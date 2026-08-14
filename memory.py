"""
チャットでの質問・回答を「会話ログ」として自動でナレッジ化するモジュール。

- 保存先は data/conversations/<thread_id>/ 配下のみ（このプロジェクト内のローカルディスク）。
- 外部サーバーやクラウドへの追加送信は一切行わない。ここでやっているのは
  「ローカルにファイルを書き込む」だけで、既存の ingest.sync_data_dir() が
  他のファイルと全く同じ扱いでローカル埋め込み・ローカルChromaに取り込む。
- 1問答ごとに1ファイル（.md）として保存するため、過去の会話を毎回re-embedding
  せずに済み、差分同期の仕組みにそのまま乗る。
- thread_id（会話スレッドID）ごとにサブフォルダを分けることで、
  rag_chain.build_agent(thread_id) が「このスレッドの会話ログ＋共通ナレッジ」だけを
  検索対象にでき、無関係な別スレッドの会話が回答に混ざらないようにしている。
"""

import re
import uuid
from datetime import datetime
from pathlib import Path

CONVERSATIONS_DIR = Path(__file__).parent / "data" / "conversations"

# 会話ログMarkdownの質問・回答見出し（本文中の区切り位置の目印として使う）。
_QUESTION_HEADER = "## 質問\n\n"
_ANSWER_HEADER = "\n\n## 回答\n\n"

# save_conversation()が書き込む「質問文字数」「回答文字数」のメタデータ行。
# 質問・回答本文そのものに"## 質問"や"## 回答"という文字列が偶然含まれていても、
# 見出しの位置を正規表現でパターンマッチするのではなく、あらかじめ記録しておいた
# 文字数ぶんだけをそのまま切り出すことで本文の中身に依存せず正確に復元できる。
_QUESTION_LENGTH_PATTERN = re.compile(r"^- 質問文字数: (\d+)$", re.MULTILINE)
_ANSWER_LENGTH_PATTERN = re.compile(r"^- 回答文字数: (\d+)$", re.MULTILINE)

# 文字数メタデータが無い旧形式ファイル（本Issue対応より前に保存された会話ログ）向けの
# フォールバック。非貪欲マッチのため、質問・回答本文に"## 質問"/"## 回答"に類する
# 文字列が含まれる場合は途中で切れうるが、後方互換のため残す。
_QUESTION_PATTERN = re.compile(r"## 質問\n\n(.*?)\n\n## 回答", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"## 回答\n\n(.*)", re.DOTALL)

# new_thread_id()が生成するuuid hex文字列を含む、英数字・ハイフン・アンダースコアのみを許可する。
# "/" や ".." のようなパス区切り・親ディレクトリ参照を含む値を拒否することで、将来thread_idに
# 外部入力がそのまま渡されるようになってもパストラバーサルが起きないようにする。
_THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_thread_id(thread_id: str) -> str:
    """thread_idが安全な文字のみで構成されているか検証し、そのまま返す。

    CONVERSATIONS_DIR配下のパス組み立てに使う前に必ず通すことで、
    ディレクトリトラバーサルや意図しないパスへのアクセスを防ぐ。
    """
    if not isinstance(thread_id, str) or not _THREAD_ID_PATTERN.match(thread_id):
        raise ValueError(f"不正なthread_idです: {thread_id!r}")
    return thread_id


def new_thread_id() -> str:
    """新しい会話スレッドのIDを生成する（「新しい会話」ボタンや初回アクセス時に使用）。"""
    return uuid.uuid4().hex[:8]


def _slugify_snippet(text: str, length: int = 20) -> str:
    """質問文の先頭部分をファイル名に使える形に変換する（人間が見て分かりやすくするため）。"""
    snippet = re.sub(r"\s+", "_", text.strip())[:length]
    snippet = re.sub(r"[^\w\-]", "", snippet, flags=re.UNICODE)
    return snippet or "conversation"


def save_conversation(question: str, answer: str, thread_id: str, is_fallback: bool = False) -> Path:
    """1回分の質問・回答を Markdown ファイルとして data/conversations/<thread_id>/ に保存する。

    is_fallback: ドキュメントに根拠が見つからず一般知識で回答した場合は True を渡す。
    Markdown本文にメタデータ行として書き込んでおき、ingest.sync_data_dir() が
    チャンクのメタデータ(is_fallback)に反映する。これにより、根拠のない一般知識の
    回答が以降の検索結果として再ヒットし、あたかもドキュメントの裏付けが
    あるかのように扱われてしまうことを防ぐ（rag_chain.retrieve_context側で除外する）。

    戻り値: 保存したファイルのパス。
    呼び出し側で ingest.sync_data_dir() を呼べば、そのままベクトルDBに反映される
    （このスレッド専用のナレッジとして、他のスレッドからは検索されない）。
    """
    thread_dir = CONVERSATIONS_DIR / _validate_thread_id(thread_id)
    thread_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}_{_slugify_snippet(question)}.md"
    path = thread_dir / filename

    content = (
        f"# 会話ログ\n\n"
        f"- 日時: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- 一般知識フォールバック: {'true' if is_fallback else 'false'}\n"
        f"- 質問文字数: {len(question)}\n"
        f"- 回答文字数: {len(answer)}\n\n"
        f"{_QUESTION_HEADER}{question}{_ANSWER_HEADER}{answer}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def _extract_qa(content: str) -> tuple[str, str]:
    """会話ログMarkdownの本文から質問・回答を抜き出す。

    「質問文字数」「回答文字数」のメタデータがあれば、見出しの直後からその文字数ぶんを
    そのまま切り出す（質問・回答本文に"## 質問"/"## 回答"のような文字列が含まれていても、
    正規表現の途中マッチに惑わされず正確に復元できる）。メタデータが無い、または見出しの
    位置が見つからない場合（本Issue対応より前に保存された旧形式ファイル）は、
    従来通り正規表現ベースの抽出にフォールバックする。
    """
    q_len_match = _QUESTION_LENGTH_PATTERN.search(content)
    a_len_match = _ANSWER_LENGTH_PATTERN.search(content)
    if q_len_match and a_len_match:
        q_start = content.find(_QUESTION_HEADER)
        if q_start != -1:
            q_start += len(_QUESTION_HEADER)
            q_end = q_start + int(q_len_match.group(1))
            a_start = content.find(_ANSWER_HEADER, q_end)
            # 質問の直後に回答見出しが続かない場合（記録された文字数と本文がズレている等の
            # 想定外のケース）は、位置がズレたまま切り出さずフォールバックに任せる。
            if a_start == q_end:
                a_start += len(_ANSWER_HEADER)
                a_end = a_start + int(a_len_match.group(1))
                return content[q_start:q_end], content[a_start:a_end]

    q_match = _QUESTION_PATTERN.search(content)
    a_match = _ANSWER_PATTERN.search(content)
    return (
        q_match.group(1).strip() if q_match else "",
        a_match.group(1).strip() if a_match else "",
    )


def _extract_question(content: str) -> str:
    question, _ = _extract_qa(content)
    return question


def _parse_created_at(path: Path) -> datetime:
    """ファイル名の先頭（save_conversationが付与するタイムスタンプ）から作成日時を復元する。

    命名規則から外れたファイル（手動で置かれた等）が万一あってもクラッシュしないよう、
    パースに失敗した場合はファイルの更新日時にフォールバックする。
    """
    try:
        return datetime.strptime(path.name[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.fromtimestamp(path.stat().st_mtime)


def list_threads() -> list[dict]:
    """data/conversations/ 配下の会話スレッド一覧を、作成日時が新しい順に返す。

    サイドバーでの過去スレッド選択UIに使う。各要素は以下のキーを持つ:
    - thread_id: スレッドID
    - created_at: 最初の会話ログのタイムスタンプ（datetime）
    - first_question: 最初の質問文（ラベル表示用の要約に使う）
    - count: そのスレッドに保存されている会話ログ件数

    会話ログが1件も無い（空の）スレッドフォルダは一覧に含めない。
    """
    if not CONVERSATIONS_DIR.exists():
        return []

    threads = []
    for thread_dir in CONVERSATIONS_DIR.iterdir():
        if not thread_dir.is_dir():
            continue
        files = sorted(thread_dir.glob("*.md"))
        if not files:
            continue
        first_file = files[0]
        threads.append(
            {
                "thread_id": thread_dir.name,
                "created_at": _parse_created_at(first_file),
                "first_question": _extract_question(first_file.read_text(encoding="utf-8")),
                "count": len(files),
            }
        )

    threads.sort(key=lambda t: t["created_at"], reverse=True)
    return threads


def load_conversation(thread_id: str) -> list[dict]:
    """指定スレッドの会話ログを時系列順（古い→新しい）に読み込んで返す。

    過去スレッドを再開する際、チャット画面に会話履歴を再現するために使う。
    各要素は {"question": str, "answer": str} の形式。
    """
    thread_dir = CONVERSATIONS_DIR / _validate_thread_id(thread_id)
    if not thread_dir.exists():
        return []

    conversations = []
    for f in sorted(thread_dir.glob("*.md")):
        question, answer = _extract_qa(f.read_text(encoding="utf-8"))
        conversations.append({"question": question, "answer": answer})
    return conversations


def conversation_count(thread_id: str | None = None) -> int:
    """保存済みの会話ログ件数を返す（サイドバー表示などに使用）。

    thread_id を指定すればそのスレッドのみ、省略すれば全スレッド合計を返す。
    """
    target_dir = CONVERSATIONS_DIR / _validate_thread_id(thread_id) if thread_id else CONVERSATIONS_DIR
    if not target_dir.exists():
        return 0
    return sum(1 for f in target_dir.rglob("*.md") if f.is_file())
