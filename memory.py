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

import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

CONVERSATIONS_DIR = Path(__file__).parent / "data" / "conversations"

# スレッドタイトルの保存先ファイル名（質問・回答のMarkdownファイルとは別管理にすることで、
# 既存の会話ログの形式・解析ロジックに一切影響しないようにする）。
THREAD_TITLE_FILENAME = "title.txt"

# 会話ログMarkdownの質問・回答・参照元見出し（本文中の区切り位置の目印として使う）。
_QUESTION_HEADER = "## 質問\n\n"
_ANSWER_HEADER = "\n\n## 回答\n\n"
_SOURCES_HEADER = "\n\n## 参照元\n\n"

# save_conversation()が書き込む「質問文字数」「回答文字数」「参照元文字数」のメタデータ行。
# 見出しの正規表現マッチではなく記録済みの文字数ぶんをそのまま切り出すことで、本文中に
# "## 質問"/"## 回答"に類する文字列が偶然含まれていても正確に復元できる。
# 参照元文字数の行が無いファイルは旧形式（sources未対応）として扱い、空リストにフォールバックする。
_QUESTION_LENGTH_PATTERN = re.compile(r"^- 質問文字数: (\d+)$", re.MULTILINE)
_ANSWER_LENGTH_PATTERN = re.compile(r"^- 回答文字数: (\d+)$", re.MULTILINE)
_SOURCES_LENGTH_PATTERN = re.compile(r"^- 参照元文字数: (\d+)$", re.MULTILINE)

# 文字数メタデータが無い旧形式ファイル向けのフォールバック（非貪欲マッチのため
# 本文に類似の文字列が含まれる場合は途中で切れうるが、後方互換のため残す）。
_QUESTION_PATTERN = re.compile(r"## 質問\n\n(.*?)\n\n## 回答", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"## 回答\n\n(.*)", re.DOTALL)

# new_thread_id()が生成するuuid hex文字列を含む、英数字・ハイフン・アンダースコアのみを許可する。
# 将来thread_idに外部入力がそのまま渡されるようになってもパストラバーサルが起きないようにする。
# api/main.py側の検証（長さ制限・resolve()によるパストラバーサル対策を追加で行う）も
# このパターンをimportして使い、許可文字ポリシーの二重管理を避ける。
THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_thread_id(thread_id: str) -> str:
    """thread_idが安全な文字のみで構成されているか検証し、そのまま返す。

    CONVERSATIONS_DIR配下のパス組み立てに使う前に必ず通すことで、
    ディレクトリトラバーサルや意図しないパスへのアクセスを防ぐ。
    """
    if not isinstance(thread_id, str) or not THREAD_ID_PATTERN.fullmatch(thread_id):
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


def _serialize_sources(sources: list) -> str:
    """参照元DocumentのリストをMarkdown内に埋め込むためJSON文字列へ変換する。"""
    return json.dumps(
        [{"metadata": doc.metadata, "page_content": doc.page_content} for doc in sources],
        ensure_ascii=False,
    )


def save_conversation(
    question: str,
    answer: str,
    thread_id: str,
    is_fallback: bool = False,
    sources: list | None = None,
) -> Path:
    """1回分の質問・回答を Markdown ファイルとして data/conversations/<thread_id>/ に保存する。

    is_fallback: ドキュメントに根拠が見つからず一般知識で回答した場合は True を渡す。
    Markdown本文にメタデータ行として書き込み、ingest.sync_data_dir() がチャンクの
    メタデータ(is_fallback)に反映する。これにより、根拠のない回答が以降の検索結果に
    再ヒットして裏付けありのように扱われることを防ぐ（rag_chain.retrieve_context側で除外）。

    sources: 回答の根拠として使われた langchain_core.documents.Document 互換のリスト
    （.metadata / .page_content を持つオブジェクト）。指定しない/空の場合は参照元セクション
    自体を書き込まず、load_conversation()側は旧形式ファイルと同じく空リストを返す。

    戻り値: 保存したファイルのパス。呼び出し側が ingest.sync_data_dir() を呼べば
    ベクトルDBに反映される。
    """
    thread_dir = CONVERSATIONS_DIR / _validate_thread_id(thread_id)
    thread_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}_{_slugify_snippet(question)}.md"
    path = thread_dir / filename

    sources_json = _serialize_sources(sources) if sources else None

    metadata_lines = (
        f"# 会話ログ\n\n"
        f"- 日時: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- 一般知識フォールバック: {'true' if is_fallback else 'false'}\n"
        f"- 質問文字数: {len(question)}\n"
        f"- 回答文字数: {len(answer)}\n"
    )
    if sources_json is not None:
        metadata_lines += f"- 参照元文字数: {len(sources_json)}\n"
    metadata_lines += "\n"

    body = f"{_QUESTION_HEADER}{question}{_ANSWER_HEADER}{answer}\n"
    if sources_json is not None:
        body += f"{_SOURCES_HEADER}{sources_json}\n"

    content = metadata_lines + body
    # 書き込み中のプロセス終了で内容が途中で切れたファイルが残らないよう、一時ファイルに
    # 書いてからos.replace()でアトミックに配置する（ingest._save_manifest()と同じパターン）。
    tmp_path = path.with_suffix(".md.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def _read_text_safe(path: Path) -> str | None:
    """会話ログファイルをUTF-8で読み込む。破損（不正なUTF-8）や権限エラー等で読めない場合はNoneを返す。

    保存時のアトミック書き込み（save_conversation）導入前に生成された壊れたファイルや、
    手動編集による破損ファイルが1件混ざっただけでlist_threads()/load_conversation()全体が
    クラッシュしないようにするためのガード。
    """
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        logger.warning("%s の読み込みに失敗したためスキップします: %s", path, e)
        return None


def _extract_qa(content: str) -> tuple[str, str]:
    """会話ログMarkdownの本文から質問・回答を抜き出す。

    文字数メタデータがあれば見出し直後からその文字数ぶんをそのまま切り出す（本文に
    "## 質問"/"## 回答"に類する文字列が含まれていても誤って途中で切れないようにするため）。
    メタデータが無い旧形式ファイルは正規表現ベースの抽出にフォールバックする。
    """
    q_len_match = _QUESTION_LENGTH_PATTERN.search(content)
    a_len_match = _ANSWER_LENGTH_PATTERN.search(content)
    if q_len_match and a_len_match:
        q_start = content.find(_QUESTION_HEADER)
        if q_start != -1:
            q_start += len(_QUESTION_HEADER)
            q_end = q_start + int(q_len_match.group(1))
            a_start = content.find(_ANSWER_HEADER, q_end)
            # 質問の直後に回答見出しが続かない（記録文字数と本文がズレている等）場合は
            # 切り出さずフォールバックに任せる。
            if a_start == q_end:
                a_start += len(_ANSWER_HEADER)
                a_end = a_start + int(a_len_match.group(1))
                return content[q_start:q_end].strip(), content[a_start:a_end].strip()

    q_match = _QUESTION_PATTERN.search(content)
    a_match = _ANSWER_PATTERN.search(content)
    return (
        q_match.group(1).strip() if q_match else "",
        a_match.group(1).strip() if a_match else "",
    )


def _extract_question(content: str) -> str:
    question, _ = _extract_qa(content)
    return question


def _extract_sources(content: str) -> list[Document]:
    """会話ログMarkdownの本文から参照元セクションを抜き出し、Documentのリストに復元する。

    参照元文字数の記録が無い（旧形式、またはsourcesを保存しなかった）場合は空リストを返す。
    JSONとして壊れている場合もクラッシュさせず空リストにフォールバックする。
    """
    len_match = _SOURCES_LENGTH_PATTERN.search(content)
    if not len_match:
        return []
    start = content.find(_SOURCES_HEADER)
    if start == -1:
        return []
    start += len(_SOURCES_HEADER)
    end = start + int(len_match.group(1))
    try:
        raw_sources = json.loads(content[start:end])
        return [Document(page_content=s["page_content"], metadata=s["metadata"]) for s in raw_sources]
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("参照元情報の復元に失敗したため空として扱います: %s", e)
        return []


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
        content = _read_text_safe(first_file)
        threads.append(
            {
                "thread_id": thread_dir.name,
                "created_at": _parse_created_at(first_file),
                "first_question": _extract_question(content) if content is not None else "",
                "count": len(files),
            }
        )

    threads.sort(key=lambda t: t["created_at"], reverse=True)
    return threads


def load_conversation(thread_id: str) -> list[dict]:
    """指定スレッドの会話ログを時系列順（古い→新しい）に読み込んで返す。

    過去スレッドを再開する際、チャット画面に会話履歴を再現するために使う。
    各要素は {"question": str, "answer": str, "created_at": datetime, "sources": list[Document]} の形式。
    sourcesは旧形式ファイル・未保存の場合は空リストになる。
    """
    thread_dir = CONVERSATIONS_DIR / _validate_thread_id(thread_id)
    if not thread_dir.exists():
        return []

    conversations = []
    for f in sorted(thread_dir.glob("*.md")):
        content = _read_text_safe(f)
        if content is None:
            continue
        question, answer = _extract_qa(content)
        conversations.append(
            {
                "question": question,
                "answer": answer,
                "created_at": _parse_created_at(f),
                "sources": _extract_sources(content),
            }
        )
    return conversations


def save_thread_title(thread_id: str, title: str) -> None:
    """スレッドにユーザー任意のタイトルを設定する。

    前後の空白を取り除いた結果が空文字列になる場合は「未設定」として扱い、
    既存のタイトルファイルがあれば削除する（自動生成ラベルへのフォールバックに戻す）。
    """
    thread_dir = CONVERSATIONS_DIR / _validate_thread_id(thread_id)
    title_path = thread_dir / THREAD_TITLE_FILENAME
    title = title.strip()
    if not title:
        title_path.unlink(missing_ok=True)
        return

    thread_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = title_path.with_suffix(".txt.tmp")
    tmp_path.write_text(title, encoding="utf-8")
    os.replace(tmp_path, title_path)


def load_thread_title(thread_id: str) -> str | None:
    """保存済みのスレッドタイトルを返す。未設定・読み込み失敗の場合はNoneを返す。"""
    title_path = CONVERSATIONS_DIR / _validate_thread_id(thread_id) / THREAD_TITLE_FILENAME
    if not title_path.exists():
        return None
    content = _read_text_safe(title_path)
    if content is None:
        return None
    return content.strip() or None


def delete_thread(thread_id: str) -> bool:
    """スレッドの会話ログ一式（data/conversations/<thread_id>/ 配下）を削除する。

    タイトルファイルも含めてディレクトリごと削除する。削除後、ベクトルDBへの反映は
    呼び出し側が ingest.sync_data_dir() を呼ぶことで行う（save_conversation()と同様、
    このモジュール自体はChroma/ingestに依存しない）。
    対象スレッドが存在しない場合はFalseを返す（api/main.py側で404判定に使う）。
    """
    thread_dir = CONVERSATIONS_DIR / _validate_thread_id(thread_id)
    if not thread_dir.is_dir():
        return False
    shutil.rmtree(thread_dir)
    return True


def conversation_count(thread_id: str | None = None) -> int:
    """保存済みの会話ログ件数を返す（サイドバー表示などに使用）。

    thread_id に None を渡す（または省略する）と全スレッド合計を返す。
    空文字列を含むそれ以外の値は save_conversation/load_conversation と同様に
    _validate_thread_id() の検証対象になり、不正な値であれば ValueError になる。
    """
    target_dir = CONVERSATIONS_DIR / _validate_thread_id(thread_id) if thread_id is not None else CONVERSATIONS_DIR
    if not target_dir.exists():
        return 0
    return sum(1 for f in target_dir.rglob("*.md") if f.is_file())
