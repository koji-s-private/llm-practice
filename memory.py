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


def new_thread_id() -> str:
    """新しい会話スレッドのIDを生成する（「新しい会話」ボタンや初回アクセス時に使用）。"""
    return uuid.uuid4().hex[:8]


def _slugify_snippet(text: str, length: int = 20) -> str:
    """質問文の先頭部分をファイル名に使える形に変換する（人間が見て分かりやすくするため）。"""
    snippet = re.sub(r"\s+", "_", text.strip())[:length]
    snippet = re.sub(r"[^\w\-]", "", snippet, flags=re.UNICODE)
    return snippet or "conversation"


def save_conversation(question: str, answer: str, thread_id: str) -> Path:
    """1回分の質問・回答を Markdown ファイルとして data/conversations/<thread_id>/ に保存する。

    戻り値: 保存したファイルのパス。
    呼び出し側で ingest.sync_data_dir() を呼べば、そのままベクトルDBに反映される
    （このスレッド専用のナレッジとして、他のスレッドからは検索されない）。
    """
    thread_dir = CONVERSATIONS_DIR / thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filename = (
        f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}_"
        f"{_slugify_snippet(question)}.md"
    )
    path = thread_dir / filename

    content = (
        f"# 会話ログ\n\n"
        f"- 日時: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## 質問\n\n{question}\n\n"
        f"## 回答\n\n{answer}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def conversation_count(thread_id: str | None = None) -> int:
    """保存済みの会話ログ件数を返す（サイドバー表示などに使用）。

    thread_id を指定すればそのスレッドのみ、省略すれば全スレッド合計を返す。
    """
    if not CONVERSATIONS_DIR.exists():
        return 0
    target_dir = CONVERSATIONS_DIR / thread_id if thread_id else CONVERSATIONS_DIR
    if not target_dir.exists():
        return 0
    return sum(1 for f in target_dir.rglob("*.md") if f.is_file())
