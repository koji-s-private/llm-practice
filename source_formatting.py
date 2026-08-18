"""参照元ドキュメントの表示用整形ロジック。

Streamlit版（app.py）とAPI版（api/main.py）の両方が、retrieve_contextツールの検索結果を
画面・レスポンスに表示する際に使う共通ロジック。
"""

from pathlib import Path

from rag_chain import GLOBAL_THREAD_ID


def format_snippet(text: str, limit: int = 300) -> str:
    """参照元プレビュー用に本文を整形する。

    limitを超える場合、単語・文の途中で不自然に切れないよう句点・改行などの区切り文字
    （見つからなければ空白）のうち末尾に近いものを探して区切る。
    """
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped

    truncated = stripped[:limit]
    break_chars = "。\n！？!?"
    best_pos = max((truncated.rfind(ch) for ch in break_chars), default=-1)
    if best_pos >= limit // 2:
        truncated = truncated[: best_pos + 1]
    else:
        space_pos = truncated.rfind(" ")
        if space_pos >= limit // 2:
            truncated = truncated[:space_pos]

    return truncated.rstrip() + "..."


def format_source_label(metadata: dict) -> str:
    """参照元ドキュメントのメタデータから表示用ラベルを組み立てる。

    - source: ファイルパス → ファイル名のみを表示
    - thread_id: 会話ログ由来のチャンクにのみ付与される（GLOBAL_THREAD_IDは
      全スレッド共通ドキュメントを表すため対象外）。付与されている場合は
      「会話ログ（スレッド: xxx）」であることが分かるように先頭に付ける
    - page: PDFのページ番号（0始まり）があれば「（p.N）」を末尾に付ける
    """
    source = metadata.get("source", "unknown")
    page = metadata.get("page")
    thread_id = metadata.get("thread_id")

    label = Path(source).name if source != "unknown" else source
    if thread_id and thread_id != GLOBAL_THREAD_ID:
        label = f"会話ログ（スレッド: {thread_id}） - {label}"
    if page is not None:
        label += f"（p.{page + 1}）"
    return label
