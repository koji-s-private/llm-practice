"""参照元ドキュメントの表示用整形ロジック。

Streamlit版（app.py）とAPI版（api/main.py）の両方が、retrieve_contextツールの検索結果を
画面・レスポンスに表示する際に使う共通ロジック。
"""

from pathlib import Path

from rag_chain import GLOBAL_THREAD_ID, RECALL_DISTANCE_THRESHOLD

# distance_score（Chroma L2距離、値が小さいほど類似）を関連度の高/中/低に振り分ける閾値。
# retrieve_context側でRECALL_DISTANCE_THRESHOLD未満に絞り込み済みのため、
# 実際に渡ってくるスコアはその範囲内（0〜RECALL_DISTANCE_THRESHOLD）に収まる想定。
assert 0.5 < RECALL_DISTANCE_THRESHOLD, "RECALL_DISTANCE_THRESHOLDが変更された場合は閾値も見直すこと"
_RELEVANCE_TIER_HIGH_MAX = 0.5
_RELEVANCE_TIER_MID_MAX = 0.9


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


def format_relevance_tier(metadata: dict) -> str | None:
    """参照元ドキュメントのdistance_scoreから、関連度を表す「高/中/低」ラベルを返す。

    過去の会話ログから復元した参照元にはdistance_scoreが記録されていないため、
    metadataに無い場合はNoneを返す（呼び出し側は表示自体をスキップする）。
    """
    score = metadata.get("distance_score")
    if score is None:
        return None
    if score <= _RELEVANCE_TIER_HIGH_MAX:
        return "🟢 関連度: 高"
    if score <= _RELEVANCE_TIER_MID_MAX:
        return "🟡 関連度: 中"
    return "🔴 関連度: 低"
