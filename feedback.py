"""AIの回答に対するユーザー評価（👍/👎）をローカルファイルに記録するモジュール。

- 保存先は data/feedback.jsonl のみ（このプロジェクト内のローカルディスク）。外部・クラウドへの
  追加送信は一切行わない。
- 拡張子 .jsonl は ingest.LOADERS の対応拡張子に含まれないため、ベクトルDBへの取り込み
  （検索対象）には含まれない。
- 1レコード1行のJSON Lines形式。scripts/evaluate_retrieval.py 等から後で集計しやすいよう、
  質問・回答・評価・スレッドID・タイムスタンプのみのシンプルな構造にする。
"""

import json
from datetime import datetime
from pathlib import Path

FEEDBACK_PATH = Path(__file__).parent / "data" / "feedback.jsonl"

RATING_UP = "up"
RATING_DOWN = "down"
_VALID_RATINGS = (RATING_UP, RATING_DOWN)


def record_feedback(question: str, answer: str, rating: str, thread_id: str) -> None:
    """1件分の回答評価を data/feedback.jsonl に1行追記する。

    rating は RATING_UP（👍）または RATING_DOWN（👎）のいずれか。
    """
    if rating not in _VALID_RATINGS:
        raise ValueError(f"不正なratingです: {rating!r}")

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "thread_id": thread_id,
        "question": question,
        "answer": answer,
        "rating": rating,
    }
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
