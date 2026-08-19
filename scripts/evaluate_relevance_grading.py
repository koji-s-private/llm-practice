"""
LLM採点（rag_chain.py の `_grade_relevance`、reranking相当）の適合率・再現率を評価するスクリプト。

rag_chain.py の retrieve_context ツールは
  1) 一次検索: ベクトル類似度で候補を広めに集める（scripts/evaluate_retrieval.py が評価対象）
  2) 二次検索: LLM（Ollama等）に候補を採点させ、本当に関連するものだけに絞り込む（_grade_relevance）
の2段階で構成されている。

このスクリプトは 2) の LLM採点だけを対象に、質問・候補文書・正解ラベル（関連/非関連）を
組にした評価セットを使って適合率(precision)・再現率(recall)・F1を計算する。
_grade_relevance は実際にOllama等のローカルLLM（setup.py の model）を呼び出すため、
Ollamaが起動していない環境（CI含む）では実行できない、手動実行専用のスクリプトである。
一次検索は使わず実行主体・実行コストが異なるため、evaluate_retrieval.py とは別スクリプトにしている。

本番の chroma_db/ や data/ には一切アクセスしない。

使い方:
    python scripts/evaluate_relevance_grading.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document

from rag_chain import _grade_relevance


def _candidate(text: str, relevant: bool) -> dict:
    return {"text": text, "relevant": relevant}


# 質問ごとに、候補文書（本文テキスト）と正解ラベル（関連=True/非関連=False）の組を用意する。
# 語彙は近いが意味的には無関係な「ノイズ候補」を混ぜ、一次検索の粗いフィルタでは弾けない
# 誤検出をLLM採点（_grade_relevance）が正しく除外できるかを検証する。
EVAL_CASES = [
    {
        "query": "電化製品はいつまで返品できますか？",
        "candidates": [
            _candidate(
                "返品ポリシー: 電化製品は購入から30日以内であれば未使用品に限り返品を受け付けます。",
                True,
            ),
            _candidate(
                "交換ポリシー: セール品・アウトレット品は返品不可ですが、"
                "サイズ違いによる交換のみ購入後14日以内であれば承ります。",
                False,
            ),
            _candidate(
                "国内配送: 国内への配送は通常3〜5営業日で到着します。送料は5000円以上のご注文で無料になります。",
                False,
            ),
        ],
    },
    {
        "query": "海外へ配送する場合、届くまでどれくらいかかりますか？",
        "candidates": [
            _candidate(
                "海外配送: 海外への配送には通常10〜15営業日かかります。関税はお客様負担となる場合があります。",
                True,
            ),
            _candidate(
                "国内配送: 国内への配送は通常3〜5営業日で到着します。送料は5000円以上のご注文で無料になります。",
                False,
            ),
            _candidate(
                "返品ポリシー: 電化製品は購入から30日以内であれば未使用品に限り返品を受け付けます。",
                False,
            ),
        ],
    },
    {
        "query": "Pythonでリストを1行で作る書き方を教えて",
        "candidates": [
            _candidate(
                "Pythonのリスト内包表記は `[式 for 変数 in イテラブル]` の形式で、ループを1行で簡潔に書ける構文です。",
                True,
            ),
            _candidate(
                "Pythonの辞書内包表記は `{キー: 値 for 変数 in イテラブル}` の形式で、辞書を1行で構築できます。",
                False,
            ),
            _candidate(
                "コミットメッセージは Conventional Commits（feat:, fix:, test: など）の形式に従う必要があります。",
                False,
            ),
        ],
    },
    {
        "query": "コミットメッセージの書き方のルールは？",
        "candidates": [
            _candidate(
                "コミットメッセージは Conventional Commits（feat:, fix:, test: など）の形式に従う必要があります。",
                True,
            ),
            _candidate(
                "このプロジェクトのブランチ命名規則は `feature/<内容>` や `fix/<内容>` のように"
                "プレフィックスを付けることになっています。",
                False,
            ),
            _candidate(
                "有給休暇の申請は、取得希望日の2週間前までに人事ポータルから申請してください。",
                False,
            ),
        ],
    },
    {
        "query": "体調不良で休むときはどう連絡すればいい？",
        "candidates": [
            _candidate(
                "体調不良で欠勤する場合は、当日の始業前までに上長にチャットで連絡してください。",
                True,
            ),
            _candidate(
                "有給休暇の申請は、取得希望日の2週間前までに人事ポータルから申請してください。",
                False,
            ),
            _candidate(
                "経費精算は、領収書を経費システムにアップロードし、支出から30日以内に申請する必要があります。",
                False,
            ),
        ],
    },
    {
        "query": "経費の領収書はいつまでに提出すればいいですか？",
        "candidates": [
            _candidate(
                "経費精算は、領収書を経費システムにアップロードし、支出から30日以内に申請する必要があります。",
                True,
            ),
            _candidate("出張旅費の払い戻しには、出張前に上長の事前承認が必要です。", False),
            _candidate(
                "パスワードポリシー: パスワードは12文字以上とし、90日ごとに変更する必要があります。",
                False,
            ),
        ],
    },
    {
        "query": "パスワードやAPIキーのセキュリティルールを教えて",
        "candidates": [
            _candidate(
                "パスワードポリシー: パスワードは12文字以上とし、90日ごとに変更する必要があります。",
                True,
            ),
            _candidate(
                "APIキーは90日ごとにローテーションし、Vaultで安全に管理する必要があります。",
                True,
            ),
            _candidate(
                "データベースのバックアップは毎晩深夜2時に自動実行され、30日分保存されます。",
                False,
            ),
        ],
    },
    {
        "query": "データベースのバックアップはどのくらいの頻度で行われますか？",
        "candidates": [
            _candidate(
                "データベースのバックアップは毎晩深夜2時に自動実行され、30日分保存されます。",
                True,
            ),
            _candidate(
                "二要素認証（2FA）は、設定ページの「セキュリティ」タブから認証アプリを使って有効化できます。",
                False,
            ),
            _candidate(
                "新入社員のオンボーディングでは、初日にPCの貸与とアカウント発行、社内システムの説明を行います。",
                False,
            ),
        ],
    },
    {
        "query": "新しく入社した人に初日にやることは？",
        "candidates": [
            _candidate(
                "新入社員のオンボーディングでは、初日にPCの貸与とアカウント発行、社内システムの説明を行います。",
                True,
            ),
            _candidate(
                "退職者のオフボーディングでは、最終出社日にアクセス権限を全て失効させ、貸与物を回収します。",
                False,
            ),
            _candidate("会議室の予約は社内カレンダーアプリから行えます。30分単位で予約可能です。", False),
        ],
    },
    {
        "query": "この検索の仕組みは何段階になっていますか？",
        "candidates": [
            _candidate(
                "このアプリの検索は2段階構成です。まずベクトル類似度で候補を広めに集め、"
                "その後LLMに採点させて関連する文書だけに絞り込みます。",
                True,
            ),
            _candidate(
                "ベクトルDBにはChromaを採用しています。ローカルに永続化でき、追加のサーバー起動が不要なためです。",
                False,
            ),
            _candidate(
                "プリンターの紙詰まりが起きた場合は、電源を切ってから用紙トレイをゆっくり引き出してください。",
                False,
            ),
        ],
    },
]


def evaluate_case(case: dict) -> tuple[float, float]:
    """1件の評価ケースについて、_grade_relevanceの適合率・再現率を返す。"""
    docs = [Document(candidate["text"]) for candidate in case["candidates"]]
    predicted_idx = set(_grade_relevance(case["query"], docs))
    relevant_idx = {i for i, candidate in enumerate(case["candidates"]) if candidate["relevant"]}

    true_positives = predicted_idx & relevant_idx
    precision = len(true_positives) / len(predicted_idx) if predicted_idx else 0.0
    recall = len(true_positives) / len(relevant_idx) if relevant_idx else 0.0
    return precision, recall


def main() -> None:
    print(f"評価ケース数: {len(EVAL_CASES)}件（Ollama等のローカルLLMを実際に呼び出すため実行に時間がかかります）\n")
    print(f"{'質問':40} | {'適合率(P)':>9} | {'再現率(R)':>9} | {'F1':>6}")
    print("-" * 76)

    precisions = []
    recalls = []
    for case in EVAL_CASES:
        precision, recall = evaluate_case(case)
        precisions.append(precision)
        recalls.append(recall)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        print(f"{case['query'][:40]:40} | {precision:>9.3f} | {recall:>9.3f} | {f1:>6.3f}")

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = (
        (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0.0
    )
    print("-" * 76)
    print(f"{'平均':40} | {avg_precision:>9.3f} | {avg_recall:>9.3f} | {avg_f1:>6.3f}")


if __name__ == "__main__":
    main()
