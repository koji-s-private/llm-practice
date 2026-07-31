"""
リランキング閾値（CANDIDATE_K / RECALL_DISTANCE_THRESHOLD）チューニング用の評価スクリプト。

rag_chain.py の retrieve_context ツールは
  1) 一次検索: ベクトル類似度で CANDIDATE_K 件を広めに取得し、
     L2距離が RECALL_DISTANCE_THRESHOLD 未満のものだけを候補として残す（粗いフィルタ）
  2) 二次検索: LLM（Ollama等）に採点させ、本当に関連するものだけに絞り込む（_grade_relevance）
の2段階で構成されている。

このスクリプトは 1) の一次検索だけを対象に、質問と正解ドキュメントのペア（評価セット）を使って
適合率(precision)・再現率(recall)を計算する。LLM採点（2）は使わない
（この評価はLLM起動の有無に依存せず、CANDIDATE_K / RECALL_DISTANCE_THRESHOLD という
1) のパラメータ単体の妥当性を検証したいため）。

本番の chroma_db/ や data/ は一切変更しない。評価用コーパスは一時ディレクトリに
専用コレクションとして作成し、実行後に破棄する。

使い方:
    python scripts/evaluate_retrieval.py
"""
import itertools
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_chain import get_embeddings

# 評価専用コーパス（本番のdata/とは無関係）。
# 閾値・候補数の違いが見えるよう、各トピックに「正解ドキュメント」と、
# 語彙は近いが意味的には無関係な「ノイズドキュメント」を対にして混ぜている
# （rag_chain.py のコメントにある「単語は近いが意味的には無関係な文書」を再現するため）。
CORPUS = [
    Document(
        "返品ポリシー: 電化製品は購入から30日以内であれば未使用品に限り返品を受け付けます。"
        "返金は元の支払い方法に対して7営業日以内に行われます。",
        metadata={"doc_id": "return_policy"},
    ),
    Document(
        "交換ポリシー: セール品・アウトレット品は返品不可ですが、サイズ違いによる交換のみ"
        "購入後14日以内であれば承ります。",
        metadata={"doc_id": "exchange_policy"},
    ),
    Document(
        "国内配送: 国内への配送は通常3〜5営業日で到着します。送料は5000円以上のご注文で無料になります。",
        metadata={"doc_id": "shipping_domestic"},
    ),
    Document(
        "海外配送: 海外への配送には通常10〜15営業日かかります。関税はお客様負担となる場合があります。",
        metadata={"doc_id": "shipping_international"},
    ),
    Document(
        "Pythonのリスト内包表記は `[式 for 変数 in イテラブル]` の形式で、"
        "ループを1行で簡潔に書ける構文です。",
        metadata={"doc_id": "python_list_comprehension"},
    ),
    Document(
        "Pythonの辞書内包表記は `{キー: 値 for 変数 in イテラブル}` の形式で、"
        "辞書を1行で構築できます。",
        metadata={"doc_id": "python_dict_comprehension"},
    ),
    Document(
        "このプロジェクトのブランチ命名規則は `feature/<内容>` や `fix/<内容>` のように"
        "プレフィックスを付けることになっています。",
        metadata={"doc_id": "git_branch_naming"},
    ),
    Document(
        "コミットメッセージは Conventional Commits（feat:, fix:, test: など）の形式に"
        "従う必要があります。",
        metadata={"doc_id": "git_commit_convention"},
    ),
    Document(
        "有給休暇の申請は、取得希望日の2週間前までに人事ポータルから申請してください。",
        metadata={"doc_id": "vacation_request"},
    ),
    Document(
        "体調不良で欠勤する場合は、当日の始業前までに上長にチャットで連絡してください。",
        metadata={"doc_id": "sick_leave_policy"},
    ),
    Document(
        "経費精算は、領収書を経費システムにアップロードし、支出から30日以内に申請する必要があります。",
        metadata={"doc_id": "expense_report"},
    ),
    Document(
        "出張旅費の払い戻しには、出張前に上長の事前承認が必要です。",
        metadata={"doc_id": "travel_reimbursement"},
    ),
    Document(
        "パスワードポリシー: パスワードは12文字以上とし、90日ごとに変更する必要があります。",
        metadata={"doc_id": "password_policy"},
    ),
    Document(
        "二要素認証（2FA）は、設定ページの「セキュリティ」タブから認証アプリを使って"
        "有効化できます。",
        metadata={"doc_id": "two_factor_auth"},
    ),
    Document(
        "APIキーは90日ごとにローテーションし、Vaultで安全に管理する必要があります。",
        metadata={"doc_id": "api_key_rotation"},
    ),
    Document(
        "データベースのバックアップは毎晩深夜2時に自動実行され、30日分保存されます。",
        metadata={"doc_id": "db_backup_schedule"},
    ),
    Document(
        "新入社員のオンボーディングでは、初日にPCの貸与とアカウント発行、"
        "社内システムの説明を行います。",
        metadata={"doc_id": "onboarding_checklist"},
    ),
    Document(
        "退職者のオフボーディングでは、最終出社日にアクセス権限を全て失効させ、"
        "貸与物を回収します。",
        metadata={"doc_id": "offboarding_checklist"},
    ),
    Document(
        "このアプリの検索は2段階構成です。まずベクトル類似度で候補を広めに集め、"
        "その後LLMに採点させて関連する文書だけに絞り込みます。",
        metadata={"doc_id": "rag_architecture"},
    ),
    Document(
        "ベクトルDBにはChromaを採用しています。ローカルに永続化でき、"
        "追加のサーバー起動が不要なためです。",
        metadata={"doc_id": "chroma_choice"},
    ),
    Document(
        "会議室の予約は社内カレンダーアプリから行えます。30分単位で予約可能です。",
        metadata={"doc_id": "meeting_room_booking"},
    ),
    Document(
        "プリンターの紙詰まりが起きた場合は、電源を切ってから用紙トレイをゆっくり"
        "引き出してください。",
        metadata={"doc_id": "printer_troubleshooting"},
    ),
]

# 質問と正解ドキュメント（doc_id）のペア。
EVAL_SET = [
    {"query": "電化製品はいつまで返品できますか？", "relevant_ids": {"return_policy"}},
    {"query": "海外へ配送する場合、届くまでどれくらいかかりますか？", "relevant_ids": {"shipping_international"}},
    {"query": "Pythonでリストを1行で作る書き方を教えて", "relevant_ids": {"python_list_comprehension"}},
    {"query": "コミットメッセージの書き方のルールは？", "relevant_ids": {"git_commit_convention"}},
    {"query": "体調不良で休むときはどう連絡すればいい？", "relevant_ids": {"sick_leave_policy"}},
    {"query": "経費の領収書はいつまでに提出すればいいですか？", "relevant_ids": {"expense_report"}},
    {"query": "二要素認証はどうやって設定しますか？", "relevant_ids": {"two_factor_auth"}},
    {"query": "データベースのバックアップはどのくらいの頻度で行われますか？", "relevant_ids": {"db_backup_schedule"}},
    {"query": "新しく入社した人に初日にやることは？", "relevant_ids": {"onboarding_checklist"}},
    {"query": "この検索の仕組みは何段階になっていますか？", "relevant_ids": {"rag_architecture"}},
]

CANDIDATE_K_VALUES = [4, 8, 12]
DISTANCE_THRESHOLD_VALUES = [1.0, 1.3, 1.5]


def build_eval_vectorstore() -> tuple[Chroma, str]:
    """評価専用の一時Chromaコレクションを作り、CORPUSを投入して返す。"""
    tmp_dir = tempfile.mkdtemp(prefix="llm_practice_eval_retrieval_")
    vector_store = Chroma(
        collection_name="eval_retrieval",
        embedding_function=get_embeddings(),
        persist_directory=tmp_dir,
    )
    vector_store.add_documents(CORPUS)
    return vector_store, tmp_dir


def evaluate(vector_store: Chroma, candidate_k: int, distance_threshold: float) -> tuple[float, float]:
    """指定したCANDIDATE_K / RECALL_DISTANCE_THRESHOLDでの平均適合率・平均再現率を返す。"""
    precisions = []
    recalls = []
    for case in EVAL_SET:
        candidates = vector_store.similarity_search_with_score(case["query"], k=candidate_k)
        retrieved_ids = {
            doc.metadata["doc_id"] for doc, score in candidates if score < distance_threshold
        }
        relevant_ids = case["relevant_ids"]
        true_positives = retrieved_ids & relevant_ids

        precision = len(true_positives) / len(retrieved_ids) if retrieved_ids else 0.0
        recall = len(true_positives) / len(relevant_ids) if relevant_ids else 0.0
        precisions.append(precision)
        recalls.append(recall)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    return avg_precision, avg_recall


def main() -> None:
    vector_store, tmp_dir = build_eval_vectorstore()
    try:
        print(f"評価用コーパス: {len(CORPUS)}件 / 評価クエリ: {len(EVAL_SET)}件\n")
        print(f"{'CANDIDATE_K':>12} | {'THRESHOLD':>9} | {'適合率(P)':>9} | {'再現率(R)':>9} | {'F1':>6}")
        print("-" * 62)
        for candidate_k, threshold in itertools.product(CANDIDATE_K_VALUES, DISTANCE_THRESHOLD_VALUES):
            precision, recall = evaluate(vector_store, candidate_k, threshold)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            print(
                f"{candidate_k:>12} | {threshold:>9.2f} | {precision:>9.3f} | "
                f"{recall:>9.3f} | {f1:>6.3f}"
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
