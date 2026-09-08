"""RAGパイプライン全体（検索→LLM採点→エージェント最終回答）の回答精度をモデルごとに比較評価するスクリプト。

scripts/evaluate_retrieval.py（一次検索の適合率・再現率）、
scripts/evaluate_relevance_grading.py（LLM採点=_grade_relevanceの適合率・再現率）とは異なり、
このスクリプトは rag_chain.build_agent() が返すエージェント一式を実際に動かし、
「最終的な回答文」に正解キーワードがどれだけ含まれているかを評価する。

LLM-as-judge（採点用の追加LLM呼び出し）は使わず、正解キーワードの文字列含有判定という
機械的・決定的な方法でスコア化する。

費用面の制約: デフォルトのプロバイダはOllama（無料・ローカル）のみで、Anthropic/OpenAI等の
有料APIは `--allow-paid-api` を明示的に指定しない限り実行できないようにしている
（うっかり実行しても課金が発生しないようにするため）。

本番の chroma_db/ や data/ は一切変更しない。評価用コーパスは一時ディレクトリに
専用コレクションとして作成し、実行後に破棄する。

使い方:
    # デフォルト（Ollama、無料）
    python scripts/evaluate_model_accuracy.py --model llama3.1

    # 複数モデルを比較
    python scripts/evaluate_model_accuracy.py --model llama3.1 --model qwen2.5

    # 有料APIモデルを評価したい場合（人間が明示的に指定した場合のみ）
    python scripts/evaluate_model_accuracy.py --model claude-sonnet-5 --provider anthropic --allow-paid-api
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document

import rag_chain
from rag_chain import get_embeddings
from setup import OLLAMA_NUM_CTX

# 評価用に検索対象とするスレッドID（build_agent()のthread_id引数）。
# 評価用コーパスのDocumentにはGLOBAL_THREAD_IDを付与しているため、実際の値は何でもよい。
EVAL_THREAD_ID = "eval-model-accuracy"

# 評価専用コーパス（本番のdata/とは無関係）。数値・固有名詞など、LLMが言い換えても
# 残りやすい具体的な事実を1文書1トピックで用意している。
CORPUS = [
    Document(
        "返品ポリシー: 電化製品は購入から30日以内であれば未使用品に限り返品を受け付けます。",
        metadata={"doc_id": "return_policy", "thread_id": rag_chain.GLOBAL_THREAD_ID},
    ),
    Document(
        "国内配送: 国内への配送は通常3〜5営業日で到着します。送料は5000円以上のご注文で無料になります。",
        metadata={"doc_id": "shipping_domestic", "thread_id": rag_chain.GLOBAL_THREAD_ID},
    ),
    Document(
        "パスワードポリシー: パスワードは12文字以上とし、90日ごとに変更する必要があります。",
        metadata={"doc_id": "password_policy", "thread_id": rag_chain.GLOBAL_THREAD_ID},
    ),
    Document(
        "有給休暇の申請は、取得希望日の2週間前までに人事ポータルから申請してください。",
        metadata={"doc_id": "vacation_request", "thread_id": rag_chain.GLOBAL_THREAD_ID},
    ),
    Document(
        "経費精算は、領収書を経費システムにアップロードし、支出から30日以内に申請する必要があります。",
        metadata={"doc_id": "expense_report", "thread_id": rag_chain.GLOBAL_THREAD_ID},
    ),
    Document(
        "データベースのバックアップは毎晩深夜2時に自動実行され、30日分保存されます。",
        metadata={"doc_id": "db_backup_schedule", "thread_id": rag_chain.GLOBAL_THREAD_ID},
    ),
    Document(
        "二要素認証（2FA）は、設定ページの「セキュリティ」タブから認証アプリを使って有効化できます。",
        metadata={"doc_id": "two_factor_auth", "thread_id": rag_chain.GLOBAL_THREAD_ID},
    ),
    Document(
        "会議室の予約は社内カレンダーアプリから行えます。30分単位で予約可能です。",
        metadata={"doc_id": "meeting_room_booking", "thread_id": rag_chain.GLOBAL_THREAD_ID},
    ),
]

# 質問と、正誤判定用の期待キーワードのペア。expected_keywords は「最終回答に全て含まれていれば
# 正解」というAND判定に使う（score_answer/summarize参照）。
EVAL_SET = [
    {"query": "電化製品はいつまで返品できますか？", "expected_keywords": ["30日"]},
    {"query": "送料はいくら以上の注文で無料になりますか？", "expected_keywords": ["5000円"]},
    {"query": "パスワードは何文字以上にする必要がありますか？", "expected_keywords": ["12文字"]},
    {"query": "有給休暇はいつまでに申請すればいいですか？", "expected_keywords": ["2週間"]},
    {"query": "経費の領収書はいつまでに提出すればいいですか？", "expected_keywords": ["30日"]},
    {"query": "データベースのバックアップは何時に実行されますか？", "expected_keywords": ["深夜2時"]},
    {"query": "二要素認証はどこから設定できますか？", "expected_keywords": ["セキュリティ"]},
    {"query": "会議室は何分単位で予約できますか？", "expected_keywords": ["30分"]},
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAGパイプライン全体（検索→LLM採点→エージェント最終回答）の回答精度をモデルごとに比較評価する"
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="評価対象のモデル名（Ollama pull済みモデル名を想定）。複数回指定すると連続比較できる",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "anthropic", "openai"],
        help="モデルのプロバイダ（デフォルト: ollama。無料・ローカルで完結する）",
    )
    parser.add_argument(
        "--allow-paid-api",
        action="store_true",
        help="anthropic/openai等の有料APIプロバイダでの評価を明示的に許可する（未指定時はollama以外を拒否する）",
    )
    return parser


def build_chat_model(model_name: str, provider: str, allow_paid_api: bool):
    """指定プロバイダのチャットモデルを構築する。

    provider="ollama"以外（有料API）はallow_paid_api=Trueが明示的に渡された場合のみ
    init_chat_model()を呼び出す。falseの場合は呼び出し自体を行わずValueErrorを送出するため、
    デフォルトの実行経路で課金が発生することはない。
    """
    if provider != "ollama" and not allow_paid_api:
        raise ValueError(
            f"provider='{provider}' は有料APIのため、--allow-paid-api を明示的に指定した場合のみ"
            "実行できます（デフォルトでは無料のOllamaモデルのみ評価対象です）。"
        )
    if provider == "ollama":
        return init_chat_model(model_name, model_provider="ollama", num_ctx=OLLAMA_NUM_CTX)
    return init_chat_model(model_name, model_provider=provider)


def build_eval_vectorstore() -> tuple[Chroma, str]:
    """評価専用の一時Chromaコレクションを作り、CORPUSを投入して返す。

    本番採用中の埋め込みモデル（rag_chain.get_embeddings()）をそのまま使い、
    埋め込みモデルの違いによる回答精度への影響が評価結果に混ざらないようにする
    （埋め込みモデル自体の比較は scripts/evaluate_retrieval.py が担当する）。
    """
    tmp_dir = tempfile.mkdtemp(prefix="llm_practice_eval_model_accuracy_")
    vector_store = Chroma(
        collection_name="eval_model_accuracy",
        embedding_function=get_embeddings(),
        persist_directory=tmp_dir,
    )
    vector_store.add_documents(CORPUS)
    return vector_store, tmp_dir


def score_answer(answer: str, expected_keywords: list[str]) -> float:
    """回答文に期待キーワードが何割含まれるか（0.0〜1.0）を返す。

    LLM-as-judgeのような追加のLLM呼び出しは行わず、部分文字列としての含有判定のみで
    機械的・決定的にスコア化する。
    """
    if not expected_keywords:
        return 0.0
    hits = sum(1 for keyword in expected_keywords if keyword in answer)
    return hits / len(expected_keywords)


def run_case(agent, case: dict) -> dict:
    """1件の評価ケースについてエージェントを実際に実行し、キーワード含有率を判定する。"""
    result = agent.invoke({"messages": [{"role": "user", "content": case["query"]}]})
    answer = result["messages"][-1].content
    score = score_answer(answer, case["expected_keywords"])
    return {"query": case["query"], "answer": answer, "score": score, "correct": score >= 1.0}


def run_eval_set(agent, eval_set: list[dict] = EVAL_SET) -> list[dict]:
    return [run_case(agent, case) for case in eval_set]


def summarize(case_results: list[dict]) -> dict:
    """正答率（期待キーワードを全て含んでいた割合）と平均キーワード含有率を返す。"""
    if not case_results:
        return {"accuracy": 0.0, "avg_keyword_rate": 0.0}
    accuracy = sum(1 for case in case_results if case["correct"]) / len(case_results)
    avg_keyword_rate = sum(case["score"] for case in case_results) / len(case_results)
    return {"accuracy": accuracy, "avg_keyword_rate": avg_keyword_rate}


def evaluate_model(model_name: str, provider: str, allow_paid_api: bool) -> dict:
    """指定モデルでEVAL_SET全件のRAGパイプラインを実行し、正答率・平均キーワード含有率を返す。

    rag_chain.get_vectorstore()を評価用の一時ベクトルストアに一時的に差し替えることで、
    本番のchroma_db/には一切アクセスせずに検索→LLM採点→最終回答までの一連の流れを検証する。
    """
    chat_model = build_chat_model(model_name, provider, allow_paid_api)
    vector_store, tmp_dir = build_eval_vectorstore()
    try:
        with mock.patch.object(rag_chain, "get_vectorstore", lambda: vector_store):
            agent = rag_chain.build_agent(thread_id=EVAL_THREAD_ID, chat_model=chat_model)
            case_results = run_eval_set(agent)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"model": model_name, "provider": provider, "cases": case_results, **summarize(case_results)}


def print_case_results(model_result: dict) -> None:
    print(f"\n[{model_result['model']}] (provider={model_result['provider']})")
    print(f"{'質問':40} | {'含有率':>7} | 判定")
    print("-" * 62)
    for case in model_result["cases"]:
        mark = "正解" if case["correct"] else "不正解"
        print(f"{case['query'][:40]:40} | {case['score']:>7.3f} | {mark}")


def print_comparison(results: list[dict]) -> None:
    print("\n[モデル比較]")
    print(f"{'モデル':<30} | {'正答率':>7} | {'平均含有率':>9}")
    print("-" * 55)
    for result in results:
        print(f"{result['model']:<30} | {result['accuracy']:>7.3f} | {result['avg_keyword_rate']:>9.3f}")


def main() -> None:
    args = build_arg_parser().parse_args()
    print(
        f"評価データセット: {len(EVAL_SET)}件"
        "（RAGパイプライン全体を実際に実行するため、モデルの応答速度によっては時間がかかります）"
    )

    results = [evaluate_model(model_name, args.provider, args.allow_paid_api) for model_name in args.model]
    for result in results:
        print_case_results(result)
    print_comparison(results)


if __name__ == "__main__":
    main()
