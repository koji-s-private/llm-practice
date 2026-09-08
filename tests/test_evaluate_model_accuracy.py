"""scripts/evaluate_model_accuracy.py のスコアリングロジック・モデル選定ガードのテスト。

実際のOllama/Anthropic/OpenAI呼び出しは行わず、`init_chat_model` / `rag_chain.build_agent` /
`build_eval_vectorstore` をフェイクに差し替えて、キーワード含有率の計算（score_answer）・
正答率の集計（summarize）・有料APIガード（build_chat_model）だけを検証する。
"""

from types import SimpleNamespace

import scripts.evaluate_model_accuracy as evaluate_model_accuracy


def _spy_rmtree(monkeypatch):
    """shutil.rmtree()の呼び出し引数（削除対象パス）を記録するスパイに差し替える。"""
    removed_dirs = []
    monkeypatch.setattr(
        evaluate_model_accuracy.shutil,
        "rmtree",
        lambda path, ignore_errors=False: removed_dirs.append(path),
    )
    return removed_dirs


def _fake_agent(answer: str):
    """agent.invoke()が指定した回答文を返すフェイクエージェント。呼び出し回数も記録する。"""
    calls = []

    def _invoke(payload):
        calls.append(payload)
        return {"messages": [SimpleNamespace(content=answer)]}

    return SimpleNamespace(invoke=_invoke), calls


# --- score_answer() ---


def test_score_answer_all_keywords_present_gives_one():
    assert evaluate_model_accuracy.score_answer("返品は30日以内です。", ["30日"]) == 1.0


def test_score_answer_no_keywords_present_gives_zero():
    assert evaluate_model_accuracy.score_answer("特に制限はありません。", ["30日"]) == 0.0


def test_score_answer_partial_match_gives_fractional_rate():
    assert evaluate_model_accuracy.score_answer("30日以内なら可能です。", ["30日", "未使用品"]) == 0.5


def test_score_answer_handles_empty_expected_keywords_without_zero_division():
    # EVAL_SETには通常存在しないが、境界値（0除算防止）として安全側の0.0を返すことを確認する。
    assert evaluate_model_accuracy.score_answer("何らかの回答", []) == 0.0


def test_score_answer_is_case_of_substring_containment_not_exact_match():
    # 前後に文字が付いた文中に含まれていても一致とみなす（部分文字列判定）。
    assert evaluate_model_accuracy.score_answer("送料は5000円以上で無料になります。", ["5000円"]) == 1.0


# --- run_case() / run_eval_set() ---


def test_run_case_marks_correct_when_all_keywords_found():
    agent, calls = _fake_agent("返品は30日以内です。")
    case = {"query": "電化製品はいつまで返品できますか？", "expected_keywords": ["30日"]}

    result = evaluate_model_accuracy.run_case(agent, case)

    assert result["correct"] is True
    assert result["score"] == 1.0
    assert result["answer"] == "返品は30日以内です。"
    assert calls == [{"messages": [{"role": "user", "content": case["query"]}]}]


def test_run_case_marks_incorrect_when_keyword_missing():
    agent, _ = _fake_agent("特に制限はありません。")
    case = {"query": "電化製品はいつまで返品できますか？", "expected_keywords": ["30日"]}

    result = evaluate_model_accuracy.run_case(agent, case)

    assert result["correct"] is False
    assert result["score"] == 0.0


def test_run_eval_set_runs_agent_once_per_case():
    agent, calls = _fake_agent("30日以内です。")
    eval_set = [
        {"query": "q1", "expected_keywords": ["30日"]},
        {"query": "q2", "expected_keywords": ["30日"]},
    ]

    results = evaluate_model_accuracy.run_eval_set(agent, eval_set)

    assert len(results) == 2
    assert len(calls) == 2


# --- summarize() ---


def test_summarize_all_correct_gives_accuracy_of_one():
    case_results = [{"correct": True, "score": 1.0}, {"correct": True, "score": 1.0}]

    summary = evaluate_model_accuracy.summarize(case_results)

    assert summary == {"accuracy": 1.0, "avg_keyword_rate": 1.0}


def test_summarize_mixed_results_averages_correctly():
    case_results = [
        {"correct": True, "score": 1.0},
        {"correct": False, "score": 0.0},
        {"correct": False, "score": 0.5},
    ]

    summary = evaluate_model_accuracy.summarize(case_results)

    assert summary["accuracy"] == 1 / 3
    assert summary["avg_keyword_rate"] == 0.5


def test_summarize_handles_empty_case_results_without_zero_division():
    assert evaluate_model_accuracy.summarize([]) == {"accuracy": 0.0, "avg_keyword_rate": 0.0}


# --- build_chat_model()（有料APIガード） ---


def test_build_chat_model_uses_ollama_by_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        evaluate_model_accuracy,
        "init_chat_model",
        lambda name, **kwargs: captured.update(name=name, kwargs=kwargs) or "fake-model",
    )

    result = evaluate_model_accuracy.build_chat_model("llama3.1", "ollama", allow_paid_api=False)

    assert result == "fake-model"
    assert captured == {
        "name": "llama3.1",
        "kwargs": {"model_provider": "ollama", "num_ctx": evaluate_model_accuracy.OLLAMA_NUM_CTX},
    }


def test_build_chat_model_rejects_paid_provider_without_explicit_flag(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("allow_paid_api=Falseの場合、init_chat_model()は呼ばれてはいけない（課金防止）")

    monkeypatch.setattr(evaluate_model_accuracy, "init_chat_model", _fail_if_called)

    try:
        evaluate_model_accuracy.build_chat_model("claude-sonnet-5", "anthropic", allow_paid_api=False)
        raised = False
    except ValueError:
        raised = True

    assert raised


def test_build_chat_model_allows_paid_provider_with_explicit_flag(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        evaluate_model_accuracy,
        "init_chat_model",
        lambda name, **kwargs: captured.update(name=name, kwargs=kwargs) or "fake-model",
    )

    result = evaluate_model_accuracy.build_chat_model("claude-sonnet-5", "anthropic", allow_paid_api=True)

    assert result == "fake-model"
    assert captured == {"name": "claude-sonnet-5", "kwargs": {"model_provider": "anthropic"}}


# --- build_arg_parser()（デフォルトでは有料APIに到達し得ないことの確認） ---


def test_arg_parser_defaults_to_ollama_provider_without_paid_api_flag():
    args = evaluate_model_accuracy.build_arg_parser().parse_args(["--model", "llama3.1"])

    assert args.model == ["llama3.1"]
    assert args.provider == "ollama"
    assert args.allow_paid_api is False


def test_arg_parser_accepts_multiple_models_for_comparison():
    args = evaluate_model_accuracy.build_arg_parser().parse_args(["--model", "llama3.1", "--model", "qwen2.5"])

    assert args.model == ["llama3.1", "qwen2.5"]


def test_arg_parser_requires_model_argument():
    # --modelが未指定の場合はargparseの標準動作としてSystemExit(2)で拒否される。
    try:
        evaluate_model_accuracy.build_arg_parser().parse_args([])
        raised = False
    except SystemExit as exc:
        raised = True
        assert exc.code == 2

    assert raised


def test_arg_parser_rejects_unknown_provider_choice():
    try:
        evaluate_model_accuracy.build_arg_parser().parse_args(["--model", "llama3.1", "--provider", "bedrock"])
        raised = False
    except SystemExit as exc:
        raised = True
        assert exc.code == 2

    assert raised


# --- build_eval_vectorstore()（本番chroma_db/を一切使わないことの確認） ---


def test_build_eval_vectorstore_uses_tempdir_and_not_production_chroma_db(monkeypatch):
    """本番のchroma_db/ディレクトリ名や既存コレクションと無関係な、独立した一時ディレクトリ・
    専用コレクション名を使っていることを確認する（Chroma/get_embeddingsはフェイクに差し替え、
    実際の埋め込みモデルダウンロードやディスクI/Oは発生させない）。"""
    captured = {}

    class _FakeChroma:
        def __init__(self, collection_name, embedding_function, persist_directory):
            captured.update(
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=persist_directory,
            )

        def add_documents(self, docs):
            captured["docs"] = docs

    monkeypatch.setattr(evaluate_model_accuracy, "Chroma", _FakeChroma)
    monkeypatch.setattr(evaluate_model_accuracy, "get_embeddings", lambda: "fake-embeddings")

    vector_store, tmp_dir = evaluate_model_accuracy.build_eval_vectorstore()

    assert isinstance(vector_store, _FakeChroma)
    assert tmp_dir == captured["persist_directory"]
    assert "chroma_db" not in tmp_dir
    assert captured["collection_name"] == "eval_model_accuracy"
    assert captured["embedding_function"] == "fake-embeddings"
    assert captured["docs"] == evaluate_model_accuracy.CORPUS


# --- evaluate_model()（オーケストレーション。重い依存はすべてフェイクに差し替える） ---


def test_evaluate_model_wires_chat_model_vectorstore_and_agent_together(monkeypatch):
    fake_chat_model = object()
    monkeypatch.setattr(
        evaluate_model_accuracy,
        "build_chat_model",
        lambda model_name, provider, allow_paid_api: fake_chat_model,
    )
    fake_store = object()
    monkeypatch.setattr(evaluate_model_accuracy, "build_eval_vectorstore", lambda: (fake_store, "/tmp/does-not-matter"))
    removed_dirs = _spy_rmtree(monkeypatch)

    captured_build_agent_kwargs = {}

    def _fake_build_agent(thread_id, chat_model):
        captured_build_agent_kwargs.update(thread_id=thread_id, chat_model=chat_model)
        agent, _ = _fake_agent("30日以内です。")
        return agent

    monkeypatch.setattr(evaluate_model_accuracy.rag_chain, "build_agent", _fake_build_agent)

    result = evaluate_model_accuracy.evaluate_model("llama3.1", "ollama", allow_paid_api=False)

    expected_kwargs = {"thread_id": evaluate_model_accuracy.EVAL_THREAD_ID, "chat_model": fake_chat_model}
    assert captured_build_agent_kwargs == expected_kwargs
    assert result["model"] == "llama3.1"
    assert result["provider"] == "ollama"
    assert len(result["cases"]) == len(evaluate_model_accuracy.EVAL_SET)
    assert removed_dirs == ["/tmp/does-not-matter"]


def test_evaluate_model_cleans_up_tmp_dir_even_if_run_fails(monkeypatch):
    monkeypatch.setattr(
        evaluate_model_accuracy, "build_chat_model", lambda model_name, provider, allow_paid_api: object()
    )
    monkeypatch.setattr(evaluate_model_accuracy, "build_eval_vectorstore", lambda: (object(), "/tmp/does-not-matter"))
    removed_dirs = _spy_rmtree(monkeypatch)

    def _raise(*args, **kwargs):
        raise RuntimeError("エージェント構築中の予期しない失敗")

    monkeypatch.setattr(evaluate_model_accuracy.rag_chain, "build_agent", _raise)

    try:
        evaluate_model_accuracy.evaluate_model("llama3.1", "ollama", allow_paid_api=False)
        raised = False
    except RuntimeError:
        raised = True

    assert raised
    assert removed_dirs == ["/tmp/does-not-matter"]


# --- main()（複数モデルの比較出力） ---


def test_main_prints_comparison_table_for_multiple_models(monkeypatch, capsys):
    monkeypatch.setattr(
        evaluate_model_accuracy,
        "build_arg_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(model=["model-a", "model-b"], provider="ollama", allow_paid_api=False)
        ),
    )

    def _fake_evaluate_model(model_name, provider, allow_paid_api):
        return {
            "model": model_name,
            "provider": provider,
            "cases": [{"query": "q1", "answer": "a1", "score": 1.0, "correct": True}],
            "accuracy": 1.0,
            "avg_keyword_rate": 1.0,
        }

    monkeypatch.setattr(evaluate_model_accuracy, "evaluate_model", _fake_evaluate_model)

    evaluate_model_accuracy.main()

    out = capsys.readouterr().out
    assert "model-a" in out
    assert "model-b" in out
    assert "モデル比較" in out


def test_main_never_calls_init_chat_model_directly_for_paid_provider_without_flag(monkeypatch):
    # allow_paid_api=Falseがmain()からevaluate_model()経由でbuild_chat_model()まで
    # 正しく伝播し、有料プロバイダ指定時にinit_chat_model()へ到達しないことを確認する。
    monkeypatch.setattr(
        evaluate_model_accuracy,
        "build_arg_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(model=["claude-sonnet-5"], provider="anthropic", allow_paid_api=False)
        ),
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("allow_paid_api=Falseの場合、init_chat_model()は呼ばれてはいけない（課金防止）")

    monkeypatch.setattr(evaluate_model_accuracy, "init_chat_model", _fail_if_called)

    try:
        evaluate_model_accuracy.main()
        raised = False
    except ValueError:
        raised = True

    assert raised
