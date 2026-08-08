"""rag_chain.py の関連度採点（_grade_relevance）とフォールバック挙動のテスト。

実際の埋め込みモデル・ベクトルDB・LLMは使わず、get_vectorstore() と
model.invoke() をテストごとに monkeypatch して、純粋なロジック
（距離フィルタ・LLM採点・「見つからない場合」の応答）だけを検証する。
"""

from types import SimpleNamespace

import rag_chain


class _FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class _FakeVectorStore:
    def __init__(self, results):
        # results: list[(doc, score)]
        self._results = results
        self.last_call = None

    def similarity_search_with_score(self, query, k, filter):
        self.last_call = {"query": query, "k": k, "filter": filter}
        return self._results


def _build_agent_with_store(monkeypatch, results):
    store = _FakeVectorStore(results)
    monkeypatch.setattr(rag_chain, "get_vectorstore", lambda: store)
    agent = rag_chain.build_agent(thread_id="thread-1")
    retrieve_context = agent.tools[0]
    return retrieve_context, store


# --- _grade_relevance ---


def test_grade_relevance_returns_empty_for_no_candidates():
    assert rag_chain._grade_relevance("質問", []) == []


def test_grade_relevance_parses_comma_separated_indices(monkeypatch):
    docs = [_FakeDocument("a"), _FakeDocument("b"), _FakeDocument("c")]
    fake_model = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="回答:1,3"))
    monkeypatch.setattr(rag_chain, "model", fake_model)

    assert rag_chain._grade_relevance("質問", docs) == [0, 2]


def test_grade_relevance_returns_empty_when_llm_says_none(monkeypatch):
    docs = [_FakeDocument("a"), _FakeDocument("b")]
    fake_model = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="回答:なし"))
    monkeypatch.setattr(rag_chain, "model", fake_model)

    assert rag_chain._grade_relevance("質問", docs) == []


def test_grade_relevance_ignores_out_of_range_indices(monkeypatch):
    docs = [_FakeDocument("a"), _FakeDocument("b")]
    # 候補は2件しかないのに "5" という範囲外の番号を返してきた場合は無視する
    fake_model = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="回答:5"))
    monkeypatch.setattr(rag_chain, "model", fake_model)

    assert rag_chain._grade_relevance("質問", docs) == []


def test_grade_relevance_ignores_numbers_in_freeform_explanation_lines(monkeypatch):
    """Issue #53のリグレッションテスト。

    LLMが1行目の「回答:」形式は守りつつ、2行目以降に自由文の説明を付け足した場合、
    その説明文中に含まれる無関係な数字（例: 年号の2024）まで拾ってしまわないことを確認する。
    """
    docs = [_FakeDocument("a"), _FakeDocument("b"), _FakeDocument("c")]
    content = "回答:1,3\n文書1は2024年に関する内容でした。文書3が最も関連しています。"
    fake_model = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content=content))
    monkeypatch.setattr(rag_chain, "model", fake_model)

    assert rag_chain._grade_relevance("質問", docs) == [0, 2]


def test_grade_relevance_falls_back_to_empty_when_answer_prefix_missing(monkeypatch):
    """Issue #53のリグレッションテスト。

    LLMが「回答:」形式のプレフィックス行を一切出力せず、完全に自由文（例:
    「文書1と文書3が関連しています。」）で回答した場合、誤って文書を採用せず
    安全側（空リスト）にフォールバックすることを確認する。
    """
    docs = [_FakeDocument("a"), _FakeDocument("b"), _FakeDocument("c")]
    content = "文書1と文書3が関連しています。"
    fake_model = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content=content))
    monkeypatch.setattr(rag_chain, "model", fake_model)

    assert rag_chain._grade_relevance("質問", docs) == []


def test_grade_relevance_parses_fullwidth_colon_answer_prefix(monkeypatch):
    """Issue #53のリグレッションテスト。全角コロン「：」を使った「回答：1,3」形式でも
    半角コロンの場合と同様に正しくパースできることを確認する。
    """
    docs = [_FakeDocument("a"), _FakeDocument("b"), _FakeDocument("c")]
    fake_model = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="回答：1,3"))
    monkeypatch.setattr(rag_chain, "model", fake_model)

    assert rag_chain._grade_relevance("質問", docs) == [0, 2]


# --- retrieve_context (build_agent の中で作られる検索ツール) ---


def test_retrieve_context_falls_back_when_no_candidates(monkeypatch):
    retrieve_context, _ = _build_agent_with_store(monkeypatch, results=[])

    content, artifact = retrieve_context.func("質問")

    assert artifact == []
    assert "見つかりませんでした" in content


def test_retrieve_context_falls_back_when_all_candidates_graded_irrelevant(monkeypatch):
    doc = _FakeDocument("無関係な内容", {"source": "a.txt"})
    retrieve_context, _ = _build_agent_with_store(monkeypatch, results=[(doc, 0.1)])
    monkeypatch.setattr(rag_chain, "_grade_relevance", lambda query, docs: [])

    content, artifact = retrieve_context.func("質問")

    assert artifact == []
    assert "見つかりませんでした" in content


def test_retrieve_context_filters_out_candidates_beyond_recall_threshold(monkeypatch):
    close_doc = _FakeDocument("近い文書", {"source": "close.txt"})
    far_doc = _FakeDocument("遠い（無関係とみなされるべき）文書", {"source": "far.txt"})
    results = [
        (close_doc, rag_chain.RECALL_DISTANCE_THRESHOLD - 0.1),
        (far_doc, rag_chain.RECALL_DISTANCE_THRESHOLD + 0.1),
    ]
    retrieve_context, _ = _build_agent_with_store(monkeypatch, results=results)

    seen_docs = {}

    def fake_grade(query, docs):
        seen_docs["docs"] = docs
        return list(range(len(docs)))

    monkeypatch.setattr(rag_chain, "_grade_relevance", fake_grade)

    retrieve_context.func("質問")

    # 距離しきい値を超えた far_doc は、LLM採点に渡す前の時点で除外されているべき
    assert seen_docs["docs"] == [close_doc]


def test_retrieve_context_returns_relevant_docs(monkeypatch):
    doc1 = _FakeDocument("関連する内容", {"source": "a.txt"})
    doc2 = _FakeDocument("これも関連する内容", {"source": "b.txt"})
    retrieve_context, _ = _build_agent_with_store(monkeypatch, results=[(doc1, 0.1), (doc2, 0.2)])
    monkeypatch.setattr(rag_chain, "_grade_relevance", lambda query, docs: [1])

    content, artifact = retrieve_context.func("質問")

    assert artifact == [doc2]
    assert "これも関連する内容" in content
    assert "a.txt" not in content


def test_retrieve_context_restricts_search_to_global_and_own_thread(monkeypatch):
    retrieve_context, store = _build_agent_with_store(monkeypatch, results=[])

    retrieve_context.func("質問")

    # filterは $and で thread_id 条件と is_fallback 条件を組み合わせた構造になっている
    and_conditions = store.last_call["filter"]["$and"]
    thread_condition = next(c for c in and_conditions if "thread_id" in c)
    allowed_ids = set(thread_condition["thread_id"]["$in"])
    assert allowed_ids == {rag_chain.GLOBAL_THREAD_ID, "thread-1"}


def test_retrieve_context_excludes_fallback_conversations_from_search_filter(monkeypatch):
    """一般知識フォールバック回答として保存された会話ログ（is_fallback=True）を
    検索対象から除外するフィルタ条件が含まれていることを確認する。"""
    retrieve_context, store = _build_agent_with_store(monkeypatch, results=[])

    retrieve_context.func("質問")

    and_conditions = store.last_call["filter"]["$and"]
    fallback_condition = next(c for c in and_conditions if "is_fallback" in c)
    assert fallback_condition == {"is_fallback": False}


# --- get_vectorstore のdocstring（Issue #81: CVE-2026-45829に関するセキュリティ注記） ---


def test_get_vectorstore_docstring_documents_known_chromadb_cve():
    """get_vectorstore() のdocstringに、既知のChromaDB脆弱性(CVE-2026-45829)への
    言及と、本実装（ローカル永続化モードのみ）では該当しない旨の説明が
    残っていることを確認する（将来のリファクタでうっかり削除されないためのガード）。
    """
    docstring = rag_chain.get_vectorstore.__doc__

    assert docstring is not None
    assert "CVE-2026-45829" in docstring
    assert "persist_directory" in docstring
