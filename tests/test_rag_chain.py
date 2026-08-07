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

    allowed_ids = set(store.last_call["filter"]["thread_id"]["$in"])
    assert allowed_ids == {rag_chain.GLOBAL_THREAD_ID, "thread-1"}


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
