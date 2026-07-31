"""scripts/evaluate_retrieval.py の適合率・再現率・F1計算ロジックのテスト。

実際の埋め込みモデル・Chromaは使わず、`similarity_search_with_score` を
差し替えたフェイクのベクトルストアを注入して、`evaluate()`（適合率・再現率の
平均計算）と `main()` の集計・出力（F1計算を含む）のみを検証する。
"""
import scripts.evaluate_retrieval as evaluate_retrieval


class _FakeDocument:
    def __init__(self, doc_id):
        self.metadata = {"doc_id": doc_id}


class _FakeVectorStore:
    """query -> (doc, score) のリストを返すフェイク。

    candidate_k / distance_threshold は evaluate() 側の呼び出し引数・後段フィルタで
    検証するため、ここでは常に登録済みの全候補をそのまま返す
    （実際のChromaもk件までしか返さないが、テストではフィルタ挙動を
    evaluate() 側のロジックとして検証したいのでcandidate_kの制限は行わない）。
    """

    def __init__(self, results_by_query):
        self._results_by_query = results_by_query
        self.calls = []

    def similarity_search_with_score(self, query, k):
        self.calls.append({"query": query, "k": k})
        return self._results_by_query[query]


# --- evaluate() ---


def test_evaluate_perfect_retrieval_gives_precision_and_recall_of_one(monkeypatch):
    monkeypatch.setattr(
        evaluate_retrieval,
        "EVAL_SET",
        [{"query": "q1", "relevant_ids": {"doc_a"}}],
    )
    store = _FakeVectorStore({"q1": [(_FakeDocument("doc_a"), 0.1)]})

    precision, recall = evaluate_retrieval.evaluate(store, candidate_k=8, distance_threshold=1.3)

    assert precision == 1.0
    assert recall == 1.0


def test_evaluate_false_positive_lowers_precision_but_not_recall(monkeypatch):
    # 正解ドキュメントに加え、しきい値未満のノイズドキュメントも混ざっている場合。
    monkeypatch.setattr(
        evaluate_retrieval,
        "EVAL_SET",
        [{"query": "q1", "relevant_ids": {"doc_a"}}],
    )
    store = _FakeVectorStore(
        {"q1": [(_FakeDocument("doc_a"), 0.1), (_FakeDocument("doc_noise"), 0.2)]}
    )

    precision, recall = evaluate_retrieval.evaluate(store, candidate_k=8, distance_threshold=1.3)

    assert precision == 0.5
    assert recall == 1.0


def test_evaluate_missed_relevant_doc_gives_zero_precision_and_recall(monkeypatch):
    # 正解ドキュメントの距離がしきい値以上で弾かれてしまい、候補が空になるケース。
    monkeypatch.setattr(
        evaluate_retrieval,
        "EVAL_SET",
        [{"query": "q1", "relevant_ids": {"doc_a"}}],
    )
    store = _FakeVectorStore({"q1": [(_FakeDocument("doc_a"), 1.3)]})

    precision, recall = evaluate_retrieval.evaluate(store, candidate_k=8, distance_threshold=1.3)

    # 境界値: score(1.3) < threshold(1.3) は成立しないため候補から除外される（`<` は厳密不等号）。
    assert precision == 0.0
    assert recall == 0.0


def test_evaluate_distance_threshold_boundary_is_exclusive(monkeypatch):
    monkeypatch.setattr(
        evaluate_retrieval,
        "EVAL_SET",
        [{"query": "q1", "relevant_ids": {"doc_a"}}],
    )
    # しきい値をわずかに下回るスコアなら候補として残る。
    store = _FakeVectorStore({"q1": [(_FakeDocument("doc_a"), 1.2999999)]})

    precision, recall = evaluate_retrieval.evaluate(store, candidate_k=8, distance_threshold=1.3)

    assert precision == 1.0
    assert recall == 1.0


def test_evaluate_averages_precision_and_recall_across_queries(monkeypatch):
    monkeypatch.setattr(
        evaluate_retrieval,
        "EVAL_SET",
        [
            {"query": "q1", "relevant_ids": {"doc_a"}},
            {"query": "q2", "relevant_ids": {"doc_b"}},
        ],
    )
    store = _FakeVectorStore(
        {
            # q1: 完全一致（precision=1.0, recall=1.0）
            "q1": [(_FakeDocument("doc_a"), 0.1)],
            # q2: 正解が見つからない（precision=0.0, recall=0.0）
            "q2": [(_FakeDocument("doc_noise"), 0.1)],
        }
    )

    precision, recall = evaluate_retrieval.evaluate(store, candidate_k=8, distance_threshold=1.3)

    assert precision == 0.5
    assert recall == 0.5


def test_evaluate_passes_candidate_k_through_to_similarity_search(monkeypatch):
    monkeypatch.setattr(
        evaluate_retrieval,
        "EVAL_SET",
        [{"query": "q1", "relevant_ids": {"doc_a"}}],
    )
    store = _FakeVectorStore({"q1": [(_FakeDocument("doc_a"), 0.1)]})

    evaluate_retrieval.evaluate(store, candidate_k=4, distance_threshold=1.3)

    assert store.calls == [{"query": "q1", "k": 4}]


def test_evaluate_handles_empty_relevant_ids_without_zero_division(monkeypatch):
    # EVAL_SET には通常存在しないが、evaluate() 単体としての境界値（0除算防止）を確認する。
    monkeypatch.setattr(
        evaluate_retrieval,
        "EVAL_SET",
        [{"query": "q1", "relevant_ids": set()}],
    )
    store = _FakeVectorStore({"q1": [(_FakeDocument("doc_a"), 0.1)]})

    precision, recall = evaluate_retrieval.evaluate(store, candidate_k=8, distance_threshold=1.3)

    assert precision == 0.0
    assert recall == 0.0


# --- main()（グリッドサーチの集計・F1計算・出力） ---


def test_main_prints_f1_for_every_candidate_k_and_threshold_combination(monkeypatch, capsys):
    # build_eval_vectorstore() は重い依存（埋め込みモデル・Chroma）を必要とするため、
    # フェイクの (vector_store, tmp_dir) を返すように差し替える。
    monkeypatch.setattr(
        evaluate_retrieval,
        "EVAL_SET",
        [{"query": "q1", "relevant_ids": {"doc_a"}}],
    )
    # candidate_k / distance_thresholdによらず常に正解ドキュメントだけがヒットするようにし、
    # グリッド全組み合わせでprecision=recall=1.0 (F1=1.000) になることを検証する。
    store = _FakeVectorStore({"q1": [(_FakeDocument("doc_a"), 0.1)]})
    monkeypatch.setattr(
        evaluate_retrieval, "build_eval_vectorstore", lambda: (store, "/tmp/does-not-matter")
    )

    evaluate_retrieval.main()

    out = capsys.readouterr().out
    expected_rows = len(evaluate_retrieval.CANDIDATE_K_VALUES) * len(
        evaluate_retrieval.DISTANCE_THRESHOLD_VALUES
    )
    assert out.count("1.000") == expected_rows * 3  # 適合率・再現率・F1がすべて1.000
    assert store.calls[0]["k"] in evaluate_retrieval.CANDIDATE_K_VALUES


def test_main_avoids_division_by_zero_when_precision_and_recall_are_zero(monkeypatch, capsys):
    monkeypatch.setattr(
        evaluate_retrieval,
        "EVAL_SET",
        [{"query": "q1", "relevant_ids": {"doc_a"}}],
    )
    # 常に無関係なドキュメントしかヒットしない → precision=recall=0.0 → F1計算で0除算が起きうる。
    store = _FakeVectorStore({"q1": [(_FakeDocument("doc_noise"), 0.1)]})
    monkeypatch.setattr(
        evaluate_retrieval, "build_eval_vectorstore", lambda: (store, "/tmp/does-not-matter")
    )

    evaluate_retrieval.main()  # 例外を送出せずに完了すること

    out = capsys.readouterr().out
    assert "0.000" in out
