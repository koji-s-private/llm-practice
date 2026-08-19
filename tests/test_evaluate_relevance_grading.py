"""scripts/evaluate_relevance_grading.py の適合率・再現率・F1計算ロジックのテスト。

実際のOllama等のLLM呼び出し（_grade_relevance）は行わず、呼び出し結果を
差し替えたフェイクに置き換えて `evaluate_case()`（1ケースの適合率・再現率計算）と
`main()` の集計・出力（F1計算を含む）のみを検証する。
"""

import scripts.evaluate_relevance_grading as evaluate_relevance_grading


def _fake_grade_relevance(results_by_query):
    """query -> 関連ありと判定するインデックスのリスト、を返すフェイクの_grade_relevance。"""

    def _fake(query, docs):
        return results_by_query[query]

    return _fake


# --- evaluate_case() ---


def test_evaluate_case_perfect_match_gives_precision_and_recall_of_one(monkeypatch):
    monkeypatch.setattr(
        evaluate_relevance_grading,
        "_grade_relevance",
        _fake_grade_relevance({"q1": [0]}),
    )
    case = {
        "query": "q1",
        "candidates": [{"text": "relevant doc", "relevant": True}, {"text": "noise doc", "relevant": False}],
    }

    precision, recall = evaluate_relevance_grading.evaluate_case(case)

    assert precision == 1.0
    assert recall == 1.0


def test_evaluate_case_false_positive_lowers_precision_but_not_recall(monkeypatch):
    # LLMが正解に加えてノイズ候補も関連ありと誤判定したケース。
    monkeypatch.setattr(
        evaluate_relevance_grading,
        "_grade_relevance",
        _fake_grade_relevance({"q1": [0, 1]}),
    )
    case = {
        "query": "q1",
        "candidates": [{"text": "relevant doc", "relevant": True}, {"text": "noise doc", "relevant": False}],
    }

    precision, recall = evaluate_relevance_grading.evaluate_case(case)

    assert precision == 0.5
    assert recall == 1.0


def test_evaluate_case_missed_relevant_doc_gives_zero_precision_and_recall(monkeypatch):
    # LLMが「なし」と判定し、正解の候補を1件も拾えなかったケース。
    monkeypatch.setattr(
        evaluate_relevance_grading,
        "_grade_relevance",
        _fake_grade_relevance({"q1": []}),
    )
    case = {
        "query": "q1",
        "candidates": [{"text": "relevant doc", "relevant": True}, {"text": "noise doc", "relevant": False}],
    }

    precision, recall = evaluate_relevance_grading.evaluate_case(case)

    assert precision == 0.0
    assert recall == 0.0


def test_evaluate_case_handles_multiple_relevant_candidates(monkeypatch):
    # 1件しか拾えなかった場合、適合率は1.0でも再現率は下がる。
    monkeypatch.setattr(
        evaluate_relevance_grading,
        "_grade_relevance",
        _fake_grade_relevance({"q1": [0]}),
    )
    case = {
        "query": "q1",
        "candidates": [
            {"text": "relevant doc 1", "relevant": True},
            {"text": "relevant doc 2", "relevant": True},
            {"text": "noise doc", "relevant": False},
        ],
    }

    precision, recall = evaluate_relevance_grading.evaluate_case(case)

    assert precision == 1.0
    assert recall == 0.5


def test_evaluate_case_handles_no_relevant_candidates_without_zero_division(monkeypatch):
    # EVAL_CASESには通常存在しないが、evaluate_case()単体としての境界値（0除算防止）を確認する。
    monkeypatch.setattr(
        evaluate_relevance_grading,
        "_grade_relevance",
        _fake_grade_relevance({"q1": []}),
    )
    case = {"query": "q1", "candidates": [{"text": "noise doc", "relevant": False}]}

    precision, recall = evaluate_relevance_grading.evaluate_case(case)

    assert precision == 0.0
    assert recall == 0.0


def test_evaluate_case_passes_documents_built_from_candidate_text(monkeypatch):
    captured = {}

    def _fake(query, docs):
        captured["query"] = query
        captured["texts"] = [doc.page_content for doc in docs]
        return [0]

    monkeypatch.setattr(evaluate_relevance_grading, "_grade_relevance", _fake)
    case = {
        "query": "q1",
        "candidates": [{"text": "doc a", "relevant": True}, {"text": "doc b", "relevant": False}],
    }

    evaluate_relevance_grading.evaluate_case(case)

    assert captured["query"] == "q1"
    assert captured["texts"] == ["doc a", "doc b"]


# --- main()（全ケースの集計・F1計算・出力） ---


def test_main_prints_f1_for_every_case(monkeypatch, capsys):
    monkeypatch.setattr(
        evaluate_relevance_grading,
        "EVAL_CASES",
        [{"query": "q1", "candidates": [{"text": "relevant doc", "relevant": True}]}],
    )
    monkeypatch.setattr(evaluate_relevance_grading, "_grade_relevance", _fake_grade_relevance({"q1": [0]}))

    evaluate_relevance_grading.main()

    out = capsys.readouterr().out
    # 1ケース分の行 + 平均行で、適合率・再現率・F1がすべて1.000になる。
    assert out.count("1.000") == 3 * 2


def test_main_avoids_division_by_zero_when_precision_and_recall_are_zero(monkeypatch, capsys):
    monkeypatch.setattr(
        evaluate_relevance_grading,
        "EVAL_CASES",
        [{"query": "q1", "candidates": [{"text": "relevant doc", "relevant": True}]}],
    )
    monkeypatch.setattr(evaluate_relevance_grading, "_grade_relevance", _fake_grade_relevance({"q1": []}))

    evaluate_relevance_grading.main()  # 例外を送出せずに完了すること

    out = capsys.readouterr().out
    assert "0.000" in out


def test_main_averages_across_multiple_cases(monkeypatch, capsys):
    monkeypatch.setattr(
        evaluate_relevance_grading,
        "EVAL_CASES",
        [
            {"query": "q1", "candidates": [{"text": "relevant doc", "relevant": True}]},
            {"query": "q2", "candidates": [{"text": "relevant doc", "relevant": True}]},
        ],
    )
    # q1は完全一致(P=1.0,R=1.0)、q2は取りこぼし(P=0.0,R=0.0) -> 平均はP=0.5,R=0.5
    monkeypatch.setattr(
        evaluate_relevance_grading,
        "_grade_relevance",
        _fake_grade_relevance({"q1": [0], "q2": []}),
    )

    evaluate_relevance_grading.main()

    out = capsys.readouterr().out
    assert "0.500" in out
