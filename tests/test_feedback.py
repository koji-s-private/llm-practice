"""feedback.py の回答評価（👍/👎）記録機能のテスト。"""

import json

import pytest

import feedback


def test_record_feedback_appends_jsonl_line(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("質問です", "回答です", feedback.RATING_UP, "thread-a")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["question"] == "質問です"
    assert record["answer"] == "回答です"
    assert record["rating"] == "up"
    assert record["thread_id"] == "thread-a"
    assert "timestamp" in record


def test_record_feedback_appends_multiple_records(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("Q1", "A1", feedback.RATING_UP, "thread-a")
    feedback.record_feedback("Q2", "A2", feedback.RATING_DOWN, "thread-a")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["rating"] == "up"
    assert json.loads(lines[1])["rating"] == "down"


def test_record_feedback_creates_parent_directory(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("Q", "A", feedback.RATING_DOWN, "thread-a")

    assert path.exists()


def test_record_feedback_rejects_invalid_rating(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    with pytest.raises(ValueError):
        feedback.record_feedback("Q", "A", "invalid", "thread-a")

    assert not path.exists()
