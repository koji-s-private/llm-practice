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


def test_record_feedback_rejects_none_rating(tmp_path, monkeypatch):
    """異常系: ratingにNoneが渡された場合も不正値として拒否する。"""
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    with pytest.raises(ValueError):
        feedback.record_feedback("Q", "A", None, "thread-a")

    assert not path.exists()


def test_record_feedback_allows_empty_question_and_answer(tmp_path, monkeypatch):
    """境界値: 質問・回答が空文字列でも記録自体は成功する（呼び出し側の責務外）。"""
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("", "", feedback.RATING_UP, "thread-a")

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["question"] == ""
    assert record["answer"] == ""


def test_record_feedback_preserves_non_ascii_characters_unescaped(tmp_path, monkeypatch):
    """日本語・絵文字を含む質問/回答が \\uXXXX にエスケープされず、
    人間が読める形式のままJSON Linesとして書き込まれる（ensure_ascii=Falseの確認）。"""
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("日本語の質問🙂", "日本語の回答📝", feedback.RATING_DOWN, "thread-a")

    raw_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert "日本語の質問🙂" in raw_line
    assert "日本語の回答📝" in raw_line
    assert "\\u" not in raw_line


def test_record_feedback_appends_without_truncating_existing_content(tmp_path, monkeypatch):
    """境界値: 既に他の内容が書き込まれたファイルに対しても、上書きせず末尾に追記する。"""
    path = tmp_path / "feedback.jsonl"
    path.write_text('{"pre_existing": true}\n', encoding="utf-8")
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    feedback.record_feedback("Q", "A", feedback.RATING_UP, "thread-a")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"pre_existing": True}
    assert json.loads(lines[1])["rating"] == "up"


def test_default_feedback_path_points_to_data_directory():
    """境界値/設定確認: モジュールデフォルトのFEEDBACK_PATHはプロジェクト直下の
    data/feedback.jsonl を指す（.gitignoreの対象パスと一致している必要がある）。"""
    assert feedback.FEEDBACK_PATH.name == "feedback.jsonl"
    assert feedback.FEEDBACK_PATH.parent.name == "data"
