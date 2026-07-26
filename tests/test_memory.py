"""memory.py の会話ログ保存・件数カウントのテスト。"""
import memory


def test_save_conversation_writes_markdown_file(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    path = memory.save_conversation(
        question="質問内容です", answer="回答内容です", thread_id="thread-a"
    )

    assert path.exists()
    assert path.parent == tmp_path / "thread-a"
    content = path.read_text(encoding="utf-8")
    assert "質問内容です" in content
    assert "回答内容です" in content


def test_save_conversation_filename_contains_question_snippet(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    path = memory.save_conversation(
        question="  日本語の 質問!!  ", answer="回答", thread_id="thread-a"
    )

    assert "日本語の_質問" in path.name
    assert path.suffix == ".md"


def test_save_conversation_isolates_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    memory.save_conversation("Q1", "A1", thread_id="thread-a")
    memory.save_conversation("Q2", "A2", thread_id="thread-b")

    assert memory.conversation_count("thread-a") == 1
    assert memory.conversation_count("thread-b") == 1
    assert memory.conversation_count() == 2


def test_conversation_count_zero_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path / "does-not-exist")

    assert memory.conversation_count() == 0
    assert memory.conversation_count("some-thread") == 0


def test_conversation_count_accumulates_multiple_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    for i in range(3):
        memory.save_conversation(f"質問{i}", f"回答{i}", thread_id="thread-a")

    assert memory.conversation_count("thread-a") == 3
