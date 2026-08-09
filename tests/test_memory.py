"""memory.py の会話ログ保存・件数カウント・過去スレッド一覧・再開機能のテスト。"""

import os
from datetime import datetime

import memory


def test_save_conversation_writes_markdown_file(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    path = memory.save_conversation(question="質問内容です", answer="回答内容です", thread_id="thread-a")

    assert path.exists()
    assert path.parent == tmp_path / "thread-a"
    content = path.read_text(encoding="utf-8")
    assert "質問内容です" in content
    assert "回答内容です" in content


def test_save_conversation_filename_contains_question_snippet(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    path = memory.save_conversation(question="  日本語の 質問!!  ", answer="回答", thread_id="thread-a")

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


# --- 一般知識フォールバックのメタデータ行 ---


def test_save_conversation_writes_fallback_true_metadata_when_is_fallback_true(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    path = memory.save_conversation(
        question="質問内容です", answer="回答内容です", thread_id="thread-a", is_fallback=True
    )

    content = path.read_text(encoding="utf-8")
    assert "- 一般知識フォールバック: true" in content


def test_save_conversation_writes_fallback_false_metadata_by_default(tmp_path, monkeypatch):
    # is_fallback を省略した場合はデフォルトでFalse扱いになる
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    path = memory.save_conversation(question="質問内容です", answer="回答内容です", thread_id="thread-a")

    content = path.read_text(encoding="utf-8")
    assert "- 一般知識フォールバック: false" in content


def test_save_conversation_writes_fallback_false_metadata_when_explicitly_false(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    path = memory.save_conversation(
        question="質問内容です", answer="回答内容です", thread_id="thread-a", is_fallback=False
    )

    content = path.read_text(encoding="utf-8")
    assert "- 一般知識フォールバック: false" in content


def test_save_conversation_fallback_metadata_line_immediately_follows_datetime_line(tmp_path, monkeypatch):
    # ingest.FALLBACK_METADATA_PATTERN が検出できる位置関係（日時行の次）で
    # 書き込まれていることを確認する
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    path = memory.save_conversation(question="質問", answer="回答", thread_id="thread-a", is_fallback=True)

    lines = path.read_text(encoding="utf-8").splitlines()
    datetime_line_index = next(i for i, line in enumerate(lines) if line.startswith("- 日時:"))
    assert lines[datetime_line_index + 1] == "- 一般知識フォールバック: true"


# --- list_threads() / load_conversation()（過去の会話スレッド一覧・再開機能） ---


def _write_log(base_dir, thread_id, filename, question="質問", answer="回答", is_fallback=False):
    """save_conversation()が生成する書式に合わせた会話ログファイルを直接書き込むテスト用ヘルパー。

    ファイル名（先頭15文字が日時）や作成順を明示的にコントロールしたいテストのために使う。
    """
    thread_dir = base_dir / thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)
    path = thread_dir / filename
    content = (
        f"# 会話ログ\n\n"
        f"- 日時: 2024-01-01 00:00:00\n"
        f"- 一般知識フォールバック: {'true' if is_fallback else 'false'}\n\n"
        f"## 質問\n\n{question}\n\n## 回答\n\n{answer}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


# --- list_threads: 正常系・境界値 ---


def test_list_threads_returns_empty_list_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path / "does-not-exist")

    assert memory.list_threads() == []


def test_list_threads_returns_empty_list_when_no_thread_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    assert memory.list_threads() == []


def test_list_threads_skips_empty_thread_folder(tmp_path, monkeypatch):
    """境界値: 会話ログが1件も無い（空の）スレッドフォルダは一覧に含めない。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    (tmp_path / "empty-thread").mkdir()

    assert memory.list_threads() == []


def test_list_threads_ignores_non_directory_entries(tmp_path, monkeypatch):
    """境界値: data/conversations/ 直下に紛れ込んだファイル（ディレクトリでないもの）は無視する。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    (tmp_path / "stray-file.txt").write_text("not a thread dir", encoding="utf-8")

    assert memory.list_threads() == []


def test_list_threads_returns_metadata_for_single_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    _write_log(tmp_path, "thread-a", "20240101_090000_abc123_q.md", question="最初の質問です", answer="回答です")

    threads = memory.list_threads()

    assert len(threads) == 1
    thread = threads[0]
    assert thread["thread_id"] == "thread-a"
    assert thread["created_at"] == datetime(2024, 1, 1, 9, 0, 0)
    assert thread["first_question"] == "最初の質問です"
    assert thread["count"] == 1


def test_list_threads_count_reflects_number_of_files(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    _write_log(tmp_path, "thread-a", "20240101_090000_aaa111_q1.md", question="Q1")
    _write_log(tmp_path, "thread-a", "20240101_091000_bbb222_q2.md", question="Q2")
    _write_log(tmp_path, "thread-a", "20240101_092000_ccc333_q3.md", question="Q3")

    threads = memory.list_threads()

    assert threads[0]["count"] == 3


def test_list_threads_uses_earliest_file_for_created_at_and_first_question(tmp_path, monkeypatch):
    """境界値: 複数ファイルがある場合、created_at/first_questionはファイル名昇順で
    最初のもの（＝最も古い会話ログ）から取得される。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    _write_log(tmp_path, "thread-a", "20240101_090000_aaa111_first.md", question="最初の質問")
    _write_log(tmp_path, "thread-a", "20240101_100000_bbb222_second.md", question="2番目の質問")

    threads = memory.list_threads()

    assert threads[0]["created_at"] == datetime(2024, 1, 1, 9, 0, 0)
    assert threads[0]["first_question"] == "最初の質問"


def test_list_threads_sorts_multiple_threads_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    _write_log(tmp_path, "thread-old", "20240101_090000_aaa111_q.md")
    _write_log(tmp_path, "thread-new", "20240201_090000_bbb222_q.md")
    _write_log(tmp_path, "thread-mid", "20240115_090000_ccc333_q.md")

    threads = memory.list_threads()

    assert [t["thread_id"] for t in threads] == ["thread-new", "thread-mid", "thread-old"]


def test_list_threads_created_at_falls_back_to_mtime_for_unparseable_filename(tmp_path, monkeypatch):
    """異常系境界値: ファイル名がsave_conversationの命名規則（先頭15文字が日時）と一致しない場合、
    strptimeが失敗しファイルのmtimeにフォールバックする。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    thread_dir = tmp_path / "thread-a"
    thread_dir.mkdir()
    path = thread_dir / "not-a-timestamp-name.md"
    path.write_text("## 質問\n\n質問\n\n## 回答\n\n回答\n", encoding="utf-8")
    fixed_mtime = datetime(2023, 5, 5, 12, 30, 0).timestamp()
    os.utime(path, (fixed_mtime, fixed_mtime))

    threads = memory.list_threads()

    assert threads[0]["created_at"] == datetime.fromtimestamp(fixed_mtime)


def test_list_threads_first_question_empty_string_when_pattern_not_matched(tmp_path, monkeypatch):
    """異常系: 質問・回答の書式が想定と異なる場合、first_questionは空文字列になる
    （クラッシュはしない）。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    thread_dir = tmp_path / "thread-a"
    thread_dir.mkdir()
    (thread_dir / "20240101_090000_aaa111_q.md").write_text("不正なフォーマットの内容", encoding="utf-8")

    threads = memory.list_threads()

    assert threads[0]["first_question"] == ""


# --- load_conversation: 正常系・異常系・境界値 ---


def test_load_conversation_returns_empty_list_when_thread_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    assert memory.load_conversation("no-such-thread") == []


def test_load_conversation_returns_empty_list_when_thread_dir_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    (tmp_path / "empty-thread").mkdir()

    assert memory.load_conversation("empty-thread") == []


def test_load_conversation_extracts_question_and_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    _write_log(tmp_path, "thread-a", "20240101_090000_aaa111_q.md", question="質問内容", answer="回答内容")

    conversations = memory.load_conversation("thread-a")

    assert conversations == [{"question": "質問内容", "answer": "回答内容"}]


def test_load_conversation_returns_entries_in_chronological_order(tmp_path, monkeypatch):
    """正常系: ファイル名（先頭が日時）昇順、すなわち古い→新しい順に返る。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    _write_log(tmp_path, "thread-a", "20240101_100000_bbb222_second.md", question="2番目", answer="回答2")
    _write_log(tmp_path, "thread-a", "20240101_090000_aaa111_first.md", question="1番目", answer="回答1")

    conversations = memory.load_conversation("thread-a")

    assert [c["question"] for c in conversations] == ["1番目", "2番目"]


def test_load_conversation_handles_multiline_answer(tmp_path, monkeypatch):
    """境界値: 回答本文に改行・空行が含まれていても、末尾まで丸ごと抽出される。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    multiline_answer = "1行目\n2行目\n\n3行目"
    _write_log(tmp_path, "thread-a", "20240101_090000_aaa111_q.md", question="質問", answer=multiline_answer)

    conversations = memory.load_conversation("thread-a")

    assert conversations[0]["answer"] == multiline_answer


def test_load_conversation_returns_empty_strings_when_content_malformed(tmp_path, monkeypatch):
    """異常系: 想定と異なる書式のファイルでもクラッシュせず、質問・回答とも空文字列になる。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    thread_dir = tmp_path / "thread-a"
    thread_dir.mkdir()
    (thread_dir / "20240101_090000_aaa111_q.md").write_text("不正な内容です", encoding="utf-8")

    conversations = memory.load_conversation("thread-a")

    assert conversations == [{"question": "", "answer": ""}]


def test_load_conversation_ignores_non_markdown_files(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    thread_dir = tmp_path / "thread-a"
    thread_dir.mkdir()
    (thread_dir / "notes.txt").write_text("無関係なファイル", encoding="utf-8")
    _write_log(tmp_path, "thread-a", "20240101_090000_aaa111_q.md", question="質問", answer="回答")

    conversations = memory.load_conversation("thread-a")

    assert len(conversations) == 1
