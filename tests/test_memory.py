"""memory.py の会話ログ保存・件数カウント・過去スレッド一覧・再開機能のテスト。"""

import os
from datetime import datetime

import pytest

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


# --- 質問・回答本文に見出し文字列（"## 質問" / "## 回答"）が
#     偶然含まれる場合でも、途中で切れずに正しく復元されること ---


def test_save_conversation_writes_question_and_answer_length_metadata(tmp_path, monkeypatch):
    """正常系: save_conversation() が質問・回答それぞれの文字数メタデータ行を書き込む。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    path = memory.save_conversation(question="質問文", answer="回答文です", thread_id="thread-a")

    content = path.read_text(encoding="utf-8")
    assert f"- 質問文字数: {len('質問文')}" in content
    assert f"- 回答文字数: {len('回答文です')}" in content


def test_load_conversation_normal_case_without_heading_like_strings(tmp_path, monkeypatch):
    """正常系（リグレッション確認）: 見出し文字列を含まない通常のケースは従来通り復元される。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    memory.save_conversation(
        question="Pythonとは何ですか？", answer="Pythonはプログラミング言語です。", thread_id="thread-a"
    )

    conversations = memory.load_conversation("thread-a")

    assert conversations == [{"question": "Pythonとは何ですか？", "answer": "Pythonはプログラミング言語です。"}]


def test_load_conversation_question_containing_answer_heading_is_restored_correctly(tmp_path, monkeypatch):
    """正常系: 質問文中に "## 回答" という文字列が含まれていても、質問・回答が途中で
    切れずに正しく復元される。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    question = "Markdownで## 回答という見出しを書くにはどうすればいいですか？"
    answer = "そのまま `## 回答` と書けば見出しになります。"
    memory.save_conversation(question=question, answer=answer, thread_id="thread-a")

    conversations = memory.load_conversation("thread-a")

    assert conversations == [{"question": question, "answer": answer}]


def test_load_conversation_answer_containing_question_heading_is_restored_correctly(tmp_path, monkeypatch):
    """正常系: 回答文中に "## 質問" という文字列が含まれていても、正しく復元される。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    question = "見出しレベル2の書き方を教えてください"
    answer = "例えば `## 質問` のように、行頭に `##` を書くと見出しになります。"
    memory.save_conversation(question=question, answer=answer, thread_id="thread-a")

    conversations = memory.load_conversation("thread-a")

    assert conversations == [{"question": question, "answer": answer}]


def test_load_conversation_both_question_and_answer_contain_heading_like_strings(tmp_path, monkeypatch):
    """複合ケース: 質問・回答の両方に "## 質問" / "## 回答" 相当の文字列が
    含まれていても、それぞれ正しく復元される。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    question = "会話ログの書式は「## 質問」の次に本文、その後「## 回答」と続きますか？"
    answer = "はい、その通りです。「## 質問」の後に質問本文、「## 回答」の後に回答本文が続きます。"
    memory.save_conversation(question=question, answer=answer, thread_id="thread-a")

    conversations = memory.load_conversation("thread-a")

    assert conversations == [{"question": question, "answer": answer}]


def test_list_threads_first_question_correct_when_question_contains_answer_heading(tmp_path, monkeypatch):
    """正常系: list_threads() の first_question も、質問文中の "## 回答" に惑わされず
    正しく（途中で切れずに）取得できる。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    question = "## 回答 という見出しの直後に本文を書く形式について教えてください"
    memory.save_conversation(question=question, answer="回答本文です", thread_id="thread-a")

    threads = memory.list_threads()

    assert threads[0]["first_question"] == question


def test_extract_qa_legacy_format_without_length_metadata_falls_back_to_regex(tmp_path, monkeypatch):
    """後方互換性: 文字数メタデータが無い旧形式ファイル（見出し文字列を含まない）は、
    従来通り正規表現ベースのフォールバックで正しく復元される。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    _write_log(tmp_path, "thread-a", "20240101_090000_aaa111_q.md", question="旧形式の質問", answer="旧形式の回答")

    conversations = memory.load_conversation("thread-a")

    assert conversations == [{"question": "旧形式の質問", "answer": "旧形式の回答"}]


def test_extract_qa_legacy_format_with_heading_like_content_is_a_known_limitation(tmp_path, monkeypatch):
    """既知の制約: 文字数メタデータが無い旧形式ファイルで、かつ質問本文中に空行を挟んで
    "## 回答" のような見出し文字列相当（"\n\n## 回答"）が含まれる場合は、非貪欲マッチの
    フォールバックが使われるため従来通り途中で切れる（この既知の制約自体を固定する
    リグレッションテストであり、修正を要求するものではない）。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    question = "前半\n\n## 回答らしき文字列\n\n後半"
    _write_log(tmp_path, "thread-a", "20240101_090000_aaa111_q.md", question=question, answer="回答本文")

    conversations = memory.load_conversation("thread-a")

    # 旧形式では非貪欲マッチにより、質問本文中の最初の "\n\n## 回答" の手前までしか復元されない
    assert conversations[0]["question"] == "前半"


def test_extract_qa_falls_back_to_regex_when_length_metadata_is_inconsistent(tmp_path, monkeypatch):
    """境界値: 文字数メタデータが本文と矛盾する（記録されたオフセットに回答見出しが
    見つからない）場合はクラッシュせず、正規表現ベースのフォールバックに切り替わる。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    path = memory.save_conversation(question="質問本文", answer="回答本文", thread_id="thread-a")
    content = path.read_text(encoding="utf-8")
    # 質問文字数メタデータを実際の値とは異なる値に書き換え、意図的に整合性を崩す
    tampered = content.replace(f"- 質問文字数: {len('質問本文')}", f"- 質問文字数: {len('質問本文') + 1}")
    path.write_text(tampered, encoding="utf-8")

    conversations = memory.load_conversation("thread-a")

    # フォールバックの正規表現でも本文自体は問題なく復元できる（見出し文字列を含まないため）
    assert conversations == [{"question": "質問本文", "answer": "回答本文"}]


# --- _validate_thread_id() / thread_id のパストラバーサル対策 ---


class TestValidateThreadId:
    """_validate_thread_id() 単体の正常系・異常系・境界値。"""

    @pytest.mark.parametrize(
        "thread_id",
        [
            "a1b2c3d4",  # new_thread_id() が生成する形式（uuid hexの先頭8文字）
            "thread-a",
            "thread_a",
            "Thread-ID_123",
            "a" * 100,  # 長さの上限は特に設けていないため、長い文字列も許可される
            "0",  # 数字1文字のみでも許可される
        ],
    )
    def test_accepts_valid_thread_id(self, thread_id):
        assert memory._validate_thread_id(thread_id) == thread_id

    @pytest.mark.parametrize(
        "thread_id",
        [
            "../../etc/passwd",
            "../secret",
            "..",
            "a/../../b",
            "/etc/passwd",
            "thread/a",
            "thread\\a",
            "",
            "thread a",  # 空白は不許可
            "thread.a",  # ドットは不許可
            "thread:a",
            "thread;rm -rf /",
            "thread\x00null",
            "日本語スレッド",  # 非ASCII文字は不許可
        ],
    )
    def test_rejects_invalid_thread_id(self, thread_id):
        with pytest.raises(ValueError):
            memory._validate_thread_id(thread_id)

    def test_rejects_non_string_thread_id(self):
        with pytest.raises(ValueError):
            memory._validate_thread_id(None)

    def test_error_message_includes_offending_value(self):
        with pytest.raises(ValueError, match=r"\.\./\.\./etc/passwd"):
            memory._validate_thread_id("../../etc/passwd")


class TestSaveConversationThreadIdValidation:
    """save_conversation() が _validate_thread_id() を経由することの確認。"""

    def test_accepts_uuid_hex_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        thread_id = memory.new_thread_id()

        path = memory.save_conversation(question="質問", answer="回答", thread_id=thread_id)

        assert path.parent == tmp_path / thread_id

    def test_accepts_hyphen_and_underscore_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        path = memory.save_conversation(question="質問", answer="回答", thread_id="my-thread_01")

        assert path.parent == tmp_path / "my-thread_01"

    def test_rejects_path_traversal_thread_id_and_does_not_create_files_outside_conversations_dir(
        self, tmp_path, monkeypatch
    ):
        conversations_dir = tmp_path / "data" / "conversations"
        conversations_dir.mkdir(parents=True)
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", conversations_dir)

        with pytest.raises(ValueError):
            memory.save_conversation(question="質問", answer="回答", thread_id="../../etc/passwd")

        # conversations_dirの外側（tmp_path直下）にファイルが作られていないことを確認する
        assert not (tmp_path / "etc").exists()

    def test_rejects_slash_containing_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        with pytest.raises(ValueError):
            memory.save_conversation(question="質問", answer="回答", thread_id="a/b")

    def test_rejects_empty_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        with pytest.raises(ValueError):
            memory.save_conversation(question="質問", answer="回答", thread_id="")


class TestLoadConversationThreadIdValidation:
    """load_conversation() が _validate_thread_id() を経由することの確認。"""

    def test_accepts_uuid_hex_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        thread_id = memory.new_thread_id()
        _write_log(tmp_path, thread_id, "20240101_090000_aaa111_q.md", question="質問", answer="回答")

        conversations = memory.load_conversation(thread_id)

        assert conversations == [{"question": "質問", "answer": "回答"}]

    def test_accepts_hyphen_and_underscore_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        _write_log(tmp_path, "my-thread_01", "20240101_090000_aaa111_q.md", question="質問", answer="回答")

        conversations = memory.load_conversation("my-thread_01")

        assert conversations == [{"question": "質問", "answer": "回答"}]

    def test_rejects_path_traversal_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        with pytest.raises(ValueError):
            memory.load_conversation("../../etc/passwd")

    def test_rejects_dotdot_alone(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        with pytest.raises(ValueError):
            memory.load_conversation("..")

    def test_rejects_empty_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        with pytest.raises(ValueError):
            memory.load_conversation("")


class TestConversationCountThreadIdValidation:
    """conversation_count() が _validate_thread_id() を経由することの確認。"""

    def test_accepts_uuid_hex_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        thread_id = memory.new_thread_id()
        memory.save_conversation("Q", "A", thread_id=thread_id)

        assert memory.conversation_count(thread_id) == 1

    def test_accepts_hyphen_and_underscore_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q", "A", thread_id="my-thread_01")

        assert memory.conversation_count("my-thread_01") == 1

    def test_rejects_path_traversal_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        with pytest.raises(ValueError):
            memory.conversation_count("../../etc/passwd")

    def test_rejects_slash_containing_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        with pytest.raises(ValueError):
            memory.conversation_count("a/b")

    def test_validates_even_when_conversations_dir_does_not_exist(self, tmp_path, monkeypatch):
        """境界値: 従来はCONVERSATIONS_DIRの存在チェックが検証より先だったため0が返っていたが、
        検証を先に行う順序に整理されたことで、ディレクトリ不在時でも不正なthread_idは
        ValueErrorになる（サイレントに0を返して見過ごされることを防ぐ）。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path / "does-not-exist")

        with pytest.raises(ValueError):
            memory.conversation_count("../../etc/passwd")

    def test_empty_string_thread_id_is_treated_as_no_thread_id_specified(self, tmp_path, monkeypatch):
        """境界値: 空文字列はPythonのif文でFalsyと判定されるため、
        `if thread_id else CONVERSATIONS_DIR` の分岐で全体カウントの経路に入り、
        _validate_thread_id() を通らない（save_conversation/load_conversationとは異なる挙動）。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q1", "A1", thread_id="thread-a")
        memory.save_conversation("Q2", "A2", thread_id="thread-b")

        assert memory.conversation_count("") == 2
