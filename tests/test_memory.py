"""memory.py の会話ログ保存・件数カウント・過去スレッド一覧・再開機能のテスト。"""

import logging
import os
import re
from datetime import datetime

import pytest
from langchain_core.documents import Document

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

    assert conversations == [
        {"question": "質問内容", "answer": "回答内容", "created_at": datetime(2024, 1, 1, 9, 0), "sources": []}
    ]


def test_load_conversation_created_at_has_expected_keys(tmp_path, monkeypatch):
    """境界値: load_conversation()の各要素は question/answer/created_at/sources の4キーのみを持つ。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    _write_log(tmp_path, "thread-a", "20240101_090000_aaa111_q.md", question="質問内容", answer="回答内容")

    conversations = memory.load_conversation("thread-a")

    assert set(conversations[0].keys()) == {"question", "answer", "created_at", "sources"}
    assert isinstance(conversations[0]["created_at"], datetime)


def test_load_conversation_created_at_falls_back_to_mtime_for_unparseable_filename(tmp_path, monkeypatch):
    """異常系境界値: ファイル名がsave_conversationの命名規則（先頭15文字が日時）と一致しない場合、
    load_conversation()のcreated_atもstrptime失敗によりファイルのmtimeにフォールバックする
    （list_threads()と同じ_parse_created_at()を共有していることの確認）。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    thread_dir = tmp_path / "thread-a"
    thread_dir.mkdir()
    path = thread_dir / "not-a-timestamp-name.md"
    path.write_text("## 質問\n\n質問\n\n## 回答\n\n回答\n", encoding="utf-8")
    fixed_mtime = datetime(2023, 5, 5, 12, 30, 0).timestamp()
    os.utime(path, (fixed_mtime, fixed_mtime))

    conversations = memory.load_conversation("thread-a")

    assert conversations[0]["created_at"] == datetime.fromtimestamp(fixed_mtime)


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

    assert conversations == [{"question": "", "answer": "", "created_at": datetime(2024, 1, 1, 9, 0), "sources": []}]


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
    path = memory.save_conversation(
        question="Pythonとは何ですか？", answer="Pythonはプログラミング言語です。", thread_id="thread-a"
    )

    conversations = memory.load_conversation("thread-a")

    assert conversations == [
        {
            "question": "Pythonとは何ですか？",
            "answer": "Pythonはプログラミング言語です。",
            "created_at": memory._parse_created_at(path),
            "sources": [],
        }
    ]


def test_load_conversation_question_containing_answer_heading_is_restored_correctly(tmp_path, monkeypatch):
    """正常系: 質問文中に "## 回答" という文字列が含まれていても、質問・回答が途中で
    切れずに正しく復元される。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    question = "Markdownで## 回答という見出しを書くにはどうすればいいですか？"
    answer = "そのまま `## 回答` と書けば見出しになります。"
    path = memory.save_conversation(question=question, answer=answer, thread_id="thread-a")

    conversations = memory.load_conversation("thread-a")

    assert conversations == [
        {"question": question, "answer": answer, "created_at": memory._parse_created_at(path), "sources": []}
    ]


def test_load_conversation_answer_containing_question_heading_is_restored_correctly(tmp_path, monkeypatch):
    """正常系: 回答文中に "## 質問" という文字列が含まれていても、正しく復元される。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    question = "見出しレベル2の書き方を教えてください"
    answer = "例えば `## 質問` のように、行頭に `##` を書くと見出しになります。"
    path = memory.save_conversation(question=question, answer=answer, thread_id="thread-a")

    conversations = memory.load_conversation("thread-a")

    assert conversations == [
        {"question": question, "answer": answer, "created_at": memory._parse_created_at(path), "sources": []}
    ]


def test_load_conversation_both_question_and_answer_contain_heading_like_strings(tmp_path, monkeypatch):
    """複合ケース: 質問・回答の両方に "## 質問" / "## 回答" 相当の文字列が
    含まれていても、それぞれ正しく復元される。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    question = "会話ログの書式は「## 質問」の次に本文、その後「## 回答」と続きますか？"
    answer = "はい、その通りです。「## 質問」の後に質問本文、「## 回答」の後に回答本文が続きます。"
    path = memory.save_conversation(question=question, answer=answer, thread_id="thread-a")

    conversations = memory.load_conversation("thread-a")

    assert conversations == [
        {"question": question, "answer": answer, "created_at": memory._parse_created_at(path), "sources": []}
    ]


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

    assert conversations == [
        {"question": "旧形式の質問", "answer": "旧形式の回答", "created_at": datetime(2024, 1, 1, 9, 0), "sources": []}
    ]


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
    saved_path = memory.save_conversation(question="質問本文", answer="回答本文", thread_id="thread-a")
    file_content = saved_path.read_text(encoding="utf-8")
    # 質問文字数メタデータを実際の値とは異なる値に書き換え、意図的に整合性を崩す
    tampered = file_content.replace(f"- 質問文字数: {len('質問本文')}", f"- 質問文字数: {len('質問本文') + 1}")
    saved_path.write_text(tampered, encoding="utf-8")

    conversations = memory.load_conversation("thread-a")

    # フォールバックの正規表現でも本文自体は問題なく復元できる（見出し文字列を含まないため）
    assert conversations == [
        {
            "question": "質問本文",
            "answer": "回答本文",
            "created_at": memory._parse_created_at(saved_path),
            "sources": [],
        }
    ]


# --- sources（参照元ドキュメント）の永続化・復元 ---


def test_save_conversation_without_sources_omits_sources_section(tmp_path, monkeypatch):
    """sourcesを渡さない場合、参照元セクションはMarkdownに書き込まれない。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    path = memory.save_conversation(question="質問", answer="回答", thread_id="thread-a")

    content = path.read_text(encoding="utf-8")
    assert "## 参照元" not in content
    assert "参照元文字数" not in content


def test_save_conversation_with_empty_sources_list_omits_sources_section(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

    path = memory.save_conversation(question="質問", answer="回答", thread_id="thread-a", sources=[])

    content = path.read_text(encoding="utf-8")
    assert "## 参照元" not in content


def test_save_and_load_conversation_roundtrips_sources(tmp_path, monkeypatch):
    """正常系: sources付きで保存した会話が、Document互換のオブジェクトとして復元される。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    sources = [
        Document(page_content="本文1", metadata={"source": "data/a.txt"}),
        Document(page_content="本文2", metadata={"source": "data/b.pdf", "page": 3}),
    ]

    memory.save_conversation(question="質問", answer="回答", thread_id="thread-a", sources=sources)
    conversations = memory.load_conversation("thread-a")

    loaded_sources = conversations[0]["sources"]
    assert len(loaded_sources) == 2
    assert loaded_sources[0].page_content == "本文1"
    assert loaded_sources[0].metadata == {"source": "data/a.txt"}
    assert loaded_sources[1].page_content == "本文2"
    assert loaded_sources[1].metadata == {"source": "data/b.pdf", "page": 3}


def test_load_conversation_without_sources_returns_empty_list(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    memory.save_conversation(question="質問", answer="回答", thread_id="thread-a")

    conversations = memory.load_conversation("thread-a")

    assert conversations[0]["sources"] == []


def test_load_conversation_legacy_file_without_sources_section_returns_empty_sources(tmp_path, monkeypatch):
    """後方互換性: sources未対応の旧形式ファイルも、sourcesは空リストとしてクラッシュせず読み込める。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    _write_log(tmp_path, "thread-a", "20240101_090000_aaa111_q.md", question="旧形式の質問", answer="旧形式の回答")

    conversations = memory.load_conversation("thread-a")

    assert conversations[0]["sources"] == []


def test_load_conversation_falls_back_to_empty_sources_when_json_is_corrupted(tmp_path, monkeypatch):
    """異常系: 参照元文字数メタデータはあるがJSONとして壊れている場合もクラッシュせず空リストにする。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    saved_path = memory.save_conversation(
        question="質問",
        answer="回答",
        thread_id="thread-a",
        sources=[Document(page_content="本文", metadata={"source": "data/a.txt"})],
    )
    content = saved_path.read_text(encoding="utf-8")
    broken = re.sub(r"(## 参照元\n\n).*", r"\1{not valid json", content, flags=re.DOTALL)
    # 壊れたJSONの文字数に合わせてメタデータの参照元文字数も更新しておく
    broken_json_len = len(broken.split("## 参照元\n\n", 1)[1])
    broken = re.sub(r"- 参照元文字数: \d+", f"- 参照元文字数: {broken_json_len}", broken)
    saved_path.write_text(broken, encoding="utf-8")

    conversations = memory.load_conversation("thread-a")

    assert conversations[0]["sources"] == []
    assert conversations[0]["question"] == "質問"
    assert conversations[0]["answer"] == "回答"


def test_save_conversation_sources_length_metadata_matches_json_length(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    sources = [Document(page_content="本文", metadata={"source": "data/a.txt"})]

    path = memory.save_conversation(question="質問", answer="回答", thread_id="thread-a", sources=sources)

    content = path.read_text(encoding="utf-8")
    expected_json = memory._serialize_sources(sources)
    assert f"- 参照元文字数: {len(expected_json)}" in content


def test_save_and_load_conversation_roundtrips_sources_with_special_characters(tmp_path, monkeypatch):
    """境界値: metadata/page_contentに日本語・ダブルクォート・改行が含まれていても、
    JSON往復（save→load）で情報が壊れずそのまま復元される。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    sources = [
        Document(
            page_content='複数行の本文です。\n"引用符"も含みます。\n日本語もOK。',
            metadata={"source": '特殊/"文字".txt', "note": "改行\nあり"},
        )
    ]

    memory.save_conversation(question="質問", answer="回答", thread_id="thread-a", sources=sources)
    conversations = memory.load_conversation("thread-a")

    loaded = conversations[0]["sources"][0]
    assert loaded.page_content == sources[0].page_content
    assert loaded.metadata == sources[0].metadata


def test_load_conversation_falls_back_to_empty_sources_when_json_entry_missing_key(tmp_path, monkeypatch):
    """異常系: JSONとしては妥当だが必須キー（page_content等）が一部欠損している場合も
    クラッシュせず空リストにフォールバックする。"""
    monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
    saved_path = memory.save_conversation(
        question="質問",
        answer="回答",
        thread_id="thread-a",
        sources=[Document(page_content="本文", metadata={"source": "data/a.txt"})],
    )
    content = saved_path.read_text(encoding="utf-8")
    partial_json = '[{"metadata": {"source": "data/a.txt"}}]'
    replaced = re.sub(r"(## 参照元\n\n).*(\n)$", rf"\g<1>{partial_json}\g<2>", content, flags=re.DOTALL)
    replaced = re.sub(r"- 参照元文字数: \d+", f"- 参照元文字数: {len(partial_json)}", replaced)
    saved_path.write_text(replaced, encoding="utf-8")

    conversations = memory.load_conversation("thread-a")

    assert conversations[0]["sources"] == []
    assert conversations[0]["question"] == "質問"
    assert conversations[0]["answer"] == "回答"


# --- save_thread_title() / load_thread_title()（スレッドの任意タイトル） ---


class TestThreadTitle:
    def test_load_thread_title_returns_none_when_not_saved(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q", "A", thread_id="thread-a")

        assert memory.load_thread_title("thread-a") is None

    def test_save_and_load_thread_title_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q", "A", thread_id="thread-a")

        memory.save_thread_title("thread-a", "経費精算について")

        assert memory.load_thread_title("thread-a") == "経費精算について"

    def test_save_and_load_thread_title_roundtrip_with_emoji(self, tmp_path, monkeypatch):
        """境界値: 絵文字（サロゲートペア）を含むタイトルも欠損なく保存・読込できる。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q", "A", thread_id="thread-a")

        memory.save_thread_title("thread-a", "経費精算 🎉📌")

        assert memory.load_thread_title("thread-a") == "経費精算 🎉📌"

    def test_save_thread_title_strips_surrounding_whitespace(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q", "A", thread_id="thread-a")

        memory.save_thread_title("thread-a", "  タイトル  ")

        assert memory.load_thread_title("thread-a") == "タイトル"

    def test_save_thread_title_overwrites_existing_title(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q", "A", thread_id="thread-a")
        memory.save_thread_title("thread-a", "最初のタイトル")

        memory.save_thread_title("thread-a", "上書き後のタイトル")

        assert memory.load_thread_title("thread-a") == "上書き後のタイトル"

    @pytest.mark.parametrize("blank_title", ["", "   ", "\n\t"])
    def test_save_thread_title_treats_blank_as_unset(self, tmp_path, monkeypatch, blank_title):
        """境界値: 空文字列・空白のみのタイトルは「未設定」として扱い保存しない。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q", "A", thread_id="thread-a")

        memory.save_thread_title("thread-a", blank_title)

        assert memory.load_thread_title("thread-a") is None

    def test_save_thread_title_blank_deletes_previously_saved_title(self, tmp_path, monkeypatch):
        """境界値: 既にタイトルが設定済みのスレッドに空文字列を保存すると未設定に戻る。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q", "A", thread_id="thread-a")
        memory.save_thread_title("thread-a", "設定済みタイトル")

        memory.save_thread_title("thread-a", "   ")

        assert memory.load_thread_title("thread-a") is None

    def test_thread_titles_are_isolated_per_thread(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q1", "A1", thread_id="thread-a")
        memory.save_conversation("Q2", "A2", thread_id="thread-b")

        memory.save_thread_title("thread-a", "スレッドAのタイトル")

        assert memory.load_thread_title("thread-a") == "スレッドAのタイトル"
        assert memory.load_thread_title("thread-b") is None

    def test_load_thread_title_returns_none_when_thread_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        assert memory.load_thread_title("no-such-thread") is None

    def test_save_thread_title_creates_thread_dir_if_missing(self, tmp_path, monkeypatch):
        """境界値: スレッドフォルダがまだ無い状態でもタイトル保存自体はクラッシュしない。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        memory.save_thread_title("new-thread", "タイトル")

        assert memory.load_thread_title("new-thread") == "タイトル"

    def test_load_thread_title_returns_none_for_corrupted_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        thread_dir = tmp_path / "thread-a"
        thread_dir.mkdir()
        (thread_dir / memory.THREAD_TITLE_FILENAME).write_bytes(b"\xff\xfe broken")

        assert memory.load_thread_title("thread-a") is None

    def test_save_thread_title_rejects_invalid_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        with pytest.raises(ValueError):
            memory.save_thread_title("../../etc/passwd", "タイトル")

    def test_load_thread_title_rejects_invalid_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        with pytest.raises(ValueError):
            memory.load_thread_title("../../etc/passwd")

    def test_save_thread_title_no_leftover_tmp_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q", "A", thread_id="thread-a")

        memory.save_thread_title("thread-a", "タイトル")

        tmp_files = list((tmp_path / "thread-a").glob("*.txt.tmp"))
        assert tmp_files == []


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

        assert conversations == [
            {"question": "質問", "answer": "回答", "created_at": datetime(2024, 1, 1, 9, 0), "sources": []}
        ]

    def test_accepts_hyphen_and_underscore_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        _write_log(tmp_path, "my-thread_01", "20240101_090000_aaa111_q.md", question="質問", answer="回答")

        conversations = memory.load_conversation("my-thread_01")

        assert conversations == [
            {"question": "質問", "answer": "回答", "created_at": datetime(2024, 1, 1, 9, 0), "sources": []}
        ]

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

    def test_rejects_empty_thread_id(self, tmp_path, monkeypatch):
        """境界値: 空文字列はNoneとは区別され、save_conversation/load_conversationと同様に
        _validate_thread_id() を通ってValueErrorになる（全体カウントにはならない）。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q1", "A1", thread_id="thread-a")
        memory.save_conversation("Q2", "A2", thread_id="thread-b")

        with pytest.raises(ValueError):
            memory.conversation_count("")

    def test_none_thread_id_returns_overall_count(self, tmp_path, monkeypatch):
        """正常系: thread_idにNone（省略時のデフォルト）を渡した場合は全スレッド合計を返す。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q1", "A1", thread_id="thread-a")
        memory.save_conversation("Q2", "A2", thread_id="thread-b")

        assert memory.conversation_count(None) == 2
        assert memory.conversation_count() == 2

    def test_valid_format_but_nonexistent_thread_id_returns_zero(self, tmp_path, monkeypatch):
        """境界値: 形式は正当だが会話ログが1件も無いthread_id（存在しないスレッド）を
        指定した場合、他スレッドの件数を巻き込まず0を返す。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q1", "A1", thread_id="thread-a")

        assert memory.conversation_count("thread-does-not-exist") == 0

    def test_rejects_dot_containing_thread_id(self, tmp_path, monkeypatch):
        """異常系: _validate_thread_id() が禁止するドットを含む値もconversation_count()側で
        ValueErrorになる（save_conversation/load_conversationと同じ検証を共有していることの確認）。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        with pytest.raises(ValueError):
            memory.conversation_count("thread.a")


# --- _read_text_safe() / list_threads() / load_conversation():
#     不正なUTF-8バイト列を含む破損ファイルへの耐性 ---


def _write_corrupted_file(base_dir, thread_id, filename):
    """不正なUTF-8バイト列を含む破損ファイルを直接書き込むテスト用ヘルパー。

    0x80-0xFF帯の単独バイトはUTF-8として不正な組み合わせになりうる（0xffは
    どのUTF-8シーケンスの先頭バイトとしても不正）ため、read_text(encoding="utf-8")で
    UnicodeDecodeErrorを再現できる。
    """
    thread_dir = base_dir / thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)
    path = thread_dir / filename
    path.write_bytes(b"# \xff\xfe broken bytes not valid utf-8")
    return path


class TestReadTextSafe:
    """_read_text_safe() 単体の正常系・異常系。"""

    def test_returns_content_for_valid_utf8_file(self, tmp_path):
        path = tmp_path / "ok.md"
        path.write_text("正常なUTF-8テキストです", encoding="utf-8")

        assert memory._read_text_safe(path) == "正常なUTF-8テキストです"

    def test_returns_none_for_invalid_utf8_bytes(self, tmp_path):
        path = tmp_path / "broken.md"
        path.write_bytes(b"\xff\xfe\x00broken")

        assert memory._read_text_safe(path) is None

    def test_logs_warning_when_decode_fails(self, tmp_path, caplog):
        path = tmp_path / "broken.md"
        path.write_bytes(b"\xff\xfe\x00broken")

        with caplog.at_level(logging.WARNING, logger="memory"):
            memory._read_text_safe(path)

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert str(path) in warnings[0].getMessage()

    def test_returns_none_for_nonexistent_file(self, tmp_path):
        """異常系: OSError(FileNotFoundError)系も捕捉されNoneを返す。"""
        path = tmp_path / "does-not-exist.md"

        assert memory._read_text_safe(path) is None

    def test_returns_none_when_permission_denied(self, tmp_path):
        """異常系: 読み込み権限が無いファイル(OSError系)も捕捉されNoneを返す。

        root権限（例: CI/コンテナ環境）ではパーミッションが無視され読み込めてしまうため、
        その場合はこのテスト自体をスキップする。
        """
        if os.name != "posix":
            pytest.skip("POSIXパーミッションに依存するテストのためスキップ")
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("root権限ではファイルパーミッションが無視されるためスキップ")

        path = tmp_path / "no-permission.md"
        path.write_text("読めないはずの内容", encoding="utf-8")
        path.chmod(0o000)
        try:
            assert memory._read_text_safe(path) is None
        finally:
            path.chmod(0o644)


class TestListThreadsWithCorruptedFile:
    """list_threads(): 破損ファイル（不正なUTF-8）混在時のクラッシュ耐性。"""

    def test_does_not_raise_when_first_file_is_corrupted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        _write_corrupted_file(tmp_path, "thread-a", "20240101_090000_aaa111_q.md")

        threads = memory.list_threads()

        assert len(threads) == 1
        assert threads[0]["thread_id"] == "thread-a"
        assert threads[0]["first_question"] == ""

    def test_corrupted_thread_is_not_excluded_and_other_threads_are_listed_correctly(self, tmp_path, monkeypatch):
        """境界値: 破損ファイルを含むスレッドが一覧から除外されず、かつ正常な
        他スレッドも正しく一覧に含まれ続けること。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        _write_corrupted_file(tmp_path, "thread-broken", "20240101_090000_aaa111_q.md")
        _write_log(tmp_path, "thread-ok", "20240102_090000_bbb222_q.md", question="正常な質問です")

        threads = memory.list_threads()

        thread_ids = {t["thread_id"] for t in threads}
        assert thread_ids == {"thread-broken", "thread-ok"}
        ok_thread = next(t for t in threads if t["thread_id"] == "thread-ok")
        assert ok_thread["first_question"] == "正常な質問です"
        broken_thread = next(t for t in threads if t["thread_id"] == "thread-broken")
        assert broken_thread["first_question"] == ""
        assert broken_thread["count"] == 1

    def test_logs_warning_but_does_not_raise(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        _write_corrupted_file(tmp_path, "thread-a", "20240101_090000_aaa111_q.md")

        with caplog.at_level(logging.WARNING, logger="memory"):
            threads = memory.list_threads()

        assert len(threads) == 1
        assert any(r.levelname == "WARNING" for r in caplog.records)


class TestLoadConversationWithCorruptedFile:
    """load_conversation(): 破損ファイル（不正なUTF-8）混在時のクラッシュ耐性。"""

    def test_skips_corrupted_file_and_returns_empty_list_when_only_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        _write_corrupted_file(tmp_path, "thread-a", "20240101_090000_aaa111_q.md")

        conversations = memory.load_conversation("thread-a")

        assert conversations == []

    def test_skips_corrupted_file_and_returns_only_valid_messages(self, tmp_path, monkeypatch):
        """境界値: 正常なファイルと破損ファイルが混在する場合、破損ファイルのみ
        スキップされ、正常なメッセージは処理が継続してすべて返る。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        _write_log(tmp_path, "thread-a", "20240101_090000_aaa111_first.md", question="1番目", answer="回答1")
        _write_corrupted_file(tmp_path, "thread-a", "20240101_093000_bbb222_broken.md")
        _write_log(tmp_path, "thread-a", "20240101_100000_ccc333_third.md", question="3番目", answer="回答3")

        conversations = memory.load_conversation("thread-a")

        assert conversations == [
            {"question": "1番目", "answer": "回答1", "created_at": datetime(2024, 1, 1, 9, 0), "sources": []},
            {"question": "3番目", "answer": "回答3", "created_at": datetime(2024, 1, 1, 10, 0), "sources": []},
        ]

    def test_logs_warning_but_does_not_raise(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        _write_corrupted_file(tmp_path, "thread-a", "20240101_090000_aaa111_q.md")

        with caplog.at_level(logging.WARNING, logger="memory"):
            conversations = memory.load_conversation("thread-a")

        assert conversations == []
        assert any(r.levelname == "WARNING" for r in caplog.records)


# --- save_conversation(): アトミック書き込み（一時ファイル + os.replace）への変更後も
#     従来の入出力仕様が壊れていないこと ---


class TestSaveConversationAtomicWrite:
    def test_no_leftover_tmp_file_after_successful_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        path = memory.save_conversation(question="質問", answer="回答", thread_id="thread-a")

        tmp_files = list(path.parent.glob("*.md.tmp"))
        assert tmp_files == []
        assert path.exists()
        assert path.suffix == ".md"

    def test_final_file_content_matches_conventional_format_after_atomic_write(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        path = memory.save_conversation(question="質問内容", answer="回答内容", thread_id="thread-a")

        content = path.read_text(encoding="utf-8")
        assert content.startswith("# 会話ログ\n\n")
        assert "## 質問\n\n質問内容" in content
        assert "## 回答\n\n回答内容" in content

    def test_uses_os_replace_to_move_tmp_file_into_place(self, tmp_path, monkeypatch):
        """os.replace()が一時ファイルパス(*.md.tmp)から最終パス(*.md)への
        アトミック配置に実際に使われていることを確認する。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        calls = []
        original_replace = os.replace

        def _spy_replace(src, dst):
            calls.append((str(src), str(dst)))
            return original_replace(src, dst)

        monkeypatch.setattr(memory.os, "replace", _spy_replace)

        path = memory.save_conversation(question="質問", answer="回答", thread_id="thread-a")

        assert len(calls) == 1
        src, dst = calls[0]
        assert src.endswith(".md.tmp")
        assert dst == str(path)

    def test_saved_conversation_is_immediately_loadable(self, tmp_path, monkeypatch):
        """回帰確認: アトミック書き込みへの変更後もload_conversation()から
        正しく読み戻せること。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        saved_path = memory.save_conversation(question="質問A", answer="回答A", thread_id="thread-a")

        conversations = memory.load_conversation("thread-a")

        assert conversations == [
            {"question": "質問A", "answer": "回答A", "created_at": memory._parse_created_at(saved_path), "sources": []}
        ]


class TestDeleteThread:
    def test_deletes_thread_directory_and_returns_true(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q", "A", thread_id="thread-a")

        result = memory.delete_thread("thread-a")

        assert result is True
        assert not (tmp_path / "thread-a").exists()

    def test_deletes_thread_with_multiple_conversation_files_and_title(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q1", "A1", thread_id="thread-a")
        memory.save_conversation("Q2", "A2", thread_id="thread-a")
        memory.save_thread_title("thread-a", "タイトル")

        result = memory.delete_thread("thread-a")

        assert result is True
        assert not (tmp_path / "thread-a").exists()

    def test_returns_false_when_thread_does_not_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        assert memory.delete_thread("no-such-thread") is False

    def test_returns_false_when_conversations_dir_itself_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path / "not-created-yet")

        assert memory.delete_thread("thread-a") is False

    def test_does_not_affect_other_threads(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        memory.save_conversation("Q1", "A1", thread_id="thread-a")
        memory.save_conversation("Q2", "A2", thread_id="thread-b")

        memory.delete_thread("thread-a")

        assert not (tmp_path / "thread-a").exists()
        assert (tmp_path / "thread-b").exists()
        assert memory.conversation_count("thread-b") == 1

    def test_rejects_invalid_thread_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)

        with pytest.raises(ValueError):
            memory.delete_thread("../../etc/passwd")

    def test_rejects_thread_id_when_path_is_a_file_not_a_directory(self, tmp_path, monkeypatch):
        """境界値: 同名のファイル（ディレクトリではない）が存在する場合は削除せずFalseを返す。"""
        monkeypatch.setattr(memory, "CONVERSATIONS_DIR", tmp_path)
        (tmp_path / "thread-a").write_text("dummy", encoding="utf-8")

        assert memory.delete_thread("thread-a") is False
        assert (tmp_path / "thread-a").exists()
