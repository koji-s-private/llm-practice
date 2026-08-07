"""ingest.py の差分同期ロジック（追加・更新・削除の判定）のテスト。

実際の埋め込みモデル・Chromaは使わず、get_vectorstore() をテストごとに
軽量なフェイクベクトルストアに monkeypatch して、data/ の内容と manifest.json
に基づく差分判定だけを検証する（ドキュメントローダー・チャンク分割は実物を使う）。
"""

import json
import os

import pytest

import ingest


class _FakeVectorStore:
    def __init__(self):
        self.docs_by_id = {}
        self._next_id = 0

    def add_documents(self, documents):
        ids = []
        for doc in documents:
            self._next_id += 1
            doc_id = f"chunk-{self._next_id}"
            self.docs_by_id[doc_id] = doc
            ids.append(doc_id)
        return ids

    def delete(self, ids):
        for chunk_id in ids:
            self.docs_by_id.pop(chunk_id, None)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma_db"

    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "PERSIST_DIR", persist_dir)
    monkeypatch.setattr(ingest, "MANIFEST_PATH", persist_dir / "manifest.json")

    store = _FakeVectorStore()
    monkeypatch.setattr(ingest, "get_vectorstore", lambda: store)

    return data_dir, store


def _write(data_dir, rel_path, text):
    path = data_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_new_file_is_added(fake_env):
    data_dir, store = fake_env
    _write(data_dir, "a.txt", "これは十分な長さのテキストです。" * 5)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": ["a.txt"], "updated": [], "removed": [], "failed": []}
    assert store.docs_by_id


def test_unchanged_file_is_noop_on_second_sync(fake_env):
    data_dir, store = fake_env
    _write(data_dir, "a.txt", "変わらないテキスト内容です。")
    ingest.sync_data_dir(verbose=False)
    docs_after_first_sync = dict(store.docs_by_id)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": [], "failed": []}
    assert store.docs_by_id == docs_after_first_sync


def test_modified_file_is_updated_and_old_chunks_replaced(fake_env):
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "元のテキスト内容です。")
    ingest.sync_data_dir(verbose=False)
    old_chunk_ids = set(store.docs_by_id.keys())

    path.write_text("変更後のまったく別のテキスト内容です。", encoding="utf-8")
    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": ["a.txt"], "removed": [], "failed": []}
    # 古いチャンクは削除され、新しいチャンクに置き換わっている
    assert old_chunk_ids.isdisjoint(store.docs_by_id.keys())
    assert store.docs_by_id


def test_removed_file_is_deleted_from_store_and_manifest(fake_env):
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "削除されるファイルの内容です。")
    ingest.sync_data_dir(verbose=False)
    assert store.docs_by_id

    path.unlink()
    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": ["a.txt"], "failed": []}
    assert store.docs_by_id == {}
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest == {}


class _FailingLoader:
    """load() で必ず例外を送出するダミーローダー（LOADERSの差し替え用）。

    Issue #18: 1ファイルの読み込み失敗が他のファイルの同期まで止めてしまわないことを
    検証するためのテストダブル。
    """

    def __init__(self, path):
        self.path = path

    def load(self):
        raise ValueError(f"{self.path} は読み込めません（壊れたファイルの想定）")


def test_one_file_load_failure_does_not_block_other_files(fake_env, monkeypatch):
    # 1ファイルの読み込みが失敗しても、他の正常なファイルは問題なく同期される
    data_dir, store = fake_env
    _write(data_dir, "good.txt", "正常に読み込めるテキストです。" * 5)
    _write(data_dir, "bad.txt", "壊れていることにするテキストです。" * 5)

    real_text_loader = ingest.LOADERS[".txt"]

    def flaky_loader(path):
        if "bad.txt" in path:
            return _FailingLoader(path)
        return real_text_loader(path)

    monkeypatch.setattr(ingest, "LOADERS", {**ingest.LOADERS, ".txt": flaky_loader})

    result = ingest.sync_data_dir(verbose=False)

    assert result == {
        "added": ["good.txt"],
        "updated": [],
        "removed": [],
        "failed": ["bad.txt"],
    }
    # 正常なファイルのチャンクはベクトルストアに登録されている
    sources = {doc.metadata.get("source") for doc in store.docs_by_id.values()}
    assert any("good.txt" in (s or "") for s in sources)


def test_pdf_load_failure_is_recorded_as_failed(fake_env, monkeypatch):
    # PDFは _load_pdf() 経由で読み込まれるため、そちらの失敗も同様にスキップされることを確認する
    data_dir, store = fake_env
    (data_dir / "broken.pdf").write_bytes(b"not a real pdf")

    def flaky_load_pdf(path, verbose=True):
        raise ValueError("壊れたPDFです")

    monkeypatch.setattr(ingest, "_load_pdf", flaky_load_pdf)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": [], "failed": ["broken.pdf"]}
    assert store.docs_by_id == {}


def test_failed_file_is_not_recorded_in_manifest_and_retried_next_sync(fake_env, monkeypatch):
    # 失敗したファイルはmanifestに記録されず、次回同期時に再度読み込みが試みられる（リトライ）
    data_dir, store = fake_env
    _write(data_dir, "bad.txt", "壊れていることにするテキストです。" * 5)

    load_attempts = []

    def always_failing_loader(path):
        load_attempts.append(path)
        return _FailingLoader(path)

    monkeypatch.setattr(ingest, "LOADERS", {**ingest.LOADERS, ".txt": always_failing_loader})

    result_1 = ingest.sync_data_dir(verbose=False)
    assert result_1["failed"] == ["bad.txt"]

    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert "bad.txt" not in manifest

    result_2 = ingest.sync_data_dir(verbose=False)
    assert result_2["failed"] == ["bad.txt"]
    # ファイルが変化していなくてもmanifestに記録がないため、2回とも読み込みが試みられている
    assert len(load_attempts) == 2


def test_all_files_failing_completes_without_raising(fake_env, monkeypatch):
    # 全ファイルが失敗しても例外を送出せず処理が正常に完了する（境界値）
    data_dir, store = fake_env
    _write(data_dir, "bad1.txt", "壊れていることにするテキスト1です。" * 5)
    _write(data_dir, "bad2.txt", "壊れていることにするテキスト2です。" * 5)

    monkeypatch.setattr(ingest, "LOADERS", {**ingest.LOADERS, ".txt": lambda path: _FailingLoader(path)})

    result = ingest.sync_data_dir(verbose=False)

    assert sorted(result["failed"]) == ["bad1.txt", "bad2.txt"]
    assert result["added"] == []
    assert result["updated"] == []
    assert result["removed"] == []
    assert store.docs_by_id == {}


def test_previously_synced_file_that_now_fails_keeps_old_chunks(fake_env, monkeypatch):
    # 読み込み失敗時は前回同期成功時点のインデックスがそのまま残る
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "最初は正常なテキストです。" * 5)
    ingest.sync_data_dir(verbose=False)
    old_chunk_ids = set(store.docs_by_id.keys())
    assert old_chunk_ids

    path.write_text("2回目の同期時は壊れていることにするテキストです。" * 5, encoding="utf-8")
    monkeypatch.setattr(ingest, "LOADERS", {**ingest.LOADERS, ".txt": lambda p: _FailingLoader(p)})

    result = ingest.sync_data_dir(verbose=False)

    assert result["failed"] == ["a.txt"]
    assert result["updated"] == []
    # 読み込み失敗時は前回同期成功時点のチャンクがそのまま（削除も置き換えもされない）
    assert store.docs_by_id.keys() == old_chunk_ids


def test_add_documents_failure_on_update_keeps_old_chunks_and_records_failed(fake_env, monkeypatch):
    # Issue #82: 更新時にadd_documents()が失敗しても、旧チャンクを消さず
    # result["failed"]に積んで次回リトライされるようにする（データが消える事故の防止）
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "最初は正常なテキストです。" * 5)
    ingest.sync_data_dir(verbose=False)
    old_chunk_ids = set(store.docs_by_id.keys())
    assert old_chunk_ids

    path.write_text("更新後のテキストです。" * 5, encoding="utf-8")

    def flaky_add_documents(documents):
        raise RuntimeError("埋め込みモデルの一時的な失敗を想定")

    monkeypatch.setattr(store, "add_documents", flaky_add_documents)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": [], "failed": ["a.txt"]}
    # 追加が失敗しても旧チャンクは削除されず残っている（検索対象から消えない）
    assert store.docs_by_id.keys() == old_chunk_ids

    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    # manifestは更新されない（旧chunk_idsのまま）ので、次回同期時も再試行される
    assert set(manifest["a.txt"]["chunk_ids"]) == old_chunk_ids


def test_add_documents_failure_on_new_file_has_no_old_chunks_to_keep(fake_env, monkeypatch):
    # 境界値: 新規ファイル追加時（entryが存在せず旧チャンクが無いケース）にadd_documents()が
    # 失敗しても、delete()が呼ばれたり例外で落ちたりせず、failedに記録されるだけで正常終了すること
    data_dir, store = fake_env
    _write(data_dir, "new.txt", "新規追加されるファイルの内容です。" * 5)

    def flaky_add_documents(documents):
        raise RuntimeError("埋め込みモデルの一時的な失敗を想定")

    monkeypatch.setattr(store, "add_documents", flaky_add_documents)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": [], "failed": ["new.txt"]}
    # そもそも旧チャンクが無いので、ベクトルストアには何も登録されないまま
    assert store.docs_by_id == {}

    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    # 失敗したファイルはmanifestに記録されないため、次回同期時に再試行される
    assert "new.txt" not in manifest


def test_add_documents_failure_for_one_file_does_not_block_other_files(fake_env, monkeypatch):
    # 複数ファイル同期中に1ファイルのadd_documents()だけが失敗しても、
    # 他のファイルの同期処理は中断されず継続すること
    data_dir, store = fake_env
    _write(data_dir, "good.txt", "正常に追加できるテキストです。" * 5)
    _write(data_dir, "bad.txt", "追加に失敗することにするテキストです。" * 5)

    real_add_documents = store.add_documents

    def flaky_add_documents(documents):
        sources = {doc.metadata.get("source", "") for doc in documents}
        if any("bad.txt" in s for s in sources):
            raise RuntimeError("埋め込みモデルの一時的な失敗を想定")
        return real_add_documents(documents)

    monkeypatch.setattr(store, "add_documents", flaky_add_documents)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {
        "added": ["good.txt"],
        "updated": [],
        "removed": [],
        "failed": ["bad.txt"],
    }
    # 失敗したファイルの分は登録されていないが、正常なファイルのチャンクは登録されている
    sources = {doc.metadata.get("source") for doc in store.docs_by_id.values()}
    assert any("good.txt" in (s or "") for s in sources)
    assert not any("bad.txt" in (s or "") for s in sources)

    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert "good.txt" in manifest
    assert "bad.txt" not in manifest


def test_unsupported_extension_is_ignored(fake_env):
    data_dir, store = fake_env
    _write(data_dir, "notes.docx", "対応していない拡張子")

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": [], "failed": []}
    assert store.docs_by_id == {}


def test_top_level_file_is_tagged_with_global_thread_id(fake_env):
    data_dir, store = fake_env
    _write(data_dir, "doc.txt", "共通ナレッジとして扱われるべき文書です。" * 5)

    ingest.sync_data_dir(verbose=False)

    thread_ids = {doc.metadata["thread_id"] for doc in store.docs_by_id.values()}
    assert thread_ids == {ingest.GLOBAL_THREAD_ID}


def test_conversation_log_is_tagged_with_its_own_thread_id(fake_env):
    data_dir, store = fake_env
    _write(
        data_dir,
        "conversations/thread-xyz/log.md",
        "このスレッド専用の会話ログです。" * 5,
    )

    ingest.sync_data_dir(verbose=False)

    thread_ids = {doc.metadata["thread_id"] for doc in store.docs_by_id.values()}
    assert thread_ids == {"thread-xyz"}


def test_legacy_flat_conversation_file_falls_back_to_global_thread_id(fake_env):
    # スレッド機能追加前の古い保存形式（サブフォルダなし）は後方互換で
    # GLOBAL_THREAD_ID として扱われる想定
    data_dir, store = fake_env
    _write(
        data_dir,
        "conversations/legacy.md",
        "サブフォルダのない古い形式の会話ログです。" * 5,
    )

    ingest.sync_data_dir(verbose=False)

    thread_ids = {doc.metadata["thread_id"] for doc in store.docs_by_id.values()}
    assert thread_ids == {ingest.GLOBAL_THREAD_ID}


@pytest.mark.parametrize(
    ("filename", "expected_name"),
    [
        ("../../etc/passwd_test", "passwd_test"),
        ("../../../tmp/evil.txt", "evil.txt"),
        ("../sibling_dir_escape.txt", "sibling_dir_escape.txt"),
    ],
)
def test_safe_upload_dest_strips_directory_traversal(fake_env, filename, expected_name):
    # ディレクトリ部分（../ 等）は無害化され、DATA_DIR配下のベース名のみのパスになる
    data_dir, _store = fake_env

    dest = ingest.safe_upload_dest(filename)

    assert dest is not None
    assert dest.parent == data_dir.resolve()
    assert dest.name == expected_name


def test_safe_upload_dest_rejects_bare_dotdot(fake_env):
    # ".." そのものは無害化してもDATA_DIR自身/外を指してしまうため拒否する
    data_dir, _store = fake_env

    dest = ingest.safe_upload_dest("..")

    assert dest is None


@pytest.mark.parametrize(
    "filename",
    ["report.pdf", "notes.txt", "my file (2).md"],
)
def test_safe_upload_dest_accepts_normal_filenames(fake_env, filename):
    data_dir, _store = fake_env

    dest = ingest.safe_upload_dest(filename)

    assert dest is not None
    assert dest.parent == data_dir.resolve()
    assert dest.name == filename


# --- data_dir_signature(): 内容を読まないstat()ベースの軽量変更検知（Issue #70） ---


def test_data_dir_signature_is_zero_when_no_target_files(fake_env):
    # 対象ファイルが1つもない場合は (0, 0.0)
    data_dir, _store = fake_env

    assert ingest.data_dir_signature() == (0, 0.0)


def test_data_dir_signature_is_zero_when_data_dir_missing(fake_env):
    # DATA_DIR自体が存在しない場合も (0, 0.0)（存在しない前提で例外にならないことも確認）
    data_dir, _store = fake_env
    data_dir.rmdir()

    assert ingest.data_dir_signature() == (0, 0.0)


def test_data_dir_signature_counts_single_file_with_its_mtime(fake_env):
    # ファイルを1つ追加すると (1, そのファイルのmtime) を返す
    data_dir, _store = fake_env
    path = _write(data_dir, "a.txt", "1件目のファイルです。")
    os.utime(path, (1_700_000_000, 1_700_000_000))

    assert ingest.data_dir_signature() == (1, 1_700_000_000.0)


def test_data_dir_signature_increases_with_second_file(fake_env):
    # ファイルをもう1つ追加すると件数が増え、最新mtimeも新しい方に更新される
    data_dir, _store = fake_env
    path1 = _write(data_dir, "a.txt", "1件目のファイルです。")
    os.utime(path1, (1_700_000_000, 1_700_000_000))
    path2 = _write(data_dir, "b.txt", "2件目のファイルです。")
    os.utime(path2, (1_700_000_100, 1_700_000_100))

    assert ingest.data_dir_signature() == (2, 1_700_000_100.0)


def test_data_dir_signature_ignores_unsupported_extension(fake_env):
    # LOADERSに無い拡張子（対象外）のファイルは件数にもmtime計算にも含まれない
    data_dir, _store = fake_env
    path = _write(data_dir, "a.txt", "対象ファイルです。")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    unsupported = _write(data_dir, "notes.docx", "対応していない拡張子です。")
    os.utime(unsupported, (1_800_000_000, 1_800_000_000))  # より新しいmtimeでも無視される

    assert ingest.data_dir_signature() == (1, 1_700_000_000.0)


def test_data_dir_signature_decreases_when_file_removed(fake_env):
    # ファイルを削除すると件数が減る
    data_dir, _store = fake_env
    path1 = _write(data_dir, "a.txt", "1件目のファイルです。")
    os.utime(path1, (1_700_000_000, 1_700_000_000))
    path2 = _write(data_dir, "b.txt", "2件目のファイルです。")
    os.utime(path2, (1_700_000_100, 1_700_000_100))
    assert ingest.data_dir_signature()[0] == 2

    path2.unlink()

    assert ingest.data_dir_signature() == (1, 1_700_000_000.0)


def test_data_dir_signature_reflects_mtime_update_on_content_change(fake_env):
    # 既存ファイルの中身だけ変更（mtime更新）すると、件数は同じでも最新mtimeが変わる
    data_dir, _store = fake_env
    path = _write(data_dir, "a.txt", "元の内容です。")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    before = ingest.data_dir_signature()
    assert before == (1, 1_700_000_000.0)

    path.write_text("変更後の内容です。", encoding="utf-8")
    os.utime(path, (1_700_000_500, 1_700_000_500))
    after = ingest.data_dir_signature()

    assert after[0] == before[0] == 1
    assert after[1] != before[1]
    assert after == (1, 1_700_000_500.0)
