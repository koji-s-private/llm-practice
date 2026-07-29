"""ingest.py の差分同期ロジック（追加・更新・削除の判定）のテスト。

実際の埋め込みモデル・Chromaは使わず、get_vectorstore() をテストごとに
軽量なフェイクベクトルストアに monkeypatch して、data/ の内容と manifest.json
に基づく差分判定だけを検証する（ドキュメントローダー・チャンク分割は実物を使う）。
"""
import json

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

    assert result == {"added": ["a.txt"], "updated": [], "removed": []}
    assert store.docs_by_id


def test_unchanged_file_is_noop_on_second_sync(fake_env):
    data_dir, store = fake_env
    _write(data_dir, "a.txt", "変わらないテキスト内容です。")
    ingest.sync_data_dir(verbose=False)
    docs_after_first_sync = dict(store.docs_by_id)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": []}
    assert store.docs_by_id == docs_after_first_sync


def test_modified_file_is_updated_and_old_chunks_replaced(fake_env):
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "元のテキスト内容です。")
    ingest.sync_data_dir(verbose=False)
    old_chunk_ids = set(store.docs_by_id.keys())

    path.write_text("変更後のまったく別のテキスト内容です。", encoding="utf-8")
    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": ["a.txt"], "removed": []}
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

    assert result == {"added": [], "updated": [], "removed": ["a.txt"]}
    assert store.docs_by_id == {}
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest == {}


def test_unsupported_extension_is_ignored(fake_env):
    data_dir, store = fake_env
    _write(data_dir, "notes.docx", "対応していない拡張子")

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": []}
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
