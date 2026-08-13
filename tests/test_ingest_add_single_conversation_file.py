"""ingest.add_single_conversation_file() / _ingest_file() のテスト。

app.py が会話ログ保存直後に呼ぶ軽量な単一ファイル同期経路の検証。
sync_data_dir()（DATA_DIR.rglob("*")による全件差分同期）とは異なり、
渡されたファイル1件だけを処理し、data/配下の他のファイルには一切触れない
（全件列挙を行わない）ことが本モジュールのテストの主眼になる。

実際の埋め込みモデル・Chromaは使わず、get_vectorstore() をテストごとに
軽量なフェイクベクトルストアに monkeypatch する（tests/test_ingest.py と同じ方針）。
"""

import json
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

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
    monkeypatch.setattr(ingest, "SYNC_LOCK_PATH", persist_dir / "sync.lock")

    store = _FakeVectorStore()
    monkeypatch.setattr(ingest, "get_vectorstore", lambda: store)

    return data_dir, store


def _write(data_dir, rel_path, text):
    path = data_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _on_disk_manifest():
    return json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))


def _conversation_content(is_fallback=False):
    return (
        "# 会話ログ\n\n"
        "- 日時: 2026-08-13 12:00:00\n"
        f"- 一般知識フォールバック: {'true' if is_fallback else 'false'}\n\n"
        "## 質問\n\n質問内容です。\n\n## 回答\n\n回答内容です。\n"
    )


# --- 正常系 ---


def test_new_conversation_file_is_added(fake_env):
    data_dir, store = fake_env
    path = _write(data_dir, "conversations/thread-1/convo.md", _conversation_content())

    status = ingest.add_single_conversation_file(path)

    assert status == "added"
    assert store.docs_by_id
    manifest = _on_disk_manifest()
    assert set(manifest.keys()) == {"conversations/thread-1/convo.md"}
    assert manifest["conversations/thread-1/convo.md"]["chunk_ids"]
    # 会話ログはそのスレッドID専用としてメタデータが付く
    for doc in store.docs_by_id.values():
        assert doc.metadata["thread_id"] == "thread-1"
        assert doc.metadata["is_fallback"] is False


def test_existing_conversation_file_is_updated_and_old_chunks_replaced(fake_env):
    data_dir, store = fake_env
    path = _write(data_dir, "conversations/thread-1/convo.md", _conversation_content())
    ingest.add_single_conversation_file(path)
    old_chunk_ids = set(store.docs_by_id.keys())

    path.write_text(_conversation_content() + "追記された内容です。" * 5, encoding="utf-8")
    status = ingest.add_single_conversation_file(path)

    assert status == "updated"
    assert old_chunk_ids.isdisjoint(store.docs_by_id.keys())
    assert store.docs_by_id


def test_unchanged_conversation_file_is_noop_on_second_call(fake_env):
    data_dir, store = fake_env
    path = _write(data_dir, "conversations/thread-1/convo.md", _conversation_content())
    ingest.add_single_conversation_file(path)
    docs_after_first = dict(store.docs_by_id)

    status = ingest.add_single_conversation_file(path)

    assert status == "unchanged"
    assert store.docs_by_id == docs_after_first


def test_fallback_conversation_is_tagged_is_fallback_true(fake_env):
    data_dir, store = fake_env
    path = _write(data_dir, "conversations/thread-1/convo.md", _conversation_content(is_fallback=True))

    status = ingest.add_single_conversation_file(path)

    assert status == "added"
    for doc in store.docs_by_id.values():
        assert doc.metadata["is_fallback"] is True


def test_manifest_file_created_after_first_call(fake_env):
    data_dir, _store = fake_env
    path = _write(data_dir, "conversations/thread-1/convo.md", _conversation_content())

    assert not ingest.MANIFEST_PATH.exists()
    ingest.add_single_conversation_file(path)
    assert ingest.MANIFEST_PATH.exists()


# --- 全件列挙を行わないことの確認（sync_data_dir()との差異が本関数の存在意義） ---


def test_does_not_touch_other_files_in_data_dir(fake_env):
    data_dir, store = fake_env
    _write(data_dir, "other.txt", "他の資料ファイルです。" * 5)
    path = _write(data_dir, "conversations/thread-1/convo.md", _conversation_content())

    status = ingest.add_single_conversation_file(path)

    assert status == "added"
    manifest = _on_disk_manifest()
    # other.txt は一切スキャン・処理対象になっていない
    assert set(manifest.keys()) == {"conversations/thread-1/convo.md"}


def test_does_not_call_data_dir_rglob(fake_env, monkeypatch):
    data_dir, _store = fake_env
    path = _write(data_dir, "conversations/thread-1/convo.md", _conversation_content())

    def _forbidden_rglob(self, pattern):
        raise AssertionError("add_single_conversation_file() は DATA_DIR.rglob() を呼ぶべきではない")

    monkeypatch.setattr(Path, "rglob", _forbidden_rglob)

    status = ingest.add_single_conversation_file(path)

    assert status == "added"


# --- 異常系 ---


class _FailingLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        raise ValueError(f"{self.path} は読み込めません（壊れたファイルの想定）")


def test_load_failure_returns_failed_and_is_not_recorded_in_manifest(fake_env, monkeypatch):
    data_dir, store = fake_env
    path = _write(data_dir, "conversations/thread-1/convo.md", _conversation_content())
    monkeypatch.setattr(ingest, "LOADERS", {**ingest.LOADERS, ".md": lambda p: _FailingLoader(p)})

    status = ingest.add_single_conversation_file(path)

    assert status == "failed"
    assert store.docs_by_id == {}
    assert _on_disk_manifest() == {}


def test_add_documents_failure_returns_failed(fake_env, monkeypatch):
    data_dir, store = fake_env
    path = _write(data_dir, "conversations/thread-1/convo.md", _conversation_content())

    def failing_add_documents(documents):
        raise RuntimeError("ベクトルストアへの追加に失敗")

    monkeypatch.setattr(store, "add_documents", failing_add_documents)

    status = ingest.add_single_conversation_file(path)

    assert status == "failed"
    assert _on_disk_manifest() == {}


def test_failed_file_is_retried_on_next_call(fake_env, monkeypatch):
    data_dir, store = fake_env
    path = _write(data_dir, "conversations/thread-1/convo.md", _conversation_content())

    # LOADERSの差し替えは、この読み込み失敗を再現したいブロックだけに限定する
    # （monkeypatch.context()を抜けると自動的に元のLOADERSに戻る）。
    with monkeypatch.context() as m:
        m.setattr(ingest, "LOADERS", {**ingest.LOADERS, ".md": lambda p: _FailingLoader(p)})
        assert ingest.add_single_conversation_file(path) == "failed"

    assert store.docs_by_id == {}
    assert _on_disk_manifest() == {}

    # LOADERSが元に戻った状態で再試行すると、失敗せず正常に追加される
    # （failed時にmanifestへ記録しないことで、次回同期時の再試行が保証される仕様の確認）。
    status = ingest.add_single_conversation_file(path)
    assert status == "added"


def test_lock_timeout_raises_and_does_not_write_manifest(fake_env, monkeypatch):
    data_dir, _store = fake_env
    path = _write(data_dir, "conversations/thread-1/convo.md", _conversation_content())
    monkeypatch.setattr(ingest, "SYNC_LOCK_TIMEOUT_SECONDS", 0.1)

    other_session_lock = FileLock(str(ingest.SYNC_LOCK_PATH))
    other_session_lock.acquire()
    try:
        with pytest.raises(Timeout):
            ingest.add_single_conversation_file(path)
    finally:
        other_session_lock.release()

    # ロック解放後は通常どおり処理できる
    status = ingest.add_single_conversation_file(path)
    assert status == "added"


# --- 境界値 ---


def test_pending_delete_chunk_ids_are_retried_on_unchanged_call(fake_env, monkeypatch):
    data_dir, store = fake_env
    path = _write(data_dir, "conversations/thread-1/convo.md", _conversation_content())
    ingest.add_single_conversation_file(path)

    path.write_text(_conversation_content() + "更新後の内容です。" * 5, encoding="utf-8")

    real_delete = store.delete
    monkeypatch.setattr(store, "delete", lambda ids: (_ for _ in ()).throw(RuntimeError("削除失敗")))
    status = ingest.add_single_conversation_file(path)
    assert status == "updated"

    manifest = _on_disk_manifest()
    entry = manifest["conversations/thread-1/convo.md"]
    assert entry.get("pending_delete_chunk_ids")

    # 削除が復旧すれば、次回の（内容変化なしの）呼び出し時に保留分の削除が再試行される
    monkeypatch.setattr(store, "delete", real_delete)
    status_again = ingest.add_single_conversation_file(path)
    assert status_again == "unchanged"
    manifest_after = _on_disk_manifest()
    assert "pending_delete_chunk_ids" not in manifest_after["conversations/thread-1/convo.md"]


def test_thread_id_metadata_matches_conversation_subdirectory(fake_env):
    data_dir, store = fake_env
    path = _write(data_dir, "conversations/thread-xyz/convo.md", _conversation_content())

    ingest.add_single_conversation_file(path)

    assert all(doc.metadata["thread_id"] == "thread-xyz" for doc in store.docs_by_id.values())


def test_empty_content_file_results_in_no_chunks_but_still_added(fake_env):
    data_dir, store = fake_env
    path = _write(data_dir, "conversations/thread-1/empty.md", "")

    status = ingest.add_single_conversation_file(path)

    assert status == "added"
    assert store.docs_by_id == {}
    manifest = _on_disk_manifest()
    assert manifest["conversations/thread-1/empty.md"]["chunk_ids"] == []
