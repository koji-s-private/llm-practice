"""ingest.py の埋め込みモデル変更検知（manifest.jsonの_embedding_modelキー）のテスト。

sync_data_dir() / add_single_conversation_file() 起動時に、manifestへ記録済みの
埋め込みモデル名と現在のrag_chain.EMBEDDING_MODEL_NAMEを比較し、不一致の場合に
警告ログを出す挙動（自動での再構築は行わない）を検証する。

実際の埋め込みモデル・Chromaは使わず、get_vectorstore() をテストごとに
軽量なフェイクベクトルストアに monkeypatch する（tests/test_ingest.py と同じ方針）。
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
    monkeypatch.setattr(ingest, "SYNC_LOCK_PATH", persist_dir / "sync.lock")
    monkeypatch.setattr(ingest, "EMBEDDING_MODEL_NAME", "model-a")

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


# --- _embedding_model_mismatch() / check_embedding_model_mismatch() ---


def test_mismatch_helper_returns_none_when_key_absent():
    assert ingest._embedding_model_mismatch({}) is None


def test_mismatch_helper_returns_none_when_key_matches(monkeypatch):
    monkeypatch.setattr(ingest, "EMBEDDING_MODEL_NAME", "model-a")
    assert ingest._embedding_model_mismatch({"_embedding_model": "model-a"}) is None


def test_mismatch_helper_returns_recorded_and_current_when_differs(monkeypatch):
    monkeypatch.setattr(ingest, "EMBEDDING_MODEL_NAME", "model-b")
    assert ingest._embedding_model_mismatch({"_embedding_model": "model-a"}) == ("model-a", "model-b")


def test_check_embedding_model_mismatch_reads_from_manifest_file(fake_env, monkeypatch):
    data_dir, _store = fake_env
    ingest.PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    ingest._save_manifest({"_embedding_model": "model-a"})

    assert ingest.check_embedding_model_mismatch() is None

    monkeypatch.setattr(ingest, "EMBEDDING_MODEL_NAME", "model-b")
    assert ingest.check_embedding_model_mismatch() == ("model-a", "model-b")


def test_check_embedding_model_mismatch_none_when_manifest_missing(fake_env):
    assert ingest.check_embedding_model_mismatch() is None


# --- sync_data_dir()経由での記録・警告 ---


def test_first_sync_records_current_embedding_model_without_warning(fake_env, caplog):
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "テスト用ファイルの内容です。" * 5)

    with caplog.at_level("WARNING"):
        ingest.sync_data_dir(verbose=False)

    manifest = _on_disk_manifest()
    assert manifest[ingest.MANIFEST_EMBEDDING_MODEL_KEY] == "model-a"
    assert not any("埋め込みモデル" in record.getMessage() for record in caplog.records)


def test_sync_warns_when_embedding_model_changed(fake_env, monkeypatch, caplog):
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "テスト用ファイルの内容です。" * 5)
    ingest.sync_data_dir(verbose=False)

    monkeypatch.setattr(ingest, "EMBEDDING_MODEL_NAME", "model-b")
    with caplog.at_level("WARNING"):
        ingest.sync_data_dir(verbose=False)

    assert any("埋め込みモデルが変更されています" in record.getMessage() for record in caplog.records)
    # 自動での再構築は行わないため、旧ファイルのチャンクはそのままDBに残り続ける
    manifest = _on_disk_manifest()
    assert manifest["a.txt"]["chunk_ids"]


def test_mismatch_is_not_overwritten_and_keeps_warning_on_next_sync(fake_env, monkeypatch, caplog):
    # rm -rf chroma_db による手動再構築までは警告が消えないことを確認する
    # （不一致検知時にmanifestの記録値を現在のモデル名へ上書きしない実装の検証）。
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "テスト用ファイルの内容です。" * 5)
    ingest.sync_data_dir(verbose=False)

    monkeypatch.setattr(ingest, "EMBEDDING_MODEL_NAME", "model-b")
    ingest.sync_data_dir(verbose=False)
    ingest.sync_data_dir(verbose=False)

    manifest = _on_disk_manifest()
    assert manifest[ingest.MANIFEST_EMBEDDING_MODEL_KEY] == "model-a"
    assert ingest.check_embedding_model_mismatch() == ("model-a", "model-b")


def test_manual_rebuild_clears_the_mismatch(fake_env, monkeypatch):
    # rm -rf chroma_db 相当（manifestごと削除）で再構築すると、次回同期時に
    # 現在のモデル名で新規に記録され直し、不一致が解消される。
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "テスト用ファイルの内容です。" * 5)
    ingest.sync_data_dir(verbose=False)

    monkeypatch.setattr(ingest, "EMBEDDING_MODEL_NAME", "model-b")
    ingest.sync_data_dir(verbose=False)
    assert ingest.check_embedding_model_mismatch() == ("model-a", "model-b")

    ingest.MANIFEST_PATH.unlink()
    ingest.sync_data_dir(verbose=False)

    assert ingest.check_embedding_model_mismatch() is None
    assert _on_disk_manifest()[ingest.MANIFEST_EMBEDDING_MODEL_KEY] == "model-b"


# --- add_single_conversation_file()経由での記録・警告 ---


def test_add_single_conversation_file_warns_when_embedding_model_changed(fake_env, monkeypatch, caplog):
    data_dir, _store = fake_env
    path = _write(data_dir, "conversations/thread-1/convo.md", "# 会話ログ\n\n質問と回答です。" * 5)
    ingest.add_single_conversation_file(path)

    monkeypatch.setattr(ingest, "EMBEDDING_MODEL_NAME", "model-b")
    with caplog.at_level("WARNING"):
        ingest.add_single_conversation_file(path)

    assert any("埋め込みモデルが変更されています" in record.getMessage() for record in caplog.records)


# --- manifestのトップレベルメタデータキーが、ファイル一覧系のAPIに漏れないこと ---


def test_embedding_model_key_excluded_from_list_indexed_files(fake_env):
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "テスト用ファイルの内容です。" * 5)
    ingest.sync_data_dir(verbose=False)

    names = [entry["name"] for entry in ingest.list_indexed_files()]
    assert ingest.MANIFEST_EMBEDDING_MODEL_KEY not in names
    assert names == ["a.txt"]


def test_manifest_file_entries_excludes_embedding_model_key():
    manifest = {
        ingest.MANIFEST_EMBEDDING_MODEL_KEY: "model-a",
        "a.txt": {"mtime": 1.0, "size": 1, "chunk_ids": ["chunk-1"]},
    }

    assert ingest._manifest_file_entries(manifest) == {"a.txt": {"mtime": 1.0, "size": 1, "chunk_ids": ["chunk-1"]}}
