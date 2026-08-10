"""ingest.py の manifest.json 読み書き（_load_manifest / _save_manifest）のテスト。

manifest.json への書き込みを一時ファイル経由のアトミックな置き換え
（os.replace）に変更し、壊れたJSONを読み込んだ場合も例外を送出せず空辞書に
フォールバックするようにした変更を検証する。
"""

import json

import pytest

import ingest


@pytest.fixture
def fake_manifest_env(tmp_path, monkeypatch):
    persist_dir = tmp_path / "chroma_db"
    manifest_path = persist_dir / "manifest.json"

    monkeypatch.setattr(ingest, "PERSIST_DIR", persist_dir)
    monkeypatch.setattr(ingest, "MANIFEST_PATH", manifest_path)

    return persist_dir, manifest_path


def test_save_then_load_roundtrip(fake_manifest_env):
    _persist_dir, manifest_path = fake_manifest_env
    manifest = {
        "a.txt": {"mtime": 123.0, "size": 10, "chunk_ids": ["chunk-1", "chunk-2"]},
        "conversations/thread-1/log.md": {
            "mtime": 456.0,
            "size": 20,
            "chunk_ids": ["chunk-3"],
        },
    }

    ingest._save_manifest(manifest)
    loaded = ingest._load_manifest()

    assert loaded == manifest
    # ファイルにも同じ内容が書き込まれている
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_save_manifest_does_not_leave_tmp_file_behind(fake_manifest_env):
    _persist_dir, manifest_path = fake_manifest_env

    ingest._save_manifest({"a.txt": {"mtime": 1.0, "size": 1, "chunk_ids": []}})

    tmp_path = manifest_path.with_suffix(".json.tmp")
    assert manifest_path.exists()
    assert not tmp_path.exists()


def test_save_manifest_creates_persist_dir_if_missing(fake_manifest_env):
    persist_dir, manifest_path = fake_manifest_env
    assert not persist_dir.exists()

    ingest._save_manifest({})

    assert persist_dir.exists()
    assert manifest_path.exists()


def test_load_manifest_returns_empty_dict_when_file_missing(fake_manifest_env):
    _persist_dir, _manifest_path = fake_manifest_env

    assert ingest._load_manifest() == {}


def test_load_manifest_falls_back_to_empty_dict_on_corrupt_json(fake_manifest_env, caplog):
    persist_dir, manifest_path = fake_manifest_env
    persist_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{ this is not valid json ", encoding="utf-8")

    with caplog.at_level("WARNING"):
        result = ingest._load_manifest()

    assert result == {}
    assert any(str(manifest_path) in record.getMessage() for record in caplog.records)


def test_load_manifest_after_save_manifest_overwrite_of_corrupt_file(
    fake_manifest_env,
):
    # 壊れたmanifest.jsonが残っている状態でも、_save_manifest()実行後は
    # 正常なJSONとして読み直せることを確認する（アトミックな置き換えの効果）。
    persist_dir, manifest_path = fake_manifest_env
    persist_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{ broken", encoding="utf-8")

    ingest._save_manifest({"a.txt": {"mtime": 1.0, "size": 1, "chunk_ids": []}})

    assert ingest._load_manifest() == {"a.txt": {"mtime": 1.0, "size": 1, "chunk_ids": []}}
