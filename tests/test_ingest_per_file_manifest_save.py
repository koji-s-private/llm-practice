"""ingest.py の「manifestをファイル1件ごとに都度保存する」変更のテスト。

_sync_data_dir_locked() は、従来は全ファイルの追加・更新・削除処理が終わった後に
まとめて1回だけ _save_manifest() を呼んでいたが、途中でプロセスが中断
（クラッシュ・強制終了など）すると、ベクトルDBには反映済みなのにmanifestには
記録されないファイルが生まれ、次回同期時に同じ内容のチャンクが重複登録される
おそれがあった。この変更ではファイル1件（追加・更新・削除）ごとにmanifestを
都度保存するようにしたため、その挙動を検証する。

実際の埋め込みモデル・Chromaは使わず、get_vectorstore() を軽量なフェイクの
ベクトルストアに monkeypatch する（tests/test_ingest.py と同じ方針）。
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


# --- 正常系: 複数ファイル同期時、manifestに全ファイルの情報が記録される ---


def test_multiple_new_files_are_all_recorded_in_manifest(fake_env):
    data_dir, store = fake_env
    _write(data_dir, "a.txt", "1件目のファイルの内容です。" * 5)
    _write(data_dir, "b.txt", "2件目のファイルの内容です。" * 5)
    _write(data_dir, "c.txt", "3件目のファイルの内容です。" * 5)

    result = ingest.sync_data_dir(verbose=False)

    assert sorted(result["added"]) == ["a.txt", "b.txt", "c.txt"]
    manifest = _on_disk_manifest()
    assert set(manifest.keys()) == {"a.txt", "b.txt", "c.txt"}
    for name, entry in manifest.items():
        assert entry["chunk_ids"]
        assert set(entry["chunk_ids"]).issubset(store.docs_by_id.keys())


def test_manifest_is_saved_incrementally_once_per_file_not_only_at_the_end(fake_env, monkeypatch):
    # _save_manifest() が全件処理後に1回だけでなく、ファイルごとに都度呼ばれることを検証する。
    # 各呼び出し時点でのmanifestサイズを記録し、単調増加していく（＝都度保存されている）ことを確認する。
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "1件目のファイルの内容です。" * 5)
    _write(data_dir, "b.txt", "2件目のファイルの内容です。" * 5)
    _write(data_dir, "c.txt", "3件目のファイルの内容です。" * 5)

    real_save_manifest = ingest._save_manifest
    snapshot_sizes = []

    def spy_save_manifest(manifest):
        real_save_manifest(manifest)
        snapshot_sizes.append(len(manifest))

    monkeypatch.setattr(ingest, "_save_manifest", spy_save_manifest)

    ingest.sync_data_dir(verbose=False)

    # 3ファイル分の追加処理 + 末尾の冪等保存の、合計4回は最低でも呼ばれる
    assert len(snapshot_sizes) >= 4
    # ファイルごとに1件ずつ増えていき、全件処理後に3で頭打ちになる（全件後まとめて保存ではない証拠）
    assert snapshot_sizes[0] == 1
    assert snapshot_sizes[-1] == 3
    assert snapshot_sizes == sorted(snapshot_sizes)


# --- 今回の修正の核心: 複数ファイル処理中の中断でも、処理済み分はmanifestに残る ---


def test_manifest_keeps_already_processed_file_when_sync_is_interrupted_mid_loop(fake_env, monkeypatch):
    # 2ファイル目の add_documents() 完了直後（新チャンク登録後）に、
    # プロセス強制終了を模した SystemExit（Exceptionのサブクラスではないため
    # 既存の `except Exception` では捕捉されない）を発生させ、
    # 「途中で中断されても、その時点までに処理済みのファイルの情報はmanifestに残る」
    # ことを確認する。ファイルの走査順（rglob）はOS依存で保証されないため、
    # 「何番目に処理されたか」ではなく「1件目は成功、2件目で中断」という
    # 呼び出し回数ベースで判定し、順序に依存しないテストにする。
    data_dir, store = fake_env
    _write(data_dir, "a.txt", "1件目のファイルの内容です。" * 5)
    _write(data_dir, "b.txt", "2件目のファイルの内容です。" * 5)

    real_add_documents = store.add_documents
    call_count = {"n": 0}

    def flaky_add_documents(documents):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return real_add_documents(documents)
        raise SystemExit("プロセスが強制終了されたことを想定")

    monkeypatch.setattr(store, "add_documents", flaky_add_documents)

    with pytest.raises(SystemExit):
        ingest.sync_data_dir(verbose=False)

    # 中断されても、1件目として処理が完了したファイルの情報はmanifestに保存されている
    manifest = _on_disk_manifest()
    assert len(manifest) == 1
    (saved_name, saved_entry) = next(iter(manifest.items()))
    assert saved_name in {"a.txt", "b.txt"}
    assert saved_entry["chunk_ids"]
    # ベクトルDB側にも、manifestに記録されたファイル分のチャンクだけが存在する
    assert set(store.docs_by_id.keys()) == set(saved_entry["chunk_ids"])

    # 中断された同期をやり直すと、残りのファイルだけが追加される（重複登録されない）
    monkeypatch.setattr(store, "add_documents", real_add_documents)
    result = ingest.sync_data_dir(verbose=False)
    assert len(result["added"]) == 1
    manifest_after_retry = _on_disk_manifest()
    assert set(manifest_after_retry.keys()) == {"a.txt", "b.txt"}


def test_manifest_keeps_already_processed_file_when_interrupted_during_update(fake_env, monkeypatch):
    # 更新処理中（旧チャンク削除 vector_store.delete() の直前後）の中断でも、
    # 直前に処理が完了した別ファイルの情報はmanifestに残ることを確認する。
    data_dir, store = fake_env
    path_a = _write(data_dir, "a.txt", "最初のaファイルの内容です。" * 5)
    path_b = _write(data_dir, "b.txt", "最初のbファイルの内容です。" * 5)
    ingest.sync_data_dir(verbose=False)

    # 両方のファイルを更新し、2回目の同期でどちらも更新対象にする
    path_a.write_text("更新後のaファイルの内容です。" * 5, encoding="utf-8")
    path_b.write_text("更新後のbファイルの内容です。" * 5, encoding="utf-8")

    real_add_documents = store.add_documents
    call_count = {"n": 0}

    def flaky_add_documents(documents):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return real_add_documents(documents)
        raise SystemExit("プロセスが強制終了されたことを想定")

    monkeypatch.setattr(store, "add_documents", flaky_add_documents)

    with pytest.raises(SystemExit):
        ingest.sync_data_dir(verbose=False)

    manifest = _on_disk_manifest()
    # 中断されても、更新が完了した1件分はmanifestに反映されている（件数自体は失われない）
    assert len(manifest) == 2
    # 更新が完了したファイルはmanifestのmtimeが更新後の実際のファイルmtimeと一致し、
    # 中断されたファイルは更新前（初回同期時）のmtimeのままになっているはず
    current_mtimes = {"a.txt": path_a.stat().st_mtime, "b.txt": path_b.stat().st_mtime}
    updated_names = [name for name, entry in manifest.items() if entry["mtime"] == current_mtimes[name]]
    stale_names = [name for name, entry in manifest.items() if entry["mtime"] != current_mtimes[name]]
    assert len(updated_names) == 1
    assert len(stale_names) == 1


# --- 削除処理の中断: 削除も1件ごとに保存されるため、途中で中断されても反映済み分は残る ---


def test_manifest_keeps_already_deleted_file_removed_when_deletion_interrupted(fake_env, monkeypatch):
    data_dir, store = fake_env
    path_a = _write(data_dir, "a.txt", "削除対象のaファイルです。" * 5)
    path_b = _write(data_dir, "b.txt", "削除対象のbファイルです。" * 5)
    ingest.sync_data_dir(verbose=False)
    manifest_before = _on_disk_manifest()
    assert set(manifest_before.keys()) == {"a.txt", "b.txt"}

    path_a.unlink()
    path_b.unlink()

    real_delete = store.delete
    call_count = {"n": 0}

    def flaky_delete(ids):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return real_delete(ids)
        raise SystemExit("プロセスが強制終了されたことを想定")

    monkeypatch.setattr(store, "delete", flaky_delete)

    with pytest.raises(SystemExit):
        ingest.sync_data_dir(verbose=False)

    manifest_after = _on_disk_manifest()
    # 1件は削除処理が完了しmanifestからも消えているが、もう1件は中断されまだ残っている
    assert len(manifest_after) == 1
    remaining_name = next(iter(manifest_after.keys()))
    assert remaining_name in {"a.txt", "b.txt"}


# --- 境界値・異常系 ---


def test_sync_empty_data_dir_returns_all_empty_results(fake_env):
    # data/ が空の場合、何も追加・更新・削除・失敗せず正常終了する
    _data_dir, store = fake_env

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": [], "failed": []}
    assert store.docs_by_id == {}


def test_sync_empty_data_dir_twice_is_idempotent(fake_env):
    # 空のdata/への同期を2回連続で行っても問題なく完了する（境界値）
    _data_dir, _store = fake_env

    result_1 = ingest.sync_data_dir(verbose=False)
    result_2 = ingest.sync_data_dir(verbose=False)

    assert result_1 == result_2 == {"added": [], "updated": [], "removed": [], "failed": []}


def test_sync_with_only_deletions_and_no_additions_or_updates(fake_env):
    # 追加・更新は無く削除のみが発生するケース（境界値）。
    # 削除以外の結果キーは空のまま、削除のみが記録されることを確認する。
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "そのうち削除されるファイルです。" * 5)
    ingest.sync_data_dir(verbose=False)

    path.unlink()
    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": ["a.txt"], "failed": []}
    assert store.docs_by_id == {}
    assert _on_disk_manifest() == {}


def test_sync_multiple_deletions_all_recorded_removed_and_manifest_emptied(fake_env):
    # 削除のみのケースでも複数ファイルを同時に扱える（境界値: 複数削除）
    data_dir, store = fake_env
    path_a = _write(data_dir, "a.txt", "aファイルの内容です。" * 5)
    path_b = _write(data_dir, "b.txt", "bファイルの内容です。" * 5)
    path_c = _write(data_dir, "c.txt", "cファイルの内容です。" * 5)
    ingest.sync_data_dir(verbose=False)

    path_a.unlink()
    path_b.unlink()
    path_c.unlink()
    result = ingest.sync_data_dir(verbose=False)

    assert sorted(result["removed"]) == ["a.txt", "b.txt", "c.txt"]
    assert result["added"] == []
    assert result["updated"] == []
    assert store.docs_by_id == {}
    assert _on_disk_manifest() == {}


def test_manifest_file_created_even_when_all_files_fail(fake_env, monkeypatch):
    # 全ファイルが失敗した場合でも、末尾の冪等保存によりmanifest.jsonは
    # （空の内容で）作成される（既存のフォールバック仕様の確認・境界値）
    data_dir, _store = fake_env

    class _FailingLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            raise ValueError("読み込み失敗を想定")

    _write(data_dir, "bad.txt", "壊れていることにするテキストです。" * 5)
    monkeypatch.setattr(ingest, "LOADERS", {**ingest.LOADERS, ".txt": lambda p: _FailingLoader(p)})

    result = ingest.sync_data_dir(verbose=False)

    assert result["failed"] == ["bad.txt"]
    assert ingest.MANIFEST_PATH.exists()
    assert _on_disk_manifest() == {}
