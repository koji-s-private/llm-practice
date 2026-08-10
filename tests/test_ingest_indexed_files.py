"""ingest.py の list_indexed_files() / delete_indexed_file() のテスト。

サイドバーのインデックス済みファイル一覧・削除機能のバックエンドロジックを、
実際のベクトルDB・埋め込みモデルを使わずに検証する。
- list_indexed_files() は manifest.json のみを読むため、_save_manifest() で
  manifest を直接組み立てて検証する。
- delete_indexed_file() は safe_upload_dest() 経由でパスを解決し実ファイルを削除するだけで、
  manifest・ベクトルDBには一切触れない設計のため、DATA_DIR上のファイル操作のみを確認する。
"""

import pytest

import ingest


@pytest.fixture
def fake_data_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma_db"

    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "PERSIST_DIR", persist_dir)
    monkeypatch.setattr(ingest, "MANIFEST_PATH", persist_dir / "manifest.json")

    return data_dir


# --- list_indexed_files() ---


def test_list_indexed_files_returns_name_and_chunk_count(fake_data_env):
    ingest._save_manifest(
        {
            "b.txt": {"mtime": 1.0, "size": 10, "chunk_ids": ["c1", "c2"]},
            "a.pdf": {"mtime": 2.0, "size": 20, "chunk_ids": ["c3"]},
        }
    )

    result = ingest.list_indexed_files()

    # ファイル名昇順で返る
    assert result == [
        {"name": "a.pdf", "chunk_count": 1},
        {"name": "b.txt", "chunk_count": 2},
    ]


def test_list_indexed_files_returns_empty_list_when_manifest_empty(fake_data_env):
    assert ingest.list_indexed_files() == []


def test_list_indexed_files_returns_empty_list_when_manifest_file_missing(fake_data_env):
    # manifest.json自体が存在しない場合も例外にならず空リストになる
    assert not ingest.MANIFEST_PATH.exists()
    assert ingest.list_indexed_files() == []


def test_list_indexed_files_excludes_conversation_logs(fake_data_env):
    ingest._save_manifest(
        {
            "doc.txt": {"mtime": 1.0, "size": 10, "chunk_ids": ["c1"]},
            "conversations/thread-1/log.md": {"mtime": 2.0, "size": 20, "chunk_ids": ["c2", "c3"]},
        }
    )

    result = ingest.list_indexed_files()

    assert result == [{"name": "doc.txt", "chunk_count": 1}]


def test_list_indexed_files_excludes_legacy_flat_conversation_log(fake_data_env):
    # サブフォルダの無い旧形式の会話ログ（conversations/xxx.md）も同様に除外される
    ingest._save_manifest(
        {
            "conversations/legacy.md": {"mtime": 1.0, "size": 10, "chunk_ids": ["c1"]},
        }
    )

    assert ingest.list_indexed_files() == []


def test_list_indexed_files_chunk_count_is_zero_when_chunk_ids_missing(fake_data_env):
    # 境界値: chunk_idsキー自体が無いエントリでも例外にならずchunk_count=0を返す
    ingest._save_manifest({"empty.txt": {"mtime": 1.0, "size": 0}})

    assert ingest.list_indexed_files() == [{"name": "empty.txt", "chunk_count": 0}]


# --- delete_indexed_file() ---


def test_delete_indexed_file_removes_existing_file_and_returns_true(fake_data_env):
    data_dir = fake_data_env
    target = data_dir / "report.txt"
    target.write_text("削除対象のファイルです。", encoding="utf-8")

    result = ingest.delete_indexed_file("report.txt")

    assert result is True
    assert not target.exists()


def test_delete_indexed_file_returns_false_for_nonexistent_file(fake_data_env):
    result = ingest.delete_indexed_file("does_not_exist.txt")

    assert result is False


def test_delete_indexed_file_does_not_touch_manifest_or_other_files(fake_data_env):
    # delete_indexed_file()自体はmanifest・DB反映を行わない設計（呼び出し元のsync_data_dir()に委ねる）
    data_dir = fake_data_env
    (data_dir / "target.txt").write_text("削除対象です。", encoding="utf-8")
    (data_dir / "other.txt").write_text("残すべきファイルです。", encoding="utf-8")
    ingest._save_manifest({"target.txt": {"mtime": 1.0, "size": 1, "chunk_ids": ["c1"]}})

    ingest.delete_indexed_file("target.txt")

    assert (data_dir / "other.txt").exists()
    # manifestはdelete_indexed_file()単体では更新されない
    manifest = ingest._load_manifest()
    assert manifest == {"target.txt": {"mtime": 1.0, "size": 1, "chunk_ids": ["c1"]}}


@pytest.mark.parametrize(
    "traversal_name",
    [
        "../../etc/passwd",
        "../../../tmp/evil.txt",
        "../sibling_dir_escape.txt",
    ],
)
def test_delete_indexed_file_rejects_directory_traversal_and_only_targets_data_dir(fake_data_env, traversal_name):
    # パストラバーサルを試みる名前でも safe_upload_dest() 経由でベースネームのみに
    # 無害化されるため、DATA_DIR外のファイルには一切触れない。
    # 同名の（無害化後の）ファイルがDATA_DIR上に存在しなければFalseで何も削除されない。
    data_dir = fake_data_env

    result = ingest.delete_indexed_file(traversal_name)

    assert result is False
    # DATA_DIR配下に余計なファイルが作られたり消えたりしていない
    assert list(data_dir.iterdir()) == []


def test_delete_indexed_file_with_traversal_name_only_deletes_basename_match_inside_data_dir(fake_data_env):
    # トラバーサル名でも無害化されたベースネーム（例: "passwd"）がDATA_DIR上に
    # たまたま存在する場合は、そのDATA_DIR配下のファイルだけが削除される
    # （DATA_DIR外への到達は一切発生しないことの確認）。
    data_dir = fake_data_env
    decoy = data_dir / "passwd"
    decoy.write_text("DATA_DIR配下にある安全なファイルです。", encoding="utf-8")

    result = ingest.delete_indexed_file("../../etc/passwd")

    assert result is True
    assert not decoy.exists()


def test_delete_indexed_file_returns_false_for_bare_dotdot(fake_data_env):
    # ".." そのものはsafe_upload_dest()がNoneを返すため、path.exists()判定にすら進まずFalse
    result = ingest.delete_indexed_file("..")

    assert result is False


def test_delete_indexed_file_ignores_directory_component_and_targets_data_dir_root(fake_data_env):
    # 境界値: ディレクトリを含む名前（例: サブフォルダ名を装ったもの）を渡しても
    # ディレクトリ部分は無視され、DATA_DIR直下の同名ファイルのみが対象になる。
    # これによりconversations/配下のファイル名を渡しても、直下に同名ファイルが
    # 無ければ何も削除されない。
    data_dir = fake_data_env
    (data_dir / "conversations").mkdir()
    (data_dir / "conversations" / "log.md").write_text("会話ログです。", encoding="utf-8")

    result = ingest.delete_indexed_file("conversations/log.md")

    assert result is False
    assert (data_dir / "conversations" / "log.md").exists()
