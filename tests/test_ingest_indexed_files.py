"""ingest.py の list_indexed_files() / delete_indexed_file() のテスト。

サイドバーのインデックス済みファイル一覧・削除機能のバックエンドロジックを、
実際のベクトルDB・埋め込みモデルを使わずに検証する。
- list_indexed_files() は manifest.json のみを読むため、_save_manifest() で
  manifest を直接組み立てて検証する。
- delete_indexed_file() は safe_relative_dest() 経由でパスを解決し実ファイルを削除するだけで、
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


def test_list_indexed_files_excludes_ghost_entry_without_chunk_ids(fake_data_env):
    # delete()失敗時にpending_delete_chunk_idsだけを持ち越した「ゴーストエントリ」
    # （chunk_idsキー自体が無いエントリ）は、data/上は既に削除済みのファイルのため一覧から除外される。
    ingest._save_manifest(
        {
            "doc.txt": {"mtime": 1.0, "size": 10, "chunk_ids": ["c1"]},
            "ghost.txt": {"pending_delete_chunk_ids": ["c2"]},
        }
    )

    assert ingest.list_indexed_files() == [{"name": "doc.txt", "chunk_count": 1}]


def test_list_indexed_files_includes_entry_with_empty_chunk_ids(fake_data_env):
    # chunk_idsキー自体は存在するが空リストの場合（0チャンクのファイル等）は
    # ゴーストエントリではないため一覧に含める
    ingest._save_manifest({"empty.txt": {"mtime": 1.0, "size": 0, "chunk_ids": []}})

    assert ingest.list_indexed_files() == [{"name": "empty.txt", "chunk_count": 0}]


def test_list_indexed_files_excludes_multiple_ghost_entries_mixed_with_normal_files(fake_data_env):
    # ゴーストエントリが複数・通常エントリが複数混在していても、通常エントリだけが
    # ファイル名昇順で正しく残ることを確認する（1件ずつの組み合わせだけでなく複数件でも
    # フィルタ条件が壊れないことの回帰防止）。
    ingest._save_manifest(
        {
            "c.txt": {"mtime": 1.0, "size": 10, "chunk_ids": ["c1"]},
            "ghost_a.txt": {"pending_delete_chunk_ids": ["g1"]},
            "a.txt": {"mtime": 2.0, "size": 20, "chunk_ids": ["c2", "c3"]},
            "ghost_b.txt": {"pending_delete_chunk_ids": ["g2", "g3"]},
        }
    )

    result = ingest.list_indexed_files()

    assert result == [
        {"name": "a.txt", "chunk_count": 2},
        {"name": "c.txt", "chunk_count": 1},
    ]


def test_list_indexed_files_excludes_ghost_entry_with_empty_pending_delete_chunk_ids(fake_data_env):
    # 境界値: pending_delete_chunk_ids自体が空リスト（あるいはキーが無い）の
    # ゴーストエントリでも、"chunk_ids"キーが無い時点で例外にならず除外される。
    ingest._save_manifest(
        {
            "ghost_empty_pending.txt": {"pending_delete_chunk_ids": []},
            "ghost_no_pending_key.txt": {},
        }
    )

    assert ingest.list_indexed_files() == []


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
    # パストラバーサルを試みる相対パスはsafe_relative_dest()がDATA_DIR外と判定しNoneを返すため、
    # DATA_DIR外のファイルには一切触れない。
    data_dir = fake_data_env

    result = ingest.delete_indexed_file(traversal_name)

    assert result is False
    # DATA_DIR配下に余計なファイルが作られたり消えたりしていない
    assert list(data_dir.iterdir()) == []


def test_delete_indexed_file_traversal_name_does_not_delete_decoy_file_inside_data_dir(fake_data_env):
    # パストラバーサルを試みる相対パスは、DATA_DIR配下にたまたま同名（ベースネーム一致）の
    # ファイルが存在していても、そのファイルを誤って削除しない（DATA_DIR外を指す時点でNoneを
    # 返し、DATA_DIR配下のどのファイルにも一切触れないことの確認）。
    data_dir = fake_data_env
    decoy = data_dir / "passwd"
    decoy.write_text("DATA_DIR配下にある無関係なファイルです。", encoding="utf-8")

    result = ingest.delete_indexed_file("../../etc/passwd")

    assert result is False
    assert decoy.exists()


def test_delete_indexed_file_returns_false_for_bare_dotdot(fake_data_env):
    # ".." はDATA_DIRの親ディレクトリを指すため、safe_relative_dest()がNoneを返しFalseになる
    result = ingest.delete_indexed_file("..")

    assert result is False


def test_delete_indexed_file_returns_false_for_empty_string(fake_data_env):
    # 空文字列はsafe_relative_dest()がDATA_DIR自身を指すパスを返してしまうため、
    # is_file()チェックが無いとIsADirectoryErrorが送出されてしまう。
    result = ingest.delete_indexed_file("")

    assert result is False


def test_delete_indexed_file_returns_false_for_current_dir(fake_data_env):
    result = ingest.delete_indexed_file(".")

    assert result is False


def test_delete_indexed_file_returns_false_for_existing_directory(fake_data_env):
    data_dir = fake_data_env
    subdir = data_dir / "manuals"
    subdir.mkdir()

    result = ingest.delete_indexed_file("manuals")

    assert result is False
    assert subdir.exists()


def test_delete_indexed_file_deletes_file_in_subfolder(fake_data_env):
    # サブフォルダ配下に手動配置されたファイル（list_indexed_files()がサブフォルダ込みの
    # 相対パスで返すもの）を指定した場合、そのサブフォルダ内の対象ファイルが正しく削除される。
    data_dir = fake_data_env
    (data_dir / "manuals").mkdir()
    target = data_dir / "manuals" / "spec.pdf"
    target.write_text("サブフォルダ内の削除対象です。", encoding="utf-8")

    result = ingest.delete_indexed_file("manuals/spec.pdf")

    assert result is True
    assert not target.exists()


def test_delete_indexed_file_in_subfolder_does_not_delete_same_named_file_at_root(fake_data_env):
    # サブフォルダ内のファイルを削除しても、DATA_DIR直下にある無関係な同名ファイルは
    # 誤って削除されない（サブフォルダ部分を切り捨てて解決していた旧実装のバグの回帰確認）。
    data_dir = fake_data_env
    (data_dir / "manuals").mkdir()
    target = data_dir / "manuals" / "spec.pdf"
    target.write_text("サブフォルダ内の削除対象です。", encoding="utf-8")
    decoy = data_dir / "spec.pdf"
    decoy.write_text("DATA_DIR直下にある無関係な同名ファイルです。", encoding="utf-8")

    result = ingest.delete_indexed_file("manuals/spec.pdf")

    assert result is True
    assert not target.exists()
    assert decoy.exists()


def test_delete_indexed_file_in_subfolder_returns_false_when_target_missing_even_if_root_file_exists(fake_data_env):
    # サブフォルダ内の対象ファイルが存在しない場合、DATA_DIR直下に同名ファイルがあっても
    # それを誤って削除せずFalseを返す（旧実装は誤ってdata_dir直下のファイルを削除・成功扱いしていた）。
    data_dir = fake_data_env
    (data_dir / "manuals").mkdir()
    decoy = data_dir / "spec.pdf"
    decoy.write_text("DATA_DIR直下にある無関係な同名ファイルです。", encoding="utf-8")

    result = ingest.delete_indexed_file("manuals/spec.pdf")

    assert result is False
    assert decoy.exists()


def test_delete_indexed_file_rejects_traversal_within_subfolder_path(fake_data_env):
    # サブフォルダを経由してDATA_DIR外へ抜けようとするパス（例: "manuals/../../etc/passwd"）も
    # safe_relative_dest()のDATA_DIR配下チェックにより拒否される。
    result = ingest.delete_indexed_file("manuals/../../etc/passwd")

    assert result is False
