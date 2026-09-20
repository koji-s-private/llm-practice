"""google_drive_sync.py のテスト。

実際のGoogle Drive API・OAuth認証は一切行わず、_get_drive_service() をフェイクの
Driveサービスクライアントに monkeypatch して、ミラー処理（export/ダウンロード判定・
スキップ・削除検知）だけを検証する（tests/test_ingest.py の fake_env フィクスチャに倣う）。
"""

import logging

import pytest

import google_drive_sync


class _FakeRequest:
    """service.files().export_media()/get_media() が返す想定のフェイクリクエスト。

    実際のMediaIoBaseDownloadはHTTPチャンク転送を行うが、テストではモジュール側の
    MediaIoBaseDownloadごとフェイクに差し替え、execute()が中身をそのまま返す単純な形にする。
    """

    def __init__(self, content: bytes):
        self.content = content

    def execute(self):
        return self.content


class _FakeMediaIoBaseDownload:
    """googleapiclient.http.MediaIoBaseDownload の最小限のフェイク実装。"""

    def __init__(self, fh, request):
        self.fh = fh
        self.request = request

    def next_chunk(self):
        self.fh.write(self.request.execute())
        return None, True


class _FakeFailingRequest:
    """execute()時に例外を送出するフェイクリクエスト（ダウンロード失敗のシミュレーション用）。"""

    def __init__(self, error: Exception):
        self.error = error

    def execute(self):
        raise self.error


class _FakeFilesResource:
    def __init__(self, drive_files, export_contents, media_contents):
        self.drive_files = drive_files
        self.export_contents = export_contents
        self.media_contents = media_contents

    def list(self, **kwargs):
        return _FakeRequest({"files": self.drive_files, "nextPageToken": None})

    def export_media(self, fileId, mimeType):
        return _FakeRequest(self.export_contents[fileId])

    def get_media(self, fileId):
        # media_contents の値が例外インスタンスの場合はダウンロード失敗を模擬する
        content = self.media_contents[fileId]
        if isinstance(content, Exception):
            return _FakeFailingRequest(content)
        return _FakeRequest(content)


class _FakeDriveService:
    def __init__(self, drive_files, export_contents=None, media_contents=None):
        self._resource = _FakeFilesResource(drive_files, export_contents or {}, media_contents or {})

    def files(self):
        return self._resource


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    drive_dir = tmp_path / "google_drive"
    monkeypatch.setattr(google_drive_sync, "GOOGLE_DRIVE_DIR", drive_dir)
    monkeypatch.setattr(google_drive_sync, "MediaIoBaseDownload", _FakeMediaIoBaseDownload)
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "fake-folder-id")
    return drive_dir


def _use_fake_service(monkeypatch, drive_files, export_contents=None, media_contents=None):
    service = _FakeDriveService(drive_files, export_contents, media_contents)
    monkeypatch.setattr(google_drive_sync, "_get_drive_service", lambda: service)
    return service


class _PaginatedFakeFilesResource:
    """files.list() が複数ページに分けてレスポンスを返す想定のフェイク（ページネーション検証用）。"""

    def __init__(self, pages: list[list[dict]], media_contents: dict):
        self.pages = pages
        self.media_contents = media_contents

    def list(self, pageToken=None, **kwargs):
        index = 0 if pageToken is None else int(pageToken)
        next_token = str(index + 1) if index + 1 < len(self.pages) else None
        return _FakeRequest({"files": self.pages[index], "nextPageToken": next_token})

    def get_media(self, fileId):
        return _FakeRequest(self.media_contents[fileId])


class _PaginatedFakeDriveService:
    def __init__(self, pages: list[list[dict]], media_contents: dict):
        self._resource = _PaginatedFakeFilesResource(pages, media_contents)

    def files(self):
        return self._resource


def test_google_docs_sheets_slides_are_exported_with_correct_extension(fake_env, monkeypatch):
    drive_dir = fake_env
    drive_files = [
        {"id": "doc1", "name": "議事録", "mimeType": "application/vnd.google-apps.document"},
        {"id": "sheet1", "name": "予算表", "mimeType": "application/vnd.google-apps.spreadsheet"},
        {"id": "slide1", "name": "説明資料", "mimeType": "application/vnd.google-apps.presentation"},
    ]
    export_contents = {"doc1": b"docx-bytes", "sheet1": b"xlsx-bytes", "slide1": b"pptx-bytes"}
    _use_fake_service(monkeypatch, drive_files, export_contents=export_contents)

    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert sorted(result["added"]) == sorted(["議事録.docx", "予算表.xlsx", "説明資料.pptx"])
    assert (drive_dir / "議事録.docx").read_bytes() == b"docx-bytes"
    assert (drive_dir / "予算表.xlsx").read_bytes() == b"xlsx-bytes"
    assert (drive_dir / "説明資料.pptx").read_bytes() == b"pptx-bytes"


def test_regular_file_is_downloaded_via_get_media(fake_env, monkeypatch):
    drive_dir = fake_env
    drive_files = [{"id": "pdf1", "name": "manual.pdf", "mimeType": "application/pdf"}]
    _use_fake_service(monkeypatch, drive_files, media_contents={"pdf1": b"%PDF-bytes"})

    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result["added"] == ["manual.pdf"]
    assert (drive_dir / "manual.pdf").read_bytes() == b"%PDF-bytes"


def test_unsupported_extension_is_skipped(fake_env, monkeypatch):
    drive_dir = fake_env
    drive_files = [{"id": "mp4-1", "name": "video.mp4", "mimeType": "video/mp4"}]
    _use_fake_service(monkeypatch, drive_files)

    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result == {
        "added": [],
        "updated": [],
        "removed": [],
        "skipped": ["video.mp4"],
        "removal_blocked_files": [],
    }
    assert list(drive_dir.iterdir()) == []


def test_file_re_downloaded_on_second_sync_is_marked_updated(fake_env, monkeypatch):
    drive_files = [{"id": "pdf1", "name": "manual.pdf", "mimeType": "application/pdf"}]
    _use_fake_service(monkeypatch, drive_files, media_contents={"pdf1": b"v1"})
    google_drive_sync.sync_google_drive_files(verbose=False)

    _use_fake_service(monkeypatch, drive_files, media_contents={"pdf1": b"v2"})
    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result == {
        "added": [],
        "updated": ["manual.pdf"],
        "removed": [],
        "skipped": [],
        "removal_blocked_files": [],
    }


def test_file_removed_from_drive_is_deleted_locally(fake_env, monkeypatch):
    drive_dir = fake_env
    drive_files = [
        {"id": "pdf1", "name": "keep.pdf", "mimeType": "application/pdf"},
        {"id": "pdf2", "name": "gone.pdf", "mimeType": "application/pdf"},
    ]
    _use_fake_service(monkeypatch, drive_files, media_contents={"pdf1": b"keep", "pdf2": b"gone"})
    google_drive_sync.sync_google_drive_files(verbose=False)
    assert (drive_dir / "gone.pdf").exists()

    _use_fake_service(monkeypatch, [drive_files[0]], media_contents={"pdf1": b"keep"})
    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result == {
        "added": [],
        "updated": ["keep.pdf"],
        "removed": ["gone.pdf"],
        "skipped": [],
        "removal_blocked_files": [],
    }
    assert not (drive_dir / "gone.pdf").exists()
    assert (drive_dir / "keep.pdf").exists()


def test_sync_is_skipped_when_folder_id_not_set(fake_env, monkeypatch):
    monkeypatch.delenv("GOOGLE_DRIVE_FOLDER_ID", raising=False)

    def _fail_if_called():
        raise AssertionError("GOOGLE_DRIVE_FOLDER_ID未設定時はDrive APIを呼び出すべきではない")

    monkeypatch.setattr(google_drive_sync, "_get_drive_service", lambda: _fail_if_called())

    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result == {
        "added": [],
        "updated": [],
        "removed": [],
        "skipped": [],
        "removal_blocked_files": [],
    }


def test_missing_client_secret_file_raises_clear_error(fake_env, monkeypatch, tmp_path):
    monkeypatch.setattr(google_drive_sync, "CLIENT_SECRET_FILE", tmp_path / "does-not-exist.json")

    with pytest.raises(RuntimeError, match="OAuthクライアントシークレットファイルが見つかりません"):
        google_drive_sync._get_drive_service()


def test_pagination_collects_all_pages(fake_env, monkeypatch):
    """フォルダ内のファイルが複数ページに分かれて返ってきても全件取得できることを確認する。"""
    drive_dir = fake_env
    pages = [
        [{"id": "pdf1", "name": "a.pdf", "mimeType": "application/pdf"}],
        [{"id": "pdf2", "name": "b.pdf", "mimeType": "application/pdf"}],
    ]
    media_contents = {"pdf1": b"a-bytes", "pdf2": b"b-bytes"}
    service = _PaginatedFakeDriveService(pages, media_contents)
    monkeypatch.setattr(google_drive_sync, "_get_drive_service", lambda: service)

    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert sorted(result["added"]) == ["a.pdf", "b.pdf"]
    assert (drive_dir / "a.pdf").read_bytes() == b"a-bytes"
    assert (drive_dir / "b.pdf").read_bytes() == b"b-bytes"


def test_download_failure_does_not_corrupt_existing_local_file(fake_env, monkeypatch):
    drive_dir = fake_env
    drive_files = [{"id": "pdf1", "name": "manual.pdf", "mimeType": "application/pdf"}]
    _use_fake_service(monkeypatch, drive_files, media_contents={"pdf1": b"original-content"})
    google_drive_sync.sync_google_drive_files(verbose=False)
    assert (drive_dir / "manual.pdf").read_bytes() == b"original-content"

    _use_fake_service(monkeypatch, drive_files, media_contents={"pdf1": RuntimeError("simulated network error")})
    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result == {
        "added": [],
        "updated": [],
        "removed": [],
        "skipped": ["manual.pdf"],
        "removal_blocked_files": [],
    }
    assert (drive_dir / "manual.pdf").read_bytes() == b"original-content"


def test_dest_path_for_rejects_relative_path_traversal(fake_env):
    drive_dir = fake_env
    drive_file = {"id": "x1", "name": "../evil.pdf", "mimeType": "application/pdf"}

    dest_path = google_drive_sync._dest_path_for(drive_file)

    assert dest_path is None or dest_path.resolve().parent == drive_dir.resolve()


def test_dest_path_for_rejects_absolute_path_override(fake_env, tmp_path):
    drive_dir = fake_env
    escape_target = tmp_path / "pwned.pdf"
    drive_file = {"id": "x2", "name": str(escape_target), "mimeType": "application/pdf"}

    dest_path = google_drive_sync._dest_path_for(drive_file)

    assert dest_path is None or dest_path.resolve().parent == drive_dir.resolve()


def test_multiple_distinct_files_use_plain_filenames(fake_env, monkeypatch):
    """同名衝突が無ければ、従来通り素のファイル名でダウンロードされる（回帰確認）。"""
    drive_dir = fake_env
    drive_files = [
        {"id": "pdf1", "name": "a.pdf", "mimeType": "application/pdf"},
        {"id": "pdf2", "name": "b.pdf", "mimeType": "application/pdf"},
    ]
    _use_fake_service(monkeypatch, drive_files, media_contents={"pdf1": b"a-bytes", "pdf2": b"b-bytes"})

    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert sorted(result["added"]) == ["a.pdf", "b.pdf"]
    assert (drive_dir / "a.pdf").read_bytes() == b"a-bytes"
    assert (drive_dir / "b.pdf").read_bytes() == b"b-bytes"


def test_duplicate_names_are_disambiguated_and_both_saved(fake_env, monkeypatch):
    """同一フォルダ内に同名（拡張子込みで完全一致）ファイルが2件あっても、両方が別名で残る。"""
    drive_dir = fake_env
    drive_files = [
        {"id": "aaaaaaaa1111", "name": "report.pdf", "mimeType": "application/pdf"},
        {"id": "bbbbbbbb2222", "name": "report.pdf", "mimeType": "application/pdf"},
    ]
    _use_fake_service(
        monkeypatch,
        drive_files,
        media_contents={"aaaaaaaa1111": b"content-A", "bbbbbbbb2222": b"content-B"},
    )

    result = google_drive_sync.sync_google_drive_files(verbose=False)

    expected_names = {"report_aaaaaaaa.pdf", "report_bbbbbbbb.pdf"}
    assert set(result["added"]) == expected_names
    # 素の名前（識別子なし）のファイルは作られない
    assert not (drive_dir / "report.pdf").exists()
    assert (drive_dir / "report_aaaaaaaa.pdf").read_bytes() == b"content-A"
    assert (drive_dir / "report_bbbbbbbb.pdf").read_bytes() == b"content-B"


def test_three_way_duplicate_names_are_all_disambiguated_and_saved(fake_env, monkeypatch):
    """3件以上の同名ファイルが存在しても、全件が別名で欠落なく保存される。"""
    drive_dir = fake_env
    drive_files = [
        {"id": "aaaaaaaa1111", "name": "dup.txt", "mimeType": "text/plain"},
        {"id": "bbbbbbbb2222", "name": "dup.txt", "mimeType": "text/plain"},
        {"id": "cccccccc3333", "name": "dup.txt", "mimeType": "text/plain"},
    ]
    media_contents = {
        "aaaaaaaa1111": b"content-A",
        "bbbbbbbb2222": b"content-B",
        "cccccccc3333": b"content-C",
    }
    _use_fake_service(monkeypatch, drive_files, media_contents=media_contents)

    result = google_drive_sync.sync_google_drive_files(verbose=False)

    expected_names = {"dup_aaaaaaaa.txt", "dup_bbbbbbbb.txt", "dup_cccccccc.txt"}
    assert set(result["added"]) == expected_names
    assert (drive_dir / "dup_aaaaaaaa.txt").read_bytes() == b"content-A"
    assert (drive_dir / "dup_bbbbbbbb.txt").read_bytes() == b"content-B"
    assert (drive_dir / "dup_cccccccc.txt").read_bytes() == b"content-C"


def test_duplicate_name_collision_logs_warning(fake_env, monkeypatch, caplog):
    """同名衝突を検出した際に警告ログが出力されることを確認する。"""
    drive_files = [
        {"id": "aaaaaaaa1111", "name": "report.pdf", "mimeType": "application/pdf"},
        {"id": "bbbbbbbb2222", "name": "report.pdf", "mimeType": "application/pdf"},
    ]
    _use_fake_service(
        monkeypatch,
        drive_files,
        media_contents={"aaaaaaaa1111": b"content-A", "bbbbbbbb2222": b"content-B"},
    )

    with caplog.at_level(logging.WARNING, logger="google_drive_sync"):
        google_drive_sync.sync_google_drive_files(verbose=False)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("report.pdf" in r.getMessage() for r in warnings)


def test_duplicate_resolved_next_sync_reverts_to_plain_name_and_removes_stale_copy(fake_env, monkeypatch):
    """次回同期で重複の片方がDrive側から削除された場合、識別子付きの旧ファイルはstaleとして
    削除され、残った方が素のファイル名で保存されることを確認する。"""
    drive_dir = fake_env
    drive_files = [
        {"id": "aaaaaaaa1111", "name": "report.pdf", "mimeType": "application/pdf"},
        {"id": "bbbbbbbb2222", "name": "report.pdf", "mimeType": "application/pdf"},
    ]
    _use_fake_service(
        monkeypatch,
        drive_files,
        media_contents={"aaaaaaaa1111": b"content-A", "bbbbbbbb2222": b"content-B"},
    )
    google_drive_sync.sync_google_drive_files(verbose=False)
    assert (drive_dir / "report_aaaaaaaa.pdf").exists()
    assert (drive_dir / "report_bbbbbbbb.pdf").exists()

    # Drive側から bbbbbbbb2222 が削除され、衝突が解消された状態を模擬する
    _use_fake_service(monkeypatch, [drive_files[0]], media_contents={"aaaaaaaa1111": b"content-A"})
    result = google_drive_sync.sync_google_drive_files(verbose=False)

    # 生存しているファイル(aaaaaaaa1111)は識別子無しの新しいdest_pathで再ダウンロードされるため、
    # 識別子付きの旧ファイル2つ（生存側・消滅側の両方）はどちらも「Drive上の現状と一致しないもの」
    # としてstale削除される。内容自体はreport.pdfとして生き残るためデータ欠落ではない。
    assert sorted(result["removed"]) == ["report_aaaaaaaa.pdf", "report_bbbbbbbb.pdf"]
    assert "report.pdf" in result["added"]
    assert not (drive_dir / "report_aaaaaaaa.pdf").exists()
    assert not (drive_dir / "report_bbbbbbbb.pdf").exists()
    assert (drive_dir / "report.pdf").read_bytes() == b"content-A"


def _sync_n_files(monkeypatch, n: int) -> None:
    drive_files = [{"id": f"id{i}", "name": f"file{i}.pdf", "mimeType": "application/pdf"} for i in range(n)]
    media_contents = {f"id{i}": f"content-{i}".encode() for i in range(n)}
    _use_fake_service(monkeypatch, drive_files, media_contents=media_contents)
    google_drive_sync.sync_google_drive_files(verbose=False)


def test_removal_blocked_when_majority_of_files_disappear_at_once(fake_env, monkeypatch, caplog):
    """既存ファイルの半数以上が一度に消えたと判定された場合、削除せずブロックすることを確認する。"""
    drive_dir = fake_env
    _sync_n_files(monkeypatch, 6)
    assert len(list(drive_dir.iterdir())) == 6

    # 6件中1件しかDrive上で確認できない（5件=約83%が消えたように見える）状態を模擬する
    _use_fake_service(
        monkeypatch,
        [{"id": "id0", "name": "file0.pdf", "mimeType": "application/pdf"}],
        media_contents={"id0": b"content-0"},
    )
    with caplog.at_level(logging.WARNING, logger="google_drive_sync"):
        result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result["removed"] == []
    assert sorted(result["removal_blocked_files"]) == [f"file{i}.pdf" for i in range(1, 6)]
    # ブロックされたファイルはローカルから削除されない
    for i in range(1, 6):
        assert (drive_dir / f"file{i}.pdf").exists()
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("削除をスキップ" in r.getMessage() for r in warnings)


def test_removal_proceeds_when_minority_of_files_disappear(fake_env, monkeypatch):
    """半数未満の消失であれば、従来通り通常の削除が実行されることを確認する（回帰確認）。"""
    drive_dir = fake_env
    _sync_n_files(monkeypatch, 6)

    # 6件中5件が引き続きDrive上で確認できる（1件=約17%の消失）状態を模擬する
    remaining = [{"id": f"id{i}", "name": f"file{i}.pdf", "mimeType": "application/pdf"} for i in range(5)]
    media_contents = {f"id{i}": f"content-{i}".encode() for i in range(5)}
    _use_fake_service(monkeypatch, remaining, media_contents=media_contents)
    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result["removed"] == ["file5.pdf"]
    assert result["removal_blocked_files"] == []
    assert not (drive_dir / "file5.pdf").exists()


def test_removal_blocked_when_drive_returns_no_files_at_all(fake_env, monkeypatch, caplog):
    """Drive APIから1件も取得できなかった場合、フォルダが空なのか取得失敗なのか区別できないため
    削除処理自体をスキップすることを確認する。"""
    drive_dir = fake_env
    _sync_n_files(monkeypatch, 2)
    assert len(list(drive_dir.iterdir())) == 2

    _use_fake_service(monkeypatch, [])
    with caplog.at_level(logging.WARNING, logger="google_drive_sync"):
        result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result["removed"] == []
    assert sorted(result["removal_blocked_files"]) == ["file0.pdf", "file1.pdf"]
    assert (drive_dir / "file0.pdf").exists()
    assert (drive_dir / "file1.pdf").exists()
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("1件も確認できません" in r.getMessage() for r in warnings)


def test_removal_not_blocked_when_existing_file_count_below_min_threshold(fake_env, monkeypatch):
    """既存ファイル数がREMOVAL_BLOCK_MIN_EXISTING_FILES未満なら、消失割合が50%以上でも
    件数条件を満たさないため通常通り削除されることを確認する（境界値）。"""
    drive_dir = fake_env
    assert google_drive_sync.REMOVAL_BLOCK_MIN_EXISTING_FILES == 5
    _sync_n_files(monkeypatch, 4)

    # 4件中3件(75%)が消えるが、既存件数が閾値(5件)未満のためブロックされない
    _use_fake_service(
        monkeypatch,
        [{"id": "id0", "name": "file0.pdf", "mimeType": "application/pdf"}],
        media_contents={"id0": b"content-0"},
    )
    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result["removal_blocked_files"] == []
    assert sorted(result["removed"]) == ["file1.pdf", "file2.pdf", "file3.pdf"]
    for name in ("file1.pdf", "file2.pdf", "file3.pdf"):
        assert not (drive_dir / name).exists()


def test_removal_blocked_at_min_existing_file_count_threshold(fake_env, monkeypatch):
    """既存ファイル数がちょうどREMOVAL_BLOCK_MIN_EXISTING_FILES件の場合も、件数条件を
    満たすものとしてブロックされることを確認する（境界値、>=の包含確認）。"""
    drive_dir = fake_env
    _sync_n_files(monkeypatch, 5)

    # 5件中3件(60%)が消える
    remaining = [{"id": f"id{i}", "name": f"file{i}.pdf", "mimeType": "application/pdf"} for i in range(2)]
    media_contents = {f"id{i}": f"content-{i}".encode() for i in range(2)}
    _use_fake_service(monkeypatch, remaining, media_contents=media_contents)
    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result["removed"] == []
    assert sorted(result["removal_blocked_files"]) == ["file2.pdf", "file3.pdf", "file4.pdf"]
    for name in ("file2.pdf", "file3.pdf", "file4.pdf"):
        assert (drive_dir / name).exists()


def test_removal_blocked_at_exactly_50_percent_ratio(fake_env, monkeypatch):
    """消失割合がちょうど50%の場合、閾値の>=条件によりブロックされることを確認する（境界値）。"""
    drive_dir = fake_env
    _sync_n_files(monkeypatch, 100)
    assert len(list(drive_dir.iterdir())) == 100

    # 100件中50件(ちょうど50%)が消える
    remaining = [{"id": f"id{i}", "name": f"file{i}.pdf", "mimeType": "application/pdf"} for i in range(50)]
    media_contents = {f"id{i}": f"content-{i}".encode() for i in range(50)}
    _use_fake_service(monkeypatch, remaining, media_contents=media_contents)
    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result["removed"] == []
    assert sorted(result["removal_blocked_files"]) == [f"file{i}.pdf" for i in range(50, 100)]


def test_removal_not_blocked_at_49_percent_ratio(fake_env, monkeypatch):
    """消失割合が50%未満（49%程度）であれば、ブロックされず通常通り削除されることを
    確認する（境界値）。"""
    drive_dir = fake_env
    _sync_n_files(monkeypatch, 100)

    # 100件中51件(49%消失)が引き続きDrive上で確認できる
    remaining = [{"id": f"id{i}", "name": f"file{i}.pdf", "mimeType": "application/pdf"} for i in range(51)]
    media_contents = {f"id{i}": f"content-{i}".encode() for i in range(51)}
    _use_fake_service(monkeypatch, remaining, media_contents=media_contents)
    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result["removal_blocked_files"] == []
    assert sorted(result["removed"]) == [f"file{i}.pdf" for i in range(51, 100)]
    for i in range(51, 100):
        assert not (drive_dir / f"file{i}.pdf").exists()


def test_failed_downloads_excluded_from_removal_and_stale_calculation(fake_env, monkeypatch):
    """ダウンロード失敗したファイルは「消えたファイル」の判定対象から除外され、削除も
    ブロック対象にもならないことを確認する（failed_namesの扱い）。"""
    drive_dir = fake_env
    _sync_n_files(monkeypatch, 6)  # file0.pdf 〜 file5.pdf

    # file0は再ダウンロード成功、file1はダウンロード失敗(消えたと誤判定されるべきでない)、
    # file2〜file4は変更なしで存在確認、file5はDrive一覧に無い(真に消えた=1/6≒17%)。
    drive_files = [
        {"id": "id0", "name": "file0.pdf", "mimeType": "application/pdf"},
        {"id": "id1", "name": "file1.pdf", "mimeType": "application/pdf"},
        {"id": "id2", "name": "file2.pdf", "mimeType": "application/pdf"},
        {"id": "id3", "name": "file3.pdf", "mimeType": "application/pdf"},
        {"id": "id4", "name": "file4.pdf", "mimeType": "application/pdf"},
    ]
    media_contents = {
        "id0": b"content-0-v2",
        "id1": RuntimeError("simulated network error"),
        "id2": b"content-2",
        "id3": b"content-3",
        "id4": b"content-4",
    }
    _use_fake_service(monkeypatch, drive_files, media_contents=media_contents)
    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result["removed"] == ["file5.pdf"]
    assert result["removal_blocked_files"] == []
    assert result["skipped"] == ["file1.pdf"]
    # ダウンロード失敗したfile1は「消えた」と誤判定されず、旧内容のまま残る
    assert (drive_dir / "file1.pdf").read_bytes() == b"content-1"
    assert not (drive_dir / "file5.pdf").exists()


def test_failed_downloads_alone_prevent_zero_confirmed_block_but_ratio_block_still_applies(
    fake_env, monkeypatch, caplog
):
    """存在確認できたファイルがダウンロード失敗のみであっても「0件確認」経路には入らず、
    通常のratio判定に進み、消失割合が閾値以上であればブロックされることを確認する。"""
    drive_dir = fake_env
    _sync_n_files(monkeypatch, 6)  # file0.pdf 〜 file5.pdf

    # Drive上にはfile0, file1のみ存在確認できるが両方ダウンロードに失敗する。
    # 残るfile2〜file5(4件, 約67%)は本当に消えたのか区別できないためブロックされることを確認する。
    drive_files = [
        {"id": "id0", "name": "file0.pdf", "mimeType": "application/pdf"},
        {"id": "id1", "name": "file1.pdf", "mimeType": "application/pdf"},
    ]
    media_contents = {
        "id0": RuntimeError("simulated network error"),
        "id1": RuntimeError("simulated network error"),
    }
    _use_fake_service(monkeypatch, drive_files, media_contents=media_contents)
    with caplog.at_level(logging.WARNING, logger="google_drive_sync"):
        result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result["removed"] == []
    assert sorted(result["removal_blocked_files"]) == ["file2.pdf", "file3.pdf", "file4.pdf", "file5.pdf"]
    # ダウンロードに失敗したfile0, file1自体はstale扱いされずそのまま残る
    assert (drive_dir / "file0.pdf").exists()
    assert (drive_dir / "file1.pdf").exists()
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("既存6件中4件" in r.getMessage() for r in warnings)
