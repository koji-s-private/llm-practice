"""google_drive_sync.py のテスト。

実際のGoogle Drive API・OAuth認証は一切行わず、_get_drive_service() をフェイクの
Driveサービスクライアントに monkeypatch して、ミラー処理（export/ダウンロード判定・
スキップ・削除検知）だけを検証する（tests/test_ingest.py の fake_env フィクスチャに倣う）。
"""

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

    assert result == {"added": [], "updated": [], "removed": [], "skipped": ["video.mp4"]}
    assert list(drive_dir.iterdir()) == []


def test_file_re_downloaded_on_second_sync_is_marked_updated(fake_env, monkeypatch):
    drive_files = [{"id": "pdf1", "name": "manual.pdf", "mimeType": "application/pdf"}]
    _use_fake_service(monkeypatch, drive_files, media_contents={"pdf1": b"v1"})
    google_drive_sync.sync_google_drive_files(verbose=False)

    _use_fake_service(monkeypatch, drive_files, media_contents={"pdf1": b"v2"})
    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result == {"added": [], "updated": ["manual.pdf"], "removed": [], "skipped": []}


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

    assert result == {"added": [], "updated": ["keep.pdf"], "removed": ["gone.pdf"], "skipped": []}
    assert not (drive_dir / "gone.pdf").exists()
    assert (drive_dir / "keep.pdf").exists()


def test_sync_is_skipped_when_folder_id_not_set(fake_env, monkeypatch):
    monkeypatch.delenv("GOOGLE_DRIVE_FOLDER_ID", raising=False)

    def _fail_if_called():
        raise AssertionError("GOOGLE_DRIVE_FOLDER_ID未設定時はDrive APIを呼び出すべきではない")

    monkeypatch.setattr(google_drive_sync, "_get_drive_service", lambda: _fail_if_called())

    result = google_drive_sync.sync_google_drive_files(verbose=False)

    assert result == {"added": [], "updated": [], "removed": [], "skipped": []}


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

    assert result == {"added": [], "updated": [], "removed": [], "skipped": ["manual.pdf"]}
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
