"""
Google Drive上のGoogle Docs/Sheets/Slides・通常ファイルを data/google_drive/ にミラーし、
既存の ingest.sync_data_dir() に載せてベクトルDBまで反映するモジュール。

- ライブラリとして: `from google_drive_sync import sync_google_drive_files` を呼び出すと、
  Google Driveの指定フォルダの内容を data/google_drive/ にミラーする（ベクトルDBへの反映は
  別途 ingest.sync_data_dir() を呼ぶ必要がある）。
- CLIとして: `python google_drive_sync.py` を実行すると、ミラー → ingest.sync_data_dir() による
  DB反映までを一気通貫で行う（`python ingest.py` のGoogle Drive連携版）。

data/google_drive/ を「ingest.pyが元々前提とするdata/配下の1サブフォルダ」として位置づけることで、
差分検知・チャンク分割・削除検知などの既存ロジックをそのまま流用している（Google Drive固有の
同期ロジックは「Driveの内容をローカルファイルとしてミラーする」ところまでに閉じている）。

OAuth 2.0（デスクトップアプリ種別、読み取り専用スコープ）で認証する。初回はブラウザでの同意操作が
必要で、以降はリフレッシュトークンにより自動更新される。Google Cloud Console側の事前セットアップ
手順は docs/google-drive-setup.md を参照。認証情報（クライアントシークレット・トークン）は
.credentials/ 配下に保存し、.gitignore で除外している。
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import ingest

logger = logging.getLogger(__name__)

# 読み取り専用スコープ（誤ってDrive側の内容を書き換えることがないよう最小権限にする）
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

GOOGLE_DRIVE_DIR = ingest.DATA_DIR / "google_drive"
CREDENTIALS_DIR = Path(__file__).parent / ".credentials"

CLIENT_SECRET_FILE = Path(os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET_FILE", CREDENTIALS_DIR / "client_secret.json"))
TOKEN_FILE = Path(os.environ.get("GOOGLE_OAUTH_TOKEN_FILE", CREDENTIALS_DIR / "token.json"))

# Google Docs/Sheets/Slidesはネイティブ形式のままではDrive上に実体ファイルが無くダウンロードできない
# ため、files.exportエンドポイントで既存ingest.LOADERSが対応する形式にエクスポートしてから取り込む。
GOOGLE_NATIVE_EXPORT_MIME_TYPES = {
    "application/vnd.google-apps.document": (
        ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    "application/vnd.google-apps.spreadsheet": (
        ".xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    "application/vnd.google-apps.presentation": (
        ".pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
}


def _log_progress(verbose: bool, message: str, *args) -> None:
    """ingest._log_progress()と同じ考え方（verbose=Trueならinfo、Falseならdebug）の進捗ログ出力窓口。"""
    logger.log(logging.INFO if verbose else logging.DEBUG, message, *args)


def _get_drive_service():
    """認証済みのDrive APIサービスクライアントを返す。

    既存トークン（TOKEN_FILE）があれば読み込み、期限切れならリフレッシュする。トークンが無い・
    リフレッシュに失敗した場合は InstalledAppFlow.run_local_server() でブラウザ経由の初回認証を行い、
    取得したトークンをTOKEN_FILEに保存する（次回以降はリフレッシュだけで済むようにするため）。
    """
    if not CLIENT_SECRET_FILE.exists():
        raise RuntimeError(
            f"OAuthクライアントシークレットファイルが見つかりません: {CLIENT_SECRET_FILE}\n"
            "Google Cloud Consoleでの事前セットアップが完了していない可能性があります。"
            "docs/google-drive-setup.md の手順を参照してください。"
        )

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.warning("トークンのリフレッシュに失敗したため、再度ブラウザ認証を行います: %s", e)
                creds = None

        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)


def _list_drive_files(service, folder_id: str) -> list[dict]:
    """指定フォルダ直下（ゴミ箱を除く）のファイル一覧を取得する。

    Drive APIは1回のレスポンスで返せる件数に上限があるため、nextPageTokenで
    ページネーションしながら全件集める。
    """
    files = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return files


def _dest_path_for(drive_file: dict) -> Path | None:
    """Driveファイル1件のローカル保存先パスを決める。対応拡張子でない場合はNoneを返す。"""
    mime_type = drive_file["mimeType"]
    name = drive_file["name"]

    if mime_type in GOOGLE_NATIVE_EXPORT_MIME_TYPES:
        export_suffix, _ = GOOGLE_NATIVE_EXPORT_MIME_TYPES[mime_type]
        return GOOGLE_DRIVE_DIR / f"{name}{export_suffix}"

    if Path(name).suffix.lower() not in ingest.LOADERS:
        return None
    return GOOGLE_DRIVE_DIR / name


def _build_download_request(service, drive_file: dict):
    mime_type = drive_file["mimeType"]
    if mime_type in GOOGLE_NATIVE_EXPORT_MIME_TYPES:
        _, export_mime_type = GOOGLE_NATIVE_EXPORT_MIME_TYPES[mime_type]
        return service.files().export_media(fileId=drive_file["id"], mimeType=export_mime_type)
    return service.files().get_media(fileId=drive_file["id"])


def _download_drive_file(service, drive_file: dict, dest_path: Path) -> None:
    request = _build_download_request(service, drive_file)
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def sync_google_drive_files(verbose: bool = True) -> dict:
    """data/google_drive/ をGoogle Driveの指定フォルダ（GOOGLE_DRIVE_FOLDER_ID）の内容にミラーする。

    戻り値: {"added": [...], "updated": [...], "removed": [...], "skipped": [...]}
    ("updated"はローカルに同名ファイルが既にあった場合。内容が実際に変わったかどうかまでは
    ここでは判定せず、後続の ingest.sync_data_dir() がmtime/sizeベースで最終判定する)

    GOOGLE_DRIVE_FOLDER_ID が未設定の場合は同期そのものをスキップし、警告ログを出して
    全キー空リストを返す（認証・API呼び出しは一切行わない）。

    1件のダウンロード・エクスポートに失敗した場合は、そのファイルの旧ローカルコピーを
    誤って削除してしまわないよう「消えたファイル」の判定対象から除外し、次回同期時に再試行する。
    """
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        logger.warning(
            "GOOGLE_DRIVE_FOLDER_ID が未設定のため、Google Drive同期をスキップします。"
            "設定方法は docs/google-drive-setup.md を参照してください。"
        )
        return {"added": [], "updated": [], "removed": [], "skipped": []}

    service = _get_drive_service()
    drive_files = _list_drive_files(service, folder_id)

    GOOGLE_DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    existing_names = {f.name for f in GOOGLE_DRIVE_DIR.iterdir() if f.is_file()}

    result = {"added": [], "updated": [], "removed": [], "skipped": []}
    downloaded_names: set[str] = set()
    failed_names: set[str] = set()

    for drive_file in drive_files:
        name = drive_file["name"]
        dest_path = _dest_path_for(drive_file)
        if dest_path is None:
            logger.warning("%s: 未対応の拡張子のためスキップします。", name)
            result["skipped"].append(name)
            continue

        try:
            _download_drive_file(service, drive_file, dest_path)
        except Exception as e:
            logger.warning("%s のダウンロードに失敗したためスキップします: %s", name, e)
            result["skipped"].append(name)
            failed_names.add(dest_path.name)
            continue

        downloaded_names.add(dest_path.name)
        status = "updated" if dest_path.name in existing_names else "added"
        result[status].append(dest_path.name)
        _log_progress(verbose, "  %s: %s", "更新" if status == "updated" else "追加", dest_path.name)

    for stale_name in sorted(existing_names - downloaded_names - failed_names):
        (GOOGLE_DRIVE_DIR / stale_name).unlink()
        result["removed"].append(stale_name)
        _log_progress(verbose, "  削除（Drive上から消えたファイル）: %s", stale_name)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Google Driveの指定フォルダをdata/google_drive/経由でベクトルDBに同期するツール"
    )
    parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    print("Google Driveの内容を data/google_drive/ にミラーしています...")
    try:
        drive_result = sync_google_drive_files()
    except RuntimeError as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)
    print(
        f"ミラー完了: 追加{len(drive_result['added'])}件 / "
        f"更新{len(drive_result['updated'])}件 / 削除{len(drive_result['removed'])}件 / "
        f"スキップ{len(drive_result['skipped'])}件"
    )

    print("data/ をベクトルDBに同期しています...")
    db_result = ingest.sync_data_dir()
    print(
        f"完了: 追加{len(db_result['added'])}件 / "
        f"更新{len(db_result['updated'])}件 / 削除{len(db_result['removed'])}件 / "
        f"失敗{len(db_result['failed'])}件"
    )


if __name__ == "__main__":
    main()
