"""
data/ 配下のドキュメント（.pdf / .txt / .md）を Chroma ベクトルDBに同期するモジュール。

- ライブラリとして: `from ingest import sync_data_dir` を app.py から呼び出し、
  起動時に自動でDBを最新状態に同期する。
- CLIとして: `python ingest.py` を手動実行しても同じ同期処理が走る。

追加・変更されたファイルのみ差分で取り込み、data/ から削除されたファイルは
ベクトルDBからも自動的に削除する（DBとdata/フォルダの内容がズレないようにするため）。
判定には chroma_db/manifest.json にファイル名・更新日時・サイズ・チャンクIDを記録している。

sync_data_dir()は複数タブ（複数Streamlitセッション）や複数プロセスから同時に呼ばれても
安全なよう、chroma_db/sync.lock を使ったファイルロックで処理全体を排他制御している
（詳細は sync_data_dir() のdocstring参照）。

data/ 直下だけでなく、data/conversations/<thread_id>/ のようなサブフォルダも再帰的に走査する
（app.py のファイルアップロード機能・会話自動保存機能で使用）。会話ログはそのスレッドIDを
チャンクのメタデータ(thread_id)として付与し、rag_chain.build_agent(thread_id) が
「共通ナレッジ＋そのスレッドの会話ログ」だけを検索できるようにしている
（別スレッドの会話ログが回答に混ざらないようにするため）。

PDFの読み込みは2段構成:
  1) PyMuPDF（高速）でまず抽出する。
  2) 抽出できた文字数が極端に少ない場合（図解・スキャンPDFの疑い）だけ、
     Docling（レイアウト認識・OCR内蔵、やや重い）で再解析する。
通常のテキストPDFは1)だけで高速に処理され、図解・スキャンPDFのような
「pypdf/PyMuPDFでは苦手なファイル」だけが2)のコストを払う仕組みにすることで、
処理速度への影響を最小限にしている。Doclingが未インストールの場合は自動的に
1)の結果のみを使う（インストールは任意）。
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from filelock import FileLock, Timeout
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_chain import COLLECTION_NAME, GLOBAL_THREAD_ID, PERSIST_DIR, get_vectorstore

try:
    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

logger = logging.getLogger(__name__)


def _log_progress(verbose: bool, message: str, *args) -> None:
    """同期処理の進捗メッセージをloggerに出力する（print()による二重出力を避けるための一本化窓口）。

    verbose=Trueならinfoレベル、Falseならdebugレベルで出力する。実際にコンソールへ
    表示されるかどうかはロガーの設定次第（例: CLIのmain()がlogging.basicConfig()で
    INFOレベルのコンソールハンドラを設定していればverbose時のみ表示される）。
    読み込み失敗などの警告は本関数を使わず、常に logger.warning() を直接呼ぶ
    （verboseの値に関わらず常に見えるべき情報のため）。
    """
    logger.log(logging.INFO if verbose else logging.DEBUG, message, *args)


DATA_DIR = Path(__file__).parent / "data"
CONVERSATIONS_DIRNAME = "conversations"
MANIFEST_PATH = PERSIST_DIR / "manifest.json"

# 同一プロセス内の複数Streamlitセッション（複数タブ）や複数プロセス（app.py/api/main.py の
# 併用など）からsync_data_dir()が同時に呼ばれた場合の排他制御用ロックファイル。
# manifest.json読み込み→ベクトルDB更新→manifest.json書き込みの一連の処理を
# 単一の実行者だけが行うようにし、read-modify-writeの競合（lost update・
# 同一チャンクの重複登録）を防ぐ。
SYNC_LOCK_PATH = PERSIST_DIR / "sync.lock"
# ロック取得を待つ最大秒数。通常の同期処理は数秒〜十数秒で終わるため、
# それより十分長い待ち時間を設けつつ、無限待機は避ける。
SYNC_LOCK_TIMEOUT_SECONDS = 60

# 拡張子ごとのローダー対応表（PDFは実際には_load_pdf()で2段構成の判定を行う）
LOADERS = {
    ".pdf": PyMuPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}

# 1ページあたりの抽出文字数がこれ未満の場合、「うまくテキスト抽出できていない
# （図解・スキャンPDFの疑いがある）」とみなしDoclingでの再解析を試みる。
MIN_CHARS_PER_PAGE_FOR_FAST_PATH = 40

# memory.save_conversation() が会話ログのMarkdownに書き込むメタデータ行を検出する正規表現。
FALLBACK_METADATA_PATTERN = re.compile(r"^-\s*一般知識フォールバック:\s*true\s*$", re.MULTILINE)


def safe_upload_dest(filename: str) -> Path | None:
    """アップロードされたファイル名を DATA_DIR 配下の安全な書き込み先パスに変換する。

    ディレクトリ部分（`../` 等）を除いた素のファイル名のみを使い、
    resolve() 後に DATA_DIR 配下から外れていないかを最終チェックする。
    DATA_DIR の外を指す場合（パストラバーサルの疑いがある場合）は None を返す。
    同名ファイルが既に存在するかどうかはチェックしない（呼び出し元が
    resolve_upload_dest() で別途重複を扱う）。
    """
    dest = (DATA_DIR / Path(filename).name).resolve()
    if dest.parent != DATA_DIR.resolve():
        return None
    return dest


def resolve_upload_dest(filename: str, taken_paths: set[Path] | None = None) -> Path | None:
    """アップロードされたファイル名から、上書きを避けた実際の書き込み先パスを求める。

    safe_upload_dest() が返すパスに既にファイルが存在する場合（＝同名ファイルの
    アップロード）、または同一アップロードバッチ内で既に使用済みのパスの場合
    （taken_paths、同じバッチ内の同名ファイル対策）は、無警告での上書きを避けるため
    "name (2).ext" のように連番サフィックスを付けた空いているパスを返す。
    呼び出し元は、戻り値のファイル名が元のファイル名と異なっていた場合に
    ユーザーへ警告を表示することを想定している。
    パストラバーサルの疑いがある場合は safe_upload_dest() と同様に None を返す。
    """
    dest = safe_upload_dest(filename)
    if dest is None:
        return None

    taken = taken_paths if taken_paths is not None else set()
    if dest not in taken and not dest.exists():
        return dest

    stem, suffix = dest.stem, dest.suffix
    counter = 2
    candidate = dest.with_name(f"{stem} ({counter}){suffix}")
    while candidate in taken or candidate.exists():
        counter += 1
        candidate = dest.with_name(f"{stem} ({counter}){suffix}")
    return candidate


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning(
            "%s の読み込みに失敗しました（壊れたJSONの可能性があります）。"
            "空のマニフェストとして扱い、全ファイルを再取り込みします。",
            MANIFEST_PATH,
        )
        return {}
    except FileNotFoundError:
        # exists()での確認直後にファイルが削除された場合（TOCTOU）に備える。
        logger.warning(
            "%s の読み込み中にファイルが見つかりませんでした。"
            "空のマニフェストとして扱い、全ファイルを再取り込みします。",
            MANIFEST_PATH,
        )
        return {}


def _save_manifest(manifest: dict) -> None:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, MANIFEST_PATH)


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _load_pdf(path: Path, verbose: bool = True) -> list:
    """PDFを読み込む（PyMuPDFで高速抽出 → 必要な場合のみDoclingでフォールバック）。"""
    try:
        fast_docs = PyMuPDFLoader(str(path)).load()
    except Exception as e:
        # PyMuPDF自体が例外を送出した場合（暗号化PDF・破損PDF・特殊なPDF構造など）。
        # Doclingが利用可能なら、レイアウト認識・OCRで読める可能性があるためフォールバックする。
        # Doclingが未インストール、またはDoclingも失敗した場合は元の例外をそのまま送出し、
        # 呼び出し元（sync_data_dir）で "failed" として記録・次回リトライさせる。
        if not DOCLING_AVAILABLE:
            raise
        _log_progress(verbose, "    → PyMuPDFでの読み込みに失敗したため、Doclingでの再解析を試みます（%s）...", e)
        docling_docs = _load_pdf_with_docling(path, verbose=verbose)
        if not docling_docs:
            raise
        docling_chars = sum(len(d.page_content.strip()) for d in docling_docs)
        _log_progress(verbose, "    → Doclingで%d文字を抽出しました。", docling_chars)
        return docling_docs

    total_chars = sum(len(d.page_content.strip()) for d in fast_docs)
    avg_chars_per_page = total_chars / max(len(fast_docs), 1)

    if avg_chars_per_page >= MIN_CHARS_PER_PAGE_FOR_FAST_PATH or not DOCLING_AVAILABLE:
        return fast_docs

    _log_progress(
        verbose,
        "    → テキスト抽出量が少ない（平均%.0f文字/ページ）ため、"
        "Doclingで図解・OCR解析を試みます（時間がかかる場合があります）...",
        avg_chars_per_page,
    )
    docling_docs = _load_pdf_with_docling(path, verbose=verbose)
    if docling_docs:
        docling_chars = sum(len(d.page_content.strip()) for d in docling_docs)
        if docling_chars > total_chars:
            _log_progress(
                verbose, "    → Doclingで%d文字を抽出しました（PyMuPDF: %d文字）。", docling_chars, total_chars
            )
            return docling_docs

    return fast_docs


def _load_pdf_with_docling(path: Path, verbose: bool = True) -> list:
    """Doclingでの再解析を試みる。失敗した場合は空リストを返す（例外は送出しない）。"""
    try:
        docling_docs = DoclingLoader(file_path=str(path), export_type=ExportType.MARKDOWN).load()
    except Exception as e:
        _log_progress(verbose, "    → Docling解析に失敗しました（%s）", e)
        return []
    for doc in docling_docs:
        doc.metadata.setdefault("source", str(path))
    return docling_docs


def _thread_id_for(rel_path: str) -> str:
    """相対パスから会話スレッドIDを判定する。

    data/conversations/<thread_id>/xxx.md → そのthread_id（このスレッドでのみ検索対象）
    data/conversations/xxx.md（サブフォルダなし、スレッド機能追加前の古い保存形式）
        → GLOBAL_THREAD_ID として扱う（後方互換。以前の会話ログも消えず、共通ナレッジとして生き続ける）
    それ以外（data/直下のファイルやアップロードファイルなど） → GLOBAL_THREAD_ID（全スレッド共通）
    """
    parts = Path(rel_path).parts
    if len(parts) >= 3 and parts[0] == CONVERSATIONS_DIRNAME:
        return parts[1]
    return GLOBAL_THREAD_ID


def _is_fallback_conversation(docs: list) -> bool:
    """分割前の生ドキュメントの本文から、一般知識フォールバック回答の会話ログかどうかを判定する。

    memory.save_conversation() が書き込む「- 一般知識フォールバック: true」という
    メタデータ行を正規表現で検出する。通常のドキュメント・アップロードファイルなど
    該当行が無いものは False になる。
    """
    return any(FALLBACK_METADATA_PATTERN.search(doc.page_content) for doc in docs)


def data_dir_signature() -> tuple[int, float]:
    """data/ の変更有無を、内容を読まずにstat()だけで軽量に判定するためのシグネチャを返す。

    sync_data_dir()が対象とするのと同じファイル集合（拡張子がLOADERSに含まれるもの）に対し、
    (ファイル数, 最新mtime) のタプルを返す。ファイルの読み込み・分割・埋め込み・
    ベクトルストア接続・manifest.jsonの読み書きは一切行わない
    （それらの重い処理を避けるための、あくまで近似的な変更検知）。
    app.py側はStreamlitが再実行されるたびにこの関数を呼び、前回値と比較することで、
    「data/に変更があった場合だけ」本格的な sync_data_dir() を呼び出す。
    DATA_DIRが存在しない、または対象ファイルが1つもない場合は (0, 0.0) を返す。
    """
    if not DATA_DIR.exists():
        return (0, 0.0)
    target_files = [f for f in DATA_DIR.rglob("*") if f.is_file() and f.suffix.lower() in LOADERS]
    if not target_files:
        return (0, 0.0)
    latest_mtime = max(f.stat().st_mtime for f in target_files)
    return (len(target_files), latest_mtime)


def sync_data_dir(verbose: bool = True) -> dict:
    """data/ の内容とベクトルDBを同期する。

    戻り値: {"added": [...], "updated": [...], "removed": [...], "failed": [...]}
    変更がなければ全て空リストになる（＝差分がなければ何もしない）。
    読み込み・分割に失敗したファイルは "failed" に積まれ、manifestには記録されない
    （＝次回同期時に再度リトライされる）。他のファイルの同期は継続される。

    manifestの保存はファイル1件ごとに行う（全件処理後にまとめて保存はしない）。
    ベクトルDBへの追加・削除が完了するたびに都度manifestを保存することで、
    途中でプロセスが中断（クラッシュ・強制終了など）してもmanifestには
    「実際にDBへ反映済みのファイル」だけが記録された状態を保ち、次回同期時に
    同じ内容のチャンクが重複登録されるのを防ぐ。

    同一プロセス内の複数Streamlitセッション（複数タブ）や複数プロセスから同時に
    呼ばれても安全なよう、manifest.json読み込み〜ベクトルDB更新〜manifest.json書き込みの
    一連の処理全体をファイルロック（SYNC_LOCK_PATH）で排他制御する。先に実行している
    処理が終わるまで後発の呼び出しはブロックされ、順番に処理される。
    SYNC_LOCK_TIMEOUT_SECONDS以内にロックを獲得できなかった場合は
    `filelock.Timeout` をそのまま送出する（呼び出し元のapp.py/api/main.py側で
    他の例外と同様にエラーとして扱われる）。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with FileLock(str(SYNC_LOCK_PATH), timeout=SYNC_LOCK_TIMEOUT_SECONDS):
            return _sync_data_dir_locked(verbose=verbose)
    except Timeout:
        logger.warning(
            "%s のロック取得がタイムアウトしました（他のセッションが同期中の可能性があります）。",
            SYNC_LOCK_PATH,
        )
        raise


def _sync_data_dir_locked(verbose: bool) -> dict:
    """sync_data_dir()の本体（呼び出し元がファイルロックを取得済みであることが前提）。"""
    vector_store = get_vectorstore()
    manifest = _load_manifest()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # data/ 直下だけでなく、data/conversations/ などのサブフォルダも再帰的に走査する
    current_files = {
        str(f.relative_to(DATA_DIR)): f for f in DATA_DIR.rglob("*") if f.is_file() and f.suffix.lower() in LOADERS
    }

    result = {"added": [], "updated": [], "removed": [], "failed": []}

    # 追加 or 変更されたファイルを取り込む
    for name, path in current_files.items():
        fingerprint = _fingerprint(path)
        entry = manifest.get(name)
        unchanged = entry and entry.get("mtime") == fingerprint["mtime"] and entry.get("size") == fingerprint["size"]
        if unchanged:
            continue

        try:
            if path.suffix.lower() == ".pdf":
                docs = _load_pdf(path, verbose=verbose)
            else:
                loader = LOADERS[path.suffix.lower()](str(path))
                docs = loader.load()
            chunks = splitter.split_documents(docs)
        except Exception as e:
            # 1ファイルの読み込み失敗（破損PDF・パスワード付きPDF・不正なエンコーディング等）で
            # 他の正常なファイルの同期まで止めないよう、ログに残してスキップする。
            # manifestには記録しないので、次回同期時に再度リトライされる。
            logger.warning("%s の読み込みに失敗したためスキップします: %s", name, e)
            result["failed"].append(name)
            continue

        # 会話ログはそのスレッドのみ、それ以外（通常ドキュメント・アップロード）は
        # 全スレッド共通で検索できるよう、チャンクにthread_idをメタデータとして付与する。
        thread_id = _thread_id_for(name)
        # 一般知識フォールバック回答の会話ログは、根拠のないままベクトルDBに再学習されると
        # 以降の検索結果として再ヒットし、あたかもドキュメントの裏付けがあるかのように
        # 扱われてしまう。全チャンクに一貫してis_fallbackキーを持たせることで、
        # rag_chain.retrieve_context側のメタデータフィルタが確実に効くようにする。
        is_fallback = _is_fallback_conversation(docs)
        for chunk in chunks:
            chunk.metadata["thread_id"] = thread_id
            chunk.metadata["is_fallback"] = is_fallback

        # 新チャンクの追加を先に行い、成功してから旧チャンクを削除する（delete→addの逆順）。
        # delete→addの順だと、削除成功後にadd_documents()が失敗した場合、旧チャンクは
        # 消えたのに新チャンクも登録されない「データが検索対象から消える」状態になり、
        # manifestも更新されないため次回同期でも気づかれず放置されてしまう。
        # add→deleteの順なら、追加が失敗しても旧チャンクがそのまま残るため安全
        # （追加成功〜削除完了までの一瞬だけ新旧チャンクが両方検索にヒットしうるが、
        # データが消えるより十分マシな許容範囲の副作用とする）。
        try:
            chunk_ids = vector_store.add_documents(documents=chunks) if chunks else []
        except Exception as e:
            logger.warning("%s のベクトルストアへの追加に失敗したためスキップします: %s", name, e)
            result["failed"].append(name)
            continue

        if entry and entry.get("chunk_ids"):
            vector_store.delete(ids=entry["chunk_ids"])

        manifest[name] = {**fingerprint, "chunk_ids": chunk_ids}
        result["updated" if entry else "added"].append(name)
        action = "更新" if entry else "追加"
        _log_progress(verbose, "%s: %s（%dチャンク）", action, name, len(chunks))

        # ファイル1件ごとにmanifestを保存する。全件処理後にまとめて保存する設計だと、
        # add_documents()でDBへの追加が完了した後・保存前にプロセスが中断した場合、
        # DBには反映済みなのにmanifestには記録されない状態になり、次回同期時に
        # 同じ内容のチャンクが重複登録されてしまう。都度保存すればその不整合を防げる。
        _save_manifest(manifest)

    # data/ から削除されたファイルをDBからも削除
    for name in list(manifest.keys()):
        if name not in current_files:
            chunk_ids = manifest[name].get("chunk_ids") or []
            if chunk_ids:
                vector_store.delete(ids=chunk_ids)
            del manifest[name]
            result["removed"].append(name)
            _log_progress(verbose, "削除: %s", name)
            # 追加・更新時と同様、削除もDB反映とmanifest保存を1件ごとに一致させる。
            _save_manifest(manifest)

    # 全ファイルが失敗し1件も追加・更新・削除が無かった場合でも、manifest.jsonが
    # 存在しない状態のまま終わらないよう最後に保存しておく（内容は変わらないため冪等）。
    _save_manifest(manifest)

    if not any(result.values()):
        _log_progress(verbose, "変更はありませんでした（すでに同期済みです）。")

    return result


def list_indexed_files() -> list[dict]:
    """サイドバーの一覧表示用に、インデックス済みファイルの情報をmanifestから取得する。

    会話ログ（data/conversations/配下）はユーザーがアップロード・削除で管理する対象ではないため
    除外し、data/直下のドキュメントのみを返す。戻り値はファイル名昇順のリストで、各要素は
    {"name": ファイル名, "chunk_count": チャンク数} の辞書。
    """
    manifest = _load_manifest()
    return [
        {"name": name, "chunk_count": len(entry.get("chunk_ids", []))}
        for name, entry in sorted(manifest.items())
        if Path(name).parts[0] != CONVERSATIONS_DIRNAME
    ]


def delete_indexed_file(name: str) -> bool:
    """インデックス済みファイルをdata/から削除する。

    ここではmanifest・ベクトルDBの更新は行わない。呼び出し元がこの後sync_data_dir()を
    呼ぶことで、data/にファイルが存在しなくなったことが検知され、DB・manifestから
    自動的に除外される（sync_data_dir()の既存の削除検知ロジックをそのまま利用する）。
    パストラバーサル対策としてsafe_upload_dest()と同じ経路でパスを解決する
    （list_indexed_files()が返すファイルはconversations/配下を除いた
    data/直下のファイルのみのため、これで十分安全に解決できる）。
    対象ファイルが存在しない場合は何もせずFalseを返す。
    """
    path = safe_upload_dest(name)
    if path is None or not path.exists():
        return False
    path.unlink()
    return True


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def print_status() -> None:
    """ベクトルDBの現在の状態を表示する（`python ingest.py --status`）。"""
    manifest = _load_manifest()

    print(f"DB保存先     : {PERSIST_DIR}")
    print(f"コレクション名: {COLLECTION_NAME}")

    if not PERSIST_DIR.exists() or not manifest:
        print("インデックス済みのファイルはまだありません。")
        print("`streamlit run app.py` を起動するか `python ingest.py` を実行してください。")
        return

    try:
        vector_store = get_vectorstore()
        chunk_count = vector_store._collection.count()
    except Exception:
        chunk_count = sum(len(v.get("chunk_ids", [])) for v in manifest.values())

    print(f"インデックス済みファイル数: {len(manifest)}件")
    print(f"チャンク数（ベクトル数）  : {chunk_count}件")
    print(f"DBフォルダのサイズ       : {_dir_size_mb(PERSIST_DIR):.1f} MB")
    print("\nファイル別チャンク数:")
    for name, entry in sorted(manifest.items()):
        print(f"  - {name}: {len(entry.get('chunk_ids', []))}チャンク")


def main():
    parser = argparse.ArgumentParser(description="data/ とベクトルDBの同期・状態確認ツール")
    parser.add_argument(
        "--status",
        action="store_true",
        help="同期は行わず、現在のDBの状態（件数・容量など）だけを表示する",
    )
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    # sync_data_dir(verbose=True)（デフォルト）が出す進捗ログ（logger.info）をCLI実行時に
    # コンソールへ表示するためのハンドラ設定。app.py/api経由（verbose=False）ではこの設定は
    # 行われず、ファイル読み込み失敗時のlogger.warning()のみがPythonの既定動作で表示される。
    # stream=sys.stdoutにするのは、同じmain()内の他のprint()出力との表示順序を揃えるため
    # （logging標準のデフォルトはstderrで、print()と混在すると出力順序が入れ替わりうる）。
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    print("data/ をベクトルDBに同期しています...")
    result = sync_data_dir()
    print(
        f"完了: 追加{len(result['added'])}件 / "
        f"更新{len(result['updated'])}件 / 削除{len(result['removed'])}件 / "
        f"失敗{len(result['failed'])}件"
    )
    print("`streamlit run app.py` でチャットを開始できます。")
    print("（DBの状態を確認したい場合は `python ingest.py --status`）")


if __name__ == "__main__":
    main()
