"""
data/ 配下のドキュメント（.pdf / .txt / .md）を Chroma ベクトルDBに同期するモジュール。

- ライブラリとして: `from ingest import sync_data_dir` を app.py から呼び出し、
  起動時に自動でDBを最新状態に同期する。
- CLIとして: `python ingest.py` を手動実行しても同じ同期処理が走る。

追加・変更されたファイルのみ差分で取り込み、data/ から削除されたファイルは
ベクトルDBからも自動的に削除する（DBとdata/フォルダの内容がズレないようにするため）。
判定には chroma_db/manifest.json にファイル名・更新日時・サイズ・チャンクIDを記録している。

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
from pathlib import Path

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

DATA_DIR = Path(__file__).parent / "data"
CONVERSATIONS_DIRNAME = "conversations"
MANIFEST_PATH = PERSIST_DIR / "manifest.json"

# 拡張子ごとのローダー対応表（PDFは実際には_load_pdf()で2段構成の判定を行う）
LOADERS = {
    ".pdf": PyMuPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}

# 1ページあたりの抽出文字数がこれ未満の場合、「うまくテキスト抽出できていない
# （図解・スキャンPDFの疑いがある）」とみなしDoclingでの再解析を試みる。
MIN_CHARS_PER_PAGE_FOR_FAST_PATH = 40


def safe_upload_dest(filename: str) -> Path | None:
    """アップロードされたファイル名を DATA_DIR 配下の安全な書き込み先パスに変換する。

    ディレクトリ部分（`../` 等）を除いた素のファイル名のみを使い、
    resolve() 後に DATA_DIR 配下から外れていないかを最終チェックする。
    DATA_DIR の外を指す場合（パストラバーサルの疑いがある場合）は None を返す。
    """
    dest = (DATA_DIR / Path(filename).name).resolve()
    if dest.parent != DATA_DIR.resolve():
        return None
    return dest


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


def _save_manifest(manifest: dict) -> None:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp_path, MANIFEST_PATH)


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _load_pdf(path: Path, verbose: bool = True) -> list:
    """PDFを読み込む（PyMuPDFで高速抽出 → 必要な場合のみDoclingでフォールバック）。"""
    fast_docs = PyMuPDFLoader(str(path)).load()
    total_chars = sum(len(d.page_content.strip()) for d in fast_docs)
    avg_chars_per_page = total_chars / max(len(fast_docs), 1)

    if avg_chars_per_page >= MIN_CHARS_PER_PAGE_FOR_FAST_PATH or not DOCLING_AVAILABLE:
        return fast_docs

    if verbose:
        print(
            f"    → テキスト抽出量が少ない（平均{avg_chars_per_page:.0f}文字/ページ）ため、"
            "Doclingで図解・OCR解析を試みます（時間がかかる場合があります）..."
        )
    try:
        docling_docs = DoclingLoader(
            file_path=str(path), export_type=ExportType.MARKDOWN
        ).load()
        docling_chars = sum(len(d.page_content.strip()) for d in docling_docs)
        if docling_docs and docling_chars > total_chars:
            for doc in docling_docs:
                doc.metadata.setdefault("source", str(path))
            if verbose:
                print(f"    → Doclingで{docling_chars}文字を抽出しました（PyMuPDF: {total_chars}文字）。")
            return docling_docs
    except Exception as e:
        if verbose:
            print(f"    → Docling解析に失敗、PyMuPDFの結果をそのまま使用します（{e}）")

    return fast_docs


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


def data_dir_signature() -> tuple[int, float]:
    """data/ の変更有無を、内容を読まずにstat()だけで軽量に判定するためのシグネチャを返す。

    sync_data_dir()が対象とするのと同じファイル集合（拡張子がLOADERSに含まれるもの）に対し、
    (ファイル数, 最新mtime) のタプルを返す。ファイルの読み込み・分割・埋め込み・
    ベクトルストア接続・manifest.jsonの読み書きは一切行わない（Issue #33で指摘された
    重い処理を避けるための、あくまで近似的な変更検知）。
    app.py側はStreamlitが再実行されるたびにこの関数を呼び、前回値と比較することで、
    「data/に変更があった場合だけ」本格的な sync_data_dir() を呼び出す（Issue #70）。
    DATA_DIRが存在しない、または対象ファイルが1つもない場合は (0, 0.0) を返す。
    """
    if not DATA_DIR.exists():
        return (0, 0.0)
    target_files = [
        f for f in DATA_DIR.rglob("*") if f.is_file() and f.suffix.lower() in LOADERS
    ]
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
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = get_vectorstore()
    manifest = _load_manifest()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # data/ 直下だけでなく、data/conversations/ などのサブフォルダも再帰的に走査する
    current_files = {
        str(f.relative_to(DATA_DIR)): f
        for f in DATA_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in LOADERS
    }

    result = {"added": [], "updated": [], "removed": [], "failed": []}

    # 追加 or 変更されたファイルを取り込む
    for name, path in current_files.items():
        fingerprint = _fingerprint(path)
        entry = manifest.get(name)
        unchanged = (
            entry
            and entry.get("mtime") == fingerprint["mtime"]
            and entry.get("size") == fingerprint["size"]
        )
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
            if verbose:
                print(f"スキップ（読み込み失敗）: {name}（{e}）")
            continue

        # 既存チャンクがあれば先に削除してから入れ直す（重複防止）
        if entry and entry.get("chunk_ids"):
            vector_store.delete(ids=entry["chunk_ids"])

        # 会話ログはそのスレッドのみ、それ以外（通常ドキュメント・アップロード）は
        # 全スレッド共通で検索できるよう、チャンクにthread_idをメタデータとして付与する。
        thread_id = _thread_id_for(name)
        for chunk in chunks:
            chunk.metadata["thread_id"] = thread_id

        chunk_ids = vector_store.add_documents(documents=chunks) if chunks else []

        manifest[name] = {**fingerprint, "chunk_ids": chunk_ids}
        result["updated" if entry else "added"].append(name)
        if verbose:
            action = "更新" if entry else "追加"
            print(f"{action}: {name}（{len(chunks)}チャンク）")

    # data/ から削除されたファイルをDBからも削除
    for name in list(manifest.keys()):
        if name not in current_files:
            chunk_ids = manifest[name].get("chunk_ids") or []
            if chunk_ids:
                vector_store.delete(ids=chunk_ids)
            del manifest[name]
            result["removed"].append(name)
            if verbose:
                print(f"削除: {name}")

    _save_manifest(manifest)

    if verbose and not any(result.values()):
        print("変更はありませんでした（すでに同期済みです）。")

    return result


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
