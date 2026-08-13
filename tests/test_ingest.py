"""ingest.py の差分同期ロジック（追加・更新・削除の判定）のテスト。

実際の埋め込みモデル・Chromaは使わず、get_vectorstore() をテストごとに
軽量なフェイクベクトルストアに monkeypatch して、data/ の内容と manifest.json
に基づく差分判定だけを検証する（ドキュメントローダー・チャンク分割は実物を使う）。
"""

import json
import logging
import os
import threading
import time

import pytest
from filelock import FileLock, Timeout
from langchain_core.documents import Document

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


def test_new_file_is_added(fake_env):
    data_dir, store = fake_env
    _write(data_dir, "a.txt", "これは十分な長さのテキストです。" * 5)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": ["a.txt"], "updated": [], "removed": [], "failed": []}
    assert store.docs_by_id


def test_unchanged_file_is_noop_on_second_sync(fake_env):
    data_dir, store = fake_env
    _write(data_dir, "a.txt", "変わらないテキスト内容です。")
    ingest.sync_data_dir(verbose=False)
    docs_after_first_sync = dict(store.docs_by_id)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": [], "failed": []}
    assert store.docs_by_id == docs_after_first_sync


def test_modified_file_is_updated_and_old_chunks_replaced(fake_env):
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "元のテキスト内容です。")
    ingest.sync_data_dir(verbose=False)
    old_chunk_ids = set(store.docs_by_id.keys())

    path.write_text("変更後のまったく別のテキスト内容です。", encoding="utf-8")
    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": ["a.txt"], "removed": [], "failed": []}
    # 古いチャンクは削除され、新しいチャンクに置き換わっている
    assert old_chunk_ids.isdisjoint(store.docs_by_id.keys())
    assert store.docs_by_id


def test_removed_file_is_deleted_from_store_and_manifest(fake_env):
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "削除されるファイルの内容です。")
    ingest.sync_data_dir(verbose=False)
    assert store.docs_by_id

    path.unlink()
    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": ["a.txt"], "failed": []}
    assert store.docs_by_id == {}
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest == {}


class _FailingLoader:
    """load() で必ず例外を送出するダミーローダー（LOADERSの差し替え用）。

    1ファイルの読み込み失敗が他のファイルの同期まで止めてしまわないことを
    検証するためのテストダブル。
    """

    def __init__(self, path):
        self.path = path

    def load(self):
        raise ValueError(f"{self.path} は読み込めません（壊れたファイルの想定）")


def test_one_file_load_failure_does_not_block_other_files(fake_env, monkeypatch):
    # 1ファイルの読み込みが失敗しても、他の正常なファイルは問題なく同期される
    data_dir, store = fake_env
    _write(data_dir, "good.txt", "正常に読み込めるテキストです。" * 5)
    _write(data_dir, "bad.txt", "壊れていることにするテキストです。" * 5)

    real_text_loader = ingest.LOADERS[".txt"]

    def flaky_loader(path):
        if "bad.txt" in path:
            return _FailingLoader(path)
        return real_text_loader(path)

    monkeypatch.setattr(ingest, "LOADERS", {**ingest.LOADERS, ".txt": flaky_loader})

    result = ingest.sync_data_dir(verbose=False)

    assert result == {
        "added": ["good.txt"],
        "updated": [],
        "removed": [],
        "failed": ["bad.txt"],
    }
    # 正常なファイルのチャンクはベクトルストアに登録されている
    sources = {doc.metadata.get("source") for doc in store.docs_by_id.values()}
    assert any("good.txt" in (s or "") for s in sources)


def test_pdf_load_failure_is_recorded_as_failed(fake_env, monkeypatch):
    # PDFは _load_pdf() 経由で読み込まれるため、そちらの失敗も同様にスキップされることを確認する
    data_dir, store = fake_env
    (data_dir / "broken.pdf").write_bytes(b"not a real pdf")

    def flaky_load_pdf(path, verbose=True):
        raise ValueError("壊れたPDFです")

    monkeypatch.setattr(ingest, "_load_pdf", flaky_load_pdf)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": [], "failed": ["broken.pdf"]}
    assert store.docs_by_id == {}


def test_load_pdf_falls_back_to_docling_when_pymupdf_raises(monkeypatch, tmp_path):
    # PyMuPDF自体が例外を送出した場合（暗号化PDF・破損PDF等）でも、
    # Doclingが利用可能ならフォールバックして読み込めることを検証する
    fake_pdf_path = tmp_path / "encrypted.pdf"
    fake_pdf_path.write_bytes(b"not a real pdf")

    class _FailingPyMuPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            raise ValueError("暗号化されたPDFは読み込めません")

    class _SucceedingDoclingLoader:
        def __init__(self, file_path, export_type):
            self.file_path = file_path

        def load(self):
            return [Document(page_content="Doclingで抽出した本文です。")]

    monkeypatch.setattr(ingest, "DOCLING_AVAILABLE", True)
    monkeypatch.setattr(ingest, "PyMuPDFLoader", _FailingPyMuPDFLoader)
    monkeypatch.setattr(ingest, "DoclingLoader", _SucceedingDoclingLoader)

    docs = ingest._load_pdf(fake_pdf_path, verbose=False)

    assert len(docs) == 1
    assert docs[0].page_content == "Doclingで抽出した本文です。"
    assert docs[0].metadata["source"] == str(fake_pdf_path)


def test_load_pdf_docling_fallback_on_pymupdf_error_respects_verbose_false(monkeypatch, tmp_path, capsys):
    # PyMuPDF例外時のDoclingフォールバック経路でも、verbose=Falseが
    # _load_pdf_with_docling() まで正しく伝播し、標準出力に何も出力されないことを検証する
    # （sync_data_dir(verbose=False)を使う app.py / api/main.py の本番導線での退行防止）
    fake_pdf_path = tmp_path / "encrypted.pdf"
    fake_pdf_path.write_bytes(b"not a real pdf")

    class _FailingPyMuPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            raise ValueError("暗号化されたPDFは読み込めません")

    class _SucceedingDoclingLoader:
        def __init__(self, file_path, export_type):
            self.file_path = file_path

        def load(self):
            return [Document(page_content="Doclingで抽出した本文です。")]

    monkeypatch.setattr(ingest, "DOCLING_AVAILABLE", True)
    monkeypatch.setattr(ingest, "PyMuPDFLoader", _FailingPyMuPDFLoader)
    monkeypatch.setattr(ingest, "DoclingLoader", _SucceedingDoclingLoader)

    ingest._load_pdf(fake_pdf_path, verbose=False)

    assert capsys.readouterr().out == ""


def test_load_pdf_raises_original_error_when_docling_also_fails(monkeypatch, tmp_path):
    # PyMuPDFが失敗し、Doclingも失敗（または未インストール）した場合は
    # 元の例外がそのまま伝播し、呼び出し元で "failed" として扱われリトライ対象になる
    fake_pdf_path = tmp_path / "broken.pdf"
    fake_pdf_path.write_bytes(b"not a real pdf")

    class _FailingPyMuPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            raise ValueError("破損したPDFです")

    class _FailingDoclingLoader:
        def __init__(self, file_path, export_type):
            self.file_path = file_path

        def load(self):
            raise RuntimeError("Doclingでも読み込めません")

    monkeypatch.setattr(ingest, "DOCLING_AVAILABLE", True)
    monkeypatch.setattr(ingest, "PyMuPDFLoader", _FailingPyMuPDFLoader)
    monkeypatch.setattr(ingest, "DoclingLoader", _FailingDoclingLoader)

    with pytest.raises(ValueError, match="破損したPDFです"):
        ingest._load_pdf(fake_pdf_path, verbose=False)


def test_load_pdf_raises_original_error_when_docling_not_installed(monkeypatch, tmp_path):
    # PyMuPDFが例外を送出し、かつDoclingが未インストール（DOCLING_AVAILABLE=False）の場合は
    # Doclingへのフォールバックを試みることなく、元の例外がそのまま伝播すること
    fake_pdf_path = tmp_path / "broken.pdf"
    fake_pdf_path.write_bytes(b"not a real pdf")

    class _FailingPyMuPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            raise ValueError("暗号化されたPDFは読み込めません")

    class _DoclingLoaderThatShouldNotBeCalled:
        def __init__(self, file_path, export_type):
            raise AssertionError("Docling未インストール時はDoclingLoaderが呼ばれてはならない")

    monkeypatch.setattr(ingest, "DOCLING_AVAILABLE", False)
    monkeypatch.setattr(ingest, "PyMuPDFLoader", _FailingPyMuPDFLoader)
    monkeypatch.setattr(ingest, "DoclingLoader", _DoclingLoaderThatShouldNotBeCalled)

    with pytest.raises(ValueError, match="暗号化されたPDFは読み込めません"):
        ingest._load_pdf(fake_pdf_path, verbose=False)


def test_load_pdf_falls_back_to_docling_when_pymupdf_text_is_too_sparse(monkeypatch, tmp_path):
    # 既存の「文字数不足時のフォールバック」機能のリグレッション確認。
    # PyMuPDFLoader自体は例外を送出しないが、抽出できた文字数が閾値未満（図解・スキャンPDFの疑い）の場合、
    # Doclingの抽出結果の方が多ければそちらを採用する
    fake_pdf_path = tmp_path / "scanned.pdf"
    fake_pdf_path.write_bytes(b"not a real pdf")

    class _SparsePyMuPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [Document(page_content="図")]  # MIN_CHARS_PER_PAGE_FOR_FAST_PATH未満

    class _SucceedingDoclingLoader:
        def __init__(self, file_path, export_type):
            self.file_path = file_path

        def load(self):
            return [Document(page_content="Doclingで抽出したより多くの本文です。" * 5)]

    monkeypatch.setattr(ingest, "DOCLING_AVAILABLE", True)
    monkeypatch.setattr(ingest, "PyMuPDFLoader", _SparsePyMuPDFLoader)
    monkeypatch.setattr(ingest, "DoclingLoader", _SucceedingDoclingLoader)

    docs = ingest._load_pdf(fake_pdf_path, verbose=False)

    assert len(docs) == 1
    assert docs[0].page_content.startswith("Doclingで抽出したより多くの本文です。")
    assert docs[0].metadata["source"] == str(fake_pdf_path)


def test_load_pdf_keeps_pymupdf_result_when_docling_extracts_fewer_chars(monkeypatch, tmp_path):
    # 文字数不足でDoclingを試みても、抽出できた文字数がPyMuPDFの結果を上回らない場合は
    # PyMuPDFの結果（fast_docs）をそのまま使う（既存挙動のリグレッション確認）
    fake_pdf_path = tmp_path / "scanned.pdf"
    fake_pdf_path.write_bytes(b"not a real pdf")

    class _SparsePyMuPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [Document(page_content="図")]

    class _WorseDoclingLoader:
        def __init__(self, file_path, export_type):
            self.file_path = file_path

        def load(self):
            return [Document(page_content="")]

    monkeypatch.setattr(ingest, "DOCLING_AVAILABLE", True)
    monkeypatch.setattr(ingest, "PyMuPDFLoader", _SparsePyMuPDFLoader)
    monkeypatch.setattr(ingest, "DoclingLoader", _WorseDoclingLoader)

    docs = ingest._load_pdf(fake_pdf_path, verbose=False)

    assert len(docs) == 1
    assert docs[0].page_content == "図"


def test_load_pdf_keeps_pymupdf_result_when_docling_fails_for_sparse_text(monkeypatch, tmp_path):
    # 文字数不足でDoclingにフォールバックしたが、Docling自体が例外を送出した場合は
    # （PyMuPDF自体は成功しているので）例外を伝播させず、PyMuPDFの結果を使う（既存挙動のリグレッション確認）
    fake_pdf_path = tmp_path / "scanned.pdf"
    fake_pdf_path.write_bytes(b"not a real pdf")

    class _SparsePyMuPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [Document(page_content="図")]

    class _FailingDoclingLoader:
        def __init__(self, file_path, export_type):
            self.file_path = file_path

        def load(self):
            raise RuntimeError("Doclingでも読み込めません")

    monkeypatch.setattr(ingest, "DOCLING_AVAILABLE", True)
    monkeypatch.setattr(ingest, "PyMuPDFLoader", _SparsePyMuPDFLoader)
    monkeypatch.setattr(ingest, "DoclingLoader", _FailingDoclingLoader)

    docs = ingest._load_pdf(fake_pdf_path, verbose=False)

    assert len(docs) == 1
    assert docs[0].page_content == "図"


def test_load_pdf_uses_pymupdf_directly_when_text_is_sufficient(monkeypatch, tmp_path):
    # 抽出文字数が閾値以上であれば、Doclingには一切フォールバックせずPyMuPDFの結果をそのまま使う
    # （既存挙動のリグレッション確認。DoclingLoaderが呼ばれないことも合わせて検証する）
    fake_pdf_path = tmp_path / "normal.pdf"
    fake_pdf_path.write_bytes(b"not a real pdf")

    class _SufficientPyMuPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [Document(page_content="十分な量のテキストです。" * 10)]

    class _DoclingLoaderThatShouldNotBeCalled:
        def __init__(self, file_path, export_type):
            raise AssertionError("文字数が十分な場合はDoclingLoaderが呼ばれてはならない")

    monkeypatch.setattr(ingest, "DOCLING_AVAILABLE", True)
    monkeypatch.setattr(ingest, "PyMuPDFLoader", _SufficientPyMuPDFLoader)
    monkeypatch.setattr(ingest, "DoclingLoader", _DoclingLoaderThatShouldNotBeCalled)

    docs = ingest._load_pdf(fake_pdf_path, verbose=False)

    assert len(docs) == 1
    assert docs[0].page_content == "十分な量のテキストです。" * 10


def test_failed_file_is_not_recorded_in_manifest_and_retried_next_sync(fake_env, monkeypatch):
    # 失敗したファイルはmanifestに記録されず、次回同期時に再度読み込みが試みられる（リトライ）
    data_dir, store = fake_env
    _write(data_dir, "bad.txt", "壊れていることにするテキストです。" * 5)

    load_attempts = []

    def always_failing_loader(path):
        load_attempts.append(path)
        return _FailingLoader(path)

    monkeypatch.setattr(ingest, "LOADERS", {**ingest.LOADERS, ".txt": always_failing_loader})

    result_1 = ingest.sync_data_dir(verbose=False)
    assert result_1["failed"] == ["bad.txt"]

    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert "bad.txt" not in manifest

    result_2 = ingest.sync_data_dir(verbose=False)
    assert result_2["failed"] == ["bad.txt"]
    # ファイルが変化していなくてもmanifestに記録がないため、2回とも読み込みが試みられている
    assert len(load_attempts) == 2


def test_all_files_failing_completes_without_raising(fake_env, monkeypatch):
    # 全ファイルが失敗しても例外を送出せず処理が正常に完了する（境界値）
    data_dir, store = fake_env
    _write(data_dir, "bad1.txt", "壊れていることにするテキスト1です。" * 5)
    _write(data_dir, "bad2.txt", "壊れていることにするテキスト2です。" * 5)

    monkeypatch.setattr(ingest, "LOADERS", {**ingest.LOADERS, ".txt": lambda path: _FailingLoader(path)})

    result = ingest.sync_data_dir(verbose=False)

    assert sorted(result["failed"]) == ["bad1.txt", "bad2.txt"]
    assert result["added"] == []
    assert result["updated"] == []
    assert result["removed"] == []
    assert store.docs_by_id == {}


def test_previously_synced_file_that_now_fails_keeps_old_chunks(fake_env, monkeypatch):
    # 読み込み失敗時は前回同期成功時点のインデックスがそのまま残る
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "最初は正常なテキストです。" * 5)
    ingest.sync_data_dir(verbose=False)
    old_chunk_ids = set(store.docs_by_id.keys())
    assert old_chunk_ids

    path.write_text("2回目の同期時は壊れていることにするテキストです。" * 5, encoding="utf-8")
    monkeypatch.setattr(ingest, "LOADERS", {**ingest.LOADERS, ".txt": lambda p: _FailingLoader(p)})

    result = ingest.sync_data_dir(verbose=False)

    assert result["failed"] == ["a.txt"]
    assert result["updated"] == []
    # 読み込み失敗時は前回同期成功時点のチャンクがそのまま（削除も置き換えもされない）
    assert store.docs_by_id.keys() == old_chunk_ids


def test_add_documents_failure_on_update_keeps_old_chunks_and_records_failed(fake_env, monkeypatch):
    # 更新時にadd_documents()が失敗しても、旧チャンクを消さず
    # result["failed"]に積んで次回リトライされるようにする（データが消える事故の防止）
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "最初は正常なテキストです。" * 5)
    ingest.sync_data_dir(verbose=False)
    old_chunk_ids = set(store.docs_by_id.keys())
    assert old_chunk_ids

    path.write_text("更新後のテキストです。" * 5, encoding="utf-8")

    def flaky_add_documents(documents):
        raise RuntimeError("埋め込みモデルの一時的な失敗を想定")

    monkeypatch.setattr(store, "add_documents", flaky_add_documents)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": [], "failed": ["a.txt"]}
    # 追加が失敗しても旧チャンクは削除されず残っている（検索対象から消えない）
    assert store.docs_by_id.keys() == old_chunk_ids

    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    # manifestは更新されない（旧chunk_idsのまま）ので、次回同期時も再試行される
    assert set(manifest["a.txt"]["chunk_ids"]) == old_chunk_ids


def test_delete_failure_after_add_success_defers_old_chunk_removal(fake_env, monkeypatch):
    # add_documents()は成功したがvector_store.delete()自体が失敗した場合、
    # 新チャンクは登録済みのまま、旧チャンクの削除はpending_delete_chunk_idsとして
    # manifestに持ち越され、次回以降に再試行されるようにする
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "最初は正常なテキストです。" * 5)
    ingest.sync_data_dir(verbose=False)
    old_chunk_ids = set(store.docs_by_id.keys())
    assert old_chunk_ids

    path.write_text("更新後のテキストです。" * 5, encoding="utf-8")

    def flaky_delete(ids):
        raise RuntimeError("ベクトルストアの一時的な削除失敗を想定")

    monkeypatch.setattr(store, "delete", flaky_delete)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": ["a.txt"], "removed": [], "failed": []}
    # 新チャンクは追加済みで、削除に失敗した旧チャンクもまだ残っている（重複状態）
    assert old_chunk_ids.issubset(store.docs_by_id.keys())
    assert len(store.docs_by_id) > len(old_chunk_ids)

    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert set(manifest["a.txt"]["pending_delete_chunk_ids"]) == old_chunk_ids


def test_pending_delete_is_retried_and_cleared_on_next_unchanged_sync(fake_env, monkeypatch):
    # 前回同期でdelete()に失敗し持ち越しになった旧チャンクは、ファイル内容に
    # 変化がない（unchanged判定の）次回同期時に再試行され、成功すれば
    # pending_delete_chunk_idsがmanifestから消え、重複が解消される
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "最初は正常なテキストです。" * 5)
    ingest.sync_data_dir(verbose=False)
    old_chunk_ids = set(store.docs_by_id.keys())

    path.write_text("更新後のテキストです。" * 5, encoding="utf-8")

    real_delete = store.delete

    def flaky_delete(ids):
        raise RuntimeError("ベクトルストアの一時的な削除失敗を想定")

    monkeypatch.setattr(store, "delete", flaky_delete)
    ingest.sync_data_dir(verbose=False)
    assert old_chunk_ids.issubset(store.docs_by_id.keys())

    monkeypatch.setattr(store, "delete", real_delete)
    result = ingest.sync_data_dir(verbose=False)

    # ファイル内容自体は変化していないため、added/updated/removedいずれにも計上されない
    assert result == {"added": [], "updated": [], "removed": [], "failed": []}
    # 保留されていた旧チャンクの削除が完了し、重複が解消されている
    assert old_chunk_ids.isdisjoint(store.docs_by_id.keys())

    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert "pending_delete_chunk_ids" not in manifest["a.txt"]


def test_delete_failure_after_add_success_logs_warning_via_logger(fake_env, monkeypatch, caplog):
    # add_documents()成功後のdelete()失敗もlogger.warning()で記録される
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "最初は正常なテキストです。" * 5)
    ingest.sync_data_dir(verbose=False)

    path.write_text("更新後のテキストです。" * 5, encoding="utf-8")

    def flaky_delete(ids):
        raise RuntimeError("ベクトルストアの一時的な削除失敗を想定")

    monkeypatch.setattr(store, "delete", flaky_delete)

    with caplog.at_level(logging.WARNING, logger="ingest"):
        ingest.sync_data_dir(verbose=False)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "a.txt" in warnings[0].getMessage()
    assert "旧チャンク削除に失敗" in warnings[0].getMessage()


def test_pending_delete_retry_failure_again_keeps_pending_ids(fake_env, monkeypatch):
    # 境界値: 保留中の旧チャンク削除を再試行してもそれ自体が再び失敗した場合、
    # pending_delete_chunk_idsはmanifestから失われず維持され、以降も再試行対象であり続ける
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "最初は正常なテキストです。" * 5)
    ingest.sync_data_dir(verbose=False)
    old_chunk_ids = set(store.docs_by_id.keys())

    path.write_text("更新後のテキストです。" * 5, encoding="utf-8")

    def flaky_delete(ids):
        raise RuntimeError("ベクトルストアの一時的な削除失敗を想定")

    monkeypatch.setattr(store, "delete", flaky_delete)
    ingest.sync_data_dir(verbose=False)
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert set(manifest["a.txt"]["pending_delete_chunk_ids"]) == old_chunk_ids

    # unchanged判定での再試行も再び失敗させる（ファイル内容は変えない）
    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": [], "failed": []}
    # 再試行が失敗してもpending_delete_chunk_idsは失われず、旧チャンクも残ったまま
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert set(manifest["a.txt"]["pending_delete_chunk_ids"]) == old_chunk_ids
    assert old_chunk_ids.issubset(store.docs_by_id.keys())


def test_pending_delete_chunk_ids_are_removed_when_file_is_deleted(fake_env, monkeypatch):
    # 境界値: pending_delete_chunk_idsが残っている状態でファイル自体がdata/から
    # 削除された場合。「削除されたファイル」処理はmanifestエントリを削除するだけで
    # pending_delete_chunk_ids（旧チャンク）をvector_store.delete()していないため、
    # 現状の実装ではこの旧チャンクがベクトルストアに永久に取り残されてしまう
    # （manifestからエントリ自体が消えるため、以後の同期でも二度と再試行されない）。
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "最初は正常なテキストです。" * 5)
    ingest.sync_data_dir(verbose=False)
    old_chunk_ids = set(store.docs_by_id.keys())

    path.write_text("更新後のテキストです。" * 5, encoding="utf-8")

    def flaky_delete(ids):
        raise RuntimeError("ベクトルストアの一時的な削除失敗を想定")

    real_delete = store.delete
    monkeypatch.setattr(store, "delete", flaky_delete)
    ingest.sync_data_dir(verbose=False)
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert set(manifest["a.txt"]["pending_delete_chunk_ids"]) == old_chunk_ids
    new_chunk_ids = set(manifest["a.txt"]["chunk_ids"])

    # deleteを復旧させた上でファイル自体を削除する
    monkeypatch.setattr(store, "delete", real_delete)
    path.unlink()
    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": ["a.txt"], "failed": []}
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert "a.txt" not in manifest
    # 新チャンクはremoved処理で削除される
    assert new_chunk_ids.isdisjoint(store.docs_by_id.keys())
    # 保留されていた旧チャンクも削除され、ベクトルストアに取り残されないべき
    assert old_chunk_ids.isdisjoint(store.docs_by_id.keys())


def test_pending_delete_chunk_ids_not_lost_when_file_updated_again_while_pending(fake_env, monkeypatch):
    # 境界値: pending_delete_chunk_idsが残っている状態でファイルがさらに再更新された場合。
    # 更新処理はmanifestエントリを{**fingerprint, "chunk_ids": ...}で丸ごと置き換えるため、
    # 現状の実装では前回持ち越されていたpending_delete_chunk_ids（v1のチャンク）が
    # 新しいmanifestエントリから消え、以後二度と削除が試みられずベクトルストアに
    # 取り残されてしまう（v2のチャンクは今回の更新でdelete対象になり削除される）。
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "バージョン1のテキストです。" * 5)
    ingest.sync_data_dir(verbose=False)
    v1_chunk_ids = set(store.docs_by_id.keys())

    path.write_text("バージョン2のテキストです。" * 5, encoding="utf-8")

    def flaky_delete(ids):
        raise RuntimeError("ベクトルストアの一時的な削除失敗を想定")

    real_delete = store.delete
    monkeypatch.setattr(store, "delete", flaky_delete)
    ingest.sync_data_dir(verbose=False)
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert set(manifest["a.txt"]["pending_delete_chunk_ids"]) == v1_chunk_ids
    v2_chunk_ids = set(manifest["a.txt"]["chunk_ids"])

    # deleteを復旧させた上でファイルをさらに更新する（v1の削除がまだ再試行されていない状態で）
    monkeypatch.setattr(store, "delete", real_delete)
    path.write_text("バージョン3のテキストです。" * 5, encoding="utf-8")
    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": ["a.txt"], "removed": [], "failed": []}
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    # v2のチャンクは今回の更新で正常に削除される
    assert v2_chunk_ids.isdisjoint(store.docs_by_id.keys())
    # v1のチャンクもpending_delete_chunk_idsとして引き継がれ、いずれ削除されるべき
    assert v1_chunk_ids.isdisjoint(store.docs_by_id.keys()) or "pending_delete_chunk_ids" in manifest["a.txt"]


def test_delete_failure_when_file_removed_defers_removal_and_retries(fake_env, monkeypatch):
    # 境界値: 「data/ から削除されたファイルをDBからも削除」ループで
    # vector_store.delete()自体が失敗した場合（chunk_idsのみ、pending_delete_chunk_idsは
    # 無い状態）、例外は握りつぶされてresult["removed"]には計上されず、manifestエントリは
    # {"pending_delete_chunk_ids": [...]}の形で維持され、次回同期時に再試行されるべき
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "削除されるファイルの内容です。" * 5)
    ingest.sync_data_dir(verbose=False)
    chunk_ids = set(store.docs_by_id.keys())
    assert chunk_ids

    real_delete = store.delete

    def flaky_delete(ids):
        raise RuntimeError("ベクトルストアの一時的な削除失敗を想定")

    monkeypatch.setattr(store, "delete", flaky_delete)
    path.unlink()

    result = ingest.sync_data_dir(verbose=False)

    # delete()失敗時は例外が握りつぶされ、removedには計上されない
    assert result == {"added": [], "updated": [], "removed": [], "failed": []}
    # チャンクは削除に失敗しているため、まだベクトルストアに残っている
    assert chunk_ids.issubset(store.docs_by_id.keys())

    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    # manifestエントリはpending_delete_chunk_idsのみを持つ形で維持される
    assert manifest["a.txt"] == {"pending_delete_chunk_ids": sorted(chunk_ids)}

    # deleteを復旧させて再度同期すると、削除が再試行されmanifestからエントリごと消える
    monkeypatch.setattr(store, "delete", real_delete)
    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": ["a.txt"], "failed": []}
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert "a.txt" not in manifest
    assert chunk_ids.isdisjoint(store.docs_by_id.keys())


def test_delete_failure_when_file_removed_logs_warning_via_logger(fake_env, monkeypatch, caplog):
    # 削除検知時点でのdelete()失敗もlogger.warning()で記録される
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "削除されるファイルの内容です。" * 5)
    ingest.sync_data_dir(verbose=False)

    def flaky_delete(ids):
        raise RuntimeError("ベクトルストアの一時的な削除失敗を想定")

    monkeypatch.setattr(store, "delete", flaky_delete)
    path.unlink()

    with caplog.at_level(logging.WARNING, logger="ingest"):
        ingest.sync_data_dir(verbose=False)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "a.txt" in warnings[0].getMessage()
    assert "削除処理中に旧チャンク削除に失敗" in warnings[0].getMessage()


def test_delete_failure_when_file_removed_merges_pending_ids_and_retries(fake_env, monkeypatch):
    # 境界値: 更新時のdelete失敗で持ち越されたpending_delete_chunk_idsがある状態で
    # ファイルが削除され、かつファイル削除時点のdelete()（chunk_ids + pending_delete_chunk_ids
    # の合算分）自体も失敗するケース。合算後のIDがpending_delete_chunk_idsとしてそのまま
    # 維持され、削除も再試行できることを確認する
    data_dir, store = fake_env
    path = _write(data_dir, "a.txt", "バージョン1のテキストです。" * 5)
    ingest.sync_data_dir(verbose=False)
    v1_chunk_ids = set(store.docs_by_id.keys())

    path.write_text("バージョン2のテキストです。" * 5, encoding="utf-8")

    real_delete = store.delete

    def flaky_delete(ids):
        raise RuntimeError("ベクトルストアの一時的な削除失敗を想定")

    monkeypatch.setattr(store, "delete", flaky_delete)
    ingest.sync_data_dir(verbose=False)
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert set(manifest["a.txt"]["pending_delete_chunk_ids"]) == v1_chunk_ids
    v2_chunk_ids = set(manifest["a.txt"]["chunk_ids"])
    all_chunk_ids = v1_chunk_ids | v2_chunk_ids

    # deleteを失敗させたままファイル自体を削除する
    path.unlink()
    result = ingest.sync_data_dir(verbose=False)

    # ファイル削除時のdelete()（v1+v2の合算分）も失敗するため、removedには計上されない
    assert result == {"added": [], "updated": [], "removed": [], "failed": []}
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["a.txt"] == {"pending_delete_chunk_ids": sorted(all_chunk_ids)}
    assert all_chunk_ids.issubset(store.docs_by_id.keys())

    # deleteを復旧させて再度同期すると、合算分がまとめて削除されmanifestからエントリごと消える
    monkeypatch.setattr(store, "delete", real_delete)
    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": ["a.txt"], "failed": []}
    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert "a.txt" not in manifest
    assert all_chunk_ids.isdisjoint(store.docs_by_id.keys())


def test_add_documents_failure_on_new_file_has_no_old_chunks_to_keep(fake_env, monkeypatch):
    # 境界値: 新規ファイル追加時（entryが存在せず旧チャンクが無いケース）にadd_documents()が
    # 失敗しても、delete()が呼ばれたり例外で落ちたりせず、failedに記録されるだけで正常終了すること
    data_dir, store = fake_env
    _write(data_dir, "new.txt", "新規追加されるファイルの内容です。" * 5)

    def flaky_add_documents(documents):
        raise RuntimeError("埋め込みモデルの一時的な失敗を想定")

    monkeypatch.setattr(store, "add_documents", flaky_add_documents)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": [], "failed": ["new.txt"]}
    # そもそも旧チャンクが無いので、ベクトルストアには何も登録されないまま
    assert store.docs_by_id == {}

    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    # 失敗したファイルはmanifestに記録されないため、次回同期時に再試行される
    assert "new.txt" not in manifest


def test_add_documents_failure_for_one_file_does_not_block_other_files(fake_env, monkeypatch):
    # 複数ファイル同期中に1ファイルのadd_documents()だけが失敗しても、
    # 他のファイルの同期処理は中断されず継続すること
    data_dir, store = fake_env
    _write(data_dir, "good.txt", "正常に追加できるテキストです。" * 5)
    _write(data_dir, "bad.txt", "追加に失敗することにするテキストです。" * 5)

    real_add_documents = store.add_documents

    def flaky_add_documents(documents):
        sources = {doc.metadata.get("source", "") for doc in documents}
        if any("bad.txt" in s for s in sources):
            raise RuntimeError("埋め込みモデルの一時的な失敗を想定")
        return real_add_documents(documents)

    monkeypatch.setattr(store, "add_documents", flaky_add_documents)

    result = ingest.sync_data_dir(verbose=False)

    assert result == {
        "added": ["good.txt"],
        "updated": [],
        "removed": [],
        "failed": ["bad.txt"],
    }
    # 失敗したファイルの分は登録されていないが、正常なファイルのチャンクは登録されている
    sources = {doc.metadata.get("source") for doc in store.docs_by_id.values()}
    assert any("good.txt" in (s or "") for s in sources)
    assert not any("bad.txt" in (s or "") for s in sources)

    manifest = json.loads(ingest.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert "good.txt" in manifest
    assert "bad.txt" not in manifest


def test_unsupported_extension_is_ignored(fake_env):
    data_dir, store = fake_env
    _write(data_dir, "notes.docx", "対応していない拡張子")

    result = ingest.sync_data_dir(verbose=False)

    assert result == {"added": [], "updated": [], "removed": [], "failed": []}
    assert store.docs_by_id == {}


def test_top_level_file_is_tagged_with_global_thread_id(fake_env):
    data_dir, store = fake_env
    _write(data_dir, "doc.txt", "共通ナレッジとして扱われるべき文書です。" * 5)

    ingest.sync_data_dir(verbose=False)

    thread_ids = {doc.metadata["thread_id"] for doc in store.docs_by_id.values()}
    assert thread_ids == {ingest.GLOBAL_THREAD_ID}


def test_conversation_log_is_tagged_with_its_own_thread_id(fake_env):
    data_dir, store = fake_env
    _write(
        data_dir,
        "conversations/thread-xyz/log.md",
        "このスレッド専用の会話ログです。" * 5,
    )

    ingest.sync_data_dir(verbose=False)

    thread_ids = {doc.metadata["thread_id"] for doc in store.docs_by_id.values()}
    assert thread_ids == {"thread-xyz"}


def test_legacy_flat_conversation_file_falls_back_to_global_thread_id(fake_env):
    # スレッド機能追加前の古い保存形式（サブフォルダなし）は後方互換で
    # GLOBAL_THREAD_ID として扱われる想定
    data_dir, store = fake_env
    _write(
        data_dir,
        "conversations/legacy.md",
        "サブフォルダのない古い形式の会話ログです。" * 5,
    )

    ingest.sync_data_dir(verbose=False)

    thread_ids = {doc.metadata["thread_id"] for doc in store.docs_by_id.values()}
    assert thread_ids == {ingest.GLOBAL_THREAD_ID}


# --- _is_fallback_conversation() / is_fallbackメタデータ ---


class _FakeRawDoc:
    """分割前の生ドキュメントのフェイク（page_contentのみ使う）。"""

    def __init__(self, page_content):
        self.page_content = page_content


def test_is_fallback_conversation_true_when_metadata_line_present():
    docs = [
        _FakeRawDoc(
            "# 会話ログ\n\n- 日時: 2026-01-01 00:00:00\n- 一般知識フォールバック: true\n\n"
            "## 質問\n\n質問です\n\n## 回答\n\n回答です\n"
        )
    ]
    assert ingest._is_fallback_conversation(docs) is True


def test_is_fallback_conversation_false_when_metadata_line_says_false():
    docs = [
        _FakeRawDoc(
            "# 会話ログ\n\n- 日時: 2026-01-01 00:00:00\n- 一般知識フォールバック: false\n\n"
            "## 質問\n\n質問です\n\n## 回答\n\n回答です\n"
        )
    ]
    assert ingest._is_fallback_conversation(docs) is False


def test_is_fallback_conversation_false_for_normal_document_without_metadata_line():
    docs = [_FakeRawDoc("これは通常のドキュメントです。会話ログのメタデータ行を含みません。")]
    assert ingest._is_fallback_conversation(docs) is False


def test_is_fallback_conversation_false_for_empty_doc_list():
    # 境界値: 空リストの場合はany()がFalseを返すため False
    assert ingest._is_fallback_conversation([]) is False


def test_is_fallback_conversation_true_if_any_doc_in_list_matches():
    # 複数ページ（PDF等）にまたがるドキュメントで、1ページでもメタデータ行があればTrue
    docs = [
        _FakeRawDoc("通常の内容"),
        _FakeRawDoc("- 一般知識フォールバック: true"),
    ]
    assert ingest._is_fallback_conversation(docs) is True


def test_sync_data_dir_tags_normal_document_as_not_fallback(fake_env):
    data_dir, store = fake_env
    _write(data_dir, "doc.txt", "通常のドキュメント内容です。" * 5)

    ingest.sync_data_dir(verbose=False)

    is_fallback_values = {doc.metadata["is_fallback"] for doc in store.docs_by_id.values()}
    assert is_fallback_values == {False}


def test_sync_data_dir_tags_fallback_conversation_log_chunks_as_fallback(fake_env):
    data_dir, store = fake_env
    content = (
        "# 会話ログ\n\n"
        "- 日時: 2026-01-01 00:00:00\n"
        "- 一般知識フォールバック: true\n\n"
        "## 質問\n\n一般知識で答えた質問です。\n\n## 回答\n\n一般知識による回答です。\n"
    )
    _write(data_dir, "conversations/thread-x/log.md", content)

    ingest.sync_data_dir(verbose=False)

    is_fallback_values = {doc.metadata["is_fallback"] for doc in store.docs_by_id.values()}
    assert is_fallback_values == {True}


def test_sync_data_dir_tags_non_fallback_conversation_log_chunks_as_not_fallback(fake_env):
    data_dir, store = fake_env
    content = (
        "# 会話ログ\n\n"
        "- 日時: 2026-01-01 00:00:00\n"
        "- 一般知識フォールバック: false\n\n"
        "## 質問\n\n文書に根拠のある質問です。\n\n## 回答\n\n文書に基づく回答です。\n"
    )
    _write(data_dir, "conversations/thread-y/log.md", content)

    ingest.sync_data_dir(verbose=False)

    is_fallback_values = {doc.metadata["is_fallback"] for doc in store.docs_by_id.values()}
    assert is_fallback_values == {False}


@pytest.mark.parametrize(
    ("filename", "expected_name"),
    [
        ("../../etc/passwd_test", "passwd_test"),
        ("../../../tmp/evil.txt", "evil.txt"),
        ("../sibling_dir_escape.txt", "sibling_dir_escape.txt"),
    ],
)
def test_safe_upload_dest_strips_directory_traversal(fake_env, filename, expected_name):
    # ディレクトリ部分（../ 等）は無害化され、DATA_DIR配下のベース名のみのパスになる
    data_dir, _store = fake_env

    dest = ingest.safe_upload_dest(filename)

    assert dest is not None
    assert dest.parent == data_dir.resolve()
    assert dest.name == expected_name


def test_safe_upload_dest_rejects_bare_dotdot(fake_env):
    # ".." そのものは無害化してもDATA_DIR自身/外を指してしまうため拒否する
    data_dir, _store = fake_env

    dest = ingest.safe_upload_dest("..")

    assert dest is None


@pytest.mark.parametrize(
    "filename",
    ["report.pdf", "notes.txt", "my file (2).md"],
)
def test_safe_upload_dest_accepts_normal_filenames(fake_env, filename):
    data_dir, _store = fake_env

    dest = ingest.safe_upload_dest(filename)

    assert dest is not None
    assert dest.parent == data_dir.resolve()
    assert dest.name == filename


# --- resolve_upload_dest(): 同名ファイルアップロード時の無警告上書き防止 ---


def test_resolve_upload_dest_returns_plain_path_when_no_conflict(fake_env):
    data_dir, _store = fake_env

    dest = ingest.resolve_upload_dest("report.pdf")

    assert dest == data_dir.resolve() / "report.pdf"


def test_resolve_upload_dest_appends_suffix_when_file_already_exists_on_disk(fake_env):
    data_dir, _store = fake_env
    _write(data_dir, "report.txt", "既存ファイルの内容です。")

    dest = ingest.resolve_upload_dest("report.txt")

    assert dest.name == "report (2).txt"
    assert not dest.exists()


def test_resolve_upload_dest_increments_suffix_until_free(fake_env):
    data_dir, _store = fake_env
    _write(data_dir, "report.txt", "1つ目")
    _write(data_dir, "report (2).txt", "2つ目")

    dest = ingest.resolve_upload_dest("report.txt")

    assert dest.name == "report (3).txt"


def test_resolve_upload_dest_avoids_duplicate_within_same_batch(fake_env):
    # data/上にはまだ存在しなくても、同一アップロードバッチ内で
    # 既に使用済みのパス（taken_paths）とは衝突しないようにする
    data_dir, _store = fake_env
    first = ingest.resolve_upload_dest("report.txt")
    taken = {first}

    second = ingest.resolve_upload_dest("report.txt", taken_paths=taken)

    assert first.name == "report.txt"
    assert second.name == "report (2).txt"


def test_resolve_upload_dest_rejects_bare_dotdot(fake_env):
    # safe_upload_dest()がNoneを返すケース（".."自体はDATA_DIR自身/外を指してしまう）をそのまま伝播する
    dest = ingest.resolve_upload_dest("..")

    assert dest is None


def test_resolve_upload_dest_appends_suffix_for_filename_without_extension(fake_env):
    # 拡張子がないファイル名でも "name (2)" のようにサフィックスが正しく付与される
    # （拡張子ありの場合と違い、末尾に直接 " (2)" が付くだけで拡張子相当の文字列は生まれない）
    data_dir, _store = fake_env
    _write(data_dir, "README", "既存のREADMEです。")

    dest = ingest.resolve_upload_dest("README")

    assert dest.name == "README (2)"
    assert dest.suffix == ""


def test_resolve_upload_dest_continues_numbering_from_existing_suffixed_file(fake_env):
    # 元のファイル名は存在せず、連番サフィックス付きの "file (2).txt" だけが
    # 既に存在するケース。この場合は無印の "file.txt" がまだ空いているため
    # そちらが返る（「まず無印から順に空きを探す」仕様であることの確認）
    data_dir, _store = fake_env
    _write(data_dir, "file (2).txt", "サフィックス付きの既存ファイル")

    dest = ingest.resolve_upload_dest("file.txt")

    assert dest.name == "file.txt"


def test_resolve_upload_dest_skips_to_next_free_number_when_lower_numbers_taken(fake_env):
    # "file.txt" と "file (2).txt" の両方が既に存在する状態でさらに "file.txt" を
    # アップロードすると、"file (3).txt" まで連番が繰り上がる（境界値）
    data_dir, _store = fake_env
    _write(data_dir, "file.txt", "1つ目")
    _write(data_dir, "file (2).txt", "2つ目")

    dest = ingest.resolve_upload_dest("file.txt")

    assert dest.name == "file (3).txt"
    assert not dest.exists()


# --- data_dir_signature(): 内容を読まないstat()ベースの軽量変更検知 ---


def test_data_dir_signature_is_zero_when_no_target_files(fake_env):
    # 対象ファイルが1つもない場合は (0, 0.0)
    data_dir, _store = fake_env

    assert ingest.data_dir_signature() == (0, 0.0)


def test_data_dir_signature_is_zero_when_data_dir_missing(fake_env):
    # DATA_DIR自体が存在しない場合も (0, 0.0)（存在しない前提で例外にならないことも確認）
    data_dir, _store = fake_env
    data_dir.rmdir()

    assert ingest.data_dir_signature() == (0, 0.0)


def test_data_dir_signature_counts_single_file_with_its_mtime(fake_env):
    # ファイルを1つ追加すると (1, そのファイルのmtime) を返す
    data_dir, _store = fake_env
    path = _write(data_dir, "a.txt", "1件目のファイルです。")
    os.utime(path, (1_700_000_000, 1_700_000_000))

    assert ingest.data_dir_signature() == (1, 1_700_000_000.0)


def test_data_dir_signature_increases_with_second_file(fake_env):
    # ファイルをもう1つ追加すると件数が増え、最新mtimeも新しい方に更新される
    data_dir, _store = fake_env
    path1 = _write(data_dir, "a.txt", "1件目のファイルです。")
    os.utime(path1, (1_700_000_000, 1_700_000_000))
    path2 = _write(data_dir, "b.txt", "2件目のファイルです。")
    os.utime(path2, (1_700_000_100, 1_700_000_100))

    assert ingest.data_dir_signature() == (2, 1_700_000_100.0)


def test_data_dir_signature_ignores_unsupported_extension(fake_env):
    # LOADERSに無い拡張子（対象外）のファイルは件数にもmtime計算にも含まれない
    data_dir, _store = fake_env
    path = _write(data_dir, "a.txt", "対象ファイルです。")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    unsupported = _write(data_dir, "notes.docx", "対応していない拡張子です。")
    os.utime(unsupported, (1_800_000_000, 1_800_000_000))  # より新しいmtimeでも無視される

    assert ingest.data_dir_signature() == (1, 1_700_000_000.0)


def test_data_dir_signature_decreases_when_file_removed(fake_env):
    # ファイルを削除すると件数が減る
    data_dir, _store = fake_env
    path1 = _write(data_dir, "a.txt", "1件目のファイルです。")
    os.utime(path1, (1_700_000_000, 1_700_000_000))
    path2 = _write(data_dir, "b.txt", "2件目のファイルです。")
    os.utime(path2, (1_700_000_100, 1_700_000_100))
    assert ingest.data_dir_signature()[0] == 2

    path2.unlink()

    assert ingest.data_dir_signature() == (1, 1_700_000_000.0)


def test_data_dir_signature_reflects_mtime_update_on_content_change(fake_env):
    # 既存ファイルの中身だけ変更（mtime更新）すると、件数は同じでも最新mtimeが変わる
    data_dir, _store = fake_env
    path = _write(data_dir, "a.txt", "元の内容です。")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    before = ingest.data_dir_signature()
    assert before == (1, 1_700_000_000.0)

    path.write_text("変更後の内容です。", encoding="utf-8")
    os.utime(path, (1_700_000_500, 1_700_000_500))
    after = ingest.data_dir_signature()

    assert after[0] == before[0] == 1
    assert after[1] != before[1]
    assert after == (1, 1_700_000_500.0)


# --- ロックによる排他制御（複数タブ・複数セッションからの同時実行対策） ---


def test_sync_data_dir_raises_timeout_when_lock_already_held(fake_env, monkeypatch):
    # 他のセッション（プロセス）がロックを保持中の場合、待機時間内に取得できなければ
    # filelock.Timeout を送出し、DBの整合性を壊すような同時書き込みを行わない。
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "十分な長さのテキストです。" * 5)
    monkeypatch.setattr(ingest, "SYNC_LOCK_TIMEOUT_SECONDS", 0.1)

    other_session_lock = FileLock(str(ingest.SYNC_LOCK_PATH))
    other_session_lock.acquire()
    try:
        with pytest.raises(Timeout):
            ingest.sync_data_dir(verbose=False)
    finally:
        other_session_lock.release()

    # ロック解放後は通常どおり同期できる（manifestが更新されていない状態のまま）
    result = ingest.sync_data_dir(verbose=False)
    assert result["added"] == ["a.txt"]


def test_sync_data_dir_releases_lock_after_success(fake_env):
    # 同期完了後はロックが解放され、後続の呼び出しがブロックされずに実行できる
    # （2回連続で問題なく完了することで、ロックが残留していないことを確認する）。
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "十分な長さのテキストです。" * 5)

    result_1 = ingest.sync_data_dir(verbose=False)
    assert result_1["added"] == ["a.txt"]

    result_2 = ingest.sync_data_dir(verbose=False)
    assert result_2 == {"added": [], "updated": [], "removed": [], "failed": []}

    # ロックはブロッキングせずすぐ取得できるはず
    lock = FileLock(str(ingest.SYNC_LOCK_PATH), timeout=1)
    with lock:
        pass


def test_sync_data_dir_releases_lock_when_processing_raises(fake_env, monkeypatch):
    # 同期処理の途中（ロック取得後）で予期しない例外が発生しても、ロックが
    # 保持されたままにならない（デッドロック状態を残さない）ことを確認する。
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "十分な長さのテキストです。" * 5)

    def boom(verbose):
        raise RuntimeError("同期処理中の想定外エラー")

    monkeypatch.setattr(ingest, "_sync_data_dir_locked", boom)

    with pytest.raises(RuntimeError, match="同期処理中の想定外エラー"):
        ingest.sync_data_dir(verbose=False)

    # 例外発生後もロックはすぐに取得できる（残留していない）
    lock = FileLock(str(ingest.SYNC_LOCK_PATH), timeout=1)
    with lock:
        pass


def test_sync_data_dir_serializes_concurrent_thread_calls(fake_env, monkeypatch):
    # 複数スレッドから同時にsync_data_dir()を呼んでも、実際の同期処理
    # （ロック区間）が重複して実行されないこと（＝排他が機能していること）を、
    # 実際にスレッドを起動して検証する。
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "スレッド経由で同期されるファイルAです。" * 3)
    _write(data_dir, "b.txt", "スレッド経由で同期されるファイルBです。" * 3)

    original_locked = ingest._sync_data_dir_locked
    counter_lock = threading.Lock()
    active_count = 0
    max_active = 0

    def slow_locked(verbose):
        nonlocal active_count, max_active
        with counter_lock:
            active_count += 1
            max_active = max(max_active, active_count)
        try:
            time.sleep(0.05)
            return original_locked(verbose=verbose)
        finally:
            with counter_lock:
                active_count -= 1

    monkeypatch.setattr(ingest, "_sync_data_dir_locked", slow_locked)

    results = []
    errors = []
    results_lock = threading.Lock()

    def worker():
        try:
            result = ingest.sync_data_dir(verbose=False)
            with results_lock:
                results.append(result)
        except Exception as exc:  # pragma: no cover - 失敗時にテストで検知させる
            with results_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 4
    # 同時に処理区間へ入ったスレッドは常に1つだけ（排他が機能している）
    assert max_active == 1
    # data/ の2ファイルが同期されるのは（4回呼ばれても）合計で1回分だけ
    total_added = sum(len(r["added"]) for r in results)
    assert total_added == 2


# --- ログ出力の一本化（print()廃止、logger経由のみ） ---
#
# sync_data_dir() / _load_pdf() / _load_pdf_with_docling() は以前 print() と
# logger.warning() を併用しており標準出力とログが二重になっていた。
# 現在は print() を一切使わず、失敗時は常に logger.warning()、進捗メッセージは
# verboseの値に応じて logger.info()（True）/ logger.debug()（False）に統一されている。
# ここではその一本化が壊れていないことを検証する。


def test_load_failure_does_not_print_to_stdout(fake_env, monkeypatch, capsys):
    # 失敗時にprint()による標準出力が発生しないこと（verbose=Trueでも）
    data_dir, _store = fake_env
    _write(data_dir, "bad.txt", "壊れていることにするテキストです。" * 5)
    monkeypatch.setattr(ingest, "LOADERS", {**ingest.LOADERS, ".txt": lambda path: _FailingLoader(path)})

    ingest.sync_data_dir(verbose=True)

    assert capsys.readouterr().out == ""


def test_load_failure_logs_warning_via_logger(fake_env, monkeypatch, caplog):
    # 読み込み失敗時はlogger.warning()で記録される（verboseの値に関わらず常に出力される）
    data_dir, _store = fake_env
    _write(data_dir, "bad.txt", "壊れていることにするテキストです。" * 5)
    monkeypatch.setattr(ingest, "LOADERS", {**ingest.LOADERS, ".txt": lambda path: _FailingLoader(path)})

    with caplog.at_level(logging.WARNING, logger="ingest"):
        ingest.sync_data_dir(verbose=False)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "bad.txt" in warnings[0].getMessage()
    assert "読み込みに失敗" in warnings[0].getMessage()


def test_add_documents_failure_logs_warning_via_logger(fake_env, monkeypatch, caplog):
    # ベクトルストアへの追加失敗時もlogger.warning()で記録される
    data_dir, _store = fake_env
    _write(data_dir, "new.txt", "新規追加されるファイルの内容です。" * 5)

    def flaky_add_documents(documents):
        raise RuntimeError("埋め込みモデルの一時的な失敗を想定")

    monkeypatch.setattr(_store, "add_documents", flaky_add_documents)

    with caplog.at_level(logging.WARNING, logger="ingest"):
        ingest.sync_data_dir(verbose=False)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "new.txt" in warnings[0].getMessage()
    assert "ベクトルストアへの追加に失敗" in warnings[0].getMessage()


def test_progress_message_logged_as_info_when_verbose_true(fake_env, caplog):
    # verbose=True の進捗メッセージ（追加）はINFOレベルで出力される
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "十分な長さのテキストです。" * 5)

    with caplog.at_level(logging.DEBUG, logger="ingest"):
        ingest.sync_data_dir(verbose=True)

    matching = [r for r in caplog.records if "追加" in r.getMessage() and "a.txt" in r.getMessage()]
    assert len(matching) == 1
    assert matching[0].levelname == "INFO"


def test_progress_message_logged_as_debug_when_verbose_false(fake_env, caplog):
    # verbose=False の進捗メッセージ（追加）はDEBUGレベルで出力される（デフォルト設定では非表示）
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "十分な長さのテキストです。" * 5)

    with caplog.at_level(logging.DEBUG, logger="ingest"):
        ingest.sync_data_dir(verbose=False)

    matching = [r for r in caplog.records if "追加" in r.getMessage() and "a.txt" in r.getMessage()]
    assert len(matching) == 1
    assert matching[0].levelname == "DEBUG"


def test_updated_progress_message_respects_verbose_level(fake_env, caplog):
    # 更新時の進捗メッセージも同様にverboseに応じたレベルで出力される
    data_dir, _store = fake_env
    path = _write(data_dir, "a.txt", "元のテキスト内容です。")
    ingest.sync_data_dir(verbose=False)
    path.write_text("変更後のまったく別のテキスト内容です。", encoding="utf-8")

    with caplog.at_level(logging.DEBUG, logger="ingest"):
        ingest.sync_data_dir(verbose=True)

    matching = [r for r in caplog.records if "更新" in r.getMessage() and "a.txt" in r.getMessage()]
    assert len(matching) == 1
    assert matching[0].levelname == "INFO"


def test_removed_progress_message_respects_verbose_level(fake_env, caplog):
    # 削除時の進捗メッセージも同様にverboseに応じたレベルで出力される
    data_dir, _store = fake_env
    path = _write(data_dir, "a.txt", "削除されるファイルの内容です。")
    ingest.sync_data_dir(verbose=False)
    path.unlink()

    with caplog.at_level(logging.DEBUG, logger="ingest"):
        ingest.sync_data_dir(verbose=False)

    matching = [r for r in caplog.records if "削除" in r.getMessage() and "a.txt" in r.getMessage()]
    assert len(matching) == 1
    assert matching[0].levelname == "DEBUG"


def test_no_changes_message_logged_as_info_when_verbose_true(fake_env, caplog):
    # 変更なしメッセージもverbose=Trueならinfoレベルで出力される
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "変わらないテキスト内容です。")
    ingest.sync_data_dir(verbose=False)

    with caplog.at_level(logging.DEBUG, logger="ingest"):
        ingest.sync_data_dir(verbose=True)

    matching = [r for r in caplog.records if "変更はありませんでした" in r.getMessage()]
    assert len(matching) == 1
    assert matching[0].levelname == "INFO"


def test_no_changes_message_logged_as_debug_when_verbose_false(fake_env, caplog):
    # 変更なしメッセージはverbose=Falseならdebugレベルで出力される（デフォルト設定では非表示）
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "変わらないテキスト内容です。")
    ingest.sync_data_dir(verbose=False)

    with caplog.at_level(logging.DEBUG, logger="ingest"):
        ingest.sync_data_dir(verbose=False)

    matching = [r for r in caplog.records if "変更はありませんでした" in r.getMessage()]
    assert len(matching) == 1
    assert matching[0].levelname == "DEBUG"


def test_no_changes_message_not_logged_when_default_level_and_verbose_false(fake_env, caplog):
    # verbose=Falseでdebugログを収集しない既定のログレベル（WARNING等）では
    # 進捗メッセージがそもそも記録されない（app.py/api経由の本番導線を想定した確認）
    data_dir, _store = fake_env
    _write(data_dir, "a.txt", "変わらないテキスト内容です。")
    ingest.sync_data_dir(verbose=False)

    with caplog.at_level(logging.WARNING, logger="ingest"):
        ingest.sync_data_dir(verbose=False)

    assert caplog.records == []


def test_docling_fallback_progress_message_respects_verbose_level(monkeypatch, tmp_path, caplog):
    # _load_pdf()内のDoclingフォールバック進捗メッセージもverboseに応じたレベルで出力される
    fake_pdf_path = tmp_path / "encrypted.pdf"
    fake_pdf_path.write_bytes(b"not a real pdf")

    class _FailingPyMuPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            raise ValueError("暗号化されたPDFは読み込めません")

    class _SucceedingDoclingLoader:
        def __init__(self, file_path, export_type):
            self.file_path = file_path

        def load(self):
            return [Document(page_content="Doclingで抽出した本文です。")]

    monkeypatch.setattr(ingest, "DOCLING_AVAILABLE", True)
    monkeypatch.setattr(ingest, "PyMuPDFLoader", _FailingPyMuPDFLoader)
    monkeypatch.setattr(ingest, "DoclingLoader", _SucceedingDoclingLoader)

    with caplog.at_level(logging.DEBUG, logger="ingest"):
        ingest._load_pdf(fake_pdf_path, verbose=True)

    matching = [r for r in caplog.records if "Doclingでの再解析を試みます" in r.getMessage()]
    assert len(matching) == 1
    assert matching[0].levelname == "INFO"


def test_docling_load_failure_progress_message_is_debug_when_verbose_false(monkeypatch, tmp_path, caplog):
    # _load_pdf_with_docling()内のDocling解析失敗メッセージも、
    # verbose=Falseならdebugレベルで出力される
    fake_pdf_path = tmp_path / "broken.pdf"
    fake_pdf_path.write_bytes(b"not a real pdf")

    class _FailingDoclingLoader:
        def __init__(self, file_path, export_type):
            self.file_path = file_path

        def load(self):
            raise RuntimeError("Doclingでも読み込めません")

    monkeypatch.setattr(ingest, "DoclingLoader", _FailingDoclingLoader)

    with caplog.at_level(logging.DEBUG, logger="ingest"):
        docs = ingest._load_pdf_with_docling(fake_pdf_path, verbose=False)

    assert docs == []
    matching = [r for r in caplog.records if "Docling解析に失敗しました" in r.getMessage()]
    assert len(matching) == 1
    assert matching[0].levelname == "DEBUG"


def test_main_configures_logging_basic_config_for_cli_console_output(monkeypatch, fake_env, capsys):
    # CLIエントリポイントmain()は、verbose=True（デフォルト）の進捗ログ（logger.info）を
    # コンソールに表示するため、INFOレベル・標準出力向けのlogging.basicConfig()を呼ぶこと
    monkeypatch.setattr("sys.argv", ["ingest.py"])
    monkeypatch.setattr(ingest, "sync_data_dir", lambda: {"added": [], "updated": [], "removed": [], "failed": []})

    basic_config_calls = []
    original_basic_config = logging.basicConfig

    def recording_basic_config(**kwargs):
        basic_config_calls.append(kwargs)
        # 実際にハンドラを増殖させ他のテストへ影響を残さないよう、実行はしない

    monkeypatch.setattr(logging, "basicConfig", recording_basic_config)

    ingest.main()

    assert len(basic_config_calls) == 1
    assert basic_config_calls[0]["level"] == logging.INFO
    import sys as sys_module

    assert basic_config_calls[0]["stream"] is sys_module.stdout

    monkeypatch.setattr(logging, "basicConfig", original_basic_config)
    out = capsys.readouterr().out
    assert "同期しています" in out
    assert "完了" in out
