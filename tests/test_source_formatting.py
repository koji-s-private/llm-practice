"""source_formatting.py（参照元表示整形の共通ロジック）の純粋関数のユニットテスト。

`format_snippet` / `format_source_label` はStreamlit・FastAPIいずれの機能にも
依存しない純粋関数のため、`app.py` / `api/main.py` を経由せずモジュールを
直接importして検証する。挙動そのものは元々 `app.py` に定義されていたものを
そのまま切り出したものであり、詳細な境界値パターン（区切り文字の探索、
GLOBAL_THREAD_IDの扱い等）は既存の `tests/test_app_source_display.py` で
`app._format_snippet` / `app._format_source_label`（再エクスポート経由）として
既にカバー済みのため、ここでは重複を避けつつ、モジュール単体としての
主要な正常系・異常系・境界値を確認する。
"""

import importlib
import sys

import pytest

import ingest
import memory
import rag_chain
import source_formatting

# --- format_snippet ---


def test_format_snippet_returns_as_is_when_under_limit():
    """正常系: limit未満の短いテキストはそのまま返り、"..."は付かない。"""
    text = "短いテキストです。"
    assert source_formatting.format_snippet(text) == text


def test_format_snippet_strips_surrounding_whitespace():
    """境界値: 前後の空白は常に取り除かれる。"""
    assert source_formatting.format_snippet("  短い  ") == "短い"


def test_format_snippet_truncates_and_appends_ellipsis_when_over_limit():
    """正常系: limitを超える場合は切り詰められ、末尾に"..."が付く。"""
    text = "あ" * 400
    result = source_formatting.format_snippet(text)
    assert result == "あ" * 300 + "..."


def test_format_snippet_empty_string_returns_empty_string():
    """異常系境界値: 空文字列を渡した場合は空文字列のまま返る（"..."は付かない）。"""
    assert source_formatting.format_snippet("") == ""


def test_format_snippet_respects_custom_limit():
    """正常系: limitを明示的に指定した場合はその値が使われる。"""
    text = "あ" * 50
    assert source_formatting.format_snippet(text, limit=10) == "あ" * 10 + "..."
    assert source_formatting.format_snippet(text, limit=50) == text


# --- format_source_label ---


def test_format_source_label_plain_document_without_thread_or_page():
    """正常系: thread_id・pageが無い通常ドキュメントはファイル名のみ表示。"""
    assert source_formatting.format_source_label({"source": "data/foo.txt"}) == "foo.txt"


def test_format_source_label_pdf_includes_one_indexed_page_number():
    """正常系・境界値: PDFのpage(0始まり)は1始まりに変換されて「（p.N）」で付与される。"""
    assert source_formatting.format_source_label({"source": "data/doc.pdf", "page": 0}) == "doc.pdf（p.1）"


def test_format_source_label_conversation_log_with_non_global_thread_id():
    """正常系: thread_idがあり、かつGLOBAL_THREAD_IDでない場合は
    「会話ログ（スレッド: xxx） - ファイル名」の形式になる。"""
    metadata = {"source": "data/conversations/abc123/log.txt", "thread_id": "abc123"}
    label = source_formatting.format_source_label(metadata)
    assert label == "会話ログ（スレッド: abc123） - log.txt"


def test_format_source_label_global_thread_id_is_treated_as_normal_document():
    """境界値: thread_idがGLOBAL_THREAD_IDの場合は会話ログ表記を付けない。"""
    label = source_formatting.format_source_label({"source": "data/foo.txt", "thread_id": rag_chain.GLOBAL_THREAD_ID})
    assert label == "foo.txt"


def test_format_source_label_unknown_source_when_missing():
    """異常系境界値: metadataが空のdictの場合は例外を送出せず、
    sourceは"unknown"のまま、pageも付与されない。"""
    assert source_formatting.format_source_label({}) == "unknown"


# --- app.py / api/main.py からの再エクスポートが同一実装を指していることの確認 ---
# （個々の境界値の網羅は test_app_source_display.py / test_api.py 側に委ねる）


def _ok_sync(verbose=False):
    return {"added": [], "updated": [], "removed": [], "failed": []}


class _FakeAgent:
    def invoke(self, payload):
        return payload


@pytest.fixture
def app_module():
    """app.py を軽量フェイク差し替え済みの状態でimportし、モジュールオブジェクトを返す。

    app.py はモジュールトップレベルで sync_data_dir() 等の重い処理を呼び出すため、
    未importの場合のみ tests/test_app_source_display.py と同じ方針でフェイクに
    差し替えてからimportする（既にimport済みなら sys.modules のキャッシュをそのまま使う）。
    """
    if "app" in sys.modules:
        yield sys.modules["app"]
        return

    originals = {
        "ingest.sync_data_dir": ingest.sync_data_dir,
        "ingest.data_dir_signature": ingest.data_dir_signature,
        "rag_chain.build_agent": rag_chain.build_agent,
        "memory.new_thread_id": memory.new_thread_id,
        "memory.conversation_count": memory.conversation_count,
        "memory.save_conversation": memory.save_conversation,
    }
    ingest.sync_data_dir = _ok_sync
    ingest.data_dir_signature = lambda: (0, 0.0)
    rag_chain.build_agent = lambda thread_id=None: _FakeAgent()
    memory.new_thread_id = lambda: "thread-test"
    memory.conversation_count = lambda thread_id: 0
    memory.save_conversation = lambda *a, **k: None
    try:
        module = importlib.import_module("app")
        yield module
    finally:
        ingest.sync_data_dir = originals["ingest.sync_data_dir"]
        ingest.data_dir_signature = originals["ingest.data_dir_signature"]
        rag_chain.build_agent = originals["rag_chain.build_agent"]
        memory.new_thread_id = originals["memory.new_thread_id"]
        memory.conversation_count = originals["memory.conversation_count"]
        memory.save_conversation = originals["memory.save_conversation"]


def test_app_module_reexports_the_same_function_objects(app_module):
    """app.py の _format_snippet / _format_source_label が
    source_formatting モジュールの同一オブジェクトを再エクスポートしていることを確認する
    （import時にコピーではなく参照を共有していれば、片方の修正漏れが起きにくい）。"""
    assert app_module._format_snippet is source_formatting.format_snippet
    assert app_module._format_source_label is source_formatting.format_source_label


def test_api_main_module_reexports_the_same_function_objects():
    """api/main.py の _format_snippet / _format_source_label も同様に
    source_formatting モジュールの同一オブジェクトを再エクスポートしていることを確認する。

    api/main.py はモジュールトップレベルで重い処理を呼び出さないため、
    app.py と異なりフェイク差し替えなしで直接importできる
    （tests/test_api.py の方針と同じ）。
    """
    from api import main as api_main

    assert api_main._format_snippet is source_formatting.format_snippet
    assert api_main._format_source_label is source_formatting.format_source_label
