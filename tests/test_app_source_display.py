"""app.py の参照元表示整形ロジック（Issue #4）のテスト。

`_format_snippet` / `_format_source_label` はどちらも Streamlit の機能（st.*）を
一切使わない純粋関数のため、`tests/test_app.py` のような `AppTest` 経由の
UIレベル検証ではなく、関数を直接呼び出す通常のユニットテストとして検証する。

ただし app.py はモジュールトップレベルで `build_agent()` など重い外部依存を
呼び出すコードを持つため、`import app` を実行するには事前に
`tests/test_app.py` と同じ軽量フェイクへの差し替えが必要になる。
このファイルではその差し替えを `app_module` フィクスチャに閉じ込め、
- import前に ingest / rag_chain / memory の該当関数を一時的にフェイクへ差し替える
- import後（module-scopedのため実際にはプロセス内で最初にimportされた1回のみ
  実行される。2回目以降は`sys.modules`のキャッシュを再利用するだけなので
  差し替え自体は空振りするが、無害）は元の値に戻す
ことで、他のテストファイル（例: `tests/test_ingest.py` が `ingest.sync_data_dir`
を実際の実装のまま直接呼び出して検証している）に影響を残さないようにする。
"""

import importlib
import sys

import pytest
from langchain_core.messages import AIMessage

import ingest
import memory
import rag_chain


class _FakeAgent:
    def invoke(self, payload):
        return {"messages": payload["messages"] + [AIMessage(content="ok")]}


def _ok_sync(verbose=False):
    return {"added": [], "updated": [], "removed": [], "failed": []}


@pytest.fixture(scope="module")
def app_module():
    """app.py を軽量フェイク差し替え済みの状態でimportし、モジュールオブジェクトを返す。"""
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
        module = sys.modules.get("app") or importlib.import_module("app")
        yield module
    finally:
        ingest.sync_data_dir = originals["ingest.sync_data_dir"]
        ingest.data_dir_signature = originals["ingest.data_dir_signature"]
        rag_chain.build_agent = originals["rag_chain.build_agent"]
        memory.new_thread_id = originals["memory.new_thread_id"]
        memory.conversation_count = originals["memory.conversation_count"]
        memory.save_conversation = originals["memory.save_conversation"]


# --- 1. _format_snippet ---


def test_format_snippet_returns_as_is_when_under_limit(app_module):
    """正常系: limit未満の短いテキストはそのまま返り、"..."は付かない。"""
    text = "短いテキストです。"
    assert app_module._format_snippet(text) == text


def test_format_snippet_returns_as_is_when_exactly_limit(app_module):
    """境界値: ちょうどlimit(300)文字の場合は切り詰めず、"..."も付かない。"""
    text = "あ" * 300
    result = app_module._format_snippet(text)
    assert result == text
    assert not result.endswith("...")


def test_format_snippet_strips_surrounding_whitespace_when_under_limit(app_module):
    """境界値: limit未満でも前後の空白は取り除かれる。"""
    assert app_module._format_snippet("  短い  ") == "短い"


def test_format_snippet_cuts_at_last_break_char_when_far_enough_from_start(app_module):
    """正常系: limitを超える場合、末尾に最も近い区切り文字（句点等）が
    limitの半分より後ろにあれば、そこで文を区切る。"""
    # 先頭150文字は区切り文字なしで埋め、200文字目に句点を置き、
    # そこから300文字目までさらに文章を続けたテキスト（全体で400文字超）を作る。
    head = "あ" * 200 + "。"  # 201文字目が句点
    tail = "い" * 200  # 300文字目までの残りを埋める
    text = head + tail
    assert len(text[:300]) == 300

    result = app_module._format_snippet(text)

    # 句点(head内、201文字目、300の半分=150より後ろ)を採用し、そこまでで区切って"..."を付与
    assert result == head + "..."
    assert result.startswith("あ" * 200 + "。")


def test_format_snippet_ignores_break_char_too_close_to_start(app_module):
    """境界値: 区切り文字がlimitの半分より手前にしかない場合は不採用になり、
    直近の空白（limitの半分より後ろにあるもの）にフォールバックする。"""
    # 10文字目に句点（150より手前なので不採用）、200文字目に半角スペース（150より後ろなので採用対象）
    text = "あ" * 10 + "。" + "い" * 189 + " " + "う" * 200
    assert len(text) > 300

    result = app_module._format_snippet(text)

    expected_cut = "あ" * 10 + "。" + "い" * 189  # 200文字目の空白の手前まで
    assert result == expected_cut + "..."


def test_format_snippet_falls_back_to_hard_cut_when_no_break_or_space(app_module):
    """異常系: 区切り文字も空白も見つからない場合は、単純にlimit文字で切って"..."を付ける。"""
    text = "あ" * 400  # 区切り文字・空白を含まない
    result = app_module._format_snippet(text)
    assert result == "あ" * 300 + "..."


@pytest.mark.parametrize("break_char", ["。", "\n", "！", "？", "!", "?"])
def test_format_snippet_recognizes_all_break_chars(app_module, break_char):
    """正常系: 句点・改行・全角/半角の感嘆符/疑問符のいずれも区切り文字として認識される。

    実装は区切り文字までを含めて切った後 `.rstrip()` してから"..."を付けるため、
    区切り文字自体が空白文字（"\n"）の場合はrstrip()で取り除かれ結果に残らない
    （句点・感嘆符・疑問符は空白文字ではないためrstrip()の影響を受けず結果に残る）。
    """
    head = "あ" * 200 + break_char
    tail = "い" * 200
    text = head + tail

    result = app_module._format_snippet(text)

    assert result == head.rstrip() + "..."


def test_format_snippet_break_char_exactly_at_half_limit_is_accepted(app_module):
    """境界値: 区切り文字の位置がちょうどlimitの半分(150)の場合は採用される
    （実装は `best_pos >= limit // 2` のため150は採用境界の内側）。"""
    limit = 300
    half = limit // 2  # 150
    head = "あ" * half + "。"  # "。"のインデックスはちょうど150
    tail = "い" * 200  # limitを確実に超えさせるための十分な余剰
    text = head + tail
    assert len(text) > limit
    assert text[:300].rfind("。") == half

    result = app_module._format_snippet(text)

    assert result == head + "..."


def test_format_snippet_default_limit_is_300(app_module):
    """境界値: limit引数を省略した場合のデフォルト値は300文字。"""
    text_299 = "あ" * 299
    text_301 = "あ" * 301

    assert app_module._format_snippet(text_299) == text_299
    assert app_module._format_snippet(text_301) == "あ" * 300 + "..."


def test_format_snippet_respects_custom_limit(app_module):
    """正常系: limitを明示的に指定した場合はその値が使われる。"""
    text = "あ" * 50
    assert app_module._format_snippet(text, limit=10) == "あ" * 10 + "..."
    assert app_module._format_snippet(text, limit=50) == text


# --- 2. _format_source_label ---


def test_format_source_label_plain_document_without_thread_or_page(app_module):
    """正常系（従来動作維持）: thread_id・pageが無い通常ドキュメントはファイル名のみ表示。"""
    label = app_module._format_source_label({"source": "data/foo.txt"})
    assert label == "foo.txt"


def test_format_source_label_extracts_filename_from_nested_path(app_module):
    """正常系: sourceがディレクトリを含むパスでも、ファイル名部分のみが表示される。"""
    label = app_module._format_source_label({"source": "data/subdir/report.txt"})
    assert label == "report.txt"


def test_format_source_label_pdf_includes_one_indexed_page_number(app_module):
    """正常系・境界値: PDFのpage(0始まり)は1始まりに変換されて「（p.N）」で付与される。
    page=0（1ページ目、Python的にはfalsy）でも `is not None` 判定のため正しく付与される。"""
    label = app_module._format_source_label({"source": "data/doc.pdf", "page": 0})
    assert label == "doc.pdf（p.1）"

    label2 = app_module._format_source_label({"source": "data/doc.pdf", "page": 4})
    assert label2 == "doc.pdf（p.5）"


def test_format_source_label_conversation_log_with_non_global_thread_id(app_module):
    """正常系: thread_idがあり、かつGLOBAL_THREAD_IDでない場合は
    「会話ログ（スレッド: xxx） - ファイル名」の形式になる。"""
    label = app_module._format_source_label({"source": "data/conversations/abc123/log.txt", "thread_id": "abc123"})
    assert label == "会話ログ（スレッド: abc123） - log.txt"


def test_format_source_label_conversation_log_with_page(app_module):
    """正常系: 会話ログ表示とページ番号表示は組み合わせても両方付与される。"""
    label = app_module._format_source_label({"source": "conv.txt", "thread_id": "abc123", "page": 2})
    assert label == "会話ログ（スレッド: abc123） - conv.txt（p.3）"


def test_format_source_label_global_thread_id_is_treated_as_normal_document(app_module):
    """境界値: thread_idがGLOBAL_THREAD_ID（"global"）の場合は、
    会話ログ由来ではない全スレッド共通ドキュメントのため、
    従来通りファイル名のみが表示される（会話ログ表記は付かない）。"""
    label = app_module._format_source_label({"source": "data/foo.txt", "thread_id": rag_chain.GLOBAL_THREAD_ID})
    assert label == "foo.txt"


def test_format_source_label_empty_string_thread_id_is_treated_as_absent(app_module):
    """境界値: thread_idが空文字列の場合はfalsyのため、未設定時と同様に扱われる。"""
    label = app_module._format_source_label({"source": "data/foo.txt", "thread_id": ""})
    assert label == "foo.txt"


def test_format_source_label_unknown_source_when_missing(app_module):
    """異常系: metadataにsourceキー自体が無い場合は"unknown"がそのまま表示される
    （Path()でファイル名抽出を試みず、従来動作を維持）。"""
    label = app_module._format_source_label({})
    assert label == "unknown"


def test_format_source_label_unknown_source_with_page(app_module):
    """異常系境界値: sourceが不明でもpageがあれば、そのまま「（p.N）」が付与される。"""
    label = app_module._format_source_label({"page": 0})
    assert label == "unknown（p.1）"
