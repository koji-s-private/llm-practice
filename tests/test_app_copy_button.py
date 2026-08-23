"""app.py の回答コピーボタン（`_copy_button_html`）のテスト。

`_copy_button_html` は Streamlit の機能（st.*）を一切使わない純粋関数のため、
`tests/test_app_source_display.py` と同様に `AppTest` 経由のUIレベル検証ではなく、
関数を直接呼び出す通常のユニットテストとして検証する。

ただし app.py はモジュールトップレベルで `build_agent()` など重い外部依存を
呼び出すコードを持つため、`import app` を実行するには事前に
`tests/test_app_source_display.py` と同じ軽量フェイクへの差し替えが必要になる。
このファイルではその差し替えを `app_module` フィクスチャに閉じ込め、他のテスト
ファイルに影響を残さないようにする（同フィクスチャの詳細な意図は
`tests/test_app_source_display.py` のdocstring参照）。
"""

import importlib
import json
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


# --- _copy_button_html ---


def test_copy_button_html_embeds_json_dumps_encoded_text(app_module):
    """正常系: 通常のテキストはjson.dumpsでエンコードされた形でJS内に埋め込まれる。"""
    text = "これはAIの回答です。"
    html = app_module._copy_button_html(text)

    assert json.dumps(text) in html
    assert "navigator.clipboard.writeText" in html
    assert '<button onclick="copyAnswer(this)">📋 回答をコピー</button>' in html


def test_copy_button_html_escapes_double_quotes(app_module):
    """異常系: ダブルクォートを含むテキストでも、JS文字列リテラルとして安全にエスケープされる。"""
    text = 'ダブルクォート"を含む回答'
    html = app_module._copy_button_html(text)

    encoded = json.dumps(text)
    assert encoded in html
    # エスケープされていない生のダブルクォート("を含む回答" の直前)がそのまま残っておらず、
    # \" にエスケープされた形でのみ埋め込まれていること（JS文字列リテラルが破壊されていない）。
    assert '\\"' in encoded
    assert f"const text = {encoded};" in html


def test_copy_button_html_escapes_single_quotes_and_backticks(app_module):
    """異常系: シングルクォート・バッククォートを含むテキストでも安全に埋め込まれる。

    json.dumpsはダブルクォート文字列リテラルとして出力するため、シングルクォートや
    バッククォートはエスケープ不要でそのまま埋め込まれても文字列リテラルは破壊されない。
    """
    text = "シングル'とバッククォート`を含む回答"
    html = app_module._copy_button_html(text)

    assert json.dumps(text) in html


def test_copy_button_html_escapes_newlines(app_module):
    """異常系: 改行を含むテキストでも、JS文字列リテラルとして1行にエスケープされる。"""
    text = "1行目\n2行目"
    html = app_module._copy_button_html(text)

    encoded = json.dumps(text)
    assert encoded in html
    # json.dumpsは改行を\nへエスケープするため、生の改行がJS文字列リテラル内に残らない。
    assert "\\n" in encoded
    assert 'const text = "1行目\n2行目";' not in html


def test_copy_button_html_does_not_break_out_of_script_tag(app_module):
    """異常系（XSS観点）: 回答テキストに `</script>` が含まれていても、
    生成されたHTMLの中でscriptタグが意図せず終端されないこと。

    `json.dumps()` はスラッシュ(`/`)をエスケープしないため、回答テキストに文字列として
    `</script>` が含まれていると、そのまま `</script>` という文字列がHTML中に出力される。
    HTMLパーサーはJS文字列リテラルの中身を見ずに `</script>` というバイト列だけを見て
    scriptタグを終端してしまうため、後続の `alert(1)</script>` 部分がHTML本文として
    解釈されてしまう（scriptタグの分断によるインジェクション）。
    """
    text = "これは</script><script>alert(1)</script>という回答です"
    html = app_module._copy_button_html(text)

    # scriptブロックの終了タグは元々のテンプレート由来の1つだけであるべきだが、
    # 回答テキスト由来の `</script>` がエスケープされていないため複数出現してしまう。
    script_close_count = html.count("</script>")
    assert script_close_count == 1, (
        f"回答テキスト中の '</script>' がエスケープされておらず、"
        f"HTML内に{script_close_count}個の閉じscriptタグが出現しました。"
        "json.dumps()はスラッシュをエスケープしないため、scriptタグが分断されXSSにつながる恐れがあります。"
    )


def test_copy_button_html_empty_string(app_module):
    """境界値: 空文字列の場合でも例外にならず、空のJS文字列リテラルが埋め込まれる。"""
    html = app_module._copy_button_html("")

    assert 'const text = "";' in html


def test_copy_button_html_returns_str(app_module):
    """正常系: 戻り値はstr型である。"""
    assert isinstance(app_module._copy_button_html("テキスト"), str)
