"""app.py のエラーハンドリング・自動再同期のテスト。

`streamlit.testing.v1.AppTest` を使い、実際にStreamlitのスクリプト実行エンジン上で
app.py を動かして検証する。重い外部依存（埋め込みモデル・Chroma・LLM）は使わず、
app.py が直接 import している以下のシンボルを monkeypatch で軽量なフェイクに
差し替える:

- `ingest.sync_data_dir`（サイドバーの再同期ボタン・ファイルアップロード時・
  トップレベルの軽量シグネチャチェックがdata/の変化を検知した場合、の各所から呼ばれる。
  data/配下を全件走査する重い経路）
- `ingest.add_single_conversation_file`（チャット応答後、save_conversationが返した
  保存先パス1件だけをその場でDBへ軽量に反映する経路。data/全件は走査しない）
- `ingest.data_dir_signature`（トップレベルで毎回呼ばれる軽量な
  変更検知。デフォルトでは実ファイルシステムを見るため、シグネチャの変化を
  意図的に起こしたいテストではmonkeypatchで差し替える）
- `rag_chain.build_agent`（フェイクエージェントを返す。`.invoke()` の成功/失敗を
  テストごとに切り替える）
- `memory.new_thread_id` / `memory.conversation_count` / `memory.save_conversation` /
  `memory.list_threads` / `memory.load_conversation` / `memory.load_thread_title` /
  `memory.save_thread_title`
  （`memory.save_conversation` は実際の実装と同様、保存先ファイルパス(Path)を返す）

app.py はモジュールトップレベルで `from ingest import ... sync_data_dir` のように
シンボルをインポートしているため、`AppTest.run()` がスクリプトを実行する
「前」に対象モジュールの属性を monkeypatch しておく必要がある
（実行時に束縛される値がその時点の属性値になるため）。

会話ログ保存後の同期トリガーの設計:
- チャット応答→会話ログ保存（save_conversation）の直後、同じturn内で
  `add_single_conversation_file(saved_path)` を呼び、保存した1ファイルだけを
  その場でDBへ反映する（data/全件を再走査する`sync_data_dir`は呼ばない）。
- 成功時（"added"/"updated"/"unchanged"）は `st.session_state.data_dir_signature` も
  その場で最新値に更新し、次回rerun時のトップレベルの軽量チェックによる
  無駄な二重同期を防ぐ。
- 失敗時（"failed"を返す、または例外送出）はシグネチャを更新しない。これにより
  次回のトップレベルの軽量チェックが「data/に未反映の変更あり」と判定し続け、
  通常の全件差分同期（`sync_data_dir`）で改めて再試行される。
"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

import feedback
import google_drive_sync
import ingest
import memory
import rag_chain

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class _FakeAgent:
    """rag_chain.build_agent() の代わりに使うフェイクエージェント。

    app.py は agent.stream(..., stream_mode="messages") で (チャンク, メタデータ) の
    タプルを逐次受け取るため、.stream() をメインで実装する。.invoke() は
    互換性のため残しているが、現在のapp.pyからは呼ばれない。
    """

    def __init__(self, answer="テスト回答です", exc=None, chunks=None):
        self.answer = answer
        self.exc = exc
        # 回答を複数チャンクに分割してyieldしたい場合はchunksを明示的に渡す。
        # 未指定の場合、例外系テスト（exc指定あり）では部分的な回答チャンクを一切出さずに
        # 直ちに失敗させ（従来のinvoke()失敗時と同じ「何も表示されないまま失敗する」挙動を再現）、
        # 正常系テストではanswer全体を1チャンクとして返す。
        if chunks is not None:
            self.chunks = chunks
        elif exc is not None:
            self.chunks = []
        else:
            self.chunks = [answer]
        self.invoke_calls = []
        self.stream_calls = []

    def invoke(self, payload):
        self.invoke_calls.append(payload)
        if self.exc is not None:
            raise self.exc
        return {"messages": payload["messages"] + [AIMessage(content=self.answer)]}

    def stream(self, payload, stream_mode="messages"):
        self.stream_calls.append(payload)
        for piece in self.chunks:
            yield AIMessageChunk(content=piece), {}
        if self.exc is not None:
            raise self.exc


class _FakeAgentWithSources:
    """ToolMessageのartifactとして参照元ドキュメントを返すフェイクエージェント。

    app.py は ToolMessage.artifact から sources を組み立てて、それが空か否かで
    save_conversation の is_fallback 引数を決めるため、そのソース有り無しを
    テストごとに切り替えられるようにする。
    """

    def __init__(self, answer="テスト回答です", artifact=None):
        self.answer = answer
        self.artifact = artifact if artifact is not None else []

    def invoke(self, payload):
        tool_message = ToolMessage(content="検索結果", artifact=self.artifact, tool_call_id="call-1")
        return {"messages": payload["messages"] + [tool_message, AIMessage(content=self.answer)]}

    def stream(self, payload, stream_mode="messages"):
        # ツール実行結果（ToolMessage）を先にyieldし、その後に回答本文チャンクをyieldする。
        # app.pyはToolMessageからartifactをsourcesへ蓄積し、AIMessageChunkのcontentのみを
        # st.write_streamで逐次描画する。
        tool_message = ToolMessage(content="検索結果", artifact=self.artifact, tool_call_id="call-1")
        yield tool_message, {}
        yield AIMessageChunk(content=self.answer), {}


class _FakeAgentWithMultipleToolCalls:
    """retrieve_contextが1ターン中に複数回呼ばれるケースを模したフェイクエージェント。

    ToolMessageを複数回yieldし、各回のartifactに重複するドキュメントが
    含まれていても sources 側で重複排除されることを確認するために使う。
    """

    def __init__(self, answer, artifacts):
        self.answer = answer
        self.artifacts = artifacts

    def stream(self, payload, stream_mode="messages"):
        for i, artifact in enumerate(self.artifacts):
            yield ToolMessage(content="検索結果", artifact=artifact, tool_call_id=f"call-{i}"), {}
        yield AIMessageChunk(content=self.answer), {}


def _ok_sync(verbose=False):
    return {"added": [], "updated": [], "removed": [], "failed": []}


# save_conversation() の戻り値（保存先パス）のデフォルトフェイク値。
# 実際の値そのものはほとんどのテストで意味を持たないため、add_single_conversation_file
# 呼び出し先まで見ないテストでは固定値のままでよい。
_FAKE_SAVED_CONVERSATION_PATH = Path("/tmp/data/conversations/thread-test/fake.md")


@pytest.fixture(autouse=True)
def _patch_light_dependencies(monkeypatch):
    """全テスト共通: 起動時同期とmemory系を軽量フェイクに差し替える。

    build_agent・data_dir_signatureの挙動・チャット後の同期挙動は、
    各テストが個別に上書きする。
    """
    monkeypatch.setattr(ingest, "sync_data_dir", _ok_sync)
    monkeypatch.setattr(ingest, "add_single_conversation_file", lambda path: "added")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: _FakeAgent())
    monkeypatch.setattr(memory, "new_thread_id", lambda: "thread-test")
    monkeypatch.setattr(memory, "conversation_count", lambda thread_id: 0)
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: _FAKE_SAVED_CONVERSATION_PATH)
    monkeypatch.setattr(memory, "list_threads", lambda: [])
    monkeypatch.setattr(memory, "load_conversation", lambda thread_id: [])
    monkeypatch.setattr(memory, "load_thread_title", lambda thread_id: None)
    monkeypatch.setattr(memory, "save_thread_title", lambda thread_id, title: None)
    monkeypatch.setattr(feedback, "record_feedback", lambda *a, **k: None)


def _run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run()
    return at


# --- 1. 起動時の sync_data_dir() 呼び出し（_sync_and_report経由） ---


def test_startup_sync_success_shows_no_error():
    """正常系: 起動時同期が成功すればエラーは表示されず、エージェントも構築される。"""
    at = _run_app()

    assert at.exception == []
    assert at.error == []
    assert "agent" in at.session_state


def test_startup_sync_failure_shows_error_but_app_keeps_running(monkeypatch):
    """異常系: 起動時同期が例外を送出しても、st.error表示のみでクラッシュせず、
    エージェント構築（後続処理）は継続される。"""

    def failing_sync(verbose=False):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ingest, "sync_data_dir", failing_sync)

    at = _run_app()

    assert at.exception == []
    assert len(at.error) == 1
    assert "ドキュメントの同期に失敗しました" in at.error[0].value
    assert "disk full" in at.error[0].value
    # 同期失敗後もアプリはクラッシュせず、エージェント構築まで到達している
    assert "agent" in at.session_state


def test_startup_sync_with_failed_files_shows_warning_and_toast(monkeypatch):
    """異常系境界値: 追加/更新/削除に加え、一部ファイルの読み込みに失敗した場合、
    st.toastで変更件数、st.warningで失敗ファイル名一覧がそれぞれ表示され、
    アプリはクラッシュせず継続する
    （ingest.sync_data_dir()の戻り値仕様: added/updated/removed/failedの4キー。
    失敗ファイルがあるケースの回帰を防ぐために追加）。"""

    def partial_failure_sync(verbose=False):
        return {
            "added": ["good.txt"],
            "updated": [],
            "removed": [],
            "failed": ["bad.pdf", "broken.txt"],
        }

    monkeypatch.setattr(ingest, "sync_data_dir", partial_failure_sync)

    at = _run_app()

    assert at.exception == []
    assert at.error == []
    assert len(at.toast) == 1
    assert "追加1" in at.toast[0].value
    assert len(at.warning) == 1
    assert "bad.pdf" in at.warning[0].value
    assert "broken.txt" in at.warning[0].value
    # 失敗があってもエージェント構築（後続処理）は継続される
    assert "agent" in at.session_state


def test_resync_button_failure_shows_error(monkeypatch):
    """異常系: サイドバーの「🔄 data/ を再同期」ボタン押下時の同期失敗もカバーされる。

    このボタンは `st.expander("今すぐ強制的に再同期したい場合")` の中に
    移動したが、`AppTest`の`at.sidebar.button`はexpander内も含めてサイドバー配下の
    ボタンを再帰的に収集するため、取得方法自体は変更不要（folded状態でも
    要素ツリーには存在し、クリック操作も通常どおり可能）。
    """
    call_count = {"n": 0}

    def flaky_sync(verbose=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"added": [], "updated": [], "removed": [], "failed": []}
        raise RuntimeError("resync fail")

    monkeypatch.setattr(ingest, "sync_data_dir", flaky_sync)

    at = _run_app()
    assert at.error == []

    resync_button = next(b for b in at.sidebar.button if "再同期" in b.label)
    at = resync_button.click().run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "ドキュメントの同期に失敗しました" in at.error[0].value
    assert "resync fail" in at.error[0].value


def test_resync_button_failed_files_shows_warning_immediately(monkeypatch):
    """異常系: サイドバーの再同期ボタン押下で同期が"failed"を返した場合、次のスクリプト
    再実行（rerun）を待たずに、このボタン押下のturn内でfailed_sync_filesが更新され
    警告バナーが即座に表示される（warning_slot経由で_show_failed_sync_files_warningが
    呼ばれることの確認）。"""
    call_count = {"n": 0}

    def flaky_sync(verbose=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"added": [], "updated": [], "removed": [], "failed": []}
        return {"added": [], "updated": [], "removed": [], "failed": ["bad.pdf"]}

    monkeypatch.setattr(ingest, "sync_data_dir", flaky_sync)

    at = _run_app()
    assert at.warning == []

    resync_button = next(b for b in at.sidebar.button if "再同期" in b.label)
    at = resync_button.click().run()

    assert at.exception == []
    # 次のrerunを待たず、このボタン押下のturn内でfailed_sync_filesと警告バナーの
    # 両方が反映される。
    assert at.session_state["failed_sync_files"] == ["bad.pdf"]
    assert len(at.warning) == 1
    assert "bad.pdf" in at.warning[0].value


def test_resync_button_recovery_clears_warning_immediately(monkeypatch):
    """異常系境界値: 前回までの同期で残っていた失敗ファイルが、再同期ボタンの押下で
    0件（復旧）になった場合、warning_slotが同じturn内でクリアされ、古い警告が
    居残らない（_show_failed_sync_files_warningのcontainer.empty()呼び出しの確認）。"""
    call_count = {"n": 0}

    def flaky_sync(verbose=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"added": [], "updated": [], "removed": [], "failed": []}
        if call_count["n"] == 2:
            return {"added": [], "updated": [], "removed": [], "failed": ["bad.pdf"]}
        return {"added": ["bad.pdf"], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", flaky_sync)

    at = _run_app()
    assert at.warning == []

    resync_button = next(b for b in at.sidebar.button if "再同期" in b.label)
    at = resync_button.click().run()
    assert len(at.warning) == 1
    assert "bad.pdf" in at.warning[0].value

    resync_button = next(b for b in at.sidebar.button if "再同期" in b.label)
    at = resync_button.click().run()

    assert at.exception == []
    assert at.session_state["failed_sync_files"] == []
    # 復旧した今回のボタン押下のturn内で、warning_slotがクリアされ古い警告が残らない。
    assert at.warning == []


def test_resync_button_success_shows_no_warning(monkeypatch):
    """正常系: 再同期ボタン押下で失敗ファイルが無い場合、warning_slotを渡すようにした
    変更後も、従来通り警告は表示されずエラーも出ない。"""
    call_count = {"n": 0}

    def counting_sync(verbose=False):
        call_count["n"] += 1
        return {"added": ["ok.txt"], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    resync_button = next(b for b in at.sidebar.button if "再同期" in b.label)
    at = resync_button.click().run()

    assert at.exception == []
    assert at.error == []
    assert at.warning == []
    assert at.session_state["failed_sync_files"] == []
    assert call_count["n"] == 2


# --- 1a. Google Drive手動同期ボタン ---


def test_google_drive_sync_button_unconfigured_shows_info(monkeypatch):
    """異常系: GOOGLE_DRIVE_FOLDER_ID未設定時（sync_google_drive_files()仕様通り全キー空
    リストが返るケース）、st.infoで未設定である旨が案内され、エラーにはならない。"""
    monkeypatch.setattr(
        google_drive_sync,
        "sync_google_drive_files",
        lambda verbose=True: {"added": [], "updated": [], "removed": [], "skipped": []},
    )
    # ドキュメントも会話履歴もある状態にし、空状態ガイダンス（st.info）が余分に
    # 出現してGoogle Drive未設定案内の件数検証と混ざらないようにする。
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "dummy.txt", "chunk_count": 1}])
    monkeypatch.setattr(memory, "conversation_count", lambda thread_id=None: 1)

    at = _run_app()
    drive_button = next(b for b in at.sidebar.button if "Google Drive" in b.label)
    at = drive_button.click().run()

    assert at.exception == []
    assert at.error == []
    assert len(at.info) == 1
    assert "未設定" in at.info[0].value
    assert at.toast == []


def test_google_drive_sync_button_success_shows_both_results(monkeypatch):
    """正常系: Drive側のミラー結果（追加/更新/削除/スキップ）がトーストで通知され、
    続けて既存のDB反映（_sync_and_report経由のingest.sync_data_dir）も実行される。"""
    monkeypatch.setattr(
        google_drive_sync,
        "sync_google_drive_files",
        lambda verbose=True: {
            "added": ["doc.docx"],
            "updated": [],
            "removed": ["old.txt"],
            "skipped": [],
        },
    )
    # ドキュメントも会話履歴もある状態にし、空状態ガイダンス（st.info）が
    # at.info == [] の検証に混ざらないようにする。
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "dummy.txt", "chunk_count": 1}])
    monkeypatch.setattr(memory, "conversation_count", lambda thread_id=None: 1)

    db_sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        db_sync_calls["n"] += 1
        return {"added": ["doc.docx"], "updated": [], "removed": ["old.txt"], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    drive_button = next(b for b in at.sidebar.button if "Google Drive" in b.label)
    at = drive_button.click().run()

    assert at.exception == []
    assert at.error == []
    assert at.info == []
    # 1回目は起動時同期、2回目がボタン押下によるDB反映
    assert db_sync_calls["n"] == 2
    assert len(at.toast) == 2
    assert "追加1" in at.toast[0].value and "削除1" in at.toast[0].value
    assert "追加1" in at.toast[1].value and "削除1" in at.toast[1].value


def test_google_drive_sync_button_missing_credentials_shows_error(monkeypatch):
    """異常系: クライアントシークレットが無い場合、sync_google_drive_files()が送出する
    RuntimeErrorを捕捉し、st.errorでわかりやすく案内する（アプリはクラッシュしない）。"""

    def raise_runtime_error(verbose=True):
        raise RuntimeError("OAuthクライアントシークレットファイルが見つかりません")

    monkeypatch.setattr(google_drive_sync, "sync_google_drive_files", raise_runtime_error)

    db_sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        db_sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    drive_button = next(b for b in at.sidebar.button if "Google Drive" in b.label)
    at = drive_button.click().run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "認証情報が見つかりません" in at.error[0].value
    # Drive側の同期に失敗した場合はDB反映も行わない（起動時の1回のみ）
    assert db_sync_calls["n"] == 1


def test_google_drive_sync_button_generic_failure_shows_error(monkeypatch):
    """異常系: Drive API呼び出し自体が予期しない例外で失敗しても、st.errorのみでアプリは
    クラッシュせず継続する。"""

    def raise_generic_error(verbose=True):
        raise TimeoutError("network timeout")

    monkeypatch.setattr(google_drive_sync, "sync_google_drive_files", raise_generic_error)

    at = _run_app()
    drive_button = next(b for b in at.sidebar.button if "Google Drive" in b.label)
    at = drive_button.click().run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "Google Driveとの同期に失敗しました" in at.error[0].value
    assert "network timeout" in at.error[0].value


def test_google_drive_sync_button_db_reflection_failed_files_shows_warning(monkeypatch):
    """境界値: Drive側のミラーには成功しても、続く_sync_and_report（ingest.sync_data_dir）が
    failedを返した場合、Google Driveボタンの押下ターン内でも既存の警告バナーが
    即座に表示される（_sync_google_drive_and_reportがwarning_slotを正しく
    _sync_and_reportへ引き継いでいることの確認）。"""
    monkeypatch.setattr(
        google_drive_sync,
        "sync_google_drive_files",
        lambda verbose=True: {
            "added": ["doc.docx"],
            "updated": [],
            "removed": [],
            "skipped": [],
        },
    )

    call_count = {"n": 0}

    def flaky_sync(verbose=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"added": [], "updated": [], "removed": [], "failed": []}
        return {"added": ["doc.docx"], "updated": [], "removed": [], "failed": ["bad.pdf"]}

    monkeypatch.setattr(ingest, "sync_data_dir", flaky_sync)

    at = _run_app()
    assert at.warning == []

    drive_button = next(b for b in at.sidebar.button if "Google Drive" in b.label)
    at = drive_button.click().run()

    assert at.exception == []
    assert at.error == []
    # Drive側のミラー・DB反映それぞれの成功トーストが出つつ、DB反映がfailedを
    # 含むため警告も同時に出る（両者は独立した通知のため排他ではない）。
    assert len(at.toast) == 2
    assert at.session_state["failed_sync_files"] == ["bad.pdf"]
    assert len(at.warning) == 1
    assert "bad.pdf" in at.warning[0].value


# --- 1b. 会話履歴のウィンドウイング（長い会話でのコンテキスト長超過対策） ---


def test_windowed_history_keeps_all_messages_when_under_budget():
    """正常系: トークン予算内に収まる短い会話では、間引かれずそのまま返る。"""
    import app

    messages = [HumanMessage(content="短い質問"), AIMessage(content="短い回答")]

    assert app._windowed_history(messages) == messages


def test_windowed_history_returns_empty_list_as_is():
    """境界値: 会話履歴が空の場合はそのまま空リストを返す（trim_messages呼び出し自体を省略）。"""
    import app

    assert app._windowed_history([]) == []


def test_windowed_history_drops_oldest_messages_when_over_budget(monkeypatch):
    """異常系: トークン予算を超える長い会話では、
    古いメッセージが間引かれ、先頭が必ずHumanMessageになる。

    conftest.pyのダミー環境変数によりデフォルトのCURRENT_PROVIDERは"anthropic"
    （予算50000トークン）になっており、以下の会話量（約8000トークン相当）では
    間引きが発生しない。ここではOllama利用時の予算（デフォルト設定で約3192
    トークン）を明示的に使うことで、間引きが発生する条件を安定して再現する。"""
    import app
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")

    long_text = "あ" * 2000  # 概算で500トークン程度
    messages = []
    for i in range(8):
        messages.append(HumanMessage(content=f"質問{i}: {long_text}"))
        messages.append(AIMessage(content=f"回答{i}: {long_text}"))

    windowed = app._windowed_history(messages)

    assert len(windowed) < len(messages)
    assert isinstance(windowed[0], HumanMessage)
    # 最も古いやりとり（質問0）は予算超過のため間引かれている
    assert not any("質問0" in m.content for m in windowed)
    # 直近のやりとり（最後の質問）は残っている
    assert any(f"質問{7}" in m.content for m in windowed)


def test_windowed_history_does_not_mutate_input_list():
    """境界値（回帰防止）: 予算超過で間引きが発生しても、呼び出し元が渡した元のリスト
    （画面表示用のst.session_state.messagesを想定）自体は一切変更されない。"""
    import app

    long_text = "あ" * 2000
    messages = []
    for i in range(8):
        messages.append(HumanMessage(content=f"質問{i}: {long_text}"))
        messages.append(AIMessage(content=f"回答{i}: {long_text}"))
    original_len = len(messages)
    original_first_content = messages[0].content

    windowed = app._windowed_history(messages)

    assert len(messages) == original_len
    assert messages[0].content == original_first_content
    assert windowed is not messages


def test_chat_streaming_sends_windowed_history_to_agent(monkeypatch):
    """正常系: 会話が長くなりOllama利用時のトークン予算を超えると、
    agent.stream()に渡すメッセージ一覧が実際に間引かれ、画面表示用の
    st.session_state.messagesはフルの履歴を保持し続ける（送信分のみが絞り込まれる）。

    conftest.pyのダミー環境変数によりデフォルトのCURRENT_PROVIDERは"anthropic"
    （予算50000トークン）になっており、以下の会話量（約8000トークン相当）では
    間引きが発生しない。ここではOllama利用時の予算（デフォルト設定で約3192
    トークン）を明示的に使うことで、間引きが発生する条件を安定して再現する。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")
    fake_agent = _FakeAgent(answer="短い回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    # 概算で1メッセージあたり数百トークンになる長さの質問を複数ターン送り、
    # Ollama利用時の既定予算(約3192トークン)を確実に超えさせる。
    long_question = "あ" * 2000
    turn_count = 8
    for i in range(turn_count):
        at = at.chat_input[0].set_value(f"質問{i}: {long_question}").run()

    assert at.exception == []
    # 画面表示用の履歴はすべてのやりとりを保持している
    assert len(at.session_state["messages"]) == turn_count * 2
    # しかし直前のagent.stream()呼び出しに渡されたメッセージは間引かれ、
    # 最も古いやりとり（質問0）は含まれない
    last_call_messages = fake_agent.stream_calls[-1]["messages"]
    assert len(last_call_messages) < turn_count * 2 + 1  # 全履歴+今回の質問 より少ない
    assert not any("質問0" in getattr(m, "content", "") for m in last_call_messages)


# --- 1c. プロバイダごとのトークン予算（_history_token_budget） ---


def test_history_token_budget_anthropic_uses_api_provider_budget(monkeypatch):
    """正常系: CURRENT_PROVIDERが"anthropic"の場合、API向けの大きい予算
    （_API_PROVIDER_HISTORY_TOKENS）が使われる。"""
    import app
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "anthropic")

    assert app._history_token_budget() == app._API_PROVIDER_HISTORY_TOKENS


def test_history_token_budget_openai_uses_api_provider_budget(monkeypatch):
    """正常系: CURRENT_PROVIDERが"openai"の場合も、Anthropicと同じくAPI向けの
    大きい予算（_API_PROVIDER_HISTORY_TOKENS）が使われる。"""
    import app
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "openai")

    assert app._history_token_budget() == app._API_PROVIDER_HISTORY_TOKENS


def test_history_token_budget_ollama_derives_from_num_ctx(monkeypatch):
    """正常系: CURRENT_PROVIDERが"ollama"の場合、予算はOLLAMA_NUM_CTXから
    余白（_OLLAMA_CONTEXT_MARGIN_TOKENS）を差し引いた値になる。"""
    import app
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")
    monkeypatch.setattr(setup, "OLLAMA_NUM_CTX", 8192)

    assert app._history_token_budget() == 8192 - app._OLLAMA_CONTEXT_MARGIN_TOKENS


def test_history_token_budget_ollama_tracks_num_ctx_changes(monkeypatch):
    """正常系: OLLAMA_NUM_CTXを変更すると、Ollama利用時の予算もそれに追従する。"""
    import app
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")
    monkeypatch.setattr(setup, "OLLAMA_NUM_CTX", 20000)

    assert app._history_token_budget() == 20000 - app._OLLAMA_CONTEXT_MARGIN_TOKENS


def test_history_token_budget_ollama_has_lower_bound_for_tiny_num_ctx(monkeypatch):
    """境界値: OLLAMA_NUM_CTXが極端に小さく余白を差し引くと負値になる場合でも、
    予算は下限(_OLLAMA_MIN_HISTORY_TOKENS)を下回らない。"""
    import app
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")
    monkeypatch.setattr(setup, "OLLAMA_NUM_CTX", 100)

    assert app._history_token_budget() == app._OLLAMA_MIN_HISTORY_TOKENS


def test_history_token_budget_falls_back_when_provider_is_none(monkeypatch):
    """異常系: CURRENT_PROVIDERが未設定(None、想定外のケース)の場合、
    安全側のフォールバック予算(_FALLBACK_HISTORY_TOKENS)が使われる。"""
    import app
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", None)

    assert app._history_token_budget() == app._FALLBACK_HISTORY_TOKENS


def test_history_token_budget_falls_back_for_unexpected_provider_value(monkeypatch):
    """異常系: CURRENT_PROVIDERが既知の3値以外の想定外の文字列の場合も、
    安全側のフォールバック予算(_FALLBACK_HISTORY_TOKENS)が使われる。"""
    import app
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "unknown-provider")

    assert app._history_token_budget() == app._FALLBACK_HISTORY_TOKENS


def test_windowed_history_keeps_all_messages_for_anthropic_over_ollama_budget(monkeypatch):
    """正常系: CURRENT_PROVIDERが"anthropic"の場合、Ollama利用時の予算
    （デフォルト設定で約3192トークン）を超える会話量でも、API向けの大きい
    予算（50000トークン）の範囲内であれば間引かれず全メッセージが維持される。"""
    import app
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "anthropic")

    long_text = "あ" * 2000  # 概算で500トークン程度
    messages = []
    for i in range(8):
        messages.append(HumanMessage(content=f"質問{i}: {long_text}"))
        messages.append(AIMessage(content=f"回答{i}: {long_text}"))

    windowed = app._windowed_history(messages)

    assert windowed == messages


def test_windowed_history_ollama_budget_change_affects_drop_result(monkeypatch):
    """正常系: CURRENT_PROVIDERが"ollama"の場合、OLLAMA_NUM_CTXを大きくすると
    それに応じて予算も増え、同じ会話量でも間引きの発生有無が変わる。"""
    import app
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")

    long_text = "あ" * 2000
    messages = []
    for i in range(8):
        messages.append(HumanMessage(content=f"質問{i}: {long_text}"))
        messages.append(AIMessage(content=f"回答{i}: {long_text}"))

    # デフォルトのOLLAMA_NUM_CTX(8192、予算約3192トークン)では間引かれる
    monkeypatch.setattr(setup, "OLLAMA_NUM_CTX", 8192)
    assert len(app._windowed_history(messages)) < len(messages)

    # OLLAMA_NUM_CTXを大きくして予算(約15000トークン)が会話量を上回れば間引かれない
    monkeypatch.setattr(setup, "OLLAMA_NUM_CTX", 20000)
    assert app._windowed_history(messages) == messages


def test_windowed_history_uses_fallback_budget_when_provider_is_none(monkeypatch):
    """異常系/境界値: CURRENT_PROVIDERが未設定(None)の場合、フォールバック予算
    (_FALLBACK_HISTORY_TOKENS=3000相当、旧固定値MAX_HISTORY_TOKENSと同じ値)が
    使われ、それを超える会話では従来通り古いメッセージが間引かれる。"""
    import app
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", None)

    long_text = "あ" * 2000
    messages = []
    for i in range(8):
        messages.append(HumanMessage(content=f"質問{i}: {long_text}"))
        messages.append(AIMessage(content=f"回答{i}: {long_text}"))

    windowed = app._windowed_history(messages)

    assert len(windowed) < len(messages)
    assert isinstance(windowed[0], HumanMessage)
    assert not any("質問0" in m.content for m in windowed)
    assert any(f"質問{7}" in m.content for m in windowed)


# --- 2. チャット処理中の agent.invoke() 呼び出し ---


def test_chat_success_appends_history_and_shows_answer(monkeypatch):
    """正常系: agent.invoke() が成功すれば回答が表示され、会話履歴にも追加される。"""
    fake_agent = _FakeAgent(answer="これが回答です")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert at.error == []
    assert len(fake_agent.stream_calls) == 1
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[0].content == "質問です"
    assert messages[1].content == "これが回答です"


def test_chat_invoke_failure_shows_ollama_message_when_provider_is_ollama(monkeypatch):
    """異常系: 使用中プロバイダがOllamaの場合、従来通り
    「Ollamaサーバーに接続できません」というメッセージが表示され、会話履歴には追加されない。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")
    fake_agent = _FakeAgent(exc=ConnectionError("connection refused"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "Ollamaサーバーに接続できません" in at.error[0].value
    assert "connection refused" in at.error[0].value
    # 例外時は履歴に追加されない（answerがNoneのまま後続処理がスキップされる）
    assert at.session_state["messages"] == []


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_chat_invoke_failure_shows_api_message_when_provider_is_cloud(monkeypatch, provider):
    """異常系: 使用中プロバイダがAnthropic/OpenAIの場合、
    Ollama決め打ちの誤ったメッセージではなく、APIへの接続失敗を示す汎用的な文言が表示される。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", provider)
    fake_agent = _FakeAgent(exc=RuntimeError("invalid api key"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "Ollamaサーバーに接続できません" not in at.error[0].value
    assert "APIへの接続に失敗しました" in at.error[0].value
    assert "invalid api key" in at.error[0].value
    assert at.session_state["messages"] == []


def test_chat_invoke_failure_shows_model_not_found_message_when_provider_is_ollama(monkeypatch):
    """異常系: プロバイダがOllamaでも、例外メッセージに"model"と"not found"が
    含まれる場合はサーバー未起動ではなくモデル未pullを案内する文言に出し分ける。

    setup._ollama_model_pulled() による起動時チェックをすり抜けたケースの保険。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")
    fake_agent = _FakeAgent(exc=RuntimeError("model 'llama3.1' not found, try pulling it first"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "Ollamaサーバーに接続できません" not in at.error[0].value
    assert "見つかりません" in at.error[0].value
    assert "ollama pull" in at.error[0].value
    assert at.session_state["messages"] == []


def test_chat_invoke_failure_model_not_found_detection_is_case_insensitive(monkeypatch):
    """境界値: "Model"/"NOT FOUND"のように大文字小文字が混在していても、
    小文字化して判定しているため同じくモデル未pull案内文言になる。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")
    fake_agent = _FakeAgent(exc=RuntimeError("Model 'llama3.1' NOT FOUND"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "見つかりません" in at.error[0].value
    assert at.session_state["messages"] == []


@pytest.mark.parametrize(
    "exc_message",
    [
        "model is overloaded, try again later",  # "model"はあるが"not found"が無い
        "endpoint not found (404)",  # "not found"はあるが"model"が無い
    ],
)
def test_chat_invoke_failure_requires_both_keywords_for_model_not_found_message(monkeypatch, exc_message):
    """境界値: "model"と"not found"の両方が揃わない場合は、モデル未pull文言ではなく
    従来通りの汎用Ollama接続失敗文言のままになる（誤検出防止の回帰確認）。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")
    fake_agent = _FakeAgent(exc=RuntimeError(exc_message))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "Ollamaサーバーに接続できません" in at.error[0].value
    assert "見つかりません" not in at.error[0].value
    assert at.session_state["messages"] == []


def test_chat_invoke_failure_shows_generic_fallback_when_provider_is_none(monkeypatch):
    """境界値: CURRENT_PROVIDER が未設定（None、想定外の状態）の場合、
    Ollama決め打ちの文言にもAPI決め打ちの文言にもならず、汎用的なフォールバック文言が
    表示される（_build_model()が何らかの理由で未実行・失敗した場合の保険）。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", None)
    fake_agent = _FakeAgent(exc=RuntimeError("unexpected"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "Ollamaサーバーに接続できません" not in at.error[0].value
    assert "APIへの接続に失敗しました" not in at.error[0].value
    assert "モデルへの接続に失敗しました" in at.error[0].value
    assert "unexpected" in at.error[0].value
    assert at.session_state["messages"] == []


def test_chat_invoke_failure_skips_auto_knowledge_save(monkeypatch):
    """異常系境界値: invoke失敗時は save_conversation が呼ばれない
    （＝data/conversations/への保存自体が発生しないため、後続の同期対象にもならない）。"""
    fake_agent = _FakeAgent(exc=RuntimeError("boom"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append(a))

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 起動時の1回のみ

    at.chat_input[0].set_value("質問です").run()

    assert save_calls == []
    assert sync_calls["n"] == 1  # invoke失敗のためsave_conversationが呼ばれず、同期も増えない


# --- 2b. ストリーミング表示（agent.stream()化）固有の挙動 ---


def test_chat_streaming_chunks_are_concatenated_into_history(monkeypatch):
    """正常系: agent.stream()が回答を複数チャンクに分けて返しても、
    st.write_streamで正しく連結された1つの回答として会話履歴に保存される。"""
    fake_agent = _FakeAgent(chunks=["これ", "が", "分割された回答です"])
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert at.error == []
    assert len(fake_agent.stream_calls) == 1
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[1].content == "これが分割された回答です"


def test_chat_streaming_clears_searching_placeholder_after_first_token(monkeypatch):
    """正常系: 最初の回答トークンが届いた時点で「🔍 検索して回答を考え中...」の
    プレースホルダーが消え、最終的な画面には残らない。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert not any("検索して回答を考え中" in m.value for m in at.markdown)


def test_chat_streaming_anthropic_content_blocks_are_extracted_as_text(monkeypatch):
    """異常系回帰: Anthropicバックエンド利用時、tools bind中のChatAnthropicは
    AIMessageChunk.content を素の文字列ではなく [{"type": "text", "text": "..."}] のような
    content blocksのlistで返す。素朴な isinstance(content, str) 判定だとこの形式を
    一度も拾えず本文が空になってしまうため、AIMessageChunk.text プロパティ経由で
    text系ブロックを正しく結合できることを確認する。"""
    fake_agent = _FakeAgent(
        chunks=[
            [{"type": "text", "text": "これは"}],
            [{"type": "text", "text": "Anthropic形式の回答です"}],
        ]
    )
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert at.error == []
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[1].content == "これはAnthropic形式の回答です"


def test_chat_streaming_exception_clears_partial_answer_from_screen(monkeypatch):
    """異常系: 一部の回答チャンクを既に描画した後に例外が送出された場合、
    途中まで描画された回答テキストが画面（markdown要素）に残らず、
    st.errorのみが表示される。"""
    fake_agent = _FakeAgent(chunks=["途中まで表示された回答"], exc=RuntimeError("stream broken"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert not any("途中まで表示された回答" in m.value for m in at.markdown)


def test_chat_streaming_tool_message_artifact_becomes_sources_expander(monkeypatch):
    """正常系: ストリーミング中に届いたToolMessageのartifactが、参照元ドキュメントの
    expanderとして正しく表示される（一括invokeからstreamに変わっても
    参照元表示のロジックが壊れていないことの回帰確認）。"""
    fake_agent = _FakeAgentWithSources(answer="文書に基づく回答", artifact=[_FakeSourceDoc()])
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert len(expanders) == 1


def test_chat_streaming_sources_expander_label_shows_count(monkeypatch):
    """正常系: 参照元expanderのタイトルには件数が付き、展開する前から量が分かる
    （狭い画面幅でも不要なタップを避けやすくする表示）。"""
    doc_a = _FakeSourceDoc(page_content="Aの内容", metadata={"source": "a.txt"})
    doc_b = _FakeSourceDoc(page_content="Bの内容", metadata={"source": "b.txt"})
    fake_agent = _FakeAgentWithSources(answer="複数件ヒットした回答", artifact=[doc_a, doc_b])
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert len(expanders) == 1
    assert expanders[0].label == "参照した箇所を見る（2件）"


def test_chat_streaming_sources_expander_label_shows_singular_count(monkeypatch):
    """境界値: 参照元が1件のみの場合でも件数表示は複数形と同じ書式（1件）になる。"""
    doc_a = _FakeSourceDoc(page_content="Aの内容", metadata={"source": "a.txt"})
    fake_agent = _FakeAgentWithSources(answer="1件だけヒットした回答", artifact=[doc_a])
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert len(expanders) == 1
    assert expanders[0].label == "参照した箇所を見る（1件）"


def test_chat_streaming_source_items_rendered_in_bordered_containers(monkeypatch):
    """正常系: 各参照元は枠線付きcontainer(border=True)で区切られ、カード状の
    一覧として表示される（狭い画面幅でも項目の境界が分かりやすいようにするため）。"""
    doc_a = _FakeSourceDoc(page_content="Aの内容", metadata={"source": "a.txt"})
    doc_b = _FakeSourceDoc(page_content="Bの内容", metadata={"source": "b.txt"})
    fake_agent = _FakeAgentWithSources(answer="複数件ヒットした回答", artifact=[doc_a, doc_b])
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expander = [e for e in at.expander if "参照した箇所を見る" in e.label][0]
    item_containers = list(expander.children.values())
    assert len(item_containers) == 2
    for container in item_containers:
        assert container.proto.flex_container.border is True
        assert len(container.markdown) == 1
        assert len(container.text) == 1


def test_chat_streaming_source_item_shows_relevance_caption(monkeypatch):
    """正常系: distance_scoreを持つ参照元は、ラベル直下に関連度キャプション
    （例: "🟢 関連度: 高"）が表示される。"""
    doc_a = _FakeSourceDoc(page_content="Aの内容", metadata={"source": "a.txt", "distance_score": 0.1})
    fake_agent = _FakeAgentWithSources(answer="関連度付きの回答", artifact=[doc_a])
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expander = [e for e in at.expander if "参照した箇所を見る" in e.label][0]
    item_container = list(expander.children.values())[0]
    assert len(item_container.caption) == 1
    assert item_container.caption[0].value == "🟢 関連度: 高"


def test_chat_streaming_source_item_hides_relevance_caption_when_score_missing(monkeypatch):
    """境界値: distance_scoreを持たない参照元（例: 過去の会話ログから復元されたケース）
    では関連度キャプション自体が表示されない。"""
    doc_a = _FakeSourceDoc(page_content="Aの内容", metadata={"source": "a.txt"})
    fake_agent = _FakeAgentWithSources(answer="関連度なしの回答", artifact=[doc_a])
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expander = [e for e in at.expander if "参照した箇所を見る" in e.label][0]
    item_container = list(expander.children.values())[0]
    assert len(item_container.caption) == 0


def test_chat_streaming_multiple_source_items_show_distinct_relevance_tiers(monkeypatch):
    """正常系: 複数の参照元それぞれについて、自身のdistance_scoreに対応した
    異なる関連度ラベルが個別に表示される（他の項目の値と混同しない）。"""
    doc_high = _FakeSourceDoc(page_content="高関連度の内容", metadata={"source": "high.txt", "distance_score": 0.2})
    doc_low = _FakeSourceDoc(page_content="低関連度の内容", metadata={"source": "low.txt", "distance_score": 1.2})
    fake_agent = _FakeAgentWithSources(answer="複数件の回答", artifact=[doc_high, doc_low])
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expander = [e for e in at.expander if "参照した箇所を見る" in e.label][0]
    item_containers = list(expander.children.values())
    captions = [c.caption[0].value for c in item_containers]
    assert captions == ["🟢 関連度: 高", "🔴 関連度: 低"]


def test_chat_streaming_dedupes_sources_across_multiple_tool_calls(monkeypatch):
    """正常系: retrieve_contextが1ターン中に複数回呼ばれ、それぞれのartifactに
    (source, page, thread_id, page_content)が同じドキュメント（＝全く同じチャンク）が
    含まれていても、「参照した箇所」には重複排除された件数のみが表示される。"""
    duplicated_doc_call1 = _FakeSourceDoc(page_content="同じチャンクの内容", metadata={"source": "doc.txt"})
    duplicated_doc_call2 = _FakeSourceDoc(page_content="同じチャンクの内容", metadata={"source": "doc.txt"})
    unique_doc = _FakeSourceDoc(page_content="別の箇所", metadata={"source": "other.txt"})
    fake_agent = _FakeAgentWithMultipleToolCalls(
        answer="複数回検索した末の回答",
        artifacts=[[duplicated_doc_call1], [duplicated_doc_call2, unique_doc]],
    )
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert len(expanders) == 1
    # doc.txtは2回検索にヒットしたが1件のみ、other.txtと合わせて計2件になる
    markdown_texts = [m.value for m in expanders[0].markdown]
    assert sum("doc.txt" in text for text in markdown_texts) == 1
    assert sum("other.txt" in text for text in markdown_texts) == 1


def test_chat_streaming_no_expander_when_retrieve_context_never_called(monkeypatch):
    """境界値: retrieve_contextが1回も呼ばれずToolMessageが1件も届かない場合、
    sourcesは空のままとなり「参照した箇所を見る」expanderは表示されない。"""
    fake_agent = _FakeAgent(answer="一般知識のみによる回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert expanders == []


def test_chat_streaming_keeps_same_source_different_page_as_distinct(monkeypatch):
    """境界値: 同じsourceでもpageが異なるドキュメントは別チャンクとして扱われ、
    重複排除されずに両方とも表示される（pageがキーの一部として機能していることの確認）。"""
    page1_doc = _FakeSourceDoc(page_content="1ページ目の内容", metadata={"source": "doc.pdf", "page": 1})
    page2_doc = _FakeSourceDoc(page_content="2ページ目の内容", metadata={"source": "doc.pdf", "page": 2})
    fake_agent = _FakeAgentWithMultipleToolCalls(
        answer="複数ページを参照した回答",
        artifacts=[[page1_doc], [page2_doc]],
    )
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert len(expanders) == 1
    markdown_texts = [m.value for m in expanders[0].markdown]
    assert sum("[1]" in text or "[2]" in text for text in markdown_texts) == 2


def test_chat_streaming_dedupes_partial_overlap_across_three_tool_calls(monkeypatch):
    """境界値: retrieve_contextが3回以上呼ばれ、一部のみが重複するケースでも、
    重複した箇所のみが正しく除外され、ユニークな箇所はすべて残る。"""
    doc_a = _FakeSourceDoc(page_content="Aの内容", metadata={"source": "a.txt"})
    doc_a_dup = _FakeSourceDoc(page_content="Aの内容", metadata={"source": "a.txt"})
    doc_b = _FakeSourceDoc(page_content="Bの内容", metadata={"source": "b.txt"})
    doc_b_dup = _FakeSourceDoc(page_content="Bの内容", metadata={"source": "b.txt"})
    doc_c = _FakeSourceDoc(page_content="Cの内容", metadata={"source": "c.txt"})
    fake_agent = _FakeAgentWithMultipleToolCalls(
        answer="3回検索した末の回答",
        artifacts=[[doc_a], [doc_b, doc_a_dup], [doc_c, doc_b_dup]],
    )
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert len(expanders) == 1
    markdown_texts = [m.value for m in expanders[0].markdown]
    for name in ("a.txt", "b.txt", "c.txt"):
        assert sum(name in text for text in markdown_texts) == 1


def test_chat_streaming_no_dedupe_when_all_sources_distinct(monkeypatch):
    """正常系（回帰確認）: 複数回のretrieve_context呼び出しがすべて別々の箇所を
    ヒットさせた通常ケースでは、重複が無いため全件がそのまま表示される。"""
    doc_a = _FakeSourceDoc(page_content="Aの内容", metadata={"source": "a.txt"})
    doc_b = _FakeSourceDoc(page_content="Bの内容", metadata={"source": "b.txt"})
    doc_c = _FakeSourceDoc(page_content="Cの内容", metadata={"source": "c.txt"})
    fake_agent = _FakeAgentWithMultipleToolCalls(
        answer="3回検索した末の回答",
        artifacts=[[doc_a], [doc_b], [doc_c]],
    )
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert len(expanders) == 1
    markdown_texts = [m.value for m in expanders[0].markdown]
    for name in ("a.txt", "b.txt", "c.txt"):
        assert sum(name in text for text in markdown_texts) == 1


def test_chat_streaming_keeps_distinct_chunks_when_page_and_thread_id_both_missing(monkeypatch):
    """境界値: pageもthread_idも付与されないソース（TextLoaderで取り込む.txt/.mdなど、
    ページ番号の概念が無いファイル）から本文の異なる2チャンクがヒットした場合でも、
    重複排除キーにpage_contentを含めているため(source, page, thread_id)だけでは
    区別できなくても正しく別々のチャンクとして扱われ、どちらも脱落せずに表示される。"""
    chunk1 = _FakeSourceDoc(page_content="sample.txtの前半チャンク", metadata={"source": "data/sample.txt"})
    chunk2 = _FakeSourceDoc(
        page_content="sample.txtの後半チャンク（本文は別物）", metadata={"source": "data/sample.txt"}
    )
    fake_agent = _FakeAgentWithMultipleToolCalls(
        answer="sample.txtを参照した回答",
        artifacts=[[chunk1, chunk2]],
    )
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert len(expanders) == 1
    markdown_texts = [m.value for m in expanders[0].markdown]
    # 本文の異なる2チャンクは、page/thread_id情報が無くても両方とも残る
    assert sum("sample.txt" in text for text in markdown_texts) == 2


def test_chat_streaming_exception_after_partial_chunks_skips_history_and_save(monkeypatch):
    """異常系: 一部の回答チャンクを既にyieldした後に例外が送出された場合でも、
    部分的な回答が会話履歴に残らず、save_conversationも呼ばれない
    （answer=Noneのまま後続処理がスキップされる従来仕様の回帰確認）。"""
    fake_agent = _FakeAgent(chunks=["途中まで", "の回答"], exc=RuntimeError("stream broken"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append(a))

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert at.session_state["messages"] == []
    assert save_calls == []


def _track_button_calls(monkeypatch):
    """DeltaGenerator.button()の呼び出し(label, key)を記録するリストを返す。

    AppTestは完全に同期実行されるため、途中で描画されたボタンが最終的な
    DOM（at.button等）には残らない場合がある。button()呼び出しそのものを
    フックすることで、実際にその呼び出しが発生したかどうかを検証できる。
    """
    button_calls = []
    original_button = DeltaGenerator.button

    def _tracking_button(self, label, *args, **kwargs):
        result = original_button(self, label, *args, **kwargs)
        button_calls.append((label, kwargs.get("key"), self))
        return result

    monkeypatch.setattr(DeltaGenerator, "button", _tracking_button)
    return button_calls


def test_chat_streaming_shows_cancel_button_with_correct_key(monkeypatch):
    """正常系: ストリーミング開始直前に、正しいラベル・key（cancel_generation）で
    キャンセルボタンが描画される。真の割り込み動作（ボタン押下でスクリプトが
    中断される挙動）はAppTestが完全同期実行のため再現できないが、ボタンが
    想定通りのkeyで生成されていること自体は button() 呼び出しのフックで確認できる。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)
    button_calls = _track_button_calls(monkeypatch)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    cancel_calls = [(label, key) for label, key, _ in button_calls if key == "cancel_generation"]
    assert cancel_calls == [("⏹️ キャンセル", "cancel_generation")]


def test_chat_streaming_success_clears_cancel_button(monkeypatch):
    """正常系: 回答生成が正常に完了すると、キャンセルボタンを描画した
    プレースホルダーに対して empty() が呼ばれ、ボタンが画面から消える。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)
    button_calls = _track_button_calls(monkeypatch)

    original_empty = DeltaGenerator.empty
    cancel_cleared = {"value": False}

    def _tracking_empty(self, *args, **kwargs):
        if any(placeholder is self for _, key, placeholder in button_calls if key == "cancel_generation"):
            cancel_cleared["value"] = True
        return original_empty(self, *args, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "empty", _tracking_empty)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert cancel_cleared["value"] is True


def test_chat_streaming_exception_also_clears_cancel_button(monkeypatch):
    """異常系: ストリーム中に例外が発生した場合も、except節でキャンセルボタンの
    プレースホルダーが empty() され、押しても意味のない状態のボタンが残らない。"""
    fake_agent = _FakeAgent(chunks=["途中まで"], exc=RuntimeError("stream broken"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)
    button_calls = _track_button_calls(monkeypatch)

    original_empty = DeltaGenerator.empty
    cancel_cleared = {"value": False}

    def _tracking_empty(self, *args, **kwargs):
        if any(placeholder is self for _, key, placeholder in button_calls if key == "cancel_generation"):
            cancel_cleared["value"] = True
        return original_empty(self, *args, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "empty", _tracking_empty)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert cancel_cleared["value"] is True


# --- 3. 会話ログ保存後の挙動（save_conversation直後にadd_single_conversation_fileで即時反映） ---


def test_post_chat_saves_conversation_and_syncs_single_file_immediately(monkeypatch):
    """正常系: チャット応答後は save_conversation で保存し、その戻り値のパスを使って
    add_single_conversation_file が同じturn内で1回だけ呼ばれる（data/全件を再走査する
    sync_data_dir は呼ばれない）。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    saved_path = Path("/tmp/data/conversations/thread-test/saved.md")
    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: (save_calls.append(a), saved_path)[1])

    add_calls = []

    def fake_add_single(path):
        add_calls.append(path)
        return "added"

    monkeypatch.setattr(ingest, "add_single_conversation_file", fake_add_single)

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 起動時の1回のみ

    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert at.error == []
    assert len(save_calls) == 1
    assert add_calls == [saved_path]  # save_conversationの戻り値パスがそのまま渡される
    assert sync_calls["n"] == 1  # data/全件を再走査するsync_data_dirは呼ばれない
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[1].content == "回答"


def test_post_chat_saves_conversation_with_accumulated_sources(monkeypatch):
    """正常系: チャット応答後のsave_conversation呼び出しに、そのターンで蓄積した
    sources（Documentのリスト）がそのままキーワード引数として渡される。"""
    doc = _FakeSourceDoc(page_content="根拠の内容", metadata={"source": "doc1.txt"})
    fake_agent = _FakeAgentWithSources(answer="文書に基づく回答", artifact=[doc])
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append((a, k)) or Path("/tmp/x.md"))

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(save_calls) == 1
    _args, kwargs = save_calls[0]
    assert kwargs["sources"] == [doc]


def test_post_chat_add_single_conversation_file_success_updates_signature_immediately(monkeypatch):
    """正常系: add_single_conversation_file が成功（added/updated/unchanged）すると、
    次回のrerunを待たずに同じturn内で st.session_state.data_dir_signature が
    最新値に更新される（次回rerun時の無駄な二重同期を防ぐため）。

    data_dir_signature() は、save_conversation で会話ログファイルが実際に
    保存された後にだけ変化するようフェイク化する（起動時のトップレベルチェックの
    時点ではまだファイルが増えていないため変化なし、というシナリオを正しく再現するため）。
    """
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    saved_path = Path("/tmp/data/conversations/t/x.md")
    state = {"file_saved": False}

    def fake_save_conversation(*a, **k):
        state["file_saved"] = True
        return saved_path

    monkeypatch.setattr(memory, "save_conversation", fake_save_conversation)
    monkeypatch.setattr(ingest, "add_single_conversation_file", lambda path: "added")
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: (2, 200.0) if state["file_saved"] else (1, 100.0))

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 起動時の1回
    assert at.session_state["data_dir_signature"] == (1, 100.0)

    at = at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    # チャット送信直後のトップレベルチェック時点ではまだファイルが未保存のため
    # 追加のsync_data_dir呼び出しは発生しない。その後の会話ログ保存→
    # add_single_conversation_fileの成功により、この場でシグネチャが最新値に更新済み。
    assert sync_calls["n"] == 1
    assert at.session_state["data_dir_signature"] == (2, 200.0)

    # そのため次回run()時のトップレベルの軽量チェックは「変更なし」と判定し、
    # 追加のsync_data_dir呼び出しは発生しない（無駄な二重同期の防止）。
    at = at.run()
    assert sync_calls["n"] == 1


def test_post_chat_add_single_conversation_file_status_failed_shows_no_error_and_skips_signature_update(
    monkeypatch,
):
    """異常系境界値: add_single_conversation_file が例外を送出せず"failed"を返した場合
    （読み込み失敗等、想定内のスキップ）、st.errorは表示されず、シグネチャも更新されない。
    シグネチャを更新しないことで、次回のトップレベルの軽量チェックが引き続き
    「未反映の変更あり」と判定し、通常の全件差分同期（sync_data_dir）で再試行される。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    saved_path = Path("/tmp/data/conversations/t/x.md")
    state = {"file_saved": False}

    def fake_save_conversation(*a, **k):
        state["file_saved"] = True
        return saved_path

    monkeypatch.setattr(memory, "save_conversation", fake_save_conversation)
    monkeypatch.setattr(ingest, "add_single_conversation_file", lambda path: "failed")
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: (2, 200.0) if state["file_saved"] else (1, 100.0))

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert at.session_state["data_dir_signature"] == (1, 100.0)

    at = at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert at.error == []
    # add_single_conversation_fileが"failed"を返した場合はシグネチャを更新しないため、
    # 実際のdata_dir_signature()は変化済みでも、セッションには反映されないまま据え置かれる。
    assert at.session_state["data_dir_signature"] == (1, 100.0)
    # チャット応答自体は正常に完了している
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[1].content == "回答"

    # シグネチャが更新されなかったため、次回rerun時のトップレベルチェックが
    # 「未反映の変更あり」と判定し、通常の全件差分同期で改めて再試行される。
    at = at.run()
    assert sync_calls["n"] == 2  # 起動時の1回 + このrerunでの再試行1回


def test_post_chat_add_single_conversation_file_failed_shows_warning_immediately(monkeypatch):
    """異常系: add_single_conversation_file が"failed"を返した場合、次のスクリプト再実行
    （rerun）を待たずに、この同じturn内でfailed_sync_filesが更新され警告バナーが表示される。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    saved_path = Path("/tmp/data/conversations/t/x.md")
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: saved_path)
    monkeypatch.setattr(ingest, "add_single_conversation_file", lambda path: "failed")

    at = _run_app()
    assert at.warning == []

    at = at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    # 次のrerunを待たず、このturn内でfailed_sync_filesと警告バナーの両方が反映される。
    assert at.session_state["failed_sync_files"] == [str(saved_path)]
    assert len(at.warning) == 1
    assert str(saved_path) in at.warning[0].value


def test_post_chat_add_single_conversation_file_failed_merges_without_duplicate_warning(monkeypatch):
    """異常系境界値: 起動時の全件同期で既に失敗ファイルが残っている状態で、同じturn中に
    会話ログの単一ファイル同期も失敗した場合、failed_sync_filesは重複なくマージされ、
    警告バナーも新規に並ばず既存のプレースホルダーが更新される（1個のまま）。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    saved_path = Path("/tmp/data/conversations/t/x.md")
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: saved_path)
    monkeypatch.setattr(ingest, "add_single_conversation_file", lambda path: "failed")

    # data/自体は変化しない想定で、起動時のsync_data_dirがすでに別ファイルの
    # 失敗を検知済みという状況を再現する。
    sig_holder = {"value": (1, 100.0)}
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: sig_holder["value"])
    monkeypatch.setattr(
        ingest,
        "sync_data_dir",
        lambda verbose=False: {"added": [], "updated": [], "removed": [], "failed": ["broken.pdf"]},
    )

    at = _run_app()
    assert len(at.warning) == 1
    assert at.session_state["failed_sync_files"] == ["broken.pdf"]

    at = at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    # 起動時の失敗（broken.pdf）と今回の失敗（saved_path）がどちらも保持され、重複もない。
    assert at.session_state["failed_sync_files"] == ["broken.pdf", str(saved_path)]
    # 同じプレースホルダーへ上書きするため、警告バナーは1個のまま増えない。
    assert len(at.warning) == 1
    assert "broken.pdf" in at.warning[0].value
    assert str(saved_path) in at.warning[0].value


def test_post_chat_add_single_conversation_file_failed_uses_path_relative_to_data_dir(tmp_path, monkeypatch):
    """境界値: 保存先パスが実際にDATA_DIR配下にある場合、failed_sync_filesおよび警告バナーには
    絶対パスではなくDATA_DIRからの相対パス文字列が使われる（sync_data_dirの失敗ファイル表記と
    形式を揃えるため）。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    data_dir = tmp_path / "data"
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    saved_path = data_dir / "conversations" / "t" / "x.md"
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: saved_path)
    monkeypatch.setattr(ingest, "add_single_conversation_file", lambda path: "failed")

    at = _run_app()
    at = at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    relative_name = str(saved_path.relative_to(data_dir))
    assert at.session_state["failed_sync_files"] == [relative_name]
    assert len(at.warning) == 1
    assert relative_name in at.warning[0].value
    # 絶対パスそのままでは表示されない（相対パスに正規化されていることの確認）。
    assert str(saved_path) not in at.warning[0].value


def test_post_chat_add_single_conversation_file_exception_shows_error_and_skips_signature_update(monkeypatch):
    """異常系: add_single_conversation_file が例外を送出した場合（ロックタイムアウト等）、
    st.errorが表示され、シグネチャも更新されない。チャット応答自体はクラッシュしない。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    saved_path = Path("/tmp/data/conversations/t/x.md")
    state = {"file_saved": False}

    def fake_save_conversation(*a, **k):
        state["file_saved"] = True
        return saved_path

    monkeypatch.setattr(memory, "save_conversation", fake_save_conversation)

    def failing_add_single(path):
        raise RuntimeError("lock timeout")

    monkeypatch.setattr(ingest, "add_single_conversation_file", failing_add_single)
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: (2, 200.0) if state["file_saved"] else (1, 100.0))

    at = _run_app()
    assert at.session_state["data_dir_signature"] == (1, 100.0)

    at = at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "会話ログの保存処理でDBへの反映に失敗しました" in at.error[0].value
    assert "lock timeout" in at.error[0].value
    assert at.session_state["data_dir_signature"] == (1, 100.0)
    # 同期失敗があっても直前のチャット応答自体は履歴に残っている
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[-1].content == "回答"


class _FakeSourceDoc:
    """save_conversationのis_fallback判定テスト用の、参照元ドキュメントのフェイク。"""

    def __init__(self, page_content="参照元の内容です", metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {"source": "doc.txt"}


def test_post_chat_saves_conversation_with_is_fallback_true_when_no_sources(monkeypatch):
    """正常系: retrieve_contextが関連文書を1件も見つけられず（sourcesが空）
    一般知識で回答した場合、save_conversationはis_fallback=Trueで呼ばれる。"""
    fake_agent = _FakeAgentWithSources(answer="一般知識による回答", artifact=[])
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append((a, k)) or Path("/tmp/x.md"))

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(save_calls) == 1
    args, kwargs = save_calls[0]
    # is_fallbackはキーワード引数として渡される想定だが、位置引数で渡された場合も考慮する
    is_fallback = kwargs.get("is_fallback", args[3] if len(args) > 3 else None)
    assert is_fallback is True


def test_post_chat_saves_conversation_with_is_fallback_false_when_sources_present(monkeypatch):
    """正常系: retrieve_contextが関連文書を見つけた（sourcesが非空）場合、
    save_conversationはis_fallback=Falseで呼ばれる。"""
    fake_agent = _FakeAgentWithSources(answer="文書に基づく回答", artifact=[_FakeSourceDoc()])
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append((a, k)) or Path("/tmp/x.md"))

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(save_calls) == 1
    args, kwargs = save_calls[0]
    is_fallback = kwargs.get("is_fallback", args[3] if len(args) > 3 else None)
    assert is_fallback is False


def test_post_chat_save_conversation_not_called_when_auto_save_memory_disabled(monkeypatch):
    """境界値: 「今の会話を記憶として保存する」トグルOFFの場合は
    save_conversation が呼ばれず、それに伴う add_single_conversation_file 呼び出しも発生しない。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append(a) or Path("/tmp/x.md"))

    add_calls = []
    monkeypatch.setattr(ingest, "add_single_conversation_file", lambda path: add_calls.append(path) or "added")

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 起動時の1回

    toggle = at.sidebar.toggle[0]
    at = toggle.set_value(False).run()

    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert at.error == []
    assert save_calls == []
    assert add_calls == []
    assert sync_calls["n"] == 1  # 事後同期は呼ばれていない（data/自体に変化がないため）


# --- 4. サイドバーの記憶設定UI（会話ID表示とナレッジ化トグルの統合） ---


def test_memory_settings_expander_integrates_thread_info(monkeypatch):
    """正常系: 「🧠 記憶設定」expander内に、記憶保存トグル・保存件数・会話ID（補足情報）が
    まとめて表示され、生の会話IDそのものはユーザー向け説明文の主語になっていない
    （説明文キャプションの先頭は「今のチャット」であり、会話IDを含まない）。

    `AppTest`の`Block`（expanderを含む）は`.caption`/`.toggle`等の子要素アクセサを
    持ち、そのブロック配下のみを再帰的に収集する。ここではあえてサイドバー全体
    （`at.sidebar.caption`）ではなく取得した「🧠 記憶設定」expanderオブジェクト自身
    から辿ることで、これらの要素が実際にこの1つのexpander配下にまとまっていること
    （＝サイドバーの他の場所に分散していないこと）まで検証する。"""
    monkeypatch.setattr(memory, "new_thread_id", lambda: "abcd1234")
    monkeypatch.setattr(memory, "conversation_count", lambda thread_id: 3)

    at = _run_app()

    memory_expanders = [e for e in at.sidebar.expander if "記憶設定" in e.label]
    assert len(memory_expanders) == 1
    expander = memory_expanders[0]

    captions = [c.value for c in expander.caption]
    # 説明文（主語）に生の会話IDが含まれていないこと
    assert any("今のチャット" in c and "覚えておいて" in c for c in captions)
    assert not any(c.strip().startswith("abcd1234") for c in captions)
    # 保存件数・会話ID（補足情報としての表示）はそれぞれこのexpander配下で確認できる
    assert any("保存済みのやりとり: 3件" in c for c in captions)
    assert any("会話ID（内部識別用）: `abcd1234`" in c for c in captions)

    toggles = [t.label for t in expander.toggle]
    assert toggles == ["今の会話を記憶として保存する"]


# --- 4.5 サイドバー先頭の使用中モデル表示（current_model_label） ---


def test_sidebar_shows_current_model_label(monkeypatch):
    """正常系: サイドバー最初のcaptionに、setup.current_model_label()の結果が
    そのまま埋め込まれて表示される。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")
    monkeypatch.setattr(setup, "CURRENT_MODEL_NAME", "llama3.1")

    at = _run_app()

    first_caption = at.sidebar.caption[0].value
    assert "使用中のモデル" in first_caption
    assert "Ollama (llama3.1)" in first_caption


def test_sidebar_shows_current_model_label_for_anthropic_fallback(monkeypatch):
    """正常系: 有料APIへフォールバック中でも、警告バナーとは別に使用中モデル名が
    サイドバー先頭に表示され続ける（境界: フォールバック警告と共存する）。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "anthropic")
    monkeypatch.setattr(setup, "CURRENT_MODEL_NAME", "claude-sonnet-5")
    monkeypatch.setattr(setup, "CURRENT_PROVIDER_FALLBACK_REASON", "Ollamaに接続できません。")

    at = _run_app()

    first_caption = at.sidebar.caption[0].value
    assert "Anthropic (claude-sonnet-5)" in first_caption
    assert len(at.sidebar.warning) == 1


# --- 5. トップレベルの軽量シグネチャチェック（data_dir_signature）そのもの ---


def test_rerun_without_signature_change_skips_resync(monkeypatch):
    """正常系: data_dir_signature() の値が前回run()と変わらなければ、
    再実行しても sync_data_dir は追加で呼ばれない（静かなno-op）。"""
    sig_holder = {"value": (1, 100.0)}
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: sig_holder["value"])

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 初回起動時は必ず同期される

    at = at.run()
    at = at.run()

    assert at.exception == []
    assert sync_calls["n"] == 1  # シグネチャが変化していないため追加の同期は起きない


def test_rerun_with_signature_change_triggers_resync(monkeypatch):
    """正常系: data_dir_signature() の値が前回run()から変化していれば、
    次回のrun()で sync_data_dir が呼ばれる。"""
    sig_holder = {"value": (1, 100.0)}
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: sig_holder["value"])

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 初回起動時

    # data/にファイルが増えた・変更されたことを想定してシグネチャを変化させる
    sig_holder["value"] = (2, 200.0)
    at = at.run()

    assert at.exception == []
    assert sync_calls["n"] == 2  # 変化を検知して同期が走る

    # シグネチャがさらに変化しなければ、その後は再び呼ばれない
    at = at.run()
    assert sync_calls["n"] == 2


# --- 6. プロダクト感のあるUI/UXへのリニューアル（画面表示文言） ---


def test_doclore_branding_title_and_tagline_are_displayed():
    """正常系: タイトルが新ブランド名「Doclore」、その下にキャッチコピーが
    表示される（`st.set_page_config` の page_title/page_icon 自体は要素ツリーに
    現れないメタ情報のため、静的検証は tests/test_theme_config.py 側で行う）。"""
    at = _run_app()

    assert at.exception == []
    assert len(at.title) == 1
    assert at.title[0].value == "📖 Doclore"
    assert any(m.value == "##### あなたの資料から、迷わず答えへ。" for m in at.markdown)


def test_sidebar_document_management_heading_has_folder_icon():
    """境界値: サイドバーの「ドキュメント管理」見出しに📂アイコンが付与され、
    かつ既存の見出しテキスト自体は変わっていない（絵文字の付け忘れ・文言の
    意図しない変更の両方を検知できるようにする）。「💬 過去の会話」「📥 会話のエクスポート」
    見出しが追加された後も、サイドバー内の見出しの並び順・各文言が保たれていることを確認する。"""
    at = _run_app()

    headings = [s.value for s in at.sidebar.subheader]
    assert headings == ["💬 過去の会話", "📥 会話のエクスポート", "📂 ドキュメント管理"]


def test_chat_input_placeholder_uses_renewed_wording():
    """正常系: チャット入力欄のプレースホルダーが刷新後の文言になっている。"""
    at = _run_app()

    assert at.chat_input[0].placeholder == "資料について気になることを聞いてみましょう"


# --- 7. ファイルアップロード時の同名ファイル上書き防止 ---


def test_upload_new_file_is_saved_as_is_without_warning(tmp_path, monkeypatch):
    """正常系: data/に同名ファイルが存在しない通常のアップロードでは、
    元のファイル名のまま保存され、st.warningは表示されない。"""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)

    at = _run_app()
    at.file_uploader[0].set_value(("report.txt", b"new content", "text/plain"))
    at = at.run()

    assert at.exception == []
    assert at.error == []
    assert at.warning == []
    saved = data_dir / "report.txt"
    assert saved.exists()
    assert saved.read_bytes() == b"new content"


def test_upload_duplicate_filename_shows_warning_and_preserves_existing_file(tmp_path, monkeypatch):
    """異常系境界値: data/に既に同名ファイルが存在する場合、
    既存ファイルを上書きせず連番サフィックス付きの別名で保存し、
    st.warningでリネームされた旨をまとめて通知する。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "report.txt").write_bytes(b"existing content")
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)

    at = _run_app()
    at.file_uploader[0].set_value(("report.txt", b"new content", "text/plain"))
    at = at.run()

    assert at.exception == []
    assert at.error == []
    assert len(at.warning) == 1
    assert "report.txt" in at.warning[0].value
    assert "report (2).txt" in at.warning[0].value
    # 既存ファイルは上書きされず内容が保たれている
    assert (data_dir / "report.txt").read_bytes() == b"existing content"
    # 新しいファイルは連番サフィックス付きの別名で保存される
    assert (data_dir / "report (2).txt").read_bytes() == b"new content"


def test_upload_same_name_files_in_one_batch_shows_warning(tmp_path, monkeypatch):
    """境界値: data/上にはまだ存在しなくても、同一アップロードバッチ内に
    同名ファイルが複数含まれる場合は2件目以降が別名で保存され、st.warningが表示される。"""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)

    at = _run_app()
    at.file_uploader[0].set_value(
        [
            ("dup.txt", b"first content", "text/plain"),
            ("dup.txt", b"second content", "text/plain"),
        ]
    )
    at = at.run()

    assert at.exception == []
    assert at.error == []
    assert len(at.warning) == 1
    assert "dup.txt" in at.warning[0].value
    assert "dup (2).txt" in at.warning[0].value
    assert (data_dir / "dup.txt").read_bytes() == b"first content"
    assert (data_dir / "dup (2).txt").read_bytes() == b"second content"


def test_upload_invalid_filename_shows_error_without_warning(tmp_path, monkeypatch):
    """異常系境界値: resolve_upload_dest()がNoneを返す場合（パストラバーサルの疑い等、
    実際の再現には st.file_uploader 自身の拡張子制限を回避する必要があるため、
    ここでは resolve_upload_dest() 自体を直接フェイクにして分岐だけを検証する）、
    該当ファイルはst.errorでスキップ表示され、リネーム扱いにならないためst.warningは
    表示されない。"""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "resolve_upload_dest", lambda filename, taken_paths=None: None)

    at = _run_app()
    at.file_uploader[0].set_value(("evil.txt", b"content", "text/plain"))
    at = at.run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "不正なファイル名のためスキップしました: evil.txt" in at.error[0].value
    assert at.warning == []


def test_upload_not_reprocessed_on_unrelated_new_chat_button_rerun(tmp_path, monkeypatch):
    """異常系境界値: st.file_uploaderの値はユーザーがアップロード欄を
    操作するかページをリロードするまでセッションに保持され続けるStreamlitの仕様のため、
    「🆕 新しい会話を始める」のようなアップロードと無関係な操作で再実行されても、
    同じファイルが繰り返し保存・再インデックスされない（report (2).txt, report (3).txt ...
    と無制限に増殖しない）こと。"""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)

    at = _run_app()
    at.file_uploader[0].set_value(("report.txt", b"content", "text/plain"))
    at = at.run()

    assert at.exception == []
    assert at.error == []
    assert at.warning == []
    assert sorted(p.name for p in data_dir.iterdir()) == ["report.txt"]

    # アップロード欄を操作せず、無関係な操作（新しい会話を始める）で再実行
    new_chat_button = next(b for b in at.sidebar.button if "新しい会話" in b.label)
    at = new_chat_button.click().run()

    assert at.exception == []
    assert at.warning == []
    assert sorted(p.name for p in data_dir.iterdir()) == ["report.txt"]

    # さらにもう一度、アップロード欄に触れないまま再実行しても増殖しない
    at = at.run()

    assert at.exception == []
    assert at.warning == []
    assert sorted(p.name for p in data_dir.iterdir()) == ["report.txt"]


def test_upload_not_reprocessed_on_unrelated_chat_send_rerun(tmp_path, monkeypatch):
    """異常系境界値: チャット送信によるスクリプト再実行でも、
    アップロード欄を操作していなければ同じファイルが再保存・再インデックスされない。"""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)

    at = _run_app()
    at.file_uploader[0].set_value(("report.txt", b"content", "text/plain"))
    at = at.run()

    assert sorted(p.name for p in data_dir.iterdir()) == ["report.txt"]

    # アップロード欄を操作せず、チャット送信（無関係な操作）で再実行
    at = at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert at.warning == []
    assert sorted(p.name for p in data_dir.iterdir()) == ["report.txt"]


def test_upload_failed_sync_shows_warning_immediately(tmp_path, monkeypatch):
    """異常系: ファイルアップロード時の同期（_sync_and_report経由）が"failed"を
    返した場合も、再同期ボタンと同様に、このアップロード処理のturn内で
    failed_sync_filesが更新され警告バナーが即座に表示される（次のrerunを待たない）。"""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)

    call_count = {"n": 0}

    def flaky_sync(verbose=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"added": [], "updated": [], "removed": [], "failed": []}
        return {"added": [], "updated": [], "removed": [], "failed": ["broken.pdf"]}

    monkeypatch.setattr(ingest, "sync_data_dir", flaky_sync)

    at = _run_app()
    assert at.warning == []

    at.file_uploader[0].set_value(("broken.pdf", b"content", "application/pdf"))
    at = at.run()

    assert at.exception == []
    # アップロードされたファイル自体は正常に保存されている（同期対象の判定はモック側の
    # 都合上、必ずしもアップロードしたファイル名と一致しない）。
    assert (data_dir / "broken.pdf").exists()
    # 次のrerunを待たず、このアップロード処理のturn内でfailed_sync_filesと
    # 警告バナーの両方が反映される。
    assert at.session_state["failed_sync_files"] == ["broken.pdf"]
    assert len(at.warning) == 1
    assert "broken.pdf" in at.warning[0].value


# --- 8. 読み込み失敗ファイルの警告永続化・自動リトライ ---


def test_failed_sync_files_warning_persists_and_retries_until_fixed(monkeypatch):
    """異常系: 読み込みに失敗したファイルがdata/内に残っている間は、
    data_dir_signature()（ファイル数+最新mtimeのみを見る軽量判定）が変化しなくても、
    次回以降のスクリプト再実行のたびに自動でsync_data_dirが再試行され、
    警告も消えずに表示され続ける。修正されて同期が成功すれば警告は消え、
    以降の再実行では余計な同期が走らなくなる（元の状態に戻る）。"""
    # data/自体は一切変化していない想定でシグネチャを固定する
    sig_holder = {"value": (1, 100.0)}
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: sig_holder["value"])

    call_count = {"n": 0}

    def flaky_sync(verbose=False):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return {"added": [], "updated": [], "removed": [], "failed": ["broken.pdf"]}
        return {"added": ["broken.pdf"], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", flaky_sync)

    at = _run_app()
    assert call_count["n"] == 1
    assert len(at.warning) == 1
    assert "broken.pdf" in at.warning[0].value

    # シグネチャは変化していないが、失敗ファイルが残っているため自動的に再試行される
    at = at.run()
    assert call_count["n"] == 2
    assert len(at.warning) == 1
    assert "broken.pdf" in at.warning[0].value

    # ファイルが修正され、今回は同期に成功したとする
    at = at.run()
    assert call_count["n"] == 3
    assert at.warning == []

    # 成功後はシグネチャが保存されるため、以降のrun()では余計な同期は走らない
    at = at.run()
    assert call_count["n"] == 3


def test_sync_success_without_failed_files_updates_signature_and_no_warning(monkeypatch):
    """正常系: 失敗ファイルが無い同期では、従来通りdata_dir_signatureがセッションに
    保存され警告は表示されない。次回run()でシグネチャが変化していなければ、
    余計な同期も走らない。"""
    sig_holder = {"value": (1, 100.0)}
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: sig_holder["value"])

    call_count = {"n": 0}

    def counting_sync(verbose=False):
        call_count["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()

    assert at.exception == []
    assert at.warning == []
    assert at.session_state["data_dir_signature"] == sig_holder["value"]
    assert at.session_state["failed_sync_files"] == []

    at = at.run()
    assert call_count["n"] == 1  # シグネチャ更新済みのため2回目のrun()では同期されない
    assert at.warning == []


def test_failed_sync_files_keep_signature_unset_until_recovered(monkeypatch):
    """異常系境界値: 失敗ファイルが残っている間はdata_dir_signatureがセッションに
    保存されず、failed_sync_filesにも失敗ファイル名がそのまま保持され続ける。
    ファイルが修正されて同期が成功すると、両方とも正常時の状態にクリアされる。"""
    sig_holder = {"value": (5, 500.0)}
    monkeypatch.setattr(ingest, "data_dir_signature", lambda: sig_holder["value"])

    call_count = {"n": 0}

    def flaky_sync(verbose=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"added": [], "updated": [], "removed": [], "failed": ["corrupt.pdf"]}
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", flaky_sync)

    at = _run_app()
    assert "data_dir_signature" not in at.session_state
    assert at.session_state["failed_sync_files"] == ["corrupt.pdf"]

    # シグネチャ未保存のため、data/自体が無変化でも次回run()で自動的に再試行される
    at = at.run()
    assert call_count["n"] == 2
    assert at.session_state["data_dir_signature"] == sig_holder["value"]
    assert at.session_state["failed_sync_files"] == []
    assert at.warning == []


# --- 9. サイドバーの過去スレッド選択・再開機能 ---


def test_past_threads_empty_state_shows_caption_and_no_selectbox():
    """正常系: 過去スレッドが0件の場合、案内キャプションのみが表示され、selectboxは描画されない
    （デフォルトフィクスチャで memory.list_threads は [] を返す）。"""
    at = _run_app()

    assert at.exception == []
    assert any("まだ保存された会話スレッドはありません。" in c.value for c in at.sidebar.caption)
    assert at.sidebar.selectbox == []


def test_filter_threads_empty_keyword_returns_all_threads():
    """正常系: キーワードが空文字列（または空白のみ）の場合は絞り込まず全件返す。"""
    import app

    threads = [
        {"thread_id": "a", "first_question": "RAGとは何ですか"},
        {"thread_id": "b", "first_question": "Streamlitの使い方"},
    ]

    assert app._filter_threads(threads, "") == threads
    assert app._filter_threads(threads, "   ") == threads


def test_filter_threads_matches_case_insensitively():
    """正常系: first_questionへの部分一致（大文字小文字を区別しない）で絞り込む。"""
    import app

    threads = [
        {"thread_id": "a", "first_question": "RAGとは何ですか"},
        {"thread_id": "b", "first_question": "Streamlitの使い方"},
    ]

    assert app._filter_threads(threads, "rag") == [threads[0]]
    assert app._filter_threads(threads, "streamlit") == [threads[1]]
    assert app._filter_threads(threads, "使い方") == [threads[1]]


def test_filter_threads_no_match_returns_empty_list():
    """境界値: 一致するスレッドが無い場合は空リストを返す。"""
    import app

    threads = [{"thread_id": "a", "first_question": "RAGとは何ですか"}]

    assert app._filter_threads(threads, "存在しないキーワード") == []


def test_filter_threads_skips_threads_without_first_question():
    """境界値: first_questionがNone/空文字列のスレッドはキーワード指定時にマッチさせない。"""
    import app

    threads = [
        {"thread_id": "a", "first_question": None},
        {"thread_id": "b", "first_question": ""},
        {"thread_id": "c", "first_question": "RAGとは何ですか"},
    ]

    assert app._filter_threads(threads, "rag") == [threads[2]]


def test_filter_threads_uppercase_keyword_matches_lowercase_text():
    """境界値: キーワード側が大文字でも、first_question側が小文字でも一致する
    （大文字小文字を区別しないことをキーワード側の大文字化でも確認する）。"""
    import app

    threads = [{"thread_id": "a", "first_question": "streamlitの使い方"}]

    assert app._filter_threads(threads, "STREAMLIT") == [threads[0]]


def test_filter_threads_strips_surrounding_whitespace_before_matching():
    """境界値: キーワードの前後に空白があっても、strip後の文字列で部分一致判定する。"""
    import app

    threads = [{"thread_id": "a", "first_question": "RAGとは何ですか"}]

    assert app._filter_threads(threads, "  rag  ") == [threads[0]]


def test_filter_threads_matches_substring_in_middle_of_text():
    """境界値: 先頭・末尾ではなく文中に含まれる部分一致でもマッチする。"""
    import app

    threads = [{"thread_id": "a", "first_question": "RAGとは何ですか"}]

    assert app._filter_threads(threads, "とは") == [threads[0]]


def test_filter_threads_returns_multiple_matches_preserving_order():
    """正常系: 複数のスレッドがキーワードに一致する場合、元の順序を保ったまま全件返す。"""
    import app

    threads = [
        {"thread_id": "a", "first_question": "RAGの使い方"},
        {"thread_id": "b", "first_question": "Streamlitの使い方"},
        {"thread_id": "c", "first_question": "全く関係ない話題"},
    ]

    assert app._filter_threads(threads, "使い方") == [threads[0], threads[1]]


def test_past_threads_selectbox_shows_formatted_labels_when_threads_exist(monkeypatch):
    """正常系: 過去スレッドが存在する場合、案内キャプションの代わりにselectboxが表示され、
    各選択肢は「日時｜質問の要約（件数）」の形式でラベル付けされる。"""
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-a",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "質問A",
                "count": 2,
            },
            {
                "thread_id": "thread-b",
                "created_at": datetime(2024, 1, 2, 9, 0),
                "first_question": "質問B",
                "count": 1,
            },
        ],
    )

    at = _run_app()

    assert at.exception == []
    assert not any("まだ保存された会話スレッドはありません。" in c.value for c in at.sidebar.caption)
    assert len(at.sidebar.selectbox) == 1
    assert at.sidebar.selectbox[0].options == [
        "2024-01-01 09:00｜質問A（2件）",
        "2024-01-02 09:00｜質問B（1件）",
    ]


def test_thread_search_input_narrows_down_selectbox_options(monkeypatch):
    """正常系: サイドバーの検索テキスト入力にキーワードを入力すると、
    selectboxの選択肢がfirst_questionの部分一致で絞り込まれる。"""
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-a",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "RAGとは何ですか",
                "count": 2,
            },
            {
                "thread_id": "thread-b",
                "created_at": datetime(2024, 1, 2, 9, 0),
                "first_question": "Streamlitの使い方",
                "count": 1,
            },
        ],
    )

    at = _run_app()
    assert at.exception == []
    assert len(at.sidebar.selectbox[0].options) == 2

    at = at.sidebar.text_input(key="thread_search").set_value("RAG").run()

    assert at.exception == []
    assert at.sidebar.selectbox[0].options == ["2024-01-01 09:00｜RAGとは何ですか（2件）"]


def test_thread_search_input_no_match_shows_caption_and_no_selectbox(monkeypatch):
    """境界値: 検索キーワードに一致するスレッドが無い場合、案内キャプションが表示され
    selectboxは描画されない。"""
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-a",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "RAGとは何ですか",
                "count": 2,
            },
        ],
    )

    at = _run_app()
    at = at.sidebar.text_input(key="thread_search").set_value("存在しないキーワード").run()

    assert at.exception == []
    assert any("該当する会話スレッドが見つかりませんでした。" in c.value for c in at.sidebar.caption)
    assert at.sidebar.selectbox == []


def test_thread_search_input_whitespace_only_keyword_shows_all_threads(monkeypatch):
    """境界値: 検索欄に空白のみを入力した場合は絞り込まれず、全件がselectboxに残る。"""
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-a",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "RAGとは何ですか",
                "count": 2,
            },
            {
                "thread_id": "thread-b",
                "created_at": datetime(2024, 1, 2, 9, 0),
                "first_question": "Streamlitの使い方",
                "count": 1,
            },
        ],
    )

    at = _run_app()
    at = at.sidebar.text_input(key="thread_search").set_value("   ").run()

    assert at.exception == []
    assert not any("該当する会話スレッドが見つかりませんでした。" in c.value for c in at.sidebar.caption)
    assert len(at.sidebar.selectbox) == 1
    assert len(at.sidebar.selectbox[0].options) == 2


def test_past_thread_label_uses_saved_title_when_set(monkeypatch):
    """正常系: タイトルが設定済みのスレッドは、自動生成ラベルの代わりにタイトルを主表示にする。"""
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-a",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "質問A",
                "count": 2,
            }
        ],
    )
    monkeypatch.setattr(memory, "load_thread_title", lambda thread_id: "経費精算について")

    at = _run_app()

    assert at.exception == []
    assert at.sidebar.selectbox[0].options == ["📌 経費精算について（2024-01-01 09:00｜質問A（2件））"]


def test_past_thread_label_falls_back_to_auto_label_when_title_unset(monkeypatch):
    """正常系: タイトル未設定のスレッドは従来通り自動生成ラベルのみが使われる
    （デフォルトフィクスチャで memory.load_thread_title は None を返す）。"""
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-a",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "質問A",
                "count": 2,
            }
        ],
    )

    at = _run_app()

    assert at.exception == []
    assert at.sidebar.selectbox[0].options == ["2024-01-01 09:00｜質問A（2件）"]


def test_thread_title_edit_form_hidden_when_current_thread_has_no_saved_conversation():
    """境界値: 現在のスレッド（st.session_state.thread_id）が保存済みスレッド一覧に
    無い場合（会話ログがまだ0件の新規スレッド等）、タイトル編集フォームは表示されない
    （デフォルトフィクスチャで memory.list_threads は [] を返す）。"""
    at = _run_app()

    assert at.exception == []
    assert not any("このスレッドのタイトルを編集" in e.label for e in at.sidebar.expander)


def test_thread_title_edit_form_shown_for_current_thread(monkeypatch):
    """正常系: 現在のスレッドが保存済みスレッド一覧に含まれる場合、
    タイトル編集フォーム（テキスト入力＋保存ボタン）が表示される。"""
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-test",  # autouseフィクスチャのnew_thread_idが返す固定値
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "質問A",
                "count": 1,
            }
        ],
    )

    at = _run_app()

    assert at.exception == []
    assert any("このスレッドのタイトルを編集" in e.label for e in at.sidebar.expander)


def test_saving_thread_title_calls_save_thread_title_with_current_thread_id(monkeypatch):
    """正常系: タイトル編集フォームで保存ボタンを押すと、現在のスレッドIDと
    入力したタイトルで memory.save_thread_title() が呼ばれ、次の描画で保存完了の
    トーストが表示された上でセッション状態のメッセージがクリアされる。
    """
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-test",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "質問A",
                "count": 1,
            }
        ],
    )
    save_calls = []
    monkeypatch.setattr(memory, "save_thread_title", lambda thread_id, title: save_calls.append((thread_id, title)))

    at = _run_app()
    title_input = next(t for t in at.sidebar.text_input if t.label == "タイトル")
    title_input.set_value("新しいタイトル")
    submit_button = next(b for b in at.sidebar.button if b.label == "💾 タイトルを保存")
    at = submit_button.click().run()

    assert at.exception == []
    assert save_calls == [("thread-test", "新しいタイトル")]
    assert len(at.toast) == 1
    assert at.toast[0].value == "タイトルを保存しました"
    assert "_thread_title_saved_message" not in at.session_state


def test_thread_selectbox_shows_current_thread_preselected_with_updated_title_after_save(monkeypatch):
    """回帰防止: タイトル保存直後のrerunでも、selectboxが現在のスレッドを選択状態のまま保持し、
    表示ラベルにも保存直後の新しいタイトルが反映される。

    Streamlitのselectboxはkeyが変わらない限り閉じた状態の表示文字列を再計算しないことがあるため、
    ラベルが変わるタイミングでkeyも変え、indexで選択状態を明示的に復元することで対処している。
    """
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-test",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "質問A",
                "count": 1,
            }
        ],
    )
    current_title = {"value": None}
    monkeypatch.setattr(memory, "load_thread_title", lambda thread_id: current_title["value"])
    monkeypatch.setattr(memory, "save_thread_title", lambda thread_id, title: current_title.__setitem__("value", title))

    at = _run_app()
    title_input = next(t for t in at.sidebar.text_input if t.label == "タイトル")
    title_input.set_value("新しいタイトル")
    submit_button = next(b for b in at.sidebar.button if b.label == "💾 タイトルを保存")
    at = submit_button.click().run()

    assert at.exception == []
    assert at.sidebar.selectbox[0].value == "thread-test"
    assert "新しいタイトル" in at.sidebar.selectbox[0].options[0]


def test_past_thread_label_truncates_long_title_so_auto_label_stays_visible(monkeypatch):
    """境界値: タイトルが長い場合、selectboxの限られた幅で自動ラベルが隠れてしまわないよう
    タイトル部分を上限文字数で切り詰める。"""
    from datetime import datetime

    long_title = "経費精算まわりの質問と回答についての非常に長いタイトルの例です" * 2
    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-a",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "質問A",
                "count": 2,
            }
        ],
    )
    monkeypatch.setattr(memory, "load_thread_title", lambda thread_id: long_title)

    at = _run_app()

    assert at.exception == []
    label = at.sidebar.selectbox[0].options[0]
    assert "2024-01-01 09:00｜質問A（2件）" in label
    assert long_title not in label


def test_thread_display_label_title_of_exactly_20_chars_is_not_truncated(monkeypatch):
    """境界値: タイトルがちょうど上限文字数(20文字)の場合は切り詰めず、"..."も付かない。"""
    from datetime import datetime

    import app

    title = "あ" * 20
    monkeypatch.setattr(app, "load_thread_title", lambda thread_id: title)
    thread = {
        "thread_id": "thread-a",
        "created_at": datetime(2024, 1, 1, 9, 0),
        "first_question": "質問A",
        "count": 1,
    }

    label = app._thread_display_label(thread)

    assert f"📌 {title}" in label
    assert "..." not in label


def test_thread_display_label_title_of_21_chars_is_truncated_with_ellipsis(monkeypatch):
    """境界値: タイトルが上限文字数を1文字超える(21文字)場合、20文字に切り詰められ"..."が付く。"""
    from datetime import datetime

    import app

    title = "あ" * 21
    monkeypatch.setattr(app, "load_thread_title", lambda thread_id: title)
    thread = {
        "thread_id": "thread-a",
        "created_at": datetime(2024, 1, 1, 9, 0),
        "first_question": "質問A",
        "count": 1,
    }

    label = app._thread_display_label(thread)

    assert f"📌 {'あ' * 20}..." in label
    assert title not in label


def test_thread_display_label_emoji_title_truncates_without_raising(monkeypatch):
    """境界値: タイトルに絵文字が含まれていても例外を送出せず切り詰められる。"""
    from datetime import datetime

    import app

    title = "🎉" * 30
    monkeypatch.setattr(app, "load_thread_title", lambda thread_id: title)
    thread = {
        "thread_id": "thread-a",
        "created_at": datetime(2024, 1, 1, 9, 0),
        "first_question": "質問A",
        "count": 1,
    }

    label = app._thread_display_label(thread)

    assert label.startswith("📌 ")
    assert label.endswith("...（2024-01-01 09:00｜質問A（1件））")
    assert title not in label


def test_thread_selector_key_changes_when_active_thread_label_changes():
    """回帰防止: 同じthread_idでもラベルが変わればselectboxのwidget keyも変わる
    （タイトル保存直後にselectboxを再マウントさせて表示を最新化するため）。"""
    import app

    key_before = app._thread_selector_key("thread-a", {"thread-a": "旧ラベル"})
    key_after = app._thread_selector_key("thread-a", {"thread-a": "新ラベル"})

    assert key_before != key_after


def test_thread_selector_key_is_unique_per_thread_even_with_same_label():
    """境界値: 複数スレッドが存在する場合、ラベルが同一でもthread_idが異なればkeyも異なる。"""
    import app

    key_a = app._thread_selector_key("thread-a", {"thread-a": "同じラベル", "thread-b": "同じラベル"})
    key_b = app._thread_selector_key("thread-b", {"thread-a": "同じラベル", "thread-b": "同じラベル"})

    assert key_a != key_b


def test_thread_selector_key_falls_back_gracefully_when_active_thread_has_no_label():
    """境界値: アクティブなthread_idがthread_labelsに存在しない場合
    （タイトル未設定・新規スレッド等）でも例外を送出せず空文字列として扱う。"""
    import app

    key = app._thread_selector_key("thread-new", {"thread-a": "ラベルA"})

    assert key == "thread_selector_thread-new_"


def test_selecting_past_thread_restores_history_and_rebuilds_agent(monkeypatch):
    """正常系: 過去スレッドをselectboxで選ぶと、そのスレッドIDに切り替わり、
    load_conversationの内容がチャット履歴として復元され、エージェントも
    選択したスレッドIDで再構築される。"""
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-past",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "過去の質問",
                "count": 1,
            }
        ],
    )
    monkeypatch.setattr(
        memory,
        "load_conversation",
        lambda thread_id: (
            [{"question": "過去の質問", "answer": "過去の回答", "created_at": datetime(2024, 1, 1, 9, 0)}]
            if thread_id == "thread-past"
            else []
        ),
    )

    built_thread_ids = []

    def fake_build_agent(thread_id=None):
        built_thread_ids.append(thread_id)
        return _FakeAgent()

    monkeypatch.setattr(rag_chain, "build_agent", fake_build_agent)

    at = _run_app()
    assert at.session_state["thread_id"] == "thread-test"  # memory.new_thread_idフェイクの初期値

    at = at.sidebar.selectbox[0].select("thread-past").run()

    assert at.exception == []
    assert at.session_state["thread_id"] == "thread-past"
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[0].content == "過去の質問"
    assert messages[1].content == "過去の回答"
    assert built_thread_ids[-1] == "thread-past"


def test_selecting_currently_active_thread_again_does_not_rebuild_agent(monkeypatch):
    """境界値: 選択値が既に表示中のスレッドと同じ場合はスキップされ、
    エージェントの再構築（build_agent呼び出し）は発生しない
    （選択操作以外の理由での再実行のたびに毎回再構築されないようにするための分岐）。"""
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-past",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "過去の質問",
                "count": 1,
            }
        ],
    )
    monkeypatch.setattr(
        memory,
        "load_conversation",
        lambda thread_id: (
            [{"question": "過去の質問", "answer": "過去の回答", "created_at": datetime(2024, 1, 1, 9, 0)}]
            if thread_id == "thread-past"
            else []
        ),
    )

    built_thread_ids = []

    def fake_build_agent(thread_id=None):
        built_thread_ids.append(thread_id)
        return _FakeAgent()

    monkeypatch.setattr(rag_chain, "build_agent", fake_build_agent)

    at = _run_app()
    at = at.sidebar.selectbox[0].select("thread-past").run()
    assert built_thread_ids == ["thread-test", "thread-past"]

    # 選択状態はそのままの状態で、他の理由（ボタン押下無し）で再実行しても
    # 追加のbuild_agent呼び出しは発生しない
    at = at.run()

    assert at.exception == []
    assert built_thread_ids == ["thread-test", "thread-past"]
    assert at.session_state["thread_id"] == "thread-past"


def test_start_new_chat_resets_thread_selector_and_does_not_pull_back_to_old_thread(monkeypatch):
    """異常系境界値（回帰防止）: 過去スレッドへ切り替えた後に「🆕 新しい会話を始める」を押すと、
    新しいスレッドIDが発行され会話履歴も空になる。selectboxのウィジェット状態
    （thread_selector）がリセットされずに残っていると、次回実行時に「選択値(過去スレッド) !=
    新しいthread_id」と誤判定されて過去スレッドへ引き戻されてしまうバグの回帰を防ぐ。"""
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-past",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "過去の質問",
                "count": 1,
            }
        ],
    )
    monkeypatch.setattr(
        memory,
        "load_conversation",
        lambda thread_id: (
            [{"question": "過去の質問", "answer": "過去の回答", "created_at": datetime(2024, 1, 1, 9, 0)}]
            if thread_id == "thread-past"
            else []
        ),
    )

    id_counter = {"n": 0}

    def fake_new_thread_id():
        id_counter["n"] += 1
        return f"new-thread-{id_counter['n']}"

    monkeypatch.setattr(memory, "new_thread_id", fake_new_thread_id)
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: _FakeAgent())

    at = _run_app()
    at = at.sidebar.selectbox[0].select("thread-past").run()
    assert at.session_state["thread_id"] == "thread-past"

    new_chat_button = next(b for b in at.sidebar.button if "新しい会話" in b.label)
    at = new_chat_button.click().run()

    assert at.exception == []
    # new-thread-1は起動時のセッション初期化（_run_app内の初回run）で消費されているため、
    # 「新しい会話を始める」クリックで発行されるのは2件目のID。
    assert at.session_state["thread_id"] == "new-thread-2"
    assert at.session_state["messages"] == []


# --- 10. サイドバーのインデックス済みファイル一覧・削除機能 ---


def test_indexed_file_list_shows_empty_caption_when_no_files(monkeypatch):
    """正常系境界値: インデックス済みファイルが0件の場合、案内キャプションのみが表示され、
    削除ボタン等は一切描画されない。"""
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [])

    at = _run_app()

    assert at.exception == []
    assert any("インデックス済みのファイルはまだありません。" in c.value for c in at.sidebar.caption)
    assert [b for b in at.sidebar.button if (b.help or "").endswith("を削除")] == []


def test_indexed_file_list_shows_files_with_chunk_counts(monkeypatch):
    """正常系: インデックス済みファイルが存在する場合、件数キャプションと
    各ファイルのファイル名・チャンク数が表示される。"""
    monkeypatch.setattr(
        ingest,
        "list_indexed_files",
        lambda: [
            {"name": "a.pdf", "chunk_count": 3},
            {"name": "b.txt", "chunk_count": 1},
        ],
    )

    at = _run_app()

    assert at.exception == []
    assert any("インデックス済みファイル: 2件" in c.value for c in at.sidebar.caption)
    markdown_values = [m.value for m in at.sidebar.markdown]
    assert any("a.pdf" in v and "3チャンク" in v for v in markdown_values)
    assert any("b.txt" in v and "1チャンク" in v for v in markdown_values)
    delete_buttons = [b for b in at.sidebar.button if b.help == "a.pdf を削除"]
    assert len(delete_buttons) == 1


def test_delete_button_click_shows_confirmation_prompt(monkeypatch):
    """正常系: 削除ボタン(🗑️)を押すと、即座には削除されず確認メッセージと
    「削除する」「キャンセル」ボタンが表示される（2段階確認の1段階目）。"""
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "report.txt", "chunk_count": 2}])
    delete_calls = []
    monkeypatch.setattr(ingest, "delete_indexed_file", lambda name: delete_calls.append(name) or True)

    at = _run_app()
    delete_button = next(b for b in at.sidebar.button if b.key == "delete_button_report.txt")
    at = delete_button.click().run()

    assert at.exception == []
    assert len(at.sidebar.warning) == 1
    assert "report.txt" in at.sidebar.warning[0].value
    assert any(b.key == "confirm_delete_report.txt" for b in at.sidebar.button)
    assert any(b.key == "cancel_delete_report.txt" for b in at.sidebar.button)
    # まだ削除は実行されていない
    assert delete_calls == []


def test_confirm_delete_calls_delete_indexed_file_and_resyncs(monkeypatch):
    """正常系: 確認プロンプトで「削除する」を押すと delete_indexed_file() が実行され、
    続けて再同期（_sync_and_report経由のsync_data_dir呼び出し）が行われる。
    削除完了後は確認プロンプト自体も消える。"""
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "report.txt", "chunk_count": 2}])
    delete_calls = []
    monkeypatch.setattr(ingest, "delete_indexed_file", lambda name: delete_calls.append(name) or True)

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 起動時の1回

    delete_button = next(b for b in at.sidebar.button if b.key == "delete_button_report.txt")
    at = delete_button.click().run()

    confirm_button = next(b for b in at.sidebar.button if b.key == "confirm_delete_report.txt")
    at = confirm_button.click().run()

    assert at.exception == []
    assert delete_calls == ["report.txt"]
    assert sync_calls["n"] == 2  # 削除確定後に再同期が呼ばれる
    assert "pending_delete_report.txt" not in at.session_state
    assert at.sidebar.warning == []


def test_confirm_delete_shows_error_when_delete_indexed_file_fails(monkeypatch):
    """異常系: delete_indexed_file() がFalseを返した場合（対象ファイルが既に無い等）、
    再同期は行わずst.errorでユーザーに失敗を通知する。"""
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "report.txt", "chunk_count": 2}])
    monkeypatch.setattr(ingest, "delete_indexed_file", lambda name: False)

    sync_calls = {"n": 0}

    def counting_sync(verbose=False):
        sync_calls["n"] += 1
        return {"added": [], "updated": [], "removed": [], "failed": []}

    monkeypatch.setattr(ingest, "sync_data_dir", counting_sync)

    at = _run_app()
    assert sync_calls["n"] == 1  # 起動時の1回

    delete_button = next(b for b in at.sidebar.button if b.key == "delete_button_report.txt")
    at = delete_button.click().run()

    confirm_button = next(b for b in at.sidebar.button if b.key == "confirm_delete_report.txt")
    at = confirm_button.click().run()

    assert at.exception == []
    assert sync_calls["n"] == 1  # 削除失敗時は再同期しない
    assert "pending_delete_report.txt" not in at.session_state
    assert "report.txt" in at.sidebar.error[0].value


def test_cancel_delete_does_not_call_delete_indexed_file(monkeypatch):
    """正常系: 確認プロンプトで「キャンセル」を押すと削除は実行されず、
    確認プロンプト自体も消える。"""
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "report.txt", "chunk_count": 2}])
    delete_calls = []
    monkeypatch.setattr(ingest, "delete_indexed_file", lambda name: delete_calls.append(name) or True)

    at = _run_app()
    delete_button = next(b for b in at.sidebar.button if b.key == "delete_button_report.txt")
    at = delete_button.click().run()
    assert len(at.sidebar.warning) == 1

    cancel_button = next(b for b in at.sidebar.button if b.key == "cancel_delete_report.txt")
    at = cancel_button.click().run()

    assert at.exception == []
    assert delete_calls == []
    assert "pending_delete_report.txt" not in at.session_state
    assert at.sidebar.warning == []


# --- 10b. サイドバーのインデックス済みファイル一覧・ダウンロード機能（Issue #209） ---


def _capture_download_button_media(monkeypatch):
    """st.download_button()に渡された実データ(bytes)・ファイル名・MIMEタイプを捕捉する。

    streamlit.testing.v1.AppTestはdownload_buttonのプロトコル上、実際のバイト列を
    直接は公開しないため、内部で使われるMediaFileManager.add()を差し替えて記録する。
    """
    import streamlit.runtime.media_file_manager as media_file_manager

    captured = {}
    original_add = media_file_manager.MediaFileManager.add

    def capturing_add(self, data, mimetype, coordinates, file_name=None, is_for_static_download=False):
        captured["data"] = data
        captured["mimetype"] = mimetype
        captured["file_name"] = file_name
        return original_add(
            self, data, mimetype, coordinates, file_name=file_name, is_for_static_download=is_for_static_download
        )

    monkeypatch.setattr(media_file_manager.MediaFileManager, "add", capturing_add)
    return captured


def test_download_button_click_shows_download_button_with_file_content(tmp_path, monkeypatch):
    """正常系: ダウンロードボタン(⬇️)を押すと、対象ファイルの実体をread_bytes()で読み込んだ上で
    st.download_buttonが表示され、正しいファイル内容・ファイル名・MIMEタイプが設定される。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "report.pdf").write_bytes(b"%PDF-1.4 dummy content")
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "report.pdf", "chunk_count": 2}])
    captured = _capture_download_button_media(monkeypatch)

    at = _run_app()
    download_button = next(b for b in at.sidebar.button if b.key == "download_button_report.pdf")
    at = download_button.click().run()

    assert at.exception == []
    assert at.error == []
    confirm_buttons = [b for b in at.sidebar.download_button if b.key == "confirm_download_report.pdf"]
    assert len(confirm_buttons) == 1
    assert captured["data"] == b"%PDF-1.4 dummy content"
    assert captured["file_name"] == "report.pdf"
    assert captured["mimetype"] == "application/pdf"


def test_download_button_unknown_extension_falls_back_to_octet_stream(tmp_path, monkeypatch):
    """境界値: mimetypes.guess_type()で種類を推測できない拡張子の場合、
    MIMEタイプはapplication/octet-streamにフォールバックする。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "report.unknownext").write_bytes(b"binary-ish content")
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "report.unknownext", "chunk_count": 1}])
    captured = _capture_download_button_media(monkeypatch)

    at = _run_app()
    download_button = next(b for b in at.sidebar.button if b.key == "download_button_report.unknownext")
    at = download_button.click().run()

    assert at.exception == []
    assert captured["mimetype"] == "application/octet-stream"


def test_download_button_missing_file_shows_error_without_crashing(tmp_path, monkeypatch):
    """異常系: manifestには存在するがdata/配下に実体が無い場合（削除済み等）、
    クラッシュせずst.errorが表示され、download_buttonも表示されない。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "ghost.txt", "chunk_count": 1}])

    at = _run_app()
    download_button = next(b for b in at.sidebar.button if b.key == "download_button_ghost.txt")
    at = download_button.click().run()

    assert at.exception == []
    assert any("ghost.txt" in e.value for e in at.sidebar.error)
    assert [b for b in at.sidebar.download_button if b.key == "confirm_download_ghost.txt"] == []
    assert "pending_download_ghost.txt" not in at.session_state


def test_cancel_download_closes_download_button(tmp_path, monkeypatch):
    """正常系: ダウンロード欄の「✕」を押すとdownload_buttonが消え、
    セッションの表示状態もクリアされる。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "report.txt").write_bytes(b"hello world")
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "report.txt", "chunk_count": 2}])
    _capture_download_button_media(monkeypatch)

    at = _run_app()
    download_button = next(b for b in at.sidebar.button if b.key == "download_button_report.txt")
    at = download_button.click().run()
    assert len([b for b in at.sidebar.download_button if b.key == "confirm_download_report.txt"]) == 1

    cancel_button = next(b for b in at.sidebar.button if b.key == "cancel_download_report.txt")
    at = cancel_button.click().run()

    assert at.exception == []
    assert [b for b in at.sidebar.download_button if b.key == "confirm_download_report.txt"] == []
    assert "pending_download_report.txt" not in at.session_state


def test_download_and_delete_confirmations_are_mutually_exclusive(tmp_path, monkeypatch):
    """正常系: ダウンロード確認と削除確認は同時に開かず、片方を開くともう片方が閉じる。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "report.txt").write_bytes(b"hello world")
    monkeypatch.setattr(ingest, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "report.txt", "chunk_count": 2}])
    _capture_download_button_media(monkeypatch)

    at = _run_app()
    delete_button = next(b for b in at.sidebar.button if b.key == "delete_button_report.txt")
    at = delete_button.click().run()
    assert len(at.sidebar.warning) == 1
    assert "pending_delete_report.txt" in at.session_state

    download_button = next(b for b in at.sidebar.button if b.key == "download_button_report.txt")
    at = download_button.click().run()

    assert at.exception == []
    assert "pending_download_report.txt" in at.session_state
    assert "pending_delete_report.txt" not in at.session_state
    assert at.sidebar.warning == []
    assert len([b for b in at.sidebar.download_button if b.key == "confirm_download_report.txt"]) == 1

    delete_button = next(b for b in at.sidebar.button if b.key == "delete_button_report.txt")
    at = delete_button.click().run()

    assert at.exception == []
    assert "pending_delete_report.txt" in at.session_state
    assert "pending_download_report.txt" not in at.session_state
    assert [b for b in at.sidebar.download_button if b.key == "confirm_download_report.txt"] == []


# --- 11. build_agent()呼び出しのtry/except保護（_build_agent_safely） ---
#
# app.py はbuild_agent()を直接呼ばず、_build_agent_safely()経由で呼び出す
# （起動時・「新しい会話を始める」・過去スレッド切り替えの3箇所）。build_agent()が
# 例外を送出しても st.error 表示のみでアプリ全体をクラッシュさせず、
# st.session_state.agent は None のまま後続処理へ進む。
# さらに agent が None のままチャット送信された場合も、AttributeErrorで
# クラッシュせずst.error表示＋st.stop()で処理を打ち切る。


def test_build_agent_safely_returns_agent_on_success(monkeypatch):
    """正常系: build_agent()が成功すればそのまま返り値（エージェント）を返し、
    st.errorは呼ばれない。

    `_build_agent_safely` はモジュールトップレベルの `from rag_chain import build_agent`
    によりapp名前空間に束縛された`build_agent`を参照するため、`rag_chain.build_agent`では
    なく`app.build_agent`を直接monkeypatchする。
    """
    import app

    fake_agent = _FakeAgent()
    monkeypatch.setattr(app, "build_agent", lambda thread_id: fake_agent)

    result = app._build_agent_safely("thread-x")

    assert result is fake_agent


def test_build_agent_safely_returns_none_and_does_not_raise_on_failure(monkeypatch):
    """異常系: build_agent()が例外を送出した場合、_build_agent_safely()は例外を
    そのまま送出せずNoneを返す（st.error呼び出しの有無はAppTest経由のテストで確認する）。"""
    import app

    def failing_build_agent(thread_id):
        raise RuntimeError("boom")

    monkeypatch.setattr(app, "build_agent", failing_build_agent)

    result = app._build_agent_safely("thread-x")

    assert result is None


def test_startup_build_agent_failure_shows_error_and_agent_is_none(monkeypatch):
    """異常系: 起動時のbuild_agent()呼び出しが例外を送出しても、
    st.errorのみが表示されクラッシュせず、st.session_state.agentはNoneのまま
    後続処理（メッセージ初期化等）が継続される。"""

    def failing_build_agent(thread_id=None):
        raise RuntimeError("agent build boom")

    monkeypatch.setattr(rag_chain, "build_agent", failing_build_agent)

    at = _run_app()

    assert at.exception == []
    assert len(at.error) == 1
    assert "RAGエージェントの初期化に失敗しました" in at.error[0].value
    assert "agent build boom" in at.error[0].value
    assert "agent" in at.session_state
    assert at.session_state["agent"] is None
    # agent構築が失敗しても、その後段のmessages初期化などは通常通り継続される
    assert at.session_state["messages"] == []


def test_start_new_chat_build_agent_failure_sets_agent_none(monkeypatch):
    """異常系: 「🆕 新しい会話を始める」押下時にbuild_agent()が失敗しても、
    クラッシュせずst.session_state.agentはNoneになる（後続のチャット送信時の
    agent Noneガードによってクラッシュが防がれる、というセーフティネットの本体を検証する）。

    _start_new_chat() 内の st.error() 呼び出し後、st.session_state.agentがNoneの
    場合はst.rerun()を呼ばない（呼び出し側のガード条件による）ため、その場の
    スクリプト実行がそのまま最後まで進み、st.error()の描画がAppTest.run()が返す
    最終的な木にも残る。at.errorにエラーメッセージが実際に残っていることも
    含めてこの実際の挙動を回帰確認用に固定する。
    """
    at = _run_app()
    assert at.session_state["agent"] is not None

    def failing_build_agent(thread_id=None):
        raise RuntimeError("new chat agent boom")

    monkeypatch.setattr(rag_chain, "build_agent", failing_build_agent)

    new_chat_button = next(b for b in at.sidebar.button if "新しい会話" in b.label)
    at = new_chat_button.click().run()

    assert at.exception == []
    assert at.session_state["agent"] is None
    assert len(at.error) == 1
    assert "RAGエージェントの初期化に失敗しました" in at.error[0].value
    assert "new chat agent boom" in at.error[0].value


def test_switch_thread_build_agent_failure_sets_agent_none(monkeypatch):
    """異常系: 過去スレッドへの切り替え時にbuild_agent()が失敗しても、
    st.errorのみが表示されクラッシュせず、st.session_state.agentはNoneになる
    （会話履歴の復元自体は先に完了しているため、messagesは維持される）。

    _switch_thread() 内の st.error() 呼び出し後、st.session_state.agentがNoneの
    場合はst.rerun()を呼ばないため、_start_new_chat()の場合と同様にst.error()の
    描画が最終的な木にも残る（詳細は同ファイル内の
    test_start_new_chat_build_agent_failure_sets_agent_none のdocstring参照）。
    """
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-past",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "過去の質問",
                "count": 1,
            }
        ],
    )
    monkeypatch.setattr(
        memory,
        "load_conversation",
        lambda thread_id: (
            [{"question": "過去の質問", "answer": "過去の回答", "created_at": datetime(2024, 1, 1, 9, 0)}]
            if thread_id == "thread-past"
            else []
        ),
    )

    at = _run_app()

    def failing_build_agent(thread_id=None):
        raise RuntimeError("switch thread agent boom")

    monkeypatch.setattr(rag_chain, "build_agent", failing_build_agent)

    at = at.sidebar.selectbox[0].select("thread-past").run()

    assert at.exception == []
    assert at.session_state["thread_id"] == "thread-past"
    assert at.session_state["agent"] is None
    assert len(at.error) == 1
    assert "RAGエージェントの初期化に失敗しました" in at.error[0].value
    assert "switch thread agent boom" in at.error[0].value
    # 履歴の復元自体は agent 構築より前に完了しているため維持される
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[0].content == "過去の質問"


def test_chat_with_none_agent_shows_error_and_stops_without_crash(monkeypatch):
    """異常系: build_agent()の失敗によりst.session_state.agentがNoneのまま
    チャットが送信された場合、agent.stream()を呼びに行かずAttributeErrorで
    クラッシュすることもなく、st.errorを表示してst.stop()で処理を打ち切る
    （会話履歴への追加やsave_conversationも行われない）。"""

    def failing_build_agent(thread_id=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(rag_chain, "build_agent", failing_build_agent)

    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append(a))

    at = _run_app()
    assert at.session_state["agent"] is None

    at = at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    # 起動時のRAGエージェント初期化失敗によるst.errorは、agentが既にsession_stateに
    # 存在する（Noneのまま）ため次のrun()では再構築処理自体を通らず表示されない。
    # このrun()で新たに表示されるのはチャット処理冒頭のagent Noneガードによる
    # st.errorのみ（st.stop()の直前に呼ばれ、その後st.rerun()は発生しないため最終的な
    # 木にそのまま残る）。
    assert len(at.error) == 1
    assert "RAGエージェントが利用できないため、回答を生成できません" in at.error[0].value
    assert at.session_state["messages"] == []
    assert save_calls == []


def test_chat_with_none_agent_after_recovery_still_shows_error_for_that_turn(monkeypatch):
    """境界値: 一度build_agent()が成功しエージェントが構築された状態から、
    何らかの理由でst.session_state.agentが明示的にNoneへ書き換わっていた場合でも
    （防御的ガードそのものの検証。実運用ではbuild_agent失敗時のみ発生する想定）、
    チャット処理はクラッシュせずガードで打ち切られる。"""
    at = _run_app()
    assert at.session_state["agent"] is not None

    at.session_state["agent"] = None
    at = at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "RAGエージェントが利用できないため、回答を生成できません" in at.error[0].value
    assert at.session_state["messages"] == []


# --- 12. 有料APIへのフォールバック時の起動時警告バナー ---


def test_provider_fallback_warning_hidden_when_ollama(monkeypatch):
    """正常系: 使用中プロバイダがOllamaの場合、有料API利用中の警告バナーは表示されない。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")

    at = _run_app()

    assert at.exception == []
    assert not any("有料API" in w.value for w in at.sidebar.warning)


def test_provider_fallback_warning_hidden_when_provider_unset(monkeypatch):
    """境界値: CURRENT_PROVIDER が未設定（None、想定外の状態）の場合も、
    誤った警告は表示しない。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", None)

    at = _run_app()

    assert at.exception == []
    assert not any("有料API" in w.value for w in at.sidebar.warning)


@pytest.mark.parametrize(("provider", "provider_label"), [("anthropic", "Anthropic"), ("openai", "OpenAI")])
def test_provider_fallback_warning_shown_with_reason_when_using_paid_api(monkeypatch, provider, provider_label):
    """異常系: Ollamaが利用できず有料APIへフォールバックした場合、サイドバーに
    プロバイダ名を含む短い警告バナーが表示され、フォールバック理由はcaptionに分離される
    （警告本文が長文化してサイドバーが縦に間延びしないようにするため）。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", provider)
    monkeypatch.setattr(setup, "CURRENT_PROVIDER_FALLBACK_REASON", "Ollamaサーバーに接続できません（未起動の可能性）。")

    at = _run_app()

    assert at.exception == []
    fallback_warnings = [w.value for w in at.sidebar.warning if "有料API" in w.value]
    assert len(fallback_warnings) == 1
    assert provider_label in fallback_warnings[0]
    assert "Ollamaサーバーに接続できません" not in fallback_warnings[0]
    reason_captions = [c.value for c in at.sidebar.caption if "Ollamaサーバーに接続できません" in c.value]
    assert len(reason_captions) == 1


def test_provider_fallback_warning_shown_without_reason_when_reason_missing(monkeypatch):
    """境界値: フォールバック理由が記録されていない（想定外）場合でも、
    警告バナー自体はプロバイダ名付きで表示される。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "anthropic")
    monkeypatch.setattr(setup, "CURRENT_PROVIDER_FALLBACK_REASON", None)

    at = _run_app()

    assert at.exception == []
    fallback_warnings = [w.value for w in at.sidebar.warning if "有料API" in w.value]
    assert len(fallback_warnings) == 1
    assert "Anthropic" in fallback_warnings[0]
    assert not any("理由:" in c.value for c in at.sidebar.caption)


def test_provider_fallback_warning_unknown_provider_falls_back_to_raw_string(monkeypatch):
    """境界値: CURRENT_PROVIDER が _PROVIDER_LABELS に存在しない想定外の文字列でも、
    KeyErrorにならずプロバイダ名をそのまま表示する。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "mystery-provider")
    monkeypatch.setattr(setup, "CURRENT_PROVIDER_FALLBACK_REASON", None)

    at = _run_app()

    assert at.exception == []
    fallback_warnings = [w.value for w in at.sidebar.warning if "有料API" in w.value]
    assert len(fallback_warnings) == 1
    assert "mystery-provider" in fallback_warnings[0]


def test_provider_fallback_warning_empty_reason_string_hides_caption(monkeypatch):
    """境界値: フォールバック理由が空文字列の場合も、Noneの場合と同様にcaptionを出さない。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "openai")
    monkeypatch.setattr(setup, "CURRENT_PROVIDER_FALLBACK_REASON", "")

    at = _run_app()

    assert at.exception == []
    fallback_warnings = [w.value for w in at.sidebar.warning if "有料API" in w.value]
    assert len(fallback_warnings) == 1
    assert "OpenAI" in fallback_warnings[0]
    assert not any("理由:" in c.value for c in at.sidebar.caption)


# --- 13. サイドバーの会話エクスポート（ダウンロード）ボタン ---


def test_conversation_to_markdown_formats_question_and_answer_pairs():
    """正常系: 質問・回答のペアがMarkdownの見出し付きで、スレッドIDと共に整形される。"""
    import app

    messages = [
        HumanMessage(content="1つ目の質問"),
        AIMessage(content="1つ目の回答"),
        HumanMessage(content="2つ目の質問"),
        AIMessage(content="2つ目の回答"),
    ]

    markdown = app._conversation_to_markdown(messages, "thread-test")

    assert "スレッドID: thread-test" in markdown
    assert "## 質問 1\n\n1つ目の質問" in markdown
    assert "## 回答 1\n\n1つ目の回答" in markdown
    assert "## 質問 2\n\n2つ目の質問" in markdown
    assert "## 回答 2\n\n2つ目の回答" in markdown


def test_conversation_to_markdown_empty_messages_still_includes_thread_id():
    """境界値: メッセージが0件でも例外にならず、スレッドIDのみ含むMarkdownを返す。"""
    import app

    markdown = app._conversation_to_markdown([], "thread-test")

    assert "スレッドID: thread-test" in markdown
    assert "## 質問" not in markdown


def test_export_download_button_disabled_when_no_messages():
    """正常系: 会話がまだ始まっていない（messagesが空）場合、エクスポートボタンは無効化される。"""
    at = _run_app()

    assert at.exception == []
    buttons = [b for b in at.sidebar.download_button if "会話をダウンロード" in b.label]
    assert len(buttons) == 1
    assert buttons[0].proto.disabled is True


def test_export_download_button_enabled_after_chat(monkeypatch):
    """正常系: チャットのやり取りが発生すると、エクスポートボタンが有効化される。"""
    fake_agent = _FakeAgent(answer="これが回答です")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()
    # サイドバーはチャット処理より前に描画されるため、その場の実行では
    # まだ更新前のmessagesを参照している。次のスクリプト再実行で反映を確認する。
    at = at.run()

    assert at.exception == []
    buttons = [b for b in at.sidebar.download_button if "会話をダウンロード" in b.label]
    assert len(buttons) == 1
    assert buttons[0].proto.disabled is False


def test_conversation_to_markdown_single_pair_is_labeled_1():
    """境界値: 質問・回答が1往復のみ（メッセージ2件）の最小構成でも正しく整形される。"""
    import app

    messages = [HumanMessage(content="質問"), AIMessage(content="回答")]

    markdown = app._conversation_to_markdown(messages, "thread-min")
    lines = markdown.splitlines()

    assert "- スレッドID: thread-min" in lines
    assert lines.count("## 質問 1") == 1
    assert lines.count("## 回答 1") == 1
    assert "## 質問 2" not in markdown
    assert "## 回答 2" not in markdown
    assert markdown.index("## 質問 1") < markdown.index("## 回答 1")


def test_conversation_to_markdown_many_pairs_numbers_sequentially():
    """境界値: 10往復（20件）の大量メッセージでも見出し番号が1件ずつずれずに連番になる。"""
    import app

    messages = []
    for i in range(1, 11):
        messages.append(HumanMessage(content=f"質問その{i}"))
        messages.append(AIMessage(content=f"回答その{i}"))

    markdown = app._conversation_to_markdown(messages, "thread-many")

    for i in range(1, 11):
        assert f"## 質問 {i}\n\n質問その{i}" in markdown
        assert f"## 回答 {i}\n\n回答その{i}" in markdown
    # 番号が重複・飛び番になっていないことも確認する（「質問 1」が「質問 10」に部分一致しないよう空白込みで数える）。
    assert markdown.count("## 質問 1\n") == 1
    assert markdown.count("## 回答 1\n") == 1


def test_conversation_to_markdown_preserves_markdown_syntax_and_newlines_verbatim():
    """異常系（想定外入力）: メッセージ本文にMarkdown記法や改行が含まれてもエスケープされず、
    そのまま保持される（意図的にエスケープしない実装のため、崩れず素通しされることを確認する）。"""
    import app

    tricky_question = "見出し風の質問です\n## 回答 99\n- 箇条書きも含む"
    tricky_answer = "コードブロックと**強調**を含む回答\n```python\nprint('hi')\n```\n[リンク](https://example.com)"
    messages = [HumanMessage(content=tricky_question), AIMessage(content=tricky_answer)]

    markdown = app._conversation_to_markdown(messages, "thread-tricky")

    lines = markdown.splitlines()
    # "## 質問 1" という見出し行自体は1つだけ存在し、本文中に紛れ込んだ
    # "## 回答 99" のような文字列が誤って見出し扱いされていないことを確認する。
    assert lines.count("## 質問 1") == 1
    assert lines.count("## 回答 1") == 1
    assert tricky_question in markdown
    assert tricky_answer in markdown


def test_conversation_to_markdown_export_timestamp_is_parseable():
    """境界値: エクスポート日時の行が %Y-%m-%d %H:%M:%S 形式でパース可能であることを確認する。"""
    from datetime import datetime

    import app

    markdown = app._conversation_to_markdown([], "thread-ts")

    timestamp_line = next(line for line in markdown.splitlines() if line.startswith("- エクスポート日時: "))
    timestamp_str = timestamp_line.removeprefix("- エクスポート日時: ")
    # フォーマット不一致なら ValueError が送出されテストが失敗する。
    datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")


def test_conversation_to_markdown_with_realistic_thread_id(monkeypatch):
    """正常系: memory.new_thread_id() が生成する実際の形式（8桁hex）のスレッドIDでも問題なく整形される。

    memory.new_thread_id は全テスト共通のautouseフィクスチャで固定値にフェイクされているため、
    ここでは一時的に元の実装へ戻して実際の生成形式を使う。
    """
    import app

    monkeypatch.undo()
    thread_id = memory.new_thread_id()

    markdown = app._conversation_to_markdown([HumanMessage(content="質問")], thread_id)

    assert f"スレッドID: {thread_id}" in markdown
    assert len(thread_id) == 8


def test_conversation_to_markdown_trailing_unanswered_question_numbered_correctly():
    """境界値: 最後の質問に回答がまだ無い（奇数件）場合でも、末尾の質問の番号がずれない。"""
    import app

    messages = [
        HumanMessage(content="質問1"),
        AIMessage(content="回答1"),
        HumanMessage(content="質問2"),
    ]

    markdown = app._conversation_to_markdown(messages, "thread-odd")

    assert "## 質問 1\n\n質問1" in markdown
    assert "## 回答 1\n\n回答1" in markdown
    assert "## 質問 2\n\n質問2" in markdown


# --- 14. 過去ターンの参照元expanderの永続化（additional_kwargs["sources"]） ---


class _FakeAgentPerTurn:
    """呼び出しターンごとに異なる回答・参照元artifactを返すフェイクエージェント。

    複数ターンにわたってsourcesが正しく紐づくこと（ターン間で混ざらないこと）を
    検証するため、turnsで渡した(answer, artifact)のタプルをstream()呼び出し順に消費する。
    """

    def __init__(self, turns):
        self.turns = turns
        self.call_count = 0

    def stream(self, payload, stream_mode="messages"):
        answer, artifact = self.turns[self.call_count]
        self.call_count += 1
        if artifact:
            tool_message = ToolMessage(content="検索結果", artifact=artifact, tool_call_id=f"call-{self.call_count}")
            yield tool_message, {}
        yield AIMessageChunk(content=answer), {}


def test_sources_expander_persists_after_next_turn_rerun(monkeypatch):
    """正常系: 1ターン目でsources付きの回答を得た後、2ターン目（sourcesなし）の
    質問を送信して画面が再描画されても、1ターン目の参照元expanderが引き続き表示される
    （additional_kwargs["sources"]でst.session_state.messages自体に永続化されているため）。"""
    turn1_doc = _FakeSourceDoc(page_content="1ターン目の参照内容", metadata={"source": "turn1.txt"})
    fake_agent = _FakeAgentPerTurn(
        turns=[
            ("1ターン目の回答", [turn1_doc]),
            ("2ターン目の回答", []),
        ]
    )
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at = at.chat_input[0].set_value("1ターン目の質問").run()

    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert len(expanders) == 1
    assert any("turn1.txt" in m.value for m in expanders[0].markdown)

    # 2ターン目を送信し、画面全体が再描画された後も1ターン目のexpanderが消えない
    at = at.chat_input[0].set_value("2ターン目の質問").run()

    assert at.exception == []
    expanders_after = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert len(expanders_after) == 1
    assert any("turn1.txt" in m.value for m in expanders_after[0].markdown)

    messages = at.session_state["messages"]
    assert len(messages) == 4
    turn1_ai_message = messages[1]
    turn2_ai_message = messages[3]
    assert len(turn1_ai_message.additional_kwargs.get("sources")) == 1
    assert turn1_ai_message.additional_kwargs["sources"][0].metadata["source"] == "turn1.txt"
    assert turn2_ai_message.additional_kwargs.get("sources") == []


def test_sources_expander_not_shown_when_turn_has_no_sources(monkeypatch):
    """正常系: sourcesが無い（一般知識フォールバック等でsourcesが空の）ターンでは、
    回答直後・再描画のいずれのタイミングでもexpanderが表示されない。"""
    fake_agent = _FakeAgent(answer="一般知識による回答（根拠文書なし）")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at = at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert expanders == []

    messages = at.session_state["messages"]
    assert messages[1].additional_kwargs.get("sources") == []

    # 無関係な操作（サイドバートグル）による再描画後も、expanderは表示されないまま
    toggle = at.sidebar.toggle[0]
    at = toggle.set_value(False).run()

    assert at.exception == []
    expanders_after = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert expanders_after == []


def test_sources_do_not_mix_across_multiple_turns(monkeypatch):
    """境界値: 複数ターン（3ターン）にわたって、各ターンごとに異なるsourcesが正しく
    紐づいて表示され、ターン間でsourcesが混ざらない
    （例: 2ターン目のexpanderに1ターン目や3ターン目のドキュメントが混入しない）。"""
    doc1 = _FakeSourceDoc(page_content="ターン1の内容", metadata={"source": "doc1.txt"})
    doc2 = _FakeSourceDoc(page_content="ターン2の内容", metadata={"source": "doc2.txt"})
    doc3 = _FakeSourceDoc(page_content="ターン3の内容", metadata={"source": "doc3.txt"})
    fake_agent = _FakeAgentPerTurn(
        turns=[
            ("ターン1の回答", [doc1]),
            ("ターン2の回答", [doc2]),
            ("ターン3の回答", [doc3]),
        ]
    )
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at = at.chat_input[0].set_value("質問1").run()
    at = at.chat_input[0].set_value("質問2").run()
    at = at.chat_input[0].set_value("質問3").run()

    assert at.exception == []
    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert len(expanders) == 3

    for expander, expected_doc, other_docs in (
        (expanders[0], "doc1.txt", ("doc2.txt", "doc3.txt")),
        (expanders[1], "doc2.txt", ("doc1.txt", "doc3.txt")),
        (expanders[2], "doc3.txt", ("doc1.txt", "doc2.txt")),
    ):
        markdown_texts = [m.value for m in expander.markdown]
        assert any(expected_doc in text for text in markdown_texts)
        for other in other_docs:
            assert not any(other in text for text in markdown_texts)

    messages = at.session_state["messages"]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    assert len(ai_messages) == 3
    assert ai_messages[0].additional_kwargs["sources"][0].metadata["source"] == "doc1.txt"
    assert ai_messages[1].additional_kwargs["sources"][0].metadata["source"] == "doc2.txt"
    assert ai_messages[2].additional_kwargs["sources"][0].metadata["source"] == "doc3.txt"


def test_switching_to_past_thread_with_legacy_ai_message_does_not_crash(monkeypatch):
    """異常系（回帰防止）: load_conversation()が"sources"キーを持たない旧形式の要素を
    返しても、_switch_thread()はクラッシュせず空リストとして扱い、参照元expanderも
    表示されない。"""
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-past",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "過去の質問",
                "count": 1,
            }
        ],
    )
    monkeypatch.setattr(
        memory,
        "load_conversation",
        lambda thread_id: (
            [{"question": "過去の質問", "answer": "過去の回答", "created_at": datetime(2024, 1, 1, 9, 0)}]
            if thread_id == "thread-past"
            else []
        ),
    )

    at = _run_app()
    at = at.sidebar.selectbox[0].select("thread-past").run()

    assert at.exception == []
    messages = at.session_state["messages"]
    assert len(messages) == 2
    restored_ai_message = messages[1]
    assert restored_ai_message.additional_kwargs.get("sources") == []

    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert expanders == []


def test_switching_to_past_thread_restores_sources_expander(monkeypatch):
    """正常系（回帰確認）: load_conversation()がsources付きの要素を返す場合、
    _switch_thread()経由で復元したAIMessageでも参照元expanderが表示される。"""
    from datetime import datetime

    doc = _FakeSourceDoc(page_content="過去ターンの参照内容", metadata={"source": "past.txt"})

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-past",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "過去の質問",
                "count": 1,
            }
        ],
    )
    monkeypatch.setattr(
        memory,
        "load_conversation",
        lambda thread_id: (
            [
                {
                    "question": "過去の質問",
                    "answer": "過去の回答",
                    "created_at": datetime(2024, 1, 1, 9, 0),
                    "sources": [doc],
                }
            ]
            if thread_id == "thread-past"
            else []
        ),
    )

    at = _run_app()
    at = at.sidebar.selectbox[0].select("thread-past").run()

    assert at.exception == []
    expanders = [e for e in at.expander if "参照した箇所を見る" in e.label]
    assert len(expanders) == 1
    assert any("past.txt" in m.value for m in expanders[0].markdown)

    messages = at.session_state["messages"]
    ai_message = [m for m in messages if isinstance(m, AIMessage)][0]
    assert ai_message.additional_kwargs["sources"] == [doc]


# --- 15. 回答の根拠バッジ（ドキュメント根拠 / 一般知識） ---


def test_answer_badge_shows_document_based_when_sources_present(monkeypatch):
    """正常系: sourcesが非空の回答には「🔍 ドキュメントに基づく回答」バッジが表示され、
    「🧠 一般知識による回答」バッジは表示されない。"""
    fake_agent = _FakeAgentWithSources(answer="文書に基づく回答", artifact=[_FakeSourceDoc()])
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    caption_texts = [c.value for c in at.caption]
    assert any("🔍 ドキュメントに基づく回答" in text for text in caption_texts)
    assert not any("🧠 一般知識による回答" in text for text in caption_texts)


def test_answer_badge_shows_general_knowledge_when_no_sources(monkeypatch):
    """正常系: sourcesが空の回答には「🧠 一般知識による回答（ドキュメントに該当情報なし）」
    バッジが表示され、「🔍 ドキュメントに基づく回答」バッジは表示されない。"""
    fake_agent = _FakeAgent(answer="一般知識のみによる回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    caption_texts = [c.value for c in at.caption]
    assert any("🧠 一般知識による回答（ドキュメントに該当情報なし）" in text for text in caption_texts)
    assert not any("🔍 ドキュメントに基づく回答" in text for text in caption_texts)


def test_answer_badge_persists_after_next_turn_rerun(monkeypatch):
    """正常系: 1ターン目（sources有り）の後に2ターン目（sources無し）を送信して
    画面が再描画されても、各ターンの回答直後に判定したバッジが両方とも残る
    （additional_kwargs["sources"]から都度再判定されるため）。"""
    turn1_doc = _FakeSourceDoc(page_content="1ターン目の参照内容", metadata={"source": "turn1.txt"})
    fake_agent = _FakeAgentPerTurn(
        turns=[
            ("1ターン目の回答", [turn1_doc]),
            ("2ターン目の回答", []),
        ]
    )
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at = at.chat_input[0].set_value("1ターン目の質問").run()
    at = at.chat_input[0].set_value("2ターン目の質問").run()

    assert at.exception == []
    caption_texts = [c.value for c in at.caption]
    assert sum("🔍 ドキュメントに基づく回答" in text for text in caption_texts) == 1
    assert sum("🧠 一般知識による回答（ドキュメントに該当情報なし）" in text for text in caption_texts) == 1


def test_answer_badge_shows_general_knowledge_for_legacy_message_without_sources_key(monkeypatch):
    """異常系（回帰防止）: load_conversation()が"sources"キーを持たない旧形式の要素を
    返す場合（_switch_thread()経由で復元されたもの）でも、クラッシュせず「一般知識」
    バッジとして扱われる。"""
    from datetime import datetime

    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-past",
                "created_at": datetime(2024, 1, 1, 9, 0),
                "first_question": "過去の質問",
                "count": 1,
            }
        ],
    )
    monkeypatch.setattr(
        memory,
        "load_conversation",
        lambda thread_id: (
            [{"question": "過去の質問", "answer": "過去の回答", "created_at": datetime(2024, 1, 1, 9, 0)}]
            if thread_id == "thread-past"
            else []
        ),
    )

    at = _run_app()
    at = at.sidebar.selectbox[0].select("thread-past").run()

    assert at.exception == []
    caption_texts = [c.value for c in at.caption]
    assert any("🧠 一般知識による回答（ドキュメントに該当情報なし）" in text for text in caption_texts)


# --- 16. チャット送信・過去スレッド復元時のタイムスタンプ表示 ---


def test_format_message_timestamp_same_day_returns_time_only():
    """正常系: 今日の日時を渡すと、日付を省いた"HH:MM"形式の文字列を返す。"""
    from datetime import datetime

    import app

    today = datetime.now().date()
    timestamp = datetime(today.year, today.month, today.day, 14, 32)

    assert app._format_message_timestamp(timestamp) == "14:32"


def test_format_message_timestamp_different_day_returns_date_and_time():
    """正常系: 今日と異なる日付を渡すと、"MM/DD HH:MM"形式で日付も添えて返す。"""
    from datetime import datetime, timedelta

    import app

    yesterday = datetime.now().date() - timedelta(days=1)
    timestamp = datetime(yesterday.year, yesterday.month, yesterday.day, 7, 29)

    expected = f"{yesterday.month:02d}/{yesterday.day:02d} 07:29"
    assert app._format_message_timestamp(timestamp) == expected


def test_format_message_timestamp_none_returns_none():
    """境界値: timestampがNone（旧形式のメッセージ等）の場合はNoneを返し、例外は送出しない。"""
    import app

    assert app._format_message_timestamp(None) is None


def test_format_message_timestamp_midnight_boundary_is_formatted_correctly():
    """境界値: 0時0分ちょうどでも例外なく"00:00"として整形される。"""
    from datetime import datetime

    import app

    today = datetime.now().date()
    timestamp = datetime(today.year, today.month, today.day, 0, 0)

    assert app._format_message_timestamp(timestamp) == "00:00"


def test_chat_send_shows_timestamp_caption_for_user_and_assistant_messages(monkeypatch):
    """正常系: チャット送信直後、質問・回答それぞれの直前に当日分の"HH:MM"形式の
    タイムスタンプキャプションが表示される。"""
    import re

    at = _run_app()
    at = at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    caption_texts = [c.value for c in at.caption]
    hhmm_captions = [c for c in caption_texts if re.fullmatch(r"\d{2}:\d{2}", c)]
    assert len(hhmm_captions) == 2


def test_chat_send_persists_timestamp_in_session_state_messages(monkeypatch):
    """正常系: 送信したメッセージのadditional_kwargs["timestamp"]に、送信時刻(datetime)が
    質問・回答の両方に同じ値で保持される。"""
    at = _run_app()
    at = at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    messages = at.session_state["messages"]
    assert len(messages) == 2
    human_ts = messages[0].additional_kwargs.get("timestamp")
    ai_ts = messages[1].additional_kwargs.get("timestamp")
    assert human_ts is not None
    assert human_ts == ai_ts


def test_switching_to_past_thread_from_different_day_shows_date_in_caption(monkeypatch):
    """正常系: 日付をまたぐ過去スレッドを復元すると、"MM/DD HH:MM"形式で
    日付付きのタイムスタンプが表示される。"""
    from datetime import datetime

    past_created_at = datetime(2024, 1, 1, 7, 29)
    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-past",
                "created_at": past_created_at,
                "first_question": "過去の質問",
                "count": 1,
            }
        ],
    )
    monkeypatch.setattr(
        memory,
        "load_conversation",
        lambda thread_id: (
            [{"question": "過去の質問", "answer": "過去の回答", "created_at": past_created_at}]
            if thread_id == "thread-past"
            else []
        ),
    )

    at = _run_app()
    at = at.sidebar.selectbox[0].select("thread-past").run()

    assert at.exception == []
    caption_texts = [c.value for c in at.caption]
    assert "01/01 07:29" in caption_texts


def test_switching_to_past_thread_from_today_shows_time_only_caption(monkeypatch):
    """境界値: 復元した過去スレッドの会話ログが今日の日付の場合、
    日付を省いた"HH:MM"形式で表示される（同日スレッドの過去メッセージでも
    日付表示にはならない）。"""
    from datetime import datetime

    today = datetime.now().date()
    created_at = datetime(today.year, today.month, today.day, 8, 15)
    monkeypatch.setattr(
        memory,
        "list_threads",
        lambda: [
            {
                "thread_id": "thread-past",
                "created_at": created_at,
                "first_question": "過去の質問",
                "count": 1,
            }
        ],
    )
    monkeypatch.setattr(
        memory,
        "load_conversation",
        lambda thread_id: (
            [{"question": "過去の質問", "answer": "過去の回答", "created_at": created_at}]
            if thread_id == "thread-past"
            else []
        ),
    )

    at = _run_app()
    at = at.sidebar.selectbox[0].select("thread-past").run()

    assert at.exception == []
    caption_texts = [c.value for c in at.caption]
    assert "08:15" in caption_texts


def test_render_loop_message_without_timestamp_key_shows_no_caption_and_does_not_crash():
    """異常系（後方互換）: additional_kwargsに"timestamp"キーを持たない旧形式の
    メッセージがセッションに残っていても、タイムスタンプキャプションなしで描画され
    クラッシュしない。"""
    import re

    at = _run_app()
    at.session_state["messages"] = [
        HumanMessage(content="旧形式の質問"),
        AIMessage(content="旧形式の回答"),
    ]
    at = at.run()

    assert at.exception == []
    markdown_texts = [m.value for m in at.markdown]
    assert "旧形式の質問" in markdown_texts
    assert "旧形式の回答" in markdown_texts
    caption_texts = [c.value for c in at.caption]
    assert not any(re.fullmatch(r"\d{2}:\d{2}", c) for c in caption_texts)
    assert not any(re.fullmatch(r"\d{2}/\d{2} \d{2}:\d{2}", c) for c in caption_texts)


# --- 8. 空状態ガイダンス（_render_empty_state_guidance） ---


def test_empty_state_guidance_shows_upload_info_when_no_documents(monkeypatch):
    """正常系: インデックス済みドキュメントが0件の場合、サイドバーからのアップロードか
    `data/` への直接配置を促す案内（st.info）が表示される。"""
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [])

    at = _run_app()

    assert at.exception == []
    assert len(at.info) == 1
    assert "ドキュメントが登録されていません" in at.info[0].value
    assert "アップロード" in at.info[0].value


def test_empty_state_guidance_hidden_when_documents_and_conversations_exist(monkeypatch):
    """正常系: ドキュメントが登録済みで、かつ会話ログも既にある（初回訪問でない）場合、
    空状態ガイダンスは一切表示されず通常表示のままになる。"""
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "dummy.txt", "chunk_count": 1}])
    monkeypatch.setattr(memory, "conversation_count", lambda thread_id=None: 1)

    at = _run_app()

    assert at.exception == []
    assert at.info == []


def test_empty_state_guidance_shows_usage_steps_for_first_time_visit(monkeypatch):
    """正常系: ドキュメントは登録済みだが会話ログが0件（初回訪問）の場合、
    ファイル配置→自動DB反映→チャットで質問、の3ステップの使い方ガイダンスが表示される。"""
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "dummy.txt", "chunk_count": 1}])
    monkeypatch.setattr(memory, "conversation_count", lambda thread_id=None: 0)

    at = _run_app()

    assert at.exception == []
    assert len(at.info) == 1
    guidance = at.info[0].value
    assert "ようこそ" in guidance
    assert "アップロード" in guidance
    assert "ベクトルDB" in guidance
    assert "質問" in guidance


def test_empty_state_guidance_disappears_on_rerun_after_first_question(monkeypatch):
    """境界値: 初回訪問の状態から実際に1件質問すると、ガイダンス表示の判定
    （_render_empty_state_guidance）はチャット入力欄より前に行われるため、
    質問直後の同一スクリプト実行内では回答と一緒にまだガイダンスが表示され続ける
    （session_state.messagesへの追記はガイダンス評価より後に行われるため）。
    その後もう一度スクリプトが再実行されると、messagesが埋まっているため
    ガイダンスは表示されなくなる。"""
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "dummy.txt", "chunk_count": 1}])
    monkeypatch.setattr(memory, "conversation_count", lambda thread_id=None: 0)
    fake_agent = _FakeAgent(answer="回答です")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    assert len(at.info) == 1  # 質問前は初回訪問ガイダンスが出ている

    at = at.chat_input[0].set_value("質問です").run()
    assert at.exception == []
    assert len(at.session_state["messages"]) == 2
    # 質問直後のこの実行では、ガイダンス評価が新規メッセージ追記より前に
    # 行われるため、回答と一緒にまだガイダンスが残っている。
    assert len(at.info) == 1

    at = at.run()

    assert at.exception == []
    assert at.info == []


def test_empty_state_guidance_not_retriggered_by_new_chat_when_other_threads_exist(monkeypatch):
    """境界値（回帰確認）: 「新しい会話を始める」で現在のスレッドのmessagesが空に
    なっても、他スレッドに既に会話ログがある（conversation_count(thread_id=None)が
    0件でない）場合は、初回訪問と誤判定されずガイダンスは表示されない
    （_render_empty_state_guidanceのdocstring記載の設計意図の確認）。"""
    monkeypatch.setattr(ingest, "list_indexed_files", lambda: [{"name": "dummy.txt", "chunk_count": 1}])
    monkeypatch.setattr(memory, "conversation_count", lambda thread_id=None: 3)

    at = _run_app()
    assert at.info == []

    new_chat_button = next(b for b in at.sidebar.button if "新しい会話" in b.label)
    at = new_chat_button.click().run()

    assert at.exception == []
    assert at.session_state["messages"] == []
    assert at.info == []


# --- 17. 直近の回答の再生成（🔄 再生成ボタン） ---


def test_regenerate_button_shown_only_for_last_ai_message():
    """正常系: 🔄再生成ボタンは、複数ターンの会話履歴があっても最後のAI回答にのみ表示される。"""
    at = _run_app()
    at.session_state["messages"] = [
        HumanMessage(content="質問1"),
        AIMessage(content="回答1"),
        HumanMessage(content="質問2"),
        AIMessage(content="回答2"),
    ]
    at = at.run()

    assert at.exception == []
    regenerate_keys = [b.key for b in at.button if b.key and b.key.startswith("regenerate_")]
    assert regenerate_keys == ["regenerate_thread-test_3"]


def test_regenerate_button_not_shown_when_no_messages():
    """境界値: 会話履歴が1件も無い状態では🔄再生成ボタンは表示されない。"""
    at = _run_app()

    assert at.exception == []
    assert at.session_state["messages"] == []
    regenerate_keys = [b.key for b in at.button if b.key and b.key.startswith("regenerate_")]
    assert regenerate_keys == []


def test_regenerate_button_not_shown_when_last_message_is_human():
    """境界値: 末尾がAIMessageでなくHumanMessage（例: 回答生成前の状態）の場合、
    🔄再生成ボタンは表示されない。"""
    at = _run_app()
    at.session_state["messages"] = [
        HumanMessage(content="質問1"),
        AIMessage(content="回答1"),
        HumanMessage(content="質問2"),
    ]
    at = at.run()

    assert at.exception == []
    regenerate_keys = [b.key for b in at.button if b.key and b.key.startswith("regenerate_")]
    assert regenerate_keys == []


def test_regenerate_click_replaces_answer_without_duplicating_history_and_skips_memory_save(monkeypatch):
    """正常系: 🔄再生成ボタン押下で、質問はそのままに末尾のAI回答だけが新しい回答に
    置き換わり、messagesの件数は変わらない。エージェントに渡す履歴にも質問が重複しない。
    再生成時はナレッジ化（save_conversation）をスキップする。"""
    fake_agent = _FakeAgent(answer="新しい回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)
    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append(a))

    at = _run_app()
    at.session_state["messages"] = [
        HumanMessage(content="質問です"),
        AIMessage(content="元の回答"),
    ]
    at = at.run()

    regenerate_button = next(b for b in at.button if b.key == "regenerate_thread-test_1")
    at = regenerate_button.click().run()

    assert at.exception == []
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[0].content == "質問です"
    assert messages[1].content == "新しい回答"
    assert save_calls == []

    assert len(fake_agent.stream_calls) == 1
    sent_messages = fake_agent.stream_calls[0]["messages"]
    assert sum(isinstance(m, HumanMessage) and m.content == "質問です" for m in sent_messages) == 1
    assert "regenerating" not in at.session_state
    assert "regenerate_original_message" not in at.session_state


def test_regenerate_click_clears_previous_feedback_recorded_state(monkeypatch):
    """正常系: 再生成すると、古い回答に対して記録済みだったフィードバック状態はクリアされ、
    新しい回答に対して改めて👍/👎を押せるようになる。"""
    fake_agent = _FakeAgent(answer="新しい回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.session_state["messages"] = [
        HumanMessage(content="質問です"),
        AIMessage(content="元の回答"),
    ]
    at.session_state["feedback_thread-test_1_recorded"] = feedback.RATING_UP
    at = at.run()

    regenerate_button = next(b for b in at.button if b.key == "regenerate_thread-test_1")
    at = regenerate_button.click().run()

    assert at.exception == []
    assert "feedback_thread-test_1_recorded" not in at.session_state


def test_regenerate_button_shown_immediately_after_answer_generated(monkeypatch):
    """正常系: 質問直後にストリーミング表示された最新回答にも、次のrerunを待たず
    その場で🔄再生成ボタンが表示される。"""
    fake_agent = _FakeAgent(answer="最初の回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at = at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    regenerate_keys = [b.key for b in at.button if b.key and b.key.startswith("regenerate_")]
    assert regenerate_keys == ["regenerate_thread-test_1"]


def test_regenerate_button_shown_immediately_after_regenerate_click(monkeypatch):
    """正常系: 🔄再生成ボタン押下による再生成直後にも、次のrerunを待たず
    その場で新しい回答に対する🔄再生成ボタンが表示される。"""
    fake_agent = _FakeAgent(answer="新しい回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.session_state["messages"] = [
        HumanMessage(content="質問です"),
        AIMessage(content="元の回答"),
    ]
    at = at.run()

    regenerate_button = next(b for b in at.button if b.key == "regenerate_thread-test_1")
    at = regenerate_button.click().run()

    assert at.exception == []
    regenerate_keys = [b.key for b in at.button if b.key and b.key.startswith("regenerate_")]
    assert regenerate_keys == ["regenerate_thread-test_1"]


def test_regenerate_cancelled_mid_generation_restores_original_answer_on_next_run():
    """異常系: 再生成中にキャンセル操作等でスクリプトが打ち切られると、
    'regenerating'フラグは既にpop済みで消える一方、退避しておいた元の回答
    （regenerate_original_message）だけがセッションに孤立して残る。
    次回のスクリプト実行でこの孤立データが自動的にmessagesへ復元されることを確認する。"""
    at = _run_app()
    original_answer = AIMessage(content="元の回答")
    at.session_state["messages"] = [HumanMessage(content="質問です")]
    at.session_state["regenerate_original_message"] = original_answer
    # "regenerating"はキャンセル時点で既にpop済みのため、あえてセットしない。

    at = at.run()

    assert at.exception == []
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[1].content == "元の回答"
    assert "regenerate_original_message" not in at.session_state


def test_regenerate_failure_restores_original_answer(monkeypatch):
    """異常系: 再生成中に例外が発生した場合、取り除いていた元の回答を復元し、
    ユーザーが既存の回答ごと失わないようにする。復元後にst.rerun()するため、
    AppTest.run()が内部の再実行まで追従した最終状態では、エラー表示は残らず
    復元済みの元の回答がその場で表示される（rerunしないとエラー表示のまま残り、
    無関係な次の操作まで元の回答が画面に出てこない）。"""
    fake_agent = _FakeAgent(exc=RuntimeError("stream broken"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)
    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append(a))

    at = _run_app()
    at.session_state["messages"] = [
        HumanMessage(content="質問です"),
        AIMessage(content="元の回答"),
    ]
    at = at.run()

    regenerate_button = next(b for b in at.button if b.key == "regenerate_thread-test_1")
    at = regenerate_button.click().run()

    assert at.exception == []
    assert at.error == []
    assert any(m.value == "元の回答" for m in at.markdown)
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[1].content == "元の回答"
    assert save_calls == []
    assert "regenerate_original_message" not in at.session_state


def test_regenerate_failure_shows_error_and_calls_rerun_once(monkeypatch):
    """異常系: 再生成中の生成失敗そのものではエラーが表示され（挙動は変わらない）、
    復元済みの回答を再描画するためst.rerun()が1回呼ばれることを確認する。
    再生成ボタン押下（それ自体もst.rerun()する）を経由すると押下時と失敗時の
    呼び出しが混ざってしまうため、失敗直前の状態（regenerating=True）を
    直接セットして生成失敗の回だけを単独で検証する。"""
    import streamlit as st

    fake_agent = _FakeAgent(exc=RuntimeError("stream broken"))
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)
    rerun_calls = []
    monkeypatch.setattr(st, "rerun", lambda *a, **k: rerun_calls.append(True))

    at = _run_app()
    at.session_state["messages"] = [HumanMessage(content="質問です")]
    at.session_state["regenerate_original_message"] = AIMessage(content="元の回答")
    at.session_state["regenerating"] = True

    at = at.run()

    assert at.exception == []
    assert len(at.error) == 1
    assert rerun_calls == [True]
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[1].content == "元の回答"
