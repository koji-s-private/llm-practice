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
  `memory.list_threads` / `memory.load_conversation`
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
from streamlit.testing.v1 import AppTest

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
    意図しない変更の両方を検知できるようにする）。「💬 過去の会話」見出しが
    追加された後も、サイドバー内の見出しの並び順・両方の文言が保たれていることを確認する。"""
    at = _run_app()

    headings = [s.value for s in at.sidebar.subheader]
    assert headings == ["💬 過去の会話", "📂 ドキュメント管理"]


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
        lambda thread_id: [{"question": "過去の質問", "answer": "過去の回答"}] if thread_id == "thread-past" else [],
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
        lambda thread_id: [{"question": "過去の質問", "answer": "過去の回答"}] if thread_id == "thread-past" else [],
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
        lambda thread_id: [{"question": "過去の質問", "answer": "過去の回答"}] if thread_id == "thread-past" else [],
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
        lambda thread_id: [{"question": "過去の質問", "answer": "過去の回答"}] if thread_id == "thread-past" else [],
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
