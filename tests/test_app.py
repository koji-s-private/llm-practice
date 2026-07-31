"""app.py のエラーハンドリング（Issue #14）のテスト。

`streamlit.testing.v1.AppTest` を使い、実際にStreamlitのスクリプト実行エンジン上で
app.py を動かして検証する。重い外部依存（埋め込みモデル・Chroma・LLM）は使わず、
app.py が直接 import している以下のシンボルを monkeypatch で軽量なフェイクに
差し替える:

- `ingest.sync_data_dir`（3箇所すべてから呼ばれる: 起動時同期・再同期ボタン・
  チャット後の自動ナレッジ同期）
- `rag_chain.build_agent`（フェイクエージェントを返す。`.invoke()` の成功/失敗を
  テストごとに切り替える）
- `memory.new_thread_id` / `memory.conversation_count` / `memory.save_conversation`

app.py はモジュールトップレベルで `from ingest import ... sync_data_dir` のように
シンボルをインポートしているため、`AppTest.run()` がスクリプトを実行する
「前」に対象モジュールの属性を monkeypatch しておく必要がある
（実行時に束縛される値がその時点の属性値になるため）。
"""
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

import ingest
import memory
import rag_chain

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class _FakeAgent:
    """rag_chain.build_agent() の代わりに使うフェイクエージェント。"""

    def __init__(self, answer="テスト回答です", exc=None):
        self.answer = answer
        self.exc = exc
        self.invoke_calls = []

    def invoke(self, payload):
        self.invoke_calls.append(payload)
        if self.exc is not None:
            raise self.exc
        return {"messages": payload["messages"] + [AIMessage(content=self.answer)]}


def _ok_sync(verbose=False):
    return {"added": [], "updated": [], "removed": [], "failed": []}


@pytest.fixture(autouse=True)
def _patch_light_dependencies(monkeypatch):
    """全テスト共通: 起動時同期とmemory系を軽量フェイクに差し替える。

    build_agent とチャット後の同期挙動は、各テストが個別に上書きする。
    """
    monkeypatch.setattr(ingest, "sync_data_dir", _ok_sync)
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: _FakeAgent())
    monkeypatch.setattr(memory, "new_thread_id", lambda: "thread-test")
    monkeypatch.setattr(memory, "conversation_count", lambda thread_id: 0)
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: None)


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
    Issue #56で失敗ファイルがあるケースの回帰を防ぐために追加）。"""

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
    """異常系: サイドバーの「🔄 data/ を再同期」ボタン押下時の同期失敗もカバーされる。"""
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


# --- 2. チャット処理中の agent.invoke() 呼び出し ---


def test_chat_success_appends_history_and_shows_answer(monkeypatch):
    """正常系: agent.invoke() が成功すれば回答が表示され、会話履歴にも追加される。"""
    fake_agent = _FakeAgent(answer="これが回答です")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert at.error == []
    assert len(fake_agent.invoke_calls) == 1
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[0].content == "質問です"
    assert messages[1].content == "これが回答です"


def test_chat_invoke_failure_shows_error_and_does_not_crash(monkeypatch):
    """異常系: agent.invoke() が例外（Ollama未起動を想定）を送出した場合、
    st.error でメッセージが表示され、会話履歴には追加されない。"""
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


def test_chat_invoke_failure_skips_auto_knowledge_save(monkeypatch):
    """異常系境界値: invoke失敗時は save_conversation / 事後sync_data_dirが呼ばれない。"""
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
    assert sync_calls["n"] == 1  # invoke失敗のため会話後の同期は呼ばれていない


# --- 3. 会話ログ保存後の sync_data_dir(verbose=False) 呼び出し ---


def test_post_chat_sync_success(monkeypatch):
    """正常系: 回答成功後の会話ログ同期も成功すればエラーは出ない。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append(a))

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert at.error == []
    assert len(save_calls) == 1


def test_post_chat_sync_failure_still_shows_answer(monkeypatch):
    """異常系: 回答自体は成功したが、その後の会話ログ同期が失敗した場合。

    回答は既に表示済みのため会話履歴には残り、同期失敗のst.errorだけが追加表示される
    （アプリは止まらない）。
    """
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    call_count = {"n": 0}

    def flaky_sync(verbose=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"added": [], "updated": [], "removed": [], "failed": []}
        raise RuntimeError("sync fail on second call")

    monkeypatch.setattr(ingest, "sync_data_dir", flaky_sync)

    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append(a))

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(at.error) == 1
    assert "会話ログの同期に失敗しました" in at.error[0].value
    assert "sync fail on second call" in at.error[0].value
    # 回答自体は表示済み・履歴にも残る（st.errorはあくまで追加同期の失敗を知らせるだけ）
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[1].content == "回答"
    assert len(save_calls) == 1


def test_post_chat_sync_not_called_when_auto_save_memory_disabled(monkeypatch):
    """境界値: 「質問・回答を自動で保存する」トグルOFFの場合は
    save_conversation も事後の sync_data_dir も呼ばれない。"""
    fake_agent = _FakeAgent(answer="回答")
    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: fake_agent)

    save_calls = []
    monkeypatch.setattr(memory, "save_conversation", lambda *a, **k: save_calls.append(a))

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
    assert sync_calls["n"] == 1  # 事後同期は呼ばれていない
