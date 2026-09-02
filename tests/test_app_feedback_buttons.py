"""app.py の回答フィードバックボタン（👍/👎）のテスト。

`streamlit.testing.v1.AppTest` で実際にapp.pyのスクリプト実行エンジン上で検証する
（フィードバックボタンはst.button/st.session_stateを使うUI機能で、純粋関数として
切り出せないため、`tests/test_app.py` と同じAppTestベースの検証方式に合わせる）。
軽量フェイクへの差し替え方針・理由は `tests/test_app.py` のdocstring参照。
"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessageChunk
from streamlit.testing.v1 import AppTest

import feedback
import ingest
import memory
import rag_chain

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

_FAKE_SAVED_CONVERSATION_PATH = Path("/tmp/data/conversations/thread-test/fake.md")


class _FakeAgent:
    def __init__(self, answer="テスト回答です"):
        self.answer = answer

    def stream(self, payload, stream_mode="messages"):
        yield AIMessageChunk(content=self.answer), {}


def _ok_sync(verbose=False):
    return {"added": [], "updated": [], "removed": [], "failed": []}


@pytest.fixture(autouse=True)
def _patch_light_dependencies(monkeypatch):
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


def _run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run()
    return at


def _feedback_buttons(at: AppTest, suffix: str):
    return [b for b in at.button if b.key and b.key.endswith(f"_{suffix}")]


def test_feedback_buttons_shown_after_answer(monkeypatch):
    """正常系: 回答が表示されたら、その下に👍/👎ボタンが1組だけ表示される。"""
    monkeypatch.setattr(feedback, "record_feedback", lambda *a, **k: None)

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert at.exception == []
    assert len(_feedback_buttons(at, "up")) == 1
    assert len(_feedback_buttons(at, "down")) == 1


def test_clicking_thumbs_up_records_feedback_with_question_and_answer(monkeypatch):
    """正常系: 👍ボタンを押すと、質問文・回答・rating・thread_idを引数に
    feedback.record_feedback が1回呼ばれる。"""
    calls = []
    monkeypatch.setattr(feedback, "record_feedback", lambda *a, **k: calls.append(a))

    at = _run_app()
    at = at.chat_input[0].set_value("質問です").run()
    up_button = _feedback_buttons(at, "up")[0]

    at = up_button.click().run()

    assert at.exception == []
    assert calls == [("質問です", "テスト回答です", feedback.RATING_UP, "thread-test")]


def test_clicking_thumbs_down_records_feedback(monkeypatch):
    """正常系: 👎ボタンを押すと、rating=RATING_DOWNでrecord_feedbackが呼ばれる。"""
    calls = []
    monkeypatch.setattr(feedback, "record_feedback", lambda *a, **k: calls.append(a))

    at = _run_app()
    at = at.chat_input[0].set_value("質問です").run()
    down_button = _feedback_buttons(at, "down")[0]

    at = down_button.click().run()

    assert at.exception == []
    assert calls == [("質問です", "テスト回答です", feedback.RATING_DOWN, "thread-test")]


def test_feedback_buttons_hidden_and_thanks_message_after_voting(monkeypatch):
    """正常系: 一度フィードバックを記録すると、ボタンは消えお礼の一言に置き換わる
    （再クリックによる重複記録を防ぐ）。"""
    monkeypatch.setattr(feedback, "record_feedback", lambda *a, **k: None)

    at = _run_app()
    at = at.chat_input[0].set_value("質問です").run()
    up_button = _feedback_buttons(at, "up")[0]

    at = up_button.click().run()

    assert _feedback_buttons(at, "up") == []
    assert _feedback_buttons(at, "down") == []
    assert any("フィードバックを記録しました" in c.value for c in at.caption)


def test_feedback_buttons_not_shown_when_generation_fails(monkeypatch):
    """異常系: 回答生成が失敗した場合は回答自体が表示されないため、
    フィードバックボタンも表示されない。"""
    monkeypatch.setattr(feedback, "record_feedback", lambda *a, **k: None)

    class _FailingAgent:
        def stream(self, payload, stream_mode="messages"):
            raise RuntimeError("boom")
            yield  # pragma: no cover - ジェネレータにするためのダミー

    monkeypatch.setattr(rag_chain, "build_agent", lambda thread_id=None: _FailingAgent())

    at = _run_app()
    at.chat_input[0].set_value("質問です").run()

    assert len(at.error) == 1
    assert _feedback_buttons(at, "up") == []
    assert _feedback_buttons(at, "down") == []
