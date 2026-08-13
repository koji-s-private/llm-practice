"""api/main.py（FastAPIバックエンド）のエンドポイントのテスト。

`fastapi.testclient.TestClient` を使い、実際にOllama/LLM/Chromaを呼び出さず、
`api.main` がモジュールトップレベルで import している以下のシンボルを monkeypatch で
軽量なフェイクに差し替えて検証する（tests/test_app.py と同じ方針）。

- `api.main.build_agent`（POST /api/chat が呼ぶ。フェイクエージェントの `.stream()` で
  トークンチャンクを模擬する）
- `api.main.sync_data_dir` / `api.main.new_thread_id` / `api.main.conversation_count` /
  `api.main.save_conversation`
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

import api.main as api_main


class _FakeChunk:
    """agent.stream(..., stream_mode="messages") が返すタプルの1要素目（メッセージチャンク）を模擬する。"""

    def __init__(self, content):
        self.content = content


class _FakeAgent:
    """rag_chain.build_agent() の代わりに使うフェイクエージェント。"""

    def __init__(self, chunks=None, exc=None):
        self.chunks = chunks or []
        self.exc = exc
        self.stream_calls = []

    def stream(self, input_, stream_mode="messages"):
        self.stream_calls.append({"input": input_, "stream_mode": stream_mode})
        if self.exc is not None:
            raise self.exc
        for chunk in self.chunks:
            yield chunk, {}


@pytest.fixture
def client():
    # sync系のテストで500応答の中身自体を検証したいテストがあるため、
    # サーバー側の未処理例外もHTTPレスポンスとして受け取れるようにする。
    return TestClient(api_main.app, raise_server_exceptions=False)


def _parse_sse_events(text: str) -> list[dict]:
    """`data: <json>\\n\\n` 形式のSSEレスポンス本文をパースしてJSONのリストにする。"""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: ")
        events.append(json.loads(block[len("data: ") :]))
    return events


# --- GET /api/health ---


def test_health_returns_ok(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- POST /api/chat ---


def test_chat_streams_tokens_and_done_event(client, monkeypatch):
    fake_agent = _FakeAgent(chunks=[_FakeChunk("こんにちは"), _FakeChunk("、"), _FakeChunk("世界")])
    monkeypatch.setattr(api_main, "build_agent", lambda thread_id: fake_agent)

    response = client.post("/api/chat", json={"thread_id": "thread-1", "message": "質問です", "history": []})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    assert events == [
        {"content": "こんにちは"},
        {"content": "、"},
        {"content": "世界"},
        {"done": True},
    ]


def test_chat_skips_chunks_with_empty_content(client, monkeypatch):
    """ツール呼び出し中の内部メッセージ等、content が空文字のチャンクはクライアントに送らない。"""
    fake_agent = _FakeAgent(chunks=[_FakeChunk(""), _FakeChunk("回答本体")])
    monkeypatch.setattr(api_main, "build_agent", lambda thread_id: fake_agent)

    response = client.post("/api/chat", json={"thread_id": "thread-1", "message": "質問", "history": []})

    events = _parse_sse_events(response.text)
    assert events == [{"content": "回答本体"}, {"done": True}]


def test_chat_reports_error_as_sse_event_without_done(client, monkeypatch):
    """agent.stream() が例外を送出した場合、エラー内容をSSEの1イベントとして返し、doneイベントは送らない。"""
    fake_agent = _FakeAgent(exc=RuntimeError("モデル呼び出しに失敗しました"))
    monkeypatch.setattr(api_main, "build_agent", lambda thread_id: fake_agent)

    response = client.post("/api/chat", json={"thread_id": "thread-1", "message": "質問", "history": []})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert events == [{"error": "モデル呼び出しに失敗しました"}]


def test_chat_reports_error_when_build_agent_itself_fails(client, monkeypatch):
    def _raise(thread_id):
        raise RuntimeError("エージェント構築に失敗しました")

    monkeypatch.setattr(api_main, "build_agent", _raise)

    response = client.post("/api/chat", json={"thread_id": "thread-1", "message": "質問", "history": []})

    events = _parse_sse_events(response.text)
    assert events == [{"error": "エージェント構築に失敗しました"}]


def test_chat_converts_history_roles_to_langchain_messages(client, monkeypatch):
    fake_agent = _FakeAgent(chunks=[_FakeChunk("OK")])
    monkeypatch.setattr(api_main, "build_agent", lambda thread_id: fake_agent)

    history = [
        {"role": "user", "content": "前回の質問"},
        {"role": "assistant", "content": "前回の回答"},
    ]
    response = client.post("/api/chat", json={"thread_id": "thread-1", "message": "今回の質問", "history": history})

    assert response.status_code == 200
    assert len(fake_agent.stream_calls) == 1
    input_messages = fake_agent.stream_calls[0]["input"]["messages"]
    assert [type(m).__name__ for m in input_messages] == ["HumanMessage", "AIMessage", "HumanMessage"]
    assert input_messages[0].content == "前回の質問"
    assert input_messages[1].content == "前回の回答"
    assert input_messages[2].content == "今回の質問"


def test_chat_accepts_empty_message_and_default_history(client, monkeypatch):
    """境界値: message が空文字でも history 省略でも400にはならず、そのままagentに渡る。"""
    fake_agent = _FakeAgent(chunks=[_FakeChunk("応答")])
    monkeypatch.setattr(api_main, "build_agent", lambda thread_id: fake_agent)

    response = client.post("/api/chat", json={"thread_id": "thread-1", "message": ""})

    assert response.status_code == 200
    input_messages = fake_agent.stream_calls[0]["input"]["messages"]
    assert len(input_messages) == 1
    assert input_messages[0].content == ""


def test_chat_missing_required_field_returns_422(client):
    response = client.post("/api/chat", json={"thread_id": "thread-1"})

    assert response.status_code == 422


# --- POST /api/chat: sources（参照元ドキュメント）イベント ---


def test_chat_streams_sources_event_with_label_and_snippet_from_tool_message_artifact(client, monkeypatch):
    """正常系: retrieve_contextツールの実行結果(ToolMessage)のartifactから参照元ドキュメントを
    取得し、contentチャンクは通常どおり送りつつ、ストリーム終了直前にsourcesイベントとして
    ラベル・スニペットを整形して送信する。"""
    doc = Document(
        page_content="これは参照元ドキュメントの本文です。",
        metadata={"source": "/data/manual.pdf", "page": 2},
    )
    tool_message = ToolMessage(content="検索結果の生テキスト", tool_call_id="call-1", artifact=[doc])
    fake_agent = _FakeAgent(chunks=[tool_message, _FakeChunk("文書に基づく回答")])
    monkeypatch.setattr(api_main, "build_agent", lambda thread_id: fake_agent)

    response = client.post("/api/chat", json={"thread_id": "thread-1", "message": "質問", "history": []})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert events == [
        {"content": "文書に基づく回答"},
        {"sources": [{"label": "manual.pdf（p.3）", "snippet": "これは参照元ドキュメントの本文です。"}]},
        {"done": True},
    ]


def test_chat_does_not_send_content_for_tool_message_itself(client, monkeypatch):
    """正常系: ToolMessage自体のcontent（検索結果の生テキスト）はcontentイベントとして
    送信されない（回答本文と混ざらないことの確認）。"""
    tool_message = ToolMessage(
        content="Source: {...}\nContent: 生の検索結果",
        tool_call_id="call-1",
        artifact=[Document(page_content="本文", metadata={"source": "a.txt"})],
    )
    fake_agent = _FakeAgent(chunks=[tool_message, _FakeChunk("回答")])
    monkeypatch.setattr(api_main, "build_agent", lambda thread_id: fake_agent)

    response = client.post("/api/chat", json={"thread_id": "thread-1", "message": "質問", "history": []})

    events = _parse_sse_events(response.text)
    contents = [e["content"] for e in events if "content" in e]
    assert contents == ["回答"]


def test_chat_no_sources_event_when_tool_message_artifact_is_empty(client, monkeypatch):
    """正常系: 参照元が見つからず一般知識で回答した場合（ToolMessage.artifactが空）は
    sourcesイベントを送信しない。"""
    tool_message = ToolMessage(
        content="関連する情報はドキュメント内に見つかりませんでした。",
        tool_call_id="call-1",
        artifact=[],
    )
    fake_agent = _FakeAgent(chunks=[tool_message, _FakeChunk("一般知識による回答")])
    monkeypatch.setattr(api_main, "build_agent", lambda thread_id: fake_agent)

    response = client.post("/api/chat", json={"thread_id": "thread-1", "message": "質問", "history": []})

    events = _parse_sse_events(response.text)
    assert events == [{"content": "一般知識による回答"}, {"done": True}]
    assert all("sources" not in e for e in events)


def test_chat_no_sources_event_when_no_tool_message_at_all(client, monkeypatch):
    """境界値: retrieve_contextツール自体が呼ばれなかった（ToolMessageが1件も無い）場合も
    sourcesイベントは送信されない。"""
    fake_agent = _FakeAgent(chunks=[_FakeChunk("回答")])
    monkeypatch.setattr(api_main, "build_agent", lambda thread_id: fake_agent)

    response = client.post("/api/chat", json={"thread_id": "thread-1", "message": "質問", "history": []})

    events = _parse_sse_events(response.text)
    assert events == [{"content": "回答"}, {"done": True}]


# --- thread_id のパストラバーサル対策 ---

_MALICIOUS_THREAD_IDS = [
    "/etc/pwned_by_thread_id",
    "../../etc/passwd",
    "..",
    "sub/dir",
    "",
]


@pytest.mark.parametrize("thread_id", _MALICIOUS_THREAD_IDS)
def test_chat_rejects_path_traversal_thread_id(client, monkeypatch, thread_id):
    """絶対パス・相対トラバーサル等の不正なthread_idは400で拒否され、agentは呼ばれない。"""
    fake_agent = _FakeAgent(chunks=[_FakeChunk("応答")])
    monkeypatch.setattr(api_main, "build_agent", lambda thread_id: fake_agent)

    response = client.post("/api/chat", json={"thread_id": thread_id, "message": "質問", "history": []})

    assert response.status_code == 400
    assert fake_agent.stream_calls == []


# --- POST /api/sync ---


def test_sync_returns_result_from_sync_data_dir(client, monkeypatch):
    fake_result = {"added": ["a.txt"], "updated": [], "removed": ["b.txt"], "failed": []}
    monkeypatch.setattr(api_main, "sync_data_dir", lambda verbose=True: fake_result)

    response = client.post("/api/sync")

    assert response.status_code == 200
    assert response.json() == fake_result


def test_sync_returns_all_empty_lists_when_no_changes(client, monkeypatch):
    """境界値: 差分が無い場合は全キーが空リストになる。"""
    empty_result = {"added": [], "updated": [], "removed": [], "failed": []}
    monkeypatch.setattr(api_main, "sync_data_dir", lambda verbose=True: empty_result)

    response = client.post("/api/sync")

    assert response.status_code == 200
    assert response.json() == empty_result


def test_sync_propagates_failure_as_server_error(client, monkeypatch):
    """異常系: sync_data_dir() が予期せぬ例外を送出した場合、未処理のまま500になる。"""

    def _raise(verbose=True):
        raise RuntimeError("ベクトルDBへの同期に失敗しました")

    monkeypatch.setattr(api_main, "sync_data_dir", _raise)

    response = client.post("/api/sync")

    assert response.status_code == 500


# --- POST /api/conversations/new ---


def test_create_new_thread_returns_generated_thread_id(client, monkeypatch):
    monkeypatch.setattr(api_main, "new_thread_id", lambda: "abcd1234")

    response = client.post("/api/conversations/new")

    assert response.status_code == 200
    assert response.json() == {"thread_id": "abcd1234"}


# --- GET /api/conversations/count ---


def test_get_conversation_count_without_thread_id(client, monkeypatch):
    captured = {}

    def _fake_count(thread_id=None):
        captured["thread_id"] = thread_id
        return 5

    monkeypatch.setattr(api_main, "conversation_count", _fake_count)

    response = client.get("/api/conversations/count")

    assert response.status_code == 200
    assert response.json() == {"thread_id": None, "count": 5}
    assert captured["thread_id"] is None


def test_get_conversation_count_with_thread_id(client, monkeypatch):
    monkeypatch.setattr(api_main, "conversation_count", lambda thread_id=None: 3 if thread_id == "thread-a" else 0)

    response = client.get("/api/conversations/count", params={"thread_id": "thread-a"})

    assert response.status_code == 200
    assert response.json() == {"thread_id": "thread-a", "count": 3}


def test_get_conversation_count_zero_boundary(client, monkeypatch):
    """境界値: 会話ログが1件も無い場合は0件を返す。"""
    monkeypatch.setattr(api_main, "conversation_count", lambda thread_id=None: 0)

    response = client.get("/api/conversations/count")

    assert response.status_code == 200
    assert response.json()["count"] == 0


@pytest.mark.parametrize("thread_id", _MALICIOUS_THREAD_IDS)
def test_get_conversation_count_rejects_path_traversal_thread_id(client, monkeypatch, thread_id):
    """絶対パス・相対トラバーサル等の不正なthread_idは400で拒否され、conversation_count()は呼ばれない。"""
    called = {"count": 0}

    def _fake_count(thread_id=None):
        called["count"] += 1
        return 0

    monkeypatch.setattr(api_main, "conversation_count", _fake_count)

    response = client.get("/api/conversations/count", params={"thread_id": thread_id})

    assert response.status_code == 400
    assert called["count"] == 0


# --- POST /api/conversations/save ---


def test_save_conversation_returns_saved_path(client, monkeypatch, tmp_path):
    saved_path = tmp_path / "thread-a" / "20260101_000000_abcdef_question.md"

    def _fake_save(question, answer, thread_id, is_fallback=False):
        assert question == "質問内容"
        assert answer == "回答内容"
        assert thread_id == "thread-a"
        assert is_fallback is False
        return saved_path

    monkeypatch.setattr(api_main, "save_conversation", _fake_save)

    response = client.post(
        "/api/conversations/save",
        json={"question": "質問内容", "answer": "回答内容", "thread_id": "thread-a"},
    )

    assert response.status_code == 200
    assert response.json() == {"path": str(saved_path)}


def test_save_conversation_missing_field_returns_422(client):
    response = client.post("/api/conversations/save", json={"question": "質問だけ"})

    assert response.status_code == 422


@pytest.mark.parametrize("thread_id", _MALICIOUS_THREAD_IDS)
def test_save_conversation_rejects_path_traversal_thread_id(client, monkeypatch, thread_id):
    """絶対パス・相対トラバーサル等の不正なthread_idは400で拒否され、save_conversation()は呼ばれない。

    data/conversations/ の外への任意ファイル書き込みを防ぐための検証（PR #97 レビュー指摘対応）。
    """
    called = {"count": 0}

    def _fake_save(question, answer, thread_id, is_fallback=False):
        called["count"] += 1
        return None

    monkeypatch.setattr(api_main, "save_conversation", _fake_save)

    response = client.post(
        "/api/conversations/save",
        json={"question": "質問内容", "answer": "回答内容", "thread_id": thread_id},
    )

    assert response.status_code == 400
    assert called["count"] == 0


# --- POST /api/conversations/save: is_fallback ---


def test_save_conversation_passes_is_fallback_true_through_to_memory(client, monkeypatch):
    """正常系: is_fallback=trueを渡した場合、save_conversationにis_fallback=Trueとして渡される
    （ドキュメントに根拠が無く一般知識で回答した会話ログとして記録される）。"""
    captured = {}

    def _fake_save(question, answer, thread_id, is_fallback=False):
        captured["is_fallback"] = is_fallback
        return "/tmp/dummy.md"

    monkeypatch.setattr(api_main, "save_conversation", _fake_save)

    response = client.post(
        "/api/conversations/save",
        json={"question": "質問内容", "answer": "回答内容", "thread_id": "thread-a", "is_fallback": True},
    )

    assert response.status_code == 200
    assert captured["is_fallback"] is True


def test_save_conversation_without_is_fallback_defaults_to_false(client, monkeypatch):
    """境界値/異常系（後方互換性）: is_fallbackを省略した場合、save_conversationには
    デフォルトのFalseが渡される（本フィールド追加前の既存クライアントの挙動を壊さない）。"""
    captured = {}

    def _fake_save(question, answer, thread_id, is_fallback=False):
        captured["is_fallback"] = is_fallback
        return "/tmp/dummy.md"

    monkeypatch.setattr(api_main, "save_conversation", _fake_save)

    response = client.post(
        "/api/conversations/save",
        json={"question": "質問内容", "answer": "回答内容", "thread_id": "thread-a"},
    )

    assert response.status_code == 200
    assert captured["is_fallback"] is False


# --- thread_id の最大長チェック ---


def test_save_conversation_accepts_thread_id_at_max_length_boundary(client, monkeypatch):
    """境界値: 上限ちょうど（64文字）のthread_idは正常に保存できる。"""
    thread_id = "a" * 64
    saved_path = f"/tmp/{thread_id}/dummy.md"

    def _fake_save(question, answer, thread_id_arg, is_fallback=False):
        assert thread_id_arg == thread_id
        return saved_path

    monkeypatch.setattr(api_main, "save_conversation", _fake_save)

    response = client.post(
        "/api/conversations/save",
        json={"question": "質問内容", "answer": "回答内容", "thread_id": thread_id},
    )

    assert response.status_code == 200
    assert response.json() == {"path": saved_path}


def test_save_conversation_accepts_thread_id_under_max_length(client, monkeypatch):
    """正常系: 上限未満（63文字）のthread_idも問題なく保存できる。"""
    thread_id = "a" * 63
    monkeypatch.setattr(
        api_main,
        "save_conversation",
        lambda question, answer, thread_id_arg, is_fallback=False: "/tmp/dummy.md",
    )

    response = client.post(
        "/api/conversations/save",
        json={"question": "質問内容", "answer": "回答内容", "thread_id": thread_id},
    )

    assert response.status_code == 200


def test_save_conversation_rejects_thread_id_exceeding_max_length(client, monkeypatch):
    """異常系: 上限を1文字超える（65文字）thread_idは400で拒否され、save_conversation()は呼ばれない。"""
    thread_id = "a" * 65
    called = {"count": 0}

    def _fake_save(question, answer, thread_id_arg, is_fallback=False):
        called["count"] += 1
        return None

    monkeypatch.setattr(api_main, "save_conversation", _fake_save)

    response = client.post(
        "/api/conversations/save",
        json={"question": "質問内容", "answer": "回答内容", "thread_id": thread_id},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "thread_id が長すぎます"
    assert called["count"] == 0


def test_get_conversation_count_rejects_thread_id_exceeding_max_length(client, monkeypatch):
    """異常系: GET /api/conversations/count でも上限超過のthread_idは400で拒否される。"""
    thread_id = "b" * 65
    called = {"count": 0}

    def _fake_count(thread_id=None):
        called["count"] += 1
        return 0

    monkeypatch.setattr(api_main, "conversation_count", _fake_count)

    response = client.get("/api/conversations/count", params={"thread_id": thread_id})

    assert response.status_code == 400
    assert response.json()["detail"] == "thread_id が長すぎます"
    assert called["count"] == 0
