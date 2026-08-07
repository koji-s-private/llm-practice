"""
ローカルドキュメントに質問できるRAGチャットアプリ「Doclore」のバックエンドAPI（FastAPI）。

Issue #88（フロントエンド移行 Step1: API層の切り出し）で、既存のPythonロジック
（`ingest.py` / `rag_chain.py` / `memory.py`）を一切変更せず、そのまま呼び出す
ラッパーとして追加した。既存のStreamlit版（`app.py`）はフォールバックとして
そのまま残しており、どちらからでも同じ `data/` / `chroma_db/` を参照する
（本APIとStreamlit版を同時に起動しても差し支えない）。

起動:
    uvicorn api.main:app --reload

想定しているフロントエンド側の使い方（Step2以降で実装予定）:
    1. POST /api/chat で質問を送信し、SSE（Server-Sent Events）で回答をトークン単位に
       受信して画面にストリーミング表示する（現行Streamlit版の `st.spinner` + 逐次表示相当）。
    2. 受信し終えた回答全文を POST /api/conversations/save で会話ログとして保存する
       （Streamlit版の `save_conversation` 呼び出しに相当。呼ぶかどうかはフロントエンド側の
       「今の会話を記憶として保存する」設定に委ねる）。
    3. data/ 配下のファイルを追加・削除した後は POST /api/sync を呼び、ベクトルDBに反映する。

ホスティングはローカル起動のみを前提としており、外部・クラウドへの追加送信は行わない
（AGENTS.mdの無料制約）。
"""

import json
from collections.abc import Generator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from ingest import sync_data_dir
from memory import conversation_count, new_thread_id, save_conversation
from rag_chain import build_agent

app = FastAPI(
    title="Doclore API",
    description="ローカルRAGチャットアプリ「Doclore」のバックエンドAPI（Issue #88）",
)

# Step2以降でVite開発サーバ（デフォルトはlocalhost:5173）から本APIを叩けるようにするための
# CORS設定。ローカル開発用途のみを想定しており、ここに列挙したポート以外からのアクセスは許可しない。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    """疎通確認用のヘルスチェックエンドポイント（サーバがローカル起動できているかの確認用）。"""
    return {"status": "ok"}


# --- チャット応答（ストリーミング） ---


class ChatMessage(BaseModel):
    """会話履歴1件分（role: "user" または "assistant"）。"""

    role: str
    content: str


class ChatRequest(BaseModel):
    """POST /api/chat のリクエストボディ。"""

    thread_id: str
    message: str
    history: list[ChatMessage] = []


def _to_langchain_messages(history: list[ChatMessage]) -> list:
    """フロントエンドから受け取った会話履歴をLangChainのメッセージ型に変換する。"""
    messages = []
    for m in history:
        if m.role == "user":
            messages.append(HumanMessage(content=m.content))
        else:
            messages.append(AIMessage(content=m.content))
    return messages


def _stream_chat_response(thread_id: str, message: str, history: list[ChatMessage]) -> Generator[str, None, None]:
    """agentの回答をSSE形式（`data: <json>\\n\\n`）のテキストとして順次yieldする。

    `create_agent` が返すエージェントは `.stream(input, stream_mode="messages")` で
    LLMのトークン単位のストリーミングに対応している（LangGraphの標準的なストリーミング
    インターフェース）。各要素は `(メッセージチャンク, メタデータ)` のタプルで、
    ツール呼び出し中の内部メッセージにはcontentが空文字のものも含まれるため、
    contentがあるものだけをクライアントに送る。
    """
    try:
        agent = build_agent(thread_id)
        input_messages = _to_langchain_messages(history) + [HumanMessage(content=message)]
        for chunk, _metadata in agent.stream({"messages": input_messages}, stream_mode="messages"):
            content = getattr(chunk, "content", "")
            if content:
                yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
    except Exception as e:
        # Streamlit版のst.error相当。ストリーミング開始後は通常のHTTPエラーレスポンスに
        # 差し替えられないため、SSEの1イベントとしてエラー内容を通知する。
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        return

    yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    """チャット応答をSSE（Server-Sent Events）でストリーミング返却する。

    現行Streamlit版（app.py）の `agent.invoke()` による一括回答表示と異なり、
    トークンが生成され次第クライアントに送信するため、フロントエンド側で
    逐次表示（タイプライター表示）を実現できる。
    """
    return StreamingResponse(
        _stream_chat_response(request.thread_id, request.message, request.history),
        media_type="text/event-stream",
    )


# --- data/ 配下ドキュメントの同期 ---


class SyncResult(BaseModel):
    """POST /api/sync のレスポンスボディ（ingest.sync_data_dir()の戻り値をそのまま返す）。"""

    added: list[str]
    updated: list[str]
    removed: list[str]
    failed: list[str]


@app.post("/api/sync", response_model=SyncResult)
def sync() -> dict:
    """data/ 配下ドキュメントをベクトルDBに同期する（ingest.sync_data_dir()のラッパー）。"""
    return sync_data_dir(verbose=False)


# --- 会話ログ（memory.py）の参照・保存 ---


class NewThreadResponse(BaseModel):
    """POST /api/conversations/new のレスポンスボディ。"""

    thread_id: str


@app.post("/api/conversations/new", response_model=NewThreadResponse)
def create_new_thread() -> dict:
    """新しい会話スレッドIDを発行する（Streamlit版の「🆕 新しい会話を始める」相当）。"""
    return {"thread_id": new_thread_id()}


class ConversationCountResponse(BaseModel):
    """GET /api/conversations/count のレスポンスボディ。"""

    thread_id: str | None
    count: int


@app.get("/api/conversations/count", response_model=ConversationCountResponse)
def get_conversation_count(thread_id: str | None = None) -> dict:
    """保存済み会話ログの件数を取得する（thread_id省略時は全スレッド合計）。"""
    return {"thread_id": thread_id, "count": conversation_count(thread_id)}


class SaveConversationRequest(BaseModel):
    """POST /api/conversations/save のリクエストボディ。"""

    question: str
    answer: str
    thread_id: str


class SaveConversationResponse(BaseModel):
    """POST /api/conversations/save のレスポンスボディ。"""

    path: str


@app.post("/api/conversations/save", response_model=SaveConversationResponse)
def save_conversation_endpoint(request: SaveConversationRequest) -> dict:
    """1回分の質問・回答を会話ログとして保存する（memory.save_conversation()のラッパー）。

    保存後のベクトルDBへの反映は行わない（Streamlit版と同様、次回の /api/sync 呼び出しに委ねる）。
    """
    path = save_conversation(request.question, request.answer, request.thread_id)
    return {"path": str(path)}
