"""
ローカルドキュメントに質問できるRAGチャットアプリ「Doclore」のバックエンドAPI（FastAPI）。

既存のPythonロジック（`ingest.py` / `rag_chain.py` / `memory.py`）を一切変更せず、
そのまま呼び出すラッパーとして構成している。既存のStreamlit版（`app.py`）はフォールバックとして
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
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from history_utils import _windowed_history
from ingest import sync_data_dir
from memory import CONVERSATIONS_DIR, THREAD_ID_PATTERN, conversation_count, new_thread_id, save_conversation
from rag_chain import build_agent
from source_formatting import format_snippet as _format_snippet
from source_formatting import format_source_label as _format_source_label

app = FastAPI(
    title="Doclore API",
    description="ローカルRAGチャットアプリ「Doclore」のバックエンドAPI",
)

# Vite開発サーバ（デフォルトはlocalhost:5173）から本APIを叩けるようにするためのCORS設定。
# ローカル開発用途のみを想定しており、ここに列挙したポート以外からのアクセスは許可しない。
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


# --- thread_id のバリデーション（パストラバーサル対策） ---

# memory.py は thread_id をそのまま `CONVERSATIONS_DIR / thread_id` としてパスに組み込む。
# 本API層ではHTTPリクエストの生のthread_idが直接渡ってくるため、絶対パスや`../`を含む値を
# 許可するとdata/conversations/の外への任意ファイル書き込み・情報漏えいにつながる。ingest.pyの
# safe_upload_dest()と同様に、許可文字種を制限した上でresolve()後の実パスも検証する。
# 許可文字の正規表現はmemory.pyのTHREAD_ID_PATTERNをそのまま使い、ポリシーが分散しないようにする。

# new_thread_id() が生成する値（uuid4().hex[:8]、8文字）に十分な余裕を持たせつつ、
# OSのファイル名長制限（一般的に255バイト程度）に触れないよう上限を設ける。
_THREAD_ID_MAX_LENGTH = 64


def _validate_thread_id(thread_id: str) -> str:
    """thread_id がファイルパスとして安全か検証し、不正であれば400エラーを送出する。"""
    if not thread_id or not THREAD_ID_PATTERN.fullmatch(thread_id):
        raise HTTPException(status_code=400, detail="thread_id の形式が不正です")
    if len(thread_id) > _THREAD_ID_MAX_LENGTH:
        raise HTTPException(status_code=400, detail="thread_id が長すぎます")
    resolved = (CONVERSATIONS_DIR / thread_id).resolve()
    if resolved.parent != CONVERSATIONS_DIR.resolve():
        raise HTTPException(status_code=400, detail="thread_id の形式が不正です")
    return thread_id


# --- チャット応答（ストリーミング） ---


class ChatMessage(BaseModel):
    """会話履歴1件分（role: "user" または "assistant"）。"""

    role: Literal["user", "assistant"]
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


def _serialize_sources(sources: list[Document]) -> list[dict]:
    """retrieve_contextツールが返した参照元ドキュメント一覧をSSE送信用のJSONに変換する。"""
    return [
        {"label": _format_source_label(doc.metadata), "snippet": _format_snippet(doc.page_content)} for doc in sources
    ]


def _stream_chat_response(thread_id: str, message: str, history: list[ChatMessage]) -> Generator[str, None, None]:
    """agentの回答をSSE形式（`data: <json>\\n\\n`）のテキストとして順次yieldする。

    `create_agent` が返すエージェントは `.stream(input, stream_mode="messages")` で
    LLMのトークン単位のストリーミングに対応している。ToolMessage（retrieve_contextツールの
    実行結果）は回答本文ではないため送信対象から除外し、代わりにartifact（取得ドキュメント）を
    蓄積してストリーム終了後に`sources`イベントとしてまとめて送信する
    （app.pyの `_stream_answer` と同じ方針）。
    AIMessageChunk.content はプロバイダによって型が異なる（str、またはAnthropicの
    content blocks list）ため、getattr(chunk, "content", "") ではなく text系ブロックを
    結合済みの .text プロパティで本文を取り出す。

    エージェントに渡す会話履歴は `_windowed_history()` でトークン予算内にウィンドウイングする
    （app.pyと同じ防御ロジック。Ollama利用時にコンテキスト長超過で古い履歴が黙って
    切り捨てられるのを防ぐ）。リクエストで受け取った `history` 自体は変更しない。
    """
    sources: list[Document] = []
    try:
        agent = build_agent(thread_id)
        windowed_history = _windowed_history(_to_langchain_messages(history))
        input_messages = windowed_history + [HumanMessage(content=message)]
        for chunk, _metadata in agent.stream({"messages": input_messages}, stream_mode="messages"):
            if isinstance(chunk, ToolMessage):
                if getattr(chunk, "artifact", None):
                    sources.extend(chunk.artifact)
                continue
            text = getattr(chunk, "text", "")
            if text:
                yield f"data: {json.dumps({'content': text}, ensure_ascii=False)}\n\n"
    except Exception as e:
        # Streamlit版のst.error相当。ストリーミング開始後は通常のHTTPエラーレスポンスに
        # 差し替えられないため、SSEの1イベントとしてエラー内容を通知する。
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        return

    if sources:
        yield f"data: {json.dumps({'sources': _serialize_sources(sources)}, ensure_ascii=False)}\n\n"

    yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    """チャット応答をSSE（Server-Sent Events）でストリーミング返却する。

    thread_id はファイルパスには使われない（Chromaのメタデータフィルタとしてのみ使用）が、
    クライアントからの直接入力である点は他のエンドポイントと同じなので、一貫性のため
    同じ形式検証を行う。
    """
    _validate_thread_id(request.thread_id)
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
    if thread_id is not None:
        _validate_thread_id(thread_id)
    return {"thread_id": thread_id, "count": conversation_count(thread_id)}


class SaveConversationRequest(BaseModel):
    """POST /api/conversations/save のリクエストボディ。

    is_fallback: ドキュメントに根拠が見つからず一般知識で回答した場合に True を渡す
    （app.pyの `save_conversation(..., is_fallback=not sources)` 相当）。
    """

    question: str
    answer: str
    thread_id: str
    is_fallback: bool = False


class SaveConversationResponse(BaseModel):
    """POST /api/conversations/save のレスポンスボディ。"""

    path: str


@app.post("/api/conversations/save", response_model=SaveConversationResponse)
def save_conversation_endpoint(request: SaveConversationRequest) -> dict:
    """1回分の質問・回答を会話ログとして保存する（memory.save_conversation()のラッパー）。

    保存後のベクトルDBへの反映は行わない（Streamlit版と同様、次回の /api/sync 呼び出しに委ねる）。
    """
    _validate_thread_id(request.thread_id)
    path = save_conversation(request.question, request.answer, request.thread_id, is_fallback=request.is_fallback)
    return {"path": str(path)}
