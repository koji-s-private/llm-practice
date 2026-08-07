import getpass
import os
import socket
import sys

from langchain.chat_models import init_chat_model

try:
    # .env ファイルから環境変数を読み込む（`python-dotenv` が必要）
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# LangSmithへのトレース送信はデフォルトOFF（本アプリは外部送信なしが前提のため）。
# .env で明示的に LANGSMITH_TRACING=true かつ LANGSMITH_API_KEY を設定した場合のみ有効化する。
# 未設定の場合は対話的なプロンプト（getpass）でブロックせず、単にトレースなしで起動する。
if os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
    if not os.environ.get("LANGSMITH_API_KEY"):
        print(
            "[setup] LANGSMITH_TRACING=true ですが LANGSMITH_API_KEY が未設定のため、"
            "LangSmithトレースを無効化して起動します。"
        )
        os.environ["LANGSMITH_TRACING"] = "false"
    elif not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")


def _ollama_available() -> bool:
    """ローカルでOllamaサーバーが起動しているかを軽くチェックする（起動が遅くならないよう短いタイムアウト）。"""
    if os.environ.get("DISABLE_OLLAMA") == "true":
        return False
    try:
        with socket.create_connection((OLLAMA_HOST, OLLAMA_PORT), timeout=0.3):
            return True
    except OSError:
        return False


def _build_model():
    """優先順位: 1) Ollama（無料・ローカル） 2) ANTHROPIC_API_KEY 3) OPENAI_API_KEY。"""
    if _ollama_available():
        print(f"[setup] Ollama を検出: {OLLAMA_MODEL}（ローカル・無料）を使用します。")
        return init_chat_model(OLLAMA_MODEL, model_provider="ollama")

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("[setup] ANTHROPIC_API_KEY を検出: Claude (claude-sonnet-5) を使用します。")
        return init_chat_model("claude-sonnet-5", model_provider="anthropic")

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        # 標準入力がTTYでない非対話環境（CI・Dockerのバックグラウンド起動等）では
        # getpass()が入力を待ち続けて無限にブロック（環境によってはEOFErrorで異常終了）するため、
        # その場合は対話入力を試みずに明確なエラーメッセージを出して即座に終了する。
        if not sys.stdin.isatty():
            raise RuntimeError(
                "Ollamaが起動しておらず、ANTHROPIC_API_KEY も OPENAI_API_KEY も未設定です。"
                "非対話環境のため対話入力を求めることができません。"
                "Ollamaを起動するか、環境変数 ANTHROPIC_API_KEY / OPENAI_API_KEY を設定してください。"
            )
        openai_key = getpass.getpass(
            "Ollamaが起動しておらず、ANTHROPIC_API_KEY も OPENAI_API_KEY も未設定です。"
            "OpenAI APIキーを入力してください: "
        )
        os.environ["OPENAI_API_KEY"] = openai_key

    print("[setup] Ollama未起動・ANTHROPIC_API_KEY未設定のため、OpenAI (gpt-5-chat-latest) にフォールバックします。")
    return init_chat_model("gpt-5-chat-latest", model_provider="openai")


model = _build_model()
