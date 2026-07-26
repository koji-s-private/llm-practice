import getpass
import os
import socket

from langchain.chat_models import init_chat_model

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"

# print("LangSmith 設定確認：")
# print("LANGSMITH_TRACING =", os.getenv("LANGSMITH_TRACING"))
# print("LANGSMITH_API_KEY =", "設定済み" if os.getenv("LANGSMITH_API_KEY") else "未設定")
# print("LANGSMITH_PROJECT =", os.getenv("LANGSMITH_PROJECT"))


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
        openai_key = getpass.getpass(
            "Ollamaが起動しておらず、ANTHROPIC_API_KEY も OPENAI_API_KEY も未設定です。"
            "OpenAI APIキーを入力してください: "
        )
        os.environ["OPENAI_API_KEY"] = openai_key

    print("[setup] Ollama未起動・ANTHROPIC_API_KEY未設定のため、OpenAI (gpt-5-chat-latest) にフォールバックします。")
    return init_chat_model("gpt-5-chat-latest", model_provider="openai")


model = _build_model()
