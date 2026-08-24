import getpass
import json
import os
import socket
import sys
import urllib.error
import urllib.request

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

# Ollamaはnum_ctx未指定だと多くのモデルで2048程度の小さいコンテキスト長がデフォルトになり、
# 会話が数往復続くだけで古い履歴や検索結果が暗黙的に切り捨てられるため、一般的なローカルPC
# （8GB〜のメモリ）でも現実的に動かせる範囲でOllama公式デフォルトより十分な余裕を持たせた値を
# 明示的に指定する。
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

# 現在実際に使用しているプロバイダ名（"ollama" / "anthropic" / "openai"）。
# _build_model() 実行時に確定させ、app.py 側から参照してエラーメッセージの出し分けに使う。
CURRENT_PROVIDER: str | None = None

# 現在実際に使用しているモデル名（例: "llama3.1"、"claude-sonnet-5"）。
# _build_model() 実行時に CURRENT_PROVIDER とあわせて確定させ、app.py がサイドバーの
# 使用中モデル表示に使う。
CURRENT_MODEL_NAME: str | None = None

# Ollamaが利用できず有料APIにフォールバックした場合の具体的な理由（未起動 / モデル未pull）。
# app.py が起動直後の警告バナー表示に使う。Ollamaをそのまま使用できた場合はNoneのまま。
CURRENT_PROVIDER_FALLBACK_REASON: str | None = None


def _ollama_available() -> bool:
    """ローカルでOllamaサーバーが起動しているかを軽くチェックする（起動が遅くならないよう短いタイムアウト）。"""
    if os.environ.get("DISABLE_OLLAMA") == "true":
        return False
    try:
        with socket.create_connection((OLLAMA_HOST, OLLAMA_PORT), timeout=0.3):
            return True
    except OSError:
        return False


def _ollama_model_pulled() -> bool:
    """OLLAMA_MODELがOllamaに実際にpull済みかを `/api/tags` で確認する。

    未pullだとモデル呼び出し時に初めて"model not found"エラーになるため事前検出する。
    Ollamaのモデル名はタグ付き（例: "llama3.1:latest"）で返るため、OLLAMA_MODELに
    タグが無い場合は暗黙のデフォルトタグ "latest" を補って比較する。APIへの到達自体に
    失敗した場合は判定不能なだけなので、安全側（pull済みとみなす）に倒す。
    """
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=0.5) as response:
            data = json.loads(response.read())
    except (OSError, urllib.error.URLError, ValueError):
        return True

    # 最上位がオブジェクトでない場合（配列・文字列など）もスキーマ不一致として判定不能扱いにする。
    if not isinstance(data, dict):
        return True

    names = {model.get("name", "") for model in data.get("models", [])}
    candidates = {OLLAMA_MODEL}
    if ":" not in OLLAMA_MODEL:
        candidates.add(f"{OLLAMA_MODEL}:latest")
    return bool(names & candidates)


def _build_model():
    """優先順位: 1) Ollama（無料・ローカル） 2) ANTHROPIC_API_KEY 3) OPENAI_API_KEY。

    選定したプロバイダ名・モデル名はモジュールレベル変数 CURRENT_PROVIDER / CURRENT_MODEL_NAME
    にも記録する（app.py が agent.invoke() 失敗時のエラーメッセージ出し分けや、サイドバーの
    使用中モデル表示に使う）。
    Ollamaが利用できずフォールバックした場合は、その理由を CURRENT_PROVIDER_FALLBACK_REASON にも
    記録する（app.py が起動直後の警告バナーで、ユーザーがOllama側を復旧しやすいように使う）。
    """
    global CURRENT_PROVIDER, CURRENT_MODEL_NAME, CURRENT_PROVIDER_FALLBACK_REASON

    CURRENT_PROVIDER_FALLBACK_REASON = None

    if _ollama_available():
        if _ollama_model_pulled():
            print(f"[setup] Ollama を検出: {OLLAMA_MODEL}（ローカル・無料、num_ctx={OLLAMA_NUM_CTX}）を使用します。")
            CURRENT_PROVIDER = "ollama"
            CURRENT_MODEL_NAME = OLLAMA_MODEL
            return init_chat_model(OLLAMA_MODEL, model_provider="ollama", num_ctx=OLLAMA_NUM_CTX)
        CURRENT_PROVIDER_FALLBACK_REASON = (
            f"Ollamaは起動していますが、モデル '{OLLAMA_MODEL}' が見つかりません（pull未実施の可能性）。"
            f"'ollama pull {OLLAMA_MODEL}' を実行するか、OLLAMA_MODEL を既存のモデル名に変更してください。"
        )
        print(f"[setup] {CURRENT_PROVIDER_FALLBACK_REASON}")
    else:
        CURRENT_PROVIDER_FALLBACK_REASON = (
            "Ollamaサーバーに接続できません（未起動の可能性）。'ollama serve' 等で起動してください。"
        )
        print(f"[setup] {CURRENT_PROVIDER_FALLBACK_REASON}")

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("[setup] ANTHROPIC_API_KEY を検出: Claude (claude-sonnet-5) を使用します。")
        CURRENT_PROVIDER = "anthropic"
        CURRENT_MODEL_NAME = "claude-sonnet-5"
        return init_chat_model("claude-sonnet-5", model_provider="anthropic")

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        # 非対話環境（CI・Dockerのバックグラウンド起動等）ではgetpass()が入力を待ち続けて
        # 無限にブロックするため、その場合は対話入力を試みずエラーメッセージを出して終了する。
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
    CURRENT_PROVIDER = "openai"
    CURRENT_MODEL_NAME = "gpt-5-chat-latest"
    return init_chat_model("gpt-5-chat-latest", model_provider="openai")


def current_model_label() -> str:
    """サイドバー表示用に「プロバイダ名 (モデル名)」形式の文字列を返す（例: "Ollama (llama3.1)"）。"""
    provider_labels = {"ollama": "Ollama", "anthropic": "Anthropic", "openai": "OpenAI"}
    provider_label = provider_labels.get(CURRENT_PROVIDER, CURRENT_PROVIDER or "不明")
    return f"{provider_label} ({CURRENT_MODEL_NAME})"


model = _build_model()
