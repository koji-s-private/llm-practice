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

# 外部APIプロバイダの候補モデル名。実在しないモデルIDを指定するとAPI呼び出し時に
# エラーになるため、_build_model() と list_available_models() / build_chat_model() の
# 選択肢を必ずこの定数で一致させる。
ANTHROPIC_MODEL = "claude-sonnet-5"
OPENAI_MODEL = "gpt-5-chat-latest"

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


def _fetch_ollama_pulled_model_names() -> set[str] | None:
    """Ollamaの `/api/tags` からpull済みモデル名一覧（タグ付き、例: "llama3.1:latest"）を取得する。

    APIへの到達自体に失敗した場合やスキーマ不一致の場合は判定不能としてNoneを返す
    （「pull済みモデルが0件」と「取得に失敗した」を呼び出し元が区別できるようにするため）。
    """
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=0.5) as response:
            data = json.loads(response.read())
    except (OSError, urllib.error.URLError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return {model.get("name", "") for model in data.get("models", [])}


def _ollama_pulled_model_names() -> set[str]:
    """一覧表示用途のpull済みモデル名一覧。取得失敗時は空集合を返す。"""
    return _fetch_ollama_pulled_model_names() or set()


def _ollama_model_pulled() -> bool:
    """OLLAMA_MODELがOllamaに実際にpull済みかを確認する。

    未pullだとモデル呼び出し時に初めて"model not found"エラーになるため事前検出する。
    Ollamaのモデル名はタグ付き（例: "llama3.1:latest"）で返るため、OLLAMA_MODELに
    タグが無い場合は暗黙のデフォルトタグ "latest" を補って比較する。一覧取得自体に
    失敗した場合は判定不能なだけなので、安全側（pull済みとみなす）に倒す。
    """
    names = _fetch_ollama_pulled_model_names()
    if names is None:
        return True

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
        print(f"[setup] ANTHROPIC_API_KEY を検出: Claude ({ANTHROPIC_MODEL}) を使用します。")
        CURRENT_PROVIDER = "anthropic"
        CURRENT_MODEL_NAME = ANTHROPIC_MODEL
        return init_chat_model(ANTHROPIC_MODEL, model_provider="anthropic")

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

    print(f"[setup] Ollama未起動・ANTHROPIC_API_KEY未設定のため、OpenAI ({OPENAI_MODEL}) にフォールバックします。")
    CURRENT_PROVIDER = "openai"
    CURRENT_MODEL_NAME = OPENAI_MODEL
    return init_chat_model(OPENAI_MODEL, model_provider="openai")


def list_available_models() -> list[dict]:
    """UI上のモデル切替選択肢として提示できるモデル一覧を返す。

    APIキーが未設定のプロバイダは、ユーザーが意図せず課金対象のAPIを呼び出して
    しまわないよう一覧に含めない。各要素は {"provider": ..., "model": ...} の形式。
    """
    models = [{"provider": "ollama", "model": name} for name in sorted(_ollama_pulled_model_names())]
    if os.environ.get("ANTHROPIC_API_KEY"):
        models.append({"provider": "anthropic", "model": ANTHROPIC_MODEL})
    if os.environ.get("OPENAI_API_KEY"):
        models.append({"provider": "openai", "model": OPENAI_MODEL})
    return models


def build_chat_model(provider: str, model_name: str):
    """指定されたprovider/model名から実際にチャットモデルインスタンスを構築する。

    UIでのモデル切替時に、_build_model()と一貫したプロバイダごとの構築ロジックを
    使い回すための関数（Ollamaサーバー自体の再起動は不要で、モデル指定を切り替えるだけでよい）。
    """
    if provider == "ollama":
        return init_chat_model(model_name, model_provider="ollama", num_ctx=OLLAMA_NUM_CTX)
    if provider in ("anthropic", "openai"):
        return init_chat_model(model_name, model_provider=provider)
    raise ValueError(f"未対応のプロバイダです: {provider}")


PROVIDER_LABELS = {"ollama": "Ollama", "anthropic": "Anthropic", "openai": "OpenAI"}


def model_label(provider: str | None, model_name: str | None) -> str:
    """「プロバイダ名 (モデル名)」形式の表示ラベルを組み立てる（例: "Ollama (llama3.1)"）。"""
    provider_label = PROVIDER_LABELS.get(provider, provider or "不明")
    return f"{provider_label} ({model_name})"


def current_model_label() -> str:
    """起動時に自動選択されたモデルのサイドバー表示用ラベルを返す。

    UIでユーザーが手動でモデルを切り替えた後の表示にはこの関数ではなく、
    切替後の状態を渡した model_label() を使う（本関数はCURRENT_PROVIDER/
    CURRENT_MODEL_NAMEという起動時固定値しか参照しないため）。
    """
    return model_label(CURRENT_PROVIDER, CURRENT_MODEL_NAME)


model = _build_model()
