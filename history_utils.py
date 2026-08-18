"""LLMに送信する会話履歴のトークン数ウィンドウイング。

Streamlit版（app.py）とAPI版（api/main.py）の両方から使う共通ロジック。
Ollamaはsetup.OLLAMA_NUM_CTXでコンテキスト長が小さく制限されるため、これを
超えると古い履歴や検索結果が黙って切り捨てられてしまう。それを防ぐため、
LLMへ渡す直前に会話履歴を予算内へ収まるよう直近優先で間引く。
"""

from langchain_core.messages import trim_messages

import setup

# 会話履歴の送信トークン予算。Ollamaはsetup.OLLAMA_NUM_CTXでコンテキスト長が小さく
# 制限されるため、システムプロンプト・検索結果・生成分の余白を差し引いた保守的な値にする。
_OLLAMA_CONTEXT_MARGIN_TOKENS = 5000
_OLLAMA_MIN_HISTORY_TOKENS = 500

# Anthropic/OpenAIはコンテキスト長を制限していないためOllamaより大きい予算を使えるが、
# 無制限にするとAPI利用料・レイテンシが増えるため現実的な上限を設ける。
_API_PROVIDER_HISTORY_TOKENS = 50000

# CURRENT_PROVIDER未設定（想定外のケース）向けの安全側フォールバック値。
_FALLBACK_HISTORY_TOKENS = 3000


def _history_token_budget() -> int:
    """実行時点のsetup.CURRENT_PROVIDERに応じて、会話履歴に割り当てるトークン予算を決める。

    Ollamaのみnum_ctxでコンテキスト長を制限しており、Anthropic/OpenAIは制限が無いため、
    固定値ではなくプロバイダに応じて動的に決める。
    """
    provider = setup.CURRENT_PROVIDER
    if provider == "ollama":
        return max(_OLLAMA_MIN_HISTORY_TOKENS, setup.OLLAMA_NUM_CTX - _OLLAMA_CONTEXT_MARGIN_TOKENS)
    if provider in ("anthropic", "openai"):
        return _API_PROVIDER_HISTORY_TOKENS
    return _FALLBACK_HISTORY_TOKENS


def _windowed_history(messages: list) -> list:
    """会話履歴をトークン予算内に収まるようウィンドウイングする（直近優先）。

    呼び出し元が保持する画面表示用・API応答用の履歴はそのまま保持しつつ、LLMへの送信直前
    だけ直近のやりとりに絞り込む。start_on="human" により、絞り込んだ結果の先頭が必ず
    HumanMessageになるようにする（エージェントが要求する会話構造を壊さないため）。
    """
    if not messages:
        return messages
    return trim_messages(
        messages,
        max_tokens=_history_token_budget(),
        token_counter="approximate",
        strategy="last",
        start_on="human",
    )
