"""LLMに送信する会話履歴のトークン数ウィンドウイング。

Streamlit版（app.py）とAPI版（api/main.py）の両方から使う共通ロジック。
Ollamaはsetup.OLLAMA_NUM_CTXでコンテキスト長が小さく制限されるため、これを
超えると古い履歴や検索結果が黙って切り捨てられてしまう。それを防ぐため、
LLMへ渡す直前に会話履歴を予算内へ収まるよう直近優先で間引く。
"""

import re

from langchain_core.messages import BaseMessage, trim_messages
from langchain_core.messages.utils import convert_to_messages, count_tokens_approximately

import setup

# ひらがな・カタカナ・CJK統一漢字（拡張A・互換漢字含む）・全角英数記号。
_CJK_PATTERN = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")

# count_tokens_approximatelyの既定値（英語想定で1トークン≈4文字）。
_CHARS_PER_TOKEN_LATIN = 4.0
# バイト単位BPEトークナイザでは日本語1文字が複数トークンに分割されることも珍しくないため、
# 安全側の見積もりとして1文字=1トークンとみなす。
_CHARS_PER_TOKEN_CJK = 1.0

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


def _message_text(message: BaseMessage) -> str:
    """メッセージのcontentをCJK比率計算用にベストエフォートで文字列化する。"""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


def _cjk_ratio(messages: list[BaseMessage]) -> float:
    """メッセージ群の全文字数に対するCJK文字数の比率を返す。"""
    text = "".join(_message_text(m) for m in convert_to_messages(messages))
    if not text:
        return 0.0
    cjk_chars = len(_CJK_PATTERN.findall(text))
    return cjk_chars / len(text)


def _effective_chars_per_token(messages: list[BaseMessage]) -> float:
    """CJK比率に応じてchars_per_tokenを4.0（英語想定）〜1.0（日本語想定）で線形補間する。"""
    ratio = _cjk_ratio(messages)
    return _CHARS_PER_TOKEN_LATIN + ratio * (_CHARS_PER_TOKEN_CJK - _CHARS_PER_TOKEN_LATIN)


def _count_tokens_ja_aware(messages: list[BaseMessage]) -> int:
    """count_tokens_approximatelyを日本語比率に応じたchars_per_tokenで呼び出す。"""
    return count_tokens_approximately(messages, chars_per_token=_effective_chars_per_token(messages))


def _windowed_history(messages: list) -> list:
    """会話履歴をトークン予算内に収まるようウィンドウイングする（直近優先）。

    呼び出し元が保持する画面表示用・API応答用の履歴はそのまま保持しつつ、LLMへの送信直前
    だけ直近のやりとりに絞り込む。start_on="human" により、絞り込んだ結果の先頭が必ず
    HumanMessageになるようにする（エージェントが要求する会話構造を壊さないため）。
    token_counterには"approximate"（chars_per_token固定4.0、英語想定）ではなく、
    日本語比率に応じてchars_per_tokenを動的に下げる_count_tokens_ja_awareを渡し、
    予算超過を防ぐ。
    """
    if not messages:
        return messages
    return trim_messages(
        messages,
        max_tokens=_history_token_budget(),
        token_counter=_count_tokens_ja_aware,
        strategy="last",
        start_on="human",
    )
