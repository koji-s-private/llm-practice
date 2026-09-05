"""history_utils.py の日本語（CJK）比率に応じたトークン数見積もりのユニットテスト。

`_windowed_history()` 経由の統合的な間引き挙動（トークン予算超過時の挙動）は
`tests/test_app.py` / `tests/test_api.py` で既にカバーされているため、ここでは
CJK比率推定まわりの純粋関数（`_cjk_ratio` / `_effective_chars_per_token` /
`_count_tokens_ja_aware`）を直接検証する。
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

import history_utils

# --- _cjk_ratio ---


def test_cjk_ratio_is_zero_for_pure_english_text():
    """正常系: 英語のみのメッセージはCJK比率0になる。"""
    messages = [HumanMessage(content="Hello world, this is English text.")]
    assert history_utils._cjk_ratio(messages) == 0.0


def test_cjk_ratio_is_close_to_one_for_pure_japanese_text():
    """正常系: 日本語（ひらがな・漢字）のみのメッセージはCJK比率がほぼ1になる。"""
    messages = [HumanMessage(content="これは日本語の文章です")]
    assert history_utils._cjk_ratio(messages) > 0.9


def test_cjk_ratio_is_between_zero_and_one_for_mixed_text():
    """境界値: 英語と日本語が混在する場合、比率は0と1の間になる。"""
    messages = [HumanMessage(content="Hello こんにちは")]
    ratio = history_utils._cjk_ratio(messages)
    assert 0.0 < ratio < 1.0


def test_cjk_ratio_is_zero_for_empty_messages():
    """境界値: 空のメッセージ一覧では0除算を起こさず0を返す。"""
    assert history_utils._cjk_ratio([]) == 0.0


def test_cjk_ratio_detects_katakana_and_fullwidth_characters():
    """正常系: カタカナ・全角英数記号もCJK文字として検出される。"""
    messages = [HumanMessage(content="カタカナ＆全角文字")]
    assert history_utils._cjk_ratio(messages) == 1.0


def test_cjk_ratio_handles_multimodal_list_content():
    """異常系/境界値: contentがマルチモーダル（list）の場合もtextブロックを連結して判定する。"""
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "日本語のテキストブロック"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
            ]
        )
    ]
    assert history_utils._cjk_ratio(messages) > 0.9


# --- _effective_chars_per_token ---


def test_effective_chars_per_token_uses_latin_default_for_english():
    """正常系: 英語のみのメッセージでは既定値4.0（英語想定）がそのまま使われる。"""
    messages = [HumanMessage(content="Hello world, this is English text.")]
    assert history_utils._effective_chars_per_token(messages) == history_utils._CHARS_PER_TOKEN_LATIN


def test_effective_chars_per_token_uses_cjk_value_for_japanese():
    """正常系: 日本語のみのメッセージでは安全側の1.0（CJK想定）に近づく。"""
    messages = [HumanMessage(content="これは日本語の文章です")]
    assert history_utils._effective_chars_per_token(messages) < 1.5


def test_effective_chars_per_token_interpolates_linearly_for_mixed_text():
    """正常系: 英日混在テキストでは4.0と1.0の間の値になる（線形補間）。"""
    messages = [HumanMessage(content="Hello こんにちは")]
    value = history_utils._effective_chars_per_token(messages)
    assert history_utils._CHARS_PER_TOKEN_CJK < value < history_utils._CHARS_PER_TOKEN_LATIN


# --- _count_tokens_ja_aware ---


def test_count_tokens_ja_aware_estimates_more_tokens_than_latin_default_for_japanese():
    """正常系: 同じ文字数でも、日本語テキストは既定のchars_per_token=4.0による
    見積もりより多いトークン数として概算される（過小評価の是正）。"""
    messages = [HumanMessage(content="あ" * 400)]

    ja_aware = history_utils._count_tokens_ja_aware(messages)
    latin_default = count_tokens_approximately(messages, chars_per_token=history_utils._CHARS_PER_TOKEN_LATIN)

    assert ja_aware > latin_default


def test_count_tokens_ja_aware_matches_plain_approximation_for_english():
    """正常系: 英語のみの場合、"approximate"（chars_per_token既定4.0）と同じ結果になる。"""
    messages = [HumanMessage(content="Hello world"), AIMessage(content="Hi there")]

    assert history_utils._count_tokens_ja_aware(messages) == count_tokens_approximately(messages)
