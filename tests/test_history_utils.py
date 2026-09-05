"""history_utils.py の日本語（CJK）比率に応じたトークン数見積もりのユニットテスト。

`_windowed_history()` 経由の統合的な間引き挙動（トークン予算超過時の挙動）は
`tests/test_app.py` / `tests/test_api.py` で既にカバーされているため、ここでは
CJK比率推定まわりの純粋関数（`_cjk_ratio` / `_effective_chars_per_token` /
`_count_tokens_ja_aware`）を直接検証する。ただし、旧ロジック（chars_per_token=4.0
固定）が日本語主体の会話で予算超過を見逃していたという今回の修正の核心（Issue #231）
については、`_windowed_history` 自体の回帰テストも本ファイルに含める。
"""

from langchain_core.messages import AIMessage, HumanMessage, trim_messages
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


def test_cjk_ratio_is_zero_for_messages_with_empty_string_content():
    """境界値: contentが空文字列のメッセージのみでも0除算を起こさず0を返す。"""
    messages = [HumanMessage(content=""), AIMessage(content="")]
    assert history_utils._cjk_ratio(messages) == 0.0


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


def test_effective_chars_per_token_follows_linear_formula_exactly():
    """正常系: CJK比率から4.0 + ratio * (1.0 - 4.0) という線形補間の式通りの値になる
    （範囲チェックだけでなく、係数のズレ等の実装ミスも検知できるようにする）。"""
    messages = [HumanMessage(content="Hello こんにちは")]
    ratio = history_utils._cjk_ratio(messages)
    expected = history_utils._CHARS_PER_TOKEN_LATIN + ratio * (
        history_utils._CHARS_PER_TOKEN_CJK - history_utils._CHARS_PER_TOKEN_LATIN
    )
    assert history_utils._effective_chars_per_token(messages) == expected


def test_effective_chars_per_token_is_latin_default_for_empty_messages():
    """境界値: 空のメッセージ一覧ではCJK比率0とみなされ、既定値4.0になる。"""
    assert history_utils._effective_chars_per_token([]) == history_utils._CHARS_PER_TOKEN_LATIN


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


def test_count_tokens_ja_aware_returns_zero_for_empty_messages():
    """境界値: 空のメッセージ一覧でも例外を起こさず0を返す。"""
    assert history_utils._count_tokens_ja_aware([]) == 0


# --- _windowed_history（旧ロジックとの回帰比較） ---


def test_windowed_history_trims_japanese_history_that_old_fixed_ratio_would_have_missed(monkeypatch):
    """回帰: 旧ロジック（trim_messagesのtoken_counter="approximate"、chars_per_token=4.0固定）
    では予算内と誤判定され間引かれなかったであろう日本語主体の会話履歴が、CJK比率を
    考慮する新ロジック（_count_tokens_ja_aware）では正しく間引かれる（Issue #231）。"""
    import setup

    monkeypatch.setattr(setup, "CURRENT_PROVIDER", "ollama")
    monkeypatch.setattr(setup, "OLLAMA_NUM_CTX", 8192)

    long_text = "あ" * 1000
    messages = []
    for i in range(4):
        messages.append(HumanMessage(content=f"質問{i}: {long_text}"))
        messages.append(AIMessage(content=f"回答{i}: {long_text}"))

    budget = history_utils._history_token_budget()

    # 前提確認: 旧ロジック（chars_per_token=4.0固定）ではこの会話量は予算内と
    # 誤判定され、trim_messagesは1件も間引かない。
    old_result = trim_messages(
        messages, max_tokens=budget, token_counter="approximate", strategy="last", start_on="human"
    )
    assert old_result == messages

    # 新ロジックでは同じ会話量・同じ予算でも正しく超過と判定され間引かれる。
    windowed = history_utils._windowed_history(messages)
    assert len(windowed) < len(messages)
    assert isinstance(windowed[0], HumanMessage)
    assert not any("質問0" in m.content for m in windowed)


def test_windowed_history_returns_empty_list_as_is():
    """境界値: 会話履歴が空の場合はそのまま空リストを返す（trim_messages呼び出し自体を省略）。"""
    assert history_utils._windowed_history([]) == []
