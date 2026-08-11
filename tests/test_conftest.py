"""tests/conftest.py の `_setdefault_even_if_empty()` ヘルパー関数の単体テスト。

`os.environ.setdefault()` は環境変数が「未設定」の場合のみ値をセットするため、
実行環境側で空文字列が事前設定されているケースに対応できない。この問題を回避する
専用ヘルパーの挙動（未設定・空文字列・既に値がある、の3パターン）を検証する。
"""

import os

import conftest

_ENV_VAR_NAME = "TEST_CONFTEST_HELPER_ENV_VAR"


def test_setdefault_even_if_empty_sets_value_when_unset(monkeypatch):
    """環境変数が未設定の場合、デフォルト値がセットされる（正常系）。"""
    monkeypatch.delenv(_ENV_VAR_NAME, raising=False)

    conftest._setdefault_even_if_empty(_ENV_VAR_NAME, "default-value")

    assert os.environ[_ENV_VAR_NAME] == "default-value"


def test_setdefault_even_if_empty_overwrites_empty_string(monkeypatch):
    """環境変数が空文字列で事前設定されている場合、デフォルト値で上書きされる（境界値）。"""
    monkeypatch.setenv(_ENV_VAR_NAME, "")

    conftest._setdefault_even_if_empty(_ENV_VAR_NAME, "default-value")

    assert os.environ[_ENV_VAR_NAME] == "default-value"


def test_setdefault_even_if_empty_keeps_existing_non_empty_value(monkeypatch):
    """環境変数に既に空でない値が設定されている場合、上書きされない（異常系にならないことの確認）。"""
    monkeypatch.setenv(_ENV_VAR_NAME, "already-configured-value")

    conftest._setdefault_even_if_empty(_ENV_VAR_NAME, "default-value")

    assert os.environ[_ENV_VAR_NAME] == "already-configured-value"
