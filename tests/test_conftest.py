"""tests/conftest.py の `_setdefault_even_if_empty()` ヘルパー関数の単体テスト。

`os.environ.setdefault()` は環境変数が「未設定」の場合のみ値をセットするため、
実行環境側で空文字列が事前設定されているケースに対応できない。この問題を回避する
専用ヘルパーの挙動（未設定・空文字列・既に値がある、の3パターン）を検証する。

また、`getpass.getpass()` が未想定のタイミングで呼ばれた場合に無応答（ハング）ではなく
即座にテスト失敗させる autouse fixture `_forbid_unexpected_getpass` の挙動も検証する。
"""

import getpass
import os

import conftest
import pytest

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


# --- autouse fixture `_forbid_unexpected_getpass` の挙動 ---
#
# `getpass.getpass()` は本来「非対話環境では呼ばれてはいけない」呼び出しであり、
# もし呼ばれてしまうと無応答（ハング）になりうる。autouse fixture により、
# 呼ばれた瞬間に例外で即座に検知できることを確認する。


def test_forbid_unexpected_getpass_raises_runtime_error_when_called():
    """異常系: 未想定の getpass.getpass() 呼び出しは無応答ではなく即座にRuntimeErrorになる。"""
    with pytest.raises(RuntimeError) as exc_info:
        getpass.getpass("password: ")

    assert "getpass.getpass()" in str(exc_info.value)


def test_forbid_unexpected_getpass_message_hints_at_conftest_env_var_gap():
    """異常系: 例外メッセージが conftest.py のダミー環境変数対応漏れを示唆する内容であること。

    調査時に原因（conftest.pyの環境変数設定漏れ）へすぐたどり着けるようにするため、
    メッセージ内容自体も検証する。
    """
    with pytest.raises(RuntimeError) as exc_info:
        getpass.getpass()

    message = str(exc_info.value)
    assert "conftest.py" in message
    assert "環境変数" in message


def test_forbid_unexpected_getpass_can_be_overridden_by_test_level_monkeypatch(monkeypatch):
    """境界値/回帰: テスト側で個別に monkeypatch.setattr してもautouse fixtureと衝突しない。

    tests/test_setup.py 内のisatty分岐テスト等は、テスト内で個別に
    `monkeypatch.setattr(setup.getpass, "getpass", ...)` を行っており、autouse fixtureの
    後段で上書きされることで従来通り動作する。同等のことを最小構成で確認する。
    """
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "overridden-value")

    assert getpass.getpass("password: ") == "overridden-value"


def test_forbid_unexpected_getpass_restored_after_scoped_override_ends(monkeypatch):
    """境界値: `monkeypatch.context()` によるスコープ限定の一時的な上書きが終了すると、
    autouse fixtureによる安全策（例外送出）に戻ることを確認する
    （`with`ブロックを抜けると、その中で行った setattr のみが取り消される）。"""
    with monkeypatch.context() as scoped:
        scoped.setattr(getpass, "getpass", lambda prompt="": "temporary-value")
        assert getpass.getpass() == "temporary-value"

    with pytest.raises(RuntimeError):
        getpass.getpass()


def test_forbid_unexpected_getpass_is_reapplied_for_every_test():
    """境界値: 別のテスト関数でも毎回autouse fixtureが再適用され、安全策が有効であることを確認する
    （前のテスト内での一時的な上書きが漏れて次のテストに影響しないことの回帰防止）。"""
    with pytest.raises(RuntimeError):
        getpass.getpass()
