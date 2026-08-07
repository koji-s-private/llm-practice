"""setup.py の _build_model() のテスト（Issue #22: 非対話環境でのgetpass()無限ブロック対策）。

`sys.stdin.isatty()` の戻り値を monkeypatch で制御し、実際に標準入力を
待ち受けさせる（テストがハングする）ことがないようにする。
`langchain.chat_models.init_chat_model` は tests/conftest.py で既にフェイクに
差し替え済みのため、実LLM・実APIキー検証は発生しない。
"""

import getpass

import pytest

import setup


class _FakeStdin:
    """`sys.stdin` の代わりに使う、`isatty()` のみを持つダミー標準入力。"""

    def __init__(self, is_tty):
        self._is_tty = is_tty

    def isatty(self):
        return self._is_tty


def _forbid_isatty():
    """isatty() が呼ばれたらテスト失敗させるためのダミー標準入力。

    Ollama検出時・APIキー設定済み時は、非対話環境かどうかの判定自体に
    到達しないはず（従来の挙動に影響しない）ことを確認するために使う。
    """

    class _Forbidden:
        def isatty(self):
            raise AssertionError("この分岐では sys.stdin.isatty() は呼ばれないはず")

    return _Forbidden()


def _clear_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DISABLE_OLLAMA", "true")


# --- 異常系: 非対話環境ではgetpass()を呼ばずRuntimeErrorを送出する ---


def test_build_model_raises_runtime_error_in_non_interactive_env_without_any_key(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setattr(setup.sys, "stdin", _FakeStdin(is_tty=False))

    def _fail_if_called(prompt):
        raise AssertionError("非対話環境では getpass.getpass() を呼んではいけない")

    monkeypatch.setattr(setup.getpass, "getpass", _fail_if_called)

    with pytest.raises(RuntimeError) as exc_info:
        setup._build_model()

    message = str(exc_info.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "OPENAI_API_KEY" in message
    assert "Ollama" in message


def test_build_model_non_interactive_error_does_not_set_openai_api_key_env(monkeypatch):
    """異常終了時に OPENAI_API_KEY が中途半端に環境変数へセットされていないことを確認する。"""
    _clear_keys(monkeypatch)
    monkeypatch.setattr(setup.sys, "stdin", _FakeStdin(is_tty=False))
    monkeypatch.setattr(setup.getpass, "getpass", lambda prompt: "should-not-be-called")

    with pytest.raises(RuntimeError):
        setup._build_model()

    assert "OPENAI_API_KEY" not in setup.os.environ


# --- 正常系: 対話環境（TTY）では従来通りgetpass()が呼ばれる ---


def test_build_model_calls_getpass_in_interactive_env_without_any_key(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setattr(setup.sys, "stdin", _FakeStdin(is_tty=True))

    calls = []

    def _fake_getpass(prompt):
        calls.append(prompt)
        return "dummy-openai-key-from-getpass"

    monkeypatch.setattr(setup.getpass, "getpass", _fake_getpass)

    model = setup._build_model()

    assert len(calls) == 1
    assert setup.os.environ["OPENAI_API_KEY"] == "dummy-openai-key-from-getpass"
    assert model is not None


# --- 正常系: 既存の優先順位分岐は非対話判定の影響を受けない ---


def test_build_model_ollama_detected_skips_isatty_check(monkeypatch):
    """Ollama検出時はキー確認自体に到達しないため、isatty()判定も行われない。"""
    monkeypatch.setattr(setup, "_ollama_available", lambda: True)
    monkeypatch.setattr(setup.sys, "stdin", _forbid_isatty())

    model = setup._build_model()

    assert model is not None


def test_build_model_anthropic_key_set_skips_isatty_check(monkeypatch):
    """ANTHROPIC_API_KEY設定済みの場合もキー確認自体に到達しないため、isatty()判定は行われない。"""
    monkeypatch.setattr(setup, "_ollama_available", lambda: False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy-key")
    monkeypatch.setattr(setup.sys, "stdin", _forbid_isatty())

    model = setup._build_model()

    assert model is not None


def test_build_model_openai_key_already_set_skips_isatty_check_and_getpass(monkeypatch):
    """OPENAI_API_KEYが既に設定済みの場合は、getpass()もisatty()判定も行われない（境界値）。"""
    monkeypatch.setattr(setup, "_ollama_available", lambda: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "already-set-key")
    monkeypatch.setattr(setup.sys, "stdin", _forbid_isatty())

    def _fail_if_called(prompt):
        raise AssertionError("OPENAI_API_KEY が既に設定済みの場合は getpass() を呼んではいけない")

    monkeypatch.setattr(setup.getpass, "getpass", _fail_if_called)

    model = setup._build_model()

    assert model is not None
    assert setup.os.environ["OPENAI_API_KEY"] == "already-set-key"


# --- getpass モジュールの参照そのものが変わっていないことの確認（回帰防止） ---


def test_setup_module_still_imports_getpass_module_directly():
    """setup.py が `import getpass`（`from getpass import getpass` ではない）を維持していることを確認する。

    テストでの `monkeypatch.setattr(setup.getpass, "getpass", ...)` が意味を持つための前提条件。
    """
    assert setup.getpass is getpass
