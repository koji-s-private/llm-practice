"""setup.py の _build_model() のテスト（Issue #22: 非対話環境でのgetpass()無限ブロック対策）。

`sys.stdin.isatty()` の戻り値を monkeypatch で制御し、実際に標準入力を
待ち受けさせる（テストがハングする）ことがないようにする。
`langchain.chat_models.init_chat_model` は tests/conftest.py で既にフェイクに
差し替え済みのため、実LLM・実APIキー検証は発生しない。
"""

import getpass
import json
import urllib.error

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


# --- CURRENT_PROVIDER（Issue #52: app.pyのエラーメッセージ出し分け用）の更新確認 ---


def test_build_model_sets_current_provider_ollama(monkeypatch):
    monkeypatch.setattr(setup, "_ollama_available", lambda: True)

    setup._build_model()

    assert setup.CURRENT_PROVIDER == "ollama"


def test_build_model_sets_current_provider_anthropic(monkeypatch):
    monkeypatch.setattr(setup, "_ollama_available", lambda: False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy-key")

    setup._build_model()

    assert setup.CURRENT_PROVIDER == "anthropic"


def test_build_model_sets_current_provider_openai(monkeypatch):
    monkeypatch.setattr(setup, "_ollama_available", lambda: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "already-set-key")

    setup._build_model()

    assert setup.CURRENT_PROVIDER == "openai"


# --- _ollama_model_pulled()（Ollama起動済みだがモデル未pullの検出） ---


class _FakeTagsResponse:
    """`urllib.request.urlopen()` の戻り値（コンテキストマネージャ）を模したフェイク。"""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _patch_tags_response(monkeypatch, payload):
    monkeypatch.setattr(
        setup.urllib.request,
        "urlopen",
        lambda url, timeout=None: _FakeTagsResponse(payload),
    )


def test_ollama_model_pulled_true_when_exact_name_present(monkeypatch):
    monkeypatch.setattr(setup, "OLLAMA_MODEL", "llama3.1:latest")
    _patch_tags_response(monkeypatch, {"models": [{"name": "llama3.1:latest"}]})

    assert setup._ollama_model_pulled() is True


def test_ollama_model_pulled_true_when_default_tag_matches_untagged_model(monkeypatch):
    """OLLAMA_MODELにタグが無い場合、暗黙の`:latest`が付与されたモデル名とも一致させる。"""
    monkeypatch.setattr(setup, "OLLAMA_MODEL", "llama3.1")
    _patch_tags_response(monkeypatch, {"models": [{"name": "llama3.1:latest"}]})

    assert setup._ollama_model_pulled() is True


def test_ollama_model_pulled_false_when_model_not_in_list(monkeypatch):
    monkeypatch.setattr(setup, "OLLAMA_MODEL", "llama3.1")
    _patch_tags_response(monkeypatch, {"models": [{"name": "mistral:latest"}]})

    assert setup._ollama_model_pulled() is False


def test_ollama_model_pulled_true_when_api_unreachable(monkeypatch):
    """/api/tagsへの到達自体に失敗した場合は判定不能として安全側（Trueのまま）に倒す。"""
    monkeypatch.setattr(setup, "OLLAMA_MODEL", "llama3.1")

    def _raise(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(setup.urllib.request, "urlopen", _raise)

    assert setup._ollama_model_pulled() is True


def test_ollama_model_pulled_true_when_response_is_not_valid_json(monkeypatch):
    """レスポンスがJSONとしてパースできない場合も安全側（Trueのまま）に倒す。"""
    monkeypatch.setattr(setup, "OLLAMA_MODEL", "llama3.1")

    class _BrokenResponse:
        def read(self):
            return b"not-json"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda url, timeout=None: _BrokenResponse())

    assert setup._ollama_model_pulled() is True


def test_ollama_model_pulled_true_when_response_is_json_array(monkeypatch):
    """レスポンスが妥当なJSONでも最上位がオブジェクトでない（配列）場合は
    スキーマ不一致として安全側（Trueのまま）に倒す（`.get()`呼び出しでの
    AttributeError送出を防ぐ）。"""
    monkeypatch.setattr(setup, "OLLAMA_MODEL", "llama3.1")
    _patch_tags_response(monkeypatch, ["llama3.1:latest"])

    assert setup._ollama_model_pulled() is True


def test_ollama_model_pulled_true_when_response_is_json_string(monkeypatch):
    """レスポンスが妥当なJSONでも最上位がオブジェクトでない（文字列）場合も同様に安全側。"""
    monkeypatch.setattr(setup, "OLLAMA_MODEL", "llama3.1")
    _patch_tags_response(monkeypatch, "unexpected-string-response")

    assert setup._ollama_model_pulled() is True


def test_build_model_falls_back_to_anthropic_when_ollama_model_not_pulled(monkeypatch, capsys):
    """サーバーは起動しているがモデル未pullの場合、Ollamaを候補から除外しフォールバックする。"""
    monkeypatch.setattr(setup, "_ollama_available", lambda: True)
    monkeypatch.setattr(setup, "_ollama_model_pulled", lambda: False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy-key")

    setup._build_model()

    assert setup.CURRENT_PROVIDER == "anthropic"
    assert "見つかりません" in capsys.readouterr().out


def test_ollama_model_pulled_false_when_models_list_is_empty(monkeypatch):
    """境界値: /api/tags には正常に到達したが1件もpull済みモデルが無い場合はFalse。"""
    monkeypatch.setattr(setup, "OLLAMA_MODEL", "llama3.1")
    _patch_tags_response(monkeypatch, {"models": []})

    assert setup._ollama_model_pulled() is False


def test_ollama_model_pulled_false_when_models_key_missing(monkeypatch):
    """境界値: レスポンスJSONは妥当だが"models"キー自体が無いスキーマ不一致の場合もFalse
    （安全側フォールバックは「APIに到達できない」場合のみで、到達できた上での
    スキーマ不一致まで安全側に倒すと誤検出を隠してしまうため）。"""
    monkeypatch.setattr(setup, "OLLAMA_MODEL", "llama3.1")
    _patch_tags_response(monkeypatch, {})

    assert setup._ollama_model_pulled() is False


def test_ollama_model_pulled_false_when_only_different_tag_present(monkeypatch):
    """境界値: 同じベース名でもタグが異なる場合は別モデル扱いでFalse
    （例: OLLAMA_MODEL="llama3.1:8b" なのにpull済みは"llama3.1:latest"のみ）。"""
    monkeypatch.setattr(setup, "OLLAMA_MODEL", "llama3.1:8b")
    _patch_tags_response(monkeypatch, {"models": [{"name": "llama3.1:latest"}]})

    assert setup._ollama_model_pulled() is False


# --- OLLAMA_NUM_CTX（Ollama利用時のコンテキスト長を明示指定し、会話が長引いた際の
# 暗黙の切り捨てを防ぐ） ---


def test_ollama_num_ctx_default_is_a_positive_int():
    """デフォルト値（未設定時）はOllama公式デフォルト(2048)より大きい妥当な正の整数。"""
    assert isinstance(setup.OLLAMA_NUM_CTX, int)
    assert setup.OLLAMA_NUM_CTX > 2048


def test_build_model_ollama_passes_num_ctx_to_init_chat_model(monkeypatch):
    """Ollama利用時は init_chat_model に num_ctx=OLLAMA_NUM_CTX を明示的に渡す。"""
    monkeypatch.setattr(setup, "_ollama_available", lambda: True)
    calls = []

    def _fake_init_chat_model(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(setup, "init_chat_model", _fake_init_chat_model)

    setup._build_model()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == setup.OLLAMA_MODEL
    assert kwargs["model_provider"] == "ollama"
    assert kwargs["num_ctx"] == setup.OLLAMA_NUM_CTX


def test_build_model_falls_back_through_to_runtime_error_when_ollama_model_not_pulled_and_no_keys(
    monkeypatch,
):
    """異常系: Ollamaは起動しているがモデル未pullで、かつクラウド側のAPIキーも
    無い非対話環境の場合、Ollamaが使えないときと同じRuntimeErrorに正しくフォールスルーする
    （モデル未pull検出がOllama以外の分岐に悪影響を与えていないことの確認）。"""
    monkeypatch.setattr(setup, "_ollama_available", lambda: True)
    monkeypatch.setattr(setup, "_ollama_model_pulled", lambda: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(setup.sys, "stdin", _FakeStdin(is_tty=False))

    with pytest.raises(RuntimeError) as exc_info:
        setup._build_model()

    message = str(exc_info.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "OPENAI_API_KEY" in message
