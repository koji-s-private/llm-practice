"""pytest共通設定。

本プロジェクトの `rag_chain.py` / `setup.py` は、実行時に埋め込みモデル
（sentence-transformers, 数百MB規模）やベクトルDB(Chroma)、LLMプロバイダの
実クライアントを読み込む/構築する。これらはテスト対象のロジック
（差分同期・関連度採点・フォールバック挙動）そのものには不要なため、
テストではこれらを軽量なフェイクに差し替える。

- `langchain_chroma` / `langchain_huggingface` はインストールしていないため、
  最小限のフェイクモジュールを sys.modules に登録する（未使用のプレースホルダ）。
- `setup.py` は起動時にモデルを自動選択するが、`ANTHROPIC_API_KEY` 等が未設定だと
  `getpass.getpass()` で対話入力を要求してテストがハングするため、ダミー値を設定する。
  実行環境によっては該当の環境変数が「未設定」ではなく空文字列で事前設定されている
  場合があり、`os.environ.setdefault()` は空文字列に対しては上書きしない
  （「未設定」の場合のみ有効なため）ため、空文字列も上書き対象にする
  `_setdefault_even_if_empty()` を使う。
- `langchain.chat_models.init_chat_model` と `langchain.agents.create_agent` は
  実際のLLM呼び出し・エージェント構築を避けるためフェイクに差し替える。
- 上記のダミー環境変数は `setup.py` が将来新たな環境変数に依存するよう変更された場合に
  対応漏れが起こりうる。対応漏れがあると非対話環境でも `getpass.getpass()` に到達し
  無応答（ハング）になりうるため、`getpass.getpass` 自体もデフォルトで例外を送出する
  ように差し替え、対応漏れを無応答ではなく即座のテスト失敗として検知できるようにする
  （`getpass.getpass()` の呼び出しを検証する側のテストは、そのテスト内で
  `monkeypatch.setattr` を使って個別に上書きする）。
"""

import getpass
import os
import sys
import types

import pytest


def _setdefault_even_if_empty(name: str, default: str) -> None:
    """環境変数が未設定、または空文字列の場合にダミー値で上書きする。

    `os.environ.setdefault()` は環境変数が「未設定」の場合のみ値をセットするため、
    実行環境側で空文字列（例: `ANTHROPIC_API_KEY=""`）が事前設定されていると
    上書きされず、`setup.py` の `getpass.getpass()` に処理が進んでテストが
    ハングしうる。これを避けるため空文字列も上書き対象にする。
    """
    if not os.environ.get(name):
        os.environ[name] = default


_setdefault_even_if_empty("DISABLE_OLLAMA", "true")
_setdefault_even_if_empty("ANTHROPIC_API_KEY", "test-dummy-key")
_setdefault_even_if_empty("LANGSMITH_API_KEY", "test-dummy-key")
_setdefault_even_if_empty("LANGSMITH_PROJECT", "test")


def _install_stub_module(name: str, **attrs) -> types.ModuleType:
    """未インストールのパッケージにのみプレースホルダを登録する。

    開発者が `requirements.txt` をフルインストールした環境で実行した場合に、
    既にインポート済みの実パッケージを誤って上書きしないようにする。
    """
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _FakeChatModel:
    """setup.model の代わりに使うダミーのチャットモデル。

    テスト側で `.invoke` を monkeypatch してから使う想定のため、
    そのままの呼び出しはテストの意図しない実LLM依存を防ぐために失敗させる。
    """

    def invoke(self, *args, **kwargs):
        raise AssertionError("model.invoke() は各テストで monkeypatch してから呼び出してください")


def _fake_init_chat_model(*args, **kwargs):
    return _FakeChatModel()


def _fake_create_agent(model, tools, system_prompt):
    return types.SimpleNamespace(model=model, tools=tools, system_prompt=system_prompt)


# rag_chain.py が要求する重い外部パッケージ（未インストール）のプレースホルダ。
# 実際にはテスト側で get_vectorstore() 自体を monkeypatch するため中身は使われない。
_install_stub_module("langchain_chroma", Chroma=object)
_install_stub_module("langchain_huggingface", HuggingFaceEmbeddings=object)

# setup.py / rag_chain.py が実LLM・実エージェントを構築しないよう差し替える。
import langchain.agents  # noqa: E402
import langchain.chat_models  # noqa: E402

langchain.chat_models.init_chat_model = _fake_init_chat_model
langchain.agents.create_agent = _fake_create_agent


@pytest.fixture(autouse=True)
def _forbid_unexpected_getpass(monkeypatch):
    """`getpass.getpass()` が呼ばれたら即座にテストを失敗させる（デフォルトの安全策）。

    上記のダミー環境変数設定に対応漏れがあると、非対話環境でも `setup.py` が
    `getpass.getpass()` に到達しうる。その場合、無応答でハングする代わりに
    明示的な例外で即座に検知できるようにする。`getpass.getpass()` の呼び出し自体を
    確認したいテストは、そのテスト内で `monkeypatch.setattr` を使って個別に上書きする。
    """

    def _raise_if_called(*args, **kwargs):
        raise RuntimeError(
            "getpass.getpass() が呼ばれました。tests/conftest.py のダミー環境変数設定に対応漏れがある可能性があります。"
        )

    monkeypatch.setattr(getpass, "getpass", _raise_if_called)
