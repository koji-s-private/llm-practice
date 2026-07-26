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
- `langchain.chat_models.init_chat_model` と `langchain.agents.create_agent` は
  実際のLLM呼び出し・エージェント構築を避けるためフェイクに差し替える。
"""
import os
import sys
import types

os.environ.setdefault("DISABLE_OLLAMA", "true")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-dummy-key")
os.environ.setdefault("LANGSMITH_API_KEY", "test-dummy-key")
os.environ.setdefault("LANGSMITH_PROJECT", "test")


def _install_stub_module(name: str, **attrs) -> types.ModuleType:
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
        raise AssertionError(
            "model.invoke() は各テストで monkeypatch してから呼び出してください"
        )


def _fake_init_chat_model(*args, **kwargs):
    return _FakeChatModel()


def _fake_create_agent(model, tools, system_prompt):
    return types.SimpleNamespace(model=model, tools=tools, system_prompt=system_prompt)


# rag_chain.py が要求する重い外部パッケージ（未インストール）のプレースホルダ。
# 実際にはテスト側で get_vectorstore() 自体を monkeypatch するため中身は使われない。
_install_stub_module("langchain_chroma", Chroma=object)
_install_stub_module("langchain_huggingface", HuggingFaceEmbeddings=object)

# setup.py / rag_chain.py が実LLM・実エージェントを構築しないよう差し替える。
import langchain.chat_models  # noqa: E402
import langchain.agents  # noqa: E402

langchain.chat_models.init_chat_model = _fake_init_chat_model
langchain.agents.create_agent = _fake_create_agent
