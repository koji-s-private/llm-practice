"""examples/配下のチュートリアル用スクリプトの整理に関するテスト。

`extract_text.py` / `models_and_prompts.py` はリポジトリ直下から `examples/` へ移動し、
トップレベルにあった実行コード（LLM呼び出し等）を `main()` に切り出して
`if __name__ == "__main__":` でガードした。本テストでは、以下を検証する。

- 旧パス（リポジトリ直下）にこれらのファイルが存在しないこと
- 新パス（`examples/`配下）に存在すること
- import するだけでは `main()` が実行されない（LLM呼び出しなどの副作用が起きない）こと
- `main` という呼び出し可能な関数がモジュールに存在すること

`setup.model` の実体は `tests/conftest.py` でフェイクの `init_chat_model` に
差し替え済みのため、import時に実際のOllama/Anthropic/OpenAI呼び出しは発生しない。
"""

import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_extract_text_does_not_exist_at_repo_root():
    assert not (REPO_ROOT / "extract_text.py").exists()


def test_models_and_prompts_does_not_exist_at_repo_root():
    assert not (REPO_ROOT / "models_and_prompts.py").exists()


def test_extract_text_exists_under_examples():
    assert (REPO_ROOT / "examples" / "extract_text.py").is_file()


def test_models_and_prompts_exists_under_examples():
    assert (REPO_ROOT / "examples" / "models_and_prompts.py").is_file()


def test_rag_interview_prep_docx_removed_from_repo():
    # どこからも参照されていなかった個人的な学習メモのため削除済み。
    assert not (REPO_ROOT / "rag_interview_prep.docx").exists()


def test_import_extract_text_module_has_no_side_effect_and_defines_main():
    # import時にLLM呼び出し（setup.modelのフェイクのinvoke()）が発生すると
    # tests/conftest.py の `_FakeChatModel.invoke` がAssertionErrorを送出するため、
    # 例外なくimportできること自体が「副作用なし」の検証になる。
    module = importlib.import_module("examples.extract_text")

    assert hasattr(module, "main")
    assert callable(module.main)


def test_import_models_and_prompts_module_has_no_side_effect_and_defines_main():
    module = importlib.import_module("examples.models_and_prompts")

    assert hasattr(module, "main")
    assert callable(module.main)
