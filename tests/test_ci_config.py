"""CI導入（GitHub Actions）に関する設定ファイルの静的検証テスト。

実際にGitHub Actions上でワークフローを走らせることはできないため、ここでは
`.github/workflows/ci.yml` と `pyproject.toml` の `[tool.ruff]` 設定を
YAML/TOMLとして読み込み、意図した内容（トリガー・実行ステップの順序・
除外設定）になっているかを静的に検証する。
"""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def ci_workflow() -> dict:
    with CI_WORKFLOW_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _on_section(workflow: dict) -> dict:
    # PyYAML(YAML 1.1仕様)は、クォートされていない `on:` キーを真偽値 True として
    # パースしてしまう既知の挙動があるため、文字列キー・真偽値キーの両方を考慮する。
    return workflow.get("on") or workflow.get(True) or {}


def test_ci_workflow_is_valid_yaml(ci_workflow):
    assert isinstance(ci_workflow, dict)
    assert "jobs" in ci_workflow


def test_ci_workflow_triggers_on_push_to_main(ci_workflow):
    on_section = _on_section(ci_workflow)
    assert "push" in on_section
    assert on_section["push"]["branches"] == ["main"]


def test_ci_workflow_triggers_on_pull_request(ci_workflow):
    on_section = _on_section(ci_workflow)
    # ブランチ指定なし(None)で全PRを対象にする
    assert "pull_request" in on_section


def test_ci_workflow_runs_ruff_check_then_format_check_then_pytest(ci_workflow):
    jobs = ci_workflow["jobs"]
    assert len(jobs) == 1
    (job,) = jobs.values()
    run_commands = [step["run"] for step in job["steps"] if "run" in step]

    assert run_commands == [
        "pip install -r requirements.txt",
        "ruff check .",
        "ruff format --check .",
        "pytest",
    ]


def test_ci_workflow_uses_checkout_and_setup_python_actions(ci_workflow):
    (job,) = ci_workflow["jobs"].values()
    uses_steps = [step["uses"] for step in job["steps"] if "uses" in step]

    assert any(u.startswith("actions/checkout@") for u in uses_steps)
    assert any(u.startswith("actions/setup-python@") for u in uses_steps)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllibはPython 3.11以降のみ標準搭載")
def test_ruff_config_excludes_tutorial_scripts_and_selects_expected_rules():
    import tomllib

    with PYPROJECT_PATH.open("rb") as f:
        config = tomllib.load(f)

    ruff_config = config["tool"]["ruff"]
    assert ruff_config["line-length"] == 120
    assert set(ruff_config["extend-exclude"]) == {
        "extract_text.py",
        "models_and_prompts.py",
    }
    assert set(ruff_config["lint"]["select"]) == {"E", "F", "W", "I"}


def test_ruff_is_listed_in_requirements():
    requirements_text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "ruff" in requirements_text.split()


def test_pyyaml_is_listed_in_requirements():
    # PyYAMLはtest_ci_config.py/test_ci_failure_guard_config.pyの`import yaml`に必須だが、
    # 他パッケージの推移的依存に頼らず明示的にバージョン固定されていることを検証する。
    requirements_text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    pyyaml_lines = [line for line in requirements_text.splitlines() if line.strip().upper().startswith("PYYAML")]
    assert len(pyyaml_lines) == 1, f"PyYAMLの記載は1行のみであること: {pyyaml_lines}"
    assert "==" in pyyaml_lines[0], "PyYAMLはバージョン固定(==)で記載されていること"
