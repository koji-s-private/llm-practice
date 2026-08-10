"""AIチームエージェント向けワークフローの依存パッケージ事前インストールに関する
設定ファイルの静的検証テスト。

`ai-team.yml` / `pr-conflict-guard.yml` / `ci-failure-guard.yml` は、coder/qa-engineer
サブエージェントがリポジトリのPythonコードを操作する前提のワークフローであり、
`ci.yml` と同じパターン（`actions/checkout` 直後の `actions/setup-python@v5` +
`pip install -r requirements.txt`）で依存パッケージを事前インストールしておく必要がある
（さもないと、都度サブエージェントが自力でインストール手順を推測する必要があり、
無駄なトークン消費や失敗の原因になる）。

実際にGitHub Actions上でワークフローを走らせることはできないため、ここでは各YAMLファイルを
読み込み、意図した内容（checkout直後にPythonセットアップ・依存インストールの2ステップが
存在すること）になっているかを静的に検証する（test_ci_config.py / test_ci_failure_guard_config.py
と同じパターン）。
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# (ワークフローファイル名, checkoutを含むjobのキー) のペア。
# ai-team.yml / ci-failure-guard.yml は単一job、pr-conflict-guard.yml は
# detect/resolveの2jobのうちcheckoutがあるのは "resolve" のみ。
TARGET_WORKFLOWS = {
    "ai-team.yml": "pm-orchestrate",
    "pr-conflict-guard.yml": "resolve",
    "ci-failure-guard.yml": "triage",
}


@pytest.fixture(scope="module", params=sorted(TARGET_WORKFLOWS))
def workflow_file(request) -> tuple[str, dict]:
    name = request.param
    with (WORKFLOWS_DIR / name).open(encoding="utf-8") as f:
        return name, yaml.safe_load(f)


def _steps_for(name: str, workflow: dict) -> list[dict]:
    job_key = TARGET_WORKFLOWS[name]
    jobs = workflow["jobs"]
    assert job_key in jobs, f"{name} に想定したjob '{job_key}' が見当たらない（実際: {list(jobs)}）"
    return jobs[job_key]["steps"]


def test_workflow_files_are_valid_yaml(workflow_file):
    name, workflow = workflow_file
    assert isinstance(workflow, dict), f"{name} がdictとしてパースできない"
    assert "jobs" in workflow


def test_checkout_is_immediately_followed_by_python_setup_and_install(workflow_file):
    # actions/checkout@の直後の2ステップが、Pythonセットアップ・依存インストールであること。
    name, workflow = workflow_file
    steps = _steps_for(name, workflow)

    checkout_indices = [i for i, step in enumerate(steps) if step.get("uses", "").startswith("actions/checkout@")]
    assert len(checkout_indices) == 1, f"{name}: actions/checkoutステップが1つだけであること"
    checkout_index = checkout_indices[0]

    assert checkout_index + 2 < len(steps), f"{name}: checkout直後に2ステップ以上残っていない"
    setup_step = steps[checkout_index + 1]
    install_step = steps[checkout_index + 2]

    assert setup_step.get("uses", "").startswith("actions/setup-python@"), (
        f"{name}: checkoutの次のステップがactions/setup-pythonではない（実際: {setup_step}）"
    )
    assert setup_step.get("with", {}).get("python-version") == "3.12", f"{name}: python-versionが3.12ではない"

    assert install_step.get("run") == "pip install -r requirements.txt", (
        f"{name}: checkoutの次の次のステップがpip installではない（実際: {install_step}）"
    )


def test_python_setup_and_install_steps_share_checkout_gating_condition(workflow_file):
    # ci-failure-guard.ymlのようにcheckoutステップがif条件付きの場合、
    # 追加したPythonセットアップ・依存インストールステップも同じif条件を
    # 引き継いでいること（さもないと、checkoutされていないのにpip installだけ
    # 走ろうとして失敗する）。
    name, workflow = workflow_file
    steps = _steps_for(name, workflow)

    (checkout_step,) = (step for step in steps if step.get("uses", "").startswith("actions/checkout@"))
    checkout_if = checkout_step.get("if")

    checkout_index = steps.index(checkout_step)
    setup_step = steps[checkout_index + 1]
    install_step = steps[checkout_index + 2]

    assert setup_step.get("if") == checkout_if, f"{name}: setup-pythonのif条件がcheckoutと一致しない"
    assert install_step.get("if") == checkout_if, f"{name}: pip installのif条件がcheckoutと一致しない"


def test_agents_md_documents_preinstalled_dependencies_for_agent_workflows():
    # AGENTS.mdに、これらのワークフローでは依存パッケージが事前インストール済みである旨が
    # 明記されていること（サブエージェントが自力でインストール手順を推測しなくて済むように）。
    agents_md = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "pip install -r requirements.txt" in agents_md
    for name in TARGET_WORKFLOWS:
        assert name in agents_md, f"AGENTS.mdに{name}への言及が見当たらない"
