"""Issue #96: CI失敗時に自動対応するworkflowに関する設定ファイルの静的検証テスト。

実際にGitHub Actions上でワークフローを走らせることはできないため、ここでは
`.github/workflows/ci-failure-guard.yml` をYAMLとして読み込み、意図した内容
(トリガー・権限・切り分けロジックの入口となるステップ構成)になっているかを
静的に検証する（test_ci_config.pyと同じパターン）。
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_FAILURE_GUARD_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci-failure-guard.yml"
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def ci_failure_guard_workflow() -> dict:
    with CI_FAILURE_GUARD_WORKFLOW_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def ci_workflow() -> dict:
    with CI_WORKFLOW_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _on_section(workflow: dict) -> dict:
    # PyYAML(YAML 1.1仕様)は、クォートされていない `on:` キーを真偽値 True として
    # パースしてしまう既知の挙動があるため、文字列キー・真偽値キーの両方を考慮する。
    return workflow.get("on") or workflow.get(True) or {}


def _sole_job(workflow: dict) -> dict:
    jobs = workflow["jobs"]
    assert len(jobs) == 1
    (job,) = jobs.values()
    return job


def test_ci_failure_guard_workflow_is_valid_yaml(ci_failure_guard_workflow):
    assert isinstance(ci_failure_guard_workflow, dict)
    assert "jobs" in ci_failure_guard_workflow


def test_ci_failure_guard_workflow_triggers_on_ci_workflow_run_completed(ci_failure_guard_workflow):
    on_section = _on_section(ci_failure_guard_workflow)
    assert "workflow_run" in on_section
    workflow_run = on_section["workflow_run"]
    assert workflow_run["workflows"] == ["CI"]
    assert "completed" in workflow_run["types"]


def test_ci_failure_guard_workflow_has_required_permissions(ci_failure_guard_workflow):
    permissions = ci_failure_guard_workflow["permissions"]
    assert permissions["contents"] == "write"
    assert permissions["pull-requests"] == "write"
    assert permissions["actions"] == "write"


def test_ci_failure_guard_workflow_has_concurrency_group(ci_failure_guard_workflow):
    assert "concurrency" in ci_failure_guard_workflow
    concurrency = ci_failure_guard_workflow["concurrency"]
    assert "group" in concurrency
    assert "workflow_run.id" in concurrency["group"]


def test_ci_failure_guard_job_only_runs_when_ci_run_failed(ci_failure_guard_workflow):
    job = _sole_job(ci_failure_guard_workflow)
    assert "github.event.workflow_run.conclusion == 'failure'" in job["if"]


def test_ci_failure_guard_workflow_invokes_claude_code_action(ci_failure_guard_workflow):
    job = _sole_job(ci_failure_guard_workflow)
    uses_steps = [step["uses"] for step in job["steps"] if "uses" in step]
    assert any(u.startswith("anthropics/claude-code-action@") for u in uses_steps)


def test_ci_failure_guard_workflow_has_rerun_step_for_infra_failures(ci_failure_guard_workflow):
    job = _sole_job(ci_failure_guard_workflow)
    run_commands = [step["run"] for step in job["steps"] if "run" in step]
    assert any("gh run rerun" in cmd for cmd in run_commands)


def _step_by_run_substring(job: dict, substring: str) -> dict:
    # `run`コマンドに指定した文字列を含む唯一のステップを返す。
    matches = [step for step in job["steps"] if "run" in step and substring in step["run"]]
    assert len(matches) == 1, f"'{substring}' を含むステップが1つだけであること: {matches}"
    return matches[0]


def test_ci_failure_guard_infra_rerun_step_is_gated_on_infra_classification(ci_failure_guard_workflow):
    # インフラ起因の自動再実行ステップは、classifyステップの結果がinfraの場合のみ実行されること。
    job = _sole_job(ci_failure_guard_workflow)
    step = _step_by_run_substring(job, "gh run rerun")
    assert step["if"] == "steps.classify.outputs.classification == 'infra'"


def test_ci_failure_guard_workflow_has_comment_step_for_unknown_classification(ci_failure_guard_workflow):
    # 切り分け不可(unknown)の場合、gh pr commentでPRへエスカレーションすること。
    job = _sole_job(ci_failure_guard_workflow)
    run_commands = [step["run"] for step in job["steps"] if "run" in step]
    assert any("gh pr comment" in cmd for cmd in run_commands)


def test_ci_failure_guard_unknown_comment_step_is_gated_on_unknown_classification(ci_failure_guard_workflow):
    job = _sole_job(ci_failure_guard_workflow)
    step = _step_by_run_substring(job, "gh pr comment")
    assert step["if"] == "steps.classify.outputs.classification == 'unknown'"


def test_ci_failure_guard_code_steps_are_gated_on_code_classification(ci_failure_guard_workflow):
    # コード起因(code)判定時のみ実行される、checkoutとclaude-code-actionの両ステップが
    # 正しくclassification == 'code'を条件としていること。
    job = _sole_job(ci_failure_guard_workflow)

    checkout_steps = [step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@")]
    assert len(checkout_steps) == 1
    assert checkout_steps[0]["if"] == "steps.classify.outputs.classification == 'code'"

    claude_steps = [step for step in job["steps"] if step.get("uses", "").startswith("anthropics/claude-code-action@")]
    assert len(claude_steps) == 1
    assert claude_steps[0]["if"] == "steps.classify.outputs.classification == 'code'"


def test_ci_failure_guard_classify_step_code_step_names_match_ci_workflow_step_names(
    ci_failure_guard_workflow, ci_workflow
):
    # classifyステップの分類ロジックが参照するステップ名(code_step_names)は、
    # ci.ymlのlint-and-testジョブの実際のステップ名と一致していなければ機能しない。
    # ここではci.ymlをYAMLとしてパースしてステップ名を取り出し、ci-failure-guard.yml側の
    # classifyステップのrun文字列にそれぞれが含まれているかを検証する。
    (ci_job,) = ci_workflow["jobs"].values()
    ci_step_names = [step["name"] for step in ci_job["steps"] if "name" in step]

    guard_job = _sole_job(ci_failure_guard_workflow)
    classify_steps = [step for step in guard_job["steps"] if step.get("id") == "classify"]
    assert len(classify_steps) == 1
    classify_run = classify_steps[0]["run"]

    # コード起因判定の対象とすべき、ci.yml側の3ステップ
    target_names = {
        "ruff check（lint）",
        "ruff format --check（フォーマットチェック）",
        "pytest",
    }
    assert target_names.issubset(set(ci_step_names))
    for name in target_names:
        assert name in classify_run, f"ci.ymlのステップ名 '{name}' がclassifyステップのrun文字列に見当たらない"
