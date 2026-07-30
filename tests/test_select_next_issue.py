""".github/scripts/select_next_issue.py の選定ロジックに対するテスト。

このスクリプトはCI(ai-team-scheduler.yml)専用でapp本体からは importされないため、
ファイルパスから直接importする。
"""

import importlib.util
import pathlib

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / ".github" / "scripts" / "select_next_issue.py"
)
spec = importlib.util.spec_from_file_location("select_next_issue", MODULE_PATH)
select_next_issue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select_next_issue)

is_cost_warning = select_next_issue.is_cost_warning
issue_referenced_in_any_pr = select_next_issue.issue_referenced_in_any_pr
pick_issue = select_next_issue.pick_issue


def test_is_cost_warning_detects_prefix():
    assert is_cost_warning("⚠️ 費用が発生する可能性があります\n\n続き")
    assert not is_cost_warning("通常のIssue本文です")
    assert not is_cost_warning("")


def test_issue_referenced_in_any_pr_matches_body_reference():
    prs = [{"number": 1, "title": "fix: foo", "body": "Closes #14\n\n詳細", "headRefName": "fix/foo"}]
    assert issue_referenced_in_any_pr(14, prs)
    assert not issue_referenced_in_any_pr(140, prs)
    assert not issue_referenced_in_any_pr(1, prs)


def test_issue_referenced_in_any_pr_matches_branch_name():
    prs = [{"number": 2, "title": "fix", "body": "", "headRefName": "issue-18-sync-fix"}]
    assert issue_referenced_in_any_pr(18, prs)
    assert not issue_referenced_in_any_pr(180, prs)


def test_pick_issue_selects_smallest_eligible_number():
    now_issues = [{"number": 18, "body": ""}, {"number": 14, "body": ""}]
    selected, heals = pick_issue(now_issues, project_items={}, open_prs=[], all_prs=[], comment_counts={})
    assert selected == 14
    assert heals == []


def test_pick_issue_skips_issue_with_cost_warning():
    now_issues = [
        {"number": 14, "body": "⚠️ 費用が発生する可能性があります"},
        {"number": 18, "body": ""},
    ]
    selected, heals = pick_issue(now_issues, project_items={}, open_prs=[], all_prs=[], comment_counts={})
    assert selected == 18
    assert heals == []


def test_pick_issue_skips_issue_with_open_pr():
    now_issues = [{"number": 14, "body": ""}, {"number": 18, "body": ""}]
    open_prs = [{"number": 30, "title": "", "body": "Closes #14", "headRefName": "fix/14"}]
    selected, heals = pick_issue(
        now_issues, project_items={}, open_prs=open_prs, all_prs=open_prs, comment_counts={}
    )
    assert selected == 18
    assert heals == []


def test_pick_issue_skips_genuinely_in_progress_issue():
    now_issues = [{"number": 14, "body": ""}, {"number": 18, "body": ""}]
    project_items = {14: {"status": "In Progress", "id": "ITEM_14"}}
    selected, heals = pick_issue(
        now_issues, project_items=project_items, open_prs=[], all_prs=[], comment_counts={14: 2}
    )
    assert selected == 18
    assert heals == []


def test_pick_issue_self_heals_stuck_in_progress_issue_with_no_pr_or_comments():
    """issue #14の実際の障害(In Progressのまま放置・PR無し・コメント無し)を再現するケース。"""
    now_issues = [{"number": 14, "body": ""}, {"number": 18, "body": ""}]
    project_items = {14: {"status": "In Progress", "id": "ITEM_14"}}
    selected, heals = pick_issue(
        now_issues, project_items=project_items, open_prs=[], all_prs=[], comment_counts={14: 0}
    )
    assert selected == 14
    assert heals == [14]


def test_pick_issue_returns_none_when_no_candidate():
    now_issues = [{"number": 14, "body": ""}]
    project_items = {14: {"status": "Under Review", "id": "ITEM_14"}}
    selected, heals = pick_issue(
        now_issues, project_items=project_items, open_prs=[], all_prs=[], comment_counts={14: 1}
    )
    assert selected is None
    assert heals == []
