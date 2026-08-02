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
main = select_next_issue.main


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


class _FakeGh:
    """main()内の gh/gh_json 呼び出しを実プロセスを起動せずに検証するためのフェイク。

    labeled_issues に {ラベル名: Issue一覧} を渡すと `issue list --label <ラベル>` を模擬する。
    実行されたコマンド一式は calls に記録され、テスト側で `gh issue edit`/`gh issue comment`
    が期待通り呼ばれたかを確認できる。
    """

    def __init__(self, labeled_issues, project_items=None, open_prs=None, all_prs=None, comments=None):
        self.labeled_issues = labeled_issues
        self.project_items = project_items or []
        self.open_prs = open_prs or []
        self.all_prs = all_prs or []
        self.comments = comments or {}
        self.calls: list[tuple] = []

    def gh_json(self, *args, token=None):
        self.calls.append(("gh_json", args))
        if args[0] == "issue" and args[1] == "list":
            label = args[args.index("--label") + 1]
            return self.labeled_issues.get(label, [])
        if args[0] == "pr" and args[1] == "list":
            return self.all_prs if "--state" in args and args[args.index("--state") + 1] == "all" else self.open_prs
        if args[0] == "project" and args[1] == "item-list":
            return {"items": self.project_items}
        if args[0] == "issue" and args[1] == "view":
            number = int(args[2])
            return {"comments": [None] * self.comments.get(number, 0)}
        raise AssertionError(f"想定外の gh_json 呼び出し: {args}")

    def gh(self, *args, token=None):
        self.calls.append(("gh", args))
        return ""


def _run_main_with_fake_gh(monkeypatch, tmp_path, fake: _FakeGh):
    monkeypatch.setenv("PROJECTS_GH_TOKEN", "dummy-token")
    monkeypatch.setenv("PROJECT_NUMBER", "3")
    monkeypatch.setenv("PROJECT_OWNER", "koji-s-private")
    monkeypatch.setenv("PROJECT_ID", "PROJECT_ID")
    monkeypatch.setenv("STATUS_FIELD_ID", "STATUS_FIELD_ID")
    monkeypatch.setenv("STATUS_TODO_ID", "STATUS_TODO_ID")
    output_file = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setattr(select_next_issue, "gh_json", fake.gh_json)
    monkeypatch.setattr(select_next_issue, "gh", fake.gh)
    main()
    return output_file


def test_main_promotes_next_label_issue_when_no_now_candidate(monkeypatch, tmp_path):
    fake = _FakeGh(labeled_issues={"now": [], "next": [{"number": 20, "body": ""}], "later": []})
    output_file = _run_main_with_fake_gh(monkeypatch, tmp_path, fake)

    assert output_file.read_text(encoding="utf-8") == "issue_number=20\n"
    assert ("gh", ("issue", "edit", "20", "--add-label", "now", "--remove-label", "next")) in fake.calls
    comment_calls = [c for c in fake.calls if c[0] == "gh" and c[1][:2] == ("issue", "comment")]
    assert len(comment_calls) == 1
    assert comment_calls[0][1][1] == "comment"


def test_main_promotes_later_label_issue_when_no_now_or_next_candidate(monkeypatch, tmp_path):
    fake = _FakeGh(labeled_issues={"now": [], "next": [], "later": [{"number": 27, "body": ""}]})
    output_file = _run_main_with_fake_gh(monkeypatch, tmp_path, fake)

    assert output_file.read_text(encoding="utf-8") == "issue_number=27\n"
    assert ("gh", ("issue", "edit", "27", "--add-label", "now", "--remove-label", "later")) in fake.calls


def test_main_prefers_next_over_later_when_both_have_candidates(monkeypatch, tmp_path):
    fake = _FakeGh(
        labeled_issues={
            "now": [],
            "next": [{"number": 20, "body": ""}],
            "later": [{"number": 5, "body": ""}],
        }
    )
    output_file = _run_main_with_fake_gh(monkeypatch, tmp_path, fake)

    assert output_file.read_text(encoding="utf-8") == "issue_number=20\n"
    assert ("gh", ("issue", "edit", "20", "--add-label", "now", "--remove-label", "next")) in fake.calls


def test_main_skips_cost_warning_issue_when_promoting(monkeypatch, tmp_path):
    fake = _FakeGh(
        labeled_issues={
            "now": [],
            "next": [
                {"number": 14, "body": "⚠️ 費用が発生する可能性があります"},
                {"number": 20, "body": ""},
            ],
            "later": [],
        }
    )
    output_file = _run_main_with_fake_gh(monkeypatch, tmp_path, fake)

    assert output_file.read_text(encoding="utf-8") == "issue_number=20\n"
    assert ("gh", ("issue", "edit", "20", "--add-label", "now", "--remove-label", "next")) in fake.calls


def test_main_does_nothing_when_no_candidate_in_any_label(monkeypatch, tmp_path):
    fake = _FakeGh(labeled_issues={"now": [], "next": [], "later": []})
    output_file = _run_main_with_fake_gh(monkeypatch, tmp_path, fake)

    assert not output_file.exists()
    edit_calls = [c for c in fake.calls if c[0] == "gh" and c[1][:2] == ("issue", "edit")]
    assert edit_calls == []
