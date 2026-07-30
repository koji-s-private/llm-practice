"""ai-team-scheduler.yml から呼ばれる、次に着手するIssueを1件だけ選ぶスクリプト。

これまではPMエージェント(Claude)がプロンプト内でこの選定ロジックを担っていたが、
選定自体はLLMの判断を必要としない決定的な処理であり、かつGitHub Actionsのrun一覧に
選定結果(Issue番号)を反映するには「トリガー発火時点で対象Issueが決まっている」必要が
あるため、選定だけを切り出してこのスクリプトが担う。

選定結果は GITHUB_OUTPUT に issue_number として書き出す(見つからない場合は書き出さない)。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

COST_WARNING_PREFIX = "⚠️ 費用が発生する可能性があります"
STUCK_STATUSES = ("In Progress", "Under Review")


def is_cost_warning(body: str) -> bool:
    return (body or "").lstrip().startswith(COST_WARNING_PREFIX)


def issue_referenced_in_any_pr(number: int, prs: list[dict]) -> bool:
    """PRの本文/タイトル/ブランチ名のいずれかでこのIssue番号が参照されていればTrue。"""
    text_pattern = re.compile(rf"#{number}(?!\d)")
    branch_pattern = re.compile(rf"(?:^|\D){number}(?:\D|$)")
    for pr in prs:
        body = pr.get("body") or ""
        title = pr.get("title") or ""
        branch = pr.get("headRefName") or ""
        if text_pattern.search(body) or text_pattern.search(title):
            return True
        if branch_pattern.search(branch):
            return True
    return False


def pick_issue(
    now_issues: list[dict],
    project_items: dict[int, dict],
    open_prs: list[dict],
    all_prs: list[dict],
    comment_counts: dict[int, int],
) -> tuple[int | None, list[int]]:
    """条件をクリアした最小番号のIssueを1件選ぶ。

    戻り値: (選ばれたIssue番号 or None, 自己修復(Statusの差し戻し)が必要なIssue番号のリスト)
    """
    heals: list[int] = []
    for issue in sorted(now_issues, key=lambda i: i["number"]):
        number = issue["number"]
        if is_cost_warning(issue.get("body", "")):
            continue
        if issue_referenced_in_any_pr(number, open_prs):
            continue

        item = project_items.get(number)
        status = item["status"] if item else None
        if status in STUCK_STATUSES:
            still_genuinely_stuck = issue_referenced_in_any_pr(number, all_prs) or comment_counts.get(number, 0) > 0
            if still_genuinely_stuck:
                continue
            # PRも一切無くコメントも無いまま In Progress/Under Review になっている
            # = 過去の実行が着手直後に異常終了して放置されたアイテムとみなし、自己修復する。
            heals.append(number)

        return number, heals

    return None, heals


def gh(*args: str, token: str | None = None) -> str:
    env = os.environ.copy()
    if token:
        env["GH_TOKEN"] = token
    result = subprocess.run(["gh", *args], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        result.check_returncode()
    return result.stdout


def gh_json(*args: str, token: str | None = None):
    return json.loads(gh(*args, token=token))


def main() -> None:
    projects_token = os.environ["PROJECTS_GH_TOKEN"]
    project_number = os.environ["PROJECT_NUMBER"]
    project_owner = os.environ["PROJECT_OWNER"]
    project_id = os.environ["PROJECT_ID"]
    status_field_id = os.environ["STATUS_FIELD_ID"]
    status_todo_id = os.environ["STATUS_TODO_ID"]

    now_issues = gh_json("issue", "list", "--state", "open", "--label", "now", "--json", "number,body")
    open_prs = gh_json("pr", "list", "--state", "open", "--json", "number,title,body,headRefName")
    all_prs = gh_json("pr", "list", "--state", "all", "--json", "number,title,body,headRefName")

    project_data = gh_json(
        "project", "item-list", project_number, "--owner", project_owner, "--format", "json", token=projects_token
    )
    project_items = {
        item["content"]["number"]: item
        for item in project_data.get("items", [])
        if item.get("content", {}).get("number") is not None
    }

    comment_counts: dict[int, int] = {}
    for issue in now_issues:
        n = issue["number"]
        item = project_items.get(n)
        if item and item["status"] in STUCK_STATUSES:
            comments = gh_json("issue", "view", str(n), "--json", "comments")["comments"]
            comment_counts[n] = len(comments)

    selected, heals = pick_issue(now_issues, project_items, open_prs, all_prs, comment_counts)

    for n in heals:
        item_id = project_items[n]["id"]
        print(f"#{n}: In Progress/Under Reviewのまま放置されていたためTodoに差し戻します")
        gh(
            "project", "item-edit",
            "--project-id", project_id,
            "--field-id", status_field_id,
            "--id", item_id,
            "--single-select-option-id", status_todo_id,
            token=projects_token,
        )
        gh(
            "issue", "comment", str(n),
            "--body",
            "前回の自動実装が着手直後に中断し、Statusが In Progress/Under Review のまま気づかれず"
            "放置されていたため、スケジューラが自動的にTodoに差し戻しました。次回以降の実行で改めて着手します。",
        )

    if selected is None:
        print("本日着手できるnowラベル付きIssueが見つかりませんでした。")
        return

    print(f"Issue #{selected} を選定しました。")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"issue_number={selected}\n")


if __name__ == "__main__":
    main()
