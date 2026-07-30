# .github/workflows/

このディレクトリのワークフローが「いつ・何をするか」の一覧です。詳細な運用ルールは [AGENTS.md](../../AGENTS.md) を参照してください。

| ワークフロー | 起動タイミング | 処理内容 |
| --- | --- | --- |
| [ai-team-scheduler.yml](ai-team-scheduler.yml) | 毎日 10:00 JST(cron) + 手動実行 | `now`ラベル付きOpen Issueの中から、LLMを使わない決定的なロジック([select_next_issue.py](../scripts/select_next_issue.py))で次に着手する1件を選び、`ai-team.yml`を起動する。放置されて`In Progress`/`Under Review`のまま固まったIssueをTodoへ差し戻す自己修復も行う |
| [ai-team.yml](ai-team.yml) | `ai-team-scheduler.yml`からの起動、または手動実行(Issue番号を指定) | 指定されたIssue1件について、PM役のエージェントが `coder`→`qa-engineer`→`reviewer` の順にサブエージェントへ実装・テスト・レビューを依頼する。reviewerは実際にGitHub PRレビューでapprove/request changesを投稿するが、**approveが出てもマージは絶対に自動実行せず、必ず人間(koji)が手動で行う**(このプロジェクト固有の恒久方針) |
| [roadmap-groomer.yml](roadmap-groomer.yml) | 常設の[📍プロダクトロードマップIssue](https://github.com/koji-s-private/llm-practice/issues/43)(`roadmap-thread`ラベル)への新規コメント + 手動実行 | オーナーがコメントした要望を読み取り、新規Issue作成(優先度ラベル付与。有料サービス必須の内容は`later`+費用注意書き)/既存Issueへの反映/クローズのいずれかを自動判断する |
| [daily-health-check.yml](daily-health-check.yml) | 毎日 9:00 JST(cron) + 手動実行 | リポジトリ全体(`rag_chain.py`, `ingest.py`, `memory.py`, `app.py`, `setup.py`など)を能動的にスキャンし、バグ・改善点・検討事項・UX提案を優先度ラベル付きでIssue化する(広く浅い定期健診)。作成したIssue番号は各実行のActions画面の「Summary」にも書き出される |

## 補足
- すべてのワークフローが `workflow_dispatch` に対応しており、Actionsタブからオーナーが任意のタイミングで手動実行できる
- スケジュール実行やコメント投稿などのイベントトリガーは、各ワークフローファイルが `main` ブランチ上に存在する内容でのみ発火する(PRのブランチ上の変更は反映されない)
- 課金が発生する可能性のある操作(有料クラウドサービスの契約・起動、有料APIの利用等)は、どのワークフローも絶対に実行しない方針
