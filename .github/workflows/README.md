# .github/workflows/

このディレクトリのワークフローが「いつ・何をするか」の一覧です。詳細な運用ルールは [AGENTS.md](../../AGENTS.md) を参照してください。

| ワークフロー | 起動タイミング | 処理内容 |
| --- | --- | --- |
| [ci.yml](ci.yml) | `main`へのpush + PR(push/pull_request) | lint(`ruff check`)・フォーマットチェック(`ruff format --check`)・テスト(`pytest`)を自動実行し、いずれかが失敗したらジョブを失敗させて問題を早期検知する |
| [ai-team-scheduler.yml](ai-team-scheduler.yml) | 毎日 10:00 JST(cron) + 手動実行 | `now`ラベル付きOpen Issueの中から、LLMを使わない決定的なロジック([select_next_issue.py](../scripts/select_next_issue.py))で次に着手する1件を選び、`ai-team.yml`を起動する。放置されて`In Progress`/`Under Review`のまま固まったIssueをTodoへ差し戻す自己修復も行う |
| [ai-team.yml](ai-team.yml) | `ai-team-scheduler.yml`からの起動、または手動実行(Issue番号を指定) | 指定されたIssue1件について、PM役のエージェントが `coder`→`qa-engineer`→`reviewer` の順にサブエージェントへ実装・テスト・レビューを依頼する。`actions/checkout`直後に`pip install -r requirements.txt`を実行済みのため、coder/qa-engineerが動く時点で依存パッケージはインストール済み。reviewerは実際にGitHub PRレビューで`--comment`のみを使い、本文冒頭でLGTM/修正必要のどちらかを明記する(PR作成者(coder)とレビュアー(reviewer)が同じGitHub App ID(`claude[bot]`)のため、GitHub上の正式な`--approve`/`--request-changes`はどちらも自己レビュー扱いで拒否されるので使わない)。**LGTMが出てもマージは絶対に自動実行せず、必ず人間(koji)が手動で行う**(このプロジェクト固有の恒久方針) |
| [roadmap-groomer.yml](roadmap-groomer.yml) | 常設の[📍プロダクトロードマップIssue](https://github.com/koji-s-private/llm-practice/issues/43)(`roadmap-thread`ラベル)への新規コメント + 手動実行 | オーナーがコメントした要望を読み取り、新規Issue作成(優先度ラベル付与。有料サービス必須の内容は`later`+費用注意書き)/既存Issueへの反映/クローズのいずれかを自動判断する |
| [daily-health-check.yml](daily-health-check.yml) | 毎日 9:00 JST(cron) + 手動実行 | リポジトリ全体(`rag_chain.py`, `ingest.py`, `memory.py`, `app.py`, `setup.py`など)を能動的にスキャンし、バグ・改善点・検討事項・UX提案を優先度ラベル付きでIssue化する(広く浅い定期健診)。作成したIssue番号は各実行のActions画面の「Summary」にも書き出される |
| [pr-conflict-guard.yml](pr-conflict-guard.yml) | オープンPR(base=main)がmainにマージされた直後(`pull_request: closed` かつ `merged == true`) | 他のオープンPRにmainとのコンフリクトが発生していないかを検知し、発生していれば`coder`サブエージェントがmainを取り込む方向でのみ自動的に解消し(mainブランチ自体は変更しない)、`qa-engineer`が再検証する。機械的に解消できない場合はpushせずPRにコメントを残す |
| [ci-failure-guard.yml](ci-failure-guard.yml) | `ci.yml`(CI)の実行完了(`workflow_run: completed`)がfailureの場合 | 失敗原因をGitHub Actions APIのジョブ・ステップ情報から切り分け、`ruff check`/`ruff format --check`/`pytest`のいずれかが失敗した「コード起因」なら`coder`サブエージェントに既存PRブランチでの修正を依頼する。前段(Set up job等)で落ちた「インフラ起因」なら`gh run rerun --failed`で自動再実行のみ行い、判断できない場合はPRにコメントを残す |
| [pr-review-on-update.yml](pr-review-on-update.yml) | オープンPRへの新規コミットpush(`pull_request: synchronize`、`main`向け) | reviewerによる初回レビューが既に投稿済みのPRに限り、`reviewer`サブエージェントによる再レビューを自動的に行う。ai-team.ymlのPMオーケストレーションが同じブランチで実行中の場合は二重実行を避けるためスキップする |

## 補足
- `workflow_dispatch` に対応しており、Actionsタブからオーナーが任意のタイミングで手動実行できるのは [ai-team-scheduler.yml](ai-team-scheduler.yml) / [ai-team.yml](ai-team.yml) / [roadmap-groomer.yml](roadmap-groomer.yml) / [daily-health-check.yml](daily-health-check.yml) の4つ。[ci.yml](ci.yml) / [pr-conflict-guard.yml](pr-conflict-guard.yml) / [ci-failure-guard.yml](ci-failure-guard.yml) / [pr-review-on-update.yml](pr-review-on-update.yml) の4つは、push/PR/CI完了などのイベントトリガーのみで起動する構成であり `workflow_dispatch` には対応していない
- スケジュール実行やコメント投稿などのイベントトリガーは、各ワークフローファイルが `main` ブランチ上に存在する内容でのみ発火する(PRのブランチ上の変更は反映されない)
- 課金が発生する可能性のある操作(有料クラウドサービスの契約・起動、有料APIの利用等)は、どのワークフローも絶対に実行しない方針
