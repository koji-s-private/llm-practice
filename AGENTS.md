# チーム開発ガイドライン（Claude Code 自動運用チーム共通ルール）

このプロジェクト（ローカルRAGチャットアプリ、Streamlit + LangChain + Chroma + Ollama）は、
GitHub Actions上で動くAIチームによって定期的にメンテナンスされています。

## コミット・PR
- コミットメッセージは Conventional Commits（feat:, fix:, test: など）を厳守
- PRの本文に必ず `Closes #<issue番号>` を入れて Issue と自動リンクさせる
- 1PRの変更ファイルは目安5枚以内。大きくなりそうなら Issue を分割する
- PR本文の「## 動作確認(エビデンス)」は、各項目に動作手順を明記する。
  pytest実行・評価スクリプト実行など実行コマンドがあるものは、実行した具体的なコマンドを
  明記した上でその生の実行結果を貼る。実機（ブラウザ操作等）による動作確認を行った場合は、
  実施した検証フロー（操作手順）を箇条書きで記載した上で結果を示す

## コード品質
- 実装を変更したら対応するテストを `tests/` に必ず追加・更新する
- テストが通らない状態でPRを作成しない
- 課金が発生する可能性のある操作（有料クラウドサービスの契約・起動、有料APIの利用等）は絶対に実行しない。実装・インフラ選定は必ず無料枠・無料ツールで完結する方法のみを採用する

## GitHub Projects 運用
- Project board: [koji-s-private/llm-practice](https://github.com/orgs/koji-s-private/projects/3)（Projects v2）
  - Status: `Todo` → `In Progress` → `Under Review` → `Done`（`Under Review` はPR作成後、reviewerのレビュー中・修正対応中に使う独自追加ステータス）
- 必要なID（`PROJECT_OWNER`, `PROJECT_NUMBER`, `PROJECT_ID`, `STATUS_FIELD_ID`,
  `STATUS_TODO_ID`, `STATUS_IN_PROGRESS_ID`, `STATUS_UNDER_REVIEW_ID`, `STATUS_DONE_ID`）は
  GitHub Actionsのリポジトリ変数（`vars.*`）として登録済みで、各ワークフローの `env:` に渡している。
  エージェントは環境変数として直接参照できるので、都度 `gh project field-list` などで調べ直す必要はない
- ステータス更新コマンド（ITEM_ID は `gh project item-list` で取得）
  ```bash
  GH_TOKEN=$PROJECTS_GH_TOKEN gh project item-edit --project-id $PROJECT_ID --field-id $STATUS_FIELD_ID \
    --id <ITEM_ID> --single-select-option-id <option-id>
  ```
- 作業開始時は `In Progress` に更新する。PR作成後は、GitHub Projects純正の「Pull request linked to issue」ワークフローが自動的に `Under Review` に変更するため、エージェントが自分で更新する必要はない（`Closes #<issue番号>` をPR本文に含めてさえいれば自動で動く）
- マージ完了後の `Done` への更新も、GitHub Projects純正の「Pull request merged」ワークフローが自動的に行う
- reviewerのapprove(LGTM)に至らず終了した場合は `In Progress` のままにせず `Under Review` のまま止め、人間が気づけるようにする
- **重要**: `claude-code-action` はセッション内で `GH_TOKEN`/`GITHUB_TOKEN` を自身のGitHub Appインストールトークン（`claude[bot]`）で上書きする。
  このbotトークンはIssue/PR操作はできるが、Organization配下のProjectsには権限がないため、
  `gh project` で始まるコマンドは必ず `GH_TOKEN=$PROJECTS_GH_TOKEN` を先頭に付けて、専用トークンに明示的に差し替えて実行すること
  （逆に issue/PR 操作は素の `gh` のままでよい）

## 役割分担
- GitHub Actions上のセッション（このガイドラインを読んでいる側）はPM/リードエンジニア役。実装そのものは行わず、Task tool 経由で以下のサブエージェントに委任すること
  - `coder`: 実装・ブランチ作成（[.claude/agents/coder.md](.claude/agents/coder.md)）
  - `qa-engineer`: テスト作成・実行（[.claude/agents/qa-engineer.md](.claude/agents/qa-engineer.md)）
  - `reviewer`: 静的解析・セキュリティ観点でのレビュー、コード変更は行わない。判定結果は実際のGitHub PRレビュー(`gh pr review --comment`)として投稿する。`--approve`・`--request-changes`はどちらも使わない（PR作成者(coder)とレビュアー(reviewer)が同じGitHub App ID(`claude[bot]`)で動作しており、GitHubが自己レビューとみなしてどちらも必ず拒否するため。LGTM/修正必要のどちらの判定かは`--comment`の本文冒頭に明記する）（[.claude/agents/reviewer.md](.claude/agents/reviewer.md)）
- 3者の作業が完了し、テストが通ってからPRを作成する
- ワークフロー本体は [.github/workflows/ai-team.yml](.github/workflows/ai-team.yml) を参照
- チケットの新規発掘は [.github/workflows/daily-health-check.yml](.github/workflows/daily-health-check.yml) が別途毎日担当する（役割が重複しないよう、ai-team.yml側はリポジトリ全体の能動的なスキャンは行わない）

## Issue選定とマージ方針（**このプロジェクトは人間承認必須。参考にした他プロジェクトとの最大の違い**）
- `now` ラベルが付いたOpen Issueのうち、`Under Review`でも`In Progress`でもなく、オープンなPRも紐づいていないものを対象にする（選定ロジックの詳細は [.github/scripts/select_next_issue.py](.github/scripts/select_next_issue.py)、起動の流れは [ai-team-scheduler.yml](.github/workflows/ai-team-scheduler.yml) / [ai-team.yml](.github/workflows/ai-team.yml) 参照）
- 3者の作業が完了しテストが通ったらPRを作成し、reviewerに実際のGitHub PRレビューでLGTM判定（`gh pr review --comment`、本文に「LGTM」と明記する慣習。coder/reviewerが同じGitHub App ID(`claude[bot]`)で動作するため、GitHub上の正式な`--approve`は自己承認扱いで必ず拒否される。そのためAIによる判定は常に`--comment`で記録し、GitHub上の正式なApprove状態の付与は行わない）をもらう
- **reviewerがapprove(LGTM)を出しても、絶対に自動マージしない。マージは必ず人間（koji）が手動で行う。** これは意図的な設計判断であり、将来的にも変更しない前提とする
- reviewerがrequest changesのまま(approveに至らない)場合も同様にマージしない。PRにこれまでの経緯を要約したコメントを残し、人間の判断を待つ

## スコープ外の発見事項の扱い
- coder / qa-engineer / reviewer が作業中に今回のIssueと無関係な問題（バグ、技術的負債、改善点）に気づいた場合、
  その場では直さずPMへの報告に「スコープ外の発見事項」として含める
- PMはそれを新しいIssueとして作成し、`found-in-review` ラベルを付ける（Statusは `Todo`。既存Issueとの重複がないか事前に確認すること）
- 優先度ラベル（`now`/`next`/`later`）は基本 `next` とする（緊急性が本当に高い場合のみ `now`）

## プロダクト方針とIssue自動作成
- 常設の[📍 プロダクトロードマップ Issue](https://github.com/koji-s-private/llm-practice/issues/43)（`roadmap-thread` ラベル）に、オーナー（koji）が機能追加・改善・方針転換をコメントで書き込む運用にしている（react-native-first-appリポジトリと同じ仕組み）
- [.github/workflows/roadmap-groomer.yml](.github/workflows/roadmap-groomer.yml) が
  ロードマップIssueへの新規コメントをトリガーに起動し、要望を次のいずれかに振り分ける
  - 新規の要望 → 新しいIssueを作成し優先度ラベル（オーナー本人の明示的な依頼のため原則 `now`。有料サービスが必須の内容は `later` + 費用注意書き）を付与
  - 既存Issueへの方針変更 → 該当Issueにコメント追記、または未着手なら本文を更新
  - 既存Issueの取り下げ → 該当Issueをクローズ
  - アクション不要な内容（雑談・確認質問など） → 何もしない
- roadmap-groomer.yml / daily-health-check.yml / ai-team-scheduler.yml / ai-team.yml はすべて `workflow_dispatch`
  に対応しており、GitHub Actionsタブからオーナーが任意のタイミングで手動実行することもできる
- 起票したIssueの重複防止のため、roadmap-groomer.yml・daily-health-check.yml・ai-team.yml（found-in-review）は
  それぞれ `gh issue list --state open` で他の起票元のIssueとも重複しないか確認してから新規作成する
