# チーム開発ガイドライン（Claude Code 自動運用チーム共通ルール）

このプロジェクト（ローカルRAGチャットアプリ、Streamlit + LangChain + Chroma + Ollama）は、
GitHub Actions上で動くAIチームによって定期的にメンテナンスされています。

## コミット・PR
- コミットメッセージは Conventional Commits（feat:, fix:, test: など）を厳守
- PRの本文に必ず `Closes #<issue番号>` を入れて Issue と自動リンクさせる
- ただし、ハーネス側の権限制約などによりIssueの受け入れ条件を完全には満たせず、部分的にしか解決できない
  PRを作成する場合は `Closes` を使わず `Related to #<issue番号>` のような表現にする（Issueを誤って
  自動クローズしないため）。この場合、GitHub Projectsの「Pull request linked to issue」自動ワークフローが
  発火せずStatusが自動的に `Under Review` へ遷移しないため、PM（またはcoder）が手動で `Under Review` に
  更新する（前例: PR #74、Issue #48対応）
- 1PRの変更ファイルは目安5枚以内。大きくなりそうなら Issue を分割する

## mainブランチの運用
- mainブランチへの直接コミット・pushは禁止。人間（オーナー）を含め全員、必ずfeatureブランチを作成し
  PR経由でのみ変更を反映する
- GitHub純正のBranch protection rules / Rulesetsによる技術的な強制ブロックは**導入していない**。
  Organization（`koji-s-private`）がGitHub Freeプランのため、privateリポジトリでのbranch protectionは
  有料プラン（GitHub Team以上）が無いと有効化できない仕様であり（APIも`Upgrade to GitHub Pro or make
  this repository public`という403を返す）、既存の「課金が発生する可能性のある操作は絶対に実行しない」
  方針により有料化・組織移管・public化のいずれも行わないため。技術的ブロックの代わりに本ルールと
  以下の自動化（PRマージ後のブランチ自動削除、コンフリクト自動解消、再レビュー自動化）で運用上担保する
- PRをマージしたら、対象ブランチは自動削除される（リポジトリ設定 `delete_branch_on_merge: true`。設定済み）
- 手動マージ方針（後述）の結果、複数のPRが並行してオープンな状態が起こり得る。1つのPRをマージしたことで
  他のオープンPRにコンフリクトが発生した場合、[.github/workflows/pr-conflict-guard.yml](.github/workflows/pr-conflict-guard.yml)
  が自動的に検知し、`coder`サブエージェントがmainを取り込む方向でのみマージ（main→feature。mainブランチ自体は
  一切変更しない）してコンフリクトを解消し、`qa-engineer`が再検証する。機械的に解消できない場合は
  無理に解消せず、PRにコメントを残して人間の判断を待つ
- reviewerがレビューした後にcoderが同じPRへ修正コミットをpushした場合（コンフリクト解消による
  push含む）、[.github/workflows/pr-review-on-update.yml](.github/workflows/pr-review-on-update.yml)
  が新規コミット（`synchronize`イベント）を検知して自動的に`reviewer`サブエージェントによる再レビューを行う
  （PMオーケストレーションのセッション内で既に再レビューループが回っている場合は二重実行しない）
- PR本文の「## 動作確認(エビデンス)」は、各項目に動作手順を明記する。
  pytest実行・評価スクリプト実行など実行コマンドがあるものは、実行した具体的なコマンドを
  明記した上でその生の実行結果を貼る。実機（ブラウザ操作等）による動作確認を行った場合は、
  実施した検証フロー（操作手順）を箇条書きで記載した上で結果を示す

## コード品質
- 実装を変更したら対応するテストを `tests/` に必ず追加・更新する
- テストが通らない状態でPRを作成しない
- 課金が発生する可能性のある操作（有料クラウドサービスの契約・起動、有料APIの利用等）は絶対に実行しない。実装・インフラ選定は必ず無料枠・無料ツールで完結する方法のみを採用する
- 新しい技術・ライブラリを追加した場合（`requirements.txt` / `pyproject.toml` への追加を伴う変更など）は、
  その都度 [README.md](README.md) の「使用技術」セクションも同じPR内で更新する（何をするためのものか・
  どのファイルでどう使われているかを追記する）。「使用技術」セクションが肥大化してきた場合は、
  `docs/tech-stack.md` のような別ファイルに切り出し、ルートの `README.md` 側にはリンクのみを残す
  （`data/README.md` / `tests/README.md` などの既存の「詳細は別ファイル」構成にならう）
- コード内コメントは日本語で記載する（新規追加・既存修正のいずれも）。ただし
  `extract_text.py` / `models_and_prompts.py` などLangChain公式チュートリアル由来の
  学習用スクリプト（Issue #23で整理予定）は対象外とする
- 使用技術（`requirements.txt` 記載の依存パッケージ等）に脆弱性が発覚した場合は、`now` ラベル相当の
  最優先でバージョンアップ等の対応を行う。無料で使える脆弱性チェック手段として
  [pip-audit](https://pypi.org/project/pip-audit/)（`pip install --user pip-audit` で一時利用すれば足り、
  `requirements.txt` への常設追加は不要）や GitHub Dependabot alerts（追加費用なしで利用可能）を使い、
  `pip-audit -r requirements.txt` のように手動実行して確認する。CIへの自動組み込みはIssue #5（CI導入）
  側の検討に委ねるため、本項では手動実行での確認手順の明文化にとどめる
- テストを実行するAIチームのワークフロー（`ai-team.yml` / `pr-conflict-guard.yml` / `ci-failure-guard.yml`）は
  `actions/checkout` の直後に `pip install -r requirements.txt` を実行済みのため、coder/qa-engineerが
  動く時点で依存パッケージはインストール済みである。新規追加した依存など、それでも不足している場合は
  サブエージェント自身が `pip install` を追加実行してよい

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
- **重要**: 同じ`claude[bot]`トークンには既定で"Workflows"権限も無いため、`.github/workflows/`配下のファイルを
  含む変更を通常の`git push`で反映しようとすると`refusing to allow a GitHub App to create or update workflow ...
  without workflows permission`で必ず拒否される（GitHub側の既知の仕様。Contents権限とは別枠で必要）。
  `.github/workflows/`配下の変更を伴うpushは、`workflow`スコープ付きの専用PATを登録した
  `$WORKFLOW_GH_TOKEN`シークレットを使って明示的に行うこと（[.claude/agents/coder.md](.claude/agents/coder.md)参照）。
  未登録の場合はcoderがその旨を報告して終了するので、無理にリトライさせないこと

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
- 選定可能な `now` Issueが1件も無い場合、その日は何もしないのではなく、`next` → `later` の順で
  同じ選定条件をクリアする最優先（Issue番号が最も小さい＝作成が最も古い）Issueを1件探し、
  見つかればラベルを `now` に自動で付け替えた上でその日の対象として選定する（昇格した旨をIssueにコメントで残す）。
  `now`/`next`/`later` のいずれにも対象が無い場合のみ、その日は何も実行しない
- 3者の作業が完了しテストが通ったらPRを作成し、reviewerに実際のGitHub PRレビューでLGTM判定（`gh pr review --comment`、本文に「LGTM」と明記する慣習。coder/reviewerが同じGitHub App ID(`claude[bot]`)で動作するため、GitHub上の正式な`--approve`は自己承認扱いで必ず拒否される。そのためAIによる判定は常に`--comment`で記録し、GitHub上の正式なApprove状態の付与は行わない）をもらう
- **reviewerがapprove(LGTM)を出しても、絶対に自動マージしない。マージは必ず人間（koji）が手動で行う。** これは意図的な設計判断であり、将来的にも変更しない前提とする
- reviewerがrequest changesのまま(approveに至らない)場合も同様にマージしない。PRにこれまでの経緯を要約したコメントを残し、人間の判断を待つ
- 何らかの理由（技術的制約、優先度変更、要件不明瞭など）によりcoder/qa-engineer/reviewerが実装・検証・レビューを
  中止する場合は、無言で終了せず必ず対象Issue（PR作成済みなら該当PR）にコメントで理由を残し、人間が状況を
  把握できるようにする

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
