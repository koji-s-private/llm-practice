# .claude/agents/

[Claude Code](https://docs.claude.com/claude-code) の [サブエージェント](https://docs.claude.com/en/docs/claude-code/sub-agents)定義ファイルを置くディレクトリです。
このプロジェクトを自動運用する「AIチーム」（PM役のセッションが `Task` ツール経由で呼び出す実装・テスト・レビュー担当）の
役割・権限・作業手順をここで定義しています。詳しい運用ルール全体は、リポジトリ直下の
[AGENTS.md](../../AGENTS.md) の「役割分担」セクションを参照してください。

## フロントマター形式

各ファイルはMarkdownの先頭にYAMLフロントマター（`---` で囲んだ設定部分）を持ちます。

| キー | 役割 |
|---|---|
| `name` | サブエージェントの識別名（`Task`ツールから呼び出す際に指定する） |
| `description` | どんな場面で使うエージェントかの説明（PM役のセッションがいつ委任すべきか判断する材料になる） |
| `tools` | 使用を許可するツールの一覧（例: `Read, Write, Edit, Bash`）。個別コマンドに絞る場合は `Bash(git diff:*)` のように許可パターンを指定できる |
| `model` | 使用するモデル（例: `sonnet`） |

フロントマターの下の本文が、そのサブエージェントに渡されるシステムプロンプト（振る舞いの指示）です。

## 現在定義されているエージェント

| ファイル | 役割 |
|---|---|
| [coder.md](coder.md) | Issueの要求に基づき実装コードを書き、ブランチを作成する |
| [qa-engineer.md](qa-engineer.md) | 実装済みコードに対して正常系・異常系・境界値のテストを作成し実行検証する |
| [reviewer.md](reviewer.md) | コーディング規約・セキュリティ・品質観点でコード変更をレビューする（コード自体は変更しない） |
| [designer.md](designer.md) | 「使いやすさ・継続して利用したいと思えるか」というUI/UX観点でレビューする（コード自体は変更しない）。`app.py` / `.streamlit/` を変更するPRのみが対象で、Playwrightで変更前後のスクリーンショットを比較する |

## 補足: なぜreviewerは正式な`--approve`ではなく`--comment`でLGTM判定を表現するのか

PR作成（`coder`）とレビュー投稿（`reviewer`）はどちらも同じGitHub Appインストール
（`claude[bot]`）の権限で動作しています。GitHubはこれを自己レビューとみなし、
`--approve` ・ `--request-changes` のどちらを使っても
`Can not approve your own pull request` 等のエラーで必ず拒否します。

そのため `reviewer` は常に `gh pr review <PR番号> --comment` でレビューを投稿し、
本文の先頭に「LGTM」または「判定: 修正が必要（REQUEST CHANGES相当）」と明記することで
判定を表現する運用にしています（詳細は [reviewer.md](reviewer.md) 参照）。GitHub上の正式な
Approve/Request changes状態の付与、および実際のマージは常に人間（koji）が行います。

同じ理由で `designer` も `--comment` でLGTM相当/要修正を表現します（詳細は [designer.md](designer.md) 参照）。

## 補足: designerとreviewerの関係

`designer` は `reviewer` と同格のゲートです。UI関連ファイル（`app.py` / `.streamlit/`）を
変更するPRに限り、reviewerのLGTM相当に加えてdesignerのLGTM相当も揃って初めて、
Under Review（人間のマージ判断待ち）の完了条件を満たしたとみなします（UI変更を含まないPRでは
designerは呼び出されず、reviewerのLGTM相当のみが条件のままです）。詳細は
[../../.github/workflows/ai-team.yml](../../.github/workflows/ai-team.yml) と
[AGENTS.md](../../AGENTS.md) の「役割分担」セクションを参照してください。
