# tests/

このプロジェクトの自動テスト（[pytest](https://docs.pytest.org/) というPythonのテストフレームワークを使用）を置くディレクトリです。
実行するには、依存パッケージをインストールした上でリポジトリ直下から次のコマンドを実行します。

```bash
pip install -r requirements.txt  # pytestを含む
pytest
```

## 方針: 重い外部依存はフェイクに差し替える

このプロジェクトの本体コード（`rag_chain.py` / `setup.py` など）は、実行時に
埋め込みモデル（[sentence-transformers](https://www.sbert.net/)、数百MB規模のダウンロードが発生するモデル）や
ベクトルDB（[Chroma](https://www.trychroma.com/)）、LLM（大規模言語モデル。Ollama/Claude/OpenAIなど）の
実クライアントを読み込みます。テストではこれらを実際に呼び出さず、`tests/conftest.py`
（pytestがテスト実行前に自動で読み込む共通設定ファイル）で軽量なフェイク（ダミーの代役）に
差し替えています。そのため、Ollamaの起動やAPIキーの設定、埋め込みモデルのダウンロードなしに
テストを実行でき、ネットワークアクセスや課金は一切発生しません。

新しくテストを追加する際も、この方針（`get_vectorstore()` や `model.invoke()` などを
`monkeypatch`で差し替える）に従ってください。

## ファイル一覧と対応する実装ファイル

新しいテストファイルを追加したら、必ず以下の表も更新すること（陳腐化を防ぐため）。

| テストファイル | 対応する実装 | 内容 |
|---|---|---|
| `conftest.py` | - | pytest共通設定。埋め込みモデル・Chroma・LLMプロバイダをフェイクに差し替える |
| `test_conftest.py` | `conftest.py` | `conftest.py` 自体のヘルパー関数（`_setdefault_even_if_empty`）と、`getpass.getpass()` の未想定呼び出しを検知するautouse fixtureのテスト |
| `test_ingest.py` | `ingest.py` | `data/` とベクトルDBの差分同期ロジック（追加・更新・削除の判定）のテスト |
| `test_ingest_manifest.py` | `ingest.py` | `chroma_db/manifest.json` の読み書き（アトミックな書き込み、壊れたJSONへのフォールバック）のテスト |
| `test_ingest_indexed_files.py` | `ingest.py` | インデックス済みファイル一覧の取得（`list_indexed_files`）とファイル削除（`delete_indexed_file`、パストラバーサル対策含む）のテスト |
| `test_ingest_per_file_manifest_save.py` | `ingest.py` | 同期処理中にファイル1件ごとにmanifestを都度保存する挙動（中断時にも処理済み分が残ることを含む）のテスト |
| `test_ingest_add_single_conversation_file.py` | `ingest.py` | 会話ログ保存直後に呼ばれる単一ファイル同期（`add_single_conversation_file`）のテスト |
| `test_rag_chain.py` | `rag_chain.py` | 検索結果の関連度採点（`_grade_relevance`）と、「見つからない場合」のフォールバック応答のテスト |
| `test_memory.py` | `memory.py` | 会話ログの保存（Markdownファイル書き込み）・件数カウントのテスト |
| `test_app.py` | `app.py` | Streamlitのチャット画面のエラーハンドリングのテスト（`streamlit.testing.v1.AppTest` を使いスクリプト実行エンジン上で検証） |
| `test_app_source_display.py` | `app.py` | 参照元スニペットの整形（`_format_snippet`）、参照元ラベル（`_format_source_label`）、過去スレッドラベル（`_format_thread_label`）の純粋関数のテスト |
| `test_source_formatting.py` | `source_formatting.py` | 参照元表示整形の共通ロジック（`format_snippet` / `format_source_label`）のユニットテストと、`app.py` / `api/main.py` の再エクスポートが同一実装を指していることの確認 |
| `test_api.py` | `api/main.py` | FastAPIバックエンドの各エンドポイント（チャットのSSEストリーミング、同期、会話ログの作成・件数取得・保存、thread_idのパストラバーサル対策）のテスト |
| `test_setup.py` | `setup.py` | LLMプロバイダの自動選択（`_build_model`）における非対話環境でのgetpass()ブロック回避、OllamaモデルのpullチェックのAPIやフォールバック挙動のテスト |
| `test_evaluate_retrieval.py` | `scripts/evaluate_retrieval.py` | 検索結果の適合率・再現率・F1計算ロジックのテスト |
| `test_select_next_issue.py` | `.github/scripts/select_next_issue.py` | AIチームのIssue選定ロジックのテスト（CI専用スクリプトのためファイルパスから直接import） |
| `test_ci_config.py` | `.github/workflows/ci.yml` / `pyproject.toml` | CIワークフロー定義（トリガー・実行ステップの順序）とruff設定（除外ファイル・ルールセット）の静的検証 |
| `test_ci_failure_guard_config.py` | `.github/workflows/ci-failure-guard.yml` | CI失敗時に自動対応するワークフロー定義（トリガー・権限・失敗要因の切り分けロジック）の静的検証 |
| `test_agent_workflow_dependency_setup.py` | `.github/workflows/ai-team.yml` / `pr-conflict-guard.yml` / `ci-failure-guard.yml` | AIチームのサブエージェント向けワークフローが、checkout直後に依存パッケージを事前インストールする構成になっていることの静的検証 |
| `test_theme_config.py` | `.streamlit/config.toml` / `app.py` | Streamlitのカスタムテーマ設定と `st.set_page_config` の引数（page_title/page_icon/layout）の静的検証 |
| `test_examples_scripts.py` | `examples/extract_text.py` / `examples/models_and_prompts.py` | チュートリアル用スクリプトが `examples/` 配下に存在すること・importだけではLLM呼び出し等の副作用が起きないことの検証 |
| `test_readme_sync.py` | `tests/README.md` | この一覧表に記載されたファイル名集合と `tests/` 配下に実在する `test_*.py` ファイル名集合が一致すること（表の陳腐化検知）のテスト |

## 新しいテストを追加する際の命名規則

- ファイル名は `test_<対応する実装ファイル名>.py`（例: `foo.py` に対して `test_foo.py`）
- テスト関数名は `test_<検証内容が分かる名前>`（例: `test_save_conversation_writes_markdown_file`）
- 正常系だけでなく、異常系（エラー時の挙動）・境界値（空データ、0件など）もできる限りカバーする
