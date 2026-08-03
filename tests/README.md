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

| テストファイル | 対応する実装 | 内容 |
|---|---|---|
| `conftest.py` | - | pytest共通設定。埋め込みモデル・Chroma・LLMプロバイダをフェイクに差し替える |
| `test_ingest.py` | `ingest.py` | `data/` とベクトルDBの差分同期ロジック（追加・更新・削除の判定）のテスト |
| `test_ingest_manifest.py` | `ingest.py` | `chroma_db/manifest.json` の読み書き（アトミックな書き込み、壊れたJSONへのフォールバック）のテスト |
| `test_rag_chain.py` | `rag_chain.py` | 検索結果の関連度採点（`_grade_relevance`）と、「見つからない場合」のフォールバック応答のテスト |
| `test_memory.py` | `memory.py` | 会話ログの保存（Markdownファイル書き込み）・件数カウントのテスト |
| `test_app.py` | `app.py` | Streamlitのチャット画面のエラーハンドリングのテスト（`streamlit.testing.v1.AppTest` を使いスクリプト実行エンジン上で検証） |
| `test_evaluate_retrieval.py` | `scripts/evaluate_retrieval.py` | 検索結果の適合率・再現率・F1計算ロジックのテスト |
| `test_select_next_issue.py` | `.github/scripts/select_next_issue.py` | AIチームのIssue選定ロジックのテスト（CI専用スクリプトのためファイルパスから直接import） |
| `test_ci_config.py` | `.github/workflows/ci.yml` / `pyproject.toml` | CIワークフロー定義（トリガー・実行ステップの順序）とruff設定（除外ファイル・ルールセット）の静的検証 |
| `test_theme_config.py` | `.streamlit/config.toml` / `app.py` | Streamlitのカスタムテーマ設定と `st.set_page_config` の引数（page_title/page_icon/layout）の静的検証（Issue #72） |

## 新しいテストを追加する際の命名規則

- ファイル名は `test_<対応する実装ファイル名>.py`（例: `foo.py` に対して `test_foo.py`）
- テスト関数名は `test_<検証内容が分かる名前>`（例: `test_save_conversation_writes_markdown_file`）
- 正常系だけでなく、異常系（エラー時の挙動）・境界値（空データ、0件など）もできる限りカバーする
