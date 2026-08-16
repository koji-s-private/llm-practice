# llm-practice

無料ツールを活用した、ローカルドキュメントQ&A（RAG）チャットアプリ「Doclore」の個人開発プロジェクトです。

## 構成

| ファイル | 役割 |
|---|---|
| `setup.py` | モデルの初期化（Ollama優先→ANTHROPIC_API_KEY→OpenAIの順にフォールバック） |
| `ingest.py` | `data/` とベクトルDB(Chroma)の差分同期（CLIとしても、app.pyからの呼び出しとしても使用） |
| `rag_chain.py` | 検索ツール付きRAGエージェント（`create_agent`）の定義 |
| `memory.py` | 質問・回答を `data/conversations/<会話ID>/` に自動保存する会話ナレッジ化機能 |
| `app.py` | Streamlitのチャット画面（ファイルアップロードUI・新しい会話ボタンを含む） |
| `.streamlit/config.toml` | Streamlitのカスタムテーマ設定（配色・フォント。Issue #72） |
| `api/main.py` | FastAPI製バックエンドAPI（`ingest.py`/`rag_chain.py`/`memory.py`をラップ。フロントエンド移行Step1, Issue #88） |
| `data/` | 質問させたいPDF/テキストファイルを置く場所（アップロードUIからもここに保存される） |
| `data/conversations/<会話ID>/` | 自動保存された過去の質問・回答（会話ログ。会話IDごとにフォルダが分かれる） |
| `examples/models_and_prompts.py` / `examples/extract_text.py` | LangChain公式チュートリアルの学習用スクリプト（アプリ本体からは未使用。`python -m examples.extract_text` のように直接実行した場合のみLLMを呼び出す） |
| `tests/` | `ingest.py` / `rag_chain.py` / `memory.py` のコアロジックに対する自動テスト |

## 主要ディレクトリ

各ディレクトリの詳細な役割・置くべきファイルの種類・命名規則などは、それぞれの配下にある
`README.md` にまとめています（下表の「役割」は概要のみ。詳細は各リンク先を参照してください）。

| ディレクトリ | 役割（概要） | 詳細 |
|---|---|---|
| `data/` | 質問対象ファイル・会話ログの保存場所 | [data/README.md](data/README.md) |
| `tests/` | 自動テスト（pytest） | [tests/README.md](tests/README.md) |
| `.claude/agents/` | AIチーム（coder/qa-engineer/reviewer）の役割定義 | [.claude/agents/README.md](.claude/agents/README.md) |
| `.github/workflows/` | GitHub Actionsワークフロー定義 | [.github/workflows/README.md](.github/workflows/README.md) |
| `docs/` | 技術方針・設計に関するドキュメント | [docs/data-model.md](docs/data-model.md)（データ構造のER図）、[docs/frontend-tech-policy.md](docs/frontend-tech-policy.md) |

ディレクトリ構成の全体像は次の通りです（一部抜粋）。

```
llm-practice/
├── app.py                  # Streamlitのチャット画面
├── api/main.py              # FastAPI製バックエンドAPI（Issue #88, ingest/rag_chain/memoryをラップ）
├── ingest.py                # data/ とベクトルDBの差分同期
├── rag_chain.py              # RAGエージェントの定義
├── memory.py                 # 会話ログの自動保存
├── setup.py                  # モデルの初期化・フォールバック
├── data/                    # 質問対象ファイル・会話ログ（詳細: data/README.md）
│   └── conversations/<会話ID>/
├── tests/                   # 自動テスト（詳細: tests/README.md）
├── scripts/                 # 検索精度の評価スクリプトなど
├── .claude/agents/           # AIチームの役割定義（詳細: .claude/agents/README.md）
└── .github/
    ├── workflows/            # GitHub Actionsワークフロー（詳細: .github/workflows/README.md）
    └── scripts/               # ワークフロー専用の補助スクリプト
```

## 使い方

```bash
# 0. 仮想環境を作成 & 有効化（初回のみ作成、以降は毎回「有効化」が必要）
python -m venv .venv
source .venv/bin/activate

# 1. 依存パッケージをインストール
pip install -r requirements.txt

# 2. .env を用意（すでに .env がある場合は不要）
cp .env.example .env
# → 課金なしで使いたい場合は下記「無料で使う（Ollama）」を先に設定
#    課金APIでよければ .env に ANTHROPIC_API_KEY または OPENAI_API_KEY を設定

# 3. data/ に質問したいファイル(.pdf / .txt / .md / .docx / .csv / .xlsx / .xls / .pptx / .html / .htm)を置く
#    （サンプルとして data/sample.txt を同梱済み）
#    Googleスプレッドシート/ドキュメント/スライドは、Googleドライブの「ダウンロード」機能で
#    それぞれ .xlsx / .docx / .pptx としてエクスポートしてから data/ に置けば同様に取り込めます

# 4. チャットアプリを起動（起動時に data/ の内容が自動でDBに反映されます）
python -m streamlit run app.py
```

### 無料で使う（Ollama）

クレジットを一切消費せず、完全ローカル・無料で会話したい場合は [Ollama](https://ollama.com/) を使います。
`setup.py` はOllamaがローカルで起動しているかを自動検知し、**起動していれば最優先で使用**します
（`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` を`.env`から消す必要はありません。優先順位が変わるだけです）。

```bash
# 1. Ollamaをインストール（Homebrewの場合）
brew install ollama

# 2. Ollamaを起動（バックグラウンドで待受を開始。Macアプリ版なら起動するだけでOK）
ollama serve &

# 3. モデルをダウンロード（初回のみ。約4.9GB、tool呼び出し対応でRAGと相性が良いモデル）
ollama pull llama3.1

# 4. あとはいつも通り
source .venv/bin/activate
python -m streamlit run app.py
```

起動時に `[setup] Ollama を検出: llama3.1（ローカル・無料）を使用します。` と表示されれば成功です。
使いたいモデルを変えたい場合は `.env` に `OLLAMA_MODEL=任意のモデル名` を追加してください
（`ollama pull` で事前にダウンロードしたモデル名を指定します）。
Ollamaが起動していてもあえて使いたくない場合は `.env` に `DISABLE_OLLAMA=true` を設定してください。

> Macのスペックによっては応答が遅くなることがあります。速度を優先したい場合は
> より軽量な `ollama pull llama3.2`（3B、軽量版）も選択肢です（`.env`の`OLLAMA_MODEL`で指定）。

> **重要**: `.venv` の有効化（`source .venv/bin/activate`）はターミナルを閉じるたびにリセットされます。
> 新しいターミナルでコマンドを実行する前は、必ずプロジェクトフォルダで `source .venv/bin/activate`
> を実行してください。詳しくは下記「[毎回の起動手順](#毎回の起動手順ターミナルを開き直したとき)」を参照。

ブラウザで `http://localhost:8501` が開き、`data/` の内容について質問できます。
その後 `data/` にファイルを追加・削除した場合は、サイドバーの「🔄 data/ を再同期」ボタンを押すだけでDBに反映されます
（アプリを再起動する必要はありません）。

まとめてファイルを取り込みたい場合や、アプリを起動せずにDBだけ更新したい場合は
`python ingest.py` をCLIで実行することもできます（内部の処理は同じです）。

### バックエンドAPI（FastAPI）を起動する場合

フロントエンド移行（Issue #88, Step1）の一環として、Streamlit版とは別にFastAPI製のバックエンドAPIも
用意しています。現時点ではAPIを呼び出すフロントエンド（React等）は未実装のため、動作確認には
`curl` やブラウザの `http://localhost:8000/docs`（Swagger UI）を使ってください。

```bash
source .venv/bin/activate
uvicorn api.main:app --reload
```

主なエンドポイント:

| メソッド・パス | 役割 |
|---|---|
| `POST /api/chat` | チャット応答をSSE（Server-Sent Events）でストリーミング返却（`rag_chain.build_agent()`のラッパー） |
| `POST /api/sync` | `data/` 配下ドキュメントをベクトルDBに同期（`ingest.sync_data_dir()`のラッパー） |
| `POST /api/conversations/new` | 新しい会話スレッドIDを発行（`memory.new_thread_id()`のラッパー） |
| `GET /api/conversations/count` | 保存済み会話ログの件数を取得（`memory.conversation_count()`のラッパー） |
| `POST /api/conversations/save` | 質問・回答を会話ログとして保存（`memory.save_conversation()`のラッパー） |
| `GET /api/health` | 疎通確認用のヘルスチェック |

Streamlit版（`app.py`）と本APIは同じ `data/` / `chroma_db/` を参照するため、どちらか一方だけを
起動して使う分には競合しません（同時起動も可能です）。

### ファイルをdata/に手動で置かなくてもいい方法

`data/` フォルダを直接触らなくても、ブラウザ上の操作だけでナレッジを増やせます。

- **ファイルアップロード**: サイドバーの「ファイルを追加」からPDF/txt/md/docx/csv/xlsx/xls/pptx/html/htmを
  ドラッグ＆ドロップすると、自動的に `data/` に保存され、DBにも即座に反映されます。
- **会話の自動ナレッジ化**: サイドバーの「🧠 記憶設定」を開くと「今の会話を記憶として保存する」
  トグルがあります（デフォルトON）。ONのとき、チャットでのやりとりが `data/conversations/<会話ID>/`
  に自動保存され、**同じ会話の中でだけ**、以降の質問の回答材料として使われます。
  OFFにしたい場合はトグルを切り替えるだけです（過去に保存済みの会話は残ります）。
  同じ「🧠 記憶設定」内で、保存済みのやりとり件数や会話ID（内部識別用）も確認できます。

### 新しい・無関係な話題を始めたいとき

サイドバー上部の「🆕 新しい会話を始める」を押すと、新しい会話ID（サイドバーの
「🧠 記憶設定」を開くと確認できます）が発行され、画面上の会話履歴もリセットされます。これにより:

- 画面に表示される会話が真っさらになる
- **AIの回答時の検索対象からも、それ以前の会話ログが除外される**
  （PDFなどの共通ナレッジは引き続き検索対象のままです）

つまり、全く別の話題を始めたいときにこのボタンを押せば、過去の無関係なやりとりが
新しい会話の回答に混ざり込むことはありません。ボタンを押さずに同じ会話を続ければ、
その会話内での文脈（過去の質問・回答）を踏まえた回答が可能です。
過去の会話に戻りたい場合の「会話の再開」機能は今のところありません（今後の発展案）。

**セキュリティについて**: アップロードされたファイルも会話ログも、保存先はこのプロジェクト内の
`data/` フォルダとローカルの`chroma_db/`のみです。埋め込みもローカルのHuggingFaceモデルで行うため、
この保存・検索の仕組み自体が外部やクラウドにデータを送信することはありません。
ただし、チャットの回答生成にOllama以外（Claude/OpenAI）を使っている場合、その質問文と検索結果は
回答生成のためにAnthropic/OpenAIのAPIへ送信されます（これは会話の自動保存機能とは無関係に、
通常のチャット機能として元々発生するものです）。この送信も含めて一切外部に出したくない場合は、
上記「無料で使う（Ollama）」を使ってください。
なお、LangSmithへのトレース送信は完全に任意（デフォルトOFF）で、`.env`で明示的に有効化しない限り発生しません。

## 使用技術

新規参画者・初学者向けに、主要なライブラリ・ミドルウェアが「何をするためのものか」「このプロジェクトの
どこで・どう使われているか」を一覧にまとめています（実装の意図・設計判断は次の
「[技術構成とベストプラクティスのポイント](#技術構成とベストプラクティスのポイント)」を参照してください）。

### UI

| ライブラリ | 役割 | このプロジェクトでの使用箇所 |
|---|---|---|
| [Streamlit](https://streamlit.io/) | Pythonだけでブラウザ上のチャットUIを構築できるフレームワーク | `app.py`。チャット画面本体（`st.chat_input` / `st.chat_message`）、サイドバー（ファイルアップロード・再同期ボタン・会話管理トグル）を実装 |

### バックエンドAPI（Issue #88）

[docs/frontend-tech-policy.md](docs/frontend-tech-policy.md)の移行計画Step1として追加。既存の
Streamlit版（`app.py`）とは別に、将来のTypeScript製フロントエンド（React + Vite）から呼び出せる
HTTP API層を提供する。

| ライブラリ | 役割 | このプロジェクトでの使用箇所 |
|---|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | Python製のWeb APIフレームワーク（型ヒントベースのリクエスト/レスポンス検証、`StreamingResponse`によるストリーミング配信に対応） | `api/main.py`。`ingest.py` / `rag_chain.py` / `memory.py` をラップするエンドポイント（`/api/chat` / `/api/sync` / `/api/conversations/*`）を定義 |
| [uvicorn](https://www.uvicorn.org/)（`uvicorn[standard]`） | FastAPIアプリをローカルで起動するASGIサーバー | `uvicorn api.main:app --reload` でローカル起動（詳細は「[バックエンドAPI（FastAPI）を起動する場合](#バックエンドapifastapiを起動する場合)」） |

### LLMエージェント・オーケストレーション（LangChain）

| ライブラリ | 役割 | このプロジェクトでの使用箇所 |
|---|---|---|
| `langchain`（`langchain[openai]`） | LLM呼び出し・エージェント構築・プロンプト管理などを抽象化するフレームワーク本体 | `setup.py`（`init_chat_model`でモデルを初期化）、`rag_chain.py`（`create_agent`で検索ツール付きRAGエージェントを構築、`@tool`で検索ツールを定義） |
| `langchain-core` | LangChainのメッセージ型・基底クラスなど（`langchain`本体に付随して導入される） | `app.py`（`AIMessage` / `HumanMessage` / `ToolMessage`で会話履歴を管理） |
| `langchain-anthropic` | LangChainからAnthropic（Claude）を呼び出すためのプロバイダ連携パッケージ | `setup.py`。`ANTHROPIC_API_KEY`設定時のフォールバック先（`init_chat_model(..., model_provider="anthropic")`） |
| `langchain-ollama` | LangChainからOllama（ローカルLLM）を呼び出すためのプロバイダ連携パッケージ | `setup.py`。Ollama起動を検出した場合の最優先モデル（`init_chat_model(..., model_provider="ollama")`） |

### 検索・ベクトルDB（RAG）

| ライブラリ / ミドルウェア | 役割 | このプロジェクトでの使用箇所 |
|---|---|---|
| [Ollama](https://ollama.com/) | LLMをローカルPC上で無料実行するためのランタイム（アプリ本体とは別プロセスで起動） | `setup.py`が`localhost:11434`への接続有無で起動を検知し、起動していれば回答生成・関連度採点に最優先で使用 |
| [Chroma](https://www.trychroma.com/)（`langchain-chroma`） | ドキュメントのベクトル（埋め込み）を保存し、類似検索するローカル動作のベクトルDB | `rag_chain.py`の`get_vectorstore()`でChromaインスタンスを生成。`chroma_db/`フォルダにローカル永続化（`ingest.py`のデータ同期、`rag_chain.py`の検索ツールから利用） |
| `langchain-huggingface` | Hugging Face製の埋め込みモデルをLangChain経由で使うための連携パッケージ | `rag_chain.py`の`get_embeddings()`（`HuggingFaceEmbeddings`） |
| `sentence-transformers` | 埋め込みモデル（`sentence-transformers/all-mpnet-base-v2`）を実際にロード・推論するエンジン（`langchain-huggingface`の内部で使用） | `rag_chain.py`の`get_embeddings()`が指定するモデルの実行エンジン |
| `langchain-text-splitters` | 長いドキュメントを検索・埋め込みに適したチャンク（断片）に分割するツール | `ingest.py`の`sync_data_dir()`（`RecursiveCharacterTextSplitter`でチャンク分割） |
| [filelock](https://py-filelock.readthedocs.io/) | クロスプラットフォーム対応のファイルロックライブラリ | `ingest.py`の`sync_data_dir()`。複数タブ（複数Streamlitセッション）や複数プロセスから同時に呼ばれても、manifest.json読み込み〜ベクトルDB更新〜書き込みを1つずつ排他的に実行するために使用（`chroma_db/sync.lock`） |

### ドキュメント読み込み

| ライブラリ | 役割 | このプロジェクトでの使用箇所 |
|---|---|---|
| `langchain-community` | PDF/テキスト/Word/CSV/HTMLファイルをLangChainのDocument形式で読み込むローダー群を提供（2026年6月にsunset済み、詳細は下記注意点を参照） | `ingest.py`（`PyMuPDFLoader` / `TextLoader` / `Docx2txtLoader` / `CSVLoader` / `BSHTMLLoader`） |
| `pymupdf` | PDFからテキストを高速抽出するエンジン（`PyMuPDFLoader`が内部で使用） | `ingest.py`の`_load_pdf()`（1段目の高速抽出） |
| `cryptography` | 暗号化（パスワード付き）PDFの復号に必要 | `pymupdf`によるPDF読み込み時に内部的に使用（`ingest.py`） |
| `docx2txt` | Word（`.docx`）ファイルからテキストを抽出するエンジン（`Docx2txtLoader`が内部で使用） | `ingest.py`の`LOADERS`（`.docx`ファイルの読み込み） |
| `openpyxl` | Excel（`.xlsx`）ファイルを読み書きするライブラリ | `ingest.py`の`_ExcelLoader`（シートごとに1つのDocumentとして読み込む自前ローダー。依存が重い`unstructured`パッケージを避けるため自前実装にしている） |
| `xlrd` | 旧バイナリ形式Excel（`.xls`）ファイルを読み込むライブラリ（`openpyxl`は`.xlsx`専用で`.xls`は読めないため別ライブラリが必要） | `ingest.py`の`_LegacyExcelLoader`（`_ExcelLoader`と同じ出力形式の自前ローダー） |
| `python-pptx` | PowerPoint（`.pptx`）ファイルを読み書きするライブラリ | `ingest.py`の`_PowerPointLoader`（スライドごとに1つのDocumentとして読み込む自前ローダー。openpyxlと同じ理由で自前実装） |
| `beautifulsoup4` | HTMLをパースするライブラリ | `ingest.py`の`LOADERS`（`BSHTMLLoader`が内部で使用し、`.html`/`.htm`ファイルからテキストを抽出） |
| `lxml` | 高速なHTML/XMLパーサー | `BSHTMLLoader`がデフォルトで使用するパーサーエンジン |
| `docling` / `langchain-docling`（任意インストール） | レイアウト認識・表構造認識・OCRに対応した高精度なドキュメント解析ライブラリ | `ingest.py`の`_load_pdf()`。PyMuPDFでの抽出文字数が極端に少ない（図解・スキャンPDFの疑いがある）場合のみフォールバックとして使用。未インストールでも動作する |

### その他

| ライブラリ | 役割 | このプロジェクトでの使用箇所 |
|---|---|---|
| `python-dotenv` | `.env`ファイルから環境変数を読み込む | `setup.py`（`load_dotenv()`）。`ANTHROPIC_API_KEY`等のAPIキーやOllama関連の設定を読み込む |
| `pytest` | テストフレームワーク | `tests/`配下の自動テスト一式（実行方法は下記「[テスト](#テスト)」を参照） |
| `PyYAML` | YAMLファイルのパース | `tests/test_ci_config.py` / `tests/test_ci_failure_guard_config.py`。GitHub Actionsのワークフロー定義（YAML）を読み込んで内容を検証するために使用（従来は他パッケージの推移的依存として導入されていたが、明示的に固定） |
| `xlwt` | 旧バイナリ形式Excel（`.xls`）ファイルを書き出すライブラリ | `tests/test_ingest.py`で`.xls`取り込みテスト用のフィクスチャファイルを生成するためだけに使用（読み込み側の`xlrd`とは別ライブラリ） |
| [ruff](https://docs.astral.sh/ruff/) | lint（静的解析）・コードフォーマットを1ツールでカバーするツール | `pyproject.toml`の`[tool.ruff]`でルールセットを設定。[.github/workflows/ci.yml](.github/workflows/ci.yml)がpush/PR時に`ruff check`・`ruff format --check`を自動実行 |

> **README肥大化時の分割方針**: このセクションが今後さらに大きくなった場合は、`docs/tech-stack.md`
> のような別ファイルに切り出し、ルートの`README.md`側にはリンクのみを残す方針とします
> （`data/README.md` / `tests/README.md`などの既存の「詳細は別ファイル」構成にならう）。
> 新しい技術・ライブラリを追加した際の更新ルールは [AGENTS.md](AGENTS.md) の
> 「コード品質」セクションを参照してください。

### 依存パッケージのバージョン管理・更新方針（Issue #64）

`requirements.txt`に記載の全パッケージは、動作確認済みのバージョンに`==`で固定しています。
LangChain関連は短期間で破壊的変更が入った実績がある（`RetrievalQA`/`create_retrieval_chain`の
非推奨化など）ため、`pip install -r requirements.txt`実行のたびに意図せず最新版へ
アップグレードされてCIやローカル環境が突然壊れることを防ぐのが目的です。

- **通常運用**: バージョン固定を維持し、`pip`による自動アップグレードは行いません。
- **更新のタイミング**: 新機能で新しいAPIが必要になった場合、または下記のセキュリティ監視で
  脆弱性が見つかった場合に、手動でバージョンを上げてから動作確認（`python -m pytest`等）を行います。
- **脆弱性監視**: 追加費用なしで使える [GitHub Dependabot alerts](https://docs.github.com/ja/code-security/dependabot/dependabot-alerts)
  と、手動実行の[pip-audit](https://pypi.org/project/pip-audit/)（`pip install --user pip-audit` &&
  `pip-audit -r requirements.txt`）を組み合わせて監視します（詳細は[AGENTS.md](AGENTS.md)の
  「コード品質」セクション参照）。Dependabotの自動更新PR（`.github/dependabot.yml`によるバージョン
  自動引き上げPR）は現時点では導入せず、alertsによる検知のみを利用します（自動更新PRの導入は
  必要になった時点で別Issueとして検討します）。

## 技術構成とベストプラクティスのポイント

2026年時点のLangChain公式ドキュメント（[docs.langchain.com/oss/python/langchain/rag](https://docs.langchain.com/oss/python/langchain/rag)）が
「汎用用途に最適」として推奨する **Agentic RAG**（`create_agent` + 検索ツール）構成を採用しています。
LangChainは1.0でメジャーアップデートされており、旧来の `RetrievalQA` や `create_retrieval_chain`
（現在は `langchain_classic` に移動した非推奨扱いのAPI）は使用していません。

- **埋め込み（検索用）**: HuggingFaceのローカルモデル（`sentence-transformers/all-mpnet-base-v2`）を使用。
  APIキー不要・無料・オフラインで動作し、埋め込みのたびに課金が発生しません。
- **回答生成・検索判断**: `setup.py` が 1) Ollama（ローカルで起動していれば最優先・無料） 2) `ANTHROPIC_API_KEY`
  （Claude, claude-sonnet-5） 3) `OPENAI_API_KEY`（OpenAI, gpt-5-chat-latest）の順に自動フォールバックします。
  このモデルを `create_agent` に渡すことで、質問に応じて検索ツール（`retrieve_context`）を
  呼ぶかどうかをエージェント自身が判断します。追加質問への対応や複数回の検索も、
  会話履歴（メッセージ配列）を踏まえて自律的に行われます。
- **PDF解析（2段構成）**: まず [PyMuPDF](https://pymupdf.readthedocs.io/) で高速に一次抽出し、
  抽出できた文字数が極端に少ない場合（図解・スキャンPDFの疑い）だけ、[Docling](https://github.com/docling-project/docling)
  （レイアウト認識・表構造認識・OCR内蔵）で再解析します。通常のテキストPDFはPyMuPDFだけで
  高速に処理され、Doclingの重い処理は「本当に必要なファイルだけ」に絞られるため、
  精度と速度を両立しています。Doclingは任意インストールで、未インストールでもPyMuPDFのみで動作します。
- **ベクトルDB**: Chroma。ローカルの `chroma_db/` フォルダに永続化（`.gitignore` 済み）。
- **自動同期**: `data/` フォルダ（`data/conversations/` などのサブフォルダも再帰的に対象）の
  追加・変更・削除は `chroma_db/manifest.json` で差分検知され、アプリ起動時・サイドバーの
  再同期ボタン・ファイルアップロード時・会話終了時に自動的にDBへ反映されます。
- **会話の自動ナレッジ化とスレッド分離**: `memory.py` が質問・回答を1件1ファイル（.md）として
  `data/conversations/<会話ID>/` にローカル保存し、他のファイルと同じ仕組みでDBに取り込みます。
  過去の会話をre-embeddingし直すことはなく、新規ファイルとして差分追加されるだけなので効率的です。
  `ingest.py`がファイルパスから会話IDをメタデータ（`thread_id`）としてチャンクに付与し、
  `rag_chain.build_agent(thread_id)`側の検索フィルタでその会話ID＋共通ナレッジ（`thread_id="global"`）
  だけに絞り込むため、無関係な別の会話の内容が回答に混ざりません。

> **既知の注意点**: PDF/テキスト読み込みのローダー（`PyMuPDFLoader`/`TextLoader`）を提供している
> `langchain-community` パッケージは2026年6月にsunset（アーカイブ・保守終了）されました。現状は動作しますが、今後は
> プロバイダごとの独立パッケージへの移行が進む見込みです。動かなくなった場合はこの点を疑ってください。

## 必要な環境変数

- 何も設定しなくてもOK: Ollamaがローカルで起動していれば自動検出され、無料で使えます（上記「無料で使う（Ollama）」参照）。
- `ANTHROPIC_API_KEY` または `OPENAI_API_KEY`（回答生成用、従量課金）: Ollamaが使えない場合のフォールバックとして、
  どちらもすでに `.env` に設定済みです。両方あれば `ANTHROPIC_API_KEY` が優先されます。
- `OLLAMA_MODEL`（任意）: 使用するOllamaのモデル名を変更したい場合のみ（デフォルト: `llama3.1`）。
- `DISABLE_OLLAMA`（任意）: `true` にすると、Ollamaが起動していてもあえて使わずAPI課金モデルに切り替えます。
- `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT`（任意・デフォルトでは無効）: LangSmithでトレースを見たい場合のみ、
  `.env` に `LANGSMITH_TRACING=true` と `LANGSMITH_API_KEY` を明示的に設定してください。設定しない限り、
  会話内容がLangSmith（外部サービス）へ送信されることはありません。

埋め込み（ドキュメント検索用）はローカル無料モデルのため、追加の環境変数は不要です。

## テスト

コアロジック（`ingest.py` の差分同期、`rag_chain.py` の関連度採点・フォールバック、
`memory.py` の会話ログ保存・件数カウント）に対する自動テストを `tests/` 配下に用意しています。

```bash
pip install -r requirements.txt  # pytestを含む
pytest
```

埋め込みモデル・Chroma・LLMプロバイダなどの重い外部依存は `tests/conftest.py` で
軽量なフェイクに差し替えているため、Ollamaの起動やAPIキーの設定、埋め込みモデルの
ダウンロードなしにテストを実行できます（ネットワークアクセス・課金は発生しません）。

さらに、lint（静的解析）・フォーマットチェックには [ruff](https://docs.astral.sh/ruff/) を使用しています。

```bash
ruff check .           # lint
ruff format --check .  # フォーマットチェック（差分を直接適用したい場合は `ruff format .`）
```

[.github/workflows/ci.yml](.github/workflows/ci.yml) がpush・PR作成時に上記の
`ruff check` / `ruff format --check` / `pytest` を自動実行し、いずれかが失敗すると
CIジョブ全体が失敗します（マージ前に問題へ気づける状態にするため）。

## 今後の発展案

- 過去の会話（会話ID）を一覧から選んで再開する機能（現状は新しい会話を始めることしかできない）

## 保守・運用

### 毎回の起動手順（ターミナルを開き直したとき）

`.venv` の有効化はターミナルのセッションごとの状態なので、ターミナルを閉じたり
PCを再起動したりすると毎回リセットされます。**「一度作ればずっと有効」ではありません。**
新しいターミナルでこのプロジェクトのコマンド（`python -m streamlit run app.py` や `python ingest.py` など）
を実行する前は、必ず以下を実行してください。

```bash
cd ~/private-practice/llm-practice   # プロジェクトフォルダに移動
source .venv/bin/activate            # 仮想環境を有効化（プロンプト先頭に (.venv) と出ればOK）
python -m streamlit run app.py
```

有効化を忘れると、`ModuleNotFoundError: No module named 'langchain'` のようなエラーになったり、
pyenvなど別のPython環境のパッケージを見に行ってしまったりします。エラーが出たら、まず
`which python` を実行し、`.venv/bin/python` になっているか確認してください。

> **なぜ `streamlit run app.py` ではなく `python -m streamlit run app.py` なのか**:
> pyenvを使っている環境では、`.venv`を有効化していても`streamlit`コマンド自体は
> pyenvのshim（`~/.pyenv/shims/streamlit`）が優先されてしまうことがあります
> （`which python`は`.venv`を指すのに`which streamlit`はpyenv側、という食い違いが起きる）。
> `python -m streamlit ...`と書けば、確認済みの`.venv`のPythonから直接streamlitモジュールを
> 呼び出すため、このPATHの取り合いを回避できます。

> VSCodeの統合ターミナルは、インタープリタとして `.venv` を選択していれば新しいターミナルを
> 開くたびに自動で有効化してくれることが多いです（プロンプトに `(.venv)` が出ていれば有効化済み）。
> 自動で有効化されない場合は、上記のコマンドを手動で実行してください。

### DBの実体

「DB」は [Chroma](https://www.trychroma.com/) というローカル動作のベクトルデータベースです。
サーバーではなく、`chroma_db/` フォルダの中にファイルとして保存されます（`.gitignore`済みなので、
Gitやこのプロジェクトの共有には含まれません）。あわせて `chroma_db/manifest.json` に
「どのファイルをいつ取り込んだか」の対応表を持っています。

```
data/
├── sample.txt
├── （アップロードしたPDF/txt/mdファイル）
└── conversations/
    └── <会話ID>/       # 会話ごとにフォルダが分かれる（例: a1b2c3d4/）
        └── xxx.md      # 1問答ごとに1ファイル

chroma_db/
├── manifest.json     # data/ のファイルとチャンクIDの対応表（このプロジェクト独自）
└── （Chroma本体のデータファイル一式）
```

各要素の詳しい関係（ファイル・manifest.json・Chromaのチャンク・会話スレッドが互いにどう
紐づいているか）はER図として [docs/data-model.md](docs/data-model.md) にまとめています。

### DBの中身を確認する

```bash
python ingest.py --status
```

インデックス済みのファイル数、チャンク（ベクトル）数、DBフォルダの容量、
ファイルごとのチャンク数が一覧表示されます。「質問に対する回答が古い/おかしい」と感じたときの
最初の確認先です。

### DBをリセットしたい場合

`chroma_db/` フォルダを削除するだけです。次回 `python -m streamlit run app.py` または `python ingest.py`
を実行したときに、`data/` の内容から自動的に作り直されます。

```bash
rm -rf chroma_db
python ingest.py
```

### 会話ログ（会話の自動ナレッジ化）の管理

`data/conversations/<会話ID>/` にたまった過去の会話は、不要になったら普通のファイルと同じように
削除できます。削除後にサイドバーの「🔄 data/ を再同期」を押す（または `python ingest.py`）と
DB側からも自動的に削除されます。

```bash
# 例: 全ての会話ログを削除したい場合
rm -rf data/conversations
python ingest.py

# 例: 特定の会話（会話ID）だけ削除したい場合
rm -rf data/conversations/<会話ID>
python ingest.py
```

雑談やテスト目的のやりとりまでナレッジ化したくない場合は、サイドバーの「🧠 記憶設定」内の
「今の会話を記憶として保存する」トグルをその場でOFFにしてください。

> 補足: 会話のスレッド分離機能を追加する前に保存された会話ログ（`data/conversations/`直下に
> サブフォルダなしで置かれているファイル）は、後方互換のため共通ナレッジ（`global`）として
> 扱われ、引き続きどの会話からも参照されます。

### バックアップ

`chroma_db/` フォルダをまるごとコピーしておけばバックアップになります
（元データの `data/` 側も残しておけば、最悪DBは失っても再構築可能です）。

### 依存パッケージの更新

```bash
pip install -r requirements.txt --upgrade
```

特に `langchain` 本体・`langchain-anthropic`・`langchain-openai` はモデル追加や
API変更が頻繁なので、動作がおかしくなったときはまずここを疑ってください。

### トラブルシューティング

| 症状 | 確認・対処 |
|---|---|
| 回答が「見つかりませんでした」ばかり | `python ingest.py --status` でファイルが取り込まれているか確認。0件なら `data/` にファイルを置いて `python ingest.py` かサイドバーの再同期ボタンを実行 |
| `anthropic.BadRequestError: credit balance is too low` 等の課金エラー | Anthropic/OpenAIのクレジット残高不足。上記「無料で使う（Ollama）」を導入するか、Consoleでクレジットを購入 |
| 起動時にAPIキーを聞かれる | Ollamaが未起動、かつ `.env` に `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` も未設定。どちらか設定するかOllamaを起動 |
| PDFが読み込めない（`DependencyError: cryptography...`） | 暗号化PDFの復号に`cryptography`が必要。`pip install -r requirements.txt`（または`pip install cryptography`）を実行 |
| PDFが読み込めない（それ以外のエラー） | `langchain-community`（PDF読み込みに使用、2026年6月にsunset）の不具合の可能性。`pip install --upgrade pymupdf langchain-community` を試す |
| PDFのテキストが文字化けする・図解のラベルが読めない | まずPyMuPDFで再取得を試みるので `rm -rf chroma_db && python ingest.py` で再インデックスしてみる。改善しない場合は`docling`/`langchain-docling`をインストールすると図解・スキャンPDF向けの高品質フォールバックが有効になる（`pip install docling langchain-docling`、初回はモデルダウンロードあり） |
| DBの内容がおかしい・壊れた | 上記「DBをリセットしたい場合」を参照 |
| 初回起動が遅い | 初回のみ埋め込みモデル（`sentence-transformers/all-mpnet-base-v2`、約420MB）を自動ダウンロードするため。2回目以降はキャッシュされ高速化されます |
| Ollamaを入れたのに課金APIが使われる | `ollama serve` が起動しているか確認（`ollama list` でエラーが出ないか）。`.env`に`DISABLE_OLLAMA=true`が残っていないかも確認 |
| `.venv`を有効化したのに `ModuleNotFoundError` や pyenv側のパッケージが読まれる | `which python`と`which streamlit`を比較。`python`は`.venv/bin/python`なのに`streamlit`だけpyenvのshimを指している場合、`streamlit run app.py`ではなく`python -m streamlit run app.py`を使う |
| アップデート後、以前は答えられていた内容に急に答えられなくなった | 会話スレッド分離機能の追加などでDBの内部構造が変わった可能性。上記「DBをリセットしたい場合」で `chroma_db` を作り直す |

### 課金について

- **Ollama（ローカル）は完全無料**です。電気代以外の追加コストはかかりません。クレジットを消費したくない場合は
  上記「無料で使う（Ollama）」の手順で導入してください。
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` はいずれもトークン量に応じた従量課金です。
  Claude.aiのPro/MaxなどのサブスクリプションやChatGPTの契約プランとは別課金・別枠なので、
  「契約プランの範囲内だから無料」にはなりません。Ollamaが起動していない時のフォールバックとしてのみ使われます。
- 埋め込み（ドキュメント検索用）はローカルのHuggingFaceモデルを使うため、バックエンドによらず課金は発生しません。
