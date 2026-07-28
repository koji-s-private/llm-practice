# llm-practice

無料ツールを活用した、ローカルドキュメントQ&A（RAG）チャットアプリの個人開発プロジェクトです。

## 構成

| ファイル | 役割 |
|---|---|
| `setup.py` | モデルの初期化（Ollama優先→ANTHROPIC_API_KEY→OpenAIの順にフォールバック） |
| `ingest.py` | `data/` とベクトルDB(Chroma)の差分同期（CLIとしても、app.pyからの呼び出しとしても使用） |
| `rag_chain.py` | 検索ツール付きRAGエージェント（`create_agent`）の定義 |
| `memory.py` | 質問・回答を `data/conversations/<会話ID>/` に自動保存する会話ナレッジ化機能 |
| `app.py` | Streamlitのチャット画面（ファイルアップロードUI・新しい会話ボタンを含む） |
| `data/` | 質問させたいPDF/テキストファイルを置く場所（アップロードUIからもここに保存される） |
| `data/conversations/<会話ID>/` | 自動保存された過去の質問・回答（会話ログ。会話IDごとにフォルダが分かれる） |
| `models_and_prompts.py` / `extract_text.py` | LangChain公式チュートリアルの学習用スクリプト（そのまま残しています） |
| `tests/` | `ingest.py` / `rag_chain.py` / `memory.py` のコアロジックに対する自動テスト |

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

# 3. data/ に質問したいファイル(.pdf / .txt / .md)を置く
#    （サンプルとして data/sample.txt を同梱済み）

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

### ファイルをdata/に手動で置かなくてもいい方法

`data/` フォルダを直接触らなくても、ブラウザ上の操作だけでナレッジを増やせます。

- **ファイルアップロード**: サイドバーの「ファイルを追加」からPDF/txt/mdをドラッグ＆ドロップすると、
  自動的に `data/` に保存され、DBにも即座に反映されます。
- **会話の自動ナレッジ化**: サイドバーの「質問・回答を自動で保存する」がONのとき
  （デフォルトON）、チャットでのやりとりが `data/conversations/<会話ID>/` に自動保存され、
  **同じ会話（同じ会話ID）の中でだけ**、以降の質問の回答材料として使われます。
  OFFにしたい場合はトグルを切り替えるだけです（過去に保存済みの会話は残ります）。

### 新しい・無関係な話題を始めたいとき

サイドバー上部の「🆕 新しい会話を始める」を押すと、新しい会話ID（表示されている
`会話ID: xxxxxxxx`）が発行され、画面上の会話履歴もリセットされます。これにより:

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

雑談やテスト目的のやりとりまでナレッジ化したくない場合は、サイドバーの
「質問・回答を自動で保存する」トグルをその場でOFFにしてください。

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
