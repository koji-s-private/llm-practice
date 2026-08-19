# Google Drive連携のセットアップ手順（人間による事前作業）

`google_drive_sync.py` を使うと、Google Drive上の指定フォルダの内容（Googleスプレッドシート/
ドキュメント/スライドや通常ファイル）を手動エクスポート・手動配置なしに `data/google_drive/`
へライブ同期し、既存の `ingest.py` の仕組みでベクトルDBまで反映できます。

この機能を使うには、**Google Cloud Consoleでのプロジェクト作成・API有効化・OAuth同意画面の設定・
OAuthクライアントIDの発行**が事前に必要です。これらはAIチーム（coder/qa-engineer）が自動化できる
操作ではなく、人間（オーナー）がブラウザで一度だけ行う手動セットアップです。

> **課金について**: 以下の手順はすべて無料の範囲内（Google Cloud Consoleの無料利用、Drive APIの
> 無料枠）で完結します。有料のGoogle Workspace契約やAPIの有料利用枠は一切必要ありません。
> 手順の中でクレジットカード登録や請求先アカウントの作成を求められることはありません。

## 1. Google Cloud Consoleでプロジェクトを作成する

1. [Google Cloud Console](https://console.cloud.google.com/) にアクセスし、Googleアカウントでログインします。
2. 画面上部のプロジェクト選択メニューから「新しいプロジェクト」を選び、任意の名前
   （例: `llm-practice-drive-sync`）で作成します。

## 2. Google Drive APIを有効化する

1. 作成したプロジェクトを選択した状態で、「APIとサービス」→「ライブラリ」に移動します。
2. 「Google Drive API」を検索し、「有効にする」をクリックします。

## 3. OAuth同意画面を設定する（Testingモード）

1. 「APIとサービス」→「OAuth同意画面」に移動します。
2. User Type は「外部」を選択します（個人のGoogleアカウントを使う場合）。
3. アプリ名（例: `llm-practice`）、ユーザーサポートメール、デベロッパーの連絡先情報を入力して保存します。
4. 公開ステータスは **「Testing（テスト）」のままにします**（本番公開審査は不要で、課金も発生しません）。
5. 「テストユーザー」に、実際に使うGoogleアカウントのメールアドレスを追加します
   （Testingモードでは、ここに追加したアカウントしか認証フローを通せません）。

## 4. OAuthクライアントIDを発行する

1. 「APIとサービス」→「認証情報」に移動し、「認証情報を作成」→「OAuthクライアントID」を選びます。
2. アプリケーションの種類は **「デスクトップアプリ」** を選択します
   （`google_drive_sync.py` はローカルPC上で `InstalledAppFlow.run_local_server()` によるブラウザ認証を
   行うため、Webアプリケーション種別ではなくデスクトップアプリ種別が必要です）。
3. 任意の名前（例: `llm-practice-desktop`）を付けて作成します。
4. 作成後に表示される「JSONをダウンロード」から、クライアントシークレットファイルをダウンロードします。

## 5. クライアントシークレットファイルを配置する

ダウンロードしたJSONファイルを、リポジトリ直下の `.credentials/client_secret.json` に配置します。

```bash
mkdir -p .credentials
mv ~/Downloads/client_secret_xxxxx.json .credentials/client_secret.json
```

`.credentials/` ディレクトリは `.gitignore` で除外済みのため、誤ってリポジトリにコミットされる
心配はありません。

## 6. 同期対象のDriveフォルダIDを`.env`に設定する

1. Google Driveで同期したいフォルダを開き、URLの末尾（`https://drive.google.com/drive/folders/`の後ろ）
   にあるIDをコピーします。
   例: `https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz` → `1AbCdEfGhIjKlMnOpQrStUvWxYz`
2. `.env` に以下を追記します（`.env.example` も参照）。

```bash
GOOGLE_DRIVE_FOLDER_ID=1AbCdEfGhIjKlMnOpQrStUvWxYz

# 任意: 配置場所を変えたい場合のみ（デフォルトは.credentials/配下）
# GOOGLE_OAUTH_CLIENT_SECRET_FILE=.credentials/client_secret.json
# GOOGLE_OAUTH_TOKEN_FILE=.credentials/token.json
```

## 7. 初回同期を実行し、ブラウザで同意する

```bash
python google_drive_sync.py
```

初回実行時はデフォルトブラウザが自動で開き、Googleアカウントでのログインとアクセス許可（読み取り専用）の
同意を求められます。同意すると、以降の実行に使うトークンが `.credentials/token.json` に自動保存され、
次回以降はブラウザ操作なしで自動的にトークンがリフレッシュされます。

実行が完了すると、指定フォルダ内のGoogleスプレッドシート/ドキュメント/スライドはそれぞれ
`.xlsx`/`.docx`/`.pptx` としてエクスポートされ、それ以外の対応拡張子（PDF/txt等）のファイルは
そのままダウンロードされて `data/google_drive/` に配置されます。同時に `ingest.sync_data_dir()` が
呼ばれ、ベクトルDBにも反映されます。

## トラブルシューティング

- `OAuthクライアントシークレットファイルが見つかりません` というエラーが出る場合:
  上記手順4〜5（OAuthクライアントIDの発行、`.credentials/client_secret.json` への配置）が
  未完了です。
- `GOOGLE_DRIVE_FOLDER_ID が未設定のため、Google Drive同期をスキップします` という警告が出る場合:
  `.env` に `GOOGLE_DRIVE_FOLDER_ID` が設定されていません（上記手順6）。
- 認証エラーが繰り返し発生する場合: `.credentials/token.json` を削除して再実行すると、
  ブラウザでの再認証フローからやり直せます。
