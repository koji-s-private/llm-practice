# frontend

Doclore（ローカルRAGチャットアプリ）の新フロントエンド（Vite + React + TypeScript）です。
既存のStreamlit版（ルートの `app.py`）を置き換える移行計画（[docs/frontend-tech-policy.md](../docs/frontend-tech-policy.md)）
のStep2（基盤構築）・Step3（チャットUI実装）まで完了しています。ファイルアップロード・会話管理UIは
Step4・Step5で追加予定です。

## 使用技術

| 用途                        | ライブラリ                                                                                                                                                                                                             | このディレクトリでの使用箇所                                                                                                            |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| ビルドツール                | [Vite](https://vitejs.dev/)                                                                                                                                                                                            | `vite.config.ts`。開発サーバ・本番ビルドを実行                                                                                          |
| フロントエンド              | [React](https://react.dev/) + TypeScript（`strict`モード）                                                                                                                                                             | `src/`配下。`.js`/`.jsx`は使用しない                                                                                                    |
| UIコンポーネント            | [Tailwind CSS](https://tailwindcss.com/) + [shadcn/ui](https://ui.shadcn.com/)                                                                                                                                         | `src/index.css`（Tailwindの読み込み）、`src/components/ui/`（shadcn/ui CLIで取り込んだコンポーネント。`components.json`が設定ファイル） |
| データ取得・状態管理        | [TanStack Query](https://tanstack.com/query)                                                                                                                                                                           | `src/main.tsx`（`QueryClientProvider`）。チャットのストリーミング送受信自体は`src/hooks/useChat.ts`が`fetch`で直接扱う                  |
| Markdown/コードブロック表示 | [react-markdown](https://github.com/remarkjs/react-markdown) + [remark-gfm](https://github.com/remarkjs/remark-gfm) + [react-syntax-highlighter](https://github.com/react-syntax-highlighter/react-syntax-highlighter) | `src/components/markdown/Markdown.tsx`。回答本文のMarkdown・コードブロックのシンタックスハイライトを表示                                |
| バックエンドAPI呼び出し     | 素の`fetch`                                                                                                                                                                                                            | `src/lib/api.ts`。`api/main.py`（FastAPI）のエンドポイントを呼び出す薄いラッパー（`/api/chat`のSSEストリーミング含む）                  |
| 静的解析                    | [ESLint](https://eslint.org/)（Flat Config） + [Prettier](https://prettier.io/)                                                                                                                                        | `eslint.config.ts` / `.prettierrc.json`                                                                                                 |
| 単体テスト                  | [Vitest](https://vitest.dev/) + [React Testing Library](https://testing-library.com/react)                                                                                                                             | `vite.config.ts`の`test`設定、`src/App.test.tsx`、`src/components/chat/*.test.tsx`、`src/lib/api.test.ts`                               |
| E2Eテスト                   | [Playwright](https://playwright.dev/)（ローカル実行）                                                                                                                                                                  | `playwright.config.ts`、`e2e/app.spec.ts`、`e2e/chat.spec.ts`（`page.route`でバックエンドをモックしチャット送信フローを検証）           |

## チャットUI（Step3）

- `src/hooks/useChat.ts`: 会話状態の管理。マウント時に `POST /api/conversations/new` でスレッドIDを発行し、
  `POST /api/chat` のSSEストリーミング応答を逐次反映する。回答完了後は `POST /api/conversations/save` で
  会話ログを自動保存する（Streamlit版のデフォルト挙動と同様）
- `src/components/chat/`: メッセージ一覧（`ChatMessageList`）・1件分の表示（`ChatMessageItem`）・
  参照元表示（`SourceList`、`api/main.py`が返す`sources`をStreamlit版の`_render_answer_provenance`相当に表示）・
  入力欄（`MessageInput`）
- `src/lib/api.ts`の`streamChat()`: `data: <json>\n\n`形式のSSEレスポンスをチャンク境界に依存せず
  パースし、`content`/`sources`/`error`/`done`イベントを非同期にyieldする

## ローカル開発サーバーの起動手順

```bash
cd frontend
npm install
npm run dev
```

`http://localhost:5173` でアクセスできます。デフォルトでは `http://localhost:8000` のFastAPI
バックエンドにAPIリクエストを送ります。別のURLを使う場合は `.env.example` を参考に `.env.local` を
作成し `VITE_API_BASE_URL` を上書きしてください。

### FastAPIバックエンド（`api/main.py`）を併用する場合

チャット画面は起動時に `POST /api/conversations/new` で会話スレッドを発行するため、実際に
チャットを送受信するにはリポジトリルートで別途FastAPIサーバーを起動しておく必要があります
（`api/main.py`自体はStep3では変更していません）。バックエンドが起動していない場合は画面上部に
エラーメッセージが表示され、入力欄が無効化されます。

```bash
# リポジトリルートで（frontendディレクトリではない）
source .venv/bin/activate
uvicorn api.main:app --reload
```

CORS設定（`api/main.py`）はVite開発サーバのデフォルトポート（`localhost:5173`）のみを許可しています。

## コマンド一覧

```bash
npm run dev          # 開発サーバー起動
npm run build         # 型チェック（tsc -b） + 本番ビルド
npm run preview       # ビルド済み成果物のプレビュー
npm run lint          # ESLintによる静的解析
npm run format         # Prettierでコード整形
npm run format:check   # Prettierの整形チェック（書き換えなし）
npm run test           # Vitestで単体テストを実行
npm run test:watch     # Vitestをウォッチモードで実行
npm run test:e2e       # Playwrightでe2eテストを実行（devサーバーを自動起動）
```

## shadcn/uiコンポーネントの追加

```bash
npx shadcn@latest add <component-name>
```

`src/components/ui/` にコンポーネントのソースがコピーされます（npm依存としてではなく、コード
そのものを取り込む方式）。
