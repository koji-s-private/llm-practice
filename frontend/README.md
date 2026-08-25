# frontend

Doclore（ローカルRAGチャットアプリ）の新フロントエンド（Vite + React + TypeScript）です。
既存のStreamlit版（ルートの `app.py`）を置き換える移行計画（[docs/frontend-tech-policy.md](../docs/frontend-tech-policy.md)）
のStep2で基盤を構築し、Step3で `POST /api/chat` を呼び出すチャットUI（メッセージ入力・
SSEストリーミング表示・Markdown/コードブロック表示・参照元表示）を実装しました。
会話の一覧・切り替え・削除等の会話管理UIは未実装（Step5以降）です。

## 使用技術

| 用途                                   | ライブラリ                                                                                                                           | このディレクトリでの使用箇所                                                                                                                                                                             |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ビルドツール                           | [Vite](https://vitejs.dev/)                                                                                                          | `vite.config.ts`。開発サーバ・本番ビルドを実行                                                                                                                                                           |
| フロントエンド                         | [React](https://react.dev/) + TypeScript（`strict`モード）                                                                           | `src/`配下。`.js`/`.jsx`は使用しない                                                                                                                                                                     |
| UIコンポーネント                       | [Tailwind CSS](https://tailwindcss.com/) + [shadcn/ui](https://ui.shadcn.com/)                                                       | `src/index.css`（Tailwindの読み込み）、`src/components/ui/`（shadcn/ui CLIで取り込んだコンポーネント。`components.json`が設定ファイル）                                                                  |
| データ取得・状態管理                   | [TanStack Query](https://tanstack.com/query)                                                                                         | `src/main.tsx`（`QueryClientProvider`）、`src/components/chat/Chat.tsx`（`useQuery`による会話スレッド発行 `POST /api/conversations/new`。チャット送信自体はSSEストリーミングのため素の`useState`で管理） |
| バックエンドAPI呼び出し                | 素の`fetch`                                                                                                                          | `src/lib/api.ts` / `src/lib/chat.ts`。`api/main.py`（FastAPI）のエンドポイントを呼び出す薄いラッパー。`chat.ts`は`POST /api/chat`のSSEレスポンスを`fetch` + `ReadableStream`でパースする                 |
| Markdown表示                           | [react-markdown](https://github.com/remarkjs/react-markdown)                                                                         | `src/components/chat/MarkdownContent.tsx`。チャット回答本文（Markdown）を描画                                                                                                                            |
| コードブロックのシンタックスハイライト | [react-syntax-highlighter](https://github.com/react-syntax-highlighter/react-syntax-highlighter) + `@types/react-syntax-highlighter` | `src/components/chat/MarkdownContent.tsx`。react-markdownの`code`コンポーネントを差し替えてコードブロックを装飾                                                                                          |
| 静的解析                               | [ESLint](https://eslint.org/)（Flat Config） + [Prettier](https://prettier.io/)                                                      | `eslint.config.ts` / `.prettierrc.json`                                                                                                                                                                  |
| 単体テスト                             | [Vitest](https://vitest.dev/) + [React Testing Library](https://testing-library.com/react)                                           | `vite.config.ts`の`test`設定、`src/App.test.tsx`                                                                                                                                                         |
| E2Eテスト                              | [Playwright](https://playwright.dev/)（ローカル実行）                                                                                | `playwright.config.ts`、`e2e/app.spec.ts`                                                                                                                                                                |

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

チャット画面を使うには、リポジトリルートで別途FastAPIサーバーを起動しておく必要があります
（`api/main.py`自体は本Issueで変更していません）。

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
