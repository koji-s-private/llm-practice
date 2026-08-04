# フロントエンド技術方針

## 背景

現在のフロントエンドは [Streamlit](https://streamlit.io/)（Python）で実装されている（`app.py`）。
個人用のローカルRAGチャットツールとして構築されてきたが、複数ファイルの一括アップロードや、会話
（チャット）の管理操作（一覧・切り替え・削除など）といった、ChatGPT等に近い操作性を今後実現していく
方針である。Streamlitの標準コンポーネントの範囲では、こうした操作の一部は実現が難しい、または
UX上の妥協が必要になる。

技術選定にあたっては、以下の制約を満たす必要がある（詳細は[AGENTS.md](../AGENTS.md)参照）。

- **無料制約**: 課金が発生する可能性がある操作は一切行わない。ホスティングも含め、無料枠・無料ツールの
  みで完結させる
- **既存資産との整合**: バックエンドロジック（`ingest.py` / `rag_chain.py` / `memory.py`）はPythonで
  LangChain・Chroma・Ollamaと密結合しており、ローカルで無料動作する構成を維持する
- **実装レビュー体制**: 本プロジェクトの実装はAIエージェントチームが担い、オーナー（koji）が最終
  レビュー・マージを行う。オーナー自身がReact/TypeScriptの実務知見を持つため、フロントエンドの実装
  品質はオーナー自身でもレビューできる

## 採用技術

**TypeScript製フロントエンド（React + Vite） + Python（FastAPI）によるバックエンドAPI化を採用する。**

JavaScript（無型）は採用しない。フロントエンドのコードはすべてTypeScriptで記述し、`.js`/`.jsx`
ファイルは作成しない（設定ファイル等でJS形式が事実上標準のものを除く）。`tsconfig.json`は`strict`
モードを有効にし、`.js`/`.jsx`の混入をCI/実装レビューで検出できるようにする。

- Streamlit継続や他のPython製UIフレームワーク（NiceGUI/Reflex等）と比較して、複数ファイル一括
  アップロードや会話管理操作を含む複雑なUI/UXを実現できる自由度が最も高い
- ホスティングは引き続きローカル完結とし（フロントエンドの静的ビルドをFastAPIから配信、または開発時は
  `vite dev`のローカルサーバを利用）、無料制約を満たす
- 既存のStreamlit版（`app.py`）は、移行完了が確認できるまでフォールバックとして残す

## UI品質担保の方針

自動化されたプロセスで不具合の発生を完全に0にすることは原理的に不可能なため、以下の複数の防御層で
リスクを実務上ほぼゼロに近づける。

1. **型安全性**: TypeScript `strict`モードを必須にし、`tsc --noEmit`をテスト工程に組み込む
2. **コンポーネント単体テスト**: Vitest + React Testing Libraryで主要コンポーネントの単体テストを
   必須化する
3. **E2E・ビジュアルリグレッションテスト**: Playwright（ローカル実行・無料）でチャット送信・ファイル
   アップロード等の主要フローをE2Eテストし、スクリーンショット比較で意図しない見た目の崩れを検知する。
   ベースライン画像との差分が出た場合は、必ず目視確認するまでマージしない
4. **段階的移行**: 一気に全機能を切り替えず、機能単位で移行し、都度オーナーが実機確認する（下記の
   移行計画を参照）
5. **オーナーによる最終レビュー**: 自動レビューに加え、オーナー自身がReact知見を活かして重要なUI変更を
   確認する

## 技術構成・採用ライブラリ

すべて無料・OSSで構成し、追加の有料契約は発生しない。

| 用途 | ライブラリ | 備考 |
|---|---|---|
| ビルドツール | [Vite](https://vitejs.dev/) | 高速なローカル開発サーバ・ビルド |
| フロントエンド | [React](https://react.dev/) + TypeScript | オーナーの既存知見を活かせる |
| UIコンポーネント | [Tailwind CSS](https://tailwindcss.com/) + [shadcn/ui](https://ui.shadcn.com/) | shadcn/uiはコピー&ペースト方式でコンポーネントを取り込むため、npm依存が肥大化しにくい。Radix UIベースでアクセシビリティも担保 |
| データ取得・状態管理 | [TanStack Query](https://tanstack.com/query) | API通信のキャッシュ・再試行等を簡潔に扱える |
| 複数ファイル一括アップロード | [react-dropzone](https://react-dropzone.js.org/) | ドラッグ&ドロップ・複数選択に対応 |
| Markdown/コードブロック表示 | [react-markdown](https://github.com/remarkjs/react-markdown) + [react-syntax-highlighter](https://github.com/react-syntax-highlighter/react-syntax-highlighter) | 現行Streamlit版の`st.markdown`相当の表示を再現 |
| バックエンドAPI | [FastAPI](https://fastapi.tiangolo.com/) | 既存のPythonロジック（`ingest.py`等）をそのまま呼び出せる |
| コンポーネント単体テスト | [Vitest](https://vitest.dev/) + [React Testing Library](https://testing-library.com/react) | Vite環境と親和性が高い |
| E2E・ビジュアルリグレッションテスト | [Playwright](https://playwright.dev/) | ローカル実行のみで完結し無料。スクリーンショット比較機能を利用 |

## 移行計画

機能単位でIssueを分割し、段階的に移行する。

1. **API層の切り出し**: `ingest.py` / `rag_chain.py` / `memory.py` をFastAPI経由で呼び出せる
   エンドポイントとして切り出す。ストリーミング応答はFastAPIの`StreamingResponse`（Server-Sent Events）
   で実装する
2. **フロントエンド基盤構築**: Vite + React + TypeScriptの雛形を作成し、Vitest/Playwrightのテスト環境・
   ESLint/Prettierの静的解析環境を整備する
3. **チャットUI実装**: メッセージ送受信・ストリーミング表示・Markdown/コードブロック表示を実装する
4. **ファイル管理UI実装**: 複数ファイルの一括アップロード、アップロード済みファイルの一覧・削除UIを
   実装する
5. **会話管理UI実装**: 会話の一覧・新規作成・削除・タイトル編集などの管理操作UIを実装する
6. **Streamlit版の廃止**: 機能パリティが確認でき次第、`app.py`と関連ファイルを削除し、READMEの起動
   手順を更新する
