import { defineConfig, devices } from '@playwright/test'

// FastAPIバックエンドが未起動でもE2Eテストが実行できるよう、`fetch`失敗時のエラー表示（App.tsx）まで
// 含めて雛形画面が表示されることのみを検証する。バックエンド疎通を伴うE2Eテストは実画面実装（Step3以降）で追加する。
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  reporter: 'html',
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
  },
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
})
