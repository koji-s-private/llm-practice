import { expect, test } from '@playwright/test'

test('チャット画面の雛形（見出し・入力欄）が表示される', async ({ page }) => {
  // /api/conversations/new の応答が無いと入力欄が有効化されないため、E2E実行環境に
  // バックエンドが無くても検証できるよう最小限のモック応答を用意する。
  await page.route('**/api/conversations/new', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ thread_id: 'e2e-thread' }),
    }),
  )

  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Doclore' })).toBeVisible()
  await expect(page.getByPlaceholder('資料について気になることを聞いてみましょう')).toBeVisible()
})
