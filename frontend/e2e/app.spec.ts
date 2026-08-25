import { expect, test } from '@playwright/test'

test('チャット画面が表示される', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Doclore' })).toBeVisible()
  await expect(page.getByPlaceholder('質問を入力してください（Shift+Enterで改行）')).toBeVisible()
})
