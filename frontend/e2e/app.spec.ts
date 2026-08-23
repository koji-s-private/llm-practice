import { expect, test } from '@playwright/test'

test('雛形画面が表示される', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Doclore' })).toBeVisible()
  await expect(page.getByRole('button', { name: '再確認する' })).toBeVisible()
})
