import { expect, test } from '@playwright/test'

// Ollamaを起動していないE2E実行環境でも検証できるよう、api/main.pyへの通信を
// page.routeでモックし、SSEストリーミング・参照元表示までのフローを検証する。
test('メッセージを送信すると回答がストリーミング表示され、参照元も確認できる', async ({ page }) => {
  await page.route('**/api/conversations/new', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ thread_id: 'e2e-thread' }),
    }),
  )
  await page.route('**/api/chat', (route) =>
    route.fulfill({
      contentType: 'text/event-stream',
      body: [
        'data: {"content": "回答"}\n\n',
        'data: {"content": "です"}\n\n',
        'data: {"sources": [{"label": "doc.txt", "snippet": "抜粋テキスト"}]}\n\n',
        'data: {"done": true}\n\n',
      ].join(''),
    }),
  )
  await page.route('**/api/conversations/save', (route) =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ path: 'dummy' }) }),
  )

  await page.goto('/')

  const textbox = page.getByPlaceholder('資料について気になることを聞いてみましょう')
  await textbox.fill('質問です')
  await textbox.press('Enter')

  await expect(page.getByText('質問です')).toBeVisible()
  await expect(page.getByText('回答です')).toBeVisible()

  await page.getByRole('button', { name: /参照した箇所を見る（1件）/ }).click()
  await expect(page.getByText('doc.txt', { exact: false })).toBeVisible()
  await expect(page.getByText('抜粋テキスト')).toBeVisible()
})
