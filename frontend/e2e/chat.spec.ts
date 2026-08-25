import { expect, test } from '@playwright/test'

// FastAPIバックエンド（Ollama含む）の起動を前提にせず、page.route()でAPIレスポンスをモックして
// フロントエンドのチャット送信フロー（送信→ストリーミング表示→参照元表示）のみを検証する。
async function mockNewThread(page: import('@playwright/test').Page, threadId = 'e2e-thread-id') {
  await page.route('**/api/conversations/new', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ thread_id: threadId }),
    })
  })
}

function sseBody(events: Record<string, unknown>[]): string {
  return events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('')
}

test.describe('チャット送信フロー', () => {
  test('メッセージを送信するとストリーミング応答と参照元が表示される', async ({ page }) => {
    await mockNewThread(page)
    await page.route('**/api/chat', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: sseBody([
          { content: 'こんにちは、' },
          { content: 'ご質問ありがとうございます。' },
          { sources: [{ label: 'guide.pdf p.1', snippet: 'ドキュメントの抜粋テキスト' }] },
          { done: true },
        ]),
      })
    })

    await page.goto('/')
    const input = page.getByPlaceholder('質問を入力してください（Shift+Enterで改行）')
    await input.fill('こんにちは')
    await input.press('Enter')

    await expect(page.getByText('こんにちは', { exact: true })).toBeVisible()
    await expect(page.getByText('こんにちは、ご質問ありがとうございます。')).toBeVisible()
    await expect(page.getByText('🔍 ドキュメントに基づく回答')).toBeVisible()

    await page.getByText('参照した箇所を見る（1件）').click()
    await expect(page.getByText('guide.pdf p.1')).toBeVisible()
    await expect(page.getByText('ドキュメントの抜粋テキスト')).toBeVisible()

    // 送信後は入力欄がクリアされる
    await expect(input).toHaveValue('')
  })

  test('sourcesが空のときは一般知識による回答である旨が表示される', async ({ page }) => {
    await mockNewThread(page)
    await page.route('**/api/chat', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: sseBody([{ content: '一般的な回答です。' }, { sources: [] }, { done: true }]),
      })
    })

    await page.goto('/')
    const input = page.getByPlaceholder('質問を入力してください（Shift+Enterで改行）')
    await input.fill('一般的な質問')
    await input.press('Enter')

    await expect(page.getByText('一般的な回答です。')).toBeVisible()
    await expect(
      page.getByText('🧠 一般知識による回答（ドキュメントに該当情報なし）'),
    ).toBeVisible()
  })

  test('サーバーがエラーイベントを返した場合エラーメッセージが表示される', async ({ page }) => {
    await mockNewThread(page)
    await page.route('**/api/chat', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: sseBody([{ error: 'モデル呼び出しに失敗しました' }]),
      })
    })

    await page.goto('/')
    const input = page.getByPlaceholder('質問を入力してください（Shift+Enterで改行）')
    await input.fill('エラーになる質問')
    await input.press('Enter')

    await expect(page.getByText('エラー: モデル呼び出しに失敗しました')).toBeVisible()
  })

  test('Shift+Enterでは送信されず改行のみ入力される', async ({ page }) => {
    await mockNewThread(page)
    let chatRequestCount = 0
    await page.route('**/api/chat', async (route) => {
      chatRequestCount += 1
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: sseBody([{ done: true }]),
      })
    })

    await page.goto('/')
    const input = page.getByPlaceholder('質問を入力してください（Shift+Enterで改行）')
    await input.fill('1行目')
    await input.press('Shift+Enter')
    await input.type('2行目')

    expect(chatRequestCount).toBe(0)
    await expect(input).toHaveValue('1行目\n2行目')
  })
})
