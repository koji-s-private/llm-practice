import { expect, test } from '@playwright/test'

// FastAPIバックエンドの起動を前提にせず、page.route()でAPIレスポンスをモックして
// ファイル管理UIのフロントエンド側の挙動（アップロード→一覧反映→削除）のみを検証する。
async function mockNewThread(page: import('@playwright/test').Page) {
  await page.route('**/api/conversations/new', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ thread_id: 'e2e-thread-id' }),
    })
  })
}

test.describe('ファイル管理フロー', () => {
  test('複数ファイルアップロード→一覧に反映→削除', async ({ page }) => {
    await mockNewThread(page)

    let indexedFiles: { name: string; chunk_count: number }[] = []

    await page.route('**/api/files', async (route) => {
      if (route.request().method() !== 'GET') {
        await route.fallback()
        return
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ files: indexedFiles }),
      })
    })

    await page.route('**/api/files/upload', async (route) => {
      indexedFiles = [
        { name: 'report.pdf', chunk_count: 3 },
        { name: 'notes.txt', chunk_count: 1 },
      ]
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          uploaded: [
            { original_name: 'report.pdf', saved_name: 'report.pdf', renamed: false },
            { original_name: 'notes.txt', saved_name: 'notes.txt', renamed: false },
          ],
        }),
      })
    })

    await page.route('**/api/files/report.pdf', async (route) => {
      indexedFiles = indexedFiles.filter((file) => file.name !== 'report.pdf')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ name: 'report.pdf' }),
      })
    })

    await page.goto('/')
    await page.getByRole('button', { name: '📁 ファイル管理' }).click()
    await expect(page.getByText('インデックス済みのファイルはまだありません。')).toBeVisible()

    await page.locator('input[type="file"]').setInputFiles([
      { name: 'report.pdf', mimeType: 'application/pdf', buffer: Buffer.from('dummy pdf content') },
      { name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('dummy text content') },
    ])

    const reportItem = page.locator('li', { hasText: 'report.pdf' })
    const notesItem = page.locator('li', { hasText: 'notes.txt' })
    await expect(reportItem).toBeVisible()
    await expect(notesItem).toBeVisible()

    await page.getByRole('button', { name: 'report.pdf を削除' }).click()
    await expect(page.getByText(/この操作は取り消せません/)).toBeVisible()
    await page.getByRole('button', { name: '削除する' }).click()

    await expect(reportItem).toHaveCount(0)
    await expect(notesItem).toBeVisible()
  })
})
