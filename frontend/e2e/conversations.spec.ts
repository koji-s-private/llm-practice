import { expect, test } from '@playwright/test'

// FastAPIバックエンドの起動を前提にせず、page.route()でAPIレスポンスをモックして
// 会話管理UI（一覧表示→スレッド切替→タイトル編集→削除、および新規会話のストリーミング完了後の
// 保存→一覧反映）のフロントエンド側の挙動のみを検証する。
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

test.describe('会話管理フロー', () => {
  test('一覧表示→スレッド切替→タイトル編集→削除', async ({ page }) => {
    await mockNewThread(page)
    await page.route('**/api/chat', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: sseBody([{ content: '回答' }, { done: true }]),
      })
    })

    let title: string | null = null
    let threadExists = true

    await page.route('**/api/conversations', async (route) => {
      if (route.request().method() !== 'GET') {
        await route.fallback()
        return
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          threads: threadExists
            ? [
                {
                  thread_id: 'past-thread',
                  created_at: '2026-01-01T09:00:00',
                  first_question: '経費精算について教えてください',
                  count: 1,
                  title,
                },
              ]
            : [],
        }),
      })
    })

    await page.route('**/api/conversations/past-thread', async (route) => {
      if (route.request().method() === 'DELETE') {
        threadExists = false
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ thread_id: 'past-thread' }),
        })
        return
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          thread_id: 'past-thread',
          turns: [
            {
              question: '経費精算について教えてください',
              answer: '経費精算は月末締めで申請してください。',
              created_at: '2026-01-01T09:00:00',
              sources: [],
            },
          ],
        }),
      })
    })

    await page.route('**/api/conversations/past-thread/title', async (route) => {
      if (route.request().method() === 'PUT') {
        const body = route.request().postDataJSON() as { title: string }
        title = body.title.trim() || null
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ thread_id: 'past-thread', title }),
        })
        return
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ thread_id: 'past-thread', title }),
      })
    })

    await page.goto('/')
    const input = page.getByPlaceholder('質問を入力してください（Shift+Enterで改行）')
    await expect(input).toBeEnabled()

    // 一覧表示→過去スレッドへの切り替え
    await page.getByRole('button', { name: '💬 会話' }).click()
    await expect(page.getByText(/経費精算について教えてください/)).toBeVisible()
    await page.getByText(/経費精算について教えてください/).click()
    await expect(page.getByText('経費精算は月末締めで申請してください。')).toBeVisible()

    // タイトル編集
    await page.getByRole('button', { name: '✏️ タイトル編集' }).click()
    await page.getByPlaceholder('例: 経費精算の質問').fill('経費精算スレッド')
    await page.getByRole('button', { name: '💾 保存' }).click()
    await expect(page.getByText(/📌 経費精算スレッド/)).toBeVisible()

    // 2段階確認削除
    await page.getByRole('button', { name: 'past-thread を削除' }).click()
    await expect(page.getByText(/この操作は取り消せません/)).toBeVisible()
    await page.getByRole('button', { name: '削除する' }).click()
    await expect(page.getByText('まだ保存された会話スレッドはありません。')).toBeVisible()
  })

  test('新規会話作成→やり取り→一覧に反映→タイトル編集→削除', async ({ page }) => {
    const threadId = 'new-thread-id'
    await mockNewThread(page, threadId)
    await page.route('**/api/chat', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: sseBody([
          { content: '経費精算は月末締めで申請してください。' },
          { sources: [{ label: 'expense.md', snippet: '経費精算の抜粋' }] },
          { done: true },
        ]),
      })
    })

    let title: string | null = null
    let threadExists = false
    const saveRequests: unknown[] = []

    await page.route('**/api/conversations/save', async (route) => {
      const body = route.request().postDataJSON() as {
        thread_id: string
        question: string
        answer: string
        is_fallback: boolean
      }
      saveRequests.push(body)
      threadExists = true
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: `data/conversations/${threadId}/1.json` }),
      })
    })

    await page.route('**/api/conversations', async (route) => {
      if (route.request().method() !== 'GET') {
        await route.fallback()
        return
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          threads: threadExists
            ? [
                {
                  thread_id: threadId,
                  created_at: '2026-01-01T09:00:00',
                  first_question: '経費精算について教えてください',
                  count: 1,
                  title,
                },
              ]
            : [],
        }),
      })
    })

    await page.route(`**/api/conversations/${threadId}/title`, async (route) => {
      if (route.request().method() === 'PUT') {
        const body = route.request().postDataJSON() as { title: string }
        title = body.title.trim() || null
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ thread_id: threadId, title }),
        })
        return
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ thread_id: threadId, title }),
      })
    })

    await page.route(`**/api/conversations/${threadId}`, async (route) => {
      if (route.request().method() === 'DELETE') {
        threadExists = false
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ thread_id: threadId }),
        })
        return
      }
      await route.fallback()
    })

    await page.goto('/')
    const input = page.getByPlaceholder('質問を入力してください（Shift+Enterで改行）')
    await expect(input).toBeEnabled()

    // 会話パネルを開いた状態で待機し、保存完了後のinvalidateQueriesによる
    // 一覧の自動更新（開きっぱなしでも反映されること）まで検証する。
    await page.getByRole('button', { name: '💬 会話' }).click()
    await expect(page.getByText('まだ保存された会話スレッドはありません。')).toBeVisible()

    await input.fill('経費精算について教えてください')
    await input.press('Enter')
    await expect(page.getByText('経費精算は月末締めで申請してください。')).toBeVisible()

    // ストリーミング完了後にPOST /api/conversations/saveが呼ばれ、一覧に新規スレッドが反映されること
    await expect(page.getByText(/経費精算について教えてください/)).toBeVisible()
    expect(saveRequests).toEqual([
      {
        thread_id: threadId,
        question: '経費精算について教えてください',
        answer: '経費精算は月末締めで申請してください。',
        is_fallback: false,
      },
    ])

    // タイトル編集
    await page.getByRole('button', { name: '✏️ タイトル編集' }).click()
    await page.getByPlaceholder('例: 経費精算の質問').fill('経費精算スレッド')
    await page.getByRole('button', { name: '💾 保存' }).click()
    await expect(page.getByText(/📌 経費精算スレッド/)).toBeVisible()

    // 2段階確認削除
    await page.getByRole('button', { name: `${threadId} を削除` }).click()
    await expect(page.getByText(/この操作は取り消せません/)).toBeVisible()
    await page.getByRole('button', { name: '削除する' }).click()
    await expect(page.getByText('まだ保存された会話スレッドはありません。')).toBeVisible()
  })
})
