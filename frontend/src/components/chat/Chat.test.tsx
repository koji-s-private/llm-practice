import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Chat } from '@/components/chat/Chat'

function sseResponse(events: Record<string, unknown>[]): Response {
  const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('')
  return new Response(body, { status: 200 })
}

function renderChat() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <Chat />
    </QueryClientProvider>,
  )
}

describe('Chat', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('sourcesイベントが一度も届かなくても、完了後に一般知識による回答である旨を表示する', async () => {
    const user = userEvent.setup()
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (url.toString().includes('/api/conversations/new')) {
          return Promise.resolve(new Response(JSON.stringify({ thread_id: 't1' })))
        }
        return Promise.resolve(sseResponse([{ content: '一般的な回答です。' }, { done: true }]))
      }),
    )

    renderChat()
    const input = await screen.findByPlaceholderText('質問を入力してください（Shift+Enterで改行）')
    await waitFor(() => expect(input).not.toBeDisabled())
    await user.type(input, '一般的な質問')
    await user.keyboard('{Enter}')

    expect(await screen.findByText('一般的な回答です。')).toBeInTheDocument()
    expect(
      await screen.findByText('🧠 一般知識による回答（ドキュメントに該当情報なし）'),
    ).toBeInTheDocument()
  })

  it('ストリーミング完了後に会話を保存し、会話スレッド一覧のキャッシュを無効化する', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      const target = url.toString()
      if (target.endsWith('/api/conversations/new') && init?.method === 'POST') {
        return Promise.resolve(new Response(JSON.stringify({ thread_id: 't1' })))
      }
      if (target.endsWith('/api/conversations/save') && init?.method === 'POST') {
        return Promise.resolve(
          new Response(JSON.stringify({ path: 'data/conversations/t1/1.json' })),
        )
      }
      if (
        target.endsWith('/api/conversations') &&
        (init === undefined || init.method === undefined)
      ) {
        return Promise.resolve(new Response(JSON.stringify({ threads: [] })))
      }
      return Promise.resolve(
        sseResponse([
          { content: '回答本文' },
          { sources: [{ label: 'doc.txt', snippet: '抜粋' }] },
          { done: true },
        ]),
      )
    })
    vi.stubGlobal('fetch', fetchMock)

    renderChat()
    const input = await screen.findByPlaceholderText('質問を入力してください（Shift+Enterで改行）')
    await waitFor(() => expect(input).not.toBeDisabled())

    // 会話パネルを開いた状態にして['threads']クエリをアクティブにしておき、
    // 保存成功時のinvalidateQueriesで再取得されることも合わせて検証する。
    await user.click(screen.getByRole('button', { name: '💬 会話' }))
    await screen.findByText('まだ保存された会話スレッドはありません。')

    await user.type(input, '質問文')
    await user.keyboard('{Enter}')

    expect(await screen.findByText('回答本文')).toBeInTheDocument()

    await waitFor(() => {
      const saveCall = fetchMock.mock.calls.find(
        ([url, init]) =>
          url.toString().endsWith('/api/conversations/save') && init?.method === 'POST',
      )
      expect(saveCall).toBeDefined()
      const body = JSON.parse((saveCall?.[1]?.body as string) ?? '{}')
      expect(body).toEqual({
        thread_id: 't1',
        question: '質問文',
        answer: '回答本文',
        is_fallback: false,
      })
    })

    await waitFor(() => {
      const threadsGetCalls = fetchMock.mock.calls.filter(
        ([url, init]) => url.toString().endsWith('/api/conversations') && init === undefined,
      )
      expect(threadsGetCalls.length).toBeGreaterThan(1)
    })
  })

  it('会話パネルから過去のスレッドを選択すると、その会話内容が表示される', async () => {
    const user = userEvent.setup()
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string, init?: RequestInit) => {
        const target = url.toString()
        if (target.endsWith('/api/conversations/new') && init?.method === 'POST') {
          return Promise.resolve(new Response(JSON.stringify({ thread_id: 't1' })))
        }
        if (target.endsWith('/api/conversations')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                threads: [
                  {
                    thread_id: 'past-1',
                    created_at: '2026-01-01T00:00:00',
                    first_question: '過去の質問',
                    count: 1,
                    title: null,
                  },
                ],
              }),
            ),
          )
        }
        if (target.endsWith('/api/conversations/past-1')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                thread_id: 'past-1',
                turns: [
                  {
                    question: '過去の質問',
                    answer: '過去の回答',
                    created_at: '2026-01-01T00:00:00',
                    sources: [],
                  },
                ],
              }),
            ),
          )
        }
        return Promise.resolve(sseResponse([{ content: '回答' }, { done: true }]))
      }),
    )

    renderChat()
    const input = await screen.findByPlaceholderText('質問を入力してください（Shift+Enterで改行）')
    await waitFor(() => expect(input).not.toBeDisabled())

    await user.click(screen.getByRole('button', { name: '💬 会話' }))
    await user.click(await screen.findByText(/過去の質問/))

    expect(await screen.findByText('過去の回答')).toBeInTheDocument()
  })

  it('「新しい会話」を押すと会話中のメッセージがクリアされ、新しいスレッドが発行される', async () => {
    const user = userEvent.setup()
    let newThreadCallCount = 0
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string, init?: RequestInit) => {
        const target = url.toString()
        if (target.endsWith('/api/conversations/new') && init?.method === 'POST') {
          newThreadCallCount += 1
          return Promise.resolve(
            new Response(JSON.stringify({ thread_id: `t${newThreadCallCount}` })),
          )
        }
        if (target.endsWith('/api/conversations')) {
          return Promise.resolve(new Response(JSON.stringify({ threads: [] })))
        }
        return Promise.resolve(sseResponse([{ content: '回答' }, { done: true }]))
      }),
    )

    renderChat()
    const input = await screen.findByPlaceholderText('質問を入力してください（Shift+Enterで改行）')
    await waitFor(() => expect(input).not.toBeDisabled())
    await user.type(input, '質問')
    await user.keyboard('{Enter}')
    expect(await screen.findByText('回答')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '💬 会話' }))
    await user.click(await screen.findByRole('button', { name: '🆕 新しい会話' }))

    expect(
      await screen.findByText('質問を入力するとここに会話が表示されます。'),
    ).toBeInTheDocument()
    await waitFor(() => expect(newThreadCallCount).toBe(2))
  })

  it('アクティブな会話スレッドを削除すると、自動的に新しい会話へ切り替わる', async () => {
    const user = userEvent.setup()
    let newThreadCallCount = 0
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string, init?: RequestInit) => {
        const target = url.toString()
        if (target.endsWith('/api/conversations/new') && init?.method === 'POST') {
          newThreadCallCount += 1
          return Promise.resolve(
            new Response(JSON.stringify({ thread_id: `t${newThreadCallCount}` })),
          )
        }
        if (target.endsWith('/api/conversations/t1') && init?.method === 'DELETE') {
          return Promise.resolve(new Response(null, { status: 200 }))
        }
        if (target.endsWith('/api/conversations')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                threads: [
                  {
                    thread_id: 't1',
                    created_at: '2026-01-01T00:00:00',
                    first_question: '自分自身のスレッド',
                    count: 1,
                    title: null,
                  },
                ],
              }),
            ),
          )
        }
        return Promise.resolve(sseResponse([{ content: '回答' }, { done: true }]))
      }),
    )

    renderChat()
    const input = await screen.findByPlaceholderText('質問を入力してください（Shift+Enterで改行）')
    await waitFor(() => expect(input).not.toBeDisabled())

    await user.click(screen.getByRole('button', { name: '💬 会話' }))
    await screen.findByText(/自分自身のスレッド/)
    await user.click(screen.getByRole('button', { name: 't1 を削除' }))
    await user.click(screen.getByRole('button', { name: '削除する' }))

    await waitFor(() => expect(newThreadCallCount).toBe(2))
    await waitFor(() =>
      expect(screen.getByText('質問を入力するとここに会話が表示されます。')).toBeInTheDocument(),
    )
  })
})
