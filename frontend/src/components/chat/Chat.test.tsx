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
})
