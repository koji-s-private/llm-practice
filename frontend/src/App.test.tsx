import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

function sseResponse(events: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder()
      for (const event of events) {
        controller.enqueue(encoder.encode(event))
      }
      controller.close()
    },
  })
  return new Response(body, { status: 200 })
}

function mockFetch() {
  return vi.fn((input: RequestInfo | URL) => {
    const url = String(input)
    if (url.endsWith('/api/conversations/new')) {
      return Promise.resolve(new Response(JSON.stringify({ thread_id: 'thread-123' })))
    }
    if (url.endsWith('/api/chat')) {
      return Promise.resolve(
        sseResponse([
          'data: {"content": "回答"}\n\n',
          'data: {"content": "です"}\n\n',
          'data: {"sources": [{"label": "doc.txt", "snippet": "抜粋テキスト"}]}\n\n',
          'data: {"done": true}\n\n',
        ]),
      )
    }
    if (url.endsWith('/api/conversations/save')) {
      return Promise.resolve(new Response(JSON.stringify({ path: 'dummy' })))
    }
    return Promise.reject(new Error(`unexpected fetch: ${url}`))
  })
}

describe('App', () => {
  it('見出し「Doclore」を表示する', () => {
    vi.stubGlobal('fetch', mockFetch())
    render(<App />)
    expect(screen.getByRole('heading', { name: 'Doclore' })).toBeInTheDocument()
  })

  it('メッセージを送信すると回答がストリーミング表示され、参照元も表示される', async () => {
    vi.stubGlobal('fetch', mockFetch())
    const user = userEvent.setup()
    render(<App />)

    const textbox = await screen.findByPlaceholderText('資料について気になることを聞いてみましょう')
    await waitFor(() => expect(textbox).toBeEnabled())

    await user.type(textbox, '質問です{Enter}')

    expect(await screen.findByText('質問です')).toBeInTheDocument()
    expect(await screen.findByText('回答です')).toBeInTheDocument()

    const toggle = await screen.findByRole('button', { name: /参照した箇所を見る（1件）/ })
    await user.click(toggle)
    expect(await screen.findByText(/doc\.txt/)).toBeInTheDocument()
  })
})
