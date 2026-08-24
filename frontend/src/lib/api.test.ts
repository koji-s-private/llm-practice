import { describe, expect, it, vi } from 'vitest'
import { streamChat } from './api'

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

describe('streamChat', () => {
  it('SSEイベントをcontent/sources/doneの順にyieldする', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(
          sseResponse([
            'data: {"content": "こん"}\n\n',
            'data: {"content": "にちは"}\n\n',
            'data: {"sources": [{"label": "a.txt", "snippet": "抜粋"}]}\n\n',
            'data: {"done": true}\n\n',
          ]),
        ),
      ),
    )

    const events = []
    for await (const event of streamChat({
      threadId: 'thread1',
      message: 'こんにちは',
      history: [],
    })) {
      events.push(event)
    }

    expect(events).toEqual([
      { type: 'content', content: 'こん' },
      { type: 'content', content: 'にちは' },
      { type: 'sources', sources: [{ label: 'a.txt', snippet: '抜粋' }] },
      { type: 'done' },
    ])
  })

  it('複数イベントが1チャンクにまとまっていても正しく分割する', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(sseResponse(['data: {"content": "a"}\n\ndata: {"content": "b"}\n\n'])),
      ),
    )

    const events = []
    for await (const event of streamChat({ threadId: 'thread1', message: 'hi', history: [] })) {
      events.push(event)
    }

    expect(events).toEqual([
      { type: 'content', content: 'a' },
      { type: 'content', content: 'b' },
    ])
  })

  it('errorイベントをyieldする', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(sseResponse(['data: {"error": "失敗しました"}\n\n']))),
    )

    const events = []
    for await (const event of streamChat({ threadId: 'thread1', message: 'hi', history: [] })) {
      events.push(event)
    }

    expect(events).toEqual([{ type: 'error', error: '失敗しました' }])
  })

  it('レスポンスがエラーの場合は例外を投げる', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(new Response(null, { status: 500 }))),
    )

    const iterate = async () => {
      const iterator = streamChat({ threadId: 'thread1', message: 'hi', history: [] })
      for await (const _event of iterator) {
        void _event
      }
    }

    await expect(iterate()).rejects.toThrow(/チャットの送信に失敗しました/)
  })
})
