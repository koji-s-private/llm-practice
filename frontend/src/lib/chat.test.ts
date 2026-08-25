import { afterEach, describe, expect, it, vi } from 'vitest'
import { type ChatStreamEvent, createNewThread, streamChat } from '@/lib/chat'

function sseResponse(chunks: string[], init?: ResponseInit): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk))
      }
      controller.close()
    },
  })
  return new Response(body, init)
}

async function collect(events: AsyncGenerator<ChatStreamEvent>): Promise<ChatStreamEvent[]> {
  const result: ChatStreamEvent[] = []
  for await (const event of events) {
    result.push(event)
  }
  return result
}

describe('streamChat', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('1チャンク1イベントのcontent/sources/doneを順にパースする', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(
          sseResponse([
            'data: {"content": "こんにちは"}\n\n',
            'data: {"sources": [{"label": "doc.pdf", "snippet": "抜粋"}]}\n\n',
            'data: {"done": true}\n\n',
          ]),
        ),
      ),
    )

    const events = await collect(streamChat({ threadId: 't1', message: 'こんにちは', history: [] }))

    expect(events).toEqual([
      { type: 'content', content: 'こんにちは' },
      { type: 'sources', sources: [{ label: 'doc.pdf', snippet: '抜粋' }] },
      { type: 'done' },
    ])
  })

  it('複数イベントが1チャンクに含まれていても分離してパースする', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(
          sseResponse([
            'data: {"content": "A"}\n\ndata: {"content": "B"}\n\ndata: {"done": true}\n\n',
          ]),
        ),
      ),
    )

    const events = await collect(streamChat({ threadId: 't1', message: 'm', history: [] }))

    expect(events).toEqual([
      { type: 'content', content: 'A' },
      { type: 'content', content: 'B' },
      { type: 'done' },
    ])
  })

  it('区切り文字(\\n\\n)がチャンク境界をまたいでも正しくパースする', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(sseResponse(['data: {"content": "分割"}\n', '\ndata: {"done": true}\n\n'])),
      ),
    )

    const events = await collect(streamChat({ threadId: 't1', message: 'm', history: [] }))

    expect(events).toEqual([{ type: 'content', content: '分割' }, { type: 'done' }])
  })

  it('JSONの内容自体が複数チャンクに分割されても正しくパースする', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(
          sseResponse(['data: {"cont', 'ent": "分割データ"}\n\n', 'data: {"done": true}\n\n']),
        ),
      ),
    )

    const events = await collect(streamChat({ threadId: 't1', message: 'm', history: [] }))

    expect(events).toEqual([{ type: 'content', content: '分割データ' }, { type: 'done' }])
  })

  it('errorイベントをパースする', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(sseResponse(['data: {"error": "モデル呼び出しに失敗しました"}\n\n'])),
      ),
    )

    const events = await collect(streamChat({ threadId: 't1', message: 'm', history: [] }))

    expect(events).toEqual([{ type: 'error', error: 'モデル呼び出しに失敗しました' }])
  })

  it('data:行を含まないイベントやdone:falseは無視する', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(
          sseResponse([
            ': keep-alive comment\n\n',
            'data: {"done": false}\n\n',
            'data: {"content": "本文"}\n\n',
          ]),
        ),
      ),
    )

    const events = await collect(streamChat({ threadId: 't1', message: 'm', history: [] }))

    expect(events).toEqual([{ type: 'content', content: '本文' }])
  })

  it('レスポンスがエラーステータスのとき例外を投げる', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(new Response(null, { status: 500 }))),
    )

    await expect(
      collect(streamChat({ threadId: 't1', message: 'm', history: [] })),
    ).rejects.toThrow('チャットの送信に失敗しました (status: 500)')
  })
})

describe('createNewThread', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('成功時はthread_idを返す', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(new Response(JSON.stringify({ thread_id: 'abc123' })))),
    )

    const result = await createNewThread()

    expect(result).toEqual({ thread_id: 'abc123' })
  })

  it('失敗時は例外を投げる', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(new Response(null, { status: 503 }))),
    )

    await expect(createNewThread()).rejects.toThrow('新しい会話の作成に失敗しました (status: 503)')
  })
})
