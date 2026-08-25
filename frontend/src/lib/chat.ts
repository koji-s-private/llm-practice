// api/main.py（FastAPI）のチャット関連エンドポイント（POST /api/chat, POST /api/conversations/new）
// を呼び出す薄いクライアント。SSEのパースはfetch + ReadableStreamで行う
// （POSTボディが必要なためEventSourceは使えない）。
import { API_BASE_URL } from '@/lib/api'

export type ChatRole = 'user' | 'assistant'

export interface ChatMessage {
  role: ChatRole
  content: string
}

export interface ChatSource {
  label: string
  snippet: string
}

export interface NewThreadResponse {
  thread_id: string
}

export async function createNewThread(): Promise<NewThreadResponse> {
  const response = await fetch(`${API_BASE_URL}/api/conversations/new`, { method: 'POST' })
  if (!response.ok) {
    throw new Error(`新しい会話の作成に失敗しました (status: ${response.status})`)
  }
  return response.json() as Promise<NewThreadResponse>
}

export type ChatStreamEvent =
  | { type: 'content'; content: string }
  | { type: 'sources'; sources: ChatSource[] }
  | { type: 'error'; error: string }
  | { type: 'done' }

interface ChatStreamPayload {
  content?: string
  sources?: ChatSource[]
  error?: string
  done?: boolean
}

function parseSseEvent(rawEvent: string): ChatStreamEvent | null {
  const dataLine = rawEvent.split('\n').find((line) => line.startsWith('data: '))
  if (!dataLine) {
    return null
  }
  const payload = JSON.parse(dataLine.slice('data: '.length)) as ChatStreamPayload
  if (payload.content !== undefined) {
    return { type: 'content', content: payload.content }
  }
  if (payload.sources !== undefined) {
    return { type: 'sources', sources: payload.sources }
  }
  if (payload.error !== undefined) {
    return { type: 'error', error: payload.error }
  }
  if (payload.done) {
    return { type: 'done' }
  }
  return null
}

export interface StreamChatParams {
  threadId: string
  message: string
  history: ChatMessage[]
  signal?: AbortSignal
}

/**
 * POST /api/chat をSSEで受信し、イベント単位で逐次yieldする非同期ジェネレータ。
 * サーバーは`\n\n`区切りでイベントを送るため、受信済みバッファをその区切りで
 * 都度分割し、境界をまたいだ分割受信（chunk途中でのカット）にも対応する。
 */
export async function* streamChat({
  threadId,
  message,
  history,
  signal,
}: StreamChatParams): AsyncGenerator<ChatStreamEvent> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thread_id: threadId, message, history }),
    signal,
  })
  if (!response.ok || !response.body) {
    throw new Error(`チャットの送信に失敗しました (status: ${response.status})`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      break
    }
    buffer += decoder.decode(value, { stream: true })

    let separatorIndex = buffer.indexOf('\n\n')
    while (separatorIndex !== -1) {
      const rawEvent = buffer.slice(0, separatorIndex)
      buffer = buffer.slice(separatorIndex + 2)
      const event = parseSseEvent(rawEvent)
      if (event) {
        yield event
      }
      separatorIndex = buffer.indexOf('\n\n')
    }
  }
}
