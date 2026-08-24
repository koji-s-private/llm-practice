// api/main.py（FastAPI）への疎通に使う共通クライアント設定。
// 開発時はViteの.envでVITE_API_BASE_URLを上書きできるが、未設定時はuvicornのデフォルト起動先を使う。
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export interface HealthResponse {
  status: string
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`)
  if (!response.ok) {
    throw new Error(`ヘルスチェックに失敗しました (status: ${response.status})`)
  }
  return response.json() as Promise<HealthResponse>
}

// --- 会話スレッド ---

export interface NewThreadResponse {
  thread_id: string
}

export async function createConversationThread(): Promise<NewThreadResponse> {
  const response = await fetch(`${API_BASE_URL}/api/conversations/new`, { method: 'POST' })
  if (!response.ok) {
    throw new Error(`会話スレッドの作成に失敗しました (status: ${response.status})`)
  }
  return response.json() as Promise<NewThreadResponse>
}

export interface SaveConversationParams {
  question: string
  answer: string
  threadId: string
  isFallback: boolean
}

export async function saveConversation(params: SaveConversationParams): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/conversations/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: params.question,
      answer: params.answer,
      thread_id: params.threadId,
      is_fallback: params.isFallback,
    }),
  })
  if (!response.ok) {
    throw new Error(`会話ログの保存に失敗しました (status: ${response.status})`)
  }
}

// --- チャット（ストリーミング） ---

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatSource {
  label: string
  snippet: string
}

export interface ChatStreamParams {
  threadId: string
  message: string
  history: ChatMessage[]
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

/**
 * POST /api/chat のSSEレスポンスを1イベントずつ非同期にyieldする。
 *
 * api/main.py はレスポンスを`data: <json>\n\n`区切りで送出するが、fetchのReadableStreamは
 * チャンク境界とイベント境界が一致するとは限らない（1イベントが複数チャンクに分割されたり、
 * 逆に複数イベントが1チャンクにまとまったりする）ため、バッファに貯めて"\n\n"単位で切り出す。
 */
export async function* streamChat(
  params: ChatStreamParams,
  signal?: AbortSignal,
): AsyncGenerator<ChatStreamEvent, void, void> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      thread_id: params.threadId,
      message: params.message,
      history: params.history,
    }),
    signal,
  })
  if (!response.ok || !response.body) {
    throw new Error(`チャットの送信に失敗しました (status: ${response.status})`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      let separatorIndex = buffer.indexOf('\n\n')
      while (separatorIndex !== -1) {
        const rawEvent = buffer.slice(0, separatorIndex).trim()
        buffer = buffer.slice(separatorIndex + 2)
        if (rawEvent.startsWith('data:')) {
          const payload = JSON.parse(rawEvent.slice('data:'.length).trim()) as ChatStreamPayload
          const event = toChatStreamEvent(payload)
          if (event) yield event
        }
        separatorIndex = buffer.indexOf('\n\n')
      }
    }
  } finally {
    reader.releaseLock()
  }
}

function toChatStreamEvent(payload: ChatStreamPayload): ChatStreamEvent | null {
  if (payload.content !== undefined) return { type: 'content', content: payload.content }
  if (payload.sources !== undefined) return { type: 'sources', sources: payload.sources }
  if (payload.error !== undefined) return { type: 'error', error: payload.error }
  if (payload.done) return { type: 'done' }
  return null
}
