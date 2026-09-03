// api/main.py（FastAPI）の会話スレッド管理関連エンドポイント
// (GET /api/conversations, GET/PUT /api/conversations/{thread_id}[/title],
// DELETE /api/conversations/{thread_id}, POST /api/conversations/save) を呼び出す薄いクライアント。
import { API_BASE_URL } from '@/lib/api'
import type { ChatSource } from '@/lib/chat'

export interface ThreadSummary {
  thread_id: string
  created_at: string
  first_question: string
  count: number
  title: string | null
}

export async function fetchThreads(): Promise<ThreadSummary[]> {
  const response = await fetch(`${API_BASE_URL}/api/conversations`)
  if (!response.ok) {
    throw new Error(`会話スレッド一覧の取得に失敗しました (status: ${response.status})`)
  }
  const data = (await response.json()) as { threads: ThreadSummary[] }
  return data.threads
}

export interface ConversationTurn {
  question: string
  answer: string
  created_at: string
  sources: ChatSource[]
}

export async function fetchConversation(threadId: string): Promise<ConversationTurn[]> {
  const response = await fetch(`${API_BASE_URL}/api/conversations/${encodeURIComponent(threadId)}`)
  if (!response.ok) {
    throw new Error(`会話内容の取得に失敗しました (status: ${response.status})`)
  }
  const data = (await response.json()) as { turns: ConversationTurn[] }
  return data.turns
}

export async function updateThreadTitle(threadId: string, title: string): Promise<string | null> {
  const response = await fetch(
    `${API_BASE_URL}/api/conversations/${encodeURIComponent(threadId)}/title`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    },
  )
  if (!response.ok) {
    throw new Error(`タイトルの保存に失敗しました (status: ${response.status})`)
  }
  const data = (await response.json()) as { title: string | null }
  return data.title
}

export interface SaveConversationParams {
  threadId: string
  question: string
  answer: string
  isFallback: boolean
}

// isFallback: ドキュメントに根拠が見つからず一般知識で回答した場合（sourcesが空）にtrueを渡す
// （app.pyの save_conversation(..., is_fallback=not sources) と同じ判定基準）。
export async function saveConversation({
  threadId,
  question,
  answer,
  isFallback,
}: SaveConversationParams): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/conversations/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      thread_id: threadId,
      question,
      answer,
      is_fallback: isFallback,
    }),
  })
  if (!response.ok) {
    throw new Error(`会話の保存に失敗しました (status: ${response.status})`)
  }
}

export async function deleteThread(threadId: string): Promise<void> {
  const response = await fetch(
    `${API_BASE_URL}/api/conversations/${encodeURIComponent(threadId)}`,
    {
      method: 'DELETE',
    },
  )
  if (!response.ok) {
    throw new Error(`会話スレッドの削除に失敗しました (status: ${response.status})`)
  }
}
