import { useCallback, useEffect, useState } from 'react'
import {
  createConversationThread,
  saveConversation,
  streamChat,
  type ChatMessage,
  type ChatSource,
} from '@/lib/api'

export interface ChatUiMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: ChatSource[]
  /** ツール検索中で最初の回答トークンがまだ届いていない状態（app.pyの検索中プレースホルダー相当） */
  isSearching?: boolean
  error?: string
}

export interface UseChatResult {
  messages: ChatUiMessage[]
  sendMessage: (text: string) => Promise<void>
  isSending: boolean
  isThreadReady: boolean
  threadError: string | null
}

/**
 * チャットの会話状態とAPI呼び出しをまとめて管理するフック。
 *
 * マウント時に POST /api/conversations/new でスレッドIDを発行し、以降の /api/chat 呼び出しに使う
 * （api/main.pyがthread_idの形式検証をしているため、ランダムなUUID文字列を自前生成するのではなく
 * サーバー側の発行結果をそのまま使う）。回答が完了したら、Streamlit版と同じデフォルト挙動として
 * 会話ログを自動保存する（保存失敗は画面上の会話継続を妨げないよう握りつぶす）。
 */
export function useChat(): UseChatResult {
  const [threadId, setThreadId] = useState<string | null>(null)
  const [threadError, setThreadError] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatUiMessage[]>([])
  const [isSending, setIsSending] = useState(false)

  useEffect(() => {
    let cancelled = false
    createConversationThread()
      .then(({ thread_id }) => {
        if (!cancelled) setThreadId(thread_id)
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setThreadError(error instanceof Error ? error.message : String(error))
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || !threadId || isSending) return

      const history: ChatMessage[] = messages.map((m) => ({ role: m.role, content: m.content }))
      const assistantId = crypto.randomUUID()
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: 'user', content: trimmed },
        { id: assistantId, role: 'assistant', content: '', isSearching: true },
      ])
      setIsSending(true)

      const updateAssistant = (patch: Partial<ChatUiMessage>) => {
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, ...patch } : m)))
      }

      let answer = ''
      let sources: ChatSource[] = []
      let streamError: string | null = null

      try {
        for await (const event of streamChat({ threadId, message: trimmed, history })) {
          if (event.type === 'content') {
            answer += event.content
            updateAssistant({ content: answer, isSearching: false })
          } else if (event.type === 'sources') {
            sources = event.sources
            updateAssistant({ sources })
          } else if (event.type === 'error') {
            streamError = event.error
            updateAssistant({ error: event.error, isSearching: false })
          }
        }
      } catch (error) {
        streamError = error instanceof Error ? error.message : String(error)
        updateAssistant({ error: streamError, isSearching: false })
      }

      // api/main.pyはsourcesが空の場合"sources"イベント自体を送らないため、正常終了時は
      // 明示的に空配列をセットし、一般知識バッジ（SourceList）を表示できるようにする。
      if (!streamError) {
        updateAssistant({ sources })
      }

      setIsSending(false)

      if (!streamError && answer) {
        saveConversation({
          question: trimmed,
          answer,
          threadId,
          isFallback: sources.length === 0,
        }).catch(() => {
          // 保存に失敗しても画面上の会話継続には影響させない（次回の手動再同期等に委ねる）。
        })
      }
    },
    [threadId, isSending, messages],
  )

  return { messages, sendMessage, isSending, isThreadReady: threadId !== null, threadError }
}
