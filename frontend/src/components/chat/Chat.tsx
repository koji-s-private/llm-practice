import { useQuery } from '@tanstack/react-query'
import { useCallback, useState } from 'react'
import { ChatInput } from '@/components/chat/ChatInput'
import { ChatMessageList } from '@/components/chat/ChatMessageList'
import type { DisplayMessage } from '@/components/chat/types'
import { type ChatMessage, type ChatStreamEvent, createNewThread, streamChat } from '@/lib/chat'

function toHistory(messages: DisplayMessage[]): ChatMessage[] {
  return messages
    .filter((message) => !message.isStreaming && !message.error)
    .map((message) => ({ role: message.role, content: message.content }))
}

function applyStreamEvent(message: DisplayMessage, event: ChatStreamEvent): DisplayMessage {
  switch (event.type) {
    case 'content':
      return { ...message, content: message.content + event.content }
    case 'sources':
      return { ...message, sources: event.sources }
    case 'error':
      return { ...message, error: event.error, isStreaming: false }
    case 'done':
      // sourcesイベントは一般知識回答時（根拠ドキュメント無し）にはサーバーから送られてこないため、
      // 完了時点で未受信ならChatSourcesの表示条件（message.sources &&）を満たすよう空配列を補う。
      return { ...message, isStreaming: false, sources: message.sources ?? [] }
  }
}

export function Chat() {
  // Step5（会話管理UI）が入るまでは、マウント時に発行したスレッド1本を使い回す。
  const newThread = useQuery({
    queryKey: ['newThread'],
    queryFn: createNewThread,
    staleTime: Infinity,
    retry: false,
  })
  const threadId = newThread.data?.thread_id

  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [isSending, setIsSending] = useState(false)

  const handleSubmit = useCallback(
    async (text: string) => {
      if (!threadId || isSending) {
        return
      }
      const history = toHistory(messages)
      const assistantId = crypto.randomUUID()
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: 'user', content: text },
        { id: assistantId, role: 'assistant', content: '', isStreaming: true },
      ])
      setIsSending(true)

      function update(event: ChatStreamEvent) {
        setMessages((prev) =>
          prev.map((message) =>
            message.id === assistantId ? applyStreamEvent(message, event) : message,
          ),
        )
      }

      try {
        for await (const event of streamChat({ threadId, message: text, history })) {
          update(event)
        }
      } catch (error) {
        update({
          type: 'error',
          error: error instanceof Error ? error.message : '通信エラーが発生しました',
        })
      } finally {
        setIsSending(false)
      }
    },
    [threadId, isSending, messages],
  )

  return (
    <div className="mx-auto flex h-svh w-full max-w-3xl flex-col">
      <header className="border-b p-4">
        <h1 className="text-xl font-semibold">Doclore</h1>
        {newThread.isError && (
          <p className="text-destructive text-xs">
            会話の初期化に失敗しました: {newThread.error.message}
          </p>
        )}
      </header>
      <ChatMessageList messages={messages} />
      <ChatInput onSubmit={(text) => void handleSubmit(text)} disabled={!threadId || isSending} />
    </div>
  )
}
