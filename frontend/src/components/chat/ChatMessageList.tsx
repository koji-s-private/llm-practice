import { useEffect, useRef } from 'react'
import type { ChatUiMessage } from '@/hooks/useChat'
import { ChatMessageItem } from './ChatMessageItem'

export function ChatMessageList({ messages }: { messages: ChatUiMessage[] }) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <p className="text-muted-foreground flex flex-1 items-center justify-center text-sm">
        資料について気になることを聞いてみましょう
      </p>
    )
  }

  return (
    <div className="flex flex-1 flex-col gap-3 overflow-y-auto py-4">
      {messages.map((message) => (
        <ChatMessageItem key={message.id} message={message} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
