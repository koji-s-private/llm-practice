import { ChatSources } from '@/components/chat/ChatSources'
import { MarkdownContent } from '@/components/chat/MarkdownContent'
import type { DisplayMessage } from '@/components/chat/types'
import { cn } from '@/lib/utils'

export function ChatMessageItem({ message }: { message: DisplayMessage }) {
  const isUser = message.role === 'user'

  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={cn(
          'max-w-[85%] rounded-2xl px-4 py-2',
          isUser ? 'bg-primary text-primary-foreground' : 'bg-muted text-foreground',
        )}
      >
        {isUser ? (
          <p className="text-sm break-words whitespace-pre-wrap">{message.content}</p>
        ) : (
          <>
            <MarkdownContent content={message.content} />
            {message.isStreaming && message.content === '' && (
              <p className="text-muted-foreground text-sm">考え中...</p>
            )}
            {message.error && (
              <p className="text-destructive mt-2 text-sm">エラー: {message.error}</p>
            )}
            {!message.isStreaming && !message.error && message.sources && (
              <ChatSources sources={message.sources} />
            )}
          </>
        )}
      </div>
    </div>
  )
}
