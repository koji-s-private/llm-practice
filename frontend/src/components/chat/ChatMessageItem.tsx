import { Loader2 } from 'lucide-react'
import { Markdown } from '@/components/markdown/Markdown'
import { cn } from '@/lib/utils'
import type { ChatUiMessage } from '@/hooks/useChat'
import { SourceList } from './SourceList'

export function ChatMessageItem({ message }: { message: ChatUiMessage }) {
  const isUser = message.role === 'user'

  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={cn(
          'max-w-[85%] rounded-2xl px-4 py-2',
          isUser ? 'bg-primary text-primary-foreground' : 'bg-muted text-foreground',
        )}
      >
        {message.isSearching && (
          <p className="text-muted-foreground flex items-center gap-1.5 text-sm">
            <Loader2 className="size-3.5 animate-spin" />
            検索して回答を考え中...
          </p>
        )}
        {!message.isSearching && message.content && <Markdown>{message.content}</Markdown>}
        {message.error && (
          <p role="alert" className="text-destructive text-sm">
            エラーが発生しました: {message.error}
          </p>
        )}
        {!isUser && !message.isSearching && !message.error && message.sources && (
          <SourceList sources={message.sources} />
        )}
      </div>
    </div>
  )
}
