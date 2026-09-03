import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { ThreadListItem } from '@/components/conversations/ThreadListItem'
import { Button } from '@/components/ui/button'
import {
  deleteThread,
  fetchThreads,
  updateThreadTitle,
  type ThreadSummary,
} from '@/lib/conversations'

const THREADS_QUERY_KEY = ['threads']

interface ThreadPanelProps {
  activeThreadId?: string
  disabled: boolean
  onSelectThread: (threadId: string) => void
  onNewThread: () => void
  onThreadDeleted: (threadId: string) => void
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback
}

// app.pyの_filter_threads()相当。最初の質問文に対するキーワードの部分一致で絞り込む。
function filterThreads(threads: ThreadSummary[], keyword: string): ThreadSummary[] {
  const trimmed = keyword.trim().toLowerCase()
  if (!trimmed) {
    return threads
  }
  return threads.filter((thread) => thread.first_question.toLowerCase().includes(trimmed))
}

export function ThreadPanel({
  activeThreadId,
  disabled,
  onSelectThread,
  onNewThread,
  onThreadDeleted,
}: ThreadPanelProps) {
  const queryClient = useQueryClient()
  const [search, setSearch] = useState('')

  const threadsQuery = useQuery({ queryKey: THREADS_QUERY_KEY, queryFn: fetchThreads })

  const titleMutation = useMutation({
    mutationFn: ({ threadId, title }: { threadId: string; title: string }) =>
      updateThreadTitle(threadId, title),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: THREADS_QUERY_KEY })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteThread,
    onSuccess: (_data, threadId) => {
      void queryClient.invalidateQueries({ queryKey: THREADS_QUERY_KEY })
      onThreadDeleted(threadId)
    },
  })

  const threads = threadsQuery.data ?? []
  const filteredThreads = filterThreads(threads, search)

  return (
    <div className="flex flex-col gap-3 p-4">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">会話スレッド</h2>
        <Button size="sm" onClick={onNewThread} disabled={disabled}>
          🆕 新しい会話
        </Button>
      </div>

      {threads.length > 0 && (
        <input
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="🔍 質問内容で絞り込み..."
          className="border-border rounded-md border px-2 py-1 text-sm"
        />
      )}

      {threadsQuery.isLoading && <p className="text-muted-foreground text-xs">読み込み中...</p>}
      {threadsQuery.isError && (
        <p className="text-destructive text-xs">
          {errorMessage(threadsQuery.error, '会話スレッド一覧の取得に失敗しました')}
        </p>
      )}
      {threads.length === 0 && !threadsQuery.isLoading && !threadsQuery.isError && (
        <p className="text-muted-foreground text-xs">まだ保存された会話スレッドはありません。</p>
      )}
      {threads.length > 0 && filteredThreads.length === 0 && (
        <p className="text-muted-foreground text-xs">
          該当する会話スレッドが見つかりませんでした。
        </p>
      )}
      {deleteMutation.isError && (
        <p className="text-destructive text-xs">
          {errorMessage(deleteMutation.error, '会話スレッドの削除に失敗しました')}
        </p>
      )}
      {titleMutation.isError && (
        <p className="text-destructive text-xs">
          {errorMessage(titleMutation.error, 'タイトルの保存に失敗しました')}
        </p>
      )}

      {filteredThreads.length > 0 && (
        <ul className="flex flex-col gap-2">
          {filteredThreads.map((thread) => (
            <ThreadListItem
              key={thread.thread_id}
              thread={thread}
              isActive={thread.thread_id === activeThreadId}
              onSelect={() => onSelectThread(thread.thread_id)}
              onDelete={() => deleteMutation.mutate(thread.thread_id)}
              isDeleting={deleteMutation.isPending && deleteMutation.variables === thread.thread_id}
              onSaveTitle={(title) => titleMutation.mutate({ threadId: thread.thread_id, title })}
              isSavingTitle={
                titleMutation.isPending && titleMutation.variables?.threadId === thread.thread_id
              }
            />
          ))}
        </ul>
      )}
    </div>
  )
}
