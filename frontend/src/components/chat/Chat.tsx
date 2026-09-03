import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useState } from 'react'
import { ChatInput } from '@/components/chat/ChatInput'
import { ChatMessageList } from '@/components/chat/ChatMessageList'
import type { DisplayMessage } from '@/components/chat/types'
import { ThreadPanel } from '@/components/conversations/ThreadPanel'
import { FileManager } from '@/components/files/FileManager'
import { Button } from '@/components/ui/button'
import {
  type ChatMessage,
  type ChatSource,
  type ChatStreamEvent,
  createNewThread,
  streamChat,
} from '@/lib/chat'
import { type ConversationTurn, fetchConversation, saveConversation } from '@/lib/conversations'

const THREADS_QUERY_KEY = ['threads']

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '不明なエラーが発生しました'
}

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

function turnToMessages(turn: ConversationTurn): DisplayMessage[] {
  return [
    { id: crypto.randomUUID(), role: 'user', content: turn.question },
    { id: crypto.randomUUID(), role: 'assistant', content: turn.answer, sources: turn.sources },
  ]
}

export function Chat() {
  const queryClient = useQueryClient()

  // マウント時に1本発行しておき、会話スレッド一覧から選択・新規作成した場合はmanualThreadIdで上書きする。
  const newThread = useQuery({
    queryKey: ['newThread'],
    queryFn: createNewThread,
    staleTime: Infinity,
    retry: false,
  })
  const [manualThreadId, setManualThreadId] = useState<string | undefined>(undefined)
  const threadId = manualThreadId ?? newThread.data?.thread_id

  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [isSending, setIsSending] = useState(false)
  const [isFileManagerOpen, setIsFileManagerOpen] = useState(false)
  const [isThreadPanelOpen, setIsThreadPanelOpen] = useState(false)

  const newThreadMutation = useMutation({
    mutationFn: createNewThread,
    onSuccess: (data) => {
      setManualThreadId(data.thread_id)
      setMessages([])
    },
  })

  const switchThreadMutation = useMutation({
    mutationFn: async (targetThreadId: string) => ({
      threadId: targetThreadId,
      turns: await fetchConversation(targetThreadId),
    }),
    onSuccess: ({ threadId: switchedThreadId, turns }) => {
      setManualThreadId(switchedThreadId)
      setMessages(turns.flatMap(turnToMessages))
    },
  })

  const saveConversationMutation = useMutation({
    mutationFn: saveConversation,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: THREADS_QUERY_KEY })
    },
  })

  const handleThreadDeleted = useCallback(
    (deletedThreadId: string) => {
      if (deletedThreadId === threadId) {
        newThreadMutation.mutate()
      }
    },
    [threadId, newThreadMutation],
  )

  const isBusy = isSending || switchThreadMutation.isPending || newThreadMutation.isPending

  const handleSubmit = useCallback(
    async (text: string) => {
      if (!threadId || isBusy) {
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

      // 保存すべき回答本文・参照元は、非同期に更新されるmessagesステートに頼らず
      // ストリームイベントから直接組み立てる（setMessages完了を待たずに保存できるようにするため）。
      let answer = ''
      let sources: ChatSource[] = []
      let hasError = false

      try {
        for await (const event of streamChat({ threadId, message: text, history })) {
          update(event)
          if (event.type === 'content') {
            answer += event.content
          } else if (event.type === 'sources') {
            sources = event.sources
          } else if (event.type === 'error') {
            hasError = true
          }
        }
        if (!hasError) {
          saveConversationMutation.mutate({
            threadId,
            question: text,
            answer,
            isFallback: sources.length === 0,
          })
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
    [threadId, isBusy, messages, saveConversationMutation],
  )

  return (
    <div className="mx-auto flex h-svh w-full max-w-3xl flex-col">
      <header className="flex items-center justify-between gap-2 border-b p-4">
        <div>
          <h1 className="text-xl font-semibold">Doclore</h1>
          {newThread.isError && (
            <p className="text-destructive text-xs">
              会話の初期化に失敗しました: {newThread.error.message}
            </p>
          )}
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsThreadPanelOpen((open) => !open)}
            aria-expanded={isThreadPanelOpen}
          >
            💬 会話
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsFileManagerOpen((open) => !open)}
            aria-expanded={isFileManagerOpen}
          >
            📁 ファイル管理
          </Button>
        </div>
      </header>
      {switchThreadMutation.isError && (
        <p className="text-destructive border-b px-4 py-2 text-xs">
          会話の切り替えに失敗しました: {errorMessage(switchThreadMutation.error)}
        </p>
      )}
      {newThreadMutation.isError && (
        <p className="text-destructive border-b px-4 py-2 text-xs">
          新しい会話の作成に失敗しました: {errorMessage(newThreadMutation.error)}
        </p>
      )}
      {saveConversationMutation.isError && (
        <p className="text-destructive border-b px-4 py-2 text-xs">
          会話の保存に失敗しました: {errorMessage(saveConversationMutation.error)}
        </p>
      )}
      {isThreadPanelOpen && (
        <div className="max-h-80 overflow-y-auto border-b">
          <ThreadPanel
            activeThreadId={threadId}
            disabled={isBusy}
            onSelectThread={(selectedThreadId) => switchThreadMutation.mutate(selectedThreadId)}
            onNewThread={() => newThreadMutation.mutate()}
            onThreadDeleted={handleThreadDeleted}
          />
        </div>
      )}
      {isFileManagerOpen && (
        <div className="max-h-80 overflow-y-auto border-b">
          <FileManager />
        </div>
      )}
      <ChatMessageList messages={messages} />
      <ChatInput onSubmit={(text) => void handleSubmit(text)} disabled={!threadId || isBusy} />
    </div>
  )
}
