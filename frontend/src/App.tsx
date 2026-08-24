import { ChatMessageList } from '@/components/chat/ChatMessageList'
import { MessageInput } from '@/components/chat/MessageInput'
import { useChat } from '@/hooks/useChat'

function App() {
  const { messages, sendMessage, isSending, isThreadReady, threadError } = useChat()

  return (
    <main className="mx-auto flex h-svh max-w-2xl flex-col px-4">
      <header className="border-border shrink-0 border-b py-4">
        <h1 className="text-xl font-semibold">Doclore</h1>
        <p className="text-muted-foreground text-sm">資料をもとに質問に答えるRAGチャットです。</p>
      </header>

      {threadError && (
        <p role="alert" className="text-destructive shrink-0 py-2 text-sm">
          会話の初期化に失敗しました: {threadError}
        </p>
      )}

      <ChatMessageList messages={messages} />

      <div className="shrink-0">
        <MessageInput
          onSend={(text) => void sendMessage(text)}
          disabled={isSending || !isThreadReady}
        />
      </div>
    </main>
  )
}

export default App
