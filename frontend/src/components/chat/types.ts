import type { ChatRole, ChatSource } from '@/lib/chat'

/** 画面表示用のメッセージ。ストリーミング中はcontentを逐次追記し、完了後にsources/errorを確定する。 */
export interface DisplayMessage {
  id: string
  role: ChatRole
  content: string
  sources?: ChatSource[]
  isStreaming?: boolean
  error?: string
}
