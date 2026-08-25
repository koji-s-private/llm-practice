import { type FormEvent, type KeyboardEvent, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'

interface ChatInputProps {
  onSubmit: (message: string) => void
  disabled: boolean
}

export function ChatInput({ onSubmit, disabled }: ChatInputProps) {
  const [value, setValue] = useState('')

  function submitMessage() {
    const trimmed = value.trim()
    if (!trimmed || disabled) {
      return
    }
    onSubmit(trimmed)
    setValue('')
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    submitMessage()
  }

  // IME変換確定のEnterで誤送信しないよう、Shift+Enterのみ改行として扱う。
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      submitMessage()
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex items-end gap-2 border-t p-4">
      <Textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="質問を入力してください（Shift+Enterで改行）"
        disabled={disabled}
        className="max-h-40"
      />
      <Button type="submit" disabled={disabled || value.trim() === ''}>
        送信
      </Button>
    </form>
  )
}
