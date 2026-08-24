import { SendHorizontal } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'

export interface MessageInputProps {
  onSend: (text: string) => void
  disabled: boolean
  placeholder?: string
}

export function MessageInput({ onSend, disabled, placeholder }: MessageInputProps) {
  const [value, setValue] = useState('')

  const send = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
  }

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault()
    send()
  }

  // Shift+Enterは改行、Enter単独は送信（Streamlit版のst.chat_inputと同様の操作感）。
  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      send()
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex items-end gap-2 py-4">
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder={placeholder ?? '資料について気になることを聞いてみましょう'}
        rows={1}
        className="border-border bg-background placeholder:text-muted-foreground focus-visible:ring-ring/50 max-h-40 min-h-9 flex-1 resize-none rounded-lg border px-3 py-2 text-sm outline-none focus-visible:ring-3 disabled:opacity-50"
      />
      <Button type="submit" disabled={disabled || value.trim().length === 0} aria-label="送信">
        <SendHorizontal />
      </Button>
    </form>
  )
}
