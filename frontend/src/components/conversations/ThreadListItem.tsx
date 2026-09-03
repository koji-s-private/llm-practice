import { type FormEvent, useState } from 'react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { ThreadSummary } from '@/lib/conversations'

interface ThreadListItemProps {
  thread: ThreadSummary
  isActive: boolean
  onSelect: () => void
  onDelete: () => void
  isDeleting: boolean
  onSaveTitle: (title: string) => void
  isSavingTitle: boolean
  disabled: boolean
}

function truncate(text: string, limit: number): string {
  const trimmed = text.trim()
  return trimmed.length <= limit ? trimmed : `${trimmed.slice(0, limit)}...`
}

function formatCreatedAt(createdAt: string): string {
  const date = new Date(createdAt)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(
    date.getMinutes(),
  )}`
}

// app.pyの_format_thread_label()/_thread_display_label()相当のラベル整形。
function formatThreadLabel(thread: ThreadSummary): string {
  const timestamp = formatCreatedAt(thread.created_at)
  const snippet = thread.first_question ? truncate(thread.first_question, 24) : '(質問内容なし)'
  const base = `${timestamp}｜${snippet}（${thread.count}件）`
  return thread.title ? `📌 ${truncate(thread.title, 20)}（${base}）` : base
}

// app.pyの_render_indexed_file_list()と同様、誤操作で消さないよう削除は2段階確認にする。
export function ThreadListItem({
  thread,
  isActive,
  onSelect,
  onDelete,
  isDeleting,
  onSaveTitle,
  isSavingTitle,
  disabled,
}: ThreadListItemProps) {
  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const [editingTitle, setEditingTitle] = useState(false)
  const [titleInput, setTitleInput] = useState(thread.title ?? '')

  function handleTitleSubmit(event: FormEvent) {
    event.preventDefault()
    onSaveTitle(titleInput)
    setEditingTitle(false)
  }

  return (
    <li
      className={cn(
        'border-border rounded-md border p-2 text-sm',
        isActive && 'border-primary bg-muted',
      )}
    >
      <button
        type="button"
        onClick={onSelect}
        disabled={isActive || disabled}
        className="w-full text-left break-words disabled:cursor-default disabled:opacity-50"
      >
        {formatThreadLabel(thread)}
      </button>

      {!confirmingDelete && !editingTitle && (
        <div className="mt-2 flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setTitleInput(thread.title ?? '')
              setEditingTitle(true)
            }}
            disabled={disabled}
          >
            ✏️ タイトル編集
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setConfirmingDelete(true)}
            aria-label={`${thread.thread_id} を削除`}
            disabled={disabled}
          >
            🗑️ 削除
          </Button>
        </div>
      )}

      {editingTitle && (
        <form onSubmit={handleTitleSubmit} className="mt-2 flex items-center gap-2">
          <input
            value={titleInput}
            onChange={(event) => setTitleInput(event.target.value)}
            placeholder="例: 経費精算の質問"
            className="border-border flex-1 rounded-md border px-2 py-1 text-xs"
            disabled={isSavingTitle || disabled}
          />
          <Button type="submit" size="sm" disabled={isSavingTitle || disabled}>
            {isSavingTitle ? '保存中...' : '💾 保存'}
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setEditingTitle(false)}
            disabled={isSavingTitle}
          >
            キャンセル
          </Button>
        </form>
      )}

      {confirmingDelete && (
        <div className="mt-2 flex items-center gap-2">
          <p className="text-muted-foreground flex-1 text-xs">
            このスレッドの会話ログをすべて削除します。この操作は取り消せません。
          </p>
          <Button
            variant="destructive"
            size="sm"
            onClick={onDelete}
            disabled={isDeleting || disabled}
          >
            {isDeleting ? '削除中...' : '削除する'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setConfirmingDelete(false)}
            disabled={isDeleting}
          >
            キャンセル
          </Button>
        </div>
      )}
    </li>
  )
}
