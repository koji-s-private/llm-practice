import { useState } from 'react'
import { Button } from '@/components/ui/button'
import type { IndexedFile } from '@/lib/files'

interface FileListItemProps {
  file: IndexedFile
  onDelete: () => void
  isDeleting: boolean
}

// app.pyの_render_indexed_file_list()と同様、誤操作で消さないよう削除は2段階確認にする。
export function FileListItem({ file, onDelete, isDeleting }: FileListItemProps) {
  const [confirming, setConfirming] = useState(false)

  return (
    <li className="border-border rounded-md border p-2 text-sm">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate">
          📄 {file.name}{' '}
          <span className="text-muted-foreground text-xs">{file.chunk_count}チャンク</span>
        </span>
        {!confirming && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setConfirming(true)}
            aria-label={`${file.name} を削除`}
          >
            🗑️ 削除
          </Button>
        )}
      </div>
      {confirming && (
        <div className="mt-2 flex items-center gap-2">
          <p className="text-muted-foreground flex-1 text-xs">
            「{file.name}」を削除します。この操作は取り消せません。
          </p>
          <Button variant="destructive" size="sm" onClick={onDelete} disabled={isDeleting}>
            {isDeleting ? '削除中...' : '削除する'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setConfirming(false)}
            disabled={isDeleting}
          >
            キャンセル
          </Button>
        </div>
      )}
    </li>
  )
}
