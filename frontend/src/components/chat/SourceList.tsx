import { ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'
import type { ChatSource } from '@/lib/api'

/**
 * 回答の根拠となった参照元一覧の表示。app.pyの`_render_answer_provenance`に相当し、
 * sourcesが空か否かで「ドキュメント根拠」か「一般知識」かのバッジを出し分ける。
 */
export function SourceList({ sources }: { sources: ChatSource[] }) {
  const [isOpen, setIsOpen] = useState(false)

  if (sources.length === 0) {
    return (
      <p className="text-muted-foreground mt-2 text-xs">
        🧠 一般知識による回答（ドキュメントに該当情報なし）
      </p>
    )
  }

  return (
    <div className="mt-2">
      <p className="text-muted-foreground text-xs">🔍 ドキュメントに基づく回答</p>
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        aria-expanded={isOpen}
        className="text-muted-foreground hover:text-foreground mt-1 inline-flex items-center gap-1 text-xs underline-offset-2 hover:underline"
      >
        {isOpen ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        参照した箇所を見る（{sources.length}件）
      </button>
      {isOpen && (
        <ul className="mt-2 flex flex-col gap-2">
          {sources.map((source, index) => (
            <li
              key={`${source.label}-${index}`}
              className="border-border rounded-md border p-2 text-xs"
            >
              <p className="font-medium">
                [{index + 1}] {source.label}
              </p>
              <p className="text-muted-foreground mt-1 whitespace-pre-wrap">{source.snippet}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
