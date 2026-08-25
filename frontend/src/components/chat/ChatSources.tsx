import type { ChatSource } from '@/lib/chat'

// app.pyの`_render_answer_provenance`相当。sourcesの有無だけで
// 「ドキュメント根拠」か「一般知識」かを判定する（save_conversationのis_fallback判定と同じ考え方）。
export function ChatSources({ sources }: { sources: ChatSource[] }) {
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
      <details className="mt-1 text-xs">
        <summary className="text-muted-foreground hover:text-foreground cursor-pointer select-none">
          参照した箇所を見る（{sources.length}件）
        </summary>
        <ul className="mt-2 space-y-2">
          {sources.map((source, index) => (
            <li key={`${source.label}-${index}`} className="border-border rounded-md border p-2">
              <p className="font-medium">
                [{index + 1}] {source.label}
              </p>
              <p className="text-muted-foreground mt-1 whitespace-pre-wrap">{source.snippet}</p>
            </li>
          ))}
        </ul>
      </details>
    </div>
  )
}
