import { lazy, Suspense } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'

// シンタックスハイライト本体（Prismの言語データ込み）は本文表示の初回描画をブロックしない
// よう遅延読み込みし、コードブロックを含まないメッセージではバンドルを取得しない。
const CodeBlock = lazy(() => import('./CodeBlock'))

// Tailwindのpreflightがul/ol/blockquote等のデフォルト装飾を打ち消すため、
// react-markdownの標準タグ出力にTailwindユーティリティで最低限の見た目を補う
// （`@tailwindcss/typography`は導入コストの割に本UIでは差分が小さいため見送り）。
const markdownComponents: Components = {
  p: ({ children }) => <p className="mb-2 leading-relaxed last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="mb-2 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2 list-decimal space-y-1 pl-5 last:mb-0">{children}</ol>,
  a: ({ children, ...props }) => (
    <a
      className="text-primary underline underline-offset-2"
      target="_blank"
      rel="noreferrer"
      {...props}
    >
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-border text-muted-foreground mb-2 border-l-2 pl-3 last:mb-0">
      {children}
    </blockquote>
  ),
  code: ({ className, children, ...props }) => {
    const match = /language-(\w+)/.exec(className ?? '')
    if (!match) {
      return (
        <code className="bg-muted rounded px-1 py-0.5 text-sm" {...props}>
          {children}
        </code>
      )
    }
    const codeText = String(children).replace(/\n$/, '')
    return (
      <Suspense
        fallback={
          <pre className="bg-muted mb-2 overflow-x-auto rounded-lg p-3 text-sm last:mb-0">
            <code>{codeText}</code>
          </pre>
        }
      >
        <CodeBlock language={match[1]} code={codeText} />
      </Suspense>
    )
  },
}

export function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="text-sm break-words">
      <ReactMarkdown components={markdownComponents}>{content}</ReactMarkdown>
    </div>
  )
}
