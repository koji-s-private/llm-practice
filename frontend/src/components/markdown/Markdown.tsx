import ReactMarkdown, { type Components } from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import remarkGfm from 'remark-gfm'

// react-markdown v9以降、code要素のcomponentsにinlineフラグが渡されないため、
// className（"language-xxx"）の有無でコードブロックとインラインコードを区別する。
const components: Components = {
  code({ className, children, ...props }) {
    const match = /language-(\w+)/.exec(className ?? '')
    if (!match) {
      return (
        <code className={className} {...props}>
          {children}
        </code>
      )
    }
    return (
      <SyntaxHighlighter
        language={match[1]}
        style={oneDark}
        PreTag="div"
        customStyle={{ margin: 0, borderRadius: 'var(--radius-md)' }}
      >
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    )
  },
}

// @tailwindcss/typography（prose）は未導入のため、Streamlit版のst.markdown相当の見た目を
// 素朴なユーティリティクラスで最低限再現する（段落間隔・リスト・コードブロックの折り返し等）。
const MARKDOWN_CLASSES =
  'text-sm leading-relaxed break-words ' +
  '[&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 ' +
  '[&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 ' +
  '[&_li]:my-0.5 [&_h1]:mt-3 [&_h1]:mb-2 [&_h1]:text-lg [&_h1]:font-semibold ' +
  '[&_h2]:mt-3 [&_h2]:mb-2 [&_h2]:text-base [&_h2]:font-semibold ' +
  '[&_h3]:mt-2 [&_h3]:mb-1 [&_h3]:font-semibold ' +
  '[&_a]:text-primary [&_a]:underline [&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground ' +
  '[&_pre]:my-2 [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:text-xs ' +
  '[&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-xs [&_pre_code]:bg-transparent [&_pre_code]:p-0 ' +
  '[&_table]:my-2 [&_table]:border-collapse [&_th]:border [&_th]:border-border [&_th]:px-2 [&_th]:py-1 [&_td]:border [&_td]:border-border [&_td]:px-2 [&_td]:py-1'

export function Markdown({ children }: { children: string }) {
  return (
    <div className={MARKDOWN_CLASSES}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {children}
      </ReactMarkdown>
    </div>
  )
}
