import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { MarkdownContent } from '@/components/chat/MarkdownContent'

describe('MarkdownContent', () => {
  it('見出しをレンダリングする', () => {
    render(<MarkdownContent content="# 見出し1" />)
    expect(screen.getByRole('heading', { level: 1, name: '見出し1' })).toBeInTheDocument()
  })

  it('箇条書きリストをレンダリングする', () => {
    render(<MarkdownContent content={'- 項目A\n- 項目B'} />)
    expect(screen.getByRole('list')).toBeInTheDocument()
    expect(screen.getByText('項目A')).toBeInTheDocument()
    expect(screen.getByText('項目B')).toBeInTheDocument()
  })

  it('インラインコードをレンダリングする', () => {
    render(<MarkdownContent content="これは`inline code`です" />)
    expect(screen.getByText('inline code').tagName).toBe('CODE')
  })

  it('フェンス付きコードブロックを遅延読み込み後にシンタックスハイライト付きでレンダリングする', async () => {
    const content = '```python\nprint("hello")\n```'
    const { container } = render(<MarkdownContent content={content} />)

    // CodeBlockはReact.lazyで遅延読み込みされるため、ハイライト済みのspanが
    // 現れるまで非同期に待つ必要がある。
    await waitFor(() => {
      expect(container.querySelectorAll('span.token').length).toBeGreaterThan(0)
    })
    expect(container.textContent).toContain('print')
    expect(container.textContent).toContain('hello')
  })

  it('登録されていない言語のコードブロックでも例外にならずコード内容を表示する', async () => {
    const content = '```not-a-real-language\nsome code\n```'
    const { container } = render(<MarkdownContent content={content} />)

    await waitFor(() => {
      expect(container.textContent).toContain('some code')
    })
  })

  it('リンクをtarget=_blank・rel=noreferrer付きでレンダリングする', () => {
    render(<MarkdownContent content="[リンク](https://example.com)" />)
    const link = screen.getByRole('link', { name: 'リンク' })
    expect(link).toHaveAttribute('href', 'https://example.com')
    expect(link).toHaveAttribute('target', '_blank')
    expect(link).toHaveAttribute('rel', 'noreferrer')
  })

  it('空文字を渡しても例外にならない', () => {
    const { container } = render(<MarkdownContent content="" />)
    expect(container).toBeInTheDocument()
  })
})

describe('MarkdownContent コードブロックの遅延読み込み', () => {
  afterEach(() => {
    vi.doUnmock('@/components/chat/CodeBlock')
    vi.resetModules()
  })

  it('読み込み完了前はプレーンなpre/codeを表示し、完了後にハイライト済みの内容へ置き換わる', async () => {
    let resolveModule!: (mod: typeof import('@/components/chat/CodeBlock')) => void
    const modulePromise = new Promise<typeof import('@/components/chat/CodeBlock')>((resolve) => {
      resolveModule = resolve
    })
    vi.doMock('@/components/chat/CodeBlock', () => modulePromise)
    vi.resetModules()

    const { MarkdownContent: IsolatedMarkdownContent } = await import(
      '@/components/chat/MarkdownContent'
    )

    const { container } = render(
      <IsolatedMarkdownContent content={'```python\nprint("hello")\n```'} />,
    )

    // 遅延読み込み中はSuspenseのfallbackとしてプレーンな<pre><code>が表示される
    const fallbackCode = container.querySelector('pre > code')
    expect(fallbackCode).toBeInTheDocument()
    expect(fallbackCode?.querySelectorAll('span.token').length).toBe(0)
    expect(container.textContent).toContain('print')

    resolveModule(await vi.importActual('@/components/chat/CodeBlock'))

    // 読み込み完了後はシンタックスハイライト済みのCodeBlockに置き換わる
    await waitFor(() => {
      expect(container.querySelectorAll('span.token').length).toBeGreaterThan(0)
    })
  })
})
