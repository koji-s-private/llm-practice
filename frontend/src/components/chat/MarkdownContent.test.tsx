import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
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

  it('フェンス付きコードブロックをシンタックスハイライト付きでレンダリングする', () => {
    const content = '```python\nprint("hello")\n```'
    const { container } = render(<MarkdownContent content={content} />)

    expect(container.querySelector('pre')).toBeInTheDocument()
    expect(container.textContent).toContain('print')
    expect(container.textContent).toContain('hello')
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
