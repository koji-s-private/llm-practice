import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ChatMessageItem } from '@/components/chat/ChatMessageItem'
import type { DisplayMessage } from '@/components/chat/types'

describe('ChatMessageItem', () => {
  it('ユーザーメッセージはプレーンテキストで右寄せ表示される', () => {
    const message: DisplayMessage = { id: '1', role: 'user', content: 'こんにちは' }
    const { container } = render(<ChatMessageItem message={message} />)

    expect(screen.getByText('こんにちは')).toBeInTheDocument()
    expect(container.querySelector('.justify-end')).toBeInTheDocument()
  })

  it('アシスタントメッセージはMarkdownとして表示される', () => {
    const message: DisplayMessage = { id: '2', role: 'assistant', content: '# 見出し' }
    render(<ChatMessageItem message={message} />)

    expect(screen.getByRole('heading', { name: '見出し' })).toBeInTheDocument()
  })

  it('ストリーミング中で内容が空のとき「考え中...」を表示する', () => {
    const message: DisplayMessage = {
      id: '3',
      role: 'assistant',
      content: '',
      isStreaming: true,
    }
    render(<ChatMessageItem message={message} />)

    expect(screen.getByText('考え中...')).toBeInTheDocument()
  })

  it('ストリーミング中でも内容が届いていれば「考え中...」を表示しない', () => {
    const message: DisplayMessage = {
      id: '4',
      role: 'assistant',
      content: '回答中です',
      isStreaming: true,
    }
    render(<ChatMessageItem message={message} />)

    expect(screen.queryByText('考え中...')).not.toBeInTheDocument()
  })

  it('エラー時はエラーメッセージを表示し、参照元は表示しない', () => {
    const message: DisplayMessage = {
      id: '5',
      role: 'assistant',
      content: '途中まで',
      error: '通信エラーが発生しました',
      sources: [{ label: 'doc.pdf', snippet: '抜粋' }],
    }
    render(<ChatMessageItem message={message} />)

    expect(screen.getByText('エラー: 通信エラーが発生しました')).toBeInTheDocument()
    expect(screen.queryByText(/参照した箇所を見る/)).not.toBeInTheDocument()
  })

  it('完了後にsourcesがあれば参照元を表示する', () => {
    const message: DisplayMessage = {
      id: '6',
      role: 'assistant',
      content: '回答本文',
      isStreaming: false,
      sources: [{ label: 'doc.pdf', snippet: '抜粋' }],
    }
    render(<ChatMessageItem message={message} />)

    expect(screen.getByText('参照した箇所を見る（1件）')).toBeInTheDocument()
  })
})
