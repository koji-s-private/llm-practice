import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ChatUiMessage } from '@/hooks/useChat'
import { ChatMessageItem } from './ChatMessageItem'

describe('ChatMessageItem', () => {
  it('ユーザーメッセージを表示する', () => {
    const message: ChatUiMessage = { id: '1', role: 'user', content: '質問です' }
    render(<ChatMessageItem message={message} />)
    expect(screen.getByText('質問です')).toBeInTheDocument()
  })

  it('検索中は回答本文の代わりにプレースホルダーを表示する', () => {
    const message: ChatUiMessage = { id: '2', role: 'assistant', content: '', isSearching: true }
    render(<ChatMessageItem message={message} />)
    expect(screen.getByText(/検索して回答を考え中/)).toBeInTheDocument()
  })

  it('Markdownの回答本文とコードブロックを表示する', () => {
    const message: ChatUiMessage = {
      id: '3',
      role: 'assistant',
      content: '**回答**です\n\n```js\nconst x = 1\n```',
      sources: [],
    }
    render(<ChatMessageItem message={message} />)
    expect(screen.getByText('回答')).toBeInTheDocument()
    expect(screen.getByText('const')).toBeInTheDocument()
  })

  it('エラーを表示する', () => {
    const message: ChatUiMessage = { id: '4', role: 'assistant', content: '', error: '通信失敗' }
    render(<ChatMessageItem message={message} />)
    expect(screen.getByRole('alert')).toHaveTextContent('通信失敗')
  })

  it('sourcesがあれば参照元リンクを表示する', () => {
    const message: ChatUiMessage = {
      id: '5',
      role: 'assistant',
      content: '回答',
      sources: [{ label: 'doc.txt', snippet: '抜粋' }],
    }
    render(<ChatMessageItem message={message} />)
    expect(screen.getByRole('button', { name: /参照した箇所を見る/ })).toBeInTheDocument()
  })
})
