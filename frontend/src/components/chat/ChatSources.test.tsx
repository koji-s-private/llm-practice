import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ChatSources } from '@/components/chat/ChatSources'

describe('ChatSources', () => {
  it('sourcesが空のとき、一般知識による回答である旨を表示する', () => {
    render(<ChatSources sources={[]} />)

    expect(
      screen.getByText('🧠 一般知識による回答（ドキュメントに該当情報なし）'),
    ).toBeInTheDocument()
    expect(screen.queryByText(/参照した箇所を見る/)).not.toBeInTheDocument()
  })

  it('sourcesがあるとき、件数と内容を表示する', () => {
    render(
      <ChatSources
        sources={[
          { label: 'doc1.pdf p.1', snippet: '本文抜粋1' },
          { label: 'doc2.pdf p.3', snippet: '本文抜粋2' },
        ]}
      />,
    )

    expect(screen.getByText('🔍 ドキュメントに基づく回答')).toBeInTheDocument()
    expect(screen.getByText('参照した箇所を見る（2件）')).toBeInTheDocument()
    expect(screen.getByText('[1] doc1.pdf p.1')).toBeInTheDocument()
    expect(screen.getByText('本文抜粋1')).toBeInTheDocument()
    expect(screen.getByText('[2] doc2.pdf p.3')).toBeInTheDocument()
    expect(screen.getByText('本文抜粋2')).toBeInTheDocument()
  })

  it('sourcesが1件のとき件数表示が単数になる', () => {
    render(<ChatSources sources={[{ label: 'doc1.pdf', snippet: '抜粋' }]} />)

    expect(screen.getByText('参照した箇所を見る（1件）')).toBeInTheDocument()
  })
})
