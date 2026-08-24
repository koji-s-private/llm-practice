import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { SourceList } from './SourceList'

describe('SourceList', () => {
  it('sourcesが空の場合は一般知識バッジのみ表示する', () => {
    render(<SourceList sources={[]} />)
    expect(screen.getByText(/一般知識による回答/)).toBeInTheDocument()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('sourcesがある場合はトグルボタンを押すと詳細が表示される', async () => {
    const user = userEvent.setup()
    render(<SourceList sources={[{ label: 'doc.txt', snippet: '抜粋テキスト' }]} />)

    expect(screen.getByText(/ドキュメントに基づく回答/)).toBeInTheDocument()
    const toggle = screen.getByRole('button', { name: /参照した箇所を見る（1件）/ })
    expect(screen.queryByText('doc.txt', { exact: false })).not.toBeInTheDocument()

    await user.click(toggle)

    expect(screen.getByText(/doc\.txt/)).toBeInTheDocument()
    expect(screen.getByText('抜粋テキスト')).toBeInTheDocument()
  })
})
