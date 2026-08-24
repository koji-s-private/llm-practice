import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { MessageInput } from './MessageInput'

describe('MessageInput', () => {
  it('入力してEnterで送信し、入力欄をクリアする', async () => {
    const user = userEvent.setup()
    const onSend = vi.fn()
    render(<MessageInput onSend={onSend} disabled={false} />)

    const textbox = screen.getByPlaceholderText('資料について気になることを聞いてみましょう')
    await user.type(textbox, 'こんにちは{Enter}')

    expect(onSend).toHaveBeenCalledWith('こんにちは')
    expect(textbox).toHaveValue('')
  })

  it('Shift+Enterでは送信せず改行する', async () => {
    const user = userEvent.setup()
    const onSend = vi.fn()
    render(<MessageInput onSend={onSend} disabled={false} />)

    const textbox = screen.getByPlaceholderText('資料について気になることを聞いてみましょう')
    await user.type(textbox, '1行目{Shift>}{Enter}{/Shift}2行目')

    expect(onSend).not.toHaveBeenCalled()
    expect(textbox).toHaveValue('1行目\n2行目')
  })

  it('disabled中は送信ボタンが無効化される', () => {
    render(<MessageInput onSend={vi.fn()} disabled={true} />)
    expect(screen.getByRole('button', { name: '送信' })).toBeDisabled()
  })

  it('空文字は送信できない', () => {
    const onSend = vi.fn()
    render(<MessageInput onSend={onSend} disabled={false} />)
    expect(screen.getByRole('button', { name: '送信' })).toBeDisabled()
  })
})
