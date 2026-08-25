import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ChatInput } from '@/components/chat/ChatInput'

describe('ChatInput', () => {
  it('通常のEnterで送信され、入力欄がクリアされる', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(<ChatInput onSubmit={onSubmit} disabled={false} />)

    const textarea = screen.getByPlaceholderText('質問を入力してください（Shift+Enterで改行）')
    await user.type(textarea, 'こんにちは')
    await user.keyboard('{Enter}')

    expect(onSubmit).toHaveBeenCalledExactlyOnceWith('こんにちは')
    expect(textarea).toHaveValue('')
  })

  it('Shift+Enterでは送信されず改行が入る', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(<ChatInput onSubmit={onSubmit} disabled={false} />)

    const textarea = screen.getByPlaceholderText('質問を入力してください（Shift+Enterで改行）')
    await user.type(textarea, '1行目{Shift>}{Enter}{/Shift}2行目')

    expect(onSubmit).not.toHaveBeenCalled()
    expect(textarea).toHaveValue('1行目\n2行目')
  })

  it('空文字（空白のみ含む）では送信されない', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(<ChatInput onSubmit={onSubmit} disabled={false} />)

    const textarea = screen.getByPlaceholderText('質問を入力してください（Shift+Enterで改行）')
    await user.type(textarea, '   ')
    await user.keyboard('{Enter}')

    expect(onSubmit).not.toHaveBeenCalled()

    const button = screen.getByRole('button', { name: '送信' })
    expect(button).toBeDisabled()
  })

  it('送信ボタンのクリックでも送信される', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(<ChatInput onSubmit={onSubmit} disabled={false} />)

    const textarea = screen.getByPlaceholderText('質問を入力してください（Shift+Enterで改行）')
    await user.type(textarea, 'ボタン送信')
    await user.click(screen.getByRole('button', { name: '送信' }))

    expect(onSubmit).toHaveBeenCalledExactlyOnceWith('ボタン送信')
  })

  it('IME変換確定のEnterでは送信されない', async () => {
    const onSubmit = vi.fn()
    render(<ChatInput onSubmit={onSubmit} disabled={false} />)

    const textarea = screen.getByPlaceholderText('質問を入力してください（Shift+Enterで改行）')
    await userEvent.type(textarea, '変換中')
    textarea.dispatchEvent(
      new KeyboardEvent('keydown', {
        key: 'Enter',
        bubbles: true,
        cancelable: true,
        isComposing: true,
      }),
    )

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('disabled時は入力・送信ボタンが無効化される', () => {
    const onSubmit = vi.fn()
    render(<ChatInput onSubmit={onSubmit} disabled={true} />)

    expect(
      screen.getByPlaceholderText('質問を入力してください（Shift+Enterで改行）'),
    ).toBeDisabled()
    expect(screen.getByRole('button', { name: '送信' })).toBeDisabled()
  })
})
