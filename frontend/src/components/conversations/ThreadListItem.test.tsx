import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ThreadListItem } from '@/components/conversations/ThreadListItem'
import type { ThreadSummary } from '@/lib/conversations'

const thread: ThreadSummary = {
  thread_id: 'thread-a',
  created_at: '2026-01-01T09:30:00',
  first_question: '経費精算について教えてください',
  count: 3,
  title: null,
}

function renderItem(overrides: Partial<Parameters<typeof ThreadListItem>[0]> = {}) {
  return render(
    <ThreadListItem
      thread={thread}
      isActive={false}
      onSelect={vi.fn()}
      onDelete={vi.fn()}
      isDeleting={false}
      onSaveTitle={vi.fn()}
      isSavingTitle={false}
      {...overrides}
    />,
  )
}

describe('ThreadListItem', () => {
  it('作成日時・冒頭質問のスニペット・保存件数を表示する', () => {
    renderItem()

    expect(screen.getByText(/2026-01-01 09:30/)).toBeInTheDocument()
    expect(screen.getByText(/経費精算について教えてください/)).toBeInTheDocument()
    expect(screen.getByText(/3件/)).toBeInTheDocument()
  })

  it('タイトルが設定済みの場合、タイトルを主表示にする', () => {
    renderItem({ thread: { ...thread, title: '経費精算スレッド' } })

    expect(screen.getByText(/📌 経費精算スレッド/)).toBeInTheDocument()
  })

  it('クリックするとonSelectが呼ばれる', async () => {
    const user = userEvent.setup()
    const onSelect = vi.fn()
    renderItem({ onSelect })

    await user.click(screen.getByText(/経費精算について教えてください/))

    expect(onSelect).toHaveBeenCalledTimes(1)
  })

  it('アクティブなスレッドは選択ボタンが無効化される', () => {
    renderItem({ isActive: true })

    expect(screen.getByRole('button', { name: /経費精算について教えてください/ })).toBeDisabled()
  })

  it('削除ボタンを押すと確認表示になり、確認前はonDeleteが呼ばれない', async () => {
    const user = userEvent.setup()
    const onDelete = vi.fn()
    renderItem({ onDelete })

    await user.click(screen.getByRole('button', { name: 'thread-a を削除' }))

    expect(screen.getByText(/この操作は取り消せません/)).toBeInTheDocument()
    expect(onDelete).not.toHaveBeenCalled()
  })

  it('確認後に「削除する」を押すとonDeleteが呼ばれる', async () => {
    const user = userEvent.setup()
    const onDelete = vi.fn()
    renderItem({ onDelete })

    await user.click(screen.getByRole('button', { name: 'thread-a を削除' }))
    await user.click(screen.getByRole('button', { name: '削除する' }))

    expect(onDelete).toHaveBeenCalledTimes(1)
  })

  it('確認後に「キャンセル」を押すと確認表示が閉じる', async () => {
    const user = userEvent.setup()
    renderItem()

    await user.click(screen.getByRole('button', { name: 'thread-a を削除' }))
    await user.click(screen.getByRole('button', { name: 'キャンセル' }))

    expect(screen.queryByText(/この操作は取り消せません/)).not.toBeInTheDocument()
  })

  it('タイトル編集を保存するとonSaveTitleが入力値付きで呼ばれる', async () => {
    const user = userEvent.setup()
    const onSaveTitle = vi.fn()
    renderItem({ onSaveTitle })

    await user.click(screen.getByRole('button', { name: '✏️ タイトル編集' }))
    const input = screen.getByPlaceholderText('例: 経費精算の質問')
    await user.type(input, '新しいタイトル')
    await user.click(screen.getByRole('button', { name: '💾 保存' }))

    expect(onSaveTitle).toHaveBeenCalledWith('新しいタイトル')
  })

  it('タイトル編集をキャンセルするとonSaveTitleが呼ばれない', async () => {
    const user = userEvent.setup()
    const onSaveTitle = vi.fn()
    renderItem({ onSaveTitle })

    await user.click(screen.getByRole('button', { name: '✏️ タイトル編集' }))
    await user.click(screen.getByRole('button', { name: 'キャンセル' }))

    expect(onSaveTitle).not.toHaveBeenCalled()
    expect(screen.queryByPlaceholderText('例: 経費精算の質問')).not.toBeInTheDocument()
  })

  it('isDeleting中は削除ボタンが無効化される', async () => {
    const user = userEvent.setup()
    renderItem({ isDeleting: true })

    await user.click(screen.getByRole('button', { name: 'thread-a を削除' }))

    expect(screen.getByRole('button', { name: '削除中...' })).toBeDisabled()
  })
})
