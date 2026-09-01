import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { FileListItem } from '@/components/files/FileListItem'

describe('FileListItem', () => {
  it('ファイル名とチャンク数を表示する', () => {
    render(
      <FileListItem
        file={{ name: 'report.pdf', chunk_count: 5 }}
        onDelete={vi.fn()}
        isDeleting={false}
      />,
    )

    expect(screen.getByText(/report\.pdf/)).toBeInTheDocument()
    expect(screen.getByText('5チャンク')).toBeInTheDocument()
  })

  it('削除ボタンを押すと確認表示になり、確認前はonDeleteが呼ばれない', async () => {
    const user = userEvent.setup()
    const onDelete = vi.fn()
    render(
      <FileListItem
        file={{ name: 'report.pdf', chunk_count: 5 }}
        onDelete={onDelete}
        isDeleting={false}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'report.pdf を削除' }))

    expect(screen.getByText(/この操作は取り消せません/)).toBeInTheDocument()
    expect(onDelete).not.toHaveBeenCalled()
  })

  it('確認後に「削除する」を押すとonDeleteが呼ばれる', async () => {
    const user = userEvent.setup()
    const onDelete = vi.fn()
    render(
      <FileListItem
        file={{ name: 'report.pdf', chunk_count: 5 }}
        onDelete={onDelete}
        isDeleting={false}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'report.pdf を削除' }))
    await user.click(screen.getByRole('button', { name: '削除する' }))

    expect(onDelete).toHaveBeenCalledTimes(1)
  })

  it('確認後に「キャンセル」を押すと確認表示が閉じる', async () => {
    const user = userEvent.setup()
    const onDelete = vi.fn()
    render(
      <FileListItem
        file={{ name: 'report.pdf', chunk_count: 5 }}
        onDelete={onDelete}
        isDeleting={false}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'report.pdf を削除' }))
    await user.click(screen.getByRole('button', { name: 'キャンセル' }))

    expect(screen.queryByText(/この操作は取り消せません/)).not.toBeInTheDocument()
    expect(onDelete).not.toHaveBeenCalled()
  })

  it('isDeleting中は削除ボタンが無効化される', async () => {
    const user = userEvent.setup()
    render(
      <FileListItem
        file={{ name: 'report.pdf', chunk_count: 5 }}
        onDelete={vi.fn()}
        isDeleting={true}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'report.pdf を削除' }))

    expect(screen.getByRole('button', { name: '削除中...' })).toBeDisabled()
  })
})
