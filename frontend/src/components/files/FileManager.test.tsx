import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { FileManager } from '@/components/files/FileManager'
import * as filesApi from '@/lib/files'

function renderFileManager() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <FileManager />
    </QueryClientProvider>,
  )
}

function getFileInput(container: HTMLElement): HTMLInputElement {
  const input = container.querySelector('input[type="file"]')
  if (!input) {
    throw new Error('file input が見つかりません')
  }
  return input as HTMLInputElement
}

describe('FileManager', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('インデックス済みファイルの一覧を表示する', async () => {
    vi.spyOn(filesApi, 'fetchIndexedFiles').mockResolvedValue([
      { name: 'a.txt', chunk_count: 2 },
      { name: 'b.pdf', chunk_count: 4 },
    ])

    renderFileManager()

    expect(await screen.findByText(/a\.txt/)).toBeInTheDocument()
    expect(screen.getByText(/b\.pdf/)).toBeInTheDocument()
  })

  it('ファイルが0件のとき案内文を表示する', async () => {
    vi.spyOn(filesApi, 'fetchIndexedFiles').mockResolvedValue([])

    renderFileManager()

    expect(
      await screen.findByText('インデックス済みのファイルはまだありません。'),
    ).toBeInTheDocument()
  })

  it('一覧取得に失敗した場合エラーメッセージを表示する', async () => {
    vi.spyOn(filesApi, 'fetchIndexedFiles').mockRejectedValue(new Error('取得失敗'))

    renderFileManager()

    expect(await screen.findByText('取得失敗')).toBeInTheDocument()
  })

  it('アップロード成功後、一覧が再取得される', async () => {
    const user = userEvent.setup()
    const fetchIndexedFiles = vi
      .spyOn(filesApi, 'fetchIndexedFiles')
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([{ name: 'new.txt', chunk_count: 1 }])
    vi.spyOn(filesApi, 'uploadFiles').mockResolvedValue([
      { original_name: 'new.txt', saved_name: 'new.txt', renamed: false },
    ])

    const { container } = renderFileManager()
    await screen.findByText('インデックス済みのファイルはまだありません。')

    const file = new File(['content'], 'new.txt', { type: 'text/plain' })
    await user.upload(getFileInput(container), file)

    expect(await screen.findByText(/new\.txt/)).toBeInTheDocument()
    expect(fetchIndexedFiles).toHaveBeenCalledTimes(2)
  })

  it('同名ファイルがリネームされた場合、警告を表示する', async () => {
    const user = userEvent.setup()
    vi.spyOn(filesApi, 'fetchIndexedFiles').mockResolvedValue([])
    vi.spyOn(filesApi, 'uploadFiles').mockResolvedValue([
      { original_name: 'report.txt', saved_name: 'report (2).txt', renamed: true },
    ])

    const { container } = renderFileManager()
    await screen.findByText('インデックス済みのファイルはまだありません。')

    const file = new File(['content'], 'report.txt', { type: 'text/plain' })
    await user.upload(getFileInput(container), file)

    expect(await screen.findByText(/report\.txt.*report \(2\)\.txt/s)).toBeInTheDocument()
  })

  it('アップロードに失敗した場合エラーメッセージを表示する', async () => {
    const user = userEvent.setup()
    vi.spyOn(filesApi, 'fetchIndexedFiles').mockResolvedValue([])
    vi.spyOn(filesApi, 'uploadFiles').mockRejectedValue(new Error('アップロード失敗'))

    const { container } = renderFileManager()
    await screen.findByText('インデックス済みのファイルはまだありません。')

    const file = new File(['content'], 'report.txt', { type: 'text/plain' })
    await user.upload(getFileInput(container), file)

    expect(await screen.findByText('アップロード失敗')).toBeInTheDocument()
  })

  it('削除確認後、削除が実行され一覧から消える', async () => {
    const user = userEvent.setup()
    vi.spyOn(filesApi, 'fetchIndexedFiles')
      .mockResolvedValueOnce([{ name: 'old.txt', chunk_count: 1 }])
      .mockResolvedValueOnce([])
    const deleteIndexedFile = vi.spyOn(filesApi, 'deleteIndexedFile').mockResolvedValue(undefined)

    renderFileManager()
    await screen.findByText(/old\.txt/)

    await user.click(screen.getByRole('button', { name: 'old.txt を削除' }))
    await user.click(screen.getByRole('button', { name: '削除する' }))

    await waitFor(() =>
      expect(deleteIndexedFile).toHaveBeenCalledWith('old.txt', expect.anything()),
    )
    await waitFor(() =>
      expect(screen.getByText('インデックス済みのファイルはまだありません。')).toBeInTheDocument(),
    )
  })
})
