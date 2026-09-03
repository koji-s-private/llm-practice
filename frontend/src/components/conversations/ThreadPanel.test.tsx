import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ThreadPanel } from '@/components/conversations/ThreadPanel'
import * as conversationsApi from '@/lib/conversations'
import type { ThreadSummary } from '@/lib/conversations'

function renderPanel(overrides: Partial<Parameters<typeof ThreadPanel>[0]> = {}) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <ThreadPanel
        activeThreadId={undefined}
        disabled={false}
        onSelectThread={vi.fn()}
        onNewThread={vi.fn()}
        onThreadDeleted={vi.fn()}
        {...overrides}
      />
    </QueryClientProvider>,
  )
}

const threadA: ThreadSummary = {
  thread_id: 'thread-a',
  created_at: '2026-01-01T09:00:00',
  first_question: '経費精算について',
  count: 2,
  title: null,
}
const threadB: ThreadSummary = {
  thread_id: 'thread-b',
  created_at: '2026-01-02T09:00:00',
  first_question: '有給休暇の申請方法',
  count: 1,
  title: null,
}

describe('ThreadPanel', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('スレッド一覧を表示する', async () => {
    vi.spyOn(conversationsApi, 'fetchThreads').mockResolvedValue([threadA, threadB])

    renderPanel()

    expect(await screen.findByText(/経費精算について/)).toBeInTheDocument()
    expect(screen.getByText(/有給休暇の申請方法/)).toBeInTheDocument()
  })

  it('スレッドが0件のとき案内文を表示する', async () => {
    vi.spyOn(conversationsApi, 'fetchThreads').mockResolvedValue([])

    renderPanel()

    expect(await screen.findByText('まだ保存された会話スレッドはありません。')).toBeInTheDocument()
  })

  it('一覧取得に失敗した場合エラーメッセージを表示する', async () => {
    vi.spyOn(conversationsApi, 'fetchThreads').mockRejectedValue(new Error('取得失敗'))

    renderPanel()

    expect(await screen.findByText('取得失敗')).toBeInTheDocument()
  })

  it('検索キーワードで絞り込める', async () => {
    const user = userEvent.setup()
    vi.spyOn(conversationsApi, 'fetchThreads').mockResolvedValue([threadA, threadB])

    renderPanel()
    await screen.findByText(/経費精算について/)

    await user.type(screen.getByPlaceholderText('🔍 質問内容で絞り込み...'), '有給')

    expect(screen.queryByText(/経費精算について/)).not.toBeInTheDocument()
    expect(screen.getByText(/有給休暇の申請方法/)).toBeInTheDocument()
  })

  it('該当するスレッドが無い場合は案内文を表示する', async () => {
    const user = userEvent.setup()
    vi.spyOn(conversationsApi, 'fetchThreads').mockResolvedValue([threadA])

    renderPanel()
    await screen.findByText(/経費精算について/)

    await user.type(screen.getByPlaceholderText('🔍 質問内容で絞り込み...'), '存在しないキーワード')

    expect(
      await screen.findByText('該当する会話スレッドが見つかりませんでした。'),
    ).toBeInTheDocument()
  })

  it('スレッドをクリックするとonSelectThreadが呼ばれる', async () => {
    const user = userEvent.setup()
    const onSelectThread = vi.fn()
    vi.spyOn(conversationsApi, 'fetchThreads').mockResolvedValue([threadA])

    renderPanel({ onSelectThread })
    await user.click(await screen.findByText(/経費精算について/))

    expect(onSelectThread).toHaveBeenCalledWith('thread-a')
  })

  it('「新しい会話」ボタンを押すとonNewThreadが呼ばれる', async () => {
    const user = userEvent.setup()
    const onNewThread = vi.fn()
    vi.spyOn(conversationsApi, 'fetchThreads').mockResolvedValue([threadA])

    renderPanel({ onNewThread })
    await screen.findByText(/経費精算について/)
    await user.click(screen.getByRole('button', { name: '🆕 新しい会話' }))

    expect(onNewThread).toHaveBeenCalledTimes(1)
  })

  it('削除確認後、削除が実行され一覧から消え、onThreadDeletedが呼ばれる', async () => {
    const user = userEvent.setup()
    const onThreadDeleted = vi.fn()
    vi.spyOn(conversationsApi, 'fetchThreads')
      .mockResolvedValueOnce([threadA])
      .mockResolvedValueOnce([])
    const deleteThread = vi.spyOn(conversationsApi, 'deleteThread').mockResolvedValue(undefined)

    renderPanel({ onThreadDeleted })
    await screen.findByText(/経費精算について/)

    await user.click(screen.getByRole('button', { name: 'thread-a を削除' }))
    await user.click(screen.getByRole('button', { name: '削除する' }))

    await waitFor(() => expect(deleteThread).toHaveBeenCalledWith('thread-a', expect.anything()))
    await waitFor(() => expect(onThreadDeleted).toHaveBeenCalledWith('thread-a'))
    await waitFor(() =>
      expect(screen.getByText('まだ保存された会話スレッドはありません。')).toBeInTheDocument(),
    )
  })

  it('タイトル編集を保存すると一覧が再取得される', async () => {
    const user = userEvent.setup()
    vi.spyOn(conversationsApi, 'fetchThreads')
      .mockResolvedValueOnce([threadA])
      .mockResolvedValueOnce([{ ...threadA, title: '新タイトル' }])
    const updateThreadTitle = vi
      .spyOn(conversationsApi, 'updateThreadTitle')
      .mockResolvedValue('新タイトル')

    renderPanel()
    await screen.findByText(/経費精算について/)

    await user.click(screen.getByRole('button', { name: '✏️ タイトル編集' }))
    await user.type(screen.getByPlaceholderText('例: 経費精算の質問'), '新タイトル')
    await user.click(screen.getByRole('button', { name: '💾 保存' }))

    await waitFor(() => expect(updateThreadTitle).toHaveBeenCalledWith('thread-a', '新タイトル'))
    expect(await screen.findByText(/📌 新タイトル/)).toBeInTheDocument()
  })

  it('disabled=trueのとき「新しい会話」ボタンが無効化される', async () => {
    vi.spyOn(conversationsApi, 'fetchThreads').mockResolvedValue([threadA])

    renderPanel({ disabled: true })
    await screen.findByText(/経費精算について/)

    expect(screen.getByRole('button', { name: '🆕 新しい会話' })).toBeDisabled()
  })

  it('activeThreadIdに一致するスレッドの選択ボタンが無効化される', async () => {
    vi.spyOn(conversationsApi, 'fetchThreads').mockResolvedValue([threadA])

    renderPanel({ activeThreadId: 'thread-a' })

    expect(await screen.findByRole('button', { name: /経費精算について/ })).toBeDisabled()
  })
})
