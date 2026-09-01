import { render } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { FileDropzone } from '@/components/files/FileDropzone'

function getFileInput(container: HTMLElement): HTMLInputElement {
  const input = container.querySelector('input[type="file"]')
  if (!input) {
    throw new Error('file input が見つかりません')
  }
  return input as HTMLInputElement
}

describe('FileDropzone', () => {
  it('ファイルを選択するとonDropが呼ばれる', async () => {
    const user = userEvent.setup()
    const onDrop = vi.fn()
    const { container } = render(<FileDropzone onDrop={onDrop} disabled={false} />)

    const file = new File(['content'], 'report.txt', { type: 'text/plain' })
    await user.upload(getFileInput(container), file)

    expect(onDrop).toHaveBeenCalledTimes(1)
    expect(onDrop.mock.calls[0][0]).toEqual([file])
  })

  it('複数ファイルを同時に選択できる', async () => {
    const user = userEvent.setup()
    const onDrop = vi.fn()
    const { container } = render(<FileDropzone onDrop={onDrop} disabled={false} />)

    const fileA = new File(['a'], 'a.txt', { type: 'text/plain' })
    const fileB = new File(['b'], 'b.txt', { type: 'text/plain' })
    await user.upload(getFileInput(container), [fileA, fileB])

    expect(onDrop).toHaveBeenCalledTimes(1)
    expect(onDrop.mock.calls[0][0]).toEqual([fileA, fileB])
  })

  it('disabled時はファイルを選択してもonDropが呼ばれない', async () => {
    const user = userEvent.setup()
    const onDrop = vi.fn()
    const { container } = render(<FileDropzone onDrop={onDrop} disabled={true} />)

    const file = new File(['content'], 'report.txt', { type: 'text/plain' })
    await user.upload(getFileInput(container), file)

    expect(onDrop).not.toHaveBeenCalled()
  })
})
