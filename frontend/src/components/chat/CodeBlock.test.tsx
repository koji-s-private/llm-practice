import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CodeBlock } from '@/components/chat/CodeBlock'

describe('CodeBlock', () => {
  it('登録済み言語のコードをシンタックスハイライト付きでレンダリングする', () => {
    const { container } = render(<CodeBlock language="python" code={'print("hello")'} />)

    expect(container.querySelector('code.language-python')).toBeInTheDocument()
    // Prismのトークン用spanが生成されていればハイライトが適用されている
    expect(container.querySelectorAll('span.token').length).toBeGreaterThan(0)
    expect(container.textContent).toContain('print')
    expect(container.textContent).toContain('hello')
  })

  it('Prismに存在しない言語を指定してもエラーにならずプレーンテキストで表示する', () => {
    const { container } = render(
      <CodeBlock language="not-a-real-language" code={'plain text body'} />,
    )

    expect(container.querySelector('code.language-not-a-real-language')).toBeInTheDocument()
    expect(container.querySelectorAll('span.token').length).toBe(0)
    expect(container.textContent).toContain('plain text body')
  })

  it('個別登録した言語一覧に含まれない言語はハイライトされずプレーンテキストで表示する', () => {
    // rubyはPrism自体には存在するが、バンドルサイズ削減のためCodeBlockのregisterLanguage対象外。
    // 対象外言語がハイライトされてしまうと、意図した言語だけを個別登録するバンドル削減方針が崩れる。
    const { container } = render(<CodeBlock language="ruby" code={'puts "hi"'} />)

    expect(container.querySelectorAll('span.token').length).toBe(0)
    expect(container.textContent).toContain('puts "hi"')
  })

  it('空文字のコードを渡しても例外にならない', () => {
    const { container } = render(<CodeBlock language="python" code="" />)

    expect(container.querySelector('code.language-python')).toBeInTheDocument()
  })
})
