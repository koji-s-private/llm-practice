import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

vi.stubGlobal(
  'fetch',
  vi.fn(() => Promise.resolve(new Response(JSON.stringify({ thread_id: 'abc12345' })))),
)

function renderWithQueryClient() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>,
  )
}

describe('App', () => {
  it('見出し「Doclore」を表示する', () => {
    renderWithQueryClient()
    expect(screen.getByRole('heading', { name: 'Doclore' })).toBeInTheDocument()
  })
})
