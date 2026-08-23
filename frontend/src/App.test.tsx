import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

vi.stubGlobal(
  'fetch',
  vi.fn(() => Promise.resolve(new Response(JSON.stringify({ status: 'ok' })))),
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

  it('API疎通確認結果を表示する', async () => {
    renderWithQueryClient()
    expect(await screen.findByText(/API疎通確認: ok/)).toBeInTheDocument()
  })
})
