import { useQuery } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { fetchHealth } from '@/lib/api'

function App() {
  const health = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    retry: false,
  })

  return (
    <main className="mx-auto flex min-h-svh max-w-2xl flex-col items-center justify-center gap-4 p-8 text-center">
      <h1 className="text-3xl font-semibold">Doclore</h1>
      <p className="text-muted-foreground">
        フロントエンド基盤（Vite + React + TypeScript）の雛形です。
      </p>
      <p data-testid="api-status" className="text-sm">
        {health.isPending && 'API疎通確認中...'}
        {health.isError && `API疎通確認に失敗しました: ${health.error.message}`}
        {health.isSuccess && `API疎通確認: ${health.data.status}`}
      </p>
      <Button onClick={() => health.refetch()}>再確認する</Button>
    </main>
  )
}

export default App
