// api/main.py（FastAPI）への疎通に使う共通クライアント設定。
// 開発時はViteの.envでVITE_API_BASE_URLを上書きできるが、未設定時はuvicornのデフォルト起動先を使う。
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export interface HealthResponse {
  status: string
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`)
  if (!response.ok) {
    throw new Error(`ヘルスチェックに失敗しました (status: ${response.status})`)
  }
  return response.json() as Promise<HealthResponse>
}
