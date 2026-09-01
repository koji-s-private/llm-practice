// api/main.py（FastAPI）のファイル管理関連エンドポイント
// (GET /api/files, POST /api/files/upload, DELETE /api/files/{name}) を呼び出す薄いクライアント。
import { API_BASE_URL } from '@/lib/api'

export interface IndexedFile {
  name: string
  chunk_count: number
}

export async function fetchIndexedFiles(): Promise<IndexedFile[]> {
  const response = await fetch(`${API_BASE_URL}/api/files`)
  if (!response.ok) {
    throw new Error(`ファイル一覧の取得に失敗しました (status: ${response.status})`)
  }
  const data = (await response.json()) as { files: IndexedFile[] }
  return data.files
}

export interface UploadedFile {
  original_name: string
  saved_name: string
  renamed: boolean
}

export async function uploadFiles(files: File[]): Promise<UploadedFile[]> {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }
  const response = await fetch(`${API_BASE_URL}/api/files/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    throw new Error(`ファイルのアップロードに失敗しました (status: ${response.status})`)
  }
  const data = (await response.json()) as { uploaded: UploadedFile[] }
  return data.uploaded
}

export async function deleteIndexedFile(name: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/files/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  })
  if (!response.ok) {
    throw new Error(`ファイルの削除に失敗しました (status: ${response.status})`)
  }
}
