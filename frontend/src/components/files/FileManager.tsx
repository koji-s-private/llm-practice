import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { FileDropzone } from '@/components/files/FileDropzone'
import { FileListItem } from '@/components/files/FileListItem'
import { deleteIndexedFile, fetchIndexedFiles, uploadFiles } from '@/lib/files'

const INDEXED_FILES_QUERY_KEY = ['indexedFiles']

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback
}

export function FileManager() {
  const queryClient = useQueryClient()
  const [uploadWarning, setUploadWarning] = useState<string | null>(null)

  const filesQuery = useQuery({ queryKey: INDEXED_FILES_QUERY_KEY, queryFn: fetchIndexedFiles })

  const uploadMutation = useMutation({
    mutationFn: uploadFiles,
    onSuccess: (uploaded) => {
      const renamed = uploaded.filter((file) => file.renamed)
      setUploadWarning(
        renamed.length > 0
          ? `同名のファイルが既に存在したため、既存ファイルを上書きせず別名で保存しました:\n${renamed
              .map((file) => `- ${file.original_name} → ${file.saved_name}`)
              .join('\n')}`
          : null,
      )
      void queryClient.invalidateQueries({ queryKey: INDEXED_FILES_QUERY_KEY })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteIndexedFile,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: INDEXED_FILES_QUERY_KEY })
    },
  })

  return (
    <div className="flex flex-col gap-3 p-4">
      <h2 className="text-sm font-semibold">ファイル管理</h2>
      <FileDropzone
        disabled={uploadMutation.isPending}
        onDrop={(files) => {
          setUploadWarning(null)
          uploadMutation.mutate(files)
        }}
      />
      {uploadMutation.isPending && (
        <p className="text-muted-foreground text-xs">アップロード中...</p>
      )}
      {uploadMutation.isError && (
        <p className="text-destructive text-xs">
          {errorMessage(uploadMutation.error, 'アップロードに失敗しました')}
        </p>
      )}
      {uploadWarning && (
        <p className="text-muted-foreground border-border rounded-md border p-2 text-xs whitespace-pre-wrap">
          {uploadWarning}
        </p>
      )}

      {filesQuery.isLoading && <p className="text-muted-foreground text-xs">読み込み中...</p>}
      {filesQuery.isError && (
        <p className="text-destructive text-xs">
          {errorMessage(filesQuery.error, 'ファイル一覧の取得に失敗しました')}
        </p>
      )}
      {filesQuery.data?.length === 0 && (
        <p className="text-muted-foreground text-xs">
          インデックス済みのファイルはまだありません。
        </p>
      )}
      {deleteMutation.isError && (
        <p className="text-destructive text-xs">
          {errorMessage(deleteMutation.error, 'ファイルの削除に失敗しました')}
        </p>
      )}
      {filesQuery.data && filesQuery.data.length > 0 && (
        <ul className="flex flex-col gap-2">
          {filesQuery.data.map((file) => (
            <FileListItem
              key={file.name}
              file={file}
              onDelete={() => deleteMutation.mutate(file.name)}
              isDeleting={deleteMutation.isPending && deleteMutation.variables === file.name}
            />
          ))}
        </ul>
      )}
    </div>
  )
}
