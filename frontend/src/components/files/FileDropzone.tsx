import { useDropzone } from 'react-dropzone'
import { cn } from '@/lib/utils'

interface FileDropzoneProps {
  onDrop: (files: File[]) => void
  disabled: boolean
}

export function FileDropzone({ onDrop, disabled }: FileDropzoneProps) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        onDrop(acceptedFiles)
      }
    },
    disabled,
  })

  return (
    <div
      {...getRootProps()}
      className={cn(
        'border-border cursor-pointer rounded-md border border-dashed p-4 text-center text-xs transition-colors',
        isDragActive && 'border-primary bg-muted',
        disabled && 'pointer-events-none opacity-50',
      )}
    >
      <input {...getInputProps()} />
      {isDragActive ? (
        <p>ここにファイルをドロップ</p>
      ) : (
        <p className="text-muted-foreground">
          ファイルをドラッグ&ドロップ、またはクリックして選択（複数選択可）
        </p>
      )}
    </div>
  )
}
