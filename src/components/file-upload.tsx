import { X } from "lucide-react";
import { type ReactNode, type RefObject, useImperativeHandle } from "react";
import { useDropzone } from "react-dropzone";

type FileUploadProps = {
  files: FileList | undefined;
  onFilesChange: (files: FileList | undefined) => void;
  disabled?: boolean;
  children?: ReactNode;
};

export type FileUploadRef = {
  openFileDialog: () => void;
};

export const FileUpload = ({
  files,
  onFilesChange,
  disabled,
  children,
  ref,
}: FileUploadProps & { ref?: RefObject<FileUploadRef | null> }) => {
  const { getRootProps, getInputProps, open, isDragActive } = useDropzone({
    accept: {
      "image/*": [],
      "audio/*": [],
    },
    multiple: true,
    disabled,
    noClick: true,
    noKeyboard: true,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        const dt = new DataTransfer();
        // Add existing files
        if (files) {
          for (const file of Array.from(files)) {
            dt.items.add(file);
          }
        }
        // Add new files
        for (const file of acceptedFiles) {
          dt.items.add(file);
        }
        onFilesChange(dt.files);
      }
    },
  });

  useImperativeHandle(ref, () => ({
    openFileDialog: () => {
      open();
    },
  }));

  const removeFile = (indexToRemove: number) => {
    if (files) {
      const dt = new DataTransfer();
      Array.from(files).forEach((file, index) => {
        if (index !== indexToRemove) {
          dt.items.add(file);
        }
      });
      onFilesChange(dt.files);
    }
  };

  return (
    <div {...getRootProps()} className="relative">
      <input {...getInputProps()} />

      {/* Drag overlay */}
      {isDragActive && (
        <div className="pointer-events-none absolute inset-0 z-50 flex items-center justify-center rounded-lg border-2 border-orange-300 border-dashed bg-orange-300/10">
          <p className="font-medium text-lg text-orange-300">Drop files here</p>
        </div>
      )}

      {/* File Previews */}
      {files && files.length > 0 && (
        <div className="mb-3 flex gap-2">
          {Array.from(files).map((file, index) => (
            <div
              className="group relative overflow-hidden rounded-lg bg-zinc-800/40"
              key={index}
            >
              {file.type.startsWith("image/") ? (
                <img
                  alt={file.name}
                  className="h-20 w-20 object-cover"
                  src={URL.createObjectURL(file)}
                />
              ) : (
                <div className="flex h-20 w-20 items-center justify-center">
                  <span className="max-w-[60px] truncate text-xs text-zinc-400">
                    {file.name}
                  </span>
                </div>
              )}
              <button
                className="absolute top-1 right-1 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-white opacity-0 transition-opacity hover:bg-red-600 group-hover:opacity-100"
                data-umami-event="file_removed"
                onClick={() => removeFile(index)}
                type="button"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Children content (chat input) */}
      {children}
    </div>
  );
};

FileUpload.displayName = "FileUpload";

export type { FileUploadProps };
