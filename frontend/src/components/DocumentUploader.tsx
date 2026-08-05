import { useRef, useState, type DragEvent } from "react";

interface Props {
  onUpload: (files: FileList) => void;
  uploading: boolean;
}

// 文档上传区：点击选择 + 拖拽（需求 FR1）。
export function DocumentUploader({ onUpload, uploading }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.length) onUpload(e.dataTransfer.files);
  };

  return (
    <div
      className={`doc-uploader${dragOver ? " dragover" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label="上传个人文档：点击选择或拖拽文件到此处"
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".pdf,.docx,.txt,.md"
        hidden
        onChange={(e) => {
          if (e.target.files?.length) onUpload(e.target.files);
          e.target.value = "";
        }}
      />
      <div className="doc-uploader-icon" aria-hidden>
        ⬆
      </div>
      <div className="doc-uploader-text">{uploading ? "上传中…" : "点击或拖拽上传文档"}</div>
      <div className="doc-uploader-hint">支持 PDF / DOCX / TXT / MD，单文件 ≤20MB，最多 10 个</div>
    </div>
  );
}
