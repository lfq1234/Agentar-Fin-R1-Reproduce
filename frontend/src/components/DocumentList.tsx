import type { PersonalDocument } from "../types/agent";
import { DocumentItem } from "./DocumentItem";

interface Props {
  documents: PersonalDocument[];
  onDelete: (id: string) => void;
  onAsk: (doc: PersonalDocument) => void;
}

// 文档列表：按上传时间倒序（需求 FR2）。
export function DocumentList({ documents, onDelete, onAsk }: Props) {
  const sorted = [...documents].sort((a, b) => b.uploadedAt.localeCompare(a.uploadedAt));
  if (sorted.length === 0) {
    return <div className="doc-empty muted">还没有文档，上传后构建你的知识图谱。</div>;
  }
  return (
    <ul className="doc-list">
      {sorted.map((d) => (
        <DocumentItem key={d.id} doc={d} onDelete={onDelete} onAsk={onAsk} />
      ))}
    </ul>
  );
}
