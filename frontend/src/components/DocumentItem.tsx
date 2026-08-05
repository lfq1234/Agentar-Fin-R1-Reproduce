import type { DocStatus, PersonalDocument } from "../types/agent";

interface Props {
  doc: PersonalDocument;
  onDelete: (id: string) => void;
  onAsk: (doc: PersonalDocument) => void;
}

const STATUS_LABEL: Record<DocStatus, string> = {
  pending: "等待解析",
  parsing: "解析中",
  done: "已解析",
  error: "解析失败",
};

// 单条文档项：状态、大小、时间、删除、基于此文档提问（需求 FR2, FR7）。
export function DocumentItem({ doc, onDelete, onAsk }: Props) {
  return (
    <li className="doc-item">
      <div className="doc-item-main">
        <span className="doc-item-name" title={doc.filename}>
          {doc.filename}
        </span>
        <span className={`doc-status doc-status-${doc.status}`}>
          {doc.status === "parsing" && <span className="doc-spinner" aria-hidden />}
          {STATUS_LABEL[doc.status]}
        </span>
      </div>
      <div className="doc-item-meta">
        <span>{(doc.size / 1024).toFixed(1)} KB</span>
        <span>{new Date(doc.uploadedAt).toLocaleString("zh-CN")}</span>
      </div>
      {doc.status === "error" && doc.error && (
        <div className="doc-item-error">⚠ {doc.error}</div>
      )}
      <div className="doc-item-actions">
        {doc.status === "done" && (
          <button className="btn btn-primary doc-item-btn" onClick={() => onAsk(doc)}>
            基于此文档提问
          </button>
        )}
        <button
          className="btn doc-item-btn"
          onClick={() => {
            if (confirm(`确认删除《${doc.filename}》？`)) onDelete(doc.id);
          }}
          aria-label={`删除 ${doc.filename}`}
        >
          删除
        </button>
      </div>
    </li>
  );
}
