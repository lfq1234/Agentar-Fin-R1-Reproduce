import { useEffect, useState } from "react";
import type { PersonalDocument } from "../types/agent";
import { usePersonalDocs } from "../hooks/usePersonalDocs";
import { useKnowledgeGraph } from "../hooks/useKnowledgeGraph";
import { DocumentUploader } from "./DocumentUploader";
import { DocumentList } from "./DocumentList";
import { KnowledgeGraphViewer } from "./KnowledgeGraphViewer";

interface Props {
  onClose: () => void;
  onInjectContext: (text: string) => void;
  onAskWithDocument: (doc: PersonalDocument) => void;
}

type Tab = "docs" | "graph";

// 个人文档面板：由 02 占位壳扩展为「文档 / 知识图谱」双标签（评审 G4 边界：外壳归 02，内容归 03）。
export function PersonalDocsPanel({ onClose, onInjectContext, onAskWithDocument }: Props) {
  const [tab, setTab] = useState<Tab>("docs");
  const kg = useKnowledgeGraph();
  const docs = usePersonalDocs(() => kg.fetchGraph());

  // 切换到图谱标签时拉取最新图谱（文档解析完成后自动刷新）。
  useEffect(() => {
    if (tab === "graph") kg.fetchGraph();
  }, [tab, kg]);

  const docById = new Map(docs.documents.map((d) => [d.id, d.filename]));
  const docOptions = docs.documents.map((d) => ({ id: d.id, name: d.filename }));

  return (
    <div className="personal-docs-panel">
      <div className="personal-docs-head">
        <h2>个人文档</h2>
        <button className="btn" onClick={onClose}>
          关闭
        </button>
      </div>

      <div className="pd-tabs" role="tablist" aria-label="个人文档视图切换">
        <button
          className={`pd-tab${tab === "docs" ? " active" : ""}`}
          role="tab"
          aria-selected={tab === "docs"}
          onClick={() => setTab("docs")}
        >
          文档
        </button>
        <button
          className={`pd-tab${tab === "graph" ? " active" : ""}`}
          role="tab"
          aria-selected={tab === "graph"}
          onClick={() => setTab("graph")}
        >
          知识图谱
        </button>
      </div>

      {docs.error && <div className="pd-error">{docs.error}</div>}

      {tab === "docs" ? (
        <div className="pd-docs-view">
          <DocumentUploader onUpload={docs.uploadFiles} uploading={docs.uploading} />
          <DocumentList documents={docs.documents} onDelete={docs.deleteDoc} onAsk={onAskWithDocument} />
        </div>
      ) : (
        <div className="pd-graph-view">
          {kg.loading && <div className="pd-loading muted">图谱加载中…</div>}
          {kg.error && <div className="pd-error">{kg.error}</div>}
          {!kg.loading && !kg.error && (
            <KnowledgeGraphViewer
              nodes={kg.filteredNodes}
              edges={kg.filteredEdges}
              docOptions={docOptions}
              selectedTypes={kg.selectedTypes}
              onToggleType={(t) =>
                kg.setSelectedTypes((prev) => (prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]))
              }
              selectedDocId={kg.selectedDocId}
              onSelectDoc={kg.setSelectedDocId}
              onInjectContext={onInjectContext}
              docById={docById}
              allTypes={kg.allTypes}
            />
          )}
        </div>
      )}
    </div>
  );
}
