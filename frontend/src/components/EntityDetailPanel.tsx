import { Fragment } from "react";
import type { GraphNode } from "../types/agent";
import { buildEntitySummary } from "../utils/graph";

interface Props {
  node: GraphNode;
  sourceDocName?: string;
  onClose: () => void;
  onInject: (text: string) => void;
}

// 实体详情侧板：双击节点打开，展示属性 + 带入对话（需求 FR5, FR7）。
export function EntityDetailPanel({ node, sourceDocName, onClose, onInject }: Props) {
  const summary = buildEntitySummary(node, sourceDocName);
  return (
    <aside className="entity-detail" aria-label="实体详情">
      <div className="entity-detail-head">
        <h3 title={node.label}>{node.label}</h3>
        <button className="btn entity-detail-close" onClick={onClose} aria-label="关闭详情">
          ✕
        </button>
      </div>
      <dl className="entity-detail-body">
        <dt>类型</dt>
        <dd>{node.type}</dd>
        {sourceDocName && (
          <>
            <dt>来源文档</dt>
            <dd>{sourceDocName}</dd>
          </>
        )}
        {node.properties &&
          Object.entries(node.properties).map(([k, v]) => (
            <Fragment key={k}>
              <dt>{k}</dt>
              <dd>{String(v)}</dd>
            </Fragment>
          ))}
      </dl>
      <button className="btn btn-primary entity-detail-inject" onClick={() => onInject(summary)}>
        带入对话
      </button>
    </aside>
  );
}
