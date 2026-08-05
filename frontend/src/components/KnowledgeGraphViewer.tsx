import { useEffect, useMemo, useRef, useState } from "react";
import { Graph, GraphEvent } from "@antv/g6";
import type { GraphEdge, GraphNode } from "../types/agent";
import { nodeColor, nodeRadius } from "../utils/graph";
import { GraphControls } from "./GraphControls";
import { EntityDetailPanel } from "./EntityDetailPanel";

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  docOptions: { id: string; name: string }[];
  selectedTypes: string[];
  onToggleType: (t: string) => void;
  selectedDocId: string;
  onSelectDoc: (id: string) => void;
  onInjectContext: (text: string) => void;
  docById: Map<string, string>; // docId -> filename
  allTypes: string[];
}

// G6 基于 Canvas 渲染，承载量远超原 SVG 方案（评审 G3 的 200 上限是针对 React+SVG 的）。
// 此处仅在节点数极大时给出筛选提示，避免无意义的全量绘制。
const TOO_MANY_NODES = 1000;

export function KnowledgeGraphViewer(props: Props) {
  const {
    nodes,
    edges,
    docOptions,
    selectedTypes,
    onToggleType,
    selectedDocId,
    onSelectDoc,
    onInjectContext,
    docById,
    allTypes,
  } = props;

  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<Graph | null>(null);
  const [zoom, setZoom] = useState(1);
  const [detailNode, setDetailNode] = useState<GraphNode | null>(null);
  const [hover, setHover] = useState<{ text: string; x: number; y: number } | null>(null);

  const nodeById = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);

  // 组装 G6 数据（节点/边均透传业务字段到 data，供样式回调使用）。
  const data = useMemo(
    () => ({
      nodes: nodes.map((n) => ({ id: n.id, data: { label: n.label, type: n.type } })),
      edges: edges.map((e) => ({
        source: e.source,
        target: e.target,
        data: { label: e.label ?? "" },
      })),
    }),
    [nodes, edges],
  );

  // 初始化 G6 实例（仅一次）。力导向与渲染全部交给 G6，辐射布局开箱即用。
  useEffect(() => {
    if (!containerRef.current) return;
    const graph = new Graph({
      container: containerRef.current,
      autoResize: true,
      animation: false,
      layout: {
        type: "radial",
        unitRadius: 130,
        linkDistance: 130,
        preventOverlap: true,
        nodeSize: 44,
        strictRadial: false,
        maxIteration: 1000,
      },
      node: {
        type: "circle",
        style: {
          size: (d: any) => nodeRadius(d.data?.type ?? "") * 2,
          fill: (d: any) => nodeColor(d.data?.type ?? ""),
          stroke: "#ffffff",
          lineWidth: 1.5,
          labelText: (d: any) => d.data?.label ?? "",
          labelFill: "#1f2937",
          labelFontSize: 11,
          labelPlacement: "bottom",
          labelMaxWidth: 110,
        },
        state: {
          selected: { stroke: "#2563eb", lineWidth: 3 },
          active: { stroke: "#2563eb", lineWidth: 2 },
          inactive: { fillOpacity: 0.15, labelOpacity: 0.15 },
        },
      },
      edge: {
        type: "line",
        style: {
          stroke: "#9ca3af",
          lineWidth: 1.5,
          endArrow: true,
          labelText: (d: any) => d.data?.label ?? "",
          labelFontSize: 10,
          labelFill: "#6b7280",
          labelBackground: true,
          labelBackgroundFill: "#ffffff",
          labelBackgroundOpacity: 0.7,
        },
        state: {
          selected: { stroke: "#2563eb" },
          active: { stroke: "#2563eb", strokeOpacity: 1 },
          inactive: { strokeOpacity: 0.08, labelOpacity: 0, endArrow: false },
        },
      },
      behaviors: [
        "drag-canvas",
        "zoom-canvas",
        { type: "drag-element" },
        { type: "hover-activate", degree: 1 },
        { type: "click-select", multiple: false, degree: 1 },
      ],
    });
    graphRef.current = graph;
    // 用户用滚轮缩放后，同步比例显示。
    graph.on(GraphEvent.AFTER_TRANSFORM, () => setZoom(graph.getZoom()));
    return () => {
      graph.destroy();
      graphRef.current = null;
    };
  }, []);

  // 数据变化 → 重新渲染并自适应视图。
  useEffect(() => {
    const graph = graphRef.current;
    if (!graph) return;
    graph.setData(data);
    void graph.render().then(() => graph.fitView({ when: "always", direction: "both" }));
  }, [data]);

  // 交互事件：双击打开详情、悬停 tooltip、点击记录选中（高亮由 click-select 行为负责）。
  useEffect(() => {
    const graph = graphRef.current;
    if (!graph) return;
    const onDblClick = (e: any) => {
      const id = e.target?.id;
      if (id && nodeById.has(id)) setDetailNode(nodeById.get(id)!);
    };
    const onPointerOver = (e: any) => {
      const id = e.target?.id;
      if (id && nodeById.has(id)) {
        const n = nodeById.get(id)!;
        const rect = containerRef.current?.getBoundingClientRect();
        const x = rect ? e.clientX - rect.left : e.clientX;
        const y = rect ? e.clientY - rect.top : e.clientY;
        setHover({ text: `${n.label}（${n.type}）`, x, y });
      }
    };
    const onPointerOut = () => setHover(null);
    graph.on("node:dblclick", onDblClick);
    graph.on("node:pointerover", onPointerOver);
    graph.on("node:pointerout", onPointerOut);
    return () => {
      graph.off("node:dblclick", onDblClick);
      graph.off("node:pointerover", onPointerOver);
      graph.off("node:pointerout", onPointerOut);
    };
  }, [nodeById]);

  const zoomBy = (factor: number) => {
    const graph = graphRef.current;
    if (!graph) return;
    void graph.zoomTo(graph.getZoom() * factor, { duration: 200 }).then(() => setZoom(graph.getZoom()));
  };

  const resetView = () => {
    const graph = graphRef.current;
    if (!graph) return;
    void graph.fitView({ when: "always", direction: "both" }).then(() => setZoom(graph.getZoom()));
  };

  if (nodes.length === 0) {
    return (
      <div className="graph-viewer graph-empty">
        <div className="muted">暂无图谱数据，上传并解析文档后即可查看知识图谱。</div>
      </div>
    );
  }

  const tooMany = nodes.length > TOO_MANY_NODES;
  const detailDocName = detailNode ? docById.get(detailNode.sourceDocId) : undefined;

  return (
    <div className="graph-viewer">
      <GraphControls
        scale={zoom}
        onZoomIn={() => zoomBy(1.2)}
        onZoomOut={() => zoomBy(1 / 1.2)}
        onReset={resetView}
        types={allTypes}
        selectedTypes={selectedTypes}
        onToggleType={onToggleType}
        docOptions={docOptions}
        selectedDocId={selectedDocId}
        onSelectDoc={onSelectDoc}
      />

      {tooMany && (
        <div className="graph-toomany">
          节点较多（{nodes.length}），建议按类型或文档筛选以提升交互流畅度。
        </div>
      )}

      <div ref={containerRef} className="graph-canvas graph-canvas-g6" />

      {hover && (
        <div className="graph-tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          {hover.text}
        </div>
      )}

      {detailNode && (
        <EntityDetailPanel
          node={detailNode}
          sourceDocName={detailDocName}
          onClose={() => setDetailNode(null)}
          onInject={onInjectContext}
        />
      )}
    </div>
  );
}
