import type { GraphEdge, GraphNode } from "../types/agent";

// ===== 03 · 图谱工具 =====
// 力导向布局与渲染交给 AntV G6（v5，`@antv/g6`）完成，本文件只保留与可视化无关的纯函数：
// 类型判定、着色、半径、邻接表、实体摘要。避免重复造轮子。

export const CORE_TYPES = new Set([
  "person",
  "people",
  "organization",
  "org",
  "institution",
  "company",
  "product",
  "fund",
  "bank",
  "enterprise",
]);

export function isCoreType(type: string): boolean {
  return CORE_TYPES.has(type.toLowerCase());
}

// 节点着色：核心实体橙、地域/属性绿、其余灰（需求 §FR4 / 技术 §6.1）。
export function nodeColor(type: string): string {
  const t = type.toLowerCase();
  if (isCoreType(t)) return "#f59e0b";
  if (t === "region" || t === "location" || t === "area" || t === "attribute" || t === "value")
    return "#22c55e";
  return "#6b7280";
}

export function nodeRadius(type: string): number {
  return isCoreType(type) ? 20 : 14;
}

// 邻接表：用于点击高亮一度邻居（G6 的 click-select / hover-activate 行为会在内部处理高亮，
// 这里保留以支撑「带入对话」时附带邻居上下文等潜在需求）。
export function buildAdjacency(edges: GraphEdge[]): Map<string, Set<string>> {
  const adj = new Map<string, Set<string>>();
  const add = (a: string, b: string) => {
    if (!adj.has(a)) adj.set(a, new Set());
    adj.get(a)!.add(b);
  };
  for (const e of edges) {
    add(e.source, e.target);
    add(e.target, e.source);
  }
  return adj;
}

// 构建实体摘要文本（用于「带入对话」联动）。
export function buildEntitySummary(node: GraphNode, sourceDocName?: string): string {
  const lines = [`【实体】${node.label}（类型：${node.type}）`];
  if (sourceDocName) lines.push(`来源文档：${sourceDocName}`);
  if (node.properties) {
    for (const [k, v] of Object.entries(node.properties)) lines.push(`${k}：${v}`);
  }
  return lines.join("\n");
}
