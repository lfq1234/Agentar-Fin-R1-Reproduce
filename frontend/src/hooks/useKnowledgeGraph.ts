import { useCallback, useEffect, useMemo, useState } from "react";
import { getKnowledgeGraph, ApiError } from "../api/client";
import type { PersonalKnowledgeGraph } from "../types/agent";

// 个人知识图谱：数据获取与筛选（需求 FR4~FR6）。
export function useKnowledgeGraph() {
  const [graph, setGraph] = useState<PersonalKnowledgeGraph | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | "all">("all");

  const fetchGraph = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const g = await getKnowledgeGraph();
      setGraph(g);
    } catch (e) {
      setError((e as ApiError).message ?? "获取图谱失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGraph();
  }, [fetchGraph]);

  // 按类型 + 来源文档筛选节点（技术 §5.2）。
  const filteredNodes = useMemo(() => {
    if (!graph) return [];
    return graph.nodes.filter(
      (n) =>
        (selectedTypes.length === 0 || selectedTypes.includes(n.type)) &&
        (selectedDocId === "all" || n.sourceDocId === selectedDocId),
    );
  }, [graph, selectedTypes, selectedDocId]);

  // 边仅保留两端都可见的，避免悬空（技术 §5.2 / 评审 G3 一致性）。
  const filteredEdges = useMemo(() => {
    if (!graph) return [];
    const ids = new Set(filteredNodes.map((n) => n.id));
    return graph.edges.filter((e) => ids.has(e.source) && ids.has(e.target));
  }, [graph, filteredNodes]);

  const allTypes = useMemo(() => {
    if (!graph) return [];
    return Array.from(new Set(graph.nodes.map((n) => n.type)));
  }, [graph]);

  return {
    graph,
    loading,
    error,
    filteredNodes,
    filteredEdges,
    allTypes,
    selectedTypes,
    setSelectedTypes,
    selectedDocId,
    setSelectedDocId,
    fetchGraph,
  };
}
