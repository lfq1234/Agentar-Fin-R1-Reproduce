import { useCallback, useEffect, useState } from "react";
import { chat, analyze, health, type ApiError } from "../api/client";
import type {
  AnalyzeRequest,
  AnalyzeResponse,
  BackendStatus,
  ChatRequest,
  SceneOption,
  UiMessage,
} from "../types/agent";

// 落库 / 续聊必须携带 user_id（G2）：默认 1，实现时务必随每次 chat 请求发送。
const DEFAULT_USER_ID = 1;

export function useChat() {
  const [message, setMessage] = useState("");
  const [scene, setScene] = useState<SceneOption>("Banking");
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [userId] = useState<number>(DEFAULT_USER_ID);
  const [history, setHistory] = useState<UiMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("unknown");

  // 挂载时探活，驱动 Header 状态点（S3 / 验收清单）。
  useEffect(() => {
    health()
      .then(() => setBackendStatus("ok"))
      .catch(() => setBackendStatus("down"));
  }, []);

  const send = useCallback(async () => {
    const text = message.trim();
    if (!text || loading) return;

    setMessage("");
    setError(null);
    setHistory((h) => [...h, { role: "user", content: text }]);
    setLoading(true);

    const req: ChatRequest = {
      message: text,
      user_id: userId,
      ...(scene !== "Auto" ? { scene } : {}),
      ...(conversationId ? { conversation_id: conversationId } : {}),
    };

    try {
      const res = await chat(req);
      if (res.conversation_id) setConversationId(res.conversation_id);
      setHistory((h) => [
        ...h,
        {
          role: "assistant",
          content: res.reply,
          compliance: res.compliance_notes,
          risk: res.risk_flags,
        },
      ]);
    } catch (e) {
      setError((e as ApiError).message ?? "请求失败");
    } finally {
      setLoading(false);
    }
  }, [message, loading, scene, conversationId, userId]);

  const runAnalyze = useCallback(
    async (targetMessage: string) => {
      const text = targetMessage.trim();
      if (!text || analyzing) return;

      setAnalyzing(true);
      setError(null);

      const req: AnalyzeRequest = {
        message: text,
        ...(scene !== "Auto" ? { scene } : {}),
      };

      try {
        const res = await analyze(req);
        setAnalyzeResult(res);
      } catch (e) {
        setError((e as ApiError).message ?? "分析失败");
      } finally {
        setAnalyzing(false);
      }
    },
    [scene, analyzing],
  );

  const reset = useCallback(() => {
    setHistory([]);
    setConversationId(null);
    setAnalyzeResult(null);
    setError(null);
  }, []);

  return {
    message,
    setMessage,
    scene,
    setScene,
    history,
    loading,
    analyzing,
    analyzeResult,
    setAnalyzeResult,
    error,
    setError,
    backendStatus,
    send,
    runAnalyze,
    reset,
  };
}
