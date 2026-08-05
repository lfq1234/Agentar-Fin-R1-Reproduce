import { useCallback, useEffect, useState } from "react";
import { chat, analyze, health, type ApiError } from "../api/client";
import type {
  AnalyzeRequest,
  AnalyzeResponse,
  BackendStatus,
  ChatRequest,
  ChatSession,
  SceneOption,
  UiMessage,
} from "../types/agent";

// 落库 / 续聊必须携带 user_id（G2）：默认 1，实现时务必随每次 chat 请求发送。
const DEFAULT_USER_ID = 1;

// 会话 ID 生成：优先 crypto.randomUUID（需安全上下文），否则降级（评审 S2）。
function genId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function createBlankSession(scene: SceneOption = "Banking"): ChatSession {
  const now = Date.now();
  return {
    id: genId(),
    title: "新对话",
    history: [],
    conversationId: null,
    scene,
    createdAt: now,
    updatedAt: now,
  };
}

// 标题派生：取第一条用户消息前 20 字，超出加省略号（评审 M2，与需求一致）。
export function deriveTitle(history: UiMessage[]): string {
  const firstUser = history.find((m) => m.role === "user");
  if (!firstUser) return "新对话";
  const text = firstUser.content.trim();
  return text.length > 20 ? text.slice(0, 20) + "…" : text;
}

// 相对时间格式化（评审 N3，使用原生 Intl，无额外依赖）。
export function formatRelativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const rtf = new Intl.RelativeTimeFormat("zh-CN", { numeric: "auto" });
  const min = 60_000;
  const hour = 60 * min;
  const day = 24 * hour;
  if (diff < min) return "刚刚";
  if (diff < hour) return rtf.format(-Math.floor(diff / min), "minute");
  if (diff < day) return rtf.format(-Math.floor(diff / hour), "hour");
  if (diff < 7 * day) return rtf.format(-Math.floor(diff / day), "day");
  return new Date(ts).toLocaleDateString("zh-CN");
}

export function useChat() {
  // 会话集合（评审：单会话 → 集合）。初始一个空白会话。
  const [sessions, setSessions] = useState<ChatSession[]>(() => [createBlankSession()]);
  const [currentSessionId, setCurrentSessionId] = useState<string>(() => sessions[0]?.id ?? "");
  const [personalDocsOpen, setPersonalDocsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false); // 移动端侧边栏显隐（评审 M3）

  // 输入与全局状态
  const [message, setMessage] = useState("");
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

  const currentSession = sessions.find((s) => s.id === currentSessionId) ?? sessions[0];

  // 发送：仅在当前会话内追加消息，并更新 conversationId / updatedAt（评审 M4 复用）。
  const send = useCallback(async () => {
    const text = message.trim();
    if (!text || loading || !currentSession) return;
    const sid = currentSessionId;

    setMessage("");
    setError(null);
    const userMsg: UiMessage = { role: "user", content: text };
    setSessions((prev) =>
      prev.map((s) =>
        s.id === sid ? { ...s, history: [...s.history, userMsg], updatedAt: Date.now() } : s,
      ),
    );
    setLoading(true);

    const req: ChatRequest = {
      message: text,
      user_id: DEFAULT_USER_ID,
      ...(currentSession.scene !== "Auto" ? { scene: currentSession.scene } : {}),
      // 已有 conversationId 则携带，复用后端会话上下文（M4）
      ...(currentSession.conversationId ? { conversation_id: currentSession.conversationId } : {}),
    };

    try {
      const res = await chat(req);
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== sid) return s;
          const history: UiMessage[] = [
            ...s.history,
            {
              role: "assistant",
              content: res.reply,
              compliance: res.compliance_notes,
              risk: res.risk_flags,
            },
          ];
          // 首次出现用户消息后，标题从「新对话」派生
          const title = s.title === "新对话" ? deriveTitle(history) : s.title;
          return {
            ...s,
            history,
            title,
            conversationId: res.conversation_id ?? s.conversationId,
            updatedAt: Date.now(),
          };
        }),
      );
    } catch (e) {
      setError((e as ApiError).message ?? "请求失败");
    } finally {
      setLoading(false);
    }
  }, [message, loading, currentSession, currentSessionId]);

  // 分析：针对当前会话最近一条助手回复（评审 N4，与会话隔离一致）。
  const runAnalyze = useCallback(async () => {
    if (analyzing || !currentSession) return;
    const last = [...currentSession.history].reverse().find((m) => m.role === "assistant");
    if (!last) return;
    const text = last.content.trim();
    if (!text) return;

    setAnalyzing(true);
    setError(null);
    const req: AnalyzeRequest = {
      message: text,
      ...(currentSession.scene !== "Auto" ? { scene: currentSession.scene } : {}),
    };
    try {
      const res = await analyze(req);
      setAnalyzeResult(res);
    } catch (e) {
      setError((e as ApiError).message ?? "分析失败");
    } finally {
      setAnalyzing(false);
    }
  }, [analyzing, currentSession]);

  // 新建会话：激活并关闭个人文档 / 收起移动端侧边栏。
  const createSession = useCallback(() => {
    const ns = createBlankSession(currentSession?.scene ?? "Banking");
    setSessions((prev) => [...prev, ns]);
    setCurrentSessionId(ns.id);
    setPersonalDocsOpen(false);
    setSidebarOpen(false);
  }, [currentSession?.scene]);

  // 删除会话：删当前则切到 updatedAt 最新的；无会话则自动新建空白。
  const deleteSession = useCallback(
    (id: string) => {
      setSessions((prev) => {
        const remaining = prev.filter((s) => s.id !== id);
        if (remaining.length === 0) {
          const ns = createBlankSession();
          setCurrentSessionId(ns.id);
          return [ns];
        }
        if (id === currentSessionId) {
          const next = [...remaining].sort((a, b) => b.updatedAt - a.updatedAt)[0];
          setCurrentSessionId(next.id);
        }
        return remaining;
      });
      setPersonalDocsOpen(false);
    },
    [currentSessionId],
  );

  // 切换会话：关闭个人文档 / 收起移动端侧边栏（评审：互斥）。
  const selectSession = useCallback((id: string) => {
    setCurrentSessionId(id);
    setPersonalDocsOpen(false);
    setSidebarOpen(false);
  }, []);

  // 场景双向绑定：修改当前会话的 scene（评审 N5）。
  const setScene = useCallback(
    (v: SceneOption) => {
      setSessions((prev) =>
        prev.map((s) => (s.id === currentSessionId ? { ...s, scene: v, updatedAt: Date.now() } : s)),
      );
    },
    [currentSessionId],
  );

  const togglePersonalDocs = useCallback(() => setPersonalDocsOpen((v) => !v), []);
  const toggleSidebar = useCallback(() => setSidebarOpen((v) => !v), []);

  const history = currentSession?.history ?? [];
  const scene = currentSession?.scene ?? "Banking";
  const lastAssistant = history.length
    ? ([...history].reverse().find((m) => m.role === "assistant") ?? null)
    : null;

  return {
    // —— 会话集合 ——
    sessions,
    currentSessionId,
    selectSession,
    createSession,
    deleteSession,
    personalDocsOpen,
    togglePersonalDocs,
    sidebarOpen,
    toggleSidebar,
    // —— 当前会话视图 ——
    history,
    scene,
    setScene,
    lastAssistant,
    // —— 输入与全局状态 ——
    message,
    setMessage,
    loading,
    analyzing,
    analyzeResult,
    setAnalyzeResult,
    error,
    setError,
    backendStatus,
    send,
    runAnalyze,
  };
}
