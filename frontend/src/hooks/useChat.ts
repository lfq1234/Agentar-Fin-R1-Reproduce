import { useCallback, useEffect, useRef, useState } from "react";
import {
  chatStream,
  analyze as apiAnalyze,
  health,
  listSessions,
  getSession,
  deleteSession as apiDeleteSession,
  type ApiError,
} from "../api/client";
import type {
  AgentTraceStep,
  AnalyzeRequest,
  AnalyzeResponse,
  BackendStatus,
  ChatRequest,
  ChatStreamEvent,
  ChatSession,
  SessionMeta,
  UiMessage,
} from "../types/agent";

// 09：user_id 由后端从 JWT 令牌解析，前端不再携带（数据隔离）。
// 智能体/场景选择：前端不再传入 scene——由后端多智能体框架（Coordinator 路由 + 专家互询/圆桌）决定谁来回答。

// 会话 ID 生成：优先 crypto.randomUUID（需安全上下文），否则降级（评审 S2）。
function genId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function createBlankSession(): ChatSession {
  const now = Date.now();
  return {
    id: genId(),
    title: "新对话",
    history: [],
    conversationId: null,
    createdAt: now,
    updatedAt: now,
  };
}

// 07：后端 SessionMeta → 前端 ChatSession（history 为空，点击会话时再拉详情）。
function metaToSession(meta: SessionMeta): ChatSession {
  return {
    id: meta.conversation_id,
    title: meta.title || "新对话",
    history: [],
    conversationId: Number(meta.conversation_id) || null,
    createdAt: meta.created_at || meta.first_at || Date.now(),
    updatedAt: meta.last_at || Date.now(),
  };
}

// 07：后端历史消息 → 前端 UiMessage（老数据可能不带 agent/avatar，降级为纯文本展示）。
// 02：trace 抽取本轮参与智能体（去重、保序）。
// Direct 通道（trace 空）退化为 ["Agentar"]，让用户至少能看到「是谁答的」。
function participantsFromTrace(trace: AgentTraceStep[] | undefined | null): string[] {
  const seen = new Set<string>();
  const list: string[] = [];
  for (const s of trace || []) {
    const k = s.agent;
    if (k && !seen.has(k)) {
      seen.add(k);
      list.push(k);
    }
  }
  // 直答通道无 trace：标注为 Agentar
  return list.length === 0 ? ["Agentar"] : list;
}

function historyMsgToUiMessage(m: {
  role: "user" | "assistant" | "agent";
  content: string;
  scene?: string | null;
  created_at?: number;
  agent?: string;
  avatar?: string;
  mention?: string | null;
  type?: string;
}): UiMessage {
  return {
    role: m.role,
    content: m.content || "",
    agent: m.agent,
    avatar: m.avatar,
    mention: m.mention,
    type: m.type,
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
  // 会话集合（评审：单会话 → 集合）。初始不预建空白——避免侧边栏出现
  // 「本地占位 + 后端真会话」的重复条目；启动后从后端拉取历史，若无再
  // 补一个空白。这样列表里的可见项总是有意义的（真历史 / 当前正在编辑的草稿）。
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>("");
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

  // SSE 流式发送：每收到一个专家消息就立即追加到历史，形成边说边出现的群聊效果。
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
      ...(currentSession.conversationId ? { conversation_id: currentSession.conversationId } : {}),
    };

    // 本轮流式产生的智能体消息，最终一起写入 history
    const agentMsgs: UiMessage[] = [];

    const pushAgentMsg = (m: UiMessage) => {
      agentMsgs.push(m);
      setSessions((prev) =>
        prev.map((s) =>
          s.id === sid
            ? { ...s, history: [...s.history, m], updatedAt: Date.now() }
            : s,
        ),
      );
    };

    try {
      const { reply, conversationId } = await chatStream(req, (ev: ChatStreamEvent) => {
        if (ev.type === "agent_message" && ev.content) {
          pushAgentMsg({
            role: "agent",
            content: ev.content,
            agent: ev.agent,
            avatar: ev.avatar,
            mention: ev.mention ?? undefined,
            type: "expert_opinion",
          });
        }
      });

      const newConversationId = conversationId ?? null;
      const newBackendId = newConversationId != null ? String(newConversationId) : sid;
      setCurrentSessionId(newBackendId);

      // 追加最终助手回复
      const finalMsg: UiMessage = { role: "assistant", content: reply };
      const finalHistory = [...agentMsgs, finalMsg];
      const title = currentSession.title === "新对话"
        ? deriveTitle([userMsg, ...finalHistory])
        : currentSession.title;

      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== sid) return s;
          return {
            ...s,
            id: newBackendId,
            history: [...s.history, finalMsg],
            title,
            conversationId: newConversationId,
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
    };
    try {
      const res = await apiAnalyze(req);
      setAnalyzeResult(res);
    } catch (e) {
      setError((e as ApiError).message ?? "分析失败");
    } finally {
      setAnalyzing(false);
    }
  }, [analyzing, currentSession]);

  // 07：登录后从后端拉取历史会话列表；空列表时补一个空白让用户能立刻开聊。
  const loadSessions = useCallback(async () => {
    try {
      const metas = await listSessions();
      if (metas.length === 0) {
        // 后端无历史：补一个本地空白，并把当前会话切到它。
        setSessions((prev) => {
          if (prev.length > 0) return prev;
          const ns = createBlankSession();
          setCurrentSessionId(ns.id);
          return [ns];
        });
        return;
      }
      const loaded = metas.map(metaToSession);
      setSessions(loaded);
      setCurrentSessionId(loaded[0].id);
    } catch (e) {
      // 历史加载失败不阻断主链路，仅静默降级（避免一启动就弹错误）。
      // eslint-disable-next-line no-console
      console.warn("加载历史会话失败", e);
      // 拉取失败也保证有可用空白会话，避免侧边栏空白 / 无法输入。
      setSessions((prev) => {
        if (prev.length > 0) return prev;
        const ns = createBlankSession();
        setCurrentSessionId(ns.id);
        return [ns];
      });
    }
  }, []);

  // 07：点击历史会话且尚未加载详情时，拉取 conversations.data 中的消息。
  const loadSessionDetail = useCallback(async (id: string) => {
    try {
      const detail = await getSession(id);
      const messages = (detail.messages || []).map(historyMsgToUiMessage);
      setSessions((prev) =>
        prev.map((s) =>
          s.id === id
            ? {
                ...s,
                title: detail.meta.title || s.title,
                history: messages,
                updatedAt: detail.meta.last_at || Date.now(),
              }
            : s,
        ),
      );
    } catch (e) {
      setError(((e as ApiError).message) ?? "加载会话详情失败");
    }
  }, []);

  // 新建会话：激活并关闭个人文档 / 收起移动端侧边栏。
  const createSession = useCallback(() => {
    const ns = createBlankSession();
    setSessions((prev) => [ns, ...prev]);
    setCurrentSessionId(ns.id);
    setPersonalDocsOpen(false);
    setSidebarOpen(false);
  }, []);

  // 删除会话：先调后端删除，再本地移除；删当前则切到 updatedAt 最新的。
  const deleteSession = useCallback(
    async (id: string) => {
      const target = sessions.find((s) => s.id === id);
      if (target?.conversationId != null) {
        try {
          await apiDeleteSession(id);
        } catch (e) {
          setError(((e as ApiError).message) ?? "删除会话失败");
          return;
        }
      }
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
    [sessions, currentSessionId],
  );

  // 切换会话：关闭个人文档 / 收起移动端侧边栏（评审：互斥）；
  // 若该会话历史未加载，则异步拉取详情。
  const selectSession = useCallback(
    (id: string) => {
      setCurrentSessionId(id);
      setPersonalDocsOpen(false);
      setSidebarOpen(false);
      const target = sessions.find((s) => s.id === id);
      if (target && target.conversationId != null && target.history.length === 0) {
        loadSessionDetail(id);
      }
    },
    [sessions, loadSessionDetail],
  );

  const togglePersonalDocs = useCallback(() => setPersonalDocsOpen((v) => !v), []);
  const toggleSidebar = useCallback(() => setSidebarOpen((v) => !v), []);

  const history = currentSession?.history ?? [];
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
    loadSessions,
    personalDocsOpen,
    togglePersonalDocs,
    sidebarOpen,
    toggleSidebar,
    // —— 当前会话视图 ——
    history,
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
