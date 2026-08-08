// 后端契约镜像（与 backend/app/routes/chat.py 对齐，已核对源码）。
// 注意：前端不再选择智能体/场景——由后端多智能体框架（Coordinator 路由 + 专家通讯）决定谁来回答。
// 因此请求体不再携带 scene，交给后端自动路由（system.py:184）。

export interface ChatRequest {
  message: string;
  // 09：user_id 不再由前端传，改由后端从 JWT 令牌解析（数据隔离）。
  conversation_id?: number;
}

export interface ChatMessage {
  role: "user" | "assistant" | "agent";
  content: string;
  agent?: string;
  avatar?: string;
  mention?: string | null;
  type?: string;
}

export interface AgentTraceStep {
  agent: string;
  // route / rag / expert_opinion / synthesize / revise
  type: string;
  content?: string | null;
  meta?: Record<string, unknown> | null;
}

export interface ChatResponse {
  conversation_id?: number | null;
  reply: string;
  compliance_notes: string[];
  risk_flags: string[];
  // 02：多人编排每步（Coordinator 路由/各专家意见/合成/回写），前端用于
  // 把"是哪位/哪些智能体参与的"显示给用户，对应后端 ChatResponse.agent_trace。
  agent_trace?: AgentTraceStep[];
  messages?: ChatMessage[];
}

// SSE 流式聊天事件（后端 run_stream() 逐步骤 yield）
export interface ChatStreamEvent {
  type: "route" | "agent_start" | "agent_message" | "done" | "error";
  agent?: string;
  avatar?: string;
  name?: string;
  content?: string;
  mention?: string;
  scene?: string;
  reply?: string;
  conversation_id?: number | null;
  detail?: string;
}

export interface AnalyzeRequest {
  message: string;
}

export interface AnalyzeResponse {
  intent?: string | null;
  slots: Record<string, string>;
  tool_plan: string[];
  expression: string;
}

// 前端消息流条目（来源为后端返回，仅内存维护，不独立持久化）。
export interface UiMessage {
  role: "user" | "assistant" | "agent";
  content: string;
  agent?: string;
  avatar?: string;
  mention?: string | null;
  type?: string;
  compliance?: string[];
  risk?: string[];
  // 02：本轮参与的所有智能体（从 res.agent_trace 去重抽取，
  // Direct 通道为空时退化为 ["Agentar"]）；用于在 assistant 气泡下方显示。
  participants?: string[];
}

export type BackendStatus = "unknown" | "ok" | "down";

// 07：后端 /api/v1/history/sessions 返回的会话元信息。
export interface SessionMeta {
  conversation_id: string; // 后端 conversations.id（字符串形式）
  scene: string | null;
  title: string | null;
  status: string;
  msg_count: number;
  total_tokens: number;
  first_at: number; // epoch ms
  last_at: number; // epoch ms
  created_at: number; // epoch ms
}

// 07：后端 /api/v1/history/sessions/{id} 返回的会话详情。
export interface SessionDetail {
  conversation_id: string;
  meta: SessionMeta;
  messages: Array<{
    role: "user" | "assistant" | "agent";
    content: string;
    scene?: string | null;
    created_at?: number;
    agent?: string;
    avatar?: string;
    mention?: string | null;
    type?: string;
  }>;
  trace?: unknown;
  has_trace: boolean;
}

// 前端本地会话（已从后端历史接口加载，刷新后重新拉取）。
export interface ChatSession {
  id: string;
  title: string;
  history: UiMessage[];
  conversationId: number | null;
  createdAt: number;
  updatedAt: number;
}

// ===== 02 · 多智能体参与者（前端契约，镜像 backend/02 框架） =====

// 后端专家 → 前端展示名 / 头像 / 主题色。后端 agent 字段统一为 AgentScope 名字键
// （如 BankingExpert / Coordinator / Direct），前端按名字映射展示。Direct 通道无
// trace 时退化为 Agentar。color 用于圆形头像底色，让不同智能体一眼可区分。
export const AGENT_DISPLAY: Record<string, { name: string; avatar: string; color: string }> = {
  Coordinator: { name: "协调者", avatar: "🤖", color: "#6366f1" },
  Agentar: { name: "Agentar", avatar: "🤖", color: "#2563eb" },
  Direct: { name: "Agentar", avatar: "🤖", color: "#2563eb" },
  rag: { name: "资料检索", avatar: "🔍", color: "#0ea5e9" },
  RAGRetriever: { name: "资料检索", avatar: "🔍", color: "#0ea5e9" },
  BankingExpert: { name: "银行专家", avatar: "🏦", color: "#16a34a" },
  SecuritiesExpert: { name: "证券专家", avatar: "📈", color: "#dc2626" },
  InsuranceExpert: { name: "保险专家", avatar: "🛡️", color: "#0891b2" },
  TrustExpert: { name: "信托专家", avatar: "🏛️", color: "#7c3aed" },
  MutualFundsExpert: { name: "基金专家", avatar: "🧺", color: "#ca8a04" },
  ComplianceReviewer: { name: "合规审核", avatar: "✅", color: "#059669" },
  RiskController: { name: "风控", avatar: "⚠️", color: "#ea580c" },
};

export interface Participant {
  key: string; // 与后端 agent_key 对齐（用于查 AGENT_DISPLAY）
  name: string;
  avatar: string;
  color: string;
}

// ===== 03 · 个人文档与知识图谱（前端契约，镜像 backend/03 接口） =====

export type DocStatus = "pending" | "parsing" | "done" | "error";

export interface PersonalDocument {
  id: string;
  filename: string;
  size: number; // bytes
  status: DocStatus;
  error?: string;
  uploadedAt: string; // ISO 8601
  summary?: string; // 可选：用于「基于此文档提问」联动（后端可返回）
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  sourceDocId: string;
  properties?: Record<string, string | number>;
}

export interface GraphEdge {
  id: string;
  source: string; // node id
  target: string; // node id
  label: string;
  sourceDocId: string;
}

export interface PersonalKnowledgeGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// ===== 09 · 用户系统与数据隔离（前端契约，镜像 backend/auth.py 与 user.py） =====

export interface AuthUser {
  id: number;
  username: string;
  email?: string | null;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterPayload {
  username: string;
  password: string;
  email?: string;
}
