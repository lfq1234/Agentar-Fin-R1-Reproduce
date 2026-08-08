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

export interface ChatResponse {
  conversation_id?: number | null;
  reply: string;
  compliance_notes: string[];
  risk_flags: string[];
  messages?: ChatMessage[];
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
