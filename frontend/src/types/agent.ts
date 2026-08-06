// 后端契约镜像（与 backend/app/routes/chat.py 对齐，已核对源码）。

export type Scene = "Banking" | "Securities" | "Insurance" | "Trust" | "MutualFunds";

export const SCENES: Scene[] = ["Banking", "Securities", "Insurance", "Trust", "MutualFunds"];

// 前端下拉选项：Auto = 不传 scene，触发后端 Coordinator 路由（G3 默认场景决策）。
export type SceneOption = Scene | "Auto";

export const SCENE_OPTIONS: SceneOption[] = ["Auto", ...SCENES];

export interface ChatRequest {
  message: string;
  scene?: Scene; // 省略时后端按默认 / Coordinator 路由处理
  // 09：user_id 不再由前端传，改由后端从 JWT 令牌解析（数据隔离）。
  conversation_id?: number;
}

export interface ChatResponse {
  scene?: Scene;
  conversation_id?: number | null;
  reply: string;
  compliance_notes: string[];
  risk_flags: string[];
}

export interface AnalyzeRequest {
  message: string;
  scene?: Scene;
}

export interface AnalyzeResponse {
  scene?: Scene;
  intent?: string | null;
  slots: Record<string, string>;
  tool_plan: string[];
  expression: string;
}

// 前端消息流条目（来源为后端返回，仅内存维护，不独立持久化）。
export interface UiMessage {
  role: "user" | "assistant";
  content: string;
  compliance?: string[];
  risk?: string[];
}

export type BackendStatus = "unknown" | "ok" | "down";

// 前端本地会话（本期仅内存维护，刷新即丢失；后续可接后端会话列表接口）。
export interface ChatSession {
  id: string;
  title: string;
  history: UiMessage[];
  conversationId: number | null;
  scene: SceneOption;
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
