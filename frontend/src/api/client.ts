import type {
  ChatRequest,
  ChatResponse,
  ChatStreamEvent,
  AnalyzeRequest,
  AnalyzeResponse,
  PersonalDocument,
  PersonalKnowledgeGraph,
  AuthUser,
  LoginCredentials,
  RegisterPayload,
  SessionMeta,
  SessionDetail,
} from "../types/agent";
import { mockDocuments, mockGraph } from "../mock/personalDocs";

// 03：后端文档/图谱接口未就绪时的 Mock 开关（需求 R1 / 评审 S6）。
// 在 .env 设 VITE_USE_MOCK=true 即可用本地样例数据跑通 UI，无需后端。
const USE_MOCK = (import.meta.env.VITE_USE_MOCK ?? "false") === "true";

// 与后端路径一致：统一带 /api 前缀（后端 include_router(prefix="/api")，vite proxy 为 /api）。
const BASE = "/api/v1";

// 统一错误类型：网络故障 status=0，便于组件层区分文案（S3）。
export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

// ===== 09：鉴权令牌本地存储 =====
export const TOKEN_KEY = "agentar_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}
export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

// 401/403 登出广播：request 拦截到鉴权失败时清除令牌并派发，App 监听后切回登录页。
export const UNAUTHORIZED_EVENT = "agentar:unauthorized";

function authHeaders(): Record<string, string> {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

interface RequestOptions {
  method?: string;
  body?: BodyInit | null;
  headers?: Record<string, string>;
  json?: unknown;
}

// 统一请求封装：自动携带令牌、拦截 401/403 触发登出。
async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders(), ...(opts.headers ?? {}) };
  if (opts.json !== undefined) headers["Content-Type"] = "application/json";
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      method: opts.method ?? "GET",
      headers,
      body: opts.json !== undefined ? JSON.stringify(opts.json) : opts.body ?? null,
    });
  } catch (e) {
    throw new ApiError(0, `网络错误，无法连接后端：${(e as Error).message}`);
  }
  // 鉴权失败：清令牌并广播登出（缺失令牌 401；令牌失效/禁用账号 403）。
  if (res.status === 401 || res.status === 403) {
    clearToken();
    window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT));
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new ApiError(res.status, `请求失败 (${res.status})${detail ? "：" + detail : ""}`);
  }
  const text = await res.text();
  return (text ? JSON.parse(text) : ({} as T)) as T;
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", json: body });
}
function get<T>(path: string): Promise<T> {
  return request<T>(path, { method: "GET" });
}

export const chat = (req: ChatRequest) => post<ChatResponse>("/chat", req);
export const analyze = (req: AnalyzeRequest) => post<AnalyzeResponse>("/analyze", req);

// SSE 流式聊天：读取 ReadableStream，逐事件回调，返回最终的 reply + conversation_id
export async function chatStream(
  req: ChatRequest,
  onEvent: (ev: ChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<{ reply: string; conversationId: number | null }> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (res.status === 401 || res.status === 403) {
    clearToken();
    window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT));
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new ApiError(res.status, `请求失败 (${res.status})${detail ? "：" + detail : ""}`);
  }

  let reply = "";
  let conversationId: number | null = null;

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const ev: ChatStreamEvent = JSON.parse(line.slice(6));
          if (ev.type === "done") {
            reply = ev.reply || "";
            conversationId = ev.conversation_id ?? null;
          } else if (ev.type === "error") {
            throw new ApiError(500, ev.detail || "流式处理错误");
          }
          onEvent(ev);
        } catch (e) {
          if (e instanceof ApiError) throw e;
        }
      }
    }
  }

  return { reply, conversationId };
}

export async function health(): Promise<{ status: string }> {
  // 健康检查挂载在 /api 前缀下（与 chat/analyze 的 /api/v1 区分），公开接口无需令牌。
  const res = await fetch("/api/health");
  if (!res.ok) throw new ApiError(res.status, `健康检查失败 (${res.status})`);
  return res.json();
}

// ===== 09：认证接口（契约镜像 backend/app/routes/auth.py） =====

// OAuth2 密码流登录：表单格式（非 JSON），成功后存储令牌并拉取当前用户。
export async function login(creds: LoginCredentials): Promise<AuthUser> {
  const form = new URLSearchParams();
  form.set("username", creds.username);
  form.set("password", creds.password);
  const res = await fetch(`${BASE}/auth/login/access-token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded", ...authHeaders() },
    body: form.toString(),
  });
  if (res.status === 401 || res.status === 403) {
    clearToken();
    window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT));
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new ApiError(res.status, `登录失败 (${res.status})${detail ? "：" + detail : ""}`);
  }
  const data = (await res.json()) as { access_token: string; token_type: string };
  setToken(data.access_token);
  return me();
}

// 注册：201 返回 UserPublic（不含密码）。
export async function register(payload: RegisterPayload): Promise<AuthUser> {
  return post<AuthUser>("/auth/register", payload);
}

// 当前用户：需令牌，401/403 由 request 统一拦截。
export async function me(): Promise<AuthUser> {
  return get<AuthUser>("/auth/me");
}

// ===== 07 · 会话历史接口（依赖鉴权，后端从 JWT 解析 user_id） =====

export async function listSessions(): Promise<SessionMeta[]> {
  return get<SessionMeta[]>("/history/sessions");
}

export async function getSession(conversation_id: string): Promise<SessionDetail> {
  return get<SessionDetail>(`/history/sessions/${encodeURIComponent(conversation_id)}`);
}

export async function deleteSession(conversation_id: string): Promise<void> {
  await request<void>(`/history/sessions/${encodeURIComponent(conversation_id)}`, { method: "DELETE" });
}

// ===== 03 · 个人文档与知识图谱接口（依赖鉴权，不再前端传 user_id） =====
// 路径与后端一致：统一带 /api 前缀（后端 include_router(prefix="/api")）。

export async function uploadDocuments(files: File[]): Promise<PersonalDocument[]> {
  if (USE_MOCK) {
    return files.map((f, i) => ({
      id: `mock-doc-${Date.now()}-${i}`,
      filename: f.name,
      size: f.size,
      status: "done",
      uploadedAt: new Date().toISOString(),
      summary: `（Mock）文档《${f.name}》的摘要，可用于提问。`,
    }));
  }
  // 不手动设 Content-Type，交由浏览器写入 multipart 边界；鉴权头由 request 自动添加。
  const res = await request<{ documents: PersonalDocument[] }>("/documents", {
    method: "POST",
    body: (() => {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      return form;
    })(),
  });
  return res.documents;
}

export async function listDocuments(): Promise<PersonalDocument[]> {
  if (USE_MOCK) return mockDocuments;
  const res = await request<{ documents: PersonalDocument[] }>("/documents", { method: "GET" });
  return res.documents;
}

export async function deleteDocument(id: string): Promise<void> {
  if (USE_MOCK) return;
  await request<void>(`/documents/${id}`, { method: "DELETE" });
}

export async function getDocumentStatus(id: string): Promise<PersonalDocument> {
  if (USE_MOCK) return mockDocuments.find((d) => d.id === id) ?? mockDocuments[0];
  return request<PersonalDocument>(`/documents/${id}/status`, { method: "GET" });
}

export async function getKnowledgeGraph(): Promise<PersonalKnowledgeGraph> {
  if (USE_MOCK) return mockGraph;
  return request<PersonalKnowledgeGraph>("/knowledge-graph", { method: "GET" });
}
