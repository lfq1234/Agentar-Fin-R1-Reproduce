import type {
  ChatRequest,
  ChatResponse,
  AnalyzeRequest,
  AnalyzeResponse,
  PersonalDocument,
  PersonalKnowledgeGraph,
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

async function post<T>(path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    throw new ApiError(0, `网络错误，无法连接后端：${(e as Error).message}`);
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new ApiError(res.status, `请求失败 (${res.status})${detail ? "：" + detail : ""}`);
  }
  return res.json() as Promise<T>;
}

export const chat = (req: ChatRequest) => post<ChatResponse>("/chat", req);
export const analyze = (req: AnalyzeRequest) => post<AnalyzeResponse>("/analyze", req);

export async function health(): Promise<{ status: string }> {
  // 健康检查挂载在 /api 前缀下（与 chat/analyze 的 /api/v1 区分）。
  const res = await fetch("/api/health");
  if (!res.ok) throw new ApiError(res.status, `健康检查失败 (${res.status})`);
  return res.json();
}

// ===== 03 · 个人文档与知识图谱接口 =====
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
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("user_id", "1");
  let res: Response;
  try {
    res = await fetch(`${BASE}/documents`, { method: "POST", body: form });
  } catch (e) {
    throw new ApiError(0, `网络错误，无法上传文档：${(e as Error).message}`);
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new ApiError(res.status, `上传失败 (${res.status})${detail ? "：" + detail : ""}`);
  }
  return (await res.json()).documents;
}

export async function listDocuments(): Promise<PersonalDocument[]> {
  if (USE_MOCK) return mockDocuments;
  const res = await fetch(`${BASE}/documents?user_id=1`);
  if (!res.ok) throw new ApiError(res.status, "获取文档列表失败");
  return (await res.json()).documents;
}

export async function deleteDocument(id: string): Promise<void> {
  if (USE_MOCK) return;
  const res = await fetch(`${BASE}/documents/${id}?user_id=1`, { method: "DELETE" });
  if (!res.ok) throw new ApiError(res.status, "删除失败");
}

export async function getDocumentStatus(id: string): Promise<PersonalDocument> {
  if (USE_MOCK) return mockDocuments.find((d) => d.id === id) ?? mockDocuments[0];
  const res = await fetch(`${BASE}/documents/${id}/status`);
  if (!res.ok) throw new ApiError(res.status, "获取状态失败");
  return await res.json();
}

export async function getKnowledgeGraph(): Promise<PersonalKnowledgeGraph> {
  if (USE_MOCK) return mockGraph;
  const res = await fetch(`${BASE}/knowledge-graph?user_id=1`);
  if (!res.ok) throw new ApiError(res.status, "获取图谱失败");
  return await res.json();
}
