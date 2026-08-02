import type { ChatRequest, ChatResponse, AnalyzeRequest, AnalyzeResponse } from "../types/agent";

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
