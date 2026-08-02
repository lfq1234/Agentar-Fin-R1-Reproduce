// 后端契约镜像（与 backend/app/routes/chat.py 对齐，已核对源码）。

export type Scene = "Banking" | "Securities" | "Insurance" | "Trust" | "MutualFunds";

export const SCENES: Scene[] = ["Banking", "Securities", "Insurance", "Trust", "MutualFunds"];

// 前端下拉选项：Auto = 不传 scene，触发后端 Coordinator 路由（G3 默认场景决策）。
export type SceneOption = Scene | "Auto";

export const SCENE_OPTIONS: SceneOption[] = ["Auto", ...SCENES];

export interface ChatRequest {
  message: string;
  scene?: Scene; // 省略时后端按默认 / Coordinator 路由处理
  user_id?: number; // 落库 / 续聊必须（G2）
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
