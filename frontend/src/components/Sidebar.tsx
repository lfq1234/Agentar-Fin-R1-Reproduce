import type { AuthUser, BackendStatus, ChatSession } from "../types/agent";
import { PersonalDocsCard } from "./PersonalDocsCard";
import { SessionList } from "./SessionList";

const STATUS_TEXT: Record<BackendStatus, string> = {
  unknown: "连接中…",
  ok: "后端在线",
  down: "后端离线",
};

interface Props {
  sessions: ChatSession[];
  currentSessionId: string;
  personalDocsOpen: boolean;
  sidebarOpen: boolean;
  user?: AuthUser | null;
  canAnalyze: boolean;
  analyzing: boolean;
  backendStatus: BackendStatus;
  onSelectSession: (id: string) => void;
  onCreateSession: () => void;
  onDeleteSession: (id: string) => void;
  onTogglePersonalDocs: () => void;
  onAnalyze: () => void;
  onLogout: () => void;
}

export function Sidebar({
  sessions,
  currentSessionId,
  personalDocsOpen,
  sidebarOpen,
  user,
  canAnalyze,
  analyzing,
  backendStatus,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
  onTogglePersonalDocs,
  onAnalyze,
  onLogout,
}: Props) {
  return (
    <aside className={`sidebar${sidebarOpen ? " open" : ""}`}>
      <div className="sidebar-user">
        <span className="sidebar-user-name" title={user?.email ?? ""}>
          {user?.username ?? "未登录"}
        </span>
        <button className="sidebar-logout" onClick={onLogout} title="退出登录">
          退出
        </button>
      </div>
      <PersonalDocsCard active={personalDocsOpen} onClick={onTogglePersonalDocs} />
      <button
        className="analyze-card"
        onClick={onAnalyze}
        disabled={!canAnalyze || analyzing}
        title="分析当前会话最近一条助手回复的意图、槽位、工具规划与表达"
      >
        <span className="analyze-card-icon" aria-hidden>
          🔍
        </span>
        <span>{analyzing ? "分析中…" : "分析最近回复"}</span>
      </button>
      <div className="backend-status" title={STATUS_TEXT[backendStatus]}>
        <span className={`status-bulb ${backendStatus}`} />
        <span className="backend-status-text">{STATUS_TEXT[backendStatus]}</span>
      </div>
      <div className="sidebar-divider" />
      <SessionList
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelect={onSelectSession}
        onDelete={onDeleteSession}
      />
      <button className="btn btn-primary new-session-btn" onClick={onCreateSession}>
        + 新对话
      </button>
    </aside>
  );
}
