import type { AuthUser, ChatSession } from "../types/agent";
import { PersonalDocsCard } from "./PersonalDocsCard";
import { SessionList } from "./SessionList";

interface Props {
  sessions: ChatSession[];
  currentSessionId: string;
  personalDocsOpen: boolean;
  sidebarOpen: boolean;
  user?: AuthUser | null;
  onSelectSession: (id: string) => void;
  onCreateSession: () => void;
  onDeleteSession: (id: string) => void;
  onTogglePersonalDocs: () => void;
  onLogout: () => void;
}

export function Sidebar({
  sessions,
  currentSessionId,
  personalDocsOpen,
  sidebarOpen,
  user,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
  onTogglePersonalDocs,
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
