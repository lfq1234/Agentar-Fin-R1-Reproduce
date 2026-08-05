import type { ChatSession } from "../types/agent";
import { PersonalDocsCard } from "./PersonalDocsCard";
import { SessionList } from "./SessionList";

interface Props {
  sessions: ChatSession[];
  currentSessionId: string;
  personalDocsOpen: boolean;
  sidebarOpen: boolean;
  onSelectSession: (id: string) => void;
  onCreateSession: () => void;
  onDeleteSession: (id: string) => void;
  onTogglePersonalDocs: () => void;
}

export function Sidebar({
  sessions,
  currentSessionId,
  personalDocsOpen,
  sidebarOpen,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
  onTogglePersonalDocs,
}: Props) {
  return (
    <aside className={`sidebar${sidebarOpen ? " open" : ""}`}>
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
