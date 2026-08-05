import type { ChatSession } from "../types/agent";
import { SessionItem } from "./SessionItem";

interface Props {
  sessions: ChatSession[];
  currentSessionId: string;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

// 历史会话列表，按 updatedAt 倒序（需求目标 1）。
export function SessionList({ sessions, currentSessionId, onSelect, onDelete }: Props) {
  const sorted = [...sessions].sort((a, b) => b.updatedAt - a.updatedAt);
  if (sorted.length === 0) {
    return (
      <div className="session-list">
        <div className="empty-sessions">暂无历史对话</div>
      </div>
    );
  }
  return (
    <div className="session-list">
      {sorted.map((s) => (
        <SessionItem
          key={s.id}
          session={s}
          active={s.id === currentSessionId}
          onSelect={() => onSelect(s.id)}
          onDelete={() => onDelete(s.id)}
        />
      ))}
    </div>
  );
}
