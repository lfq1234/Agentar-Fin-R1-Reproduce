import type { ChatSession } from "../types/agent";
import { formatRelativeTime } from "../hooks/useChat";

interface Props {
  session: ChatSession;
  active: boolean;
  onSelect: () => void;
  onDelete: () => void;
}

// 单条会话项：标题 + 相对时间 + hover 删除按钮（需求目标 1）。
export function SessionItem({ session, active, onSelect, onDelete }: Props) {
  return (
    <div
      className={`session-item${active ? " active" : ""}`}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter") onSelect();
      }}
    >
      <div className="session-meta">
        <div className="session-title">{session.title}</div>
        <div className="session-time">{formatRelativeTime(session.updatedAt)}</div>
      </div>
      <button
        className="session-delete"
        aria-label="删除会话"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
      >
        ×
      </button>
    </div>
  );
}
