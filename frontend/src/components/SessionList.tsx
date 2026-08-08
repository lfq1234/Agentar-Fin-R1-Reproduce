import type { ChatSession } from "../types/agent";
import { SessionItem } from "./SessionItem";

interface Props {
  sessions: ChatSession[];
  currentSessionId: string;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

// 历史会话列表，按 updatedAt 倒序（需求目标 1）。
// 过滤规则：本地空白草稿（无后端 conversationId 且未开聊）只在「当前激活」
// 时显示——避免侧边栏出现「本地占位 + 后端真会话」这种重复占位。
function isVisible(s: ChatSession, currentId: string): boolean {
  const isEmptyDraft = s.conversationId == null && s.history.length === 0;
  if (isEmptyDraft && s.id !== currentId) return false;
  return true;
}

export function SessionList({ sessions, currentSessionId, onSelect, onDelete }: Props) {
  const sorted = [...sessions]
    .filter((s) => isVisible(s, currentSessionId))
    .sort((a, b) => b.updatedAt - a.updatedAt);
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
