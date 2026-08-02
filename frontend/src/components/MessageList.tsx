import type { UiMessage } from "../types/agent";
import { MessageBubble } from "./MessageBubble";

export function MessageList({ messages }: { messages: UiMessage[] }) {
  if (messages.length === 0) {
    return (
      <div className="empty-hint">
        向金融 Agent 提问，验证意图识别 / 槽位填充 / 工具规划 / 表达生成。
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map((m, i) => (
        <MessageBubble key={i} msg={m} />
      ))}
    </div>
  );
}
