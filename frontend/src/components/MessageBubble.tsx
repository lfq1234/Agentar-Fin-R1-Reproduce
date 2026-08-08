import type { UiMessage } from "../types/agent";
import { ComplianceRisk } from "./ComplianceRisk";

export function MessageBubble({ msg }: { msg: UiMessage }) {
  const isUser = msg.role === "user";
  const isAgent = msg.role === "agent";

  if (isAgent) {
    return (
      <div className="bubble-row agent">
        <div className="agent-avatar" title={msg.agent ?? "智能体"}>
          {msg.avatar ?? "🤖"}
        </div>
        <div className="agent-bubble">
          <div className="agent-header">
            <span className="agent-name">{msg.agent ?? "智能体"}</span>
            {msg.mention && (
              <span className="agent-mention">@{msg.mention}</span>
            )}
          </div>
          <p className="agent-content">{msg.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bubble-row ${isUser ? "user" : "assistant"}`}>
      <div className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}>
        <p className="bubble-content">{msg.content}</p>
        {!isUser && (msg.compliance || msg.risk) && (
          <ComplianceRisk compliance={msg.compliance ?? []} risk={msg.risk ?? []} />
        )}
      </div>
    </div>
  );
}
