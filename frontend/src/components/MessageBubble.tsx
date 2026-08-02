import type { UiMessage } from "../types/agent";
import { ComplianceRisk } from "./ComplianceRisk";

export function MessageBubble({ msg }: { msg: UiMessage }) {
  const isUser = msg.role === "user";

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
