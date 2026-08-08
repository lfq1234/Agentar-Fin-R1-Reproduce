import type { UiMessage } from "../types/agent";
import { AGENT_DISPLAY, type Participant } from "../types/agent";
import { ComplianceRisk } from "./ComplianceRisk";

// 把后端 agent_key 翻译成前端展示的「{name, avatar}」。未知键名用兜底圆圈，不让用户看到 raw key。
function toParticipants(keys: string[] | undefined | null): Participant[] {
  const list: Participant[] = [];
  for (const key of keys || []) {
    const m = AGENT_DISPLAY[key];
    list.push(m ? { key, name: m.name, avatar: m.avatar } : { key, name: key, avatar: "•" });
  }
  return list;
}

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

  // assistant：除正文外，尾部展示「本次参与者」chip 行，让用户
  // 一眼看出是 Direct 单模型直答（["Agentar"]）还是 Multi 圆桌（多专家）。
  const participants = isUser ? [] : toParticipants(msg.participants);

  return (
    <div className={`bubble-row ${isUser ? "user" : "assistant"}`}>
      <div className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}>
        <p className="bubble-content">{msg.content}</p>
        {!isUser && (msg.compliance || msg.risk) && (
          <ComplianceRisk compliance={msg.compliance ?? []} risk={msg.risk ?? []} />
        )}
        {!isUser && participants.length > 0 && (
          <div className="participants-row" aria-label="本轮参与的智能体">
            <span className="participants-label">本轮参与</span>
            <ul className="participants-chips">
              {participants.map((p) => (
                <li key={p.key} className="participant-chip" title={p.key}>
                  <span className="participant-avatar" aria-hidden>
                    {p.avatar}
                  </span>
                  <span className="participant-name">{p.name}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
