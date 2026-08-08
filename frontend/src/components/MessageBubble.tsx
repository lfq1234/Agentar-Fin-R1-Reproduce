import type { UiMessage } from "../types/agent";
import { AGENT_DISPLAY } from "../types/agent";
import { ComplianceRisk } from "./ComplianceRisk";

// 后端 agent 字段是 AgentScope 名字键（如 "BankingExpert"），前端按名字从
// AGENT_DISPLAY 映射出中文名 + 头像 + 主题色。未知键名兜底，不让用户看到 raw key。
function agentDisplay(key: string | undefined): {
  key: string;
  name: string;
  avatar: string;
  color: string;
} {
  const k = key ?? "";
  const m = AGENT_DISPLAY[k];
  if (m) return { key: k, name: m.name, avatar: m.avatar, color: m.color };
  return { key: k, name: k || "智能体", avatar: "🤖", color: "#64748b" };
}

export function MessageBubble({ msg }: { msg: UiMessage }) {
  const isUser = msg.role === "user";
  const isAgent = msg.role === "agent";

  if (isAgent) {
    const a = agentDisplay(msg.agent);
    return (
      <div className="bubble-row agent">
        <div className="agent-avatar" title={a.name} style={{ background: a.color }}>
          {a.avatar}
        </div>
        <div className="agent-bubble">
          <div className="agent-header">
            <span className="agent-name">{a.name}</span>
            {msg.mention && <span className="agent-mention">@{msg.mention}</span>}
          </div>
          <p className="agent-content">{msg.content}</p>
        </div>
      </div>
    );
  }

  // assistant / 最终回复：头部显式标注「谁综合作答」（头像 + 名字 + 主题色），
  // 直接回应「看不到哪个智能体回答」的诉求——Direct 单答也一眼可见。
  const speaker = agentDisplay(msg.agent);
  // 本轮参与者（从 res.agent_trace 抽取；Direct 退化为 ["Agentar"]），去重保序。
  const participants = (msg.participants ?? [])
    .map((k) => agentDisplay(k))
    .filter((p, i, arr) => arr.findIndex((x) => x.key === p.key) === i);

  return (
    <div className={`bubble-row ${isUser ? "user" : "assistant"}`}>
      <div className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}>
        {!isUser && (
          <div className="msg-speaker">
            <span className="msg-avatar" aria-hidden style={{ background: speaker.color }}>
              {speaker.avatar}
            </span>
            <span className="msg-speaker-name">{speaker.name}</span>
            <span className="msg-speaker-tag">作答</span>
          </div>
        )}
        <p className="bubble-content">{msg.content}</p>
        {!isUser && (msg.compliance || msg.risk) && (
          <ComplianceRisk compliance={msg.compliance ?? []} risk={msg.risk ?? []} />
        )}
        {!isUser && participants.length > 0 && (
          <div className="participants-row" aria-label="本轮参与的智能体">
            <span className="participants-label">参与</span>
            <ul className="participants-chips">
              {participants.map((p) => (
                <li key={p.key} className="participant-chip" title={p.name}>
                  <span
                    className="participant-avatar"
                    aria-hidden
                    style={{ background: p.color }}
                  >
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
