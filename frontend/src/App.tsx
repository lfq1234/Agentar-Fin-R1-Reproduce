import { useState } from "react";

const SCENES = ["Banking", "Securities", "Insurance", "Trust", "MutualFunds"] as const;

export function App() {
  const [message, setMessage] = useState("");
  const [scene, setScene] = useState<string>(SCENES[0]);
  const [reply, setReply] = useState("");
  const [loading, setLoading] = useState(false);

  async function send() {
    if (!message.trim()) return;
    setLoading(true);
    try {
      const res = await fetch("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, scene }),
      });
      const data = await res.json();
      setReply(data.reply ?? JSON.stringify(data));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 760, margin: "40px auto", fontFamily: "system-ui, sans-serif" }}>
      <h1>Agentar-Fin-R1 复现 · 交互演示</h1>
      <p style={{ color: "#666" }}>
        向金融 Agent 提问，验证意图识别 / 槽位填充 / 工具规划 / 表达生成。
      </p>

      <label style={{ display: "block", marginBottom: 12 }}>
        场景：
        <select value={scene} onChange={(e) => setScene(e.target.value)}>
          {SCENES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>

      <div style={{ display: "flex", gap: 8 }}>
        <input
          style={{ flex: 1, padding: 8 }}
          value={message}
          placeholder="例如：瑞士法郎兑加元现在报价多少？"
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
        />
        <button onClick={send} disabled={loading} style={{ padding: "8px 16px" }}>
          {loading ? "…" : "发送"}
        </button>
      </div>

      {reply && (
        <pre
          style={{
            background: "#f5f5f5",
            padding: 16,
            marginTop: 16,
            whiteSpace: "pre-wrap",
            borderRadius: 8,
          }}
        >
          {reply}
        </pre>
      )}
    </main>
  );
}
