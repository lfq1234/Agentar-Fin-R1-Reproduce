import type { AnalyzeResponse } from "../types/agent";

interface Props {
  result: AnalyzeResponse | null;
  loading: boolean;
  onClose: () => void;
}

// 结构化分析四字段分块展示（需求 §3 目标 4 / 技术 §6）。
export function AnalyzePanel({ result, loading, onClose }: Props) {
  if (!result && !loading) return null;

  return (
    <aside className="analyze-panel">
      <div className="analyze-head">
        <h3>结构化分析</h3>
        <button className="btn" onClick={onClose} disabled={loading}>
          关闭
        </button>
      </div>

      {loading && <p className="muted">分析中…</p>}

      {result && (
        <div className="analyze-body">
          <section>
            <h4>意图 (intent)</h4>
            <p>{result.intent || "—"}</p>
          </section>
          <section>
            <h4>槽位 (slots)</h4>
            {Object.keys(result.slots).length === 0 ? (
              <p>—</p>
            ) : (
              <ul>
                {Object.entries(result.slots).map(([k, v]) => (
                  <li key={k}>
                    <b>{k}</b>: {v}
                  </li>
                ))}
              </ul>
            )}
          </section>
          <section>
            <h4>工具规划 (tool_plan)</h4>
            {result.tool_plan.length === 0 ? (
              <p>—</p>
            ) : (
              <ol>
                {result.tool_plan.map((t, i) => (
                  <li key={i}>{t}</li>
                ))}
              </ol>
            )}
          </section>
          <section>
            <h4>表达 (expression)</h4>
            <p>{result.expression || "—"}</p>
          </section>
        </div>
      )}
    </aside>
  );
}
