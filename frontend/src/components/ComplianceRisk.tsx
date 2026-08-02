interface Props {
  compliance: string[];
  risk: string[];
}

// 合规提示 / 风险标记列表；两者皆空则完全隐藏（渲染规则：为空隐藏对应区块）。
export function ComplianceRisk({ compliance, risk }: Props) {
  if (compliance.length === 0 && risk.length === 0) return null;

  return (
    <div className="compliance-risk">
      {compliance.length > 0 && (
        <div className="cr-block">
          <span className="cr-label cr-ok">合规提示</span>
          <ul>
            {compliance.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
      )}
      {risk.length > 0 && (
        <div className="cr-block">
          <span className="cr-label cr-warn">风险标记</span>
          <ul>
            {risk.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
