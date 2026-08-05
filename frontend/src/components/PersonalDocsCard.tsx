interface Props {
  active: boolean;
  onClick: () => void;
}

// 左上角个人文档入口卡片（需求目标 2）。
export function PersonalDocsCard({ active, onClick }: Props) {
  return (
    <button
      className={`personal-docs-card${active ? " active" : ""}`}
      onClick={onClick}
      aria-pressed={active}
    >
      <span className="personal-docs-icon" aria-hidden>
        📄
      </span>
      <span>个人文档</span>
    </button>
  );
}
