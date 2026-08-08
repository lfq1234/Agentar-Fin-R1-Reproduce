interface Props {
  onToggleSidebar: () => void;
}

// 顶部条：仅保留汉堡菜单。
// 「分析」与「后端状态」均已移到 Sidebar，避免顶部拥挤。
export function Header({ onToggleSidebar }: Props) {
  return (
    <header className="app-header">
      <div className="app-title">
        <button className="hamburger" onClick={onToggleSidebar} aria-label="切换侧边栏">
          ☰
        </button>
      </div>
    </header>
  );
}
