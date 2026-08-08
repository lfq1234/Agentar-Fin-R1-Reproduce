import type { BackendStatus } from "../types/agent";

interface Props {
  onToggleSidebar: () => void;
  backendStatus: BackendStatus;
}

// 顶部条：左 ☰ + 在线状态点。
// 「分析」已移到 Sidebar 个人文档下方，「新对话」在 Sidebar 底部，避免顶部拥挤。
export function Header({ onToggleSidebar, backendStatus }: Props) {
  return (
    <header className="app-header">
      <div className="app-title">
        <button className="hamburger" onClick={onToggleSidebar} aria-label="切换侧边栏">
          ☰
        </button>
        <span
          className={`status-bulb ${backendStatus}`}
          title={backendStatus === "ok" ? "后端在线" : backendStatus === "down" ? "后端离线" : "连接中…"}
        />
      </div>
    </header>
  );
}
