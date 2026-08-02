import { SceneSelect } from "./SceneSelect";
import type { BackendStatus, SceneOption } from "../types/agent";

interface Props {
  scene: SceneOption;
  onSceneChange: (v: SceneOption) => void;
  onReset: () => void;
  backendStatus: BackendStatus;
}

const STATUS_TEXT: Record<BackendStatus, string> = {
  unknown: "连接中…",
  ok: "后端在线",
  down: "后端离线",
};

export function Header({ scene, onSceneChange, onReset, backendStatus }: Props) {
  return (
    <header className="app-header">
      <div className="app-title">
        <h1>Agentar-Fin-R1 复现 · 交互演示</h1>
        <span className={`status-dot ${backendStatus}`}>{STATUS_TEXT[backendStatus]}</span>
      </div>
      <div className="header-actions">
        <SceneSelect value={scene} onChange={onSceneChange} />
        <button className="btn" onClick={onReset}>
          新对话
        </button>
      </div>
    </header>
  );
}
