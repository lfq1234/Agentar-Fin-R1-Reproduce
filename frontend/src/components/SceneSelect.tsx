import { SCENE_OPTIONS, type SceneOption } from "../types/agent";

interface Props {
  value: SceneOption;
  onChange: (v: SceneOption) => void;
}

export function SceneSelect({ value, onChange }: Props) {
  return (
    <label className="scene-select">
      场景：
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as SceneOption)}
      >
        {SCENE_OPTIONS.map((s) => (
          <option key={s} value={s}>
            {s === "Auto" ? "自动路由" : s}
          </option>
        ))}
      </select>
    </label>
  );
}
