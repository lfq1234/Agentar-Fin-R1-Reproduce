interface Props {
  scale: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  types: string[];
  selectedTypes: string[];
  onToggleType: (t: string) => void;
  docOptions: { id: string; name: string }[];
  selectedDocId: string;
  onSelectDoc: (id: string) => void;
}

// 图谱浮动控件：缩放 / 重置 / 按类型筛选 / 按文档筛选（需求 FR5, FR6）。
export function GraphControls({
  scale,
  onZoomIn,
  onZoomOut,
  onReset,
  types,
  selectedTypes,
  onToggleType,
  docOptions,
  selectedDocId,
  onSelectDoc,
}: Props) {
  return (
    <div className="graph-controls" role="toolbar" aria-label="图谱控件">
      <div className="graph-zoom">
        <button className="graph-ctrl-btn" onClick={onZoomOut} aria-label="缩小" title="缩小">
          −
        </button>
        <span className="graph-zoom-val">{Math.round(scale * 100)}%</span>
        <button className="graph-ctrl-btn" onClick={onZoomIn} aria-label="放大" title="放大">
          ＋
        </button>
        <button className="graph-ctrl-btn" onClick={onReset} aria-label="重置视图" title="重置视图">
          ⟲
        </button>
      </div>

      <div className="graph-filter">
        <label className="graph-filter-label">类型</label>
        <div className="graph-filter-types">
          {types.map((t) => (
            <label key={t} className="graph-type-chip">
              <input
                type="checkbox"
                checked={selectedTypes.includes(t)}
                onChange={() => onToggleType(t)}
              />
              {t}
            </label>
          ))}
          {types.length === 0 && <span className="muted">暂无</span>}
        </div>
        <label className="graph-filter-label" htmlFor="graph-doc-filter">
          来源文档
        </label>
        <select
          id="graph-doc-filter"
          className="graph-doc-select"
          value={selectedDocId}
          onChange={(e) => onSelectDoc(e.target.value)}
        >
          <option value="all">全部文档</option>
          {docOptions.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
