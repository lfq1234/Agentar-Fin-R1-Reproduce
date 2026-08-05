interface Props {
  onClose: () => void;
}

// 个人文档占位面板（MVP，仅入口与占位文案；需求 §2.2 排除具体功能）。
export function PersonalDocsPanel({ onClose }: Props) {
  return (
    <div className="personal-docs-panel">
      <div className="personal-docs-head">
        <h2>个人文档</h2>
        <button className="btn" onClick={onClose}>
          关闭
        </button>
      </div>
      <p className="muted">
        个人文档功能即将上线，可用于上传 / 管理您的金融知识库文档。
      </p>
    </div>
  );
}
