interface Props {
  message: string | null;
  onDismiss: () => void;
}

// 后端 / 模型不可用时顶部提示，避免白屏（S3 / 验收清单）。
export function ErrorBanner({ message, onDismiss }: Props) {
  if (!message) return null;

  return (
    <div className="error-banner" role="alert">
      <span>{message}</span>
      <button className="error-close" onClick={onDismiss} aria-label="关闭">
        ×
      </button>
    </div>
  );
}
