import { useEffect } from "react";

interface Props {
  message: string | null;
  onDismiss: () => void;
}

export function ErrorModal({ message, onDismiss }: Props) {
  useEffect(() => {
    if (!message) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onDismiss();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [message, onDismiss]);

  if (!message) return null;

  return (
    <div className="error-modal-overlay" role="presentation" onClick={onDismiss}>
      <div className="error-modal" role="alertdialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <div className="error-modal-head">
          <span className="error-modal-icon">⚠️</span>
          <h3>请求失败</h3>
        </div>
        <p className="error-modal-body">{message}</p>
        <div className="error-modal-actions">
          <button className="btn btn-primary" onClick={onDismiss}>
            知道了
          </button>
        </div>
      </div>
    </div>
  );
}
