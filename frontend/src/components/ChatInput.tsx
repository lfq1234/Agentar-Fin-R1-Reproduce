interface Props {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  loading: boolean;
  variant?: "compact" | "hero";
}

export function ChatInput({ value, onChange, onSend, loading, variant = "compact" }: Props) {
  const isHero = variant === "hero";
  return (
    <div className={`chat-input ${isHero ? "chat-input-hero" : ""}`}>
      {isHero && (
        <button
          type="button"
          className="input-round-btn input-round-btn-outline"
          title="附件（待实现）"
          onClick={() => alert("附件上传功能待实现")}
        >
          +
        </button>
      )}
      <input
        className="chat-input-field"
        value={value}
        placeholder={isHero ? "有什么我能帮您的吗？" : "例如：瑞士法郎兑加元现在报价多少？"}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSend();
          }
        }}
        disabled={loading}
      />
      <button
        className={`input-round-btn ${isHero ? "input-round-btn-primary" : "btn btn-primary"}`}
        onClick={onSend}
        disabled={loading || !value.trim()}
        aria-label="发送"
      >
        {loading ? "…" : "➤"}
      </button>
    </div>
  );
}
