interface Props {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  loading: boolean;
}

export function ChatInput({ value, onChange, onSend, loading }: Props) {
  return (
    <div className="chat-input">
      <input
        className="chat-input-field"
        value={value}
        placeholder="例如：瑞士法郎兑加元现在报价多少？"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSend();
          }
        }}
        disabled={loading}
      />
      <button className="btn btn-primary" onClick={onSend} disabled={loading}>
        {loading ? "…" : "发送"}
      </button>
    </div>
  );
}
