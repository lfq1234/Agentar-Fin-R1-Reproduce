import { useChat } from "./hooks/useChat";
import { Header } from "./components/Header";
import { MessageList } from "./components/MessageList";
import { ChatInput } from "./components/ChatInput";
import { AnalyzePanel } from "./components/AnalyzePanel";
import { ErrorBanner } from "./components/ErrorBanner";

export function App() {
  const {
    message,
    setMessage,
    scene,
    setScene,
    history,
    loading,
    analyzing,
    analyzeResult,
    setAnalyzeResult,
    error,
    setError,
    backendStatus,
    send,
    runAnalyze,
    reset,
  } = useChat();

  // 最近一条助手回复作为分析目标（S4：全局分析入口，非每条消息内嵌）。
  const lastAssistant = [...history].reverse().find((m) => m.role === "assistant");

  return (
    <div className="app-shell">
      <Header scene={scene} onSceneChange={setScene} onReset={reset} backendStatus={backendStatus} />
      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      <div className="app-main">
        <section className="chat-area">
          <MessageList messages={history} />
          <div className="composer">
            <ChatInput value={message} onChange={setMessage} onSend={send} loading={loading} />
            <button
              className="btn"
              disabled={analyzing || !lastAssistant}
              onClick={() => lastAssistant && runAnalyze(lastAssistant.content)}
            >
              {analyzing ? "分析中…" : "分析最近回复"}
            </button>
          </div>
        </section>

        <AnalyzePanel result={analyzeResult} loading={analyzing} onClose={() => setAnalyzeResult(null)} />
      </div>
    </div>
  );
}
