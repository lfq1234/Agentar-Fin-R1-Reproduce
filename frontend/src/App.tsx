import { useCallback } from "react";
import { useChat } from "./hooks/useChat";
import type { PersonalDocument } from "./types/agent";
import { Sidebar } from "./components/Sidebar";
import { Header } from "./components/Header";
import { MessageList } from "./components/MessageList";
import { ChatInput } from "./components/ChatInput";
import { AnalyzePanel } from "./components/AnalyzePanel";
import { ErrorModal } from "./components/ErrorModal";
import { PersonalDocsPanel } from "./components/PersonalDocsPanel";
import { SceneSelect } from "./components/SceneSelect";

const STATUS_TEXT: Record<string, string> = {
  unknown: "连接中…",
  ok: "后端在线",
  down: "后端离线",
};

function LeftToolbar({
  scene,
  onSceneChange,
  onNewSession,
  onAnalyze,
  canAnalyze,
  backendStatus,
}: {
  scene: Parameters<typeof SceneSelect>[0]["value"];
  onSceneChange: (v: Parameters<typeof SceneSelect>[0]["value"]) => void;
  onNewSession: () => void;
  onAnalyze: () => void;
  canAnalyze: boolean;
  backendStatus: string;
}) {
  return (
    <aside className="chat-leftbar">
      <div className="leftbar-group">
        <label className="leftbar-label">场景</label>
        <SceneSelect value={scene} onChange={onSceneChange} />
      </div>

      <div className="leftbar-group">
        <label className="leftbar-label">操作</label>
        <button className="leftbar-btn" onClick={onAnalyze} disabled={!canAnalyze} title="分析最近一条助手回复">
          分析
        </button>
        <button className="leftbar-btn" onClick={onNewSession}>
          新对话
        </button>
      </div>

      <div className="leftbar-footer">
        <span className={`status-bulb ${backendStatus}`} title={STATUS_TEXT[backendStatus]} />
        <span className="leftbar-status">{STATUS_TEXT[backendStatus]}</span>
      </div>
    </aside>
  );
}

export function App() {
  const {
    sessions,
    currentSessionId,
    selectSession,
    createSession,
    deleteSession,
    personalDocsOpen,
    togglePersonalDocs,
    sidebarOpen,
    toggleSidebar,
    history,
    scene,
    setScene,
    lastAssistant,
    message,
    setMessage,
    loading,
    analyzing,
    analyzeResult,
    setAnalyzeResult,
    error,
    setError,
    backendStatus,
    send,
    runAnalyze,
  } = useChat();

  const isEmpty = history.length === 0;

  // 03 联动：把实体/文档摘要追加到当前会话输入框（关闭面板后可见），技术文档 §8。
  const handleInjectContext = useCallback(
    (text: string) => {
      setMessage((prev) => (prev ? prev + "\n" + text : text));
      togglePersonalDocs();
    },
    [setMessage, togglePersonalDocs],
  );

  // 03 联动：新建会话并以文档摘要作为首条上下文（需求 FR7）。
  const handleAskWithDocument = useCallback(
    (doc: PersonalDocument) => {
      createSession();
      const summary = doc.summary
        ? `关于文档《${doc.filename}》：${doc.summary}`
        : `关于文档《${doc.filename}》，请基于其内容回答：`;
      setMessage(summary);
    },
    [createSession, setMessage],
  );

  return (
    <div className="app-shell">
      <Sidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        personalDocsOpen={personalDocsOpen}
        sidebarOpen={sidebarOpen}
        onSelectSession={selectSession}
        onCreateSession={createSession}
        onDeleteSession={deleteSession}
        onTogglePersonalDocs={togglePersonalDocs}
      />

      <ErrorModal message={error} onDismiss={() => setError(null)} />

      <div className="app-main">
        {personalDocsOpen ? (
          <PersonalDocsPanel
            onClose={() => togglePersonalDocs()}
            onInjectContext={handleInjectContext}
            onAskWithDocument={handleAskWithDocument}
          />
        ) : isEmpty ? (
          <div className="hero-page">
            <button className="hero-hamburger" onClick={toggleSidebar} aria-label="切换侧边栏">
              ☰
            </button>
            <div className="hero-content">
              <h1 className="hero-title">向金融 Agent 提问</h1>
              <p className="hero-subtitle">验证意图识别 / 槽位填充 / 工具规划 / 表达生成</p>
              <ChatInput
                variant="hero"
                value={message}
                onChange={setMessage}
                onSend={send}
                loading={loading}
              />
              <div className="hero-footer">
                <SceneSelect value={scene} onChange={setScene} />
                <span className={`status-dot ${backendStatus}`}>{STATUS_TEXT[backendStatus]}</span>
              </div>
            </div>
          </div>
        ) : (
          <>
            <Header onToggleSidebar={toggleSidebar} backendStatus={backendStatus} />

            <LeftToolbar
              scene={scene}
              onSceneChange={setScene}
              onNewSession={createSession}
              onAnalyze={runAnalyze}
              canAnalyze={!!lastAssistant}
              backendStatus={backendStatus}
            />

            <section className="chat-area">
              <MessageList messages={history} />
              <div className="composer composer-floating">
                <ChatInput
                  variant="hero"
                  value={message}
                  onChange={setMessage}
                  onSend={send}
                  loading={loading}
                />
              </div>
            </section>

            <AnalyzePanel result={analyzeResult} loading={analyzing} onClose={() => setAnalyzeResult(null)} />
          </>
        )}
      </div>
    </div>
  );
}
