import { useCallback, useEffect, useState } from "react";
import { useChat } from "./hooks/useChat";
import { clearToken, getToken, me, UNAUTHORIZED_EVENT } from "./api/client";
import type { AuthUser, PersonalDocument } from "./types/agent";
import { Sidebar } from "./components/Sidebar";
import { Header } from "./components/Header";
import { MessageList } from "./components/MessageList";
import { ChatInput } from "./components/ChatInput";
import { AnalyzePanel } from "./components/AnalyzePanel";
import { ErrorModal } from "./components/ErrorModal";
import { PersonalDocsPanel } from "./components/PersonalDocsPanel";
import { LoginPage } from "./components/LoginPage";

const STATUS_TEXT: Record<string, string> = {
  unknown: "连接中…",
  ok: "后端在线",
  down: "后端离线",
};

function LeftToolbar({
  onNewSession,
  onAnalyze,
  canAnalyze,
  backendStatus,
}: {
  onNewSession: () => void;
  onAnalyze: () => void;
  canAnalyze: boolean;
  backendStatus: string;
}) {
  return (
    <aside className="chat-leftbar">
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
  // 09：登录门禁。启动时若有令牌则校验有效性，无效则清令牌回到登录页。
  const [user, setUser] = useState<AuthUser | null>(null);
  const [booting, setBooting] = useState(true);

  const logout = useCallback(() => {
    clearToken();
    setUser(null);
  }, []);

  // 启动校验：携带本地令牌请求 /auth/me，成功则进入聊天界面，失败则回到登录页。
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!getToken()) {
        setBooting(false);
        return;
      }
      try {
        const u = await me();
        if (!cancelled) setUser(u);
      } catch {
        if (!cancelled) clearToken();
      } finally {
        if (!cancelled) setBooting(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // 运行时鉴权失败（401/403）由 client 派发，统一登出。
  useEffect(() => {
    const onUnauthorized = () => setUser(null);
    window.addEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
  }, []);

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

  // 未登录：渲染登录页（启动校验中显示最小占位，避免闪烁）。
  if (booting) {
    return (
      <div className="app-shell">
        <div className="auth-boot">校验登录中…</div>
      </div>
    );
  }
  if (!user) {
    return <LoginPage onLoggedIn={(u) => setUser(u)} />;
  }

  return (
    <div className="app-shell">
      <Sidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        personalDocsOpen={personalDocsOpen}
        sidebarOpen={sidebarOpen}
        user={user}
        onSelectSession={selectSession}
        onCreateSession={createSession}
        onDeleteSession={deleteSession}
        onTogglePersonalDocs={togglePersonalDocs}
        onLogout={logout}
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
                <span className={`status-dot ${backendStatus}`}>{STATUS_TEXT[backendStatus]}</span>
              </div>
            </div>
          </div>
        ) : (
          <>
            <Header onToggleSidebar={toggleSidebar} backendStatus={backendStatus} />

            <LeftToolbar
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
