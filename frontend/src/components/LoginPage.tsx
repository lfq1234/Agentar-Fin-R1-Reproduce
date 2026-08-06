import { useState } from "react";
import { login, register, type ApiError } from "../api/client";
import type { AuthUser, LoginCredentials, RegisterPayload } from "../types/agent";

// 09：登录 / 注册门禁页。对齐现有 hero 风格（居中卡片、主色按钮）。
export function LoginPage({ onLoggedIn }: { onLoggedIn: (u: AuthUser) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "login") {
        const creds: LoginCredentials = { username: username.trim(), password };
        const user = await login(creds);
        onLoggedIn(user);
      } else {
        const payload: RegisterPayload = {
          username: username.trim(),
          password,
          email: email.trim() || undefined,
        };
        await register(payload);
        // 注册成功后自动登录，直接进入聊天界面。
        const user = await login({ username: username.trim(), password });
        onLoggedIn(user);
      }
    } catch (err) {
      setError((err as ApiError).message ?? "操作失败，请重试");
    } finally {
      setBusy(false);
    }
  };

  const switchMode = () => {
    setMode((m) => (m === "login" ? "register" : "login"));
    setError(null);
  };

  return (
    <div className="auth-page">
      <form className="auth-card" onSubmit={submit}>
        <h1 className="auth-title">{mode === "login" ? "登录金融 Agent" : "注册新账号"}</h1>
        <p className="auth-sub">
          {mode === "login" ? "登录后对话与文档按账号隔离" : "账号 3-50 字，密码至少 8 位"}
        </p>

        <label className="auth-label" htmlFor="auth-username">账号</label>
        <input
          id="auth-username"
          className="auth-input"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="用户名"
          autoFocus
          autoComplete="username"
        />

        {mode === "register" && (
          <>
            <label className="auth-label" htmlFor="auth-email">邮箱（可选）</label>
            <input
              id="auth-email"
              className="auth-input"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              autoComplete="email"
            />
          </>
        )}

        <label className="auth-label" htmlFor="auth-password">密码</label>
        <input
          id="auth-password"
          className="auth-input"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="密码"
          autoComplete={mode === "login" ? "current-password" : "new-password"}
        />

        {error && <div className="auth-error">{error}</div>}

        <button className="btn btn-primary auth-submit" type="submit" disabled={busy}>
          {busy ? "处理中…" : mode === "login" ? "登录" : "注册并登录"}
        </button>

        <button type="button" className="auth-switch" onClick={switchMode}>
          {mode === "login" ? "没有账号？去注册" : "已有账号？去登录"}
        </button>

        {mode === "login" && (
          <div className="auth-hint">体验账号：alice / alice1234，bob / bob1234</div>
        )}
      </form>
    </div>
  );
}
