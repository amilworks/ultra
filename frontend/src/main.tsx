import { Component, StrictMode, type ErrorInfo, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/inter/latin.css";
import "@fontsource/inter/latin-ext.css";
import "@fontsource/jetbrains-mono/latin.css";
import "@fontsource/jetbrains-mono/latin-ext.css";
import { App } from "./App";
import "./styles.css";

type AppErrorBoundaryProps = {
  children: ReactNode;
};

type AppErrorBoundaryState = {
  hasError: boolean;
  message: string;
};

class AppErrorBoundary extends Component<AppErrorBoundaryProps, AppErrorBoundaryState> {
  state: AppErrorBoundaryState = {
    hasError: false,
    message: "",
  };

  static getDerivedStateFromError(error: unknown): AppErrorBoundaryState {
    const message = error instanceof Error ? error.message : String(error);
    return { hasError: true, message };
  }

  componentDidCatch(error: unknown, errorInfo: ErrorInfo): void {
    // Avoid crashing the whole app on unexpected render errors.
    if (import.meta.env.DEV) {
      console.error("App render failure", error, errorInfo);
    }
  }

  render(): ReactNode {
    if (!this.state.hasError) {
      return this.props.children;
    }
    return (
      <main
        style={{
          minHeight: "100svh",
          display: "grid",
          placeItems: "center",
          padding: "24px",
          background: "var(--bg-page)",
          color: "var(--text-main)",
        }}
      >
        <div style={{ maxWidth: "640px", textAlign: "center" }}>
          <h1 style={{ margin: 0, fontSize: "1.25rem" }}>Something went wrong</h1>
          <p style={{ marginTop: "0.65rem", color: "var(--text-muted)" }}>
            The interface hit an unexpected render error.
          </p>
          {this.state.message ? (
            <p
              style={{
                marginTop: "0.65rem",
                fontFamily:
                  '"JetBrains Mono","SF Mono","Menlo",monospace',
                fontSize: "0.8rem",
                color: "var(--text-muted)",
              }}
            >
              {this.state.message}
            </p>
          ) : null}
          <button
            type="button"
            style={{
              marginTop: "1rem",
              border: "1px solid var(--line)",
              borderRadius: "999px",
              background: "var(--bg-panel-strong)",
              color: "var(--text-main)",
              padding: "8px 14px",
              cursor: "pointer",
            }}
            onClick={() => window.location.reload()}
          >
            Reload
          </button>
        </div>
      </main>
    );
  }
}

const root = document.getElementById("root");
if (!root) {
  throw new Error("Root element not found.");
}

createRoot(root).render(
  <StrictMode>
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  </StrictMode>
);
