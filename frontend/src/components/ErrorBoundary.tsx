import { Component, type ErrorInfo, type ReactNode } from "react";
import { CircleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import { reportClientError, type ClientErrorSource } from "@/lib/client-diagnostics";

type FallbackArgs = {
  error: Error;
  reset: () => void;
};

type ErrorBoundaryProps = {
  children: ReactNode;
  /** Custom fallback; defaults to a calm inline notice. */
  fallback?: (args: FallbackArgs) => ReactNode;
  /** When any value here changes, a caught error is cleared and children re-render. */
  resetKeys?: ReadonlyArray<unknown>;
  /** Diagnostics label for where this boundary lives. */
  source?: ClientErrorSource;
  onError?: (error: Error, info: ErrorInfo) => void;
};

type ErrorBoundaryState = {
  error: Error | null;
};

const keysChanged = (
  previous: ReadonlyArray<unknown> | undefined,
  next: ReadonlyArray<unknown> | undefined
): boolean => {
  if (previous === next) {
    return false;
  }
  if (!previous || !next || previous.length !== next.length) {
    return true;
  }
  return previous.some((value, index) => !Object.is(value, next[index]));
};

/**
 * Contains render-time throws to a subtree instead of letting them reach the
 * single top-level boundary (which replaces the entire app). A failed message
 * or lazy chunk degrades to a small inline notice while the rest of the UI —
 * the transcript, the composer, navigation — stays alive and usable.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: unknown): ErrorBoundaryState {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: unknown, info: ErrorInfo): void {
    const normalized = error instanceof Error ? error : new Error(String(error));
    reportClientError(normalized, {
      source: this.props.source ?? "error-boundary",
      componentStack: info.componentStack ?? undefined,
    });
    this.props.onError?.(normalized, info);
  }

  componentDidUpdate(previousProps: ErrorBoundaryProps): void {
    if (this.state.error && keysChanged(previousProps.resetKeys, this.props.resetKeys)) {
      this.reset();
    }
  }

  reset = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    const { error } = this.state;
    if (!error) {
      return this.props.children;
    }
    if (this.props.fallback) {
      return this.props.fallback({ error, reset: this.reset });
    }
    return <InlineErrorFallback reset={this.reset} />;
  }
}

function InlineErrorFallback({ reset }: { reset: () => void }): ReactNode {
  return (
    <div
      role="alert"
      className="my-2 flex items-start gap-2.5 rounded-xl border bg-card px-3.5 py-3 text-sm shadow-sm"
    >
      <CircleAlert aria-hidden className="mt-0.5 size-4 shrink-0 text-destructive/80" />
      <div className="min-w-0 flex-1">
        <p className="font-medium text-foreground">This part couldn’t be displayed.</p>
        <p className="mt-0.5 text-muted-foreground">
          The rest of your conversation is unaffected.
        </p>
      </div>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={reset}
        className="shrink-0 rounded-full"
      >
        Try again
      </Button>
    </div>
  );
}
