import React from "react";
import { cn } from "../../lib/cn";
import { LazyMarkdown } from "./lazy-markdown";

type ReasoningContextValue = {
  open: boolean;
  setOpen: (value: boolean) => void;
};

const ReasoningContext = React.createContext<ReasoningContextValue | null>(null);

function useReasoningContext(): ReasoningContextValue {
  const context = React.useContext(ReasoningContext);
  if (!context) {
    throw new Error("Reasoning components must be used within Reasoning.");
  }
  return context;
}

type ReasoningProps = {
  children: React.ReactNode;
  className?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  isStreaming?: boolean;
};

export function Reasoning({
  children,
  className,
  open,
  onOpenChange,
  isStreaming,
}: ReasoningProps) {
  const [internalOpen, setInternalOpen] = React.useState(Boolean(open));
  const isControlled = typeof open === "boolean";
  const streamingEnded = isStreaming === false;
  const actualOpen = streamingEnded ? false : isControlled ? open : internalOpen;

  const setOpen = React.useCallback(
    (nextOpen: boolean) => {
      const nextActualOpen = streamingEnded ? false : nextOpen;
      if (!isControlled) {
        setInternalOpen(nextActualOpen);
      }
      onOpenChange?.(nextActualOpen);
    },
    [isControlled, onOpenChange, streamingEnded]
  );

  return (
    <ReasoningContext.Provider value={{ open: actualOpen, setOpen }}>
      <section className={cn("pk-reasoning", className)}>{children}</section>
    </ReasoningContext.Provider>
  );
}

export function ReasoningTrigger({
  children,
  className,
  onClick,
  ...props
}: React.HTMLAttributes<HTMLButtonElement>) {
  const { open, setOpen } = useReasoningContext();
  return (
    <button
      {...props}
      type="button"
      className={cn("pk-reasoning-trigger", className)}
      onClick={(event) => {
        onClick?.(event);
        if (!event.defaultPrevented) {
          setOpen(!open);
        }
      }}
    >
      {children}
      <span aria-hidden>{open ? "▲" : "▼"}</span>
    </button>
  );
}

type ReasoningContentProps = React.HTMLAttributes<HTMLDivElement> & {
  contentClassName?: string;
  markdown?: boolean;
};

export function ReasoningContent({
  children,
  className,
  contentClassName,
  markdown = false,
  ...props
}: ReasoningContentProps) {
  const { open } = useReasoningContext();
  if (!open) {
    return null;
  }

  const canRenderMarkdown = markdown && typeof children === "string";

  return (
    <div {...props} className={cn("pk-reasoning-content", className)}>
      <div className={cn("pk-reasoning-content-inner", contentClassName)}>
        {canRenderMarkdown ? (
          <React.Suspense
            fallback={<div style={{ whiteSpace: "pre-wrap" }}>{children}</div>}
          >
            <LazyMarkdown>{children}</LazyMarkdown>
          </React.Suspense>
        ) : (
          children
        )}
      </div>
    </div>
  );
}
