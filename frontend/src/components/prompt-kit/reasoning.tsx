import React from "react";
import { cn } from "../../lib/cn";
import { Markdown } from "./markdown";

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
  const actualOpen = isControlled ? open : internalOpen;

  const setOpen = React.useCallback(
    (nextOpen: boolean) => {
      if (!isControlled) {
        setInternalOpen(nextOpen);
      }
      onOpenChange?.(nextOpen);
    },
    [isControlled, onOpenChange]
  );

  React.useEffect(() => {
    if (isStreaming === false && actualOpen) {
      setOpen(false);
    }
  }, [isStreaming, actualOpen, setOpen]);

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
        {canRenderMarkdown ? <Markdown>{children}</Markdown> : children}
      </div>
    </div>
  );
}
