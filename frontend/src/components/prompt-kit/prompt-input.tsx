import React from "react";
import { cn } from "../../lib/cn";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

type PromptInputContextValue = {
  value: string;
  onValueChange?: (value: string) => void;
  onSubmit?: () => void;
  isLoading: boolean;
  maxHeight: number | string;
  disabled?: boolean;
};

const PromptInputContext = React.createContext<PromptInputContextValue | null>(null);

type PromptInputProps = {
  isLoading?: boolean;
  value: string;
  onValueChange?: (value: string) => void;
  maxHeight?: number | string;
  onSubmit?: () => void;
  children?: React.ReactNode;
  className?: string;
  disabled?: boolean;
};

export function PromptInput({
  isLoading = false,
  value,
  onValueChange,
  maxHeight = 240,
  onSubmit,
  children,
  className,
  disabled = false,
}: PromptInputProps) {
  return (
    <TooltipProvider>
      <PromptInputContext.Provider
        value={{ value, onValueChange, onSubmit, isLoading, maxHeight, disabled }}
      >
        <form
          className={cn("pk-prompt-input", className)}
          onSubmit={(event) => {
            event.preventDefault();
            if (!isLoading && !disabled) {
              onSubmit?.();
            }
          }}
        >
          {children}
        </form>
      </PromptInputContext.Provider>
    </TooltipProvider>
  );
}

type PromptInputTextareaProps = React.ComponentProps<"textarea"> & {
  disableAutosize?: boolean;
};

export const PromptInputTextarea = React.forwardRef<
  HTMLTextAreaElement,
  PromptInputTextareaProps
>(function PromptInputTextarea(
  {
    disableAutosize = false,
    className,
    onKeyDown,
    onChange,
    value,
    ...props
  },
  forwardedRef
) {
  const context = React.useContext(PromptInputContext);
  const textareaRef = React.useRef<HTMLTextAreaElement | null>(null);
  const controlledValue = value ?? context?.value ?? "";
  const maxHeight = context?.maxHeight ?? 240;

  React.useLayoutEffect(() => {
    if (disableAutosize) {
      return;
    }
    const node = textareaRef.current;
    if (!node) {
      return;
    }
    node.style.height = "auto";
    const nextHeight = Math.min(
      node.scrollHeight,
      typeof maxHeight === "number" ? maxHeight : Number(maxHeight) || 240
    );
    node.style.height = `${Math.max(44, nextHeight)}px`;
  }, [controlledValue, disableAutosize, maxHeight]);

  return (
    <textarea
      {...props}
      ref={(node) => {
        textareaRef.current = node;
        if (typeof forwardedRef === "function") {
          forwardedRef(node);
        } else if (forwardedRef) {
          forwardedRef.current = node;
        }
      }}
      value={controlledValue}
      className={cn("pk-prompt-input-textarea", className)}
      disabled={context?.isLoading || context?.disabled || props.disabled}
      onChange={(event) => {
        context?.onValueChange?.(event.target.value);
        onChange?.(event);
      }}
      onKeyDown={(event) => {
        onKeyDown?.(event);
      }}
    />
  );
});

export function PromptInputActions({
  children,
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("flex items-center gap-2", className)} {...props}>
      {children}
    </div>
  );
}

type PromptInputActionProps = {
  className?: string;
  tooltip: React.ReactNode;
  children: React.ReactNode;
  side?: "top" | "bottom" | "left" | "right";
} & React.ComponentProps<typeof Tooltip>;

export function PromptInputAction({
  tooltip,
  className,
  children,
  side = "top",
  ...props
}: PromptInputActionProps) {
  const disabled = Boolean(React.useContext(PromptInputContext)?.disabled);

  return (
    <Tooltip {...props}>
      <TooltipTrigger
        asChild
        disabled={disabled}
        onClick={(event) => event.stopPropagation()}
      >
        {children}
      </TooltipTrigger>
      <TooltipContent side={side} className={className}>
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}
