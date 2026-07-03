import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import React from "react";
import { LazyMarkdown } from "./lazy-markdown";
import type { MarkdownProps } from "./markdown";

export type MessageProps = {
  children: React.ReactNode;
  className?: string;
} & React.HTMLProps<HTMLDivElement>;

export function Message({
  children,
  className,
  ...props
}: MessageProps) {
  return (
    <div className={cn("pk-message flex gap-3", className)} {...props}>
      {children}
    </div>
  );
}

export type MessageContentProps = {
  children: React.ReactNode;
  markdown?: boolean;
  className?: string;
} & Omit<MarkdownProps, "children" | "className"> &
  React.HTMLAttributes<HTMLDivElement>;

export function MessageContent({
  children,
  markdown = false,
  className,
  components,
  id,
  ...props
}: MessageContentProps) {
  const classNames = cn(
    "pk-message-content rounded-lg bg-secondary p-2 text-foreground break-words whitespace-normal",
    className
  );

  if (markdown) {
    const markdownText = typeof children === "string" ? children : String(children ?? "");
    return (
      <React.Suspense
        fallback={
          <div id={id} className={classNames} {...props}>
            <div className="pk-message-content-plain whitespace-pre-wrap">
              {markdownText}
            </div>
          </div>
        }
      >
        <LazyMarkdown id={id} components={components} className={classNames}>
          {markdownText}
        </LazyMarkdown>
      </React.Suspense>
    );
  }

  return (
    <div id={id} className={classNames} {...props}>
      <div className="pk-message-content-plain">{children}</div>
    </div>
  );
}

export type MessageActionsProps = {
  children: React.ReactNode;
  className?: string;
} & React.HTMLProps<HTMLDivElement>;

export function MessageActions({
  children,
  className,
  ...props
}: MessageActionsProps) {
  return (
    <div className={cn("pk-message-actions text-muted-foreground flex items-center gap-2", className)} {...props}>
      {children}
    </div>
  );
}

export type MessageActionProps = {
  className?: string;
  tooltip: React.ReactNode;
  children: React.ReactNode;
  side?: "top" | "bottom" | "left" | "right";
} & React.ComponentProps<typeof Tooltip>;

export function MessageAction({
  tooltip,
  children,
  className,
  side = "top",
  ...props
}: MessageActionProps) {
  return (
    <TooltipProvider>
      <Tooltip {...props}>
        <TooltipTrigger asChild>{children}</TooltipTrigger>
        <TooltipContent side={side} className={className}>
          {tooltip}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
