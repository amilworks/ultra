import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import React from "react";
import { Markdown } from "./markdown";

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

export type MessageAvatarProps = {
  src?: string;
  alt?: string;
  fallback?: string;
  delayMs?: number;
  className?: string;
};

export function MessageAvatar({
  src,
  alt,
  fallback,
  delayMs,
  className,
}: MessageAvatarProps) {
  return (
    <Avatar className={cn("pk-message-avatar h-8 w-8 shrink-0", className)}>
      <AvatarImage src={src} alt={alt} />
      {fallback ? <AvatarFallback delayMs={delayMs}>{fallback}</AvatarFallback> : null}
    </Avatar>
  );
}

export type MessageContentProps = {
  children: React.ReactNode;
  markdown?: boolean;
  className?: string;
} & React.ComponentProps<typeof Markdown> &
  React.HTMLProps<HTMLDivElement>;

export function MessageContent({
  children,
  markdown = false,
  className,
  ...props
}: MessageContentProps) {
  const classNames = cn(
    "pk-message-content rounded-lg bg-secondary p-2 text-foreground break-words whitespace-normal",
    className
  );

  if (markdown) {
    return (
      <Markdown className={classNames} {...props}>
        {children as string}
      </Markdown>
    );
  }

  return (
    <div className={classNames} {...props}>
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
