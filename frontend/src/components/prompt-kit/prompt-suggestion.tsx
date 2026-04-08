import React from "react";
import { cn } from "../../lib/cn";
import { Button } from "@/components/ui/button";

type PromptSuggestionProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "destructive" | "outline" | "ghost";
  size?: "default" | "sm" | "lg" | "icon";
  highlight?: string;
};

const highlightText = (text: string, highlight?: string): React.ReactNode => {
  const query = highlight?.trim();
  if (!query) {
    return text;
  }

  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "ig");
  const parts = text.split(regex);
  if (parts.length <= 1) {
    return text;
  }

  return parts.map((part, index) => {
    if (part.toLowerCase() === query.toLowerCase()) {
      return (
        <mark key={`${part}-${index}`} className="pk-prompt-suggestion-highlight">
          {part}
        </mark>
      );
    }
    return <React.Fragment key={`${part}-${index}`}>{part}</React.Fragment>;
  });
};

export function PromptSuggestion({
  children,
  variant = "outline",
  size = "lg",
  highlight,
  className,
  ...props
}: PromptSuggestionProps) {
  const content =
    typeof children === "string" ? highlightText(children, highlight) : children;

  return (
    <Button
      {...props}
      type={props.type ?? "button"}
      variant={
        variant === "destructive"
          ? "destructive"
          : variant === "ghost"
            ? "ghost"
            : variant === "default"
              ? "default"
              : "outline"
      }
      size={size === "icon" ? "icon" : size === "sm" ? "sm" : size === "lg" ? "default" : "default"}
      className={cn(
        "pk-prompt-suggestion",
        `pk-prompt-suggestion-${variant}`,
        `pk-prompt-suggestion-${size}`,
        className
      )}
    >
      {content}
    </Button>
  );
}
