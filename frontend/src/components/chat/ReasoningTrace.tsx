import { ChevronDown } from "lucide-react";

import { cn } from "@/lib/utils";

export type ReasoningTraceProps = {
  /** The coordinator's accumulated chain-of-thought for this turn. */
  text: string;
  className?: string;
};

// The persisted, collapsed-by-default "Thought process" disclosure shown beneath a completed answer,
// so a reader can open the model's reasoning after the turn (the live ChatRunSteps timeline unmounts
// at turn end). A native <details> — no framer-motion, no infinite animation — keeps it
// reduced-motion friendly and consistent with the live-reasoning affordance.
export function ReasoningTrace({ text, className }: ReasoningTraceProps) {
  const trimmed = text.trim();
  if (!trimmed) {
    return null;
  }
  return (
    <details className={cn("reasoning-trace", className)}>
      <summary className="reasoning-trace-summary">
        <ChevronDown className="reasoning-trace-chevron size-3.5" aria-hidden />
        <span>Thought process</span>
      </summary>
      <div className="reasoning-trace-body">{trimmed}</div>
    </details>
  );
}
