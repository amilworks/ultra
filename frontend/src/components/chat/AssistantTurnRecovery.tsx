import { Button } from "@/components/ui/button";
import { CircleAlert, CircleSlash, Pencil, RotateCcw } from "lucide-react";

type AssistantTurnRecoveryProps = {
  status: "stopped" | "failed";
  errorReason?: string;
  onRetry: () => void;
  onEdit: () => void;
};

// Calm inline recovery affordance for a stopped or failed assistant turn —
// understated muted status line + ghost Retry/Edit actions, matching the app's
// "Could not …" voice. Replaces the previous silent dead-end.
export function AssistantTurnRecovery({
  status,
  errorReason,
  onRetry,
  onEdit,
}: AssistantTurnRecoveryProps) {
  const stopped = status === "stopped";
  return (
    <div className="mt-1.5 flex flex-col gap-1.5" role="status" aria-live="polite">
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-sm">
        <span className="inline-flex items-center gap-1.5 text-muted-foreground">
          {stopped ? (
            <CircleSlash aria-hidden className="size-3.5" />
          ) : (
            <CircleAlert aria-hidden className="size-3.5 text-destructive/70" />
          )}
          {stopped ? "You stopped this response." : "This response couldn’t be completed."}
        </span>
        <div className="flex items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 gap-1.5 rounded-full px-2.5 text-muted-foreground hover:text-foreground"
            onClick={onRetry}
          >
            <RotateCcw className="size-3.5" />
            Retry
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 gap-1.5 rounded-full px-2.5 text-muted-foreground hover:text-foreground"
            onClick={onEdit}
          >
            <Pencil className="size-3.5" />
            Edit
          </Button>
        </div>
      </div>
      {!stopped && errorReason ? (
        <p className="font-mono text-[11px] leading-relaxed text-muted-foreground/75">
          {errorReason}
        </p>
      ) : null}
    </div>
  );
}
