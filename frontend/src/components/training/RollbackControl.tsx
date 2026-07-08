// The rollback + override + dismiss + freeze dialogs (§14.6 R/O/D/F).
// Rollback is instant and UNGATED; the override requires a recorded reason
// (validated by submit-attempt + inline error, never a disabled confirm).

import { useState } from "react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Label } from "@/components/ui/label";

export type TrainingDialogKind = "rollback" | "override" | "dismiss" | "freeze" | null;

export function TrainingDialogs({
  kind,
  targetVersionId,
  activeVersionId,
  goldDraftSummary,
  onConfirm,
  onClose,
}: {
  kind: TrainingDialogKind;
  targetVersionId: string;
  activeVersionId: string;
  goldDraftSummary: string;
  onConfirm: (kind: Exclude<TrainingDialogKind, null>, reason?: string) => void;
  onClose: () => void;
}) {
  const [reason, setReason] = useState("");
  const [reasonError, setReasonError] = useState(false);

  const close = () => {
    setReason("");
    setReasonError(false);
    onClose();
  };

  if (!kind) {
    return null;
  }

  const copy = {
    rollback: {
      title: `Roll back to ${targetVersionId}?`,
      body: `Production flips back immediately — no benchmark is required to return to a known-good version. ${activeVersionId} will be retired, and this action is recorded in the audit log.`,
      confirm: "Roll back",
    },
    override: {
      title: "Promote to active without a generalization check?",
      body: "No held-out survey data exists yet, so this candidate has not been measured on unseen geography — the gate has verified anti-forgetting and the production operating point only. This promotion is recorded in the audit log as an override.",
      confirm: "Promote anyway",
    },
    dismiss: {
      title: `Dismiss candidate ${targetVersionId}?`,
      body: "The candidate is marked rejected and will not be promoted. Your current model is unchanged. This is recorded in the audit log.",
      confirm: "Dismiss candidate",
    },
    freeze: {
      title: "Freeze the gold set?",
      body: `${goldDraftSummary} Once frozen, the gold set never changes — every future candidate takes this exact exam.`,
      confirm: "Freeze gold set",
    },
  }[kind];

  return (
    <AlertDialog open onOpenChange={(open) => (!open ? close() : undefined)}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{copy.title}</AlertDialogTitle>
          <AlertDialogDescription>{copy.body}</AlertDialogDescription>
        </AlertDialogHeader>
        {kind === "override" ? (
          <div style={{ display: "grid", gap: "0.4rem" }}>
            <Label htmlFor="training-override-reason">Reason — required, recorded in the audit log</Label>
            <textarea
              id="training-override-reason"
              value={reason}
              onChange={(event) => setReason(event.target.value)}
              rows={3}
              style={{
                border: "1px solid var(--line)",
                borderRadius: "var(--radius-sm)",
                padding: "0.5rem",
                background: "transparent",
                color: "var(--text-main)",
                font: "inherit",
              }}
            />
            {reasonError ? <p className="training-inline-error">A reason is required.</p> : null}
          </div>
        ) : null}
        <AlertDialogFooter>
          <AlertDialogCancel onClick={close}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={(event) => {
              if (kind === "override" && reason.trim() === "") {
                event.preventDefault();
                setReasonError(true);
                return;
              }
              onConfirm(kind, reason.trim() || undefined);
              close();
            }}
          >
            {copy.confirm}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
