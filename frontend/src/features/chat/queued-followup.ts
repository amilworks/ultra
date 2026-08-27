import { notesSearchScopeState } from "@/lib/notesAccess";

/**
 * Seals Notes search consent while separate composer messages are folded into
 * one queued follow-up. The newest explicit UI choice wins. Without one, a
 * new directly typed Notes request can supersede an older removal, while an
 * ordinary or paste-only addition preserves the prior choice.
 */
export const mergeQueuedNoteSearchScopeOverride = ({
  existingOverride,
  incomingOverride,
  incomingText,
  incomingExcludedReferenceText,
}: {
  existingOverride: boolean | null;
  incomingOverride: boolean | null;
  incomingText: string;
  incomingExcludedReferenceText: readonly string[];
}): boolean | null => {
  if (incomingOverride !== null) return incomingOverride;
  if (
    notesSearchScopeState(incomingText, incomingExcludedReferenceText, null).active
  ) {
    return true;
  }
  return existingOverride;
};
