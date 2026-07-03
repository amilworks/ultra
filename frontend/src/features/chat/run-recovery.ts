export type RecoverableRunMessage = {
  role?: string;
  runId?: string | null;
  content?: string | null;
  liveStream?: unknown;
  status?: "stopped" | "failed" | string | null;
};

const LEGACY_RUN_RESULT_404_PATTERN =
  /^(?:error:\s*)?request failed with status 404\b/i;

export const isLegacyRunLookup404Content = (content: string | null | undefined): boolean =>
  LEGACY_RUN_RESULT_404_PATTERN.test(String(content ?? "").trim());

export const shouldRecoverRunResultMessage = (
  message: RecoverableRunMessage,
  options: { isStreamingMessage?: boolean; isLocalRunActive?: boolean } = {}
): boolean => {
  if (message.role !== "assistant" || !message.runId) {
    return false;
  }
  if (options.isLocalRunActive) {
    return false;
  }
  // A turn the user stopped, or that we have already marked failed, must not be
  // silently re-recovered — that would overwrite the calm Retry/Edit affordance.
  if (message.status === "stopped" || message.status === "failed") {
    return false;
  }
  if (options.isStreamingMessage && message.liveStream) {
    return false;
  }
  if (options.isStreamingMessage && !message.liveStream) {
    return true;
  }
  const content = String(message.content ?? "").trim();
  if (!content) {
    return true;
  }
  return isLegacyRunLookup404Content(content);
};

export const shouldDropHydratedLegacy404Message = (
  message: RecoverableRunMessage
): boolean =>
  message.role === "assistant" &&
  !message.runId &&
  isLegacyRunLookup404Content(message.content);

export const latestRunEventSequence = (
  events: Array<{ sequence?: unknown; payload?: { sequence?: unknown } | null } | null | undefined>
): number =>
  events.reduce((latest, event) => {
    const rawSequence = event?.payload?.sequence ?? event?.sequence ?? 0;
    const sequence = Math.floor(Number(rawSequence));
    return Number.isFinite(sequence) && sequence > latest ? sequence : latest;
  }, 0);
