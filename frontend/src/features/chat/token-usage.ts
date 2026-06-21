import type { RunEvent, RunTokenUsage } from "@/types";

type TokenUsageMessage = {
  responseMetadata?: Record<string, unknown> | null;
  runEvents?: RunEvent[] | null;
};

const toRecord = (value: unknown): Record<string, unknown> | null =>
  typeof value === "object" && value !== null ? (value as Record<string, unknown>) : null;

const positiveInteger = (value: unknown): number => {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? Math.floor(numeric) : 0;
};

const readTokenUsageRecord = (value: unknown): RunTokenUsage | null => {
  const record = toRecord(value);
  if (!record) {
    return null;
  }
  const input = positiveInteger(record.input_tokens);
  const output = positiveInteger(record.output_tokens);
  const explicitTotal = positiveInteger(record.total_tokens);
  const totalTokens = explicitTotal > 0 ? explicitTotal : input + output;
  if (!(totalTokens > 0)) {
    return null;
  }
  return {
    input_tokens: input,
    output_tokens: output,
    total_tokens: totalTokens,
    model: typeof record.model === "string" ? record.model : undefined,
  };
};

const tokenUsageFromPayload = (payload: unknown): RunTokenUsage | null => {
  const record = toRecord(payload);
  if (!record) {
    return null;
  }
  return readTokenUsageRecord(record) ?? readTokenUsageRecord(record.usage);
};

const eventTypeForUsage = (event: RunEvent): string => {
  const payload = toRecord(event.payload);
  return String(event.event_type || payload?.event_kind || payload?.event_type || "")
    .trim()
    .toLowerCase();
};

const stableUsageKeyPart = (value: unknown): string | null => {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed || null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? String(Math.floor(numeric)) : null;
};

const usageDedupeKey = (event: RunEvent): string | null => {
  const payload = toRecord(event.payload);
  if (!payload) {
    return null;
  }
  const usageEventId = stableUsageKeyPart(payload.usage_event_id);
  if (usageEventId) {
    return `usage:${usageEventId}`;
  }
  const eventId = stableUsageKeyPart(payload.event_id);
  if (eventId) {
    return `event:${eventId}`;
  }
  const sequence = stableUsageKeyPart(
    payload.sequence ?? (event as { sequence?: unknown }).sequence
  );
  if (!sequence) {
    return null;
  }
  const runId = stableUsageKeyPart(payload.run_id);
  return ["sequence", runId, eventTypeForUsage(event), sequence]
    .filter(Boolean)
    .join(":");
};

const aggregateLiveTokenUsage = (events: RunEvent[] | null | undefined): RunTokenUsage | null => {
  if (!Array.isArray(events) || events.length === 0) {
    return null;
  }
  const seenKeys = new Set<string>();
  let inputTokens = 0;
  let outputTokens = 0;
  let totalTokens = 0;
  let model: string | undefined;
  let modelMixed = false;

  events.forEach((event) => {
    if (eventTypeForUsage(event) !== "run.token_usage") {
      return;
    }
    const usage = tokenUsageFromPayload(event.payload);
    if (!usage) {
      return;
    }
    const key = usageDedupeKey(event);
    if (key) {
      if (seenKeys.has(key)) {
        return;
      }
      seenKeys.add(key);
    }
    inputTokens += usage.input_tokens;
    outputTokens += usage.output_tokens;
    totalTokens += usage.total_tokens;
    if (usage.model) {
      if (!model) {
        model = usage.model;
      } else if (model !== usage.model) {
        modelMixed = true;
      }
    }
  });

  if (!(totalTokens > 0)) {
    return null;
  }
  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: totalTokens,
    model: modelMixed ? undefined : model,
  };
};

const extractFinalRunTokenUsage = (
  events: RunEvent[] | null | undefined
): RunTokenUsage | null => {
  if (!Array.isArray(events)) {
    return null;
  }
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (eventTypeForUsage(event) !== "run.completed") {
      continue;
    }
    const usage = tokenUsageFromPayload(event.payload);
    if (usage) {
      return usage;
    }
  }
  return null;
};

export const extractRunTokenUsage = (message: TokenUsageMessage): RunTokenUsage | null => {
  const liveUsage = aggregateLiveTokenUsage(message.runEvents);
  if (liveUsage) {
    return liveUsage;
  }

  const metadataUsage = readTokenUsageRecord(message.responseMetadata?.usage);
  if (metadataUsage) {
    return metadataUsage;
  }

  return extractFinalRunTokenUsage(message.runEvents);
};
