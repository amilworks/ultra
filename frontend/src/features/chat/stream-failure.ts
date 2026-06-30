// Calm, categorized classification of a streaming-turn failure. The app already preserves any
// partial answer and never leaves a stuck spinner (the handler's finally clears `sending`); this
// adds the missing piece — a human, category-aware headline + a retryable signal — so a mid-stream
// failure reads as a calm, actionable notice instead of a raw stack/status string. Pure + testable;
// the caller passes the HTTP status (when the error is an ApiError) and the normalized detail text.

export type StreamFailureCategory =
  | "transient_transport"
  | "rate_limited"
  | "auth"
  | "run_failed"
  | "unknown";

export type StreamFailureClassification = {
  category: StreamFailureCategory;
  retryable: boolean;
  // A single calm sentence shown above the technical detail.
  headline: string;
};

const TRANSPORT_HINTS = [
  "load failed",
  "failed to fetch",
  "network request failed",
  "networkerror",
  "the network connection was lost",
  "terminated",
  "connection",
];

export const classifyStreamFailure = (
  status: number | null | undefined,
  detail: string
): StreamFailureClassification => {
  const text = String(detail ?? "").toLowerCase();

  if (status === 401 || status === 403) {
    return {
      category: "auth",
      retryable: false,
      headline: "Your session expired — sign in again to continue this conversation.",
    };
  }
  if (status === 429 || text.includes("rate limit") || text.includes("too many requests")) {
    return {
      category: "rate_limited",
      retryable: true,
      headline: "The model is handling a lot right now — give it a moment, then retry.",
    };
  }
  // Transport drops are the most common mid-stream failure and are always worth a retry; the
  // partial answer (if any) is kept above the notice.
  if (TRANSPORT_HINTS.some((hint) => text.includes(hint))) {
    return {
      category: "transient_transport",
      retryable: true,
      headline:
        "The connection dropped while the model was responding — any partial answer is kept above. Retry to continue.",
    };
  }
  if (typeof status === "number" && status >= 500) {
    return {
      category: "run_failed",
      retryable: true,
      headline: "The run hit a server error before it finished — retry to try again.",
    };
  }
  return {
    category: "unknown",
    // With no status (a thrown stream error) a retry is usually worth offering; a clean 4xx is not.
    retryable: typeof status !== "number" || status >= 500,
    headline: "This response could not be completed.",
  };
};

// Compose the calm headline with the technical detail for the inline error card: the human sentence
// leads, the raw detail follows in parentheses for debugging without dominating the surface.
export const composeStreamFailureReason = (
  classification: StreamFailureClassification,
  detail: string
): string => {
  const trimmed = String(detail ?? "").trim();
  if (!trimmed || trimmed.toLowerCase() === classification.headline.toLowerCase()) {
    return classification.headline;
  }
  return `${classification.headline} (${trimmed})`;
};
