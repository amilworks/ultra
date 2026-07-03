import { useEffect, useRef, useState } from "react";
import { queueEffectUpdate } from "@/lib/queueEffectUpdate";
import type { TokenUsageResponse } from "@/types";

// Loads the signed-in user's token usage for the blank-chat hero exactly once
// per identity key, with a ref-based request dedupe (no refetch loops when the
// hero unmounts/remounts while the identity is unchanged).
export function useBlankChatTokenUsage({
  enabled,
  key,
  load,
  normalizeError,
}: {
  enabled: boolean;
  key: string;
  load: () => Promise<TokenUsageResponse>;
  normalizeError: (error: unknown) => string;
}): {
  usage: TokenUsageResponse | null;
  loading: boolean;
  error: string | null;
} {
  const [usage, setUsage] = useState<TokenUsageResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestRef = useRef({
    key: "",
    inFlight: false,
    loaded: false,
    failed: false,
  });

  useEffect(() => {
    requestRef.current = {
      key,
      inFlight: false,
      loaded: false,
      failed: false,
    };
    return queueEffectUpdate(() => {
      setUsage(null);
      setError(null);
      setLoading(false);
    });
  }, [key]);

  useEffect(() => {
    const requestState = requestRef.current;
    if (!enabled || requestState.key !== key) {
      return;
    }
    if (requestState.inFlight || requestState.loaded || requestState.failed) {
      return;
    }
    let cancelled = false;
    requestState.inFlight = true;
    const cancelQueuedLoading = queueEffectUpdate(() => {
      if (cancelled) {
        return;
      }
      setLoading(true);
      setError(null);
    });
    void load()
      .then((response) => {
        if (!cancelled) {
          requestState.loaded = true;
          setUsage(response);
        }
      })
      .catch((loadError) => {
        if (!cancelled) {
          requestState.failed = true;
          setError(normalizeError(loadError));
        }
      })
      .finally(() => {
        requestState.inFlight = false;
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
      requestState.inFlight = false;
      cancelQueuedLoading();
    };
  }, [enabled, key, load, normalizeError]);

  return { usage, loading, error };
}
