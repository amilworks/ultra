import { useEffect, useRef, useState } from "react";

import { cn } from "@/lib/utils";
import {
  fetchFigureCaption,
  figureCaptionKey,
  getCachedFigureCaption,
} from "@/lib/figureCaptionCache";
import type { ApiClient } from "@/lib/api";

type Props = {
  runId: string | null | undefined;
  path: string | null | undefined;
  apiClient: ApiClient;
  className?: string;
};

// A calm, academic-style caption beneath a run-output figure. The caption is
// generated + cached by the backend (a grounded VLM describing only what's visible);
// here it is fetched lazily the first time the figure scrolls into view and rendered
// in the muted "Figure. …" style of a paper. Renders an (invisible) sentinel until a
// caption arrives, so a figure with no caption adds no visual weight.
export function FigureCaption({ runId, path, apiClient, className }: Props) {
  const key = runId && path ? figureCaptionKey(runId, path) : "";
  const ref = useRef<HTMLParagraphElement>(null);
  const startedRef = useRef(false);
  const [caption, setCaption] = useState<string>(() => (key ? getCachedFigureCaption(key) ?? "" : ""));

  useEffect(() => {
    if (!runId || !path || !key || caption || startedRef.current) {
      return;
    }
    const node = ref.current;
    if (!node) {
      return;
    }
    let cancelled = false;
    const begin = () => {
      if (startedRef.current) {
        return;
      }
      startedRef.current = true;
      // fetchFigureCaption resolves immediately for an already-cached key, so the
      // caption is set in this .then (not synchronously in the effect body).
      fetchFigureCaption(key, () => apiClient.getRunArtifactCaption(runId, path))
        .then((result) => {
          if (!cancelled && result) {
            setCaption(result);
          }
        })
        .catch(() => undefined);
    };
    // Already resolved this session (incl. a cached "" = no caption): apply now,
    // no need to wait for the figure to scroll into view.
    if (getCachedFigureCaption(key) !== undefined || typeof IntersectionObserver === "undefined") {
      begin();
      return () => {
        cancelled = true;
      };
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          begin();
          observer.disconnect();
        }
      },
      { rootMargin: "200px" }
    );
    observer.observe(node);
    return () => {
      cancelled = true;
      observer.disconnect();
    };
  }, [key, runId, path, caption, apiClient]);

  return (
    <p ref={ref} className={cn("chat-figure-caption", !caption && "chat-figure-caption-empty", className)}>
      {caption ? (
        <>
          <span className="chat-figure-caption-label">Figure.</span> {caption}
        </>
      ) : null}
    </p>
  );
}
