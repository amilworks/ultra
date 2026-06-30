import { ChevronDown } from "lucide-react";

import { MarkdownResponseStream } from "@/components/prompt-kit/markdown-response-stream";

export type LiveStreamRegionProps = {
  messageId: string;
  liveStream: AsyncIterable<string>;
  // When true (a multi-step agentic run), the running first-person narration folds into an opt-in
  // "live reasoning" disclosure so the calm step timeline above stays the primary surface. A plain
  // reply (false) streams inline for responsiveness.
  foldIntoReasoning: boolean;
  onComplete: () => void;
};

// The streaming-response region. The <details> keeps the stream body MOUNTED while collapsed, so the
// MarkdownResponseStream keeps consuming the (single-consumer) liveStream and accumulating — expanding
// reveals the full text with no lost tokens. The body is identical in both modes.
export function LiveStreamRegion({
  messageId,
  liveStream,
  foldIntoReasoning,
  onComplete,
}: LiveStreamRegionProps) {
  const body = (
    <div
      id={messageId}
      className="pk-message-content rounded-lg bg-transparent p-0 text-foreground break-words whitespace-normal"
      // a11y: the text updates ~60fps while streaming. aria-busy tells assistive tech the region is
      // mid-update and aria-live="off" suppresses per-frame announcements.
      aria-busy="true"
      aria-live="off"
    >
      <MarkdownResponseStream
        className="w-full bg-transparent p-0 text-foreground"
        textStream={liveStream}
        onComplete={onComplete}
      />
    </div>
  );

  if (!foldIntoReasoning) {
    return body;
  }

  return (
    <details className="live-reasoning mt-0.5">
      <summary className="live-reasoning-summary">
        <ChevronDown className="live-reasoning-chevron size-3.5" aria-hidden />
        <span>Show live reasoning</span>
      </summary>
      <div className="live-reasoning-body">{body}</div>
    </details>
  );
}
