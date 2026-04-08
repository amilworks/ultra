import React from "react";

type Segment =
  | { type: "text"; value: string }
  | { type: "code"; value: string; lang?: string };

const CODE_FENCE_RE = /```([a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g;

const splitMessage = (content: string): Segment[] => {
  const segments: Segment[] = [];
  let lastIndex = 0;
  for (const match of content.matchAll(CODE_FENCE_RE)) {
    const index = match.index ?? 0;
    if (index > lastIndex) {
      segments.push({ type: "text", value: content.slice(lastIndex, index) });
    }
    segments.push({
      type: "code",
      value: match[2] ?? "",
      lang: (match[1] ?? "").trim() || undefined,
    });
    lastIndex = index + match[0].length;
  }
  if (lastIndex < content.length) {
    segments.push({ type: "text", value: content.slice(lastIndex) });
  }
  if (segments.length === 0) {
    segments.push({ type: "text", value: content });
  }
  return segments;
};

export function MessageContent({ content }: { content: string }) {
  const segments = React.useMemo(() => splitMessage(content), [content]);
  return (
    <div className="message-content">
      {segments.map((segment, index) => {
        if (segment.type === "code") {
          return (
            <pre key={`code-${index}`} className="code-block">
              {segment.lang ? <span className="code-lang">{segment.lang}</span> : null}
              <code>{segment.value}</code>
            </pre>
          );
        }
        return (
          <p key={`text-${index}`} className="message-text">
            {segment.value}
          </p>
        );
      })}
    </div>
  );
}

