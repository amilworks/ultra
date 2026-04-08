import { useMemo } from "react";
import { Markdown } from "./markdown";
import { type ResponseStreamProps, useTextStream } from "./response-stream";

const FENCE_PATTERN = /^(\s*)(`{3,}|~{3,})(.*)$/;

const closeOpenFence = (source: string): string => {
  const lines = source.split("\n");
  let activeFence: { marker: "`" | "~"; length: number } | null = null;

  for (const line of lines) {
    const match = line.match(FENCE_PATTERN);
    if (!match) {
      continue;
    }

    const token = match[2] ?? "";
    const marker = token[0] as "`" | "~";

    if (!activeFence) {
      activeFence = { marker, length: token.length };
      continue;
    }

    if (activeFence.marker === marker && token.length >= activeFence.length) {
      activeFence = null;
    }
  }

  if (!activeFence) {
    return source;
  }

  return `${source}${source.endsWith("\n") ? "" : "\n"}${activeFence.marker.repeat(activeFence.length)}`;
};

const closeOpenMathBlock = (source: string): string => {
  const blockFenceCount = Array.from(source.matchAll(/^\s*\$\$\s*$/gm)).length;
  if (blockFenceCount % 2 === 0) {
    return source;
  }
  return `${source}${source.endsWith("\n") ? "" : "\n"}$$`;
};

const normalizeStreamingMarkdown = (source: string): string => {
  if (!source.trim()) {
    return source;
  }
  return closeOpenMathBlock(closeOpenFence(source));
};

export type MarkdownResponseStreamProps = Omit<ResponseStreamProps, "as">;

export function MarkdownResponseStream({
  textStream,
  mode = "typewriter",
  speed = 20,
  className,
  onComplete,
  fadeDuration,
  segmentDelay,
  characterChunkSize,
}: MarkdownResponseStreamProps) {
  const { displayedText } = useTextStream({
    textStream,
    mode,
    speed,
    onComplete,
    fadeDuration,
    segmentDelay,
    characterChunkSize,
  });
  const normalizedPreview = useMemo(
    () => normalizeStreamingMarkdown(displayedText),
    [displayedText]
  );

  return <Markdown className={className}>{normalizedPreview}</Markdown>;
}

export { normalizeStreamingMarkdown };
