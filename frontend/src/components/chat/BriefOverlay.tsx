import {
  type CSSProperties,
  type MouseEvent,
  type ReactNode,
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

import {
  type BriefFileToken,
  type BriefFileSegment,
  parseBriefSegments,
} from "@/features/chat/brief-tokens";
import { cn } from "@/lib/utils";

/**
 * Paints the brief's tokens over the composer's textarea.
 *
 * The textarea keeps every character and every behaviour; this layer only adds
 * ink. It mirrors the textarea's typography and padding exactly, lays the same
 * text out in the same box, and draws a translucent pill under each token run
 * — translucent, so the textarea's own glyphs show through and no text is ever
 * painted twice. Prose runs render transparent: they exist only to push the
 * pills into the right place.
 *
 * The prefix slot holds the chips that are NOT text — the workflow and the
 * intelligence mode — laid inline at the head of the first line. Their measured
 * width is reported back so the textarea can indent its first line by exactly
 * that much, keeping both layers on one grid.
 */

export type BriefOverlayFileDetails = {
  title: string;
  gone?: boolean;
};

type BriefOverlayProps = {
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  text: string;
  registry: readonly BriefFileToken[];
  /** Real filename, kind, size — the hover title — and whether the file is gone. */
  fileDetails: (fileId: string) => BriefOverlayFileDetails | null;
  prefix?: ReactNode;
  onPrefixWidthChange?: (width: number) => void;
  /** A pointer press on a token: the caller places the caret after it. */
  onTokenPointerDown?: (segment: BriefFileSegment) => void;
  /** Any value that changes when the textarea's typography or padding may have
   *  changed underneath us (slim/expanded/collapsed flips), so the mirror is
   *  re-read in the same frame. */
  syncKey?: string;
  className?: string;
};

const MIRRORED_STYLE_PROPERTIES = [
  "fontFamily",
  "fontSize",
  "fontWeight",
  "fontStyle",
  "fontFeatureSettings",
  "fontVariantNumeric",
  "letterSpacing",
  "lineHeight",
  "paddingTop",
  "paddingRight",
  "paddingBottom",
  "paddingLeft",
  "tabSize",
] as const;
// NOT mirrored: text-indent. The textarea indents its first line to make room
// for the prefix chips, but in the overlay those chips ARE the first line's
// leading content — mirroring the indent as well would shift the overlay's
// text a second time and land the chips under the first word (seen live).

export function BriefOverlay({
  textareaRef,
  text,
  registry,
  fileDetails,
  prefix,
  onPrefixWidthChange,
  onTokenPointerDown,
  syncKey,
  className,
}: BriefOverlayProps) {
  const contentRef = useRef<HTMLDivElement | null>(null);
  const prefixRef = useRef<HTMLSpanElement | null>(null);
  const [mirroredStyle, setMirroredStyle] = useState<CSSProperties>({});
  const [box, setBox] = useState<CSSProperties>({});
  const [scrollTop, setScrollTop] = useState(0);
  const [mountTick, setMountTick] = useState(0);

  // Sit exactly on the textarea's box. The textarea is one child among the
  // card body's several (attach button, actions row), so the overlay cannot
  // simply fill its parent — it copies the textarea's offset rectangle and
  // follows it through autosize and the slim/expanded transitions.
  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      // A sibling textarea can attach its ref after this effect's first run
      // (React walks layout effects and refs in one pass, in tree order).
      // Look again next frame instead of waiting for the text to change.
      const frame = window.requestAnimationFrame(() => setMountTick((tick) => tick + 1));
      return () => window.cancelAnimationFrame(frame);
    }
    const measure = () =>
      setBox((previous) => {
        const next: CSSProperties = {
          top: textarea.offsetTop,
          left: textarea.offsetLeft,
          width: textarea.offsetWidth,
          height: textarea.offsetHeight,
        };
        return previous.top === next.top &&
          previous.left === next.left &&
          previous.width === next.width &&
          previous.height === next.height
          ? previous
          : next;
      });
    measure();
    if (typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(measure);
    observer.observe(textarea);
    return () => observer.disconnect();
  }, [textareaRef, text, syncKey, mountTick]);

  // Copy the textarea's live typography into the overlay. Re-read whenever the
  // text or the caller's sync key changes — the composer moves its own padding
  // and metrics across slim / expanded / collapsed states via CSS attributes
  // driven by exactly those inputs — so the overlay follows in the same frame.
  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea || typeof window === "undefined") {
      return;
    }
    const computed = window.getComputedStyle(textarea);
    const next: Record<string, string> = {};
    for (const property of MIRRORED_STYLE_PROPERTIES) {
      const value = computed[property as keyof CSSStyleDeclaration];
      if (typeof value === "string") {
        next[property] = value;
      }
    }
    setMirroredStyle((previous) => {
      const previousRecord = previous as Record<string, string | number | undefined>;
      const changed = Object.keys(next).some((key) => previousRecord[key] !== next[key]);
      return changed ? (next as CSSProperties) : previous;
    });
  }, [textareaRef, text, prefix, syncKey]);

  // Follow the textarea's own scroll (it scrolls past its max-height).
  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    const sync = () => setScrollTop(textarea.scrollTop);
    sync();
    textarea.addEventListener("scroll", sync, { passive: true });
    return () => textarea.removeEventListener("scroll", sync);
  }, [textareaRef]);

  // Report the prefix width so the textarea can indent its first line to match.
  useLayoutEffect(() => {
    const node = prefixRef.current;
    if (!onPrefixWidthChange) {
      return;
    }
    if (!node) {
      onPrefixWidthChange(0);
      return;
    }
    const report = () => onPrefixWidthChange(node.getBoundingClientRect().width);
    report();
    if (typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(report);
    observer.observe(node);
    return () => observer.disconnect();
  }, [onPrefixWidthChange, prefix]);

  const handleTokenMouseDown = useCallback(
    (segment: BriefFileSegment) => (event: MouseEvent<HTMLSpanElement>) => {
      event.preventDefault();
      onTokenPointerDown?.(segment);
    },
    [onTokenPointerDown]
  );

  const segments = parseBriefSegments(text, registry);

  return (
    <div className={cn("brief-overlay", className)} style={box}>
      <div
        ref={contentRef}
        className="brief-overlay-content"
        style={{ ...mirroredStyle, textIndent: 0, transform: `translateY(${-scrollTop}px)` }}
      >
        {prefix ? (
          <span ref={prefixRef} className="brief-overlay-prefix">
            {prefix}
          </span>
        ) : null}
        {/* The mirrored text is decoration for sighted readers — the textarea
            already carries it. The prefix chips above are real controls and
            stay in the tree; only the mirror is hidden from assistive tech. */}
        <span className="brief-overlay-mirror" aria-hidden="true">
        {segments.map((segment) => {
          if (segment.kind === "text") {
            return (
              <span key={segment.start} className="brief-overlay-text">
                {segment.text}
              </span>
            );
          }
          const details = fileDetails(segment.fileId);
          return (
            <span
              key={segment.start}
              className={cn("brief-token", details?.gone && "brief-token-gone")}
              data-file-id={segment.fileId}
              title={details?.title ?? segment.label}
              onMouseDown={handleTokenMouseDown(segment)}
            >
              {segment.text}
            </span>
          );
        })}
        {/* A trailing newline in the text would otherwise collapse: keep the
            overlay's last line box alive so its height mirrors the textarea. */}
        {text.endsWith("\n") ? <span className="brief-overlay-text">{"​"}</span> : null}
        </span>
      </div>
    </div>
  );
}
