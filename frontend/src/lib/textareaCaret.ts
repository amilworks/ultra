/**
 * Where a caret sits inside a <textarea>, in pixels.
 *
 * Browsers expose no API for this, so the standard approach is used: a hidden
 * mirror block that copies every style the textarea uses to lay out its text,
 * holds the text up to the caret plus a zero-width marker, and reports the
 * marker's offset. The composer uses it to anchor the @ picker at the caret
 * on desktop. Everything is defensive: the answer is advisory (the caller
 * falls back to a fixed anchor), so no failure here may ever throw.
 */

const MIRRORED_PROPERTIES = [
  "boxSizing",
  "width",
  "paddingTop",
  "paddingRight",
  "paddingBottom",
  "paddingLeft",
  "borderTopWidth",
  "borderRightWidth",
  "borderBottomWidth",
  "borderLeftWidth",
  "fontFamily",
  "fontSize",
  "fontWeight",
  "fontStyle",
  "fontVariant",
  "fontStretch",
  "fontFeatureSettings",
  "fontVariantNumeric",
  "letterSpacing",
  "lineHeight",
  "textIndent",
  "textTransform",
  "wordSpacing",
  "tabSize",
  "whiteSpace",
  "overflowWrap",
  "wordBreak",
] as const;

export type TextareaCaretPosition = {
  /** Distance from the textarea's left border edge to the caret, in px. */
  left: number;
  /** Distance from the textarea's top border edge to the caret's line top, in px. */
  top: number;
  /** The caret's line box height, in px. */
  height: number;
};

export const measureTextareaCaret = (
  textarea: HTMLTextAreaElement,
  position: number
): TextareaCaretPosition | null => {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return null;
  }
  let mirror: HTMLDivElement | null = null;
  try {
    const computed = window.getComputedStyle(textarea);
    mirror = document.createElement("div");
    const style = mirror.style;
    style.position = "absolute";
    style.visibility = "hidden";
    style.pointerEvents = "none";
    style.top = "0";
    style.left = "-9999px";
    style.overflow = "hidden";
    style.whiteSpace = "pre-wrap";
    style.overflowWrap = "break-word";
    for (const property of MIRRORED_PROPERTIES) {
      const value = computed[property as keyof CSSStyleDeclaration];
      if (typeof value === "string" && value.length > 0) {
        (style as unknown as Record<string, string>)[property] = value;
      }
    }
    // The textarea's scrollbar, if any, narrows its content box; the mirror
    // has none, so match the content width explicitly.
    style.width = `${textarea.clientWidth}px`;
    style.boxSizing = "border-box";
    const clamped = Math.max(0, Math.min(position, textarea.value.length));
    const before = textarea.value.slice(0, clamped);
    mirror.textContent = before;
    const marker = document.createElement("span");
    // A zero-width marker still needs content to occupy a line box.
    marker.textContent = "​";
    mirror.appendChild(marker);
    document.body.appendChild(mirror);
    // Line height resolves in this order: the computed value when numeric,
    // the marker's own box, then 1.5× the font size, then a plain default —
    // so a "normal" line-height or a layout-less test DOM never yields NaN.
    const lineHeight = Number.parseFloat(computed.lineHeight);
    const fontSize = Number.parseFloat(computed.fontSize);
    const height =
      Number.isFinite(lineHeight) && lineHeight > 0
        ? lineHeight
        : marker.offsetHeight > 0
          ? marker.offsetHeight
          : Number.isFinite(fontSize) && fontSize > 0
            ? fontSize * 1.5
            : 24;
    return {
      left: marker.offsetLeft,
      top: marker.offsetTop - textarea.scrollTop,
      height,
    };
  } catch {
    return null;
  } finally {
    mirror?.remove();
  }
};
