import { fireEvent, render } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";

import { BriefOverlay } from "./BriefOverlay";

const registry = [{ label: "scan.tif", fileId: "f-scan" }];

const mountTextarea = () => {
  const ref = createRef<HTMLTextAreaElement>();
  const textarea = document.createElement("textarea");
  textarea.value = "in @scan.tif to";
  document.body.appendChild(textarea);
  (ref as { current: HTMLTextAreaElement | null }).current = textarea;
  return { ref, textarea };
};

describe("BriefOverlay", () => {
  it("draws a pill per token carrying the real filename as its title", () => {
    const { ref, textarea } = mountTextarea();
    const { container } = render(
      <BriefOverlay
        textareaRef={ref}
        text="in @scan.tif to"
        registry={registry}
        fileDetails={() => ({ title: "scan.tif · TIF · 12 MB" })}
      />
    );
    const token = container.querySelector(".brief-token");
    expect(token).not.toBeNull();
    expect(token?.textContent).toBe("@scan.tif");
    expect(token?.getAttribute("title")).toBe("scan.tif · TIF · 12 MB");
    expect(token?.getAttribute("data-file-id")).toBe("f-scan");
    // Prose stays in the flow (it positions the pills) but is never ink.
    expect(container.querySelectorAll(".brief-overlay-text")).toHaveLength(2);
    // The layer is decoration: never announced, never a focus stop.
    expect(container.querySelector(".brief-overlay")?.getAttribute("aria-hidden")).toBeNull();
    expect(container.querySelector(".brief-overlay-mirror")?.getAttribute("aria-hidden")).toBe("true");
    textarea.remove();
  });

  it("marks a token whose file is gone", () => {
    const { ref, textarea } = mountTextarea();
    const { container } = render(
      <BriefOverlay
        textareaRef={ref}
        text="@scan.tif"
        registry={registry}
        fileDetails={() => ({ title: "scan.tif", gone: true })}
      />
    );
    expect(container.querySelector(".brief-token-gone")).not.toBeNull();
    textarea.remove();
  });

  it("reports a pointer press on a token so the caret can be placed after it", () => {
    const { ref, textarea } = mountTextarea();
    const onTokenPointerDown = vi.fn();
    const { container } = render(
      <BriefOverlay
        textareaRef={ref}
        text="in @scan.tif to"
        registry={registry}
        fileDetails={() => null}
        onTokenPointerDown={onTokenPointerDown}
      />
    );
    fireEvent.mouseDown(container.querySelector(".brief-token")!);
    expect(onTokenPointerDown).toHaveBeenCalledTimes(1);
    expect(onTokenPointerDown.mock.calls[0][0]).toMatchObject({ fileId: "f-scan", start: 3, end: 12 });
    textarea.remove();
  });

  it("reports the prefix width, and zero when there is no prefix", () => {
    const { ref, textarea } = mountTextarea();
    const onPrefixWidthChange = vi.fn();
    const { rerender } = render(
      <BriefOverlay
        textareaRef={ref}
        text=""
        registry={[]}
        fileDetails={() => null}
        prefix={<span className="brief-chip">Pro</span>}
        onPrefixWidthChange={onPrefixWidthChange}
      />
    );
    expect(onPrefixWidthChange).toHaveBeenCalled();
    rerender(
      <BriefOverlay
        textareaRef={ref}
        text=""
        registry={[]}
        fileDetails={() => null}
        onPrefixWidthChange={onPrefixWidthChange}
      />
    );
    expect(onPrefixWidthChange).toHaveBeenLastCalledWith(0);
    textarea.remove();
  });
});
