// Shared wheel-gesture classifier for the scientific viewer's 2D surfaces, so the
// deep-zoom canvas and the direct-plane canvas behave identically. The modern
// image-viewer convention (Figma/Maps/macOS): a trackpad PINCH and ⌘/ctrl-scroll
// zoom; a trackpad two-finger SCROLL pans; a discrete mouse wheel zooms. Trackpad
// scroll is distinguished from a mouse wheel heuristically — it carries a horizontal
// component or fine/fractional pixel deltas, whereas a mouse wheel sends coarse
// vertical steps. Pure + unit-tested.
export type WheelLike = {
  deltaX: number;
  deltaY: number;
  deltaMode: number;
  ctrlKey: boolean;
  metaKey: boolean;
};

export const classifyWheelGesture = (event: WheelLike): "zoom" | "pan" => {
  if (event.ctrlKey || event.metaKey) {
    return "zoom"; // pinch (macOS fires ctrlKey) or ⌘/ctrl-scroll
  }
  if (event.deltaMode !== 0) {
    return "zoom"; // line/page deltas come from a classic mouse wheel
  }
  if (event.deltaX !== 0) {
    return "pan"; // horizontal component → trackpad two-finger scroll
  }
  // Pure-vertical pixel scroll: fine/fractional steps are a trackpad pan; coarse
  // integer steps (typical mouse wheel notch ~100px) are a zoom.
  if (!Number.isInteger(event.deltaY) || Math.abs(event.deltaY) < 50) {
    return "pan";
  }
  return "zoom";
};
