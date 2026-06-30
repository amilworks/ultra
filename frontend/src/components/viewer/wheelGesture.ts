// Shared wheel-gesture classifier for the scientific viewer's 2D surfaces, so the
// deep-zoom canvas and the direct-plane canvas behave identically. Lens keeps drag
// as the primary pan gesture; vertical wheel / trackpad scroll zooms so a fitted
// image does not feel inert on Mac trackpads. Horizontal-dominant trackpad scroll
// still pans for users who intentionally side-scroll across a zoomed image.
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
  if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
    return "pan"; // horizontal-dominant trackpad scroll
  }
  return "zoom";
};
