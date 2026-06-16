/**
 * Receding-chrome visibility policy for the scientific viewer.
 *
 * The viewer's glass overlays (zoom readout, scale bar, orientation / slice /
 * clip cues, control buttons) recede to a faint, still-discoverable level after
 * the pointer goes idle, and snap back on pointer activity, hover, or keyboard
 * focus. Fading is OPACITY-ONLY so faded controls stay in the tab order and the
 * accessibility tree, and prefers-reduced-motion keeps everything fully visible.
 */

export type ChromeVisibility = "visible" | "faded";

export interface ChromeVisibilityInput {
  /** Keyboard focus is within a chrome control (forces visible for a11y). */
  focusWithin: boolean;
  /** Milliseconds since the last pointer activity over the viewer. */
  idleMs: number;
  /** prefers-reduced-motion — when true, chrome never fades. */
  reducedMotion: boolean;
  /** Idle duration before chrome recedes (ms). */
  idleThreshold?: number;
}

export const CHROME_IDLE_THRESHOLD_MS = 2500;

/**
 * Pure decision: should the chrome be visible or faded? Reduced-motion and
 * keyboard focus always win (accessibility); otherwise chrome recedes once the
 * pointer has been idle past the threshold. Pointer hover is handled in CSS
 * (:hover) so it is intentionally not an input here.
 */
export function resolveChromeVisibility({
  focusWithin,
  idleMs,
  reducedMotion,
  idleThreshold = CHROME_IDLE_THRESHOLD_MS,
}: ChromeVisibilityInput): ChromeVisibility {
  if (reducedMotion || focusWithin) {
    return "visible";
  }
  return idleMs >= idleThreshold ? "faded" : "visible";
}

export interface ChromeFadeController {
  /** Show chrome now and (re)arm the idle-fade timer. Call on pointer activity. */
  reveal(): void;
  /** Force chrome to the faded state immediately. */
  fadeNow(): void;
  /** Cancel the timer and detach. */
  dispose(): void;
}

/**
 * Imperative driver wiring the policy to a DOM node. Writes `data-chrome-faded`
 * on `root` (CSS targets `[data-chrome-faded='true'] .viewer-chrome-fade`). Kept
 * ref/DOM-based (no React state) so high-frequency pointermove never triggers a
 * re-render. Reduced-motion pins chrome visible: the idle timer is never armed
 * and fadeNow() is a no-op.
 */
export function createChromeFadeController(
  root: HTMLElement,
  options: { reducedMotion?: boolean; idleThreshold?: number } = {}
): ChromeFadeController {
  const reducedMotion = options.reducedMotion ?? false;
  const idleThreshold = options.idleThreshold ?? CHROME_IDLE_THRESHOLD_MS;
  let timer = 0;
  const cancel = () => {
    if (timer) {
      window.clearTimeout(timer);
      timer = 0;
    }
  };
  const setFaded = (faded: boolean) => {
    const next = faded ? "true" : "false";
    if (root.dataset.chromeFaded !== next) {
      root.dataset.chromeFaded = next;
    }
  };
  const reveal = () => {
    cancel();
    setFaded(false);
    if (reducedMotion) {
      return;
    }
    timer = window.setTimeout(() => setFaded(true), idleThreshold);
  };
  const fadeNow = () => {
    cancel();
    if (!reducedMotion) {
      setFaded(true);
    }
  };
  return { reveal, fadeNow, dispose: cancel };
}

/** Safe prefers-reduced-motion read (jsdom has no matchMedia). */
export function prefersReducedMotionSafe(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
    : false;
}
