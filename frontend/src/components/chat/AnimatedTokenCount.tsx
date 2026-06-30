import { useEffect, useRef, useState } from "react";

import { formatTokens } from "@/lib/format";

type AnimatedTokenCountProps = {
  value: number;
  durationMs?: number;
};

// Increments smaller than this snap instantly instead of running a ~600ms rAF animation — a
// few-token bump is imperceptible to animate and only adds frame churn. Larger jumps still ease.
const MIN_ANIMATE_STEP = 50;

const normalizeTokenValue = (value: number): number =>
  Number.isFinite(value) && value > 0 ? Math.round(value) : 0;

const easeOutCubic = (progress: number): number => {
  const clamped = Math.min(1, Math.max(0, progress));
  return 1 - Math.pow(1 - clamped, 3);
};

const prefersReducedMotion = (): boolean =>
  typeof window !== "undefined" &&
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function AnimatedTokenCount({ value, durationMs = 600 }: AnimatedTokenCountProps) {
  const targetValue = normalizeTokenValue(value);
  const [displayValue, setDisplayValue] = useState(targetValue);
  const displayValueRef = useRef(targetValue);

  useEffect(() => {
    displayValueRef.current = displayValue;
  }, [displayValue]);

  useEffect(() => {
    const startValue = displayValueRef.current;
    // Snap (no rAF animation) when animating would only churn frames for no perceptual gain: a
    // sub-MIN_ANIMATE_STEP increment (pure flicker), a hidden tab (rAF is throttled/paused there,
    // so an animation would stall mid-count), reduced-motion, or a non-increasing/instant target.
    // The displayed number is always set to the exact target, so the final count stays accurate.
    const documentHidden = typeof document !== "undefined" && document.hidden;
    const shouldJump =
      targetValue <= startValue ||
      targetValue - startValue < MIN_ANIMATE_STEP ||
      documentHidden ||
      durationMs <= 0 ||
      prefersReducedMotion() ||
      typeof window === "undefined" ||
      typeof window.requestAnimationFrame !== "function";

    if (shouldJump) {
      displayValueRef.current = targetValue;
      setDisplayValue(targetValue);
      return;
    }

    const delta = targetValue - startValue;
    let frameId = 0;
    let startedAt: number | null = null;

    const tick = (timestamp: number) => {
      if (startedAt === null) {
        startedAt = timestamp;
      }
      const progress = Math.min(1, (timestamp - startedAt) / durationMs);
      const nextValue =
        progress >= 1 ? targetValue : Math.round(startValue + delta * easeOutCubic(progress));

      displayValueRef.current = nextValue;
      setDisplayValue((previous) => (previous === nextValue ? previous : nextValue));

      if (progress < 1) {
        frameId = window.requestAnimationFrame(tick);
      }
    };

    frameId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frameId);
  }, [durationMs, targetValue]);

  return <span aria-label={`${targetValue.toLocaleString()} tokens`}>{formatTokens(displayValue)} tokens</span>;
}
