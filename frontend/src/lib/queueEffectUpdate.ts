// Defers a state update out of the effect body onto a microtask, returning a
// cancel function for the effect cleanup. Keeps reset-then-load effect chains
// from tearing renders under React 18/19 strict scheduling.
export const queueEffectUpdate = (callback: () => void): (() => void) => {
  let cancelled = false;
  window.queueMicrotask(() => {
    if (!cancelled) {
      callback();
    }
  });
  return () => {
    cancelled = true;
  };
};
