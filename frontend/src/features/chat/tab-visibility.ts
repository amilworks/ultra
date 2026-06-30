// Shared visibility-gating foundation: a backgrounded tab should do no redundant live work (polling,
// rAF ticks). Centralizing the check + subscription keeps the poll, ticker, and multi-tab
// coordination consistent. (The live SSE response is deliberately NOT paused on hide — you want an
// in-flight answer to keep arriving so it is ready when you return — only redundant polling defers.)

export const isTabHidden = (): boolean =>
  typeof document !== "undefined" && document.visibilityState === "hidden";

export const onVisibilityChange = (handler: (hidden: boolean) => void): (() => void) => {
  if (typeof document === "undefined") {
    return () => {};
  }
  const listener = () => handler(document.visibilityState === "hidden");
  document.addEventListener("visibilitychange", listener);
  return () => document.removeEventListener("visibilitychange", listener);
};
