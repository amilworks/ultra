// Tail-anchored message windowing: keep the DOM bounded for very long threads by mounting only the
// most-recent `windowSize` messages, with the rest reachable via a "show earlier" affordance. The
// window tracks the tail, so the streaming/newest turn is always included. Pure + testable.

export const MESSAGE_WINDOW_SIZE = 50;

export const windowTailMessages = <T>(
  messages: T[],
  windowSize: number
): { visible: T[]; hiddenCount: number } => {
  if (windowSize <= 0 || messages.length <= windowSize) {
    return { visible: messages, hiddenCount: 0 };
  }
  const visible = messages.slice(messages.length - windowSize);
  return { visible, hiddenCount: messages.length - visible.length };
};
