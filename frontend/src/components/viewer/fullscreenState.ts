/**
 * Fullscreen state machine + key mapping for the scientific viewer.
 *
 * Pure helpers so the toggle policy and key handling are unit-testable without the
 * DOM Fullscreen API (jsdom does not implement it). The component owns the actual
 * requestFullscreen()/exitFullscreen() calls and syncs state from the native
 * 'fullscreenchange' event.
 */

export type FullscreenAction = "toggle" | "exit";

/** Next desired fullscreen flag for an action. */
export function nextFullscreen(current: boolean, action: FullscreenAction): boolean {
  if (action === "exit") {
    return false;
  }
  return !current;
}

/**
 * Map a keydown to a fullscreen action, or null to ignore. F toggles, Escape
 * exits. Ignored when the user is typing in a field or holding a modifier, so we
 * never hijack form input, the window/level + cutaway inputs, or browser shortcuts.
 * (Escape always maps to "exit"; the caller is responsible for ignoring it when not
 * already fullscreen, so Escape still closes a host dialog/sheet in that case.)
 */
export function keyToFullscreenAction(
  key: string,
  typingTarget: boolean,
  modifierActive = false
): FullscreenAction | null {
  if (typingTarget || modifierActive) {
    return null;
  }
  if (key === "f" || key === "F") {
    return "toggle";
  }
  if (key === "Escape" || key === "Esc") {
    return "exit";
  }
  return null;
}

/** Whether an event target is a text-entry control we must not hijack. */
export function isTypingTarget(target: EventTarget | null): boolean {
  if (typeof HTMLElement === "undefined" || !(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
    return true;
  }
  return target.isContentEditable === true;
}
