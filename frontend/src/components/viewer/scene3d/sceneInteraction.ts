/**
 * OrbitControls emits `start`, `change`, and `end` synchronously for every wheel event.
 * Without a settle latch the renderer enters and leaves interactive LoD before the next
 * animation frame, so zooming still sorts the full settled scene. This controller keeps
 * one burst active without coupling the policy to React, three.js, or the DOM.
 */

export const SCENE_INTERACTION_SETTLE_MS = 240;

/**
 * Spark sorts at roughly 30 Hz in Lens, so submitting the same ordered scene at a 120 Hz
 * panel rate only repeats expensive blend work. Fifteen milliseconds yields one render
 * every other 120 Hz callback while retaining every frame on a conventional 60 Hz panel.
 */
export const SCENE_CONTINUOUS_FRAME_INTERVAL_MS = 15;

export const isContinuousSceneFrameDue = (lastRenderedAt: number, now: number): boolean =>
  !Number.isFinite(lastRenderedAt) ||
  !Number.isFinite(now) ||
  now < lastRenderedAt ||
  now - lastRenderedAt >= SCENE_CONTINUOUS_FRAME_INTERVAL_MS;

export type SceneInteractionController = {
  start: () => void;
  end: () => void;
  dispose: () => void;
};

export const createSceneInteractionController = (
  onActiveChange: (active: boolean) => void,
  settleMs = SCENE_INTERACTION_SETTLE_MS
): SceneInteractionController => {
  let active = false;
  let disposed = false;
  let settleTimer: ReturnType<typeof globalThis.setTimeout> | null = null;
  const delay = Number.isFinite(settleMs) && settleMs >= 0 ? settleMs : SCENE_INTERACTION_SETTLE_MS;

  const cancelSettle = () => {
    if (settleTimer !== null) {
      globalThis.clearTimeout(settleTimer);
      settleTimer = null;
    }
  };

  return {
    start: () => {
      if (disposed) {
        return;
      }
      cancelSettle();
      if (!active) {
        active = true;
        onActiveChange(true);
      }
    },
    end: () => {
      if (disposed || !active) {
        return;
      }
      cancelSettle();
      settleTimer = globalThis.setTimeout(() => {
        settleTimer = null;
        if (disposed || !active) {
          return;
        }
        active = false;
        onActiveChange(false);
      }, delay);
    },
    dispose: () => {
      disposed = true;
      cancelSettle();
    },
  };
};
