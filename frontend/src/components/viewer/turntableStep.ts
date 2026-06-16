/**
 * Idle-turntable gate for the 3D volume viewer.
 *
 * The turntable is an OPT-IN showcase rotation (toggled with Space): it only spins
 * when the user has explicitly enabled it AND the scene is settled AND the pointer
 * has been idle past a threshold. prefers-reduced-motion disables it entirely. Any
 * real interaction cancels it (the caller flips `active` off on control 'start').
 *
 * Returns the OrbitControls `autoRotateSpeed` to apply (0 = do not rotate), so the
 * decision is a single pure function the caller can unit-test and feed straight
 * into `controls.autoRotate = step > 0`.
 */
export interface TurntableStepInput {
  /** ms since the last real (user) interaction. */
  idleMs: number;
  /** prefers-reduced-motion — when true the turntable never runs. */
  reducedMotion: boolean;
  /** Ray-march sampling has reached full quality (don't spin a half-resolved volume). */
  settled: boolean;
  /** User opted in (Space toggle). */
  active: boolean;
  /** Idle dwell before the turntable starts (ms). */
  idleThreshold?: number;
  /** Rotation speed when running (OrbitControls autoRotateSpeed units). */
  speed?: number;
}

export const TURNTABLE_IDLE_THRESHOLD_MS = 4000;
export const TURNTABLE_SPEED = 0.6;

export function turntableStep({
  idleMs,
  reducedMotion,
  settled,
  active,
  idleThreshold = TURNTABLE_IDLE_THRESHOLD_MS,
  speed = TURNTABLE_SPEED,
}: TurntableStepInput): number {
  if (!active || reducedMotion || !settled || idleMs < idleThreshold) {
    return 0;
  }
  return speed;
}
