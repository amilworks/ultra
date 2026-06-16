import { describe, expect, it } from "vitest";

import {
  TURNTABLE_IDLE_THRESHOLD_MS,
  TURNTABLE_SPEED,
  turntableStep,
} from "./turntableStep";

const base = {
  idleMs: TURNTABLE_IDLE_THRESHOLD_MS + 1,
  reducedMotion: false,
  settled: true,
  active: true,
};

describe("turntableStep", () => {
  it("returns the rotate speed when opted in, settled, and idle past threshold", () => {
    expect(turntableStep(base)).toBe(TURNTABLE_SPEED);
  });

  it("returns 0 when not opted in", () => {
    expect(turntableStep({ ...base, active: false })).toBe(0);
  });

  it("returns 0 under reduced motion even when otherwise eligible", () => {
    expect(turntableStep({ ...base, reducedMotion: true })).toBe(0);
  });

  it("returns 0 until the volume sampling has settled", () => {
    expect(turntableStep({ ...base, settled: false })).toBe(0);
  });

  it("returns 0 before the idle threshold", () => {
    expect(turntableStep({ ...base, idleMs: TURNTABLE_IDLE_THRESHOLD_MS - 1 })).toBe(0);
  });

  it("honors a custom speed", () => {
    expect(turntableStep({ ...base, speed: 1.2 })).toBe(1.2);
  });
});
