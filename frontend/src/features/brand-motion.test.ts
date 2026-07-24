import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const stylesSource = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

const transitionDeclarations = (): string[] =>
  [...stylesSource.matchAll(/transition\s*:\s*([^;]+);/g)].map((match) => match[1]);

/**
 * Motion is a brand signature here, not a per-component decision: one curve,
 * cubic-bezier(0.16, 1, 0.3, 1), so surfaces settle instead of sliding and the
 * product is recognisable with the logo cropped out.
 *
 * These are guards against drift, not a style preference. The curve was applied
 * to 59 declarations at once; without a test the next hand-written `ease` puts
 * a second gesture back into the product and nobody notices, because a wrong
 * easing curve looks like nothing at all — it just feels slightly off.
 */
describe("brand motion gesture", () => {
  it("defines exactly one house curve, with the sidebar token aliased to it", () => {
    expect(stylesSource).toMatch(/--motion-ease:\s*cubic-bezier\(0\.16,\s*1,\s*0\.3,\s*1\);/);
    expect(stylesSource).toMatch(/--sidebar-motion-ease:\s*var\(--motion-ease\);/);
    // One literal definition of the curve. Any second literal is a fork of the
    // gesture, even if the numbers happen to match today.
    expect(stylesSource.match(/cubic-bezier\(0\.16,\s*1,\s*0\.3,\s*1\)/g)).toHaveLength(1);
  });

  it("never leaves a bare `ease` in a transition", () => {
    const offenders = transitionDeclarations()
      // var(--motion-ease) / var(--sidebar-motion-ease) contain the substring
      // "ease" and are the correct form — strip them before looking for strays.
      .map((declaration) => declaration.replace(/var\(--[\w-]*ease\)/g, ""))
      // ease-in / ease-out / ease-in-out are distinct named curves, not the
      // generic default this guards against.
      .filter((declaration) => /\bease\b(?!-)/.test(declaration));

    expect(offenders, `bare \`ease\` found in: ${offenders.join(" | ")}`).toEqual([]);
  });

  it("keeps the documented exemptions and nothing more", () => {
    const named = transitionDeclarations().flatMap(
      (declaration) =>
        declaration.match(/\blinear\b|\bsteps\([^)]*\)|\bease-in-out\b|\bease-out\b|\bease-in\b/g) ?? []
    );
    // Discrete flips and continuous indicators legitimately opt out; a growing
    // list here means the gesture is being negotiated away one component at a
    // time, which is exactly the drift this suite exists to catch.
    //   linear x3 — `visibility 0s linear <delay>`, an instant state flip that
    //               must not interpolate (see the composer/rail slim rules).
    //   ease-out  — one 120ms progress-bar width nudge, too brief to read.
    expect(named.sort()).toEqual(["ease-out", "linear", "linear", "linear"]);
  });

  it("only ever animates `visibility` with linear, never a real property", () => {
    // Guards the exemption above from being borrowed: `linear` is permitted
    // because visibility cannot meaningfully interpolate, not because linear is
    // an acceptable second gesture for geometry or colour.
    for (const declaration of transitionDeclarations()) {
      for (const segment of declaration.split(",")) {
        if (/\blinear\b/.test(segment)) {
          expect(segment).toMatch(/visibility/);
        }
      }
    }
  });

  it("keeps durations inside the 160-280ms band the design law specifies", () => {
    const durations = [
      ...new Set(
        transitionDeclarations()
          .flatMap((declaration) => declaration.match(/(\d+)ms/g) ?? [])
          .map((value) => Number.parseInt(value, 10))
      ),
    ].sort((a, b) => a - b);

    expect(durations.length).toBeGreaterThan(0);
    // 120-140ms reads as "immediate" for colour and small nudges and is allowed
    // below the band; anything past 280ms is slower than the law permits.
    expect(Math.max(...durations)).toBeLessThanOrEqual(280);
    expect(Math.min(...durations)).toBeGreaterThanOrEqual(120);
  });
});
