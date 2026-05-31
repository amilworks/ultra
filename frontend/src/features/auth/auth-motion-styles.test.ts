import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const readSource = (relativePath: string): string =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8");

describe("auth motion styles", () => {
  it("keeps login motion keyboard-like, hover-driven, and reduced-motion aware", () => {
    const styles = readSource("src/styles.css");
    const authScreen = readSource("src/components/auth/AuthScreen.tsx");

    expect(styles).toMatch(/\.auth-hero-typewriter/);
    expect(styles).toMatch(/\.auth-hero-typewriter-caret/);
    expect(styles).toMatch(/@keyframes auth-hero-caret/);
    expect(styles).not.toMatch(/@keyframes auth-hero-flip/);
    expect(styles).not.toMatch(/rotateX/);
    expect(styles).toMatch(/\.auth-card::before/);
    expect(styles).toMatch(/\.auth-card:hover::before/);
    expect(styles).toMatch(/@keyframes auth-card-sheen/);
    expect(styles).toMatch(/@media \(prefers-reduced-motion: reduce\)/);
    expect(styles).toMatch(/\.auth-screen-logo\s*{[^}]*font-weight:\s*300;/s);
    expect(styles).toMatch(/\.auth-screen-hero h1\s*{[^}]*font-weight:\s*300;/s);
    expect(authScreen).toMatch(/const HERO_PHRASE_DWELL_MS = 12_000;/);
    expect(authScreen).not.toMatch(/}, 2200\)/);
  });
});
