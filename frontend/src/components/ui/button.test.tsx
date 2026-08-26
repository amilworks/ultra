import { describe, expect, it } from "vitest";

import { buttonVariants } from "./button";

describe("buttonVariants", () => {
  it("keeps secondary outline actions quiet until hover or focus", () => {
    const classes = buttonVariants({ variant: "outline" });

    expect(classes).toContain("border-transparent");
    expect(classes).toContain("shadow-none");
    expect(classes).toContain("bg-muted/50");
    expect(classes).not.toContain("bg-background");
    expect(classes).not.toContain("shadow-xs");
  });
});
