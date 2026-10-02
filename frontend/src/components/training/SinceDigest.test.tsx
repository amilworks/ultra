import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SinceDigest } from "./SinceDigest";

describe("SinceDigest", () => {
  it("does not treat an empty first-visit digest as proof that training never ran", () => {
    render(
      <SinceDigest
        events={[]}
        demoted={[]}
        firstVisit
        onOpenHistory={() => {}}
      />
    );

    expect(screen.getByText("This page tracks your detector's retraining.")).toBeInTheDocument();
    expect(screen.queryByText(/Nothing has run yet/i)).not.toBeInTheDocument();
  });
});
