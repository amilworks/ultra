import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PhaseMetadataSummary } from "./PhaseMetadataSummary";

describe("PhaseMetadataSummary", () => {
  it("renders DREAM.3D phase names and provenance without claiming detection", () => {
    const provenance =
      "Read from stored DREAM.3D PhaseName metadata; no phase-identification algorithm was run.";

    render(
      <PhaseMetadataSummary
        phaseNames={["Primary"]}
        source="stored_metadata"
        provenance={provenance}
      />
    );

    expect(screen.getByText("Stored phase metadata")).toBeInTheDocument();
    expect(screen.getByText("Primary")).toBeInTheDocument();
    expect(screen.getByText(provenance)).toBeInTheDocument();
    expect(screen.queryByText(/detected phases/i)).not.toBeInTheDocument();
  });
});
