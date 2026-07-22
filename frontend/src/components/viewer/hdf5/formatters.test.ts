import { describe, expect, it } from "vitest";

import { describePhaseMetadata } from "./formatters";

describe("describePhaseMetadata", () => {
  it("labels DREAM.3D phase names as stored metadata, not detected phases", () => {
    const provenance =
      "Read from stored DREAM.3D PhaseName metadata; no phase-identification algorithm was run.";

    const description = describePhaseMetadata("stored_metadata", provenance);

    expect(description).toEqual({
      label: "Stored phase metadata",
      detail: provenance,
    });
    expect(description.label).not.toMatch(/detect/i);
  });

  it("uses neutral wording for legacy payloads without source metadata", () => {
    const description = describePhaseMetadata(null, null);

    expect(description).toEqual({ label: "Reported phase names", detail: null });
    expect(description.label).not.toMatch(/detect/i);
  });
});
