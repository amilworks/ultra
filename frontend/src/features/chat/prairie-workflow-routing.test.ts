import { describe, expect, it } from "vitest";

import { getComposerWorkflowById } from "@/components/chat/composer-workflows";

describe("prairie workflow routing", () => {
  it("sends prairie detection through the prairie-dog-detection Skill (no forced dispatch tool)", () => {
    const workflow = getComposerWorkflowById("detect_prairie_dog");

    // The RareSpot dispatch tool was retired for the sandbox-run Skill, so the
    // workflow no longer forces a tool or a rarespot_ecology workflow hint — the
    // prompt + keywords trigger the Skill via progressive disclosure.
    expect(workflow?.selectedToolNames).toEqual([]);
    expect(workflow?.workflowHint).toBeNull();
    expect(workflow?.requiresAttachedFiles).toBe(true);
  });
});
