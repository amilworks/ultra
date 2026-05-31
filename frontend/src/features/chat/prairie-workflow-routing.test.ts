import { describe, expect, it } from "vitest";

import { getComposerWorkflowById } from "@/components/chat/composer-workflows";

describe("prairie workflow routing", () => {
  it("routes uploaded prairie detection requests to the RareSpot queue tool", () => {
    const workflow = getComposerWorkflowById("detect_prairie_dog");

    expect(workflow?.selectedToolNames).toEqual(["rarespot_ecology_inference"]);
    expect(workflow?.workflowHint?.id).toBe("rarespot_ecology");
    expect(workflow?.requiresAttachedFiles).toBe(true);
  });
});
