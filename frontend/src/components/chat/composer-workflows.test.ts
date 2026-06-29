import { describe, expect, it } from "vitest";

import {
  filterComposerWorkflows,
  getComposerWorkflowById,
} from "./composer-workflows";
import {
  composeComposerWorkflowPromptForModel,
  slashWorkflowSearchQuery,
  visiblePromptAfterComposerWorkflowSelection,
} from "./composer-workflow-prompt";

describe("composer workflow prompt handling", () => {
  it("keeps the curated workflow scaffold out of the visible composer", () => {
    const workflow = getComposerWorkflowById("quantitative_analysis");
    expect(workflow).toBeTruthy();

    const visiblePrompt = visiblePromptAfterComposerWorkflowSelection(
      workflow!,
      "/quantitative"
    );

    expect(visiblePrompt).toBe("");
    expect(visiblePrompt).not.toContain("Run a focused quantitative analysis");
  });

  it("preserves user text after a slash command while removing the command token", () => {
    const workflow = getComposerWorkflowById("quantitative_analysis");
    expect(workflow).toBeTruthy();

    expect(
      visiblePromptAfterComposerWorkflowSelection(
        workflow!,
        "/quant analyze the cell count distribution"
      )
    ).toBe("analyze the cell count distribution");
  });

  it("filters slash workflows by the command token while leaving trailing prompt text alone", () => {
    const query = slashWorkflowSearchQuery("/quant analyze the cell count distribution");

    expect(query).toBe("quant");
    expect(filterComposerWorkflows(query).map((workflow) => workflow.id)).toContain(
      "quantitative_analysis"
    );
  });

  it("sends the workflow scaffold to the model without showing it in the composer", () => {
    const workflow = getComposerWorkflowById("quantitative_analysis");
    expect(workflow).toBeTruthy();

    const prompt = composeComposerWorkflowPromptForModel(
      workflow!,
      "analyze the cell count distribution"
    );

    expect(prompt).toContain("Run a focused quantitative analysis");
    expect(prompt).toContain("analyze the cell count distribution");
  });
});
