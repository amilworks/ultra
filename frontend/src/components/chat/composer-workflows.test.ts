import { describe, expect, it } from "vitest";

import {
  COMPOSER_WORKFLOWS,
  COMPOSER_WORKFLOW_GROUP_ORDER,
  type ComposerWorkflowId,
  filterComposerWorkflows,
  getComposerWorkflowById,
} from "./composer-workflows";
import {
  composeComposerWorkflowPromptForModel,
  slashWorkflowSearchQuery,
  visiblePromptAfterComposerWorkflowSelection,
} from "./composer-workflow-prompt";

// The only tool pack the BisQue backend actually gates on is the literal "bisque"
// (see _should_register_bisque_tools in agent.py). Per-action names like
// "search_bisque_resources" were decorative and several pointed at tools that do
// not exist. These guards keep the curated menu honest: every command maps to a
// real backend capability.
const REAL_TOOL_PACKS = new Set(["bisque"]);

// Commands that were removed because they pointed at a non-existent backend tool
// or duplicated another command. They must not return.
const RETIRED_WORKFLOW_IDS: string[] = [
  "search_bisque_resources",
  "bisque_advanced_search",
  "load_bisque_resource",
  "bisque_download_dataset",
  "bisque_add_to_dataset",
  "bisque_add_gobjects",
  "add_tags_to_resource",
  "bisque_fetch_xml",
  "delete_bisque_resource",
  "detect_yolo",
  "estimate_depth_pro",
  "scientific_calculator",
  "chemistry_workbench",
];

describe("composer workflow catalog is focused and honest", () => {
  it("keeps a calm, focused command count", () => {
    // 11 active + 1 coming-soon. Guard against the list creeping back to 25.
    expect(COMPOSER_WORKFLOWS.length).toBeLessThanOrEqual(12);
    expect(COMPOSER_WORKFLOWS.filter((w) => !w.comingSoon).length).toBeGreaterThan(0);
  });

  it("never advertises an orphan tool pack", () => {
    for (const workflow of COMPOSER_WORKFLOWS) {
      for (const tool of workflow.selectedToolNames) {
        expect(REAL_TOOL_PACKS.has(tool)).toBe(true);
      }
    }
  });

  it("registers every BisQue command via the real 'bisque' tool pack", () => {
    const bisque = COMPOSER_WORKFLOWS.filter((w) => w.category === "BisQue");
    expect(bisque.length).toBeGreaterThan(0);
    for (const workflow of bisque) {
      expect(workflow.selectedToolNames).toEqual(["bisque"]);
    }
  });

  it("does not resurrect retired or dead-backend commands", () => {
    for (const id of RETIRED_WORKFLOW_IDS) {
      expect(getComposerWorkflowById(id as ComposerWorkflowId)).toBeNull();
    }
  });

  it("groups every workflow under a known, ordered category", () => {
    for (const workflow of COMPOSER_WORKFLOWS) {
      expect(COMPOSER_WORKFLOW_GROUP_ORDER).toContain(workflow.category);
    }
  });
});

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
