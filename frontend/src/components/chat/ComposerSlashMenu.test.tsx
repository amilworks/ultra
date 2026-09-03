import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ComposerSlashMenu, workflowSlugs } from "./ComposerSlashMenu";
import {
  COMPOSER_WORKFLOW_GROUP_ORDER,
  filterComposerWorkflows,
  getComposerWorkflowById,
  toComposerWorkflowPresetState,
  type ComposerWorkflowDefinition,
} from "./composer-workflows";

HTMLElement.prototype.scrollIntoView ??= vi.fn();

describe("ComposerSlashMenu", () => {
  const workflowGroupsForQuery = (query: string) => {
    const grouped = new Map<
      ComposerWorkflowDefinition["category"],
      ComposerWorkflowDefinition[]
    >();
    filterComposerWorkflows(query).forEach((workflow) => {
      const existing = grouped.get(workflow.category) ?? [];
      existing.push(workflow);
      grouped.set(workflow.category, existing);
    });
    return COMPOSER_WORKFLOW_GROUP_ORDER.map((category) => ({
      category,
      items: grouped.get(category) ?? [],
    })).filter((group) => group.items.length > 0);
  };

  it("renders a calm command list without panel chrome", () => {
    render(
      <ComposerSlashMenu
        mode="workflow"
        workflowGroups={workflowGroupsForQuery("")}
        activeWorkflowId="goal_driven_build"
        onSelectWorkflow={vi.fn()}
      />
    );

    expect(screen.queryByText("Slash workflows")).not.toBeInTheDocument();
    expect(
      screen.queryByText("Choose a structured workflow for this turn.")
    ).not.toBeInTheDocument();
    expect(screen.queryByText("type `/`")).not.toBeInTheDocument();
    expect(screen.getByText("Goal-driven build")).toBeInTheDocument();
    expect(screen.getByText(/Scoped iterate-until-verified loop/i)).toBeInTheDocument();
  });

  it("selects a workflow when a command row is clicked", () => {
    const onSelectWorkflow = vi.fn();
    render(
      <ComposerSlashMenu
        mode="workflow"
        workflowGroups={workflowGroupsForQuery("quant")}
        activeWorkflowId="quantitative_analysis"
        onSelectWorkflow={onSelectWorkflow}
      />
    );

    fireEvent.click(screen.getByTestId("composer-workflow-quantitative_analysis"));

    expect(onSelectWorkflow).toHaveBeenCalledTimes(1);
    expect(onSelectWorkflow.mock.calls[0]?.[0]).toMatchObject({
      id: "quantitative_analysis",
    });
  });

  it("selects a workflow on mouse down before the composer focus changes", () => {
    const onSelectWorkflow = vi.fn();
    render(
      <ComposerSlashMenu
        mode="workflow"
        workflowGroups={workflowGroupsForQuery("quant")}
        activeWorkflowId="quantitative_analysis"
        onSelectWorkflow={onSelectWorkflow}
      />
    );

    fireEvent.mouseDown(screen.getByTestId("composer-workflow-quantitative_analysis"));

    expect(onSelectWorkflow).toHaveBeenCalledTimes(1);
    expect(onSelectWorkflow.mock.calls[0]?.[0]).toMatchObject({
      id: "quantitative_analysis",
    });
  });

  it("gives every row a slug no sibling shares, in the mono margin", () => {
    render(
      <ComposerSlashMenu
        mode="workflow"
        workflowGroups={workflowGroupsForQuery("")}
        activeWorkflowId={null}
        onSelectWorkflow={vi.fn()}
      />
    );
    const asides = [...document.querySelectorAll(".composer-slash-menu .composer-menu-aside")]
      .map((node) => node.textContent ?? "")
      .filter((text) => text.startsWith("/"));
    expect(asides.length).toBeGreaterThan(5);
    expect(new Set(asides).size).toBe(asides.length);
    expect(asides).toContain("/image");
    const slugs = workflowSlugs(filterComposerWorkflows(""));
    expect(slugs.get("bisque_download_resource")).not.toBe(slugs.get("find_bisque_assets"));
  });

  it("labels the confirm without a count until something is chosen", () => {
    render(
      <ComposerSlashMenu
        mode="resource_picker"
        preset={toComposerWorkflowPresetState(getComposerWorkflowById("image_analysis")!)}
        resources={[]}
        selectedResourceIds={new Set()}
      />
    );
    expect(screen.getByText("Use resources")).toBeInTheDocument();
  });
});
