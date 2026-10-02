import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { FileUpload } from "@/components/prompt-kit/file-upload";

import { Composer, type ComposerProps } from "./Composer";

const renderComposer = (overrides: Partial<ComposerProps> = {}) => {
  const props: ComposerProps = {
    value: "",
    onValueChange: vi.fn(),
    tokens: [],
    goneFileIds: [],
    onTokensChange: vi.fn(),
    tokenDetails: () => null,
    hydrated: true,
    running: false,
    readMode: false,
    phone: false,
    welcomeStage: false,
    menuOpen: false,
    hasFiles: false,
    workflow: null,
    onChangeWorkflow: vi.fn(),
    onRemoveWorkflow: vi.fn(),
    mode: "high",
    onChangeMode: vi.fn(),
    mention: null,
    onMentionChange: vi.fn(),
    onMentionActivate: vi.fn(),
    onMentionPick: vi.fn(),
    submitDisabled: true,
    canSteer: false,
    onSubmit: vi.fn(),
    onSteer: vi.fn(),
    onQueue: vi.fn(),
    onStop: vi.fn(),
    onOpenResources: vi.fn(),
    onStartWorkflow: vi.fn(),
    ...overrides,
  };

  render(
    <FileUpload onFilesAdded={vi.fn()}>
      <Composer {...props} />
    </FileUpload>
  );

  const modeLabel = props.mode === "pro" ? "Pro" : "High";
  fireEvent.pointerDown(
    screen.getByRole("button", { name: `Intelligence: ${modeLabel}` }),
    {
      button: 0,
      ctrlKey: false,
    }
  );

  return props;
};

describe("Composer intelligence menu", () => {
  it("exposes the selected mode as a checked radio menu item", () => {
    const onChangeMode = vi.fn();
    renderComposer({ onChangeMode });

    expect(screen.getByRole("group", { name: "Intelligence" })).toBeInTheDocument();
    expect(screen.getByRole("menuitemradio", { name: "High" })).toHaveAttribute(
      "aria-checked",
      "true"
    );
    const proMode = screen.getByRole("menuitemradio", { name: "Pro" });
    expect(proMode).toHaveAttribute("aria-checked", "false");

    fireEvent.click(proMode);
    expect(onChangeMode).toHaveBeenCalledWith("pro");
    expect(onChangeMode).toHaveBeenCalledTimes(1);
  });

  it("reports Pro as the sole selected mode when controlled by Pro state", () => {
    renderComposer({ mode: "pro" });

    expect(screen.getByRole("menuitemradio", { name: "High" })).toHaveAttribute(
      "aria-checked",
      "false"
    );
    expect(screen.getByRole("menuitemradio", { name: "Pro" })).toHaveAttribute(
      "aria-checked",
      "true"
    );
  });

  it("keeps the active mode action available for clearing adjacent workflow state", () => {
    const onChangeMode = vi.fn();
    renderComposer({ onChangeMode });

    fireEvent.click(screen.getByRole("menuitemradio", { name: "High" }));
    expect(onChangeMode).toHaveBeenCalledWith("high");
    expect(onChangeMode).toHaveBeenCalledTimes(1);
  });
});
