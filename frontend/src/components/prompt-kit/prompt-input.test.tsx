import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PromptInput, PromptInputAction } from "./prompt-input";

function renderSubmitAction({
  disabled = false,
  onSubmit = vi.fn(),
}: {
  disabled?: boolean;
  onSubmit?: () => void;
} = {}) {
  const view = render(
    <PromptInput value="Ready" onSubmit={onSubmit}>
      <PromptInputAction disabled={disabled} tooltip="Send prompt">
        <button type="submit" disabled={disabled} aria-label="Send prompt">
          Send
        </button>
      </PromptInputAction>
    </PromptInput>
  );
  return {
    ...view,
    button: screen.getByRole("button", { name: "Send prompt" }),
  };
}

describe("PromptInputAction", () => {
  it("opens its tooltip from keyboard focus without changing the child control", async () => {
    const { button, container } = renderSubmitAction();

    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(container.querySelector("button button")).toBeNull();
    expect(button).toHaveAccessibleName("Send prompt");

    fireEvent.focus(button);

    expect(await screen.findByRole("tooltip")).toHaveTextContent("Send prompt");
  });

  it("keeps mouse-open content available through hover transfer and dismisses on Escape", async () => {
    const { button } = renderSubmitAction();

    fireEvent.pointerMove(button, { pointerType: "mouse" });

    const tooltip = await screen.findByRole("tooltip");
    const tooltipContent = tooltip.closest('[data-slot="tooltip-content"]');
    expect(tooltipContent).not.toBeNull();

    fireEvent.pointerLeave(button, {
      pointerType: "mouse",
      clientX: 0,
      clientY: 0,
    });
    fireEvent.pointerMove(tooltipContent as HTMLElement, {
      pointerType: "mouse",
      clientX: 0,
      clientY: 0,
    });

    expect(screen.getByRole("tooltip")).toHaveTextContent("Send prompt");

    fireEvent.keyDown(document, { key: "Escape" });

    await waitFor(() => {
      expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    });
  });

  it("lets one enabled submit-button click submit exactly once", () => {
    const onSubmit = vi.fn();
    const { button } = renderSubmitAction({ onSubmit });

    fireEvent.click(button);

    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("renders an explicitly disabled child unwrapped with no tooltip or submit", () => {
    const onSubmit = vi.fn();
    const { button, container } = renderSubmitAction({ disabled: true, onSubmit });

    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(container.querySelector('[data-slot="tooltip"]')).toBeNull();
    expect(container.querySelector('[data-slot="tooltip-trigger"]')).toBeNull();

    fireEvent.focus(button);
    fireEvent.pointerMove(button, { pointerType: "mouse" });
    fireEvent.click(button);

    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
