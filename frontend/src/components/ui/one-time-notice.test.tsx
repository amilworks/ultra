import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ONE_TIME_NOTICE_STORAGE_KEY, OneTimeNotice } from "./one-time-notice";

const renderNotice = (audienceId = "scientist@example.edu") =>
  render(
    <OneTimeNotice
      noticeId="pro-mode-v1"
      audienceId={audienceId}
      title="Pro mode is now available"
      description="Choose Pro for complex scientific work."
    >
      <button type="button">High</button>
    </OneTimeNotice>
  );

describe("OneTimeNotice", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("anchors an accessible release notice without replacing its control", () => {
    renderNotice();

    expect(screen.getByRole("button", { name: "High" })).toBeVisible();
    expect(screen.getByRole("button", { name: "High" }).parentElement).toHaveAttribute(
      "data-notice-state",
      "open"
    );
    expect(screen.getByRole("dialog", { name: "Pro mode is now available" })).toBeVisible();
    expect(screen.getByText("Choose Pro for complex scientific work.")).toBeVisible();
    expect(screen.getByLabelText("Dismiss Pro mode is now available")).toBeVisible();
  });

  it("persists dismissal for the same notice and audience only", async () => {
    const first = renderNotice();

    fireEvent.click(screen.getByLabelText("Dismiss Pro mode is now available"));
    expect(screen.queryByRole("dialog", { name: "Pro mode is now available" })).toBeNull();

    await waitFor(() => {
      expect(window.localStorage.getItem(ONE_TIME_NOTICE_STORAGE_KEY)).toContain(
        "scientist@example.edu:pro-mode-v1"
      );
    });

    first.unmount();
    renderNotice();
    expect(screen.queryByRole("dialog", { name: "Pro mode is now available" })).toBeNull();

    renderNotice("another-scientist@example.edu");
    expect(screen.getByRole("dialog", { name: "Pro mode is now available" })).toBeVisible();
  });

  it("does not consume an announcement when product code moves focus", () => {
    render(
      <>
        <input aria-label="Composer" />
        <OneTimeNotice
          noticeId="focus-safe-v1"
          audienceId="scientist@example.edu"
          title="A focused release"
          description="The composer may focus itself."
        >
          <button type="button">Anchor</button>
        </OneTimeNotice>
      </>
    );

    fireEvent.focus(screen.getByRole("textbox", { name: "Composer" }));

    expect(screen.getByRole("dialog", { name: "A focused release" })).toBeVisible();
    expect(window.localStorage.getItem(ONE_TIME_NOTICE_STORAGE_KEY)).not.toContain(
      "focus-safe-v1"
    );
  });

  it("does not consume an announcement when responsive layout unmounts it", () => {
    const view = renderNotice();

    view.unmount();

    expect(window.localStorage.getItem(ONE_TIME_NOTICE_STORAGE_KEY)).not.toContain(
      "scientist@example.edu:pro-mode-v1"
    );
    renderNotice();
    expect(screen.getByRole("dialog", { name: "Pro mode is now available" })).toBeVisible();
  });

  it("supports an optional release action and marks the notice seen", async () => {
    const onSelect = vi.fn();
    render(
      <OneTimeNotice
        noticeId="new-viewer-v1"
        audienceId="scientist@example.edu"
        title="A new viewer is ready"
        description="Open the viewer directly from a resource."
        action={{ label: "Open viewer", onSelect }}
      >
        <button type="button">View</button>
      </OneTimeNotice>
    );

    fireEvent.click(screen.getByRole("button", { name: "Open viewer" }));

    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("dialog", { name: "A new viewer is ready" })).toBeNull();
    await waitFor(() => {
      expect(window.localStorage.getItem(ONE_TIME_NOTICE_STORAGE_KEY)).toContain(
        "scientist@example.edu:new-viewer-v1"
      );
    });
  });

  it("can be held back until a release is eligible", () => {
    render(
      <OneTimeNotice
        noticeId="future-release-v1"
        audienceId="scientist@example.edu"
        title="Future release"
        description="Not ready yet."
        enabled={false}
      >
        <button type="button">Anchor</button>
      </OneTimeNotice>
    );

    expect(screen.getByRole("button", { name: "Anchor" })).toBeVisible();
    expect(screen.queryByRole("dialog", { name: "Future release" })).toBeNull();
  });

  it("recovers from a malformed receipt store", () => {
    window.localStorage.setItem(ONE_TIME_NOTICE_STORAGE_KEY, "null");

    renderNotice();

    expect(screen.getByRole("dialog", { name: "Pro mode is now available" })).toBeVisible();
    expect(() => {
      fireEvent.click(screen.getByLabelText("Dismiss Pro mode is now available"));
    }).not.toThrow();
  });
});
