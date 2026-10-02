import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ResourceQuickPeek } from "./ResourceQuickPeek";
import type { ResourceRecord, ResourceTextHead } from "@/types";

describe("CSV preview identity", () => {
  const resource: ResourceRecord = {
    file_id: "survey",
    original_name: "survey_2026.csv",
    content_type: "text/csv",
    size_bytes: 20,
    sha256: "survey-sha",
    created_at: "2026-06-07T12:00:00Z",
    source_type: "upload",
    resource_kind: "table",
    has_thumbnail: false,
  };

  describe("ResourceQuickPeek", () => {
    it.each(["loaded", "failed"])("names the focused preview while loading and when %s", async (outcome) => {
      let resolve!: (head: ResourceTextHead) => void;
      let reject!: (error: Error) => void;
      const pending = new Promise<ResourceTextHead>((res, rej) => {
        resolve = res;
        reject = rej;
      });
      const fetchHead = vi.fn(() => pending);
      render(<ResourceQuickPeek resource={resource} fetchHead={fetchHead} />);
      const trigger = screen.getByRole("button", { name: "Quick look at survey_2026.csv" });
      trigger.focus();
      fireEvent.click(trigger);

      const dialog = screen.getByRole("dialog");
      expect(dialog).toHaveAccessibleName("survey_2026.csv");
      expect(dialog).toHaveFocus();
      expect(screen.getByText("Loading…")).toBeInTheDocument();
      expect(fetchHead).toHaveBeenCalledWith("survey", 8192);

      await act(async () => {
        if (outcome === "failed") {
          reject(new Error("Could not load preview"));
        } else {
          resolve({
            file_id: resource.file_id,
            original_name: resource.original_name,
            content_type: "text/csv",
            format: "csv",
            total_size_bytes: 20,
            offset: 0,
            returned_bytes: 20,
            next_offset: 20,
            truncated: false,
            encoding: "utf-8",
            eol: "lf",
            line_count: 2,
            approx_total_lines: 2,
            text: "site_id,count\nA-014,3",
          });
        }
      });

      expect(screen.queryByText("Loading…")).not.toBeInTheDocument();
      expect(dialog).toHaveAccessibleName("survey_2026.csv");
      expect(dialog).toHaveTextContent(outcome === "failed" ? "Could not load preview" : "A-014,3");
      fireEvent.keyDown(dialog, { key: "Escape" });
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
      expect(trigger).toHaveFocus();
    });
  });
});

describe("Markdown preview identity", () => {
  const resource: ResourceRecord = {
    file_id: "readme",
    original_name: "README.md",
    content_type: "text/markdown",
    size_bytes: 100,
    sha256: "fixture",
    created_at: "2026-09-21T00:00:00Z",
    source_type: "upload",
    resource_kind: "document",
    has_thumbnail: false,
  };

  const head: ResourceTextHead = {
    file_id: resource.file_id,
    original_name: resource.original_name,
    content_type: "text/markdown",
    format: "markdown",
    total_size_bytes: 8,
    offset: 0,
    returned_bytes: 8,
    next_offset: 8,
    truncated: false,
    encoding: "utf-8",
    eol: "none",
    line_count: 1,
    approx_total_lines: 1,
    text: "# Survey",
  };

  describe("ResourceQuickPeek", () => {
    it("identifies the preview by filename while loading and after content arrives", async () => {
      let resolveHead!: (value: ResourceTextHead) => void;
      const fetchHead = vi.fn(() => new Promise<ResourceTextHead>((resolve) => {
        resolveHead = resolve;
      }));
      render(<ResourceQuickPeek resource={resource} fetchHead={fetchHead} />);

      fireEvent.click(screen.getByRole("button", { name: "Quick look at README.md" }));

      const dialog = await screen.findByRole("dialog", { name: "README.md" });
      expect(dialog).toHaveTextContent("Loading…");
      await act(async () => resolveHead(head));
      expect(dialog).toHaveTextContent("# Survey");
      expect(dialog).toHaveAccessibleName("README.md");
    });

    it("keeps the resource identity available when its preview fails", async () => {
      const fetchHead = vi.fn().mockRejectedValue(new Error("Could not load preview"));
      render(<ResourceQuickPeek resource={resource} fetchHead={fetchHead} />);

      fireEvent.click(screen.getByRole("button", { name: "Quick look at README.md" }));

      await screen.findByText("Could not load preview");
      expect(screen.getByRole("dialog", { name: "README.md" })).toHaveTextContent(
        "Could not load preview"
      );
    });
  });
});

describe("Keyboard reading", () => {
  const resource: ResourceRecord = {
    file_id: "readme",
    original_name: "README.md",
    content_type: "text/markdown",
    size_bytes: 4096,
    sha256: "readme-sha",
    created_at: "2026-09-22T00:00:00Z",
    source_type: "upload",
    resource_kind: "document",
    has_thumbnail: false,
  };

  const head: ResourceTextHead = {
    file_id: resource.file_id,
    original_name: resource.original_name,
    content_type: "text/markdown",
    format: "markdown",
    total_size_bytes: 4096,
    offset: 0,
    returned_bytes: 4096,
    next_offset: 4096,
    truncated: false,
    encoding: "utf-8",
    eol: "lf",
    line_count: 14,
    approx_total_lines: 14,
    text: Array.from({ length: 14 }, (_, i) => `Survey line ${i + 1}: observation notes`).join("\n"),
  };

  describe("ResourceQuickPeek keyboard reading", () => {
    it("makes a loaded preview a named keyboard scroll target without stealing focus", async () => {
      let resolveHead!: (value: ResourceTextHead) => void;
      const fetchHead = vi.fn(() => new Promise<ResourceTextHead>((resolve) => { resolveHead = resolve; }));
      render(<ResourceQuickPeek resource={resource} fetchHead={fetchHead} />);
      const trigger = screen.getByRole("button", { name: "Quick look at README.md" });
      trigger.focus();
      fireEvent.click(trigger);
      const dialog = screen.getByRole("dialog");
      await waitFor(() => expect(dialog).toHaveFocus());
      expect(screen.queryByRole("region", { name: "Preview of README.md" })).not.toBeInTheDocument();

      await act(async () => resolveHead(head));
      const preview = screen.getByRole("region", { name: "Preview of README.md" });
      expect(preview.tabIndex).toBe(0);
      expect(preview).toHaveTextContent("Survey line 14");
      expect(dialog).toHaveFocus();
      preview.focus();
      expect(preview).toHaveFocus();
      fireEvent.keyDown(preview, { key: "Escape" });
      await waitFor(() => expect(trigger).toHaveFocus());
      expect(fetchHead).toHaveBeenCalledWith("readme", 8192);
    });

    it("focuses the scroll target when reopening a cached preview", async () => {
      const fetchHead = vi.fn().mockResolvedValue(head);
      render(<ResourceQuickPeek resource={resource} fetchHead={fetchHead} />);
      const trigger = screen.getByRole("button", { name: "Quick look at README.md" });
      fireEvent.click(trigger);
      await screen.findByText(/Survey line 14/);
      fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
      fireEvent.click(trigger);
      const preview = screen.getByRole("region", { name: "Preview of README.md" });
      await waitFor(() => expect(preview).toHaveFocus());
      expect(fetchHead).toHaveBeenCalledTimes(1);
    });

    it("offers the same keyboard target after a failed preview is retried", async () => {
      const fetchHead = vi.fn().mockRejectedValueOnce(new Error("Preview unavailable")).mockResolvedValueOnce(head);
      render(<ResourceQuickPeek resource={resource} fetchHead={fetchHead} />);
      const trigger = screen.getByRole("button", { name: "Quick look at README.md" });
      fireEvent.click(trigger);
      await screen.findByText("Preview unavailable");
      expect(screen.queryByRole("region", { name: "Preview of README.md" })).not.toBeInTheDocument();
      fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
      await waitFor(() => expect(trigger).toHaveFocus());
      fireEvent.click(trigger);
      const preview = await screen.findByRole("region", { name: "Preview of README.md" });
      expect(preview.tabIndex).toBe(0);
      expect(fetchHead).toHaveBeenCalledTimes(2);
    });
  });
});

describe("Detached previews", () => {
  const resource: ResourceRecord = {
    file_id: "readme",
    original_name: "README.md",
    content_type: "text/markdown",
    size_bytes: 4096,
    sha256: "readme-sha",
    created_at: "2026-09-24T00:00:00Z",
    source_type: "upload",
    resource_kind: "document",
    has_thumbnail: false,
  };

  const head: ResourceTextHead = {
    file_id: resource.file_id,
    original_name: resource.original_name,
    content_type: "text/markdown",
    format: "markdown",
    total_size_bytes: 4096,
    offset: 0,
    returned_bytes: 4096,
    next_offset: 4096,
    truncated: false,
    encoding: "utf-8",
    eol: "lf",
    line_count: 14,
    approx_total_lines: 14,
    text: Array.from({ length: 14 }, (_, i) => `Survey line ${i + 1}: observation notes`).join("\n"),
  };

  type ObservedTrigger = {
    notify: (isIntersecting: boolean, ratio?: number) => void;
    disconnect: ReturnType<typeof vi.fn>;
  };
  let observations: ObservedTrigger[];

  beforeEach(() => {
    observations = [];
    vi.stubGlobal("IntersectionObserver", class {
      constructor(callback: IntersectionObserverCallback) {
        let target: Element;
        const disconnect = vi.fn();
        this.disconnect = disconnect;
        this.observe = (element: Element) => { target = element; };
        observations.push({
          disconnect,
          notify: (isIntersecting, intersectionRatio = isIntersecting ? 1 : 0) => {
            callback([{ target, isIntersecting, intersectionRatio } as IntersectionObserverEntry], this as unknown as IntersectionObserver);
          },
        });
      }
      observe: (element: Element) => void;
      disconnect: ReturnType<typeof vi.fn>;
    });
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  function trigger() {
    return screen.getByRole("button", { name: "Quick look at README.md" });
  }

  function notify(isIntersecting: boolean, ratio?: number) {
    act(() => observations[observations.length - 1]?.notify(isIntersecting, ratio));
  }

  describe("ResourceQuickPeek trigger visibility", () => {
    it("closes an offscreen preview and restores keyboard position without scrolling back", async () => {
      const fetchHead = vi.fn().mockResolvedValue(head);
      render(<ResourceQuickPeek resource={resource} fetchHead={fetchHead} />);
      trigger().focus();
      fireEvent.click(trigger());
      await screen.findByText(/Survey line 14/);
      await waitFor(() => expect(screen.getByRole("dialog")).toContainElement(document.activeElement as HTMLElement));
      const focus = vi.spyOn(trigger(), "focus");

      notify(false);

      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
      await waitFor(() => expect(trigger()).toHaveFocus());
      expect(focus).toHaveBeenCalledWith({ preventScroll: true });
      expect(observations[0].disconnect).toHaveBeenCalled();
    });

    it("keeps a partially visible trigger open and preserves normal Escape return", async () => {
      render(<ResourceQuickPeek resource={resource} fetchHead={vi.fn().mockResolvedValue(head)} />);
      fireEvent.click(trigger());
      await screen.findByText(/Survey line 14/);
      notify(true, 0.2);
      expect(screen.getByRole("dialog")).toBeInTheDocument();
      fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
      await waitFor(() => expect(trigger()).toHaveFocus());
    });

    it("stays closed when a detached preview finishes loading and reuses its result on reopen", async () => {
      let resolveHead!: (value: ResourceTextHead) => void;
      const fetchHead = vi.fn(() => new Promise<ResourceTextHead>(resolve => { resolveHead = resolve; }));
      render(<ResourceQuickPeek resource={resource} fetchHead={fetchHead} />);
      fireEvent.click(trigger());
      expect(screen.getByText("Loading…")).toBeInTheDocument();
      notify(false);
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
      await act(async () => resolveHead(head));
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

      fireEvent.click(trigger());
      expect(screen.getByText(/Survey line 14/)).toBeInTheDocument();
      expect(fetchHead).toHaveBeenCalledTimes(1);
      act(() => observations[0].notify(false));
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    it("does not steal focus moved outside during detachment dismissal", async () => {
      render(<><ResourceQuickPeek resource={resource} fetchHead={vi.fn().mockResolvedValue(head)} /><button>Refresh resources</button></>);
      fireEvent.click(trigger());
      await screen.findByText(/Survey line 14/);
      notify(false);
      screen.getByRole("button", { name: "Refresh resources" }).focus();
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
      // Radix restores focus in a later task after its portal unmounts.
      await act(async () => { await new Promise(resolve => setTimeout(resolve, 10)); });
      expect(screen.getByRole("button", { name: "Refresh resources" })).toHaveFocus();
    });

    it("disconnects an open preview on unmount", async () => {
      const { unmount } = render(<ResourceQuickPeek resource={resource} fetchHead={vi.fn().mockResolvedValue(head)} />);
      fireEvent.click(trigger());
      await screen.findByText(/Survey line 14/);
      unmount();
      expect(observations[0]?.disconnect).toHaveBeenCalled();
    });
  });
});
