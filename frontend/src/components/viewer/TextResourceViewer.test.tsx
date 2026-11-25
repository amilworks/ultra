import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";

beforeAll(() => {
  if (!window.matchMedia) {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: (query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        addListener: () => undefined,
        removeListener: () => undefined,
        dispatchEvent: () => false,
      }),
    });
  }
});

import type { ApiClient } from "@/lib/api";
import type { ResourceCsvRows, ResourceTextHead, UploadedFileRecord } from "@/types";

import { TextResourceViewer } from "./TextResourceViewer";
import { stringifyYamlPreview } from "./yaml-preview";

function file(overrides: Partial<UploadedFileRecord>): UploadedFileRecord {
  return {
    file_id: "f1",
    original_name: "data.csv",
    content_type: "text/csv",
    size_bytes: 2048,
    sha256: "abc",
    created_at: "2026-06-22T00:00:00Z",
    ...overrides,
  } as UploadedFileRecord;
}

function fakeClient(overrides: Partial<ApiClient>): ApiClient {
  return {
    resourceDownloadUrl: (id: string) => `/v2/resources/${id}/download`,
    ...overrides,
  } as unknown as ApiClient;
}

async function renderYaml(source: string, truncated = false) {
  const name = "calibration.yaml";
  const contentType = "application/yaml";
  const lineCount = source.split("\n").length;
  const bytes = new TextEncoder().encode(source).length;
  const head: ResourceTextHead = {
    file_id: "f1",
    original_name: name,
    content_type: contentType,
    format: "yaml",
    total_size_bytes: bytes + (truncated ? 10 : 0),
    returned_bytes: bytes,
    offset: 0,
    next_offset: bytes,
    truncated,
    encoding: "utf-8",
    eol: "lf",
    line_count: lineCount,
    approx_total_lines: lineCount,
    text: source,
  };
  const apiClient = fakeClient({ resourceTextHead: vi.fn().mockResolvedValue(head) });
  const { container } = render(
    <TextResourceViewer file={file({ original_name: name, content_type: contentType })} kind="yaml" apiClient={apiClient} />
  );
  const visibleLines = () => Array.from(container.querySelectorAll(".text-viewer-line"), line => line.textContent).join("\n");
  await waitFor(() => expect(visibleLines()).toBe(source));
  return { json: screen.getByRole("radio", { name: "JSON" }), visibleLines };
}

describe("TextResourceViewer", () => {
  it("renders a CSV as a table with header, rows, and footer", async () => {
    const csv: ResourceCsvRows = {
      file_id: "f1",
      original_name: "data.csv",
      delimiter: ",",
      columns: ["id", "name", "score"],
      rows: [
        ["1", "alpha", "10"],
        ["2", "beta", "20"],
      ],
      offset_bytes: 0,
      next_offset_bytes: 40,
      returned_rows: 2,
      has_more: false,
      approx_total_rows: 2,
      total_size_bytes: 2048,
    };
    const apiClient = fakeClient({ resourceCsvRows: vi.fn().mockResolvedValue(csv) });

    render(<TextResourceViewer file={file({})} kind="csv" apiClient={apiClient} />);

    expect(await screen.findByText("data.csv")).toBeInTheDocument();
    expect(screen.getByText("CSV")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("alpha")).toBeInTheDocument());
    expect(screen.getByText("beta")).toBeInTheDocument();
    // Column header present.
    expect(screen.getByRole("columnheader", { name: "name" })).toBeInTheDocument();
    // Footer summarizes the table.
    expect(screen.getByText(/3 columns/)).toBeInTheDocument();
  });

  it("renders JSON pretty-printed and converts to YAML via the toggle", async () => {
    const head: ResourceTextHead = {
      file_id: "f1",
      original_name: "config.json",
      content_type: "application/json",
      format: "json",
      total_size_bytes: 40,
      offset: 0,
      returned_bytes: 40,
      next_offset: 40,
      truncated: false,
      encoding: "utf-8",
      eol: "lf",
      line_count: 1,
      approx_total_lines: 1,
      text: '{"model":"deepseek_v4","enabled":true}',
    };
    const apiClient = fakeClient({ resourceTextHead: vi.fn().mockResolvedValue(head) });

    render(<TextResourceViewer file={file({ original_name: "config.json", content_type: "application/json" })} kind="json" apiClient={apiClient} />);

    // Pretty-printed JSON shows the key on its own line.
    await waitFor(() => expect(screen.getByText(/"model"/)).toBeInTheDocument());

    // Toggle to YAML (Radix renders single-mode items as radios; the YAML
    // highlighter splits a "key:" across spans, so assert on the surface text).
    fireEvent.click(screen.getByText("YAML"));
    await waitFor(() => {
      const surface = document.querySelector(".text-viewer-surface");
      expect(surface?.textContent ?? "").toContain("model: deepseek_v4");
    });
  });

  it("converts a complete YAML resource with shared calibration values to JSON", async () => {
    const { json, visibleLines } = await renderYaml("base: &spacing [2, 4]\npixels: *spacing");
    fireEvent.click(json);
    await waitFor(() => expect(JSON.parse(visibleLines())).toEqual({ base: [2, 4], pixels: [2, 4] }));
  });

  it("keeps a valid YAML prefix in its source format when the rest of the file is absent", async () => {
    const { json } = await renderYaml("pixels: [2, 4]", true);
    expect(json).toBeDisabled();
    expect(screen.getByRole("radio", { name: "YAML" })).toHaveAttribute("aria-checked", "true");
  });

  it("returns to the source view and explains an unsuccessful conversion", async () => {
    const { json, visibleLines } = await renderYaml("pixels: [");
    fireEvent.click(json);
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Showing the original"));
    expect(json).toBeDisabled();
    expect(screen.getByRole("radio", { name: "YAML" })).toHaveAttribute("aria-checked", "true");
    expect(visibleLines()).toBe("pixels: [");
  });

  it("accounts for repeated values when budgeting the YAML preview", () => {
    const calibration = { name: "microscopy calibration", units: "micrometers" };
    expect(JSON.parse(stringifyYamlPreview([calibration], 80))).toEqual([calibration]);
    expect(() => stringifyYamlPreview([calibration, calibration], 80)).toThrow(RangeError);
  });

  it("caps the wrapped <pre> so enabling wrap on a large head cannot freeze the main thread", async () => {
    const bigText = "x".repeat(250_000); // > WRAP_CHAR_CAP (200k)
    const head: ResourceTextHead = {
      file_id: "f1",
      original_name: "big.txt",
      content_type: "text/plain",
      format: "text",
      total_size_bytes: 250_000,
      offset: 0,
      returned_bytes: 250_000,
      next_offset: 250_000,
      truncated: false,
      encoding: "utf-8",
      eol: "lf",
      line_count: 1,
      approx_total_lines: 1,
      text: bigText,
    };
    const apiClient = fakeClient({ resourceTextHead: vi.fn().mockResolvedValue(head) });
    render(<TextResourceViewer file={file({ original_name: "big.txt", content_type: "text/plain" })} kind="text" apiClient={apiClient} />);

    await screen.findByText("big.txt");
    fireEvent.click(screen.getByLabelText("Toggle line wrap"));
    await waitFor(() => expect(screen.getByText(/Wrapped preview limited/)).toBeInTheDocument());
    // Only the capped slice is in the DOM, not the full 250k characters.
    const pre = document.querySelector(".text-viewer-pre-wrap");
    expect((pre?.textContent ?? "").length).toBeLessThanOrEqual(200_000);
  });

  it("shows a calm large-file banner when the head is truncated", async () => {
    const head: ResourceTextHead = {
      file_id: "f1",
      original_name: "huge.log",
      content_type: "text/plain",
      format: "text",
      total_size_bytes: 5_000_000,
      offset: 0,
      returned_bytes: 2_000_000,
      next_offset: 2_000_000,
      truncated: true,
      encoding: "utf-8",
      eol: "lf",
      line_count: 1200,
      approx_total_lines: 3000,
      text: "line a\nline b\nline c\n",
    };
    const apiClient = fakeClient({ resourceTextHead: vi.fn().mockResolvedValue(head) });

    render(<TextResourceViewer file={file({ original_name: "huge.log", content_type: "text/plain" })} kind="text" apiClient={apiClient} />);

    expect(await screen.findByText(/Large file/)).toBeInTheDocument();
    expect(screen.getByText(/Download for the full file/)).toBeInTheDocument();
  });
});
