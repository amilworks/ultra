import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FileText } from "lucide-react";

import { ResourceThumbnailPreview } from "./ResourceThumbnailPreview";
import type { ResourceRecord, ResourceTextHead } from "@/types";

// jsdom has no IntersectionObserver, so the component takes its immediate-fetch
// fallback path — perfect for asserting the rendered preview.

function resource(overrides: Partial<ResourceRecord>): ResourceRecord {
  return {
    file_id: "t1",
    original_name: "f.json",
    content_type: "application/json",
    size_bytes: 100,
    sha256: "x",
    created_at: "2026-06-22T00:00:00Z",
    resource_kind: "document",
    source_type: "upload",
    has_thumbnail: false,
    ...overrides,
  } as ResourceRecord;
}

function head(text: string, format: string): ResourceTextHead {
  return {
    file_id: "t1",
    original_name: "f",
    content_type: "text/plain",
    format,
    total_size_bytes: text.length,
    offset: 0,
    returned_bytes: text.length,
    next_offset: text.length,
    truncated: false,
    encoding: "utf-8",
    eol: "lf",
    line_count: text.split("\n").length,
    approx_total_lines: text.split("\n").length,
    text,
  };
}

describe("ResourceThumbnailPreview", () => {
  it("renders a JSON source snippet", async () => {
    const fetchHead = vi.fn().mockResolvedValue(head('{\n  "model": "deepseek_v4"\n}', "json"));
    render(
      <ResourceThumbnailPreview
        resource={resource({ file_id: "tjson", original_name: "config.json" })}
        kind="json"
        fetchHead={fetchHead}
        fallbackIcon={FileText}
        fallbackLabel="Document"
      />
    );
    await waitFor(() => {
      const snippet = document.querySelector(".resource-thumb-snippet");
      expect(snippet?.textContent ?? "").toContain('"model"');
    });
  });

  it("renders a CSV grid (header + rows)", async () => {
    const csv = "site_id,species,count\nA-014,C. ludovicianus,34\nB-002,C. gunnisoni,52\n";
    const fetchHead = vi.fn().mockResolvedValue(head(csv, "csv"));
    render(
      <ResourceThumbnailPreview
        resource={resource({ file_id: "tcsv", original_name: "survey.csv", resource_kind: "table" })}
        kind="csv"
        fetchHead={fetchHead}
        fallbackIcon={FileText}
        fallbackLabel="Table"
      />
    );
    await waitFor(() => expect(document.querySelector(".resource-thumb-grid")).toBeInTheDocument());
    expect(screen.getByText("site_id")).toBeInTheDocument();
    expect(screen.getByText("A-014")).toBeInTheDocument();
  });

  it("renders a Markdown mini (heading)", async () => {
    const md = "# Survey pipeline\n\nBatch detection over transects.\n";
    const fetchHead = vi.fn().mockResolvedValue(head(md, "markdown"));
    render(
      <ResourceThumbnailPreview
        resource={resource({ file_id: "tmd", original_name: "README.md" })}
        kind="markdown"
        fetchHead={fetchHead}
        fallbackIcon={FileText}
        fallbackLabel="Document"
      />
    );
    await waitFor(() => expect(screen.getByText("Survey pipeline")).toBeInTheDocument());
  });

  it("renders Python as a bounded source preview with a PY identity chip", async () => {
    const python = "from pathlib import Path\n\ndef compute_xrd(sample):\n    return sample.peaks\n";
    const fetchHead = vi.fn().mockResolvedValue(head(python, "text"));
    render(
      <ResourceThumbnailPreview
        resource={resource({
          file_id: "tpy",
          original_name: "compute_xrd.py",
          content_type: "application/octet-stream",
          resource_kind: "file",
        })}
        kind="text"
        fetchHead={fetchHead}
        fallbackIcon={FileText}
        fallbackLabel="Python"
      />
    );

    await waitFor(() => {
      expect(document.querySelector(".resource-thumb-snippet")?.textContent).toContain(
        "compute_xrd"
      );
    });
    expect(document.querySelector(".resource-thumb-chip")).toHaveTextContent("PY");
    expect(fetchHead).toHaveBeenCalledWith("tpy", expect.any(Number));
  });
});
