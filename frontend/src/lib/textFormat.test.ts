import { describe, expect, it } from "vitest";

import {
  classifyTextResource,
  extensionOf,
  formatBytes,
  formatChipLabel,
  isGzipName,
} from "./textFormat";

describe("classifyTextResource", () => {
  it("classifies by extension first (content_type is unreliable for chunked uploads)", () => {
    expect(classifyTextResource({ original_name: "data.csv", content_type: "application/octet-stream" })).toBe("csv");
    expect(classifyTextResource({ original_name: "manifest.json", content_type: "application/octet-stream" })).toBe(
      "json"
    );
    expect(classifyTextResource({ original_name: "config.yaml", content_type: "" })).toBe("yaml");
    expect(classifyTextResource({ original_name: "feed.xml" })).toBe("xml");
    expect(classifyTextResource({ original_name: "README.md" })).toBe("markdown");
    expect(classifyTextResource({ original_name: "run.log" })).toBe("text");
    expect(classifyTextResource({ original_name: "notes.tsv" })).toBe("csv");
  });

  it("looks past a trailing .gz", () => {
    expect(classifyTextResource({ original_name: "big.csv.gz" })).toBe("csv");
    expect(classifyTextResource({ original_name: "data.json.gz" })).toBe("json");
  });

  it("falls back to table kind and textual content_type when there is no known extension", () => {
    expect(classifyTextResource({ original_name: "export", resource_kind: "table" })).toBe("csv");
    expect(classifyTextResource({ original_name: "blob", content_type: "application/json" })).toBe("json");
    expect(classifyTextResource({ original_name: "blob", content_type: "text/plain" })).toBe("text");
  });

  it("does not mistake binary or generic documents and CIFTI matrices for text", () => {
    for (const original_name of ["report.pdf", "slides.ppt", "slides.pptx", "letter.doc", "letter.docx"]) {
      expect(
        classifyTextResource({ original_name, content_type: "application/octet-stream", resource_kind: "document" })
      ).toBeNull();
    }
    expect(classifyTextResource({ original_name: "rest.dtseries.nii", resource_kind: "document" })).toBeNull();
    expect(classifyTextResource({ original_name: "extensionless", resource_kind: "document" })).toBeNull();
  });

  it("never claims image or video resources (so the image viewer still wins)", () => {
    expect(classifyTextResource({ original_name: "cells.ome.tif", content_type: "image/tiff" })).toBeNull();
    expect(classifyTextResource({ original_name: "scan.png", resource_kind: "image" })).toBeNull();
    expect(classifyTextResource({ original_name: "clip.mp4", content_type: "video/mp4" })).toBeNull();
    expect(classifyTextResource({ original_name: "model.h5" })).toBeNull();
    expect(classifyTextResource(null)).toBeNull();
  });
});

describe("extensionOf / isGzipName", () => {
  it("returns the final extension, ignoring .gz", () => {
    expect(extensionOf("a/b/data.CSV")).toBe("csv");
    expect(extensionOf("data.csv.gz")).toBe("csv");
    expect(extensionOf("noext")).toBe("");
  });
  it("detects gzip names", () => {
    expect(isGzipName("data.csv.gz")).toBe(true);
    expect(isGzipName("data.csv")).toBe(false);
  });
});

describe("formatChipLabel", () => {
  it("uppercases the format and distinguishes TSV", () => {
    expect(formatChipLabel("json")).toBe("JSON");
    expect(formatChipLabel("markdown")).toBe("MD");
    expect(formatChipLabel("csv", "data.tsv")).toBe("TSV");
    expect(formatChipLabel("csv", "data.csv")).toBe("CSV");
  });
});

describe("formatBytes", () => {
  it("renders calm rounded sizes in base 1000", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1500)).toBe("1.5 KB");
    expect(formatBytes(500_000_000)).toBe("500 MB");
    expect(formatBytes(2_410_000_000)).toBe("2.4 GB");
  });
});
