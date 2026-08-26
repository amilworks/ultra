import { describe, expect, it } from "vitest";

import {
  classifyTextResource,
  extensionOf,
  fileTypeLabel,
  formatBytes,
  formatChipLabel,
  isGzipName,
  isSourceCodeName,
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

  it("classifies source code and notebooks from filenames even when uploads use octet-stream", () => {
    for (const original_name of [
      "compute_xrd.py",
      "viewer.tsx",
      "analysis.R",
      "query.sql",
      "worker.go",
      "pipeline.sh",
    ]) {
      expect(
        classifyTextResource({ original_name, content_type: "application/octet-stream" })
      ).toBe("text");
    }
    expect(
      classifyTextResource({
        original_name: "experiment.ipynb",
        content_type: "application/octet-stream",
      })
    ).toBe("json");
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

  it("lets an explicit image or video kind win over a misleading text extension", () => {
    expect(classifyTextResource({ original_name: "plot.json", resource_kind: "image" })).toBeNull();
    expect(classifyTextResource({ original_name: "capture.py", resource_kind: "video" })).toBeNull();
    expect(classifyTextResource({ original_name: "plot.json", content_type: "image/png" })).toBeNull();
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
  it("uses compact filename-aware labels for source and data previews", () => {
    expect(formatChipLabel("json")).toBe("JSON");
    expect(formatChipLabel("markdown")).toBe("MD");
    expect(formatChipLabel("csv", "data.tsv")).toBe("TSV");
    expect(formatChipLabel("csv", "data.csv")).toBe("CSV");
    expect(formatChipLabel("text", "compute_xrd.py")).toBe("PY");
    expect(formatChipLabel("text", "pipeline.sh")).toBe("SH");
    expect(formatChipLabel("json", "analysis.ipynb")).toBe("IPYNB");
  });
});

describe("file identity labels", () => {
  it("names source formats in plain language and scientific binaries precisely", () => {
    expect(isSourceCodeName("compute_xrd.py")).toBe(true);
    expect(isSourceCodeName("NPH_shunt_002_70yo.nii.gz")).toBe(false);
    expect(fileTypeLabel("compute_xrd.py")).toBe("Python");
    expect(fileTypeLabel("viewer.tsx")).toBe("TypeScript");
    expect(fileTypeLabel("experiment.ipynb")).toBe("Jupyter notebook");
    expect(fileTypeLabel("NPH_shunt_002_70yo.nii.gz")).toBe("NIfTI");
    expect(fileTypeLabel("volume.h5")).toBe("HDF5");
  });

  it("falls back to a short uppercase extension without inventing a file type", () => {
    expect(fileTypeLabel("mesh.ply")).toBe("PLY");
    expect(fileTypeLabel("archive.unknownformat")).toBeNull();
    expect(fileTypeLabel("README")).toBeNull();
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
