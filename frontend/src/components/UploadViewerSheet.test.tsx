import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import type { UploadedFileRecord } from "@/types";

import { UploadViewerWorkspace } from "./UploadViewerSheet";

const pdfUpload: UploadedFileRecord = {
  file_id: "file_pdf",
  original_name: "bright-4b.pdf",
  content_type: "application/pdf",
  size_bytes: 4_250_000,
  sha256: "sha-pdf",
  created_at: "2026-06-07T12:30:00Z",
};

describe("UploadViewerWorkspace PDF uploads", () => {
  const renderWorkspace = (file: UploadedFileRecord) => {
    const getUploadViewer = vi.fn(async () => {
      throw new Error("PDF uploads should not load scientific image metadata");
    });
    const uploadDisplayUrl = vi.fn(
      (fileId: string) => `https://ultra.example.org/v2/uploads/${fileId}/display`
    );
    const apiClient = {
      getUploadViewer,
      uploadDisplayUrl,
    } as unknown as ApiClient;

    render(
      <UploadViewerWorkspace
        uploadedFiles={[file]}
        bisqueLinksByFileId={{}}
        apiClient={apiClient}
        active
      />
    );

    return { getUploadViewer };
  };

  it("renders a PDF reader without hydrating the scientific image viewer", async () => {
    const { getUploadViewer } = renderWorkspace(pdfUpload);

    expect(await screen.findByRole("region", { name: "PDF reader" })).toBeInTheDocument();
    expect(screen.getByText("bright-4b.pdf")).toBeInTheDocument();
    expect(screen.getByText("Document · PDF · 4.3 MB")).toBeInTheDocument();
    expect(screen.getByTitle("PDF viewer for bright-4b.pdf")).toHaveAttribute(
      "src",
      "https://ultra.example.org/v2/uploads/file_pdf/display"
    );
    expect(getUploadViewer).not.toHaveBeenCalled();
  });

  it("uses the PDF reader for legacy records with a PDF extension", async () => {
    const { getUploadViewer } = renderWorkspace({
      ...pdfUpload,
      content_type: "application/octet-stream",
    });

    expect(await screen.findByRole("region", { name: "PDF reader" })).toBeInTheDocument();
    expect(getUploadViewer).not.toHaveBeenCalled();
  });
});
