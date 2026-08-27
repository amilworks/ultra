import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import { NoteContextPicker } from "./NoteContextPicker";

const noteItem = {
  note_id: "note_1",
  title: "Calibration log",
  snippet: "Daily drift observations",
  pinned: false,
  revision: 3,
  updated_at: "2026-08-25T12:00:00Z",
};

describe("NoteContextPicker", () => {
  it("explains the conversation boundary and attaches the chosen revision", async () => {
    const onSelect = vi.fn();
    const apiClient = {
      listNotes: vi.fn().mockResolvedValue({ notes: [noteItem], total_count: 1 }),
    } as unknown as ApiClient;
    render(
      <NoteContextPicker
        apiClient={apiClient}
        open
        selectedNoteIds={[]}
        onOpenChange={vi.fn()}
        onSelect={onSelect}
      />
    );

    expect(
      screen.getByText("Ultra can read it for this message. Content used in chat becomes part of this conversation.")
    ).toBeInTheDocument();
    fireEvent.click(await screen.findByText("Calibration log"));
    expect(onSelect).toHaveBeenCalledWith({
      note_id: "note_1",
      title: "Calibration log",
      revision: 3,
    });
  });

  it("makes the eight-Note cap visible", async () => {
    const apiClient = {
      listNotes: vi.fn().mockResolvedValue({ notes: [noteItem], total_count: 1 }),
    } as unknown as ApiClient;
    render(
      <NoteContextPicker
        apiClient={apiClient}
        open
        selectedNoteIds={Array.from({ length: 8 }, (_, index) => `note_${index + 2}`)}
        onOpenChange={vi.fn()}
        onSelect={vi.fn()}
      />
    );

    expect(await screen.findByText("Eight Notes are attached. Remove one before adding another.")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Calibration log").closest("[cmdk-item]")).toHaveAttribute("data-disabled", "true"));
  });
});
