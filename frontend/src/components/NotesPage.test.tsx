import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { NotesPage } from "./NotesPage";
import { ApiError, type ApiClient, type NoteRecord, type NoteWritePayload } from "@/lib/api";

vi.mock("@/components/notes/MarkdownNoteEditor", async () => {
  const React = await import("react");

  function MarkdownNoteEditor({
    defaultMarkdown,
    onMarkdownChange,
    onBlur,
    bindApi,
  }: {
    defaultMarkdown: string;
    onMarkdownChange: (markdown: string) => void;
    onBlur: () => void;
    bindApi: (api: {
      exec: () => void;
      applyLink: () => void;
      removeLink: () => void;
      insertMarkdown: (markdown: string) => void;
      focus: () => void;
    } | null) => void;
  }) {
    const [value, setValue] = React.useState(defaultMarkdown);
    const valueRef = React.useRef(defaultMarkdown);
    const bodyRef = React.useRef<HTMLTextAreaElement | null>(null);

    React.useEffect(() => {
      bindApi({
        exec: () => undefined,
        applyLink: () => undefined,
        removeLink: () => undefined,
        insertMarkdown: (markdown) => {
          const next = valueRef.current + markdown;
          valueRef.current = next;
          setValue(next);
          onMarkdownChange(`${next}\n`);
        },
        focus: () => bodyRef.current?.focus(),
      });
      return () => bindApi(null);
    }, [bindApi, onMarkdownChange]);

    return (
      <textarea
        ref={bodyRef}
        aria-label="Note body"
        value={value}
        onBlur={onBlur}
        onChange={(event) => {
          const next = event.target.value;
          valueRef.current = next;
          setValue(next);
          // Milkdown's serializer appends a trailing newline; the page owns
          // trimming it without rewriting the durable Markdown source.
          onMarkdownChange(`${next}\n`);
        }}
      />
    );
  }

  return { MarkdownNoteEditor };
});

const now = "2026-08-08T18:00:00.000Z";

const note = (noteId: string, overrides: Partial<NoteRecord> = {}): NoteRecord => ({
  note_id: noteId,
  title: `Note ${noteId}`,
  body_markdown: `Body ${noteId}`,
  pinned: false,
  editor_mode: "markdown",
  revision: 1,
  content_digest: `digest-${noteId}`,
  created_at: now,
  updated_at: now,
  ...overrides,
});

const apiFor = (records: NoteRecord[] = []) => {
  const byId = new Map(records.map((record) => [record.note_id, record]));
  const api = {
    listNotes: vi.fn().mockResolvedValue({
      notes: records.map((record) => ({
        note_id: record.note_id,
        title: record.title,
        snippet: record.body_markdown,
        pinned: record.pinned,
        revision: record.revision,
        updated_at: record.updated_at,
      })),
      total_count: records.length,
    }),
    createNote: vi.fn(async (payload: NoteWritePayload) =>
      note("created", {
        title: payload.title ?? "",
        body_markdown: payload.body_markdown ?? "",
        pinned: payload.pinned ?? false,
        editor_mode: payload.editor_mode ?? "markdown",
      })
    ),
    getNote: vi.fn(async (noteId: string) => byId.get(noteId) ?? note(noteId)),
    updateNote: vi.fn(async (noteId: string, payload: NoteWritePayload) =>
      note(noteId, {
        title: payload.title ?? "",
        body_markdown: payload.body_markdown ?? "",
        pinned: payload.pinned ?? false,
        editor_mode: payload.editor_mode ?? "markdown",
        revision: (payload.expected_revision ?? 0) + 1,
        content_digest: `updated-${noteId}`,
      })
    ),
    deleteNote: vi.fn().mockResolvedValue(undefined),
    listResources: vi.fn().mockResolvedValue({ count: 0, resources: [] }),
    uploadFiles: vi.fn().mockResolvedValue({ uploaded: [] }),
    resourceDownloadUrl: vi.fn((fileId: string) => `/resources/${fileId}`),
  };
  return { api: api as unknown as ApiClient, mocks: api };
};

const advance = async (milliseconds = 0) => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(milliseconds);
    await Promise.resolve();
  });
};

const startDraft = async () => {
  const newButtons = screen.getAllByRole("button", { name: "New note" });
  fireEvent.click(newButtons[0]);
  await advance();
  await advance();
  return screen.getByLabelText("Note body") as HTMLTextAreaElement;
};

describe("NotesPage writing flow", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    });
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("focuses the body and creates nothing until the draft has meaningful text", async () => {
    const { api, mocks } = apiFor();
    render(<NotesPage apiClient={api} />);
    await advance();

    const body = await startDraft();
    expect(body).toHaveFocus();
    expect(mocks.createNote).not.toHaveBeenCalled();

    await advance(800);
    expect(mocks.createNote).not.toHaveBeenCalled();

    fireEvent.change(body, { target: { value: "First observation" } });
    expect(screen.queryByLabelText("Note title, optional")).not.toBeInTheDocument();

    await advance(699);
    expect(mocks.createNote).not.toHaveBeenCalled();
    await advance(1);

    expect(mocks.createNote).toHaveBeenCalledWith({
      title: "",
      body_markdown: "First observation",
      pinned: false,
      editor_mode: "markdown",
    });
    const row = screen.getByRole("button", { name: /First observation/ });
    expect(row).toBeInTheDocument();
    expect(row.querySelector(".notes-row-snippet")).toBeEmptyDOMElement();
    expect(body).toHaveFocus();
  });

  it("discards an untouched local draft with Escape and never calls the API", async () => {
    const { api, mocks } = apiFor();
    render(<NotesPage apiClient={api} />);
    await advance();
    await startDraft();

    fireEvent.keyDown(window, { key: "Escape" });
    await advance();

    expect(screen.getByText("Select a note or start a new one.")).toBeInTheDocument();
    expect(mocks.createNote).not.toHaveBeenCalled();
    expect(mocks.deleteNote).not.toHaveBeenCalled();
  });

  it("only exposes Use in chat when the caller enables model Notes context", async () => {
    const first = note("first", { title: "Field protocol", revision: 7 });
    const { api } = apiFor([first]);
    const disabledView = render(<NotesPage apiClient={api} />);
    await advance();
    await advance();

    fireEvent.pointerDown(screen.getByRole("button", { name: "More note actions" }), {
      button: 0,
      ctrlKey: false,
    });
    await advance();
    expect(screen.queryByRole("menuitem", { name: "Use in chat" })).not.toBeInTheDocument();
    disabledView.unmount();

    const onUseInChat = vi.fn();
    render(<NotesPage apiClient={api} onUseInChat={onUseInChat} />);
    await advance();
    await advance();
    fireEvent.pointerDown(screen.getByRole("button", { name: "More note actions" }), {
      button: 0,
      ctrlKey: false,
    });
    await advance();
    fireEvent.click(screen.getByRole("menuitem", { name: "Use in chat" }));
    await advance();

    expect(onUseInChat).toHaveBeenCalledWith({
      note_id: "first",
      title: "Field protocol",
      revision: 7,
    });
  });

  it("keeps the title optional until the writer explicitly asks for it", async () => {
    const { api, mocks } = apiFor();
    render(<NotesPage apiClient={api} />);
    await advance();
    await startDraft();

    expect(screen.queryByLabelText("Note title, optional")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Add a title" }));
    await advance();

    const title = screen.getByLabelText("Note title, optional");
    expect(title).toHaveFocus();
    fireEvent.change(title, { target: { value: "Treatment response" } });
    await advance(700);

    expect(mocks.createNote).toHaveBeenCalledWith({
      title: "Treatment response",
      body_markdown: "",
      pinned: false,
      editor_mode: "markdown",
    });
  });

  it("serializes a create followed by edits so the newest draft wins", async () => {
    const { api, mocks } = apiFor();
    let resolveCreate: ((record: NoteRecord) => void) | null = null;
    mocks.createNote.mockImplementation(
      () =>
        new Promise<NoteRecord>((resolve) => {
          resolveCreate = resolve;
        })
    );
    render(<NotesPage apiClient={api} />);
    await advance();
    const body = await startDraft();

    fireEvent.change(body, { target: { value: "Initial finding" } });
    await advance(700);
    expect(mocks.createNote).toHaveBeenCalledTimes(1);
    expect(mocks.updateNote).not.toHaveBeenCalled();

    fireEvent.change(body, { target: { value: "Revised finding" } });
    await act(async () => {
      resolveCreate?.(
        note("created", { title: "", body_markdown: "Initial finding" })
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mocks.updateNote).toHaveBeenCalledWith("created", {
      title: "",
      body_markdown: "Revised finding",
      pinned: false,
      editor_mode: "markdown",
      expected_revision: 1,
    });
  });

  it("orders deletion after an in-flight create so no server note is stranded", async () => {
    const { api, mocks } = apiFor();
    let resolveCreate: ((record: NoteRecord) => void) | null = null;
    mocks.createNote.mockImplementation(
      () =>
        new Promise<NoteRecord>((resolve) => {
          resolveCreate = resolve;
        })
    );
    render(<NotesPage apiClient={api} />);
    await advance();
    const body = await startDraft();

    fireEvent.change(body, { target: { value: "Temporary finding" } });
    await advance(700);
    expect(mocks.createNote).toHaveBeenCalledTimes(1);

    const more = screen.getByRole("button", { name: "More note actions" });
    fireEvent.pointerDown(more, { button: 0, ctrlKey: false });
    await advance();
    fireEvent.click(screen.getByRole("menuitem", { name: "Discard draft" }));
    await advance();
    fireEvent.click(screen.getByRole("button", { name: "Discard draft" }));
    await advance();

    expect(mocks.deleteNote).not.toHaveBeenCalled();
    await act(async () => {
      resolveCreate?.(note("created", { title: "", body_markdown: "Temporary finding" }));
      await Promise.resolve();
      await Promise.resolve();
    });
    await advance();

    expect(mocks.deleteNote).toHaveBeenCalledWith("created");
    expect(screen.getByText("Select a note or start a new one.")).toBeInTheDocument();
  });

  it("does not attach a slow upload to a different note after navigation", async () => {
    const first = note("first", { title: "First note", body_markdown: "First body" });
    const second = note("second", { title: "Second note", body_markdown: "Second body" });
    const { api, mocks } = apiFor([first, second]);
    let resolveUpload: ((value: { uploaded: unknown[] }) => void) | null = null;
    mocks.uploadFiles.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveUpload = resolve;
        })
    );
    const { container } = render(<NotesPage apiClient={api} />);
    await advance();
    await advance();

    const input = container.querySelector<HTMLInputElement>('input[type="file"]');
    expect(input).not.toBeNull();
    fireEvent.change(input!, {
      target: { files: [new File(["pixels"], "capture.tif", { type: "image/tiff" })] },
    });
    expect(mocks.uploadFiles).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: /Second note/ }));
    await advance();
    expect(screen.getByLabelText("Note body")).toHaveValue("Second body");

    await act(async () => {
      resolveUpload?.({ uploaded: [{}] });
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByLabelText("Note body")).toHaveValue("Second body");
    expect(mocks.updateNote).not.toHaveBeenCalled();
  });

  it("keeps a failed note load selected and offers an in-place retry", async () => {
    const first = note("first", { title: "First note", body_markdown: "Recovered body" });
    const { api, mocks } = apiFor([first]);
    mocks.getNote.mockRejectedValueOnce(new Error("request timed out")).mockResolvedValue(first);
    render(<NotesPage apiClient={api} />);
    await advance();
    await advance();

    expect(screen.getByRole("alert")).toHaveTextContent("Couldn’t open this note");
    expect(screen.getByRole("alert")).toHaveTextContent("request timed out");
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await advance();

    expect(screen.getByLabelText("Note body")).toHaveValue("Recovered body");
    expect(mocks.getNote).toHaveBeenCalledTimes(2);
  });

  it("clears the specific-note deep link when mobile navigation returns to the list", async () => {
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: true,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));
    const first = note("first", { title: "First note", body_markdown: "First body" });
    const { api } = apiFor([first]);
    const onActiveNoteChange = vi.fn();

    render(
      <NotesPage
        apiClient={api}
        initialNoteId="first"
        onActiveNoteChange={onActiveNoteChange}
      />
    );
    await advance();
    await advance();

    expect(screen.getByLabelText("Note body")).toHaveValue("First body");
    fireEvent.click(screen.getByRole("button", { name: "Notes" }));
    await advance();

    expect(onActiveNoteChange).toHaveBeenLastCalledWith(null);
  });

  it("mirrors browser Back and Forward between the mobile list and a warm Note", async () => {
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: true,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));
    const first = note("first", { title: "First note", body_markdown: "First body" });
    const { api } = apiFor([first]);
    const view = render(
      <NotesPage apiClient={api} initialNoteId="first" listRequestVersion={0} />
    );
    await advance();
    await advance();
    expect(screen.getByTestId("notes-page")).not.toHaveAttribute("data-mobile-list");

    view.rerender(
      <NotesPage apiClient={api} initialNoteId={null} listRequestVersion={1} />
    );
    await advance();
    expect(screen.getByTestId("notes-page")).toHaveAttribute("data-mobile-list", "true");

    view.rerender(
      <NotesPage apiClient={api} initialNoteId="first" listRequestVersion={1} />
    );
    await advance();
    expect(screen.getByTestId("notes-page")).not.toHaveAttribute("data-mobile-list");
  });

  it("keeps the current note open when sync fails instead of navigating away", async () => {
    const first = note("first", { title: "First note", body_markdown: "First body" });
    const second = note("second", { title: "Second note", body_markdown: "Second body" });
    const { api, mocks } = apiFor([first, second]);
    mocks.updateNote.mockRejectedValue(new Error("network unavailable"));
    render(<NotesPage apiClient={api} />);
    await advance();
    await advance();

    const body = screen.getByLabelText("Note body");
    fireEvent.change(body, { target: { value: "Unsynced finding" } });
    await advance(700);
    expect(screen.getByText("Couldn’t sync")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Second note/ }));
    await advance();

    expect(mocks.getNote).not.toHaveBeenCalledWith("second");
    expect(screen.getByLabelText("Note body")).toHaveValue("Unsynced finding");
    expect(screen.getByText("Couldn’t sync")).toBeInTheDocument();
  });

  it("preserves local writing on a revision conflict and requires an explicit choice", async () => {
    const first = note("first", { title: "Protocol", body_markdown: "Original" });
    const latest = note("first", {
      title: "Protocol",
      body_markdown: "Updated elsewhere",
      revision: 2,
      content_digest: "latest",
    });
    const { api, mocks } = apiFor([first]);
    mocks.updateNote.mockRejectedValueOnce(
      new ApiError("note revision conflict", 409, { code: "note_revision_conflict" })
    );
    mocks.getNote.mockResolvedValueOnce(first).mockResolvedValueOnce(latest);
    render(<NotesPage apiClient={api} />);
    await advance();
    await advance();

    const body = screen.getByLabelText("Note body");
    fireEvent.change(body, { target: { value: "My unsaved revision" } });
    await advance(700);

    expect(body).toHaveValue("My unsaved revision");
    expect(screen.getByRole("alert")).toHaveTextContent("This note changed elsewhere");
    expect(screen.getByRole("button", { name: "Use latest" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save my version" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Use latest" }));
    await advance();
    expect(screen.getByLabelText("Note body")).toHaveValue("Updated elsewhere");
  });

  it("freezes automatic writes after conflict and overwrites only after explicit approval", async () => {
    const first = note("first", { title: "Protocol", body_markdown: "Original" });
    const latest = note("first", {
      title: "Protocol",
      body_markdown: "Updated elsewhere",
      revision: 2,
      content_digest: "latest",
    });
    const { api, mocks } = apiFor([first]);
    mocks.updateNote.mockRejectedValueOnce(
      new ApiError("note revision conflict", 409, { code: "note_revision_conflict" })
    );
    mocks.getNote.mockResolvedValueOnce(first).mockResolvedValueOnce(latest);
    render(<NotesPage apiClient={api} />);
    await advance();
    await advance();

    const body = screen.getByLabelText("Note body");
    fireEvent.change(body, { target: { value: "My reviewed version" } });
    await advance(700);
    expect(mocks.updateNote).toHaveBeenCalledTimes(1);

    fireEvent.blur(body);
    await advance(1000);
    expect(mocks.updateNote).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "Save my version" }));
    await advance();
    expect(mocks.updateNote).toHaveBeenCalledTimes(2);
    expect(mocks.updateNote).toHaveBeenLastCalledWith("first", {
      title: "Protocol",
      body_markdown: "My reviewed version",
      pinned: false,
      editor_mode: "markdown",
      expected_revision: 2,
    });
  });

  it("opens recent Ultra resources from the quiet paperclip action", async () => {
    const { api, mocks } = apiFor();
    render(<NotesPage apiClient={api} />);
    await advance();
    await startDraft();

    fireEvent.click(screen.getByRole("button", { name: "Link an Ultra resource" }));
    await advance(160);

    expect(screen.getByRole("dialog", { name: "Link an Ultra resource" })).toBeInTheDocument();
    expect(mocks.listResources).toHaveBeenCalledWith({ limit: 10, query: undefined });
  });
});
