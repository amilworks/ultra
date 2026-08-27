import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { NotesPage } from "./NotesPage";
import { ApiError, type ApiClient, type NoteRecord, type NoteWritePayload } from "@/lib/api";
import { writeNoteDraftRecovery } from "@/lib/noteDraftRecovery";

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
        content_updated_at: record.content_updated_at,
        updated_at: record.updated_at,
      })),
      total_count: records.length,
    }),
    createNote: vi.fn(async (payload: NoteWritePayload, idempotencyKey?: string) => {
      void idempotencyKey;
      return note("created", {
        title: payload.title ?? "",
        body_markdown: payload.body_markdown ?? "",
        pinned: payload.pinned ?? false,
        editor_mode: payload.editor_mode ?? "markdown",
      });
    }),
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

const installMemoryStorage = (): Storage => {
  const values = new Map<string, string>();
  const storage = {
    get length() {
      return values.size;
    },
    clear: () => values.clear(),
    getItem: (key: string) => values.get(key) ?? null,
    key: (index: number) => [...values.keys()][index] ?? null,
    removeItem: (key: string) => {
      values.delete(key);
    },
    setItem: (key: string, value: string) => {
      values.set(key, value);
    },
  } as Storage;
  Object.defineProperty(window, "localStorage", { configurable: true, value: storage });
  return storage;
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
    installMemoryStorage();
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

    expect(mocks.createNote).toHaveBeenCalledWith(
      {
        title: "",
        body_markdown: "First observation",
        pinned: false,
        editor_mode: "markdown",
      },
      expect.stringMatching(/^note-create:/)
    );
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

    expect(mocks.createNote).toHaveBeenCalledWith(
      {
        title: "Treatment response",
        body_markdown: "",
        pinned: false,
        editor_mode: "markdown",
      },
      expect.stringMatching(/^note-create:/)
    );
  });

  it("lets Shift+Tab leave the title backwards and Escape then Tab leave raw source", async () => {
    const first = note("first", {
      title: "Protocol",
      body_markdown: "Raw body",
      editor_mode: "plaintext",
    });
    const { api } = apiFor([first]);
    render(<NotesPage apiClient={api} />);
    await advance();
    await advance();

    const title = screen.getByLabelText("Note title, optional");
    const body = screen.getByLabelText("Note body") as HTMLTextAreaElement;
    title.focus();
    expect(fireEvent.keyDown(title, { key: "Tab", shiftKey: true })).toBe(true);
    expect(body).not.toHaveFocus();

    title.focus();
    expect(fireEvent.keyDown(title, { key: "Tab" })).toBe(false);
    expect(body).toHaveFocus();

    body.setSelectionRange(0, 0);
    expect(fireEvent.keyDown(body, { key: "Tab" })).toBe(false);
    expect(body).toHaveValue("  Raw body");
    fireEvent.keyDown(body, { key: "Escape" });
    expect(fireEvent.keyDown(body, { key: "Tab" })).toBe(true);
    expect(body).toHaveValue("  Raw body");
  });

  it("updates the visible Edited time from content_updated_at after autosave", async () => {
    const first = note("first", {
      title: "Protocol",
      body_markdown: "Original",
      content_updated_at: "2026-01-01T00:00:00.000Z",
    });
    const { api, mocks } = apiFor([first]);
    render(<NotesPage apiClient={api} />);
    await advance();
    await advance();

    const savedAt = new Date(Date.now()).toISOString();
    mocks.updateNote.mockResolvedValueOnce(
      note("first", {
        title: "Protocol",
        body_markdown: "Updated",
        revision: 2,
        content_updated_at: savedAt,
        updated_at: savedAt,
      })
    );
    fireEvent.change(screen.getByLabelText("Note body"), {
      target: { value: "Updated" },
    });
    await advance(700);

    expect(screen.getByText("Edited just now")).toBeInTheDocument();
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

  it.each([
    ["lost transport response", new TypeError("response lost")],
    ["server error", new ApiError("temporarily unavailable", 503, { code: "unavailable" })],
  ])("replays an uncertain create after a %s with the same key and original payload before saving newer edits", async (_case, failure) => {
    const { api, mocks } = apiFor();
    mocks.createNote
      .mockRejectedValueOnce(failure)
      .mockResolvedValueOnce(note("created", { title: "", body_markdown: "Initial finding" }));
    render(<NotesPage apiClient={api} recoveryScope="researcher" />);
    await advance();
    const body = await startDraft();

    fireEvent.change(body, { target: { value: "Initial finding" } });
    await advance(700);
    expect(screen.getByText("Couldn’t sync")).toBeInTheDocument();
    fireEvent.change(body, { target: { value: "Revised after uncertain response" } });
    fireEvent.blur(body);
    await advance();

    expect(mocks.createNote).toHaveBeenCalledTimes(2);
    const firstCreate = mocks.createNote.mock.calls[0];
    const replayCreate = mocks.createNote.mock.calls[1];
    expect(replayCreate[1]).toBe(firstCreate[1]);
    expect(replayCreate[0]).toEqual(firstCreate[0]);
    expect(mocks.updateNote).toHaveBeenCalledWith("created", {
      title: "",
      body_markdown: "Revised after uncertain response",
      pinned: false,
      editor_mode: "markdown",
      expected_revision: 1,
    });
  });

  it("keeps an uncertain create frozen when its exact replay meets a current 401", async () => {
    const { api, mocks } = apiFor();
    mocks.createNote
      .mockRejectedValueOnce(new TypeError("response lost"))
      .mockRejectedValueOnce(
        new ApiError("sign in again", 401, { error: "unauthorized" })
      )
      .mockResolvedValueOnce(
        note("created", { title: "", body_markdown: "Initial finding" })
      );
    render(<NotesPage apiClient={api} recoveryScope="researcher" />);
    await advance();
    const body = await startDraft();

    fireEvent.change(body, { target: { value: "Initial finding" } });
    await advance(700);
    fireEvent.change(body, { target: { value: "Edit after uncertainty" } });
    fireEvent.blur(body);
    await advance();

    expect(mocks.createNote).toHaveBeenCalledTimes(2);
    expect(mocks.createNote.mock.calls[1]).toEqual(mocks.createNote.mock.calls[0]);
    expect(screen.getByText("Couldn’t sync")).toBeInTheDocument();

    fireEvent.change(body, { target: { value: "Latest local edit" } });
    fireEvent.blur(body);
    await advance();

    expect(mocks.createNote).toHaveBeenCalledTimes(3);
    expect(mocks.createNote.mock.calls[2]).toEqual(mocks.createNote.mock.calls[0]);
    expect(mocks.updateNote).toHaveBeenCalledWith("created", {
      title: "",
      body_markdown: "Latest local edit",
      pinned: false,
      editor_mode: "markdown",
      expected_revision: 1,
    });
  });

  it("releases a restored uncertain create after stable validation so it can be corrected", async () => {
    const { api, mocks } = apiFor();
    writeNoteDraftRecovery(window.localStorage, "researcher", {
      note_id: "__ultra_local_note_draft__",
      title: "",
      body_markdown: "Frozen invalid payload",
      pinned: false,
      editor_mode: "markdown",
      expected_revision: 0,
      create_key: "frozen-create-key",
      create_attempt: {
        title: "",
        body_markdown: "Frozen invalid payload",
        pinned: false,
        editor_mode: "markdown",
      },
    });
    mocks.createNote
      .mockRejectedValueOnce(
        new ApiError("create did not commit", 400, {
          code: "note_create_not_committed",
        })
      )
      .mockResolvedValueOnce(
        note("created", { title: "", body_markdown: "Corrected payload" })
      );

    render(<NotesPage apiClient={api} recoveryScope="researcher" />);
    await advance();
    await advance(700);

    expect(mocks.createNote).toHaveBeenCalledWith(
      {
        title: "",
        body_markdown: "Frozen invalid payload",
        pinned: false,
        editor_mode: "markdown",
      },
      "frozen-create-key"
    );
    const body = screen.getByLabelText("Note body");
    expect(body).toBeEnabled();
    expect(screen.getByText("Couldn’t sync")).toBeInTheDocument();

    fireEvent.change(body, { target: { value: "Corrected payload" } });
    fireEvent.blur(body);
    await advance();

    expect(mocks.createNote).toHaveBeenCalledTimes(2);
    expect(mocks.createNote.mock.calls[1]?.[0]).toMatchObject({
      body_markdown: "Corrected payload",
    });
    expect(mocks.createNote.mock.calls[1]?.[1]).not.toBe("frozen-create-key");
  });

  it("starts a fresh create with corrected content after a deterministic rejection", async () => {
    const { api, mocks } = apiFor();
    mocks.createNote
      .mockRejectedValueOnce(new ApiError("title is too long", 400, { error: "title is too long" }))
      .mockResolvedValueOnce(note("created", { title: "", body_markdown: "Corrected finding" }));
    render(<NotesPage apiClient={api} recoveryScope="researcher" />);
    await advance();
    const body = await startDraft();

    fireEvent.change(body, { target: { value: "Rejected finding" } });
    await advance(700);
    expect(screen.getByText("Couldn’t sync")).toBeInTheDocument();

    fireEvent.change(body, { target: { value: "Corrected finding" } });
    fireEvent.blur(body);
    await advance();

    expect(mocks.createNote).toHaveBeenCalledTimes(2);
    const firstCreate = mocks.createNote.mock.calls[0];
    const correctedCreate = mocks.createNote.mock.calls[1];
    expect(correctedCreate[0]).toEqual({
      title: "",
      body_markdown: "Corrected finding",
      pinned: false,
      editor_mode: "markdown",
    });
    expect(correctedCreate[1]).not.toBe(firstCreate[1]);
    expect(mocks.updateNote).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Note body")).toHaveValue("Corrected finding");
  });

  it("restores an exact same-revision draft before autosave and acknowledges it only after save", async () => {
    const first = note("first", { title: "Protocol", body_markdown: "Server copy", revision: 4 });
    const { api, mocks } = apiFor([first]);
    writeNoteDraftRecovery(window.localStorage, "researcher", {
      note_id: "first",
      title: "Protocol",
      body_markdown: "Recovered unsaved copy",
      pinned: false,
      editor_mode: "markdown",
      expected_revision: 4,
    });

    render(<NotesPage apiClient={api} recoveryScope="researcher" />);
    await advance();
    await advance();

    expect(screen.getByLabelText("Note body")).toHaveValue("Recovered unsaved copy");
    expect(screen.getByText("Recovered unsaved changes")).toBeInTheDocument();
    expect(mocks.updateNote).not.toHaveBeenCalled();
    await advance(700);
    expect(mocks.updateNote).toHaveBeenCalledWith("first", {
      title: "Protocol",
      body_markdown: "Recovered unsaved copy",
      pinned: false,
      editor_mode: "markdown",
      expected_revision: 4,
    });
    expect(window.localStorage.length).toBe(0);
  });

  it("keeps recovered writing open for review when the server revision advanced", async () => {
    const first = note("first", { title: "Protocol", body_markdown: "Server changed", revision: 5 });
    const { api, mocks } = apiFor([first]);
    writeNoteDraftRecovery(window.localStorage, "researcher", {
      note_id: "first",
      title: "Protocol",
      body_markdown: "Recovered revision four",
      pinned: false,
      editor_mode: "markdown",
      expected_revision: 4,
    });

    render(<NotesPage apiClient={api} recoveryScope="researcher" />);
    await advance();
    await advance();

    expect(screen.getByLabelText("Note body")).toHaveValue("Recovered revision four");
    expect(screen.getByRole("alert")).toHaveTextContent("Your edits are safe in this editor");
    await advance(1000);
    expect(mocks.updateNote).not.toHaveBeenCalled();
  });

  it("acknowledges an old-revision recovery when its content already matches the server", async () => {
    const first = note("first", {
      title: "Protocol",
      body_markdown: "Already committed",
      revision: 5,
    });
    const { api, mocks } = apiFor([first]);
    writeNoteDraftRecovery(window.localStorage, "researcher", {
      note_id: "first",
      title: "Protocol",
      body_markdown: "Already committed",
      pinned: false,
      editor_mode: "markdown",
      expected_revision: 4,
    });

    render(<NotesPage apiClient={api} recoveryScope="researcher" />);
    await advance();
    await advance();

    expect(screen.getByLabelText("Note body")).toHaveValue("Already committed");
    expect(screen.queryByText(/changed elsewhere/i)).not.toBeInTheDocument();
    expect(mocks.updateNote).not.toHaveBeenCalled();
    expect(window.localStorage.length).toBe(0);
  });

  it("turns recovered edits into a new draft when the original Note was deleted", async () => {
    const first = note("first", { title: "Deleted protocol" });
    const { api, mocks } = apiFor([first]);
    writeNoteDraftRecovery(window.localStorage, "researcher", {
      note_id: "first",
      title: "Recovered protocol",
      body_markdown: "Exact offline edits",
      pinned: true,
      editor_mode: "markdown",
      expected_revision: 1,
    });
    mocks.getNote.mockRejectedValue(
      new ApiError("not found", 404, { error: "not found" })
    );

    render(<NotesPage apiClient={api} recoveryScope="researcher" />);
    await advance();
    await advance();

    expect(screen.getByLabelText("Note body")).toHaveValue("Exact offline edits");
    expect(screen.getByText("Recovered edits are ready as a new note.")).toBeInTheDocument();
    expect(screen.queryByText("Deleted protocol")).not.toBeInTheDocument();
    await advance(700);
    expect(mocks.createNote).toHaveBeenCalledWith(
      {
        title: "Recovered protocol",
        body_markdown: "Exact offline edits",
        pinned: true,
        editor_mode: "markdown",
      },
      expect.stringMatching(/^note-create:/)
    );
  });

  it("uses the explicit recent endpoint to open the newest content instead of a pinned browse row", async () => {
    const pinnedOld = note("old", {
      title: "Pinned old",
      pinned: true,
      content_updated_at: "2026-08-01T00:00:00Z",
    });
    const newest = note("newest", {
      title: "Newest content",
      content_updated_at: "2026-08-27T00:00:00Z",
    });
    const { api, mocks } = apiFor([pinnedOld, newest]);
    mocks.listNotes.mockImplementation(async (options?: { sort?: string }) => ({
      notes: (options?.sort === "recent" ? [newest] : [pinnedOld, newest]).map((record) => ({
        note_id: record.note_id,
        title: record.title,
        snippet: record.body_markdown,
        pinned: record.pinned,
        revision: record.revision,
        content_updated_at: record.content_updated_at,
        updated_at: record.updated_at,
      })),
      total_count: 2,
    }));

    render(<NotesPage apiClient={api} />);
    await advance();
    await advance();

    expect(mocks.listNotes).toHaveBeenCalledWith({ sort: "recent", limit: 1, offset: 0 });
    expect(mocks.getNote).toHaveBeenCalledWith("newest");
    expect(screen.getByLabelText("Note body")).toHaveValue("Body newest");
  });

  it("does not let a delayed recent-note lookup override an explicit row choice", async () => {
    const recent = note("recent", { title: "Recent note" });
    const chosen = note("chosen", { title: "Chosen note" });
    const { api, mocks } = apiFor([recent, chosen]);
    let resolveRecent: ((value: { notes: NoteRecord[]; total_count: number }) => void) | null = null;
    let resolveChosen: ((value: NoteRecord) => void) | null = null;
    mocks.listNotes.mockImplementation(
      (options?: { sort?: string }) =>
        options?.sort === "recent"
          ? new Promise((resolve) => {
              resolveRecent = resolve;
            })
          : Promise.resolve({
              notes: [recent, chosen].map((record) => ({
                note_id: record.note_id,
                title: record.title,
                snippet: record.body_markdown,
                pinned: record.pinned,
                revision: record.revision,
                updated_at: record.updated_at,
              })),
              total_count: 2,
            })
    );
    mocks.getNote.mockImplementation(
      (noteId: string) =>
        noteId === "chosen"
          ? new Promise((resolve) => {
              resolveChosen = resolve;
            })
          : Promise.resolve(recent)
    );
    render(<NotesPage apiClient={api} />);
    await advance();
    await advance();

    fireEvent.click(screen.getByRole("button", { name: /Chosen note/ }));
    await advance();
    expect(mocks.getNote).toHaveBeenCalledWith("chosen");

    await act(async () => {
      resolveRecent?.({ notes: [recent], total_count: 2 });
      await Promise.resolve();
    });
    expect(mocks.getNote).not.toHaveBeenCalledWith("recent");

    await act(async () => {
      resolveChosen?.(chosen);
      await Promise.resolve();
    });
    expect(screen.getByLabelText("Note body")).toHaveValue("Body chosen");
  });

  it("keeps server relevance order flat and paginates an active search", async () => {
    const browse = note("browse", { title: "Browse note" });
    const firstResult = note("result_b", { title: "Second alphabetically" });
    const secondResult = note("result_a", { title: "First alphabetically" });
    const thirdResult = note("result_c", { title: "Third result" });
    const { api, mocks } = apiFor([browse, firstResult, secondResult, thirdResult]);
    mocks.listNotes.mockImplementation(async (options?: { query?: string; offset?: number; sort?: string }) => {
      const rows = options?.query
        ? options.offset === 4
          ? [thirdResult]
          : options.offset
            ? [secondResult, thirdResult]
            : [firstResult, secondResult]
        : options?.sort === "recent"
          ? [browse]
          : [browse];
      return {
        notes: rows.map((record) => ({
          note_id: record.note_id,
          title: record.title,
          snippet: record.body_markdown,
          pinned: record.pinned,
          revision: record.revision,
          content_updated_at: record.content_updated_at,
          updated_at: record.updated_at,
        })),
        total_count: options?.query ? 5 : 1,
      };
    });
    const { container } = render(<NotesPage apiClient={api} />);
    await advance();
    await advance();

    fireEvent.change(screen.getByRole("searchbox", { name: "Search notes" }), {
      target: { value: "protocol" },
    });
    await advance(180);
    expect(
      [...container.querySelectorAll(".notes-row-title")].map((element) => element.textContent)
    ).toEqual(["Second alphabetically", "First alphabetically"]);
    expect(container.querySelector(".notes-group-label")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    await advance();
    expect(
      [...container.querySelectorAll(".notes-row-title")].map((element) => element.textContent)
    ).toEqual(["Second alphabetically", "First alphabetically", "Third result"]);
    expect(mocks.listNotes).toHaveBeenLastCalledWith({
      query: "protocol",
      limit: 50,
      offset: 2,
    });
    expect(screen.getByRole("button", { name: "Load more" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    await advance();
    expect(
      [...container.querySelectorAll(".notes-row-title")].map((element) => element.textContent)
    ).toEqual(["Second alphabetically", "First alphabetically", "Third result"]);
    expect(mocks.listNotes).toHaveBeenLastCalledWith({
      query: "protocol",
      limit: 50,
      offset: 4,
    });
    expect(screen.queryByRole("button", { name: "Load more" })).not.toBeInTheDocument();
  });

  it("never shows browse or prior-query rows as matches after the active search fails", async () => {
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
    const browse = note("browse", { title: "Browse result" });
    const alpha = note("alpha", { title: "Alpha match" });
    const { api, mocks } = apiFor([browse, alpha]);
    mocks.listNotes.mockImplementation(async (options?: { query?: string }) => {
      if (options?.query === "broken" || options?.query === "beta") {
        throw new Error(`Search failed for ${options.query}`);
      }
      const rows = options?.query === "alpha" ? [alpha] : [browse];
      return {
        notes: rows.map((record) => ({
          note_id: record.note_id,
          title: record.title,
          snippet: record.body_markdown,
          pinned: record.pinned,
          revision: record.revision,
          updated_at: record.updated_at,
        })),
        total_count: rows.length,
      };
    });
    render(<NotesPage apiClient={api} />);
    await advance();
    expect(screen.getByText("Browse result")).toBeInTheDocument();

    const search = screen.getByRole("searchbox", { name: "Search notes" });
    fireEvent.change(search, { target: { value: "broken" } });
    expect(screen.queryByText("Browse result")).not.toBeInTheDocument();
    await advance(180);
    expect(screen.getByRole("alert")).toHaveTextContent("Search failed for broken");
    expect(screen.queryByText("Browse result")).not.toBeInTheDocument();

    fireEvent.change(search, { target: { value: "alpha" } });
    await advance(180);
    expect(screen.getByText("Alpha match")).toBeInTheDocument();
    fireEvent.change(search, { target: { value: "beta" } });
    expect(screen.queryByText("Alpha match")).not.toBeInTheDocument();
    await advance(180);
    expect(screen.getByRole("alert")).toHaveTextContent("Search failed for beta");
    expect(screen.queryByText("Alpha match")).not.toBeInTheDocument();
  });

  it("decrements the consumed browse offset when deleting a loaded row", async () => {
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
    const first = note("first", { title: "First loaded" });
    const second = note("second", { title: "Second loaded" });
    const third = note("third", { title: "Third page" });
    const { api, mocks } = apiFor([first, second, third]);
    let serverRows = [first, second, third];
    mocks.listNotes.mockImplementation(async (options?: { offset?: number }) => {
      const offset = options?.offset ?? 0;
      const rows = offset === 0 ? serverRows.slice(0, 2) : serverRows.slice(offset, offset + 2);
      return {
        notes: rows.map((record) => ({
          note_id: record.note_id,
          title: record.title,
          snippet: record.body_markdown,
          pinned: record.pinned,
          revision: record.revision,
          updated_at: record.updated_at,
        })),
        total_count: serverRows.length,
      };
    });
    mocks.deleteNote.mockImplementation(async (noteId: string) => {
      serverRows = serverRows.filter((record) => record.note_id !== noteId);
    });
    render(<NotesPage apiClient={api} />);
    await advance();

    fireEvent.click(screen.getByRole("button", { name: /First loaded/ }));
    await advance();
    fireEvent.pointerDown(screen.getByRole("button", { name: "More note actions" }), {
      button: 0,
      ctrlKey: false,
    });
    await advance();
    fireEvent.click(screen.getByRole("menuitem", { name: "Delete note" }));
    await advance();
    fireEvent.click(screen.getByRole("button", { name: "Delete note" }));
    await advance();

    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    await advance();
    expect(mocks.listNotes).toHaveBeenLastCalledWith({
      query: undefined,
      limit: 50,
      offset: 1,
    });
    expect(screen.getByText("Third page")).toBeInTheDocument();
  });

  it.each([
    ["lost transport response", new TypeError("response lost")],
    [
      "malformed successful response",
      new Error(
        "Ultra received an incomplete Note delete receipt. The result is uncertain; retry the exact request before continuing."
      ),
    ],
  ])("treats retry 404 as successful deletion reconciliation after a %s", async (_label, firstError) => {
    const existing = note("existing", { title: "Delete me" });
    const { api, mocks } = apiFor([existing]);
    mocks.deleteNote
      .mockRejectedValueOnce(firstError)
      .mockRejectedValueOnce(new ApiError("not found", 404, { error: "not found" }));
    render(<NotesPage apiClient={api} />);
    await advance();
    await advance();

    for (let attempt = 0; attempt < 2; attempt += 1) {
      fireEvent.pointerDown(screen.getByRole("button", { name: "More note actions" }), {
        button: 0,
        ctrlKey: false,
      });
      await advance();
      fireEvent.click(screen.getByRole("menuitem", { name: "Delete note" }));
      await advance();
      fireEvent.click(screen.getByRole("button", { name: "Delete note" }));
      await advance();
    }

    expect(mocks.deleteNote).toHaveBeenNthCalledWith(1, "existing");
    expect(mocks.deleteNote).toHaveBeenNthCalledWith(2, "existing");
    expect(screen.getByText("Select a note or start a new one.")).toBeInTheDocument();
    expect(screen.queryByText("Delete me")).not.toBeInTheDocument();
  });

  it("treats a first delete 404 as successful concurrent deletion", async () => {
    const existing = note("existing", { title: "Already gone" });
    const { api, mocks } = apiFor([existing]);
    mocks.deleteNote.mockRejectedValueOnce(
      new ApiError("not found", 404, { error: "not found" })
    );
    render(<NotesPage apiClient={api} />);
    await advance();
    await advance();

    fireEvent.pointerDown(screen.getByRole("button", { name: "More note actions" }), {
      button: 0,
      ctrlKey: false,
    });
    await advance();
    fireEvent.click(screen.getByRole("menuitem", { name: "Delete note" }));
    await advance();
    fireEvent.click(screen.getByRole("button", { name: "Delete note" }));
    await advance();

    expect(mocks.deleteNote).toHaveBeenCalledOnce();
    expect(screen.getByText("Select a note or start a new one.")).toBeInTheDocument();
    expect(screen.queryByText("Already gone")).not.toBeInTheDocument();
  });

  it("restarts browse pagination after unpinning moves a row across the boundary", async () => {
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
    const pinned = note("pinned", {
      title: "Pinned old",
      pinned: true,
      content_updated_at: "2026-08-01T00:00:00Z",
    });
    const second = note("second", {
      title: "Newest unpinned",
      content_updated_at: "2026-08-27T00:00:00Z",
    });
    const third = note("third", {
      title: "Middle unpinned",
      content_updated_at: "2026-08-20T00:00:00Z",
    });
    const { api, mocks } = apiFor([pinned, second, third]);
    let serverRows = [pinned, second, third];
    mocks.listNotes.mockImplementation(async (options?: { offset?: number }) => {
      const offset = options?.offset ?? 0;
      const rows = serverRows.slice(offset, offset + 2);
      return {
        notes: rows.map((record) => ({
          note_id: record.note_id,
          title: record.title,
          snippet: record.body_markdown,
          pinned: record.pinned,
          revision: record.revision,
          content_updated_at: record.content_updated_at,
          updated_at: record.updated_at,
        })),
        total_count: serverRows.length,
      };
    });
    mocks.updateNote.mockImplementation(async (noteId: string, payload: NoteWritePayload) => {
      const updated = note(noteId, {
        title: payload.title,
        body_markdown: payload.body_markdown,
        pinned: payload.pinned,
        editor_mode: payload.editor_mode,
        revision: (payload.expected_revision ?? 0) + 1,
        content_updated_at: pinned.content_updated_at,
      });
      serverRows = [second, third, updated];
      return updated;
    });
    render(<NotesPage apiClient={api} />);
    await advance();

    fireEvent.click(screen.getByRole("button", { name: /Pinned, Pinned old/ }));
    await advance();
    fireEvent.click(screen.getByRole("button", { name: "Unpin note" }));
    await advance();
    expect(mocks.listNotes).toHaveBeenLastCalledWith({
      query: undefined,
      limit: 50,
      offset: 0,
    });

    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    await advance();
    expect(mocks.listNotes).toHaveBeenLastCalledWith({
      query: undefined,
      limit: 50,
      offset: 2,
    });
    expect(screen.getByText("Pinned old")).toBeInTheDocument();
    expect(screen.getByText("Middle unpinned")).toBeInTheDocument();
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

  it("reconciles an uncertain create before discarding so a committed server Note is deleted", async () => {
    const { api, mocks } = apiFor();
    mocks.createNote
      .mockRejectedValueOnce(new TypeError("response lost"))
      .mockResolvedValueOnce(note("created", { title: "", body_markdown: "Temporary finding" }));
    render(<NotesPage apiClient={api} recoveryScope="researcher" />);
    await advance();
    const body = await startDraft();
    fireEvent.change(body, { target: { value: "Temporary finding" } });
    await advance(700);

    fireEvent.pointerDown(screen.getByRole("button", { name: "More note actions" }), {
      button: 0,
      ctrlKey: false,
    });
    await advance();
    fireEvent.click(screen.getByRole("menuitem", { name: "Discard draft" }));
    await advance();
    fireEvent.click(screen.getByRole("button", { name: "Discard draft" }));
    await advance();

    expect(mocks.createNote).toHaveBeenCalledTimes(2);
    expect(mocks.createNote.mock.calls[1]).toEqual(mocks.createNote.mock.calls[0]);
    expect(mocks.deleteNote).toHaveBeenCalledWith("created");
    expect(screen.getByText("Select a note or start a new one.")).toBeInTheDocument();
  });

  it("retains an uncertain local draft when discard reconciliation is still offline", async () => {
    const { api, mocks } = apiFor();
    mocks.createNote.mockRejectedValue(new TypeError("offline"));
    render(<NotesPage apiClient={api} recoveryScope="researcher" />);
    await advance();
    const body = await startDraft();
    fireEvent.change(body, { target: { value: "Keep this exact draft" } });
    await advance(700);

    fireEvent.pointerDown(screen.getByRole("button", { name: "More note actions" }), {
      button: 0,
      ctrlKey: false,
    });
    await advance();
    fireEvent.click(screen.getByRole("menuitem", { name: "Discard draft" }));
    await advance();
    fireEvent.click(screen.getByRole("button", { name: "Discard draft" }));
    await advance();

    expect(mocks.createNote).toHaveBeenCalledTimes(2);
    expect(mocks.deleteNote).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Note body")).toHaveValue("Keep this exact draft");
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Couldn’t confirm whether this draft was already created"
    );
    expect(window.localStorage.length).toBeGreaterThan(0);
  });

  it("discards a local draft immediately after a deterministic create rejection", async () => {
    const { api, mocks } = apiFor();
    mocks.createNote.mockRejectedValue(
      new ApiError("title is too long", 400, { error: "title is too long" })
    );
    render(<NotesPage apiClient={api} recoveryScope="researcher" />);
    await advance();
    const body = await startDraft();
    fireEvent.change(body, { target: { value: "Rejected draft" } });
    await advance(700);

    fireEvent.pointerDown(screen.getByRole("button", { name: "More note actions" }), {
      button: 0,
      ctrlKey: false,
    });
    await advance();
    fireEvent.click(screen.getByRole("menuitem", { name: "Discard draft" }));
    await advance();
    fireEvent.click(screen.getByRole("button", { name: "Discard draft" }));
    await advance();

    expect(mocks.createNote).toHaveBeenCalledTimes(1);
    expect(mocks.deleteNote).not.toHaveBeenCalled();
    expect(screen.getByText("Select a note or start a new one.")).toBeInTheDocument();
    expect(window.localStorage.length).toBe(0);
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

  it("does not consume a captured New note until a blocked draft saves, then retries once", async () => {
    const first = note("first", { title: "First note", body_markdown: "First body" });
    const { api, mocks } = apiFor([first]);
    mocks.updateNote
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(
        note("first", {
          title: "First note",
          body_markdown: "Unsynced current draft",
          revision: 2,
        })
      );
    const onInitialDraftConsumed = vi.fn();
    const view = render(
      <NotesPage apiClient={api} onInitialDraftConsumed={onInitialDraftConsumed} />
    );
    await advance();
    await advance();
    fireEvent.change(screen.getByLabelText("Note body"), {
      target: { value: "Unsynced current draft" },
    });

    view.rerender(
      <NotesPage
        apiClient={api}
        initialDraft={{ key: "capture_1", bodyMarkdown: "Exact captured text" }}
        onInitialDraftConsumed={onInitialDraftConsumed}
      />
    );
    await advance();
    expect(mocks.updateNote).toHaveBeenCalledTimes(1);
    expect(onInitialDraftConsumed).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Note body")).toHaveValue("Unsynced current draft");
    await advance(1000);
    expect(mocks.updateNote).toHaveBeenCalledTimes(1);

    fireEvent.pointerDown(screen.getByRole("button", { name: /Save status: Couldn’t sync/ }), {
      button: 0,
      ctrlKey: false,
    });
    await advance();
    fireEvent.click(screen.getByRole("menuitem", { name: "Retry sync" }));
    await advance();
    await advance();

    expect(mocks.updateNote).toHaveBeenCalledTimes(2);
    expect(onInitialDraftConsumed).toHaveBeenCalledWith("capture_1");
    expect(screen.getByLabelText("Note body")).toHaveValue("Exact captured text");
  });

  it("creates a captured New note after one pause without another keystroke", async () => {
    const { api, mocks } = apiFor();
    const onInitialDraftConsumed = vi.fn();
    render(
      <NotesPage
        apiClient={api}
        initialDraft={{ key: "capture_1", bodyMarkdown: "Exact captured text" }}
        onInitialDraftConsumed={onInitialDraftConsumed}
      />
    );
    await advance();

    expect(screen.getByLabelText("Note body")).toHaveValue("Exact captured text");
    expect(onInitialDraftConsumed).toHaveBeenCalledWith("capture_1");
    expect(mocks.createNote).not.toHaveBeenCalled();
    await advance(700);

    expect(mocks.createNote).toHaveBeenCalledTimes(1);
    expect(mocks.createNote.mock.calls[0][0]).toMatchObject({
      body_markdown: "Exact captured text",
    });
  });

  it("keeps a captured New note local after one failed automatic create", async () => {
    const { api, mocks } = apiFor();
    mocks.createNote.mockRejectedValue(new TypeError("offline"));
    render(
      <NotesPage
        apiClient={api}
        recoveryScope="researcher"
        initialDraft={{ key: "capture_1", bodyMarkdown: "Keep this capture" }}
      />
    );
    await advance(700);
    await advance(2_000);

    expect(mocks.createNote).toHaveBeenCalledTimes(1);
    expect(screen.getByLabelText("Note body")).toHaveValue("Keep this capture");
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

  it("reconciles a lost PATCH response when the latest server content matches exactly", async () => {
    const first = note("first", { title: "Protocol", body_markdown: "Original" });
    const committed = note("first", {
      title: "Protocol",
      body_markdown: "Committed exact edit",
      revision: 2,
      content_digest: "committed",
    });
    const { api, mocks } = apiFor([first]);
    mocks.updateNote
      .mockRejectedValueOnce(new TypeError("response lost"))
      .mockRejectedValueOnce(
        new ApiError("note revision conflict", 409, { code: "note_revision_conflict" })
      );
    mocks.getNote.mockResolvedValueOnce(first).mockResolvedValueOnce(committed);
    render(<NotesPage apiClient={api} recoveryScope="researcher" />);
    await advance();
    await advance();

    const body = screen.getByLabelText("Note body");
    fireEvent.change(body, { target: { value: "Committed exact edit" } });
    await advance(700);
    expect(screen.getByText("Couldn’t sync")).toBeInTheDocument();

    fireEvent.blur(body);
    await advance();
    expect(mocks.updateNote).toHaveBeenCalledTimes(2);
    expect(screen.queryByText(/changed elsewhere/i)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Save status: Saved/ })).toBeInTheDocument();
    expect(window.localStorage.length).toBe(0);
  });

  it("saves edits as a new Note when the original is deleted during autosave", async () => {
    const first = note("first", { title: "Protocol", body_markdown: "Original" });
    const { api, mocks } = apiFor([first]);
    mocks.updateNote.mockRejectedValueOnce(
      new ApiError("not found", 404, { error: "not found" })
    );
    render(<NotesPage apiClient={api} recoveryScope="researcher" />);
    await advance();
    await advance();

    fireEvent.change(screen.getByLabelText("Note body"), {
      target: { value: "Edits after another tab deleted it" },
    });
    await advance(700);

    expect(mocks.updateNote).toHaveBeenCalledOnce();
    expect(mocks.createNote).toHaveBeenCalledWith(
      {
        title: "Protocol",
        body_markdown: "Edits after another tab deleted it",
        pinned: false,
        editor_mode: "markdown",
      },
      expect.stringMatching(/^note-create:/)
    );
    expect(screen.getByText("Recovered edits are ready as a new note.")).toBeInTheDocument();
    expect(screen.getByLabelText("Note body")).toHaveValue(
      "Edits after another tab deleted it"
    );
  });

  it("flushes the freshest local draft before account departure", async () => {
    const first = note("first", { title: "Protocol", body_markdown: "Original" });
    const { api, mocks } = apiFor([first]);
    let flushForLogout: (() => Promise<boolean>) | null = null;
    render(
      <NotesPage
        apiClient={api}
        recoveryScope="researcher"
        onLogoutFlushReady={(flush) => {
          flushForLogout = flush;
        }}
      />
    );
    await advance();
    await advance();

    fireEvent.change(screen.getByLabelText("Note body"), {
      target: { value: "Typed immediately before sign-out" },
    });
    expect(mocks.updateNote).not.toHaveBeenCalled();
    let confirmed = false;
    await act(async () => {
      confirmed = (await flushForLogout?.()) ?? false;
    });

    expect(confirmed).toBe(true);
    expect(mocks.updateNote).toHaveBeenCalledWith("first", {
      title: "Protocol",
      body_markdown: "Typed immediately before sign-out",
      pinned: false,
      editor_mode: "markdown",
      expected_revision: 1,
    });
    expect(window.localStorage.length).toBe(0);
  });

  it("hands an in-flight unmount save to account departure after switching panels", async () => {
    const first = note("first", { title: "Protocol", body_markdown: "Original" });
    const { api, mocks } = apiFor([first]);
    let resolveUpdate: ((record: NoteRecord) => void) | null = null;
    mocks.updateNote.mockImplementation(
      () =>
        new Promise<NoteRecord>((resolve) => {
          resolveUpdate = resolve;
        })
    );
    let flushHandoff: (() => Promise<boolean>) | null = null;
    const view = render(
      <NotesPage
        apiClient={api}
        recoveryScope="researcher"
        onLogoutFlushReady={(flush) => {
          // Mirrors App: a lazy-panel cleanup must not erase the last writer.
          if (flush) flushHandoff = flush;
        }}
      />
    );
    await advance();
    await advance();
    fireEvent.change(screen.getByLabelText("Note body"), {
      target: { value: "Typed, then switched to Chat" },
    });

    view.unmount();
    expect(mocks.updateNote).toHaveBeenCalledOnce();
    let departureSettled = false;
    const departure = flushHandoff!().then((confirmed) => {
      departureSettled = true;
      return confirmed;
    });
    await advance();
    expect(departureSettled).toBe(false);

    await act(async () => {
      resolveUpdate?.(
        note("first", {
          title: "Protocol",
          body_markdown: "Typed, then switched to Chat",
          revision: 2,
        })
      );
      await Promise.resolve();
    });
    expect(await departure).toBe(true);
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
