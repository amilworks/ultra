import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const typesSource = readFileSync(path.join(process.cwd(), "src/types.ts"), "utf8");
const mockApiSource = readFileSync(
  path.join(process.cwd(), "scripts/mock-api.mjs"),
  "utf8"
);

const blockFrom = (start: string, end: string): string => {
  const startIndex = appSource.indexOf(start);
  expect(startIndex, `missing block: ${start}`).toBeGreaterThan(-1);
  const endIndex = appSource.indexOf(end, startIndex);
  expect(endIndex, `unterminated block: ${start}`).toBeGreaterThan(startIndex);
  return appSource.slice(startIndex, endIndex + end.length);
};

describe("model Notes release gates", () => {
  it("treats missing or failed public config as disabled", () => {
    expect(appSource).toContain(
      "const [modelNotesReadEnabled, setModelNotesReadEnabled] = useState(false)"
    );
    expect(appSource).toContain(
      "setModelNotesReadEnabled(payload.features?.model_notes_read === true)"
    );
    expect(typesSource).toMatch(/model_notes_read\?: boolean/);
  });

  it("hides both browser entry points until the flag is explicitly true", () => {
    expect(appSource).toMatch(
      /onUseInChat=\{\s*modelNotesReadEnabled \? attachNoteToActiveConversation : undefined\s*\}/
    );
    expect(appSource).toMatch(
      /onOpenNotes=\{\s*modelNotesReadEnabled\s*\? \(\) => setComposerNotePickerOpen\(true\)\s*: undefined\s*\}/
    );
    expect(appSource).toMatch(
      /\{modelNotesReadEnabled && composerNotePickerOpen \? \(\s*<Suspense fallback=\{null\}>\s*<LazyNoteContextPicker/
    );
    expect(appSource).toContain(
      'const loadNoteContextPickerModule = () => import("./components/chat/NoteContextPicker")'
    );
    const attachMenu = blockFrom("function ComposerAttachMenu", "export function App()");
    expect(attachMenu).toMatch(/onOpenNotes\?: \(\) => void/);
    expect(attachMenu).toMatch(/\{onOpenNotes \? \(/);
  });

  it("keeps the browser mock opted in for visual and interaction QA", () => {
    expect(mockApiSource.match(/features: \{ model_notes_read: true \}/g)).toHaveLength(2);
  });

  it("loads render-heavy Notes chat surfaces only when they are needed", () => {
    expect(appSource).toContain(
      'const loadNoteRunContextModule = () => import("./components/chat/NoteRunContext")'
    );
    expect(appSource).toMatch(
      /!isStreamingAssistant && showNoteRunContext \? \(\s*<Suspense fallback=\{null\}>\s*<LazyNoteRunContext/
    );
    expect(appSource).toMatch(
      /toolName === "read_note" \|\| toolName === "propose_note_append"/
    );
  });

  it("fails mixed Notes/analysis turns before reads, imports, uploads, or runs", () => {
    const submit = blockFrom(
      "const handleSubmit = async",
      "// Keep a stable handle to the latest handleSubmit"
    );
    const preflight = submit.indexOf("notesTurnHasUnsupportedAnalysisContext({");
    expect(preflight).toBeGreaterThan(-1);
    expect(submit).toContain("NOTES_TEXT_ONLY_GUIDANCE");
    expect(submit).toContain("pendingFileCount: conversation.pendingFiles.length");
    expect(submit).toContain("activeUploadCount: conversation.stagedUploadFileIds.length");
    expect(submit).toContain("selectionContext: conversation.activeSelectionContext");
    expect(submit).toContain("workflowSelected: Boolean(composerWorkflowPreset)");
    expect(submit).toContain("externalResourceCount: bisqueUrls.length");
    expect(preflight).toBeLessThan(submit.indexOf("await resealTurnNotes"));
    expect(preflight).toBeLessThan(submit.indexOf("apiClient.importBisqueResources"));
    expect(preflight).toBeLessThan(submit.indexOf("await uploadPendingFiles"));
    expect(preflight).toBeLessThan(submit.indexOf("apiClient.chatStream"));
    expect(submit).toMatch(/if \(bisqueUrls\.length === 0 && !requestedNoteAccess\)/);
    expect(submit).toMatch(
      /if \(\s*!requestedNoteAccess &&\s*shouldInferBisqueToolsForTurn\(/
    );
  });

  it("serializes append-proposal authority as a typed Note scope field", () => {
    expect(typesSource).toMatch(/allow_append_proposal: boolean/);
    const submit = blockFrom(
      "const handleSubmit = async",
      "// Keep a stable handle to the latest handleSubmit"
    );
    expect(submit).toMatch(
      /noteAccessForTurn\(\s*text,\s*selectedNotesForTurn,\s*excludedNoteIntentTextForTurn,\s*noteSearchScopeOverrideForTurn\s*\)/
    );
  });

  it("queues every follow-up when the active run is already Note-scoped", () => {
    const steer = blockFrom("const steerFollowup = useCallback", ".steerRun(runId");
    const noteGate = steer.indexOf("assistantRunOriginatedWithNotes(");
    const queued = steer.indexOf("queueFollowup();", noteGate);
    expect(noteGate).toBeGreaterThan(-1);
    expect(queued).toBeGreaterThan(noteGate);
    expect(queued).toBeLessThan(steer.indexOf(".steerRun(runId"));
  });
});
