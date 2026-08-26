import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

// Source pins for the Lens deep-link wiring in App.tsx. App is a 14k-line shell that
// cannot be rendered in a unit test, so these pin the load-bearing shapes of the
// wiring: one shared opener, registered for chat pills and the lightbox, reused by
// deep-link restore, and a URL sync that can neither strip a deep link on cold load
// nor push a second history entry for a view it has already written.
const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const navUrlSource = readFileSync(path.join(process.cwd(), "src/lib/navUrl.ts"), "utf8");
const viewerSource = readFileSync(
  path.join(process.cwd(), "src/components/ScientificViewerPage.tsx"),
  "utf8"
);

const sliceBetween = (source: string, start: string, end: string): string => {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  if (from < 0 || to < 0) {
    throw new Error(`marker not found: ${from < 0 ? start : end}`);
  }
  return source.slice(from, to);
};

const opener = sliceBetween(appSource, "const openLensByFileIds = useCallback(", "registerLensOpener((fileIds)");
// The opener's own body, without the Retry / Open-chat helpers declared after it.
const openerBody = sliceBetween(opener, "const openLensByFileIds = useCallback(", "const lensRequestedFileIds");

describe("Lens deep-link opener", () => {
  it("registers one shared opener for chat pills and the figure lightbox", () => {
    expect(appSource).toMatch(/import \{ registerLensOpener \} from "\.\/lib\/lensNavigation";/);
    const effect = sliceBetween(
      appSource,
      "useEffect(() => {\n    registerLensOpener(",
      "}, [openLensByFileIds]);"
    );
    expect(effect).toMatch(/registerLensOpener\(\(fileIds\) => \{\s*void openLensByFileIds\(fileIds\);/);
    expect(effect).toMatch(/registerLightboxOpenInLens\(\(fileId\) => \{\s*void openLensByFileIds\(\[fileId\]\);/);
    expect(effect).toContain("registerLensOpener(null);");
    expect(effect).toContain("registerLightboxOpenInLens(null);");
  });

  it("routes deep links, Back/Forward and the viewer's Retry through the same opener", () => {
    expect(appSource).toMatch(
      /const restoreViewerContextForFileIds = useCallback\(\s*\(fileIds: string\[\]\): Promise<void> => openLensByFileIds\(fileIds\),/
    );
    expect(appSource).toContain("void restoreViewerContextForFileIds(initial.resourceFileIds);");
    expect(appSource).toContain("void restoreViewerContextForFileIds(nav.resourceFileIds);");
    // Retry re-asks for exactly the ids the context recorded as requested.
    expect(opener).toContain("const lensRequestedFileIds = resourceViewerContext?.requestedFileIds;");
    expect(opener).toMatch(/const retryLensOpen = useCallback\(\(\): void => \{\s*if \(lensRequestedFileIds && lensRequestedFileIds\.length > 0\) \{\s*void openLensByFileIds\(lensRequestedFileIds\);/);
    expect(appSource).toContain("onRetry={retryLensOpen}");
  });

  it("normalizes with the URL layer's own function and cap, never a local copy", () => {
    expect(appSource).toMatch(
      /import \{\s*buildNavUrl,\s*LENS_MAX_FILE_IDS,\s*navStateKey,\s*normalizeLensFileIds,\s*parseNavFromSearch,\s*type NavState,\s*\} from "\.\/lib\/navUrl";/
    );
    expect(appSource).not.toContain("LENS_OPEN_MAX_FILES");
    expect(opener).toContain("const ids = normalizeLensFileIds(fileIds);");
    expect(opener).not.toContain("uniqueFileIds(");
    expect(opener).not.toMatch(/\.slice\(0, /);
    expect(opener).toContain("ids.length === LENS_MAX_FILE_IDS && fileIds.length > ids.length");
    expect(opener).toContain("Promise.allSettled(ids.map((id) => apiClient.getResource(id)))");
  });

  it("records the requested ids on every outcome through a single state path", () => {
    expect(opener).toContain(
      "const outcome: LensOpenOutcome = { requestedFileIds: ids, unavailableFileIds, failedFileIds };"
    );
    // Survivors open through the ordinary viewer state path, never a bespoke one...
    expect(opener).toContain(
      "openUploadedFilesInViewer(found.map(resourceToUploadedFile), bisqueLinksByFileId, outcome);"
    );
    // ...and so does a total miss, so the URL sync sees one final state either way.
    expect(opener).toContain("openUploadedFilesInViewer([], {}, outcome);");
    expect(openerBody).not.toContain("setResourceViewerContext(");
    // The shared entry point only admits an empty file set when an outcome rides along.
    const enter = sliceBetween(
      appSource,
      "const openUploadedFilesInViewer = useCallback(",
      "const openConversationFilesInViewer"
    );
    expect(enter).toContain("lensOpen?: LensOpenOutcome");
    expect(enter).toContain("if (selectedFiles.length === 0 && !lensOpen) {");
    expect(enter).toContain("...lensOpen,");
  });

  it("classifies 401 as a session problem, 404/403/410 as unavailable, anything else as failed", () => {
    expect(appSource).toContain("const LENS_UNAVAILABLE_STATUSES: ReadonlySet<number> = new Set([403, 404, 410]);");
    const classify = sliceBetween(opener, "settled.forEach((result, index) => {", "const outcome: LensOpenOutcome");
    const apiErrorBranch = classify.indexOf("if (reason instanceof ApiError) {");
    const unauthorized = classify.indexOf("if (reason.status === 401) {");
    const unavailable = classify.indexOf("if (LENS_UNAVAILABLE_STATUSES.has(reason.status)) {");
    const unavailablePush = classify.indexOf("unavailableFileIds.push(ids[index]);");
    // Two failed pushes: the malformed-record guard inside the fulfilled branch
    // (a 2xx without a usable record) and the rejection fall-through. The
    // fall-through is the LAST one and must sit after the unavailable push.
    const failedPush = classify.lastIndexOf("failedFileIds.push(ids[index]);");
    expect(apiErrorBranch).toBeGreaterThan(-1);
    expect(unauthorized).toBeGreaterThan(apiErrorBranch);
    expect(unavailable).toBeGreaterThan(unauthorized);
    expect(unavailablePush).toBeGreaterThan(unavailable);
    // The fall-through (non-ApiError, other statuses, network) lands in failed.
    expect(failedPush).toBeGreaterThan(unavailablePush);
    expect(classify.match(/failedFileIds\.push/g)).toHaveLength(2);
    // A 401 surfaces as a toast and never as either notice.
    expect(opener).toMatch(/else if \(unauthorized\) \{\s*showErrorToast\(/);
  });

  it("feeds the viewer both miss lists plus Retry and Open chat", () => {
    expect(appSource).toContain("unavailableFileIds={resourceViewerContext?.unavailableFileIds ?? []}");
    expect(appSource).toContain("failedFileIds={resourceViewerContext?.failedFileIds ?? []}");
    expect(appSource).toContain("onOpenChat={openChatPanelFromLens}");
    const openChat = sliceBetween(appSource, "const openChatPanelFromLens = useCallback(", "}, []);");
    expect(openChat).toContain('setActivePanel("chat");');
    expect(openChat).toContain("setResourceViewerContext(null);");
    expect(viewerSource).toContain('title: "This resource isn\'t available"');
    expect(viewerSource).toContain('body: "It may have been removed, or it isn\'t shared with you."');
    expect(viewerSource).toContain('title: "This resource couldn\'t be loaded"');
    expect(viewerSource).toContain('body: "Check your connection and try again."');
    expect(viewerSource).toContain("window.history.length > 1");
  });

  it("never leaves the app for a Lens link", () => {
    // window.location.assign is reserved for auth redirects; every Lens path stays
    // in-app so ?conversation= survives and Back returns to the thread.
    const lensRegion = sliceBetween(
      appSource,
      "const openUploadedFilesInViewer = useCallback(",
      "const stageResourcesForConversation = ("
    );
    expect(lensRegion).not.toContain("window.location.assign");
    expect(lensRegion).not.toContain("window.location.href =");
    expect(lensRegion).not.toContain("window.open(");
  });
});

describe("Lens deep-link cold load", () => {
  const restore = sliceBetween(
    appSource,
    "// One-time restore on load",
    "// State -> URL:"
  );
  const sync = sliceBetween(appSource, "// State -> URL:", "// Back/Forward:");
  const popstate = sliceBetween(appSource, "// Back/Forward:", "const stageResourcesForConversation = (");

  it("pre-arms the URL dedupe key with the deep-linked state before any state write", () => {
    const preArm = restore.indexOf("lastNavKeyRef.current = navStateKey(initial);");
    const panelWrite = restore.indexOf("setActivePanel(initial.panel);");
    expect(preArm).toBeGreaterThan(-1);
    expect(panelWrite).toBeGreaterThan(preArm);
  });

  it("pre-arms both restore paths with ids normalized by the same function the opener uses", () => {
    // Both keys are computed from parseNavFromSearch output...
    expect(appSource).toContain("const initial = initialNavRef.current;");
    expect(appSource).toMatch(/initialNavRef = useRef<NavState>\([\s\S]*?parseNavFromSearch\(window\.location\.search\)/);
    expect(popstate).toMatch(
      /const nav = parseNavFromSearch\(window\.location\.search\);[\s\S]*?lastNavKeyRef\.current = navStateKey\(nav\);/
    );
    // ...and parseNavFromSearch normalizes through normalizeLensFileIds, the same
    // function openLensByFileIds applies, so the pre-armed key always equals the key
    // of the state that open will write (even for duplicate ids or more than the cap).
    const parse = sliceBetween(navUrlSource, "export const parseNavFromSearch", "export const navStateKey");
    expect(parse).toContain("normalizeLensFileIds(resourceRaw.split(\",\"))");
    expect(navUrlSource).toContain("export const LENS_MAX_FILE_IDS = 24;");
  });

  it("gates the URL sync on restored STATE so the same-commit stale panel is never written", () => {
    expect(restore).toContain("setNavRestored(true);");
    expect(sync).toContain("!navRestored ||");
    expect(sync).not.toContain("!navRestoredRef.current");
    expect(sync).toMatch(/\}, \[[^\]]*navRestored\]\);/);
  });

  it("only ever replaces (never pushes) on cold load and pushes afterwards", () => {
    expect(restore).toContain("window.history.replaceState(window.history.state, \"\", canonicalUrl);");
    expect(restore).not.toContain("pushState");
    expect(sync).not.toContain("replaceState");
    expect(sync).toContain('window.history.pushState({}, "", nextUrl);');
  });

  it("dedupes writes by nav key, with no sticky suppress flag", () => {
    expect(sync).toContain("if (key === lastNavKeyRef.current) {");
    expect(sync).toContain("lastNavKeyRef.current = key;");
    expect(appSource).not.toMatch(/suppressNavSync|skipNextNavWrite|suppressUrlWrite/);
  });

  it("skips the intermediate empty Lens state while an open is resolving, without a sticky flag", () => {
    expect(sync).toContain('activePanel === "scientific-viewer" && pendingLensOpenRef.current !== null');
    // Armed before the first await, cleared by the same call after its final state write.
    const arm = opener.indexOf("pendingLensOpenRef.current = ids;");
    const firstAwait = opener.indexOf("await Promise.allSettled");
    const lastWrite = opener.lastIndexOf("openUploadedFilesInViewer([], {}, outcome);");
    const clear = opener.lastIndexOf("pendingLensOpenRef.current = null;");
    expect(arm).toBeGreaterThan(-1);
    expect(firstAwait).toBeGreaterThan(arm);
    expect(lastWrite).toBeGreaterThan(firstAwait);
    expect(clear).toBeGreaterThan(lastWrite);
  });

  it("writes the REQUESTED ids to the URL so re-opening that URL yields the same key", () => {
    // This is the Back-button guarantee: the URL encodes the request, the restore
    // re-issues the request, the context records the request, the key matches.
    const derive = sliceBetween(appSource, "const viewerResourceFileIds = useMemo(() => {", "}, [resourceViewerContext]);");
    const requested = derive.indexOf("const requestedIds = resourceViewerContext?.requestedFileIds ?? [];");
    const returnRequested = derive.indexOf("return requestedIds;");
    const fallback = derive.indexOf("return (resourceViewerContext?.uploadedFiles ?? []).map((file) => file.file_id);");
    expect(requested).toBeGreaterThan(-1);
    expect(returnRequested).toBeGreaterThan(requested);
    expect(fallback).toBeGreaterThan(returnRequested);
    expect(derive).not.toContain("unavailableFileIds");
    expect(sync).toContain("resourceFileIds: viewerResourceFileIds,");
  });
  it("treats a fulfilled lookup without a usable record as a load failure, never a throw", () => {
    // Found in the mock harness: a 2xx without the { resource } envelope made
    // openResourceInViewer's helpers throw inside the opener — an unhandled
    // rejection with no user feedback. The fulfilled branch must guard.
    expect(appSource).toMatch(/const isUsableResourceRecord = \(value: unknown\): value is ResourceRecord =>/);
    expect(appSource).toMatch(
      /if \(result\.status === "fulfilled"\) \{\s*if \(isUsableResourceRecord\(result\.value\)\) \{\s*found\.push\(result\.value\);\s*\} else \{\s*failedFileIds\.push\(ids\[index\]\);/
    );
  });
});
