// URL-as-navigation-state for the app shell. The app has no router; navigation is
// React state in App.tsx. These pure helpers reflect "which panel + which Lens
// resource" in the query string so the browser Back/Forward buttons work, the view
// survives a refresh, and a Lens view is a shareable/deep-linkable URL.
//
// Design notes:
//  - Query params (not paths) keep it consistent with the existing ?conversation=<id>
//    sync and need no server-side SPA-fallback config.
//  - `view` encodes the panel ("resources" | "admin" | "training" | "lens"); chat is
//    the absence of `view`. `resource` carries the Lens file id(s).
//  - buildNavUrl PRESERVES every other param (notably conversation), so this layer and
//    the conversation-URL layer never clobber each other — each only owns its own keys.

import { parseUltraResourceRef } from "./ultraResource";

// Mirrors App.tsx's ActivePanel union (kept local to avoid importing the 14.8k-line App).
export type NavPanel = "chat" | "resources" | "notes" | "admin" | "training" | "scientific-viewer";

export type NavState = {
  panel: NavPanel;
  // Lens resource file id(s); only meaningful when panel === "scientific-viewer".
  resourceFileIds: string[];
  // Open Resources collection (folder); only meaningful when panel === "resources".
  // In the URL so Back leaves a folder the way it was entered and a refresh
  // reopens the same folder instead of dropping to the collection root.
  resourceCollectionId: string | null;
};

const PANEL_TO_VIEW: Record<NavPanel, string | null> = {
  chat: null,
  resources: "resources",
  notes: "notes",
  admin: "admin",
  training: "training",
  "scientific-viewer": "lens",
};

const VIEW_TO_PANEL: Record<string, NavPanel> = {
  resources: "resources",
  notes: "notes",
  admin: "admin",
  training: "training",
  lens: "scientific-viewer",
};

const VIEW_PARAM = "view";
const RESOURCE_PARAM = "resource";
const COLLECTION_PARAM = "collection";

// The most file ids one Lens view opens. A chat turn can cite many files; Lens
// opens the first few and ignores the rest rather than firing an unbounded
// fan-out of catalog lookups from one click or one URL.
export const LENS_MAX_FILE_IDS = 24;

// The ONE normalization every Lens id list goes through: trimmed, empties
// dropped, deduped (first occurrence wins), capped at LENS_MAX_FILE_IDS.
// Parsing a URL, building a URL, building a deep link, keying a nav state for
// the write-dedupe, and the opener in App.tsx all apply this same function, so
// a URL and the state it restores always agree on which ids they mean. If they
// could disagree (a duplicate, a 25th id), restoring a URL would produce a
// "different" state, which would push a new history entry, and Back would land
// on the same URL again — forever.
export function normalizeLensFileIds(ids: readonly string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of ids) {
    const id = raw.trim();
    if (id.length === 0 || seen.has(id)) {
      continue;
    }
    seen.add(id);
    out.push(id);
    if (out.length === LENS_MAX_FILE_IDS) {
      break;
    }
  }
  return out;
}

// Build the relative URL (pathname + search + hash) for a nav state, preserving all
// other query params present on `current`.
export const buildNavUrl = (
  current: { pathname: string; search: string; hash: string },
  nav: NavState,
): string => {
  const params = new URLSearchParams(current.search);
  const view = PANEL_TO_VIEW[nav.panel];
  if (view) {
    params.set(VIEW_PARAM, view);
  } else {
    params.delete(VIEW_PARAM);
  }
  const fileIds = nav.panel === "scientific-viewer" ? normalizeLensFileIds(nav.resourceFileIds) : [];
  if (fileIds.length > 0) {
    params.set(RESOURCE_PARAM, fileIds.join(","));
  } else {
    params.delete(RESOURCE_PARAM);
  }
  const collectionId = nav.panel === "resources" ? (nav.resourceCollectionId ?? "").trim() : "";
  if (collectionId) {
    params.set(COLLECTION_PARAM, collectionId);
  } else {
    params.delete(COLLECTION_PARAM);
  }
  const search = params.toString();
  return `${current.pathname}${search ? `?${search}` : ""}${current.hash}`;
};

// Parse the nav state out of a query string.
export const parseNavFromSearch = (search: string): NavState => {
  const params = new URLSearchParams(search);
  const view = (params.get(VIEW_PARAM) ?? "").trim().toLowerCase();
  const panel = VIEW_TO_PANEL[view] ?? "chat";
  const resourceRaw = (params.get(RESOURCE_PARAM) ?? "").trim();
  const resourceFileIds =
    panel === "scientific-viewer" && resourceRaw ? normalizeLensFileIds(resourceRaw.split(",")) : [];
  const collectionRaw = (params.get(COLLECTION_PARAM) ?? "").trim();
  const resourceCollectionId = panel === "resources" && collectionRaw ? collectionRaw : null;
  return { panel, resourceFileIds, resourceCollectionId };
};

// A stable identity for a nav state, for deduping URL writes (the resource list only
// matters in Lens).
export const navStateKey = (nav: NavState): string => {
  if (nav.panel === "scientific-viewer") {
    return `scientific-viewer|${[...nav.resourceFileIds].join(",")}`;
  }
  if (nav.panel === "resources") {
    return `resources|${nav.resourceCollectionId ?? ""}`;
  }
  return nav.panel;
};

// --- Lens deep links ---------------------------------------------------------
// The chat agent cites Ultra resources with a link that must open OUR Lens for
// that exact file. The wire shape is shared with the Go control plane
// (runResourceHit.lens_url) and the Python tools: "/?view=lens&resource=<id>",
// ids comma-joined in one param, optionally prefixed by the public origin.

// A catalog file id: opaque, URL-safe, bounded. Anything else is not a file id
// and is never turned into a navigation target.
const LENS_FILE_ID_PATTERN = /^[A-Za-z0-9_.:-]{1,128}$/;
const ULTRA_RESOURCE_SCHEME = "ultra://resource/";
const BISQUE_PATH_PATTERN = /\/(client_service\/view|data_service\/|image_service\/)/i;

// lensDeepLinkFor builds the relative Lens deep link for one or more file ids.
// Each id is percent-encoded individually; the separator stays a literal comma
// so the link reads the same as what buildNavUrl writes for the same state.
export function lensDeepLinkFor(fileIds: string | readonly string[]): string {
  const ids = normalizeLensFileIds(typeof fileIds === "string" ? [fileIds] : fileIds);
  const query = new URLSearchParams({ [VIEW_PARAM]: PANEL_TO_VIEW["scientific-viewer"] as string });
  if (ids.length === 0) {
    return `/?${query.toString()}`;
  }
  return `/?${query.toString()}&${RESOURCE_PARAM}=${ids.map((id) => encodeURIComponent(id)).join(",")}`;
}

export type ResolvedLensLink = { fileIds: string[]; href: string };

const resolveFromIds = (candidates: string[]): ResolvedLensLink | null => {
  const fileIds = normalizeLensFileIds(candidates);
  if (fileIds.length === 0 || !fileIds.every((id) => LENS_FILE_ID_PATTERN.test(id))) {
    return null;
  }
  return { fileIds, href: lensDeepLinkFor(fileIds) };
};

const parseLensHrefAsUrl = (href: string, origin: string): URL | null => {
  if (href.startsWith("/") || href.startsWith("?")) {
    // A relative form: "/?view=lens&resource=…" or "?view=lens&resource=…".
    // Protocol-relative "//host/…" is a foreign origin in disguise.
    if (href.startsWith("//")) {
      return null;
    }
    try {
      return new URL(href, origin);
    } catch {
      return null;
    }
  }
  let parsed: URL;
  try {
    parsed = new URL(href);
  } catch {
    return null;
  }
  let expected: URL;
  try {
    expected = new URL(origin);
  } catch {
    return null;
  }
  return parsed.origin === expected.origin ? parsed : null;
};

// resolveLensLink decides whether a markdown href is a link into our Lens and,
// if so, which file ids it opens. Accepts the relative form, the same-origin
// absolute form, and the portable "ultra://resource/<id>[/<name>]" reference.
// Returns null for anything else — foreign origins, BisQue viewer/data/image
// URLs, missing or malformed ids — so those hrefs keep their ordinary rendering.
export function resolveLensLink(href: string, origin: string): ResolvedLensLink | null {
  const candidate = href.trim();
  if (!candidate) {
    return null;
  }
  if (candidate.startsWith(ULTRA_RESOURCE_SCHEME)) {
    const ref = parseUltraResourceRef(candidate);
    return ref ? resolveFromIds([ref.fileId]) : null;
  }
  if (BISQUE_PATH_PATTERN.test(candidate)) {
    return null;
  }
  const url = parseLensHrefAsUrl(candidate, origin);
  if (!url) {
    return null;
  }
  // The deep link lives at the app root; any other path is not a Lens link.
  if (url.pathname !== "/" && url.pathname !== "") {
    return null;
  }
  const nav = parseNavFromSearch(url.search);
  if (nav.panel !== "scientific-viewer" || nav.resourceFileIds.length === 0) {
    return null;
  }
  return resolveFromIds(nav.resourceFileIds);
}
