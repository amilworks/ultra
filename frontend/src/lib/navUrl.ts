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
  const fileIds = nav.panel === "scientific-viewer" ? nav.resourceFileIds.filter((id) => id.trim().length > 0) : [];
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
    panel === "scientific-viewer" && resourceRaw
      ? resourceRaw
          .split(",")
          .map((id) => id.trim())
          .filter((id) => id.length > 0)
      : [];
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
