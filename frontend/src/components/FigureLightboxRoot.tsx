import { Suspense, lazy, useSyncExternalStore } from "react";

import {
  closeFigureLightbox,
  getFigureLightboxState,
  subscribeFigureLightbox,
} from "@/lib/figureLightbox";

const LazyFigureLightboxOverlay = lazy(() =>
  import("./FigureLightboxOverlay").then((module) => ({ default: module.FigureLightboxOverlay }))
);

// Mounted once near the app root. Subscribes to the module store and renders the
// (lazy) overlay only while a figure set is open, so the heavy viewer code never
// loads until a figure is actually opened.
export function FigureLightboxRoot() {
  const state = useSyncExternalStore(subscribeFigureLightbox, getFigureLightboxState, () => null);
  if (!state) {
    return null;
  }
  return (
    <Suspense fallback={null}>
      <LazyFigureLightboxOverlay
        key={state.figures.map((figure) => figure.url).join("|")}
        figures={state.figures}
        initialIndex={state.index}
        onClose={closeFigureLightbox}
      />
    </Suspense>
  );
}
