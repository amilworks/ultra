// A tiny module-level store for the in-app figure lightbox. Any chat component
// can open it by importing openFigureLightbox(...) — no context threading through
// the (very large) App tree. A single <FigureLightboxRoot/> subscribes and renders
// the overlay; its "Open in Lens" button reads App's one registered opener from
// lib/lensNavigation, the same registry the chat pills use.

export type LightboxFigure = {
  url: string;
  downloadUrl?: string;
  title: string;
  /** An alternate version (e.g. the un-annotated original) to add to the set. */
  originalUrl?: string;
  originalTitle?: string;
  /** When set, the figure can be opened in the full Lens scientific viewer. */
  fileId?: string | null;
};

export type FigureLightboxState = { figures: LightboxFigure[]; index: number } | null;

let state: FigureLightboxState = null;
const listeners = new Set<() => void>();

const emit = () => {
  for (const listener of listeners) {
    listener();
  }
};

// openFigureLightbox shows the lightbox for a set of figures starting at `index`.
// Figures without a url are dropped; opening an empty set is a no-op.
export function openFigureLightbox(figures: LightboxFigure[], index = 0): void {
  const valid = figures.filter((figure) => figure && typeof figure.url === "string" && figure.url.length > 0);
  if (valid.length === 0) {
    return;
  }
  const start = valid.findIndex((figure) => figure.url === figures[index]?.url);
  state = { figures: valid, index: start >= 0 ? start : Math.max(0, Math.min(index, valid.length - 1)) };
  emit();
}

export function closeFigureLightbox(): void {
  if (state === null) {
    return;
  }
  state = null;
  emit();
}

export function subscribeFigureLightbox(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getFigureLightboxState(): FigureLightboxState {
  return state;
}
