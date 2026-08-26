// A tiny module-level registry for "open these files in Lens". Chat markdown is
// rendered through module-level components (overrides passed via props never
// reach the streaming surface, and the block memo comparator would freeze any
// closure), so the pill cannot receive a handler through React. App registers
// its opener once; the pill reads it at click time, never at render time.

export type LensOpener = (fileIds: string[]) => void;

let opener: LensOpener | null = null;

export function registerLensOpener(next: LensOpener | null): void {
  opener = next;
}

export function getLensOpener(): LensOpener | null {
  return opener;
}
