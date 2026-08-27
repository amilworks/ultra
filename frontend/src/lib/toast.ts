import type * as Sonner from "sonner";

export type SonnerModule = typeof Sonner;
export type ToastSuccessOptions = Parameters<SonnerModule["toast"]["success"]>[1];
export type ToastErrorOptions = Parameters<SonnerModule["toast"]["error"]>[1];
export type ToastOptions = Parameters<SonnerModule["toast"]>[1];

let sonnerModulePromise: Promise<SonnerModule> | null = null;

export const loadSonnerModule = (): Promise<SonnerModule> => {
  sonnerModulePromise ??= import("sonner").catch((error: unknown) => {
    sonnerModulePromise = null;
    throw error;
  });
  return sonnerModulePromise;
};

export const showSuccessToast = (
  message: string,
  options?: ToastSuccessOptions
): void => {
  void loadSonnerModule().then(({ toast }) => {
    toast.success(message, options);
  });
};

export const showErrorToast = (
  message: string,
  options?: ToastErrorOptions
): void => {
  void loadSonnerModule().then(({ toast }) => {
    toast.error(message, options);
  });
};

export const showActionToast = (message: string, options?: ToastOptions): void => {
  void loadSonnerModule().then(({ toast }) => {
    toast(message, options);
  });
};

export const dismissToast = (id: string | number | undefined): void => {
  if (id === undefined) return;
  void loadSonnerModule().then(({ toast }) => toast.dismiss(id));
};

export const dismissAllToasts = (): void => {
  void loadSonnerModule().then(({ toast }) => toast.dismiss());
};

/**
 * A neutral toast carrying a single reversal.
 *
 * Deliberately not `toast.success`: removing a message is not a success, it is a
 * change the user may want back. Success styling on a destructive action reads
 * as congratulation. Eight seconds is long enough to notice and act on without
 * the toast becoming furniture.
 */
export const showUndoToast = (
  message: string,
  onUndo: () => void,
  options?: { description?: string; durationMs?: number }
): void => {
  void loadSonnerModule().then(({ toast }) => {
    toast(message, {
      description: options?.description,
      duration: options?.durationMs ?? 8000,
      action: { label: "Undo", onClick: onUndo },
    });
  });
};
