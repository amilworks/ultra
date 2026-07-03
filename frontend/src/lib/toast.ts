import type * as Sonner from "sonner";

export type SonnerModule = typeof Sonner;
export type ToastSuccessOptions = Parameters<SonnerModule["toast"]["success"]>[1];
export type ToastErrorOptions = Parameters<SonnerModule["toast"]["error"]>[1];

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
