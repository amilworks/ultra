type ActivePanel = "chat" | "resources" | "notes" | "admin" | "training" | "scientific-viewer";

export const MISSING_REQUESTED_CONVERSATION_MESSAGE =
  "Requested chat was not found. Opened the latest available conversation instead.";

export const shouldShowAppShellBanner = (
  activePanel: ActivePanel,
  message: string | null
): boolean => {
  if (!message) {
    return false;
  }
  return activePanel === "chat" || message !== MISSING_REQUESTED_CONVERSATION_MESSAGE;
};
