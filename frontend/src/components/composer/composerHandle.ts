import type { BriefFileToken } from "@/features/chat/brief-tokens";

/* The contract between the app and whatever edits the composer's text. Both
   editors — the ProseMirror one and the plain-textarea fallback that covers the
   first paint — implement it, so the app never knows which is mounted. */

export type ComposerFileToken = BriefFileToken;

export type ComposerFocusOptions = {
  preventScroll?: boolean;
  /** "end" puts the caret after the last character; "keep" leaves it alone. */
  caret?: "end" | "keep";
};

export type ComposerMention = {
  /** What was typed after the "@", possibly nothing yet. */
  query: string;
};

export type ComposerCaretRect = { left: number; top: number; bottom: number };

export type ComposerTokenDetails = {
  /** Tooltip: the real file name, kind and size. */
  title: string;
  /** Short mono kind chip (GZ, TIF, CSV…). */
  kind: string;
  /** The file left the library since the token was written. */
  gone?: boolean;
};

export interface ComposerHandle {
  /** The editable element — for checks like `element === document.activeElement`. */
  readonly element: HTMLElement | null;
  readonly disabled: boolean;
  /** The text the run would receive, tokens serialised as `@label`. */
  readonly value: string;
  focus(options?: ComposerFocusOptions): void;
  isFocused(): boolean;
  /** Replace the whole text. Tokens whose `@label` survives keep their pills. */
  setValue(text: string): void;
  /** Insert at the caret (or append when nothing is focused). */
  insertText(text: string): void;
  /** Replace the active `@query` run with a token; append one when no run is active. */
  acceptMention(token: ComposerFileToken): void;
  /** A file that arrived by another path — drop, upload, library picker. */
  appendToken(token: ComposerFileToken): void;
  removeToken(fileId: string): void;
  /** Turn a token back into a bare "@" so the picker reopens in its place. */
  reopenMentionFor(fileId: string): void;
  /** Viewport rect of the active mention's "@", for anchoring the picker. */
  mentionRect(): ComposerCaretRect | null;
}

export type ComposerEditorProps = {
  value: string;
  /** Known tokens for this conversation — how `@label` text becomes a pill. */
  tokens: readonly ComposerFileToken[];
  goneFileIds: readonly string[];
  disabled: boolean;
  placeholder: string;
  ariaLabel: string;
  mentionOpen: boolean;
  listboxId?: string;
  activeOptionId?: string;
  tokenDetails: (fileId: string) => ComposerTokenDetails | null;
  onValueChange: (text: string) => void;
  /** The tokens present in the text, in order, whenever that set changes. */
  onTokensChange: (tokens: ComposerFileToken[]) => void;
  onMentionChange: (mention: ComposerMention | null) => void;
  onFocusChange: (focused: boolean) => void;
  /** First look at every key. Return true (after preventDefault) to consume it. */
  onKeyDown?: (event: KeyboardEvent) => boolean;
  /** Plain Enter and ⌘/Ctrl-Enter. Shift-Enter never reaches this: it is a new line. */
  onEnter: (event: KeyboardEvent) => boolean;
  /** Return true when the paste was consumed (files, data-shaped text). */
  onPaste?: (event: ClipboardEvent) => boolean;
  onTokenRemoveClick?: (fileId: string) => void;
  /** The editor is mounted and can take focus. */
  onReady?: () => void;
};
