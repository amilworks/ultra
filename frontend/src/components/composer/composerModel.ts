import { briefSummary } from "@/features/chat/brief-tokens";

/* The composer's explicit states. Everything the layout does is a function of
   these few values — no CSS attribute cascade decides geometry on its own. */

export type ComposerStage = "rest" | "composing" | "running";
export type ComposerLayout = "desktop" | "phone";

export type ComposerStageInputs = {
  running: boolean;
  focused: boolean;
  hasText: boolean;
  hasTokens: boolean;
  hasFiles: boolean;
  hasWorkflow: boolean;
  menuOpen: boolean;
  welcomeStage: boolean;
};

export const deriveComposerStage = (inputs: ComposerStageInputs): ComposerStage => {
  if (inputs.running) {
    return "running";
  }
  if (
    inputs.focused ||
    inputs.hasText ||
    inputs.hasTokens ||
    inputs.hasFiles ||
    inputs.hasWorkflow ||
    inputs.menuOpen ||
    inputs.welcomeStage
  ) {
    return "composing";
  }
  return "rest";
};

/** The key contract, stated where the keys are: ↵ queues during a run, ⌘↵ steers. */
export const composerKeysHint = (running: boolean): string =>
  running ? "queue for after · ⌘↵ steer" : "send · ⇧↵ new line";

export type ComposerPlaceholderInputs = {
  hydrated: boolean;
  welcomeStage: boolean;
  readMode: boolean;
  running: boolean;
  hasTokens: boolean;
  hasFiles: boolean;
  phone: boolean;
};

export const composerPlaceholder = (inputs: ComposerPlaceholderInputs): string => {
  if (!inputs.hydrated) {
    return "Loading chat…";
  }
  if (inputs.welcomeStage) {
    return "Describe a question, dataset, or experiment…";
  }
  if (inputs.readMode && !inputs.running) {
    return "Just start typing";
  }
  if (inputs.running) {
    return "Steer this run, or queue for after";
  }
  // The grammar cues ride along only where they fit: a phone bar's status box
  // is ~186px beside the mode tag, and a truncated cue reads as a defect. On
  // phones the + menu carries the same two routes.
  if (!inputs.phone && !inputs.hasTokens && !inputs.hasFiles) {
    return "Ask Ultra — @ to bring in a file, / for a workflow";
  }
  return "Ask Ultra";
};

export const composerSummary = briefSummary;

/** The @ picker's desktop width; the anchor is clamped so it never overflows the surface. */
export const COMPOSER_MENTION_WIDTH = 380;

export const clampMentionAnchor = (left: number, hostWidth: number): number =>
  Math.max(0, Math.min(left, Math.max(0, hostWidth - COMPOSER_MENTION_WIDTH)));
