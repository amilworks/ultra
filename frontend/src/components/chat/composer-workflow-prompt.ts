import type {
  ComposerWorkflowDefinition,
  ComposerWorkflowPresetState,
} from "./composer-workflows";

const normalizeWorkflowToken = (value: string): string =>
  value
    .trim()
    .toLowerCase()
    .replace(/^\/+/, "")
    .replace(/\s+/g, " ");

const escapeRegExp = (value: string): string =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const cleanWorkflowTrailingText = (value: string): string =>
  value.replace(/^[\s:.,;/-]+/, "").trim();

export const slashWorkflowSearchQuery = (value: string): string => {
  const prompt = String(value || "").trimStart();
  if (!prompt.startsWith("/")) {
    return "";
  }
  const remainder = prompt.slice(1).trimStart();
  return remainder.split(/\s+/)[0] ?? "";
};

export const visiblePromptAfterComposerWorkflowSelection = (
  workflow: ComposerWorkflowDefinition | ComposerWorkflowPresetState,
  currentPrompt: string
): string => {
  const prompt = String(currentPrompt || "").trimStart();
  if (!prompt.startsWith("/")) {
    return prompt.trim();
  }
  const remainder = prompt.slice(1).trimStart();
  if (!remainder) {
    return "";
  }
  const commandCandidates = [
    workflow.label,
    workflow.id,
    workflow.id.replace(/[_-]+/g, " "),
    ...("keywords" in workflow ? workflow.keywords : []),
  ]
    .map((candidate) => normalizeWorkflowToken(candidate))
    .filter(Boolean)
    .sort((a, b) => b.length - a.length);

  for (const candidate of commandCandidates) {
    const match = new RegExp(`^${escapeRegExp(candidate)}(?:\\b|\\s|$)`, "i").exec(
      normalizeWorkflowToken(remainder)
    );
    if (!match) {
      continue;
    }
    return cleanWorkflowTrailingText(remainder.slice(match[0].length));
  }

  const [, ...rest] = remainder.split(/\s+/);
  return cleanWorkflowTrailingText(rest.join(" "));
};

export const composeComposerWorkflowPromptForModel = (
  workflow: ComposerWorkflowPresetState | null | undefined,
  userPrompt: string
): string => {
  const prompt = String(userPrompt || "").trim();
  const scaffold = String(workflow?.prompt ?? "").trimEnd();
  if (!scaffold) {
    return prompt;
  }
  if (!prompt) {
    return scaffold;
  }
  const joiner = /[:：]\s*$/.test(scaffold) ? " " : "\n\n";
  return `${scaffold}${joiner}${prompt}`;
};

export type ComposerWorkflowTurnContract = {
  modelPrompt: string;
  selectedToolNames: string[];
  workflowHint: ComposerWorkflowPresetState["workflowHint"];
};

/**
 * Resolves the workflow-owned portion of a chat turn's wire contract.
 *
 * Notes access is a sealed, Notes-only surface. A persistent visual preset such
 * as Pro Mode may remain selected for the next turn, but no persisted scaffold,
 * tool selection, or workflow hint is allowed to alter the current Notes turn.
 */
export const composerWorkflowTurnContract = (
  workflow: ComposerWorkflowPresetState | null | undefined,
  userPrompt: string,
  notesAccessRequested: boolean,
  selectedToolNamesForTurn: readonly string[] = []
): ComposerWorkflowTurnContract => {
  if (notesAccessRequested) {
    return {
      modelPrompt: String(userPrompt || "").trim(),
      selectedToolNames: [],
      workflowHint: null,
    };
  }
  return {
    modelPrompt: composeComposerWorkflowPromptForModel(workflow, userPrompt),
    selectedToolNames: Array.from(
      new Set([...(workflow?.selectedToolNames ?? []), ...selectedToolNamesForTurn])
    ),
    workflowHint: workflow?.workflowHint ?? null,
  };
};

export const composerWorkflowPresetAfterSubmit = (
  workflow: ComposerWorkflowPresetState | null | undefined
): ComposerWorkflowPresetState | null =>
  workflow?.persistsAcrossTurns ? workflow : null;
