import type { NoteAccessContext, SelectedNoteReference, SelectionContext } from "@/types";

const MAX_NOTE_INTENT_EXCLUSIONS = 20;
export const NOTES_INTENT_EXCLUSIONS_OVERFLOW =
  "__ULTRA_NOTES_INTENT_EXCLUSIONS_OVERFLOW_V1__";

/**
 * Keeps exact paste provenance bounded. Once more fragments exist than can be
 * represented safely, retain a content-free overflow marker instead of
 * dropping the oldest authority-bearing paste.
 */
export const boundedNoteIntentExclusions = (
  values: readonly unknown[]
): string[] => {
  if (values.some((value) => typeof value !== "string")) {
    return [NOTES_INTENT_EXCLUSIONS_OVERFLOW];
  }
  const normalized = values.filter(
    (value): value is string => typeof value === "string" && value.length > 0
  );
  if (
    normalized.includes(NOTES_INTENT_EXCLUSIONS_OVERFLOW) ||
    normalized.length > MAX_NOTE_INTENT_EXCLUSIONS
  ) {
    return [NOTES_INTENT_EXCLUSIONS_OVERFLOW];
  }
  return normalized;
};

const NOTES_REFERENCE =
  /\b(?:my\s+(?:notes?|notebook|lab\s+(?:notes?|log|notebook))|ultra\s+notes?|(?:other|related|all)\s+notes?|notes?\s+(?:app|page|library)|(?:in|from|to)\s+(?:my\s+)?notes?)\b/i;
// Natural Note references usually put the title between "my" and "note"
// ("find my Field Protocol note"). Keep the span deliberately short so an
// unrelated later use of the word "note" cannot accidentally grant access.
const TITLED_NOTE_REFERENCE =
  /\bmy\s+(?:(?:[\p{L}\p{N}][\p{L}\p{N}._+#'’/-]*)\s+){0,6}notes?\b/iu;
const OWN_NOTE_REFERENCE = /\b(?:(?:the|a|my)\s+)?note\s+i\s+(?:wrote|saved|made|created)\b/i;
const SELECTED_NOTE_REFERENCE =
  /\b(?:(?:the|my)\s+)?(?:attached|selected)\s+notes?\b|\bthis\s+note\b/i;
const NOTES_RETRIEVAL_ACTION =
  /\b(?:search|find|look\s+(?:in|through|up|for)|read|check|scan|use|review|open|remember|recall)\b/i;
const NOTES_DISCOVERY_ACTION =
  /\b(?:search|find|look\s+(?:in|through|up|for)|remember|recall)\b/i;
const RELATED_NOTES_REFERENCE = /\b(?:other|related|another|all)\s+notes?\b/i;
const NOTES_MUTATION_ACTION = /\b(?:add|append|save|update|write|record|jot)\b/i;
const NEGATED_MUTATION_CLAUSE =
  /\b(?:do\s+not|don't|dont|never|without|avoid(?:ing)?)\b[^.!?;]{0,120}\b(?:add|append|save|update|write|record|jot)\b|\bnot\b[^.!?;]{0,40}\b(?:add|append|save|update|write|record|jot)\b/i;
const UNCERTAIN_MUTATION_CLAUSE =
  /\b(?:explain\s+how|how\s+(?:do|can|could|would|should)|should\s+(?:ultra|you|i|we)|whether\s+(?:i|we)\s+should|why\s+(?:would|should)\s+(?:i|we)|what\s+if|if\s+(?:i|we)|did(?:\s+not|n't)\s+ask)\b[^.!?;]{0,120}\b(?:add|append|save|update|write|record|jot)\b/i;
const POLITE_MUTATION_QUESTION =
  /\b(?:can|could|would)\s+(?:ultra|you)\b[^.!?;]{0,80}\b(?:add|append|save|update|write|record|jot)\b/i;
const CONCRETE_MUTATION_CONTENT =
  /\b(?:this|that|these|those|it|today(?:'s)?|the\s+(?:following|result|results|summary|finding|findings|text|context|answer|analysis|observation|observations|measurement|measurements|protocol|link))\b/i;
// “Search Notes” is an unambiguous product-level request even without the
// possessive “my”. Keep this direct form plural so generic prose such as
// “review the note below” does not unlock account-wide search.
const DIRECT_NOTES_REQUEST =
  /\b(?:search|find|look\s+(?:in|through|for)|read|check|scan|review|open)\s+(?:the\s+)?notes\b/i;
const WRITING_RECALL = /\b(?:what|where|when)\s+did\s+i\s+(?:write|note|jot|save)\b/i;
const NEGATED_SEARCH =
  /\b(?:do\s+not|don't|dont|never|without)\s+(?:searching|search|reading|read|checking|check|using|use|looking\s+(?:in|through)|adding|add|appending|append|saving|save|updating|update|writing|write|recording|record|jotting|jot)(?:\s+[\p{L}\p{N}'’/-]+){0,5}\s+(?:my\s+(?:notes?|notebook|lab\s+(?:notes?|log|notebook))|notes?)\b/iu;
const INSTRUCTIONAL_ONLY =
  /\b(?:how\s+(?:do|can|would|should)\s+i|can\s+you\s+(?:explain|show|tell)\s+me\s+how\s+to)\s+(?:search|find|use|read|add|append|save|update|write|record|jot)(?:\s+[\p{L}\p{N}'’/-]+){0,5}\s+(?:my\s+(?:notes?|notebook|lab\s+(?:notes?|log|notebook))|notes?)\b/iu;

const blankLike = (value: string): string => value.replace(/[^\n]/g, " ");

const stripDelimitedQuotes = (value: string): string => {
  const pairs: Record<string, string> = { '"': '"', "'": "'", "“": "”", "‘": "’", "«": "»" };
  const characters = Array.from(value);
  for (let index = 0; index < characters.length; index += 1) {
    const opener = characters[index];
    const closer = pairs[opener];
    if (!closer) continue;
    // An ASCII apostrophe inside a word is punctuation, not a quoted payload.
    if (opener === "'" && /[\p{L}\p{N}]/u.test(characters[index - 1] ?? "")) continue;
    let end = index + 1;
    for (; end < characters.length; end += 1) {
      if (characters[end] !== closer || characters[end - 1] === "\\") continue;
      if (closer === "'" && /[\p{L}\p{N}]/u.test(characters[end + 1] ?? "")) continue;
      break;
    }
    if (end >= characters.length) {
      for (let cursor = index; cursor < characters.length; cursor += 1) {
        if (characters[cursor] !== "\n") characters[cursor] = " ";
      }
      break;
    }
    for (let cursor = index; cursor <= end; cursor += 1) {
      if (characters[cursor] !== "\n") characters[cursor] = " ";
    }
    index = end;
  }
  return characters.join("");
};

/**
 * Notes scope is authority, so instructions presented as reference data can
 * never mint it. Preserve line boundaries while blanking Markdown/HTML code,
 * blockquotes, quoted spans, and exact fragments the composer observed being
 * pasted. The remaining prose is the only text eligible to request access.
 */
export const notesAuthorityText = (
  text: string,
  excludedReferenceTexts: readonly string[] = []
): string => {
  const boundedExclusions = boundedNoteIntentExclusions(excludedReferenceTexts);
  if (boundedExclusions.includes(NOTES_INTENT_EXCLUSIONS_OVERFLOW)) {
    return "";
  }
  let authority = text.replace(/\r\n?/g, "\n");
  for (const rawFragment of boundedExclusions) {
    const fragment = String(rawFragment || "").replace(/\r\n?/g, "\n");
    if (!fragment) continue;
    const index = authority.indexOf(fragment);
    if (index >= 0) {
      authority =
        authority.slice(0, index) + blankLike(fragment) + authority.slice(index + fragment.length);
    }
  }

  authority = authority
    .replace(/<\s*(?:blockquote|pre|code)\b[^>]*>[\s\S]*?<\s*\/\s*(?:blockquote|pre|code)\s*>/gi, (match) => blankLike(match));

  const lines = authority.split("\n");
  let fence: "```" | "~~~" | null = null;
  authority = lines
    .map((line) => {
      const trimmed = line.trimStart();
      if (fence) {
        if (trimmed.startsWith(fence)) fence = null;
        return " ".repeat(line.length);
      }
      if (trimmed.startsWith("```") || trimmed.startsWith("~~~")) {
        fence = trimmed.startsWith("```") ? "```" : "~~~";
        return " ".repeat(line.length);
      }
      if (/^\s*>/.test(line) || /^(?: {4}|\t)/.test(line)) {
        return " ".repeat(line.length);
      }
      return line;
    })
    .join("\n")
    .replace(/(`+)[\s\S]*?\1/g, (match) => blankLike(match));

  return stripDelimitedQuotes(authority).replace(/\s+/g, " ").trim();
};

/**
 * Grants broad Notes search only for an explicit request in the current user
 * turn. Ordinary phrases such as "note that" never opt a run into Notes.
 */
export const notesSearchRequested = (
  text: string,
  excludedReferenceTexts: readonly string[] = []
): boolean => {
  const normalized = notesAuthorityText(text, excludedReferenceTexts);
  if (!normalized || NEGATED_SEARCH.test(normalized) || INSTRUCTIONAL_ONLY.test(normalized)) {
    return false;
  }
  return (
    WRITING_RECALL.test(normalized) ||
    DIRECT_NOTES_REQUEST.test(normalized) ||
    ((NOTES_REFERENCE.test(normalized) ||
      TITLED_NOTE_REFERENCE.test(normalized) ||
      OWN_NOTE_REFERENCE.test(normalized)) &&
      (NOTES_RETRIEVAL_ACTION.test(normalized) || NOTES_MUTATION_ACTION.test(normalized)))
  );
};

const firstMatch = (
  value: string,
  pattern: RegExp
): { index: number; length: number } | null => {
  const match = pattern.exec(value);
  return match && match.index >= 0 ? { index: match.index, length: match[0].length } : null;
};

const wordCount = (value: string): number =>
  value.match(/[\p{L}\p{N}]+/gu)?.length ?? 0;

/**
 * Grants only the ability to create an append proposal that the browser still
 * has to review and commit. The authority must be a direct mutation request in
 * this exact user turn; quoted, fenced, blockquoted, or paste-provenance text
 * is blanked before matching. A nearby Notes target is required so ordinary
 * requests such as "add a chart" cannot accidentally expose the proposal tool.
 */
export const noteAppendProposalRequested = (
  text: string,
  excludedReferenceTexts: readonly string[] = []
): boolean => {
  const normalized = notesAuthorityText(text, excludedReferenceTexts);
  if (!normalized || NEGATED_SEARCH.test(normalized) || INSTRUCTIONAL_ONLY.test(normalized)) {
    return false;
  }

  return normalized.split(/[.!?;]+/).some((clause) => {
    if (
      !clause.trim() ||
      NEGATED_MUTATION_CLAUSE.test(clause) ||
      UNCERTAIN_MUTATION_CLAUSE.test(clause) ||
      (POLITE_MUTATION_QUESTION.test(clause) && !CONCRETE_MUTATION_CONTENT.test(clause))
    ) {
      return false;
    }
    const action = firstMatch(clause, NOTES_MUTATION_ACTION);
    if (!action) return false;
    const target = [
      firstMatch(clause, NOTES_REFERENCE),
      firstMatch(clause, TITLED_NOTE_REFERENCE),
      firstMatch(clause, OWN_NOTE_REFERENCE),
      firstMatch(clause, SELECTED_NOTE_REFERENCE),
    ]
      .filter((match): match is { index: number; length: number } => match !== null)
      .sort((left, right) => left.index - right.index)[0];
    if (!target) return false;
    const between =
      action.index < target.index
        ? clause.slice(action.index + action.length, target.index)
        : clause.slice(target.index + target.length, action.index);
    if (wordCount(between) > 10) return false;
    // A target named before the mutation needs an explicit connective:
    // "Find my Field Protocol note and add today's result." When the action
    // comes first, bounded proximity covers "write this to my lab log" and
    // "update my Field Protocol note" without accepting a distant mention.
    return action.index < target.index || /(?:,|\b(?:and|then)\b|[-—])/i.test(between);
  });
};

export const uniqueSelectedNotes = (
  notes: readonly SelectedNoteReference[],
  limit = 8
): SelectedNoteReference[] => {
  const seen = new Set<string>();
  const result: SelectedNoteReference[] = [];
  for (const note of notes) {
    const noteId = String(note.note_id || "").trim();
    if (!noteId || seen.has(noteId)) continue;
    const revision = Number(note.revision);
    seen.add(noteId);
    result.push({
      note_id: noteId,
      ...(Number.isSafeInteger(revision) && revision > 0 ? { revision } : {}),
    });
    if (result.length >= limit) break;
  }
  return result;
};

export const noteAccessForTurn = (
  text: string,
  selectedNotes: readonly SelectedNoteReference[],
  excludedReferenceTexts: readonly string[] = []
): NoteAccessContext | null => {
  const notes = uniqueSelectedNotes(selectedNotes);
  const authorityText = notesAuthorityText(text, excludedReferenceTexts);
  const wantsNotes = notesSearchRequested(authorityText);
  const allowAppendProposal = noteAppendProposalRequested(text, excludedReferenceTexts);
  // An attached Note is already an exact target. "Add this to my notes"
  // therefore stays selected-only; broad authority is needed only when the
  // turn explicitly asks Ultra to discover/read Notes, or no Note was attached.
  if (
    wantsNotes &&
    (notes.length === 0 ||
      NOTES_DISCOVERY_ACTION.test(authorityText) ||
      DIRECT_NOTES_REQUEST.test(authorityText) ||
      RELATED_NOTES_REFERENCE.test(authorityText) ||
      WRITING_RECALL.test(authorityText))
  ) {
    return { mode: "search", notes, allow_append_proposal: allowAppendProposal };
  }
  return notes.length > 0
    ? { mode: "selected", notes, allow_append_proposal: allowAppendProposal }
    : null;
};

export const NOTES_TEXT_ONLY_GUIDANCE =
  "Notes context currently works with chat text alone. Remove files and other analysis selections before sending.";

export const notesTurnHasUnsupportedAnalysisContext = ({
  pendingFileCount = 0,
  activeUploadCount = 0,
  selectionContext = null,
  workflowSelected = false,
  selectedToolNames = [],
  externalResourceCount = 0,
}: {
  pendingFileCount?: number;
  activeUploadCount?: number;
  selectionContext?: SelectionContext | null;
  workflowSelected?: boolean;
  selectedToolNames?: readonly string[];
  externalResourceCount?: number;
}): boolean => {
  const context = withoutNoteAccess(selectionContext);
  const hasArtifactHandles = Object.values(context?.artifact_handles ?? {}).some(
    (handles) => handles.length > 0
  );
  const hasCapabilityBearingSelection = Boolean(
    (context?.focused_file_ids?.length ?? 0) > 0 ||
      (context?.resource_uris?.length ?? 0) > 0 ||
      (context?.dataset_uris?.length ?? 0) > 0 ||
      hasArtifactHandles ||
      (context?.suggested_tool_names?.length ?? 0) > 0
  );
  return (
    pendingFileCount > 0 ||
    activeUploadCount > 0 ||
    externalResourceCount > 0 ||
    workflowSelected ||
    selectedToolNames.length > 0 ||
    hasCapabilityBearingSelection
  );
};

type NoteTurnMessage = {
  id: string;
  role: string;
  content?: string;
  steering?: unknown;
  noteReferences?: readonly SelectedNoteReference[];
  excludedNoteIntentText?: readonly string[];
};

/** Finds the non-steering user turn that owns an assistant run. */
export const assistantRunOriginatedWithNotes = (
  messages: readonly NoteTurnMessage[],
  assistantMessageId: string | null | undefined
): boolean => {
  if (!assistantMessageId) return false;
  const assistantIndex = messages.findIndex((message) => message.id === assistantMessageId);
  if (assistantIndex < 0) return false;
  for (let index = assistantIndex - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "user" || message.steering) continue;
    return (
      noteAccessForTurn(
        message.content ?? "",
        message.noteReferences ?? [],
        message.excludedNoteIntentText ?? []
      ) !== null
    );
  }
  return false;
};

export const withNoteAccess = (
  context: SelectionContext | null,
  noteAccess: NoteAccessContext | null
): SelectionContext | null => {
  if (!noteAccess) return context;
  return { ...(context ?? {}), note_access: noteAccess };
};

export const withoutNoteAccess = (context: SelectionContext | null): SelectionContext | null => {
  if (!context?.note_access) return context;
  const rest: SelectionContext = { ...context };
  delete rest.note_access;
  const hasOtherContext = Object.entries(rest).some(([, value]) => {
    if (Array.isArray(value)) return value.length > 0;
    if (value && typeof value === "object") return Object.keys(value).length > 0;
    return value !== null && value !== undefined && value !== "";
  });
  return hasOtherContext ? rest : null;
};
