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

const PERSONAL_NOTES_COLLECTION =
  /\bmy\s+(?:notes?|notebook|lab\s+(?:notes?|log|notebook))\b/i;
const PRODUCT_NOTES_COLLECTION =
  /\bultra\s+notes?\b|\bnotes?\s+(?:app|page|library)\b/i;
const RELATED_NOTES_REFERENCE = /\b(?:other|related|another|all)\s+notes?\b/i;
// Natural Note references usually put the title between "my" and "note"
// ("find my Field Protocol note"). Keep the span deliberately short so an
// unrelated later use of the word "note" cannot accidentally grant access.
const TITLED_NOTE_REFERENCE =
  /\bmy\s+(?:(?:[\p{L}\p{N}][\p{L}\p{N}._+#'’/-]*)\s+){1,6}notes?\b/iu;
const OWN_NOTE_REFERENCE =
  /\b(?:(?:the|a|my)\s+)?note\s+i\s+(?:wrote|saved|made|created)\b/i;
const SELECTED_NOTE_REFERENCE =
  /\b(?:(?:the|my)\s+)?(?:attached|selected)\s+notes?\b|\bthis\s+note\b/i;
// Personal wording is not sufficient when the sentence identifies Notes as
// reference material inside the current prompt, document, or attachment.
// Product-qualified Ultra Notes references remain eligible authority.
const CONTEXTUAL_NOTES_REFERENCE =
  /\b(?:(?:(?:my|a|the)\s+)?notes?\s+(?:below|above|here|attached|provided|included)|(?:(?:my|a|the)\s+)?notes?\s+(?:in|on|from|inside|within)\s+(?:(?:(?:this|that|my|our|uploaded|attached|selected)|(?:a|an|the)(?:\s+(?:uploaded|attached|selected))?)\s+)?(?:pdf|attachment|document|file|email|message|section|slide|page|upload|report)|(?:(?:my|a|the)\s+)?notes?\s+(?:in|on|from|inside|within)\s+[\w.-]+\.(?:pdf|docx?|txt|md)|my\s+(?:meeting|document|pdf|slide|section|email|attachment|file|page)\s+notes?|my\s+notes?\s+from\s+(?:this|that|the|my|our)\s+meeting)\b/i;
const UNIQUE_PERSONAL_NOTE_TARGET =
  /\bmy\s+(?:notebook|lab\s+(?:notes?|log|notebook))\b/i;
const NOTES_RETRIEVAL_ACTION =
  /\b(?:search|find|look\s+(?:in|through|up|for)|read|check|scan|use|review|open|show|list|summarize|explain|compare|answer|tell)\b/i;
const NOTES_DISCOVERY_ACTION =
  /\b(?:search|find|look\s+(?:in|through|up|for))\b/i;
const NOTES_MUTATION_ACTION = /\b(?:add|append|save|update|write|record|jot)\b/i;
const CONCRETE_MUTATION_CONTENT =
  /\b(?:this|that|these|those|it|today(?:'s)?|the\s+(?:following|result|results|summary|finding|findings|text|context|answer|analysis|observation|observations|measurement|measurements|protocol|link))\b/i;
const DIRECT_RETRIEVAL_IMPERATIVE =
  /^(?:(?:please|kindly)\s+)?(search|find|look\s+(?:in|through|up|for)|read|check|scan|use|review|open|show|list|summarize|explain|compare|answer|tell)\b/i;
const DIRECT_RETRIEVAL_REQUEST =
  /^(?:can|could|would|will)\s+you(?:\s+please)?\s+(search|find|look\s+(?:in|through|up|for)|read|check|scan|use|review|open|show|list|summarize|explain|compare|answer|tell)\b/i;
const DIRECT_MUTATION_IMPERATIVE =
  /^(?:(?:please|kindly)\s+)?(add|append|save|update|write|record|jot)\b/i;
const DIRECT_MUTATION_REQUEST =
  /^(?:can|could|would|will)\s+you(?:\s+please)?\s+(add|append|save|update|write|record|jot)\b/i;
const COMPOUND_MUTATION_REQUEST =
  /\b(?:and|then)\s+(?:(?:please|kindly)\s+)?(?:add|append|save|update|write|record|jot)\b/i;
const NEGATED_RETRIEVAL_ACTION =
  /\b(?:do\s+not|don['’]t|dont|never|cannot|can['’]t|without|avoid(?:ing)?)\b[^.!?;]{0,120}\b(?:search(?:ing)?|find(?:ing)?|open(?:ing)?|review(?:ing)?|scan(?:ning)?|read(?:ing)?|check(?:ing)?|use|using|look(?:ing)?\s+(?:in|through|up|for)|show(?:ing)?|list(?:ing)?|summariz(?:e|ing)|explain(?:ing)?|compar(?:e|ing)|answer(?:ing)?|tell(?:ing)?)\b/i;
const NEGATED_MUTATION_ACTION =
  /\b(?:do\s+not|don['’]t|dont|never|cannot|can['’]t|without|avoid(?:ing)?)\b[^.!?;]{0,120}\b(?:add(?:ing)?|append(?:ing)?|save|saving|update|updating|write|writing|record(?:ing)?|jot(?:ting)?)\b/i;
const EXCLUDED_NOTES_TARGET =
  /\b(?:not|except|excluding|but\s+not)\b[^.!?;]{0,60}\b(?:my\s+(?:notes?|notebook|lab\s+(?:notes?|log|notebook))|ultra\s+notes?|notes?\s+(?:app|page|library))\b/i;
const WRITING_RECALL =
  /^(?:what|where|when)\s+did\s+i\s+(?:write|note|jot|save)(?:\s+down)?\b/i;
const PAST_NOTES_RECALL =
  /^(?:did\s+i\s+(?:write|note|jot|save|record)|(?:have|had)\s+i\s+(?:written|noted|jotted|saved|recorded))(?:\s+[\p{L}\p{N}'’/-]+){0,8}\s+(?:in|to)\s+my\s+(?:notes?|notebook|lab\s+(?:notes?|log|notebook))\b/iu;
const RECENT_NOTE_REQUEST =
  /^(?:(?:what|which)\s+(?:is|was)\s+my\s+(?:(?:most\s+)?recent|latest|last|newest)\s+notes?|(?:show|open|read|find|get)\s+(?:me\s+)?my\s+(?:(?:most\s+)?recent|latest|last|newest)\s+notes?)\b/i;
const RECENT_WRITING_RECALL =
  /^(?:(?:(?:did\s+i(?:\s+(?:not|never))?|didn['’]t\s+i)\s+(?:write|save|add|record|jot))(?:\s+[\p{L}\p{N}'’/-]+){0,8}\s+(?:in|to)\s+my\s+(?:(?:most\s+)?recent|latest|last|newest)\s+notes?|what(?:['’]s|\s+is)\s+(?:the\s+)?(?:(?:most\s+)?recent|latest|last|newest)\s+(?:thing|entry|item)\s+i\s+(?:wrote|saved|added|recorded|jotted)(?:\s+[\p{L}\p{N}'’/-]+){0,4}\s+(?:in|to)\s+my\s+(?:notes?|notebook|lab\s+(?:notes?|log|notebook)))\b/iu;
const POSSESSIVE_NOTES_RETRIEVAL =
  /^(?:(?:show(?:\s+me)?|list)\s+my\s+notes?|do\s+i\s+have\s+(?:a|any)\s+notes?|which\s+of\s+my\s+notes?\s+(?:mention|mentions|contain|contains|include|includes|say|says)|what\s+(?:do|does)\s+my\s+notes?\s+(?:mention|mentions|contain|contains|include|includes|say|says)|do\s+my\s+notes?\s+(?:mention|mentions|contain|contains|include|includes|say)|what(?:['’]s|\s+is)\s+in\s+my\s+notes?)\b/i;
const PERSONAL_NOTE_EXISTENCE_RECALL =
  /^do\s+i\s+have\s+(?:a|any)\s+notes?\b/i;
const PERSONAL_NAMED_MUTATION_RECALL =
  /^(?:did|didn['’]t)\s+i\s+(?:add|append|save|update|write|record|jot)\b/i;
const FIRST_PERSON_MUTATION_RECALL =
  /^(?:(?:did|didn['’]t)\s+i|(?:what|where|when)\s+did\s+i)\s+(?:add|append|save|update|write|record|jot)\b/i;
const BARE_WITHDRAWAL =
  /^(?:(?:actually\s+)?no|never\s*mind|scratch\s+that|cancel(?:\s+that)?|forget\s+(?:it|that)|stop|(?:please\s+)?(?:don['’]t|dont|do\s+not)(?:\s+(?:(?:do\s+)?(?:that|it)|anymore))?)$/i;
const INSTRUCTIONAL_FRAME =
  /\b(?:explain|show|tell)\s+(?:me\s+)?how\s+to\b|^how\s+(?:do|can|could|would|should)\b/i;
const REPORTED_REFERENCE_FRAME =
  /^(?:(?:the\s+)?(?:assistant|model|system|prompt|instruction|example|command|source|author|paper|email)\b[^.!?;]{0,100}\b(?:said|says|stated|states|asked|asks|instructed|suggests?|suggested|recommends?|recommended|was|is)\b|(?:i|we|you|they)\s+(?:said|wrote|suggests?|suggested|recommends?|recommended)\b|(?:did|why\s+did)\b[^.!?;]{0,100}\b(?:say|tell|ask|instruct|suggest|recommend)\b)|\b(?:was|is)\s+(?:the|an?)\s+(?:instruction|command|example|prompt|request)\b|\b(?:(?:my|the|the\s+assistant['’]s)\s+)?(?:plan|proposal|idea|intention|request|claim)\b[^.!?;]{0,80}\b(?:to\s+(?:search|find|read|review|use|add|append|save|update|write|record|jot)|that\b[^.!?;]{0,40}\b(?:searched|read|added|appended|saved|updated|wrote|recorded))\b|\b(?:sentence|phrase|wording|text)\b[^.!?;]{0,60}\b(?:search|find|read|add|append|save|update|write|record|jot)\b|\bwhether\b[^.!?;]{0,60}\bshould\b[^.!?;]{0,40}\b(?:search|find|read|review|use|add|append|save|update|write|record|jot)\b/i;
const CONTENT_GROUNDING_PREDICATE =
  /\b(?:about|using|from)\b|\b(?:what|whether)\b[^.!?;]{0,80}\b(?:notes?|notebook)\b[^.!?;]{0,30}\b(?:say|says|said|mention|mentions|contain|contains|include|includes)\b/i;
const STOP_RETRIEVAL_ACTION =
  /\bstop\s+(?:searching|finding|looking|reading|checking|scanning|using|reviewing|opening|showing|listing|summarizing|explaining|comparing|answering|telling)\b/i;
const ONLY_SELECTED_NOTE_RE =
  /\bonly\s+(?:(?:the|my)\s+)?(?:selected|attached)?\s*(?:this\s+)?notes?\b/i;
const NON_NOTES_MUTATION_DESTINATION =
  /\b(?:add|append|save|update|write|record|jot)\b[^.!?;]{0,80}\b(?:to|in|into)\s+(?:(?:this|the|my)\s+)?(?:answer|response|report|email|document|message|slide|file|chart)\b/i;
const TRAILING_GENERIC_WITHDRAWAL =
  /\b(?:and|but|then)\s+(?:please\s+)?(?:don['’]t|dont|do\s+not)\s*$/i;
const LEADING_META_ACTION_FRAME =
  /^(?:explain|review|summarize|compare)\s+(?:this|the|an?)\s+(?:command|instruction|example|sentence|request|prompt)\b[^.!?;,:—–]{0,80}[.!?;,:—–]\s*(?:then\s+)?(?:(?:please|kindly)\s+)?(?:search|find|read|review|use|add|append|save|update|write|record|jot)\b[^.!?;]{0,120}\b(?:notes?|notebook|lab\s+(?:notes?|log))\b/i;

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
    // Exact paste provenance is an authority boundary, not a best-effort
    // cleanup hint. If a recorded fragment was edited or removed, we can no
    // longer prove which nearby words were typed, so inferred authority stays
    // closed until the user chooses the explicit Search Notes override.
    if (index < 0) return "";
    // Value-only provenance cannot identify which copy was pasted. Blanking
    // an arbitrary first match could erase a quoted/reference occurrence and
    // leave the actual pasted instruction live, so duplicates and overlaps
    // fail closed until the explicit browser override is chosen.
    if (authority.indexOf(fragment, index + 1) >= 0) return "";
    authority =
      authority.slice(0, index) + blankLike(fragment) + authority.slice(index + fragment.length);
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

type AuthorityDecision = "grant" | "deny" | null;

type NotesAuthorityEvaluation = {
  search: AuthorityDecision;
  append: AuthorityDecision;
  /** Whether a selected Note is insufficient for the requested retrieval. */
  searchBroadWhenSelected: boolean;
  /** Whether an unselected mutation names a Note that must first be found. */
  appendNeedsDiscovery: boolean;
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

const authorityClauses = (value: string): string[] => {
  const clauses: string[] = [];
  const primary = value.split(
    /[.!?;]+|\s+(?=(?:but|however|instead|actually)\b)|\s+then\s+(?=(?:do\s+not|don['’]t|dont|never|stop)\b)|\s*[—–]\s*/i
  );
  const commaDirective =
    /,\s*(?=(?:but|however|instead|actually|then|do\s+not|don['’]t|dont|never|please|kindly|search|find|look|read|check|scan|use|review|open|show|list|summarize|explain|compare|answer|tell|add|append|save|update|write|record|jot)\b)/i;
  for (const segment of primary) {
    const pieces = commaDirective.test(segment) && !REPORTED_REFERENCE_FRAME.test(segment)
      ? segment.split(commaDirective)
      : [segment];
    for (const piece of pieces) {
      const clause = piece
        .trim()
        .replace(/^(?:(?:but|however|instead|actually|then)\s*,?\s*)+/i, "")
        .trim();
      if (clause) clauses.push(clause);
    }
  }
  return clauses;
};

const hasAccountNotesTarget = (clause: string): boolean => {
  if (PRODUCT_NOTES_COLLECTION.test(clause)) return true;
  if (CONTEXTUAL_NOTES_REFERENCE.test(clause)) return false;
  return (
    PERSONAL_NOTES_COLLECTION.test(clause) ||
    RELATED_NOTES_REFERENCE.test(clause) ||
    (TITLED_NOTE_REFERENCE.test(clause) && !SELECTED_NOTE_REFERENCE.test(clause)) ||
    OWN_NOTE_REFERENCE.test(clause)
  );
};

const hasAnyAppendTarget = (clause: string): boolean =>
  hasAccountNotesTarget(clause) || SELECTED_NOTE_REFERENCE.test(clause);

const appendTargetNeedsDiscovery = (clause: string): boolean =>
  (TITLED_NOTE_REFERENCE.test(clause) && !SELECTED_NOTE_REFERENCE.test(clause)) ||
  OWN_NOTE_REFERENCE.test(clause) ||
  UNIQUE_PERSONAL_NOTE_TARGET.test(clause);

const personalRecallRequest = (clause: string): boolean =>
  (PERSONAL_NOTE_EXISTENCE_RECALL.test(clause) &&
    !CONTEXTUAL_NOTES_REFERENCE.test(clause)) ||
  (hasAccountNotesTarget(clause) &&
    (WRITING_RECALL.test(clause) ||
      PAST_NOTES_RECALL.test(clause) ||
      RECENT_NOTE_REQUEST.test(clause) ||
      RECENT_WRITING_RECALL.test(clause) ||
      POSSESSIVE_NOTES_RETRIEVAL.test(clause)));

const directRetrievalAction = (clause: string): string | null =>
  DIRECT_RETRIEVAL_IMPERATIVE.exec(clause)?.[1] ??
  DIRECT_RETRIEVAL_REQUEST.exec(clause)?.[1] ??
  null;

const directMutationForm = (
  clause: string,
  retrievalAction: string | null
): { polite: boolean } | null => {
  if (DIRECT_MUTATION_IMPERATIVE.test(clause)) return { polite: false };
  if (DIRECT_MUTATION_REQUEST.test(clause)) return { polite: true };
  if (retrievalAction && COMPOUND_MUTATION_REQUEST.test(clause)) {
    return { polite: false };
  }
  return null;
};

const mutationTargetIsNearAction = (clause: string): boolean => {
  const action = firstMatch(clause, NOTES_MUTATION_ACTION);
  if (!action) return false;
  const target = [
    firstMatch(clause, PERSONAL_NOTES_COLLECTION),
    firstMatch(clause, PRODUCT_NOTES_COLLECTION),
    firstMatch(clause, RELATED_NOTES_REFERENCE),
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
  return action.index < target.index || /(?:,|\b(?:and|then)\b|[-—])/i.test(between);
};

const evaluateNotesAuthorityText = (normalized: string): NotesAuthorityEvaluation => {
  const result: NotesAuthorityEvaluation = {
    search: null,
    append: null,
    searchBroadWhenSelected: false,
    appendNeedsDiscovery: false,
  };
  if (LEADING_META_ACTION_FRAME.test(normalized)) return result;

  for (const clause of authorityClauses(normalized)) {
    if (REPORTED_REFERENCE_FRAME.test(clause)) continue;
    if (TRAILING_GENERIC_WITHDRAWAL.test(clause)) {
      result.search = "deny";
      result.append = "deny";
      result.searchBroadWhenSelected = false;
      result.appendNeedsDiscovery = false;
      continue;
    }
    if (BARE_WITHDRAWAL.test(clause)) {
      result.search = "deny";
      result.append = "deny";
      result.searchBroadWhenSelected = false;
      result.appendNeedsDiscovery = false;
      continue;
    }

    const accountTarget = hasAccountNotesTarget(clause);
    const instructional = INSTRUCTIONAL_FRAME.test(clause);
    const searchDenied =
      (EXCLUDED_NOTES_TARGET.test(clause) &&
        (NOTES_RETRIEVAL_ACTION.test(clause) || result.search === "grant")) ||
      ((NEGATED_RETRIEVAL_ACTION.test(clause) || STOP_RETRIEVAL_ACTION.test(clause)) &&
        (accountTarget ||
          result.search === "grant"));
    const appendDenied =
      NEGATED_MUTATION_ACTION.test(clause) &&
      (hasAnyAppendTarget(clause) ||
        result.append === "grant" ||
        /\b(?:it|that|this|anything)\b/i.test(clause));
    if (searchDenied) {
      result.search = "deny";
      result.searchBroadWhenSelected = false;
    }
    if (appendDenied) {
      result.append = "deny";
      result.appendNeedsDiscovery = false;
    }

    if (
      !instructional &&
      !searchDenied &&
      (personalRecallRequest(clause) ||
        (FIRST_PERSON_MUTATION_RECALL.test(clause) &&
          accountTarget))
    ) {
      result.search = "grant";
      result.searchBroadWhenSelected = true;
    } else if (
      !searchDenied &&
      PERSONAL_NAMED_MUTATION_RECALL.test(clause) &&
      accountTarget &&
      (TITLED_NOTE_REFERENCE.test(clause) || UNIQUE_PERSONAL_NOTE_TARGET.test(clause))
    ) {
      // “Did I add this to my Field Protocol note?” is a narrow lookup,
      // never fresh mutation consent. It must discover the named Note even
      // when an unrelated Note happens to be attached.
      result.search = "grant";
      result.searchBroadWhenSelected = true;
    }

    const retrievalAction = directRetrievalAction(clause);
    const selectedOnlyRetrieval =
      Boolean(retrievalAction) &&
      SELECTED_NOTE_REFERENCE.test(clause) &&
      ONLY_SELECTED_NOTE_RE.test(clause);
    if (selectedOnlyRetrieval) {
      result.search = "deny";
      result.searchBroadWhenSelected = false;
    }
    const normalizedRetrievalAction = retrievalAction?.toLocaleLowerCase() ?? "";
    const retrievalActionIsGrounded =
      !/^(?:explain|compare|answer|tell)$/.test(normalizedRetrievalAction) ||
      CONTENT_GROUNDING_PREDICATE.test(clause);
    const bareProductSearch =
      !CONTEXTUAL_NOTES_REFERENCE.test(clause) &&
      (/^(?:(?:please|kindly)\s+)?search\s+notes\b/i.test(clause) ||
        /^(?:can|could|would|will)\s+you(?:\s+please)?\s+search\s+notes\b/i.test(clause));
    if (
      !instructional &&
      !searchDenied &&
      !selectedOnlyRetrieval &&
      retrievalAction &&
      retrievalActionIsGrounded &&
      (accountTarget || bareProductSearch)
    ) {
      result.search = "grant";
      result.searchBroadWhenSelected =
        bareProductSearch ||
        PERSONAL_NOTES_COLLECTION.test(clause) ||
        PRODUCT_NOTES_COLLECTION.test(clause) ||
        RELATED_NOTES_REFERENCE.test(clause) ||
        (TITLED_NOTE_REFERENCE.test(clause) && !SELECTED_NOTE_REFERENCE.test(clause)) ||
        OWN_NOTE_REFERENCE.test(clause) ||
        UNIQUE_PERSONAL_NOTE_TARGET.test(clause) ||
        NOTES_DISCOVERY_ACTION.test(retrievalAction);
    }

    const mutationForm = directMutationForm(clause, retrievalAction);
    if (
      !appendDenied &&
      !instructional &&
      mutationForm &&
      !NON_NOTES_MUTATION_DESTINATION.test(clause) &&
      hasAnyAppendTarget(clause) &&
      mutationTargetIsNearAction(clause) &&
      (!mutationForm.polite || CONCRETE_MUTATION_CONTENT.test(clause))
    ) {
      const needsDiscovery = appendTargetNeedsDiscovery(clause);
      result.append = "grant";
      result.appendNeedsDiscovery = needsDiscovery;
      if (needsDiscovery && result.search !== "deny") {
        if (result.search !== "grant") {
          result.search = "grant";
          result.searchBroadWhenSelected = false;
        }
      }
    }
  }
  return result;
};

const evaluateNotesAuthority = (
  text: string,
  excludedReferenceTexts: readonly string[] = []
): NotesAuthorityEvaluation => {
  const normalized = notesAuthorityText(text, excludedReferenceTexts);
  return normalized
    ? evaluateNotesAuthorityText(normalized)
    : {
        search: null,
        append: null,
        searchBroadWhenSelected: false,
        appendNeedsDiscovery: false,
      };
};

/**
 * Grants broad Notes search only for a direct request or narrow personal
 * recall question in the current user turn. Reports, hypotheticals,
 * capability questions, contextual “the notes,” and reference data stay inert.
 */
export const notesSearchRequested = (
  text: string,
  excludedReferenceTexts: readonly string[] = []
): boolean => evaluateNotesAuthority(text, excludedReferenceTexts).search === "grant";

export type NotesSearchScopeState = {
  active: boolean;
  recoverableFromReferenceText: boolean;
};

/**
 * Projects the composer’s visible one-turn Notes scope. Pasted instructions
 * stay inert until the user makes the separate, structured choice to enable
 * search; removing an automatically inferred scope likewise stays removed for
 * the rest of that draft.
 */
export const notesSearchScopeState = (
  text: string,
  excludedReferenceTexts: readonly string[] = [],
  searchScopeOverride: boolean | null = null
): NotesSearchScopeState => {
  const inferred = notesSearchRequested(text, excludedReferenceTexts);
  return {
    active: searchScopeOverride ?? inferred,
    recoverableFromReferenceText:
      searchScopeOverride === null &&
      !inferred &&
      excludedReferenceTexts.length > 0 &&
      notesSearchRequested(text),
  };
};

export const shouldResetNotesSearchScope = (
  prompt: string,
  override: boolean | null
): boolean => prompt.length === 0 && override !== null;

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
): boolean => evaluateNotesAuthority(text, excludedReferenceTexts).append === "grant";

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
  excludedReferenceTexts: readonly string[] = [],
  searchScopeOverride: boolean | null = null
): NoteAccessContext | null => {
  const notes = uniqueSelectedNotes(selectedNotes);
  const authority = evaluateNotesAuthority(text, excludedReferenceTexts);
  const wantsNotes = searchScopeOverride ?? authority.search === "grant";
  const allowAppendProposal =
    authority.append === "grant" &&
    (!authority.appendNeedsDiscovery || wantsNotes);
  // An attached Note is already an exact target. "Add this to my notes"
  // therefore stays selected-only; broad authority is needed only when the
  // turn explicitly asks Ultra to discover/read Notes, or no Note was attached.
  if (
    wantsNotes &&
    (notes.length === 0 ||
      searchScopeOverride === true ||
      authority.searchBroadWhenSelected ||
      authority.appendNeedsDiscovery)
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
