import type { RemoteMutationIntent } from "../types";

const REMOTE_BISQUE = String.raw`(?:my\s+|our\s+|the\s+)?(?:linked\s+)?bisque(?:\s+(?:account|server|instance))?`;
const NON_CONSENT = new RegExp(
  String.raw`\b(?:do\s+not|don't|never|without|did(?:\s+not|n't)\s+ask|if\s+i|if\s+we|explain\s+how|how\s+(?:do|can|could|would)|can\s+(?:ultra|you)|could\s+(?:ultra|you)|would\s+(?:ultra|you))\b`,
  "i"
);
const UPLOAD_REQUEST = new RegExp(
  String.raw`(?:^|[.!?]\s*|\bplease\s+|\bthen\s+|\band\s+)` +
    String.raw`(?:upload|push|publish|send|copy)\b[^.!?\n]{0,120}\b(?:to|into|onto|on|in)\s+${REMOTE_BISQUE}\b`,
  "i"
);
const DATASET_REQUEST = new RegExp(
  String.raw`(?:^|[.!?]\s*|\bplease\s+|\bthen\s+|\band\s+)` +
    String.raw`(?:create|make|group|organize)\b[^.!?\n]{0,120}(?:\b${REMOTE_BISQUE}\s+dataset\b|\bdataset\b[^.!?\n]{0,48}\b(?:in|on|within|under)\s+${REMOTE_BISQUE}\b)`,
  "i"
);

const RUN_MODULE_VERB = String.raw`(?:run|execute|launch|start|apply)`;
const RUN_MODULE_REQUEST = new RegExp(
  String.raw`(?:^|[.!?]\s*|\bplease\s+|\bthen\s+|\band\s+|\bnow\s+)` +
    String.raw`${RUN_MODULE_VERB}\b[^.!?\n]{0,120}(?:\bmodule\b|\b(?:on|in|via|using)\s+${REMOTE_BISQUE}\b)`,
  "i"
);
const RUN_MODULE_DENIAL = new RegExp(
  String.raw`\b(?:do\s+not|don't|never|without)\b[^.!?\n]{0,48}\b${RUN_MODULE_VERB}\b`,
  "i"
);

// This is deliberately a conservative client-side confirmation extractor.
// The resulting bounded enum is authenticated and persisted by the Go control
// plane; transformed model prompts and free-form metadata are never authority.
export function remoteMutationIntentsForUserText(text: string): RemoteMutationIntent[] {
  const currentTurn = String(text ?? "").trim();
  if (!currentTurn || NON_CONSENT.test(currentTurn)) {
    return [];
  }
  const intents: RemoteMutationIntent[] = [];
  if (UPLOAD_REQUEST.test(currentTurn)) {
    intents.push("bisque.upload");
  }
  if (DATASET_REQUEST.test(currentTurn)) {
    intents.push("bisque.create_dataset");
  }
  if (RUN_MODULE_REQUEST.test(currentTurn) && !RUN_MODULE_DENIAL.test(currentTurn)) {
    intents.push("bisque.run_module");
  }
  return intents;
}
