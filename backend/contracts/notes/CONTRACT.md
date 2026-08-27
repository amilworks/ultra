# Notes model-context contract

This contract defines the production boundary between the authenticated browser,
the Go control plane, and the coordinator Python worker. Notes are private,
user-owned data. Absence of explicit per-run Notes scope means no model access.

## Run scope

The browser may place one reserved object in `selection_context`:

```json
{
  "note_access": {
    "mode": "selected",
    "notes": [{ "note_id": "note_123", "revision": 4 }],
    "allow_append_proposal": true
  }
}
```

- `mode: "selected"` authorizes reads of only the non-empty, owner-validated
  note list.
- `mode: "search"` authorizes bounded search and reads of search-discovered
  notes owned by the run user. It may also carry selected-note references.
- At most 20 unique references may be persisted. Unknown, malformed, foreign,
  or missing references fail closed; the control plane stores only its canonical
  owner-validated shape.
- Scope is immutable run metadata. Notes content, titles, and search queries are
  never placed in the job envelope.
- `allow_append_proposal` defaults to false. It authorizes only submission of a
  browser-review proposal; the authenticated browser remains the sole commit
  authority. The control plane also stamps the reserved boolean run metadata
  key `model_notes_proposals_enabled`, and the proposal tool requires both
  values to be true.
- Notes scope currently cannot be combined with selected files, resources,
  datasets, knowledge context, workflow hints, or selected tools. The control
  plane rejects those combinations instead of letting a narrow Notes runtime
  silently ignore user selections. Ordinary message text remains supported.
- Cleanroom/protected runs and delegated agents have no Notes authority.

New Notes-scoped runs are rejected with `503` while the model Notes read gate is
disabled. A matching idempotent replay of an existing run remains available
during a kill-switch event. A raw retry that omitted revisions is compared with
the canonical stored scope and reuses its sealed revisions without rereading a
changed or deleted Note; different mode, note IDs, explicit revision, or append-
proposal flag conflicts.

## Browser intent and draft provenance

Notes access is authority, so only instructions the user types as the current
turn may request it. Text presented as reference material cannot mint search or
append-proposal authority. Before intent detection, the browser blanks quoted
spans, Markdown or HTML blockquotes, inline and fenced code, and the exact text
fragments observed through composer paste events. Removing those fragments must
preserve the surrounding typed instruction; a user can still type an explicit
request alongside pasted source material.

The browser persists each unsent draft and its bounded paste-exclusion list in
one versioned storage snapshot. Writes for a conversation are serialized, and a
snapshot is acknowledged only after its storage write succeeds; a failed or
late older write must neither suppress retry nor overwrite a newer draft and
provenance snapshot. Edit, retry, queue, cancel, steering recovery, reload, and
sign-out paths preserve or clear draft text and provenance together.

Restored legacy, partial, or corrupt records keep their visible draft but treat
the whole restored text as reference material for Notes intent. More than 20
paste fragments collapse to the printable, content-free sticky marker
`__ULTRA_NOTES_INTENT_EXCLUSIONS_OVERFLOW_V1__`. That marker is stored with the
draft and disables implicit Notes search and append-proposal authority until the
draft is cleared; it is never evicted to make room for later fragments. These
fail-closed states do not prevent sending the ordinary chat message.

Notes-scoped runs do not accept live steering. Their Note access and append-
proposal consent are immutable for the turn, so a later steer cannot be applied
without creating conflicting authority. `POST /v2/runs/{run_id}/steer` returns
the existing typed `409 steering_closed`, and the browser preserves the text and
paste provenance as a queued follow-up turn. Notes-only workers construct no
steering inbox or finalization barrier, so a steer accepted by an older control
plane during a rolling deploy cannot enter the model; existing missed-steer
recovery remains responsible for the queued follow-up. Ordinary runs retain live
steering.

## Worker authority

Every worker request uses the existing worker credential plus all three
lease-binding headers:

- `X-Ultra-Worker-Token`
- `X-Ultra-Run-Id`
- `X-Ultra-Worker-Id`
- `X-Ultra-Run-Lease-Token`

The path run ID and header run ID must match. The control plane requires a
running run and a live matching worker lease, resolves the owner from the stored
run, and rechecks the immutable Note scope. Caller-supplied user or organization
identifiers are never authority. A foreign note is indistinguishable from a
missing note.

## Coordinator tools and HTTP endpoints

Only the coordinator may receive these tools. A Notes-enabled run uses a narrow
agent surface containing only the authorized Notes tools; it does not inherit
filesystem, execute, memory, output, artifact, caller-tool, Builder, subagent,
async-delegation, map, or tool-program capabilities. Expanding that surface is
a privacy-boundary change requiring a new audit.

### `search_notes`

`POST /v2/runs/{run_id}/note-search`

```json
{ "query": "microscopy calibration", "sort": "relevance", "limit": 8 }
```

The tool defaults are `query = ""`, `sort = "relevance"`, and `limit = 8`.
`sort` is either `relevance` or `recent`. Relevance requires a non-blank query;
recent permits an empty query for requests such as “my most recent Note.” A
non-empty recent query still applies the lexical filter, but ordering remains
strict content recency. Queries are bounded to 512 characters and the limit is
capped at 20.

Relevance order is exact title, title substring, `content_updated_at DESC`,
then `note_id ASC`. Recent order is `content_updated_at DESC`, then
`note_id ASC`; it never considers pin state or title rank. Pin and editor-mode
changes advance the Note revision and general `updated_at`, but do not advance
`content_updated_at`; title/body writes, appends, and undo do. This same
relevance order is used by an active browser Notes query. The ordinary blank-
query browser list defaults to pin, content recency, and Note ID, while
`GET /v2/notes?sort=recent` omits pin rank for chooser surfaces.

Results contain only bounded metadata: note ID, title, match-centered snippet,
pin state, revision, general update time, and content update time. The response
uses `limit + 1` for `has_more`, returns `next_cursor`, and does not run a
separate exact-count query for the worker. A continuation request sends the
same normalized query and sort plus that opaque cursor. The cursor is bounded
and bound to the leased run, query, sort, a store-authoritative search-start
snapshot, and the last `(rank, content_updated_at, note_id)` sort key (rank is
omitted semantically for recent order). Every page filters to
`content_updated_at <= snapshot` and uses exclusive keyset traversal. A
title/body mutation advances content recency and therefore leaves that in-flight
snapshot instead of duplicating or reordering a row; pin/editor-only mutations
do not affect search order. Malformed, legacy offset, cross-run, or mismatched
cursors are rejected instead of silently restarting or changing the result set.

### `read_note`

`POST /v2/runs/{run_id}/note-read`

```json
{ "note_id": "note_123", "max_chars": 8000, "cursor": "opaque-if-continuing" }
```

The server returns a UTF-8-safe body chunk of at most 16,000 characters plus
`note_id`, `title`, `revision`, `content_digest`, byte range, `has_more`, an
opaque continuation cursor, and a short-lived opaque `read_token`. Cursors bind
the note, revision, and byte offset; a changed note yields a conflict instead of
combining revisions. Per-run call and byte budgets are enforced server-side.

### `propose_note_append`

`POST /v2/runs/{run_id}/note-append-proposals`

```json
{
  "note_id": "note_123",
  "expected_revision": 4,
  "body_markdown": "Exact text for browser review.",
  "read_token": "opaque",
  "idempotency_key": "host-derived"
}
```

The tool exists only when the current run goal explicitly asks Ultra to add,
append, save, write, record, update, or jot material in a Note. Negated,
hypothetical, capability-only, and how-to prompts do not enable it. The worker
derives the idempotency key from the run and tool-call identity; the model cannot
provide or see it.

The control plane requires a valid read token for the same run, owner, note, and
revision. It stores an expiring exact-text proposal but does not mutate the
Note. Same key and request returns the original proposal; the same key with a
different request returns `note_append_idempotency_conflict`. Identical
run/note/revision/body proposals are
deduplicated while pending or committed. An expired proposal does not prevent a
new explicit request from creating a fresh proposal.

## Browser approval and optimistic concurrency

All browser Note patches include `expected_revision`. Every durable Note
mutation increments the monotonic revision; stale writes return HTTP 409 with
`note_revision_conflict`. The editor preserves the local draft on conflict.

The authenticated browser fetches a proposal by ID and shows its exact target
and Markdown. The user may edit the proposed text before choosing **Add to
note**. Commit is an atomic, owner-scoped append against the proposal's base
revision and the 2 MiB Note limit. It produces a content-free idempotent receipt
with revisions, byte count, digests, and timestamps. The proposal body is erased
after commit. A mandatory background sweep also erases exact proposal text
shortly after expiry, independently of optional resource-retention settings.

Undo is conditional: it succeeds only while the Note still has the committed
revision and exact appended suffix. It never overwrites newer writing. Hard Note
deletion cascades through grants, proposals, and receipts. Per-run aggregate
usage contains no Note identity or content and is deleted with the run.

The model has no direct create, rename, replace, prepend, pin, delete, or commit
operation.

## Browser create retry safety

`POST /v2/notes` remains compatible with legacy callers that omit an
idempotency key. The new local-draft flow sends an optional `Idempotency-Key`
header containing a stable opaque key of at most 256 bytes. The first committed
create returns `201`; a same-key retry of the same normalized title, body, pin,
and editor mode returns the original Note with `200`, even though the server may
have generated a different candidate Note ID for the retry. The same key with a
different normalized request returns `409` with
`note_create_idempotency_conflict`. There is no title/body similarity or
heuristic deduplication.

For an authenticated request with exactly one valid key and a decodable body,
the owner/key/request-digest receipt lookup happens before mutable title, body
size, or editor-mode validation. A live exact historical replay therefore still
returns `200` if validation rules tighten after its original commit; a live
same-key/different-request conflict still wins before validation. Only after a
confirmed receipt miss does application validation run. Such a rejection is
`400` with `note_create_not_committed`, explicitly proving that this request did
not commit. Header/decode failures and proxy-generated `400`/`413` responses do
not carry that code and remain ambiguous. This route does not use `422`.

The durable create receipt is owner-scoped and content-free: while its Note is
live it stores only the random client key, private request digest, Note ID, and
timestamp. Hard Note deletion clears both the Note ID and content-derived
request digest but deliberately preserves the owner/key tombstone. Every later
same-key replay—whether its body matches or differs—returns the same
owner-scoped `410` with `note_create_replay_deleted`; it never confirms deleted
text or Note identity and never resurrects the Note. This terminal code lets a
browser stop retrying an effect that no longer survives without weakening the
privacy tombstone.

## Browser exact-capture append

A human selection saved to a new Note first opens a body-first local draft; it
does not persist an untitled Note or mint model Notes authority. When the user
chooses an existing target Note, the authenticated browser may use the direct
capture endpoint without involving a model:

`POST /v2/notes/{note_id}/append`

- Header: `Idempotency-Key`, a stable opaque client key of at most 256 bytes.
- Body: `{ "body_markdown": "exact captured text", "expected_revision": 4 }`.
- First success is `201`; an exact retry is `200` with the original receipt.

The control plane binds the key to owner, Note ID, expected revision, and exact
body digest before mutation. The same key with another request returns `409`
with `note_append_idempotency_conflict`; a stale Note returns `409` with
`note_revision_conflict`. A replay lookup precedes revision/liveness checks, so
a transport retry remains stable after the successful append. The transaction
row-locks the owner Note and applies the revision CAS once. If the Note body is
blank, the captured bytes become the exact body. Otherwise the server inserts
only the required blank-line separator before the exact captured bytes. The
final body remains subject to the 2 MiB Note limit.

The live owner/key/request-digest receipt lookup also precedes mutable append
validation. An exact historical replay returns `200` even if today's validation
would reject a new equivalent request, while a mismatched digest still returns
`note_append_idempotency_conflict`. After a confirmed receipt miss, application
validation—including the transactional combined Note-size bound—rejects with
`400` and `note_append_not_committed`. Generic
header/decode/proxy `400` or `413` responses remain ambiguous, and the route does
not use `422`. If hard deletion has removed both the target Note and its cascaded
direct-append receipt, retry returns the owner-scoped terminal `404`
`note_append_target_unavailable`: no owner-visible append effect survives.

For historical-retry cleanup, the only definitive release codes are
`note_create_not_committed`, `note_append_not_committed`,
`note_create_replay_deleted`, and `note_append_target_unavailable`. HTTP status
alone is never proof that a response-loss request did not previously commit.

The durable direct-append operation stores no title or body, only owner/Note
identity, the private retry/request digests, suffix byte range/hash, revisions,
content digests, and timestamps. The browser response joins the current title
transiently and otherwise remains content-free. Hard Note deletion cascades the
receipt.

`POST /v2/note-direct-append-operations/{operation_id}/undo` is owner-scoped
and idempotent. It removes the append only when the Note is still at the exact
post-append revision and still ends with the exact stored suffix; any newer
content or metadata mutation returns `note_undo_conflict` and is never
overwritten.

## Durable trace policy

Notes tool payloads are projected before generic event normalization. The same
turn-wide boundary applies defensively to every internal stream once
`note_access` is present, even if an unexpected tool or subagent event arrives.

- Never persist a search query, title, snippet, body chunk, cursor, read token,
  proposal Markdown, idempotency key, or generic output preview.
- Search completion may retain result count and `has_more`.
- Read completion may retain note ID, revision, returned byte count, and
  `has_more`. The body-derived content digest is not retained in durable events.
- Proposal completion may retain proposal ID, note ID, expected revision,
  expiry, and status.
- Tool name, call ID, lifecycle status, and a stable content-free error code may
  be retained.
- Coordinator reasoning, unexpected subagent messages, generic tool arguments
  and output previews, execute progress, provider error detail, and trace-lens
  model-input capture are content-free or disabled for the entire run. The
  browser scrubs prior and future reasoning when it sees the redacted marker and
  persists only generic activity metadata.
- Conversation-title generation remains request-only: Note-derived response or
  artifact text is neither sent to the title model nor used by its deterministic
  fallback. The final assistant answer is the only Note-derived prose that may
  be persisted as conversation content.
- A run carrying the reserved `selection_context.note_access` marker starts a
  durable privacy lineage for its thread. Every later run in that thread is
  server-marked, because a plain-text follow-up may restate Note-derived prose.
  No run from a lineage-bearing thread is eligible for cross-conversation
  episodic search, including recency-only search. Final answers remain
  available through direct or source-thread conversation surfaces; unscoped
  run lists omit their response text. Cross-conversation reads also classify
  the complete thread dynamically so unmarked descendants written before or
  during a rolling upgrade remain protected.

Note content used by the model necessarily becomes part of that run's in-memory
model context and can be reflected in the conversation. The Notes-only runner
does not hydrate, wire, or write the ordinary durable LangGraph checkpointer, so
body chunks and opaque read tokens never enter checkpoint storage. A delivery
failure restarts the Notes graph from the sealed job envelope; owner/run budgets,
read grants, revision checks, and proposal idempotency remain server-authoritative
across attempts. Ordinary non-Notes runs retain durable resume behavior. During a
rolling worker deploy, any legacy Notes checkpoint is ignored by the new runner
and cannot restore an accepted steer or Note tool output; terminal ACK cleanup
removes the legacy row. The UI must not promise that deleting a source Note
rewrites an active in-memory model context or an existing conversation.

## Production rollout order

Roll out schema and the compatible Go control plane first. This closes live
steering for Notes runs before any worker changes and makes clients queue the
text through the established `steering_closed` path. The deployed
upgrade adds `content_updated_at` as nullable, sets the default for old-writer
inserts, avoids a table-wide backfill and new ordinary recency indexes, and
reads legacy rows through `COALESCE(content_updated_at, updated_at)`; fresh
databases remain `NOT NULL`. Deploy Python workers next so Notes runs omit both
steering and durable checkpoints and their strict Note search response parser
understands `pinned`, `content_updated_at`, `sort`, and v2 cursors. Deploy the
revision/idempotency-aware frontend last. Only after the
frontend compatibility window should strict expected-revision enforcement be
enabled, followed by model Notes read and then proposal creation gates.
