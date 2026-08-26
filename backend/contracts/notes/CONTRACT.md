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
{ "query": "microscopy calibration", "limit": 8 }
```

The query is non-empty and bounded to 512 characters. The limit is capped at
20. Results are owner-scoped, lexically matched, relevance ordered, and contain
only bounded metadata: note ID, title, match-centered snippet, pin state,
revision, and update time. The response uses `limit + 1` for `has_more`; it
does not run a separate exact-count query for the worker.

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
different request conflicts. Identical run/note/revision/body proposals are
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

Note content used by the model necessarily becomes part of that run's model
context and can be reflected in the conversation. While a run is resumable, its
ordinary durable LangGraph checkpoint may temporarily contain the tool result.
For a terminal outcome, the worker first confirms the JetStream acknowledgement
(`ack_sync` when supported) and only then deletes the durable checkpoint. If
acknowledgement fails, it flushes and retains the checkpoint so redelivery can
finish safely without recomputing the run. Restored checkpoint state is
classified explicitly: absent starts, pending resumes, and completed waits for
the original terminal event to become control-plane authority via delayed
redelivery; a completed graph never receives the original prompt again. A
failed boundary flush is not treated as durable and keeps the freshest
worker-local slice instead of clearing it. A redelivery that finds the run
already terminal likewise acknowledges before initializing cleanup and deleting
the durable row, including on a fresh worker. An hourly worker sweep
removes failed-cleanup checkpoint rows after the configured 72-hour retention
window only when the parent run is terminal; queued, running, and waiting runs
are never deleted merely because of age. A worker also flushes and clears its
process-local slice whenever it relinquishes a non-terminal delivery. The UI
must not promise that deleting a source Note rewrites an active model context or
an existing conversation.
