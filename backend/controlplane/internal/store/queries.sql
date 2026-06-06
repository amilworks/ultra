-- name: CreateThread :one
INSERT INTO control_threads (thread_id, user_id, title, status, created_at, updated_at, latest_run_id, checkpoint_id, summary, metadata)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
RETURNING *;

-- name: GetThread :one
SELECT * FROM control_threads WHERE thread_id = $1;

-- name: GetThreadForUser :one
SELECT * FROM control_threads WHERE thread_id = $1 AND user_id = $2;

-- name: CountThreads :one
SELECT COUNT(*) FROM control_threads
WHERE ($1::text = '' OR status = $1);

-- name: CountThreadsForUser :one
SELECT COUNT(*) FROM control_threads
WHERE user_id = $1
  AND ($2::text = '' OR status = $2);

-- name: ListThreads :many
SELECT * FROM control_threads
WHERE ($1::text = '' OR status = $1)
ORDER BY updated_at DESC
LIMIT $2 OFFSET $3;

-- name: ListThreadsForUser :many
SELECT * FROM control_threads
WHERE user_id = $1
  AND ($2::text = '' OR status = $2)
ORDER BY updated_at DESC
LIMIT $3 OFFSET $4;

-- name: InsertThreadMessage :one
INSERT INTO control_thread_messages (message_id, thread_id, role, content, created_at, metadata, run_id)
VALUES ($1, $2, $3, $4, $5, $6, $7)
RETURNING *;

-- name: ListThreadMessages :many
SELECT * FROM control_thread_messages WHERE thread_id = $1 ORDER BY created_at ASC;

-- name: ListThreadMessagesForUser :many
SELECT m.*
FROM control_thread_messages m
JOIN control_threads t ON t.thread_id = m.thread_id
WHERE m.thread_id = $1
  AND t.user_id = $2
ORDER BY m.created_at ASC;

-- name: CreateRun :one
INSERT INTO control_runs (run_id, thread_id, user_id, goal, status, workflow_kind, mode, created_at, updated_at, metadata)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
RETURNING *;

-- name: SetThreadLatestRun :exec
UPDATE control_threads SET latest_run_id = $2, updated_at = $3 WHERE thread_id = $1;

-- name: GetRun :one
SELECT * FROM control_runs WHERE run_id = $1;

-- name: GetRunForUser :one
SELECT * FROM control_runs WHERE run_id = $1 AND user_id = $2;

-- name: ListRuns :many
SELECT * FROM control_runs
WHERE ($1::text = '' OR thread_id = $1)
  AND ($2::text = '' OR status = $2)
ORDER BY updated_at DESC
LIMIT $3 OFFSET $4;

-- name: ListRunsForUser :many
SELECT * FROM control_runs
WHERE user_id = $1
  AND ($2::text = '' OR thread_id = $2)
  AND ($3::text = '' OR status = $3)
ORDER BY updated_at DESC
LIMIT $4 OFFSET $5;

-- name: UpdateRunStatus :one
UPDATE control_runs
SET status = $2,
    response_text = NULLIF($3, ''),
    error = NULLIF($4, ''),
    updated_at = $5,
    started_at = CASE WHEN $2 = 'running' AND started_at IS NULL THEN $5 ELSE started_at END,
    completed_at = CASE WHEN $2 IN ('succeeded', 'failed', 'canceled') THEN $5 ELSE completed_at END
WHERE run_id = $1
  AND status NOT IN ('succeeded', 'failed', 'canceled')
RETURNING *;

-- name: NextRunEventSequence :one
SELECT COALESCE(MAX(sequence_number), 0) + 1 AS next_sequence FROM control_run_events WHERE run_id = $1;

-- name: AppendRunEvent :one
INSERT INTO control_run_events (
  event_id, sequence_number, run_id, thread_id, event_kind, event_type,
  node_name, task_id, checkpoint_id, scope_id, agent_role, level, ts, message, payload
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
RETURNING *;

-- name: GetRunEvent :one
SELECT * FROM control_run_events WHERE event_id = $1;

-- name: ListRunEvents :many
SELECT *
FROM (
  SELECT * FROM control_run_events WHERE run_id = $1 ORDER BY sequence_number DESC LIMIT $2
) recent_events
ORDER BY sequence_number ASC;

-- name: ListRunEventsForUser :many
SELECT *
FROM (
  SELECT e.*
  FROM control_run_events e
  JOIN control_runs r ON r.run_id = e.run_id
  WHERE e.run_id = $1
    AND r.user_id = $2
  ORDER BY e.sequence_number DESC
  LIMIT $3
) recent_events
ORDER BY sequence_number ASC;

-- name: ListRunEventsAfter :many
SELECT *
FROM control_run_events
WHERE run_id = $1 AND sequence_number > $2
ORDER BY sequence_number ASC
LIMIT $3;

-- name: ListRunEventsAfterForUser :many
SELECT e.*
FROM control_run_events e
JOIN control_runs r ON r.run_id = e.run_id
WHERE e.run_id = $1
  AND r.user_id = $2
  AND e.sequence_number > $3
ORDER BY e.sequence_number ASC
LIMIT $4;

-- name: CreateArtifact :one
INSERT INTO control_artifacts (
  artifact_id, run_id, thread_id, kind, path, source_path, preview_path, title,
  result_group_id, mime_type, size_bytes, sha256, storage_uri, tool_name, category,
  created_at, updated_at, metadata
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
RETURNING *;

-- name: ListRunArtifacts :many
SELECT * FROM control_artifacts WHERE run_id = $1 ORDER BY created_at DESC LIMIT $2;

-- name: ListRunArtifactsForUser :many
SELECT a.*
FROM control_artifacts a
JOIN control_runs r ON r.run_id = a.run_id
WHERE a.run_id = $1
  AND r.user_id = $2
ORDER BY a.created_at DESC
LIMIT $3;

-- name: GetArtifact :one
SELECT * FROM control_artifacts WHERE artifact_id = $1;

-- name: GetArtifactForUser :one
SELECT a.*
FROM control_artifacts a
JOIN control_runs r ON r.run_id = a.run_id
WHERE a.artifact_id = $1
  AND r.user_id = $2;
