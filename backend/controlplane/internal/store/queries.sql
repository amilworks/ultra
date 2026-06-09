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

-- name: UpsertResource :one
INSERT INTO control_resources (
  resource_id, owner_user_id, owner_org_id, owner_role, original_name, content_type,
  size_bytes, sha256, storage_uri, storage_path, source_type, resource_kind,
  source_uri, project_id, status, created_at, updated_at, deleted_at, retention_expires_at, metadata
)
VALUES (
  $1, $2, $3, $4, $5, $6,
  $7, $8, $9, $10, $11, $12,
  $13, $14, $15, $16, $17, $18, $19, $20
)
ON CONFLICT (resource_id) DO UPDATE SET
  owner_user_id = EXCLUDED.owner_user_id,
  owner_org_id = EXCLUDED.owner_org_id,
  owner_role = EXCLUDED.owner_role,
  original_name = EXCLUDED.original_name,
  content_type = EXCLUDED.content_type,
  size_bytes = EXCLUDED.size_bytes,
  sha256 = EXCLUDED.sha256,
  storage_uri = EXCLUDED.storage_uri,
  storage_path = EXCLUDED.storage_path,
  source_type = EXCLUDED.source_type,
  resource_kind = EXCLUDED.resource_kind,
  source_uri = EXCLUDED.source_uri,
  project_id = EXCLUDED.project_id,
  status = EXCLUDED.status,
  updated_at = EXCLUDED.updated_at,
  deleted_at = EXCLUDED.deleted_at,
  retention_expires_at = EXCLUDED.retention_expires_at,
  metadata = EXCLUDED.metadata
RETURNING *;

-- name: GetResourceForUser :one
SELECT r.*
FROM control_resources r
WHERE r.resource_id = $1
  AND (
    (
      r.owner_user_id = $2
      AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $3)
    )
    OR EXISTS (
      SELECT 1
      FROM control_resource_share_grants g
      WHERE g.resource_id = r.resource_id
        AND g.status = 'active'
        AND (
          (
            COALESCE(g.grantee_user_id, '') <> ''
            AND g.grantee_user_id = $2
            AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $3)
          )
          OR (
            COALESCE(g.grantee_user_id, '') = ''
            AND COALESCE(g.grantee_org_id, '') <> ''
            AND g.grantee_org_id = $3
          )
        )
    )
  )
  AND r.status = 'active';

-- name: GetResourceForOwner :one
SELECT *
FROM control_resources
WHERE resource_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3);

-- name: ListResourcesForUser :many
WITH visible_resources AS (
  SELECT r.resource_id, r.owner_user_id, r.owner_org_id, r.owner_role, r.original_name,
         r.content_type, r.size_bytes, r.sha256, r.storage_uri, r.storage_path,
         r.source_type, r.resource_kind, r.source_uri, r.project_id, r.status AS resource_status,
         r.created_at, r.updated_at, r.deleted_at, r.retention_expires_at, r.metadata,
         (
           r.owner_user_id = $1
           AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $2)
         ) AS is_owner,
         (
           SELECT count(*)::bigint
           FROM control_resource_share_grants g
           WHERE g.resource_id = r.resource_id
             AND g.status = 'active'
         ) AS active_owner_grant_count,
         (
           SELECT count(*)::bigint
           FROM control_resource_share_grants g
           WHERE g.resource_id = r.resource_id
             AND g.status = 'active'
             AND COALESCE(g.grantee_user_id, '') = '__public__'
         ) AS active_public_grant_count,
         (
           SELECT count(*)::bigint
           FROM control_resource_share_grants g
           WHERE g.resource_id = r.resource_id
             AND g.status = 'active'
             AND (
               (
                 COALESCE(g.grantee_user_id, '') <> ''
                 AND g.grantee_user_id = $1
                 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $2)
               )
               OR (
                 COALESCE(g.grantee_user_id, '') = ''
                 AND COALESCE(g.grantee_org_id, '') <> ''
                 AND g.grantee_org_id = $2
               )
             )
         ) AS active_matching_grant_count
  FROM control_resources r
  LEFT JOIN control_resource_search_documents sd ON sd.resource_id = r.resource_id
  WHERE (
      (
        r.owner_user_id = $1
        AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $2)
      )
      OR (
        r.status = 'active'
        AND EXISTS (
          SELECT 1
          FROM control_resource_share_grants g
          WHERE g.resource_id = r.resource_id
            AND g.status = 'active'
            AND (
              COALESCE(g.grantee_user_id, '') = '__public__'
              OR
              (
                COALESCE(g.grantee_user_id, '') <> ''
                AND g.grantee_user_id = $1
                AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $2)
              )
              OR (
                COALESCE(g.grantee_user_id, '') = ''
                AND COALESCE(g.grantee_org_id, '') <> ''
                AND g.grantee_org_id = $2
              )
            )
        )
      )
    )
    AND r.status = $3
    AND ($4::text = '' OR r.resource_kind = $4)
    AND ($5::text = '' OR r.source_type = $5)
    AND ($6::text = '' OR r.project_id = $6)
    AND (
      $7::text = ''
      OR sd.search_vector @@ plainto_tsquery('simple', $7::text)
      OR lower(COALESCE(sd.search_text, '')) LIKE '%' || lower($7::text) || '%'
    )
    AND (
      cardinality($8::text[]) = 0
      OR COALESCE(r.metadata->'tag_keys', '[]'::jsonb) ?& $8::text[]
    )
    AND (
      cardinality($9::text[]) = 0
      OR NOT EXISTS (
        SELECT 1
        FROM unnest($9::text[]) AS metadata_filters(filter)
        CROSS JOIN LATERAL (
          SELECT split_part(metadata_filters.filter, ':', 1) AS path,
                 split_part(metadata_filters.filter, ':', 2) AS operator,
                 substring(metadata_filters.filter from '^[^:]*:[^:]*:(.*)$') AS expected
        ) mf
        CROSS JOIN LATERAL (
          SELECT r.metadata #> regexp_split_to_array(mf.path, E'\\.') AS actual_json,
                 r.metadata #>> regexp_split_to_array(mf.path, E'\\.') AS actual_text
        ) mv
        WHERE NOT (
          (mf.operator = 'exists' AND mv.actual_json IS NOT NULL)
          OR (mf.operator = 'eq' AND lower(COALESCE(mv.actual_text, '')) = lower(mf.expected))
          OR (mf.operator = 'contains' AND lower(COALESCE(mv.actual_text, mv.actual_json::text, '')) LIKE '%' || lower(mf.expected) || '%')
          OR (mf.operator = 'lt' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric < mf.expected::numeric ELSE false END)
          OR (mf.operator = 'lte' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric <= mf.expected::numeric ELSE false END)
          OR (mf.operator = 'gt' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric > mf.expected::numeric ELSE false END)
          OR (mf.operator = 'gte' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric >= mf.expected::numeric ELSE false END)
        )
      )
    )
    AND ($10::timestamptz IS NULL OR r.created_at >= $10)
    AND ($11::timestamptz IS NULL OR r.created_at <= $11)
    AND (
      $12::text = ''
      OR $12::text = 'all'
      OR ($12::text = 'caption_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('succeeded', 'completed'))
      OR ($12::text = 'metadata_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('succeeded', 'completed'))
      OR ($12::text = 'tags_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('succeeded', 'completed'))
      OR ($12::text = 'qc_complete' AND lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('succeeded', 'completed'))
      OR ($12::text = 'dedupe_checked' AND lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('succeeded', 'completed'))
      OR ($12::text = 'organization_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('succeeded', 'completed'))
      OR ($12::text = 'data_agent_ready' AND (
        lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('succeeded', 'completed')
        OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('succeeded', 'completed')
        OR lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('succeeded', 'completed')
        OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('succeeded', 'completed')
        OR lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('succeeded', 'completed')
        OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('succeeded', 'completed')
      ))
      OR ($12::text = 'needs_caption' AND lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) NOT IN ('succeeded', 'completed'))
      OR ($12::text = 'needs_metadata' AND lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) NOT IN ('succeeded', 'completed'))
      OR ($12::text = 'data_agent_failed' AND (
        lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('failed', 'error')
        OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('failed', 'error')
        OR lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('failed', 'error')
        OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('failed', 'error')
        OR lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('failed', 'error')
        OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('failed', 'error')
      ))
    )
    AND (
      cardinality($14::text[]) = 0
      OR NOT EXISTS (
        SELECT 1
        FROM unnest($14::text[]) AS descriptor_filters(filter)
        WHERE NOT (
          COALESCE(r.metadata->'tag_keys', '[]'::jsonb) ? lower(descriptor_filters.filter)
          OR lower(COALESCE(r.metadata->>'label', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata->>'descriptor', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata->>'scientific_descriptor', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata->>'diagnosis', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata->>'modality', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata->>'organism', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata->>'species', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,caption}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,summary}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,caption}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,summary}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,label}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,descriptor}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,scientific_descriptor}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,summary}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,summary}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata->'labels') = 'array' THEN r.metadata->'labels' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata->'descriptors') = 'array' THEN r.metadata->'descriptors' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata->'scientific_descriptors') = 'array' THEN r.metadata->'scientific_descriptors' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata->'diagnoses') = 'array' THEN r.metadata->'diagnoses' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata #> '{data_agent,extract_metadata,labels}') = 'array' THEN r.metadata #> '{data_agent,extract_metadata,labels}' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata #> '{data_agent,extract_metadata,descriptors}') = 'array' THEN r.metadata #> '{data_agent,extract_metadata,descriptors}' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata #> '{data_agent,extract_metadata,scientific_descriptors}') = 'array' THEN r.metadata #> '{data_agent,extract_metadata,scientific_descriptors}' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
        )
      )
    )
),
summarized_resources AS (
  SELECT resource_id, owner_user_id, owner_org_id, owner_role, original_name, content_type,
         size_bytes, sha256, storage_uri, storage_path, source_type, resource_kind, source_uri,
         project_id, resource_status, created_at, updated_at, deleted_at, retention_expires_at,
         metadata,
         CASE
           WHEN active_public_grant_count > 0 THEN 'public'
           WHEN is_owner AND active_owner_grant_count > 0 THEN 'shared_by_me'
           WHEN NOT is_owner AND active_matching_grant_count > 0 THEN 'shared_with_me'
           ELSE 'private'
         END::text AS share_status,
         CASE
           WHEN active_public_grant_count > 0 THEN active_public_grant_count
           WHEN is_owner THEN active_owner_grant_count
           ELSE active_matching_grant_count
         END::bigint AS active_grant_count,
         (active_public_grant_count = 0 AND is_owner AND active_owner_grant_count > 0)::bool AS shared_by_me,
         (NOT is_owner AND active_matching_grant_count > 0)::bool AS shared_with_me
  FROM visible_resources
)
SELECT resource_id, owner_user_id, owner_org_id, owner_role, original_name, content_type,
       size_bytes, sha256, storage_uri, storage_path, source_type, resource_kind, source_uri,
       project_id, resource_status AS status, created_at, updated_at, deleted_at, retention_expires_at, metadata,
       share_status, active_grant_count, shared_by_me, shared_with_me
FROM summarized_resources
WHERE (
    $13::text = ''
    OR $13::text = 'all'
    OR share_status = $13
    OR ($13::text = 'shared' AND share_status IN ('shared_by_me', 'shared_with_me', 'public'))
  )
ORDER BY created_at DESC, resource_id ASC
LIMIT $15 OFFSET $16;

-- name: ListResources :many
SELECT *
FROM control_resources
ORDER BY created_at DESC, resource_id ASC
LIMIT $1 OFFSET $2;

-- name: CountResourcesForUser :one
WITH visible_resources AS (
  SELECT r.resource_id, r.original_name, r.content_type, r.sha256, r.source_uri, r.project_id,
         r.source_type, r.resource_kind, r.status AS resource_status, r.metadata,
         (
           r.owner_user_id = $1
           AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $2)
         ) AS is_owner,
         (
           SELECT count(*)::bigint
           FROM control_resource_share_grants g
           WHERE g.resource_id = r.resource_id
             AND g.status = 'active'
         ) AS active_owner_grant_count,
         (
           SELECT count(*)::bigint
           FROM control_resource_share_grants g
           WHERE g.resource_id = r.resource_id
             AND g.status = 'active'
             AND COALESCE(g.grantee_user_id, '') = '__public__'
         ) AS active_public_grant_count,
         (
           SELECT count(*)::bigint
           FROM control_resource_share_grants g
           WHERE g.resource_id = r.resource_id
             AND g.status = 'active'
             AND (
               (
                 COALESCE(g.grantee_user_id, '') <> ''
                 AND g.grantee_user_id = $1
                 AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $2)
               )
               OR (
                 COALESCE(g.grantee_user_id, '') = ''
                 AND COALESCE(g.grantee_org_id, '') <> ''
                 AND g.grantee_org_id = $2
               )
             )
         ) AS active_matching_grant_count
  FROM control_resources r
  LEFT JOIN control_resource_search_documents sd ON sd.resource_id = r.resource_id
  WHERE (
      (
        r.owner_user_id = $1
        AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $2)
      )
      OR (
        r.status = 'active'
        AND EXISTS (
          SELECT 1
          FROM control_resource_share_grants g
          WHERE g.resource_id = r.resource_id
            AND g.status = 'active'
            AND (
              COALESCE(g.grantee_user_id, '') = '__public__'
              OR
              (
                COALESCE(g.grantee_user_id, '') <> ''
                AND g.grantee_user_id = $1
                AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $2)
              )
              OR (
                COALESCE(g.grantee_user_id, '') = ''
                AND COALESCE(g.grantee_org_id, '') <> ''
                AND g.grantee_org_id = $2
              )
            )
        )
      )
    )
    AND r.status = $3
    AND ($4::text = '' OR r.resource_kind = $4)
    AND ($5::text = '' OR r.source_type = $5)
    AND ($6::text = '' OR r.project_id = $6)
    AND (
      $7::text = ''
      OR sd.search_vector @@ plainto_tsquery('simple', $7::text)
      OR lower(COALESCE(sd.search_text, '')) LIKE '%' || lower($7::text) || '%'
    )
    AND (
      cardinality($8::text[]) = 0
      OR COALESCE(r.metadata->'tag_keys', '[]'::jsonb) ?& $8::text[]
    )
    AND (
      cardinality($9::text[]) = 0
      OR NOT EXISTS (
        SELECT 1
        FROM unnest($9::text[]) AS metadata_filters(filter)
        CROSS JOIN LATERAL (
          SELECT split_part(metadata_filters.filter, ':', 1) AS path,
                 split_part(metadata_filters.filter, ':', 2) AS operator,
                 substring(metadata_filters.filter from '^[^:]*:[^:]*:(.*)$') AS expected
        ) mf
        CROSS JOIN LATERAL (
          SELECT r.metadata #> regexp_split_to_array(mf.path, E'\\.') AS actual_json,
                 r.metadata #>> regexp_split_to_array(mf.path, E'\\.') AS actual_text
        ) mv
        WHERE NOT (
          (mf.operator = 'exists' AND mv.actual_json IS NOT NULL)
          OR (mf.operator = 'eq' AND lower(COALESCE(mv.actual_text, '')) = lower(mf.expected))
          OR (mf.operator = 'contains' AND lower(COALESCE(mv.actual_text, mv.actual_json::text, '')) LIKE '%' || lower(mf.expected) || '%')
          OR (mf.operator = 'lt' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric < mf.expected::numeric ELSE false END)
          OR (mf.operator = 'lte' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric <= mf.expected::numeric ELSE false END)
          OR (mf.operator = 'gt' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric > mf.expected::numeric ELSE false END)
          OR (mf.operator = 'gte' AND CASE WHEN mv.actual_text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' AND mf.expected ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN mv.actual_text::numeric >= mf.expected::numeric ELSE false END)
        )
      )
    )
    AND ($10::timestamptz IS NULL OR r.created_at >= $10)
    AND ($11::timestamptz IS NULL OR r.created_at <= $11)
    AND (
      $12::text = ''
      OR $12::text = 'all'
      OR ($12::text = 'caption_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('succeeded', 'completed'))
      OR ($12::text = 'metadata_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('succeeded', 'completed'))
      OR ($12::text = 'tags_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('succeeded', 'completed'))
      OR ($12::text = 'qc_complete' AND lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('succeeded', 'completed'))
      OR ($12::text = 'dedupe_checked' AND lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('succeeded', 'completed'))
      OR ($12::text = 'organization_ready' AND lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('succeeded', 'completed'))
      OR ($12::text = 'data_agent_ready' AND (
        lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('succeeded', 'completed')
        OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('succeeded', 'completed')
        OR lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('succeeded', 'completed')
        OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('succeeded', 'completed')
        OR lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('succeeded', 'completed')
        OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('succeeded', 'completed')
      ))
      OR ($12::text = 'needs_caption' AND lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) NOT IN ('succeeded', 'completed'))
      OR ($12::text = 'needs_metadata' AND lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) NOT IN ('succeeded', 'completed'))
      OR ($12::text = 'data_agent_failed' AND (
        lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,status}', '')) IN ('failed', 'error')
        OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,status}', '')) IN ('failed', 'error')
        OR lower(COALESCE(r.metadata #>> '{data_agent,batch_tag_resources,status}', '')) IN ('failed', 'error')
        OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,status}', '')) IN ('failed', 'error')
        OR lower(COALESCE(r.metadata #>> '{data_agent,deduplicate_resources,status}', '')) IN ('failed', 'error')
        OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,status}', '')) IN ('failed', 'error')
      ))
    )
    AND (
      cardinality($14::text[]) = 0
      OR NOT EXISTS (
        SELECT 1
        FROM unnest($14::text[]) AS descriptor_filters(filter)
        WHERE NOT (
          COALESCE(r.metadata->'tag_keys', '[]'::jsonb) ? lower(descriptor_filters.filter)
          OR lower(COALESCE(r.metadata->>'label', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata->>'descriptor', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata->>'scientific_descriptor', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata->>'diagnosis', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata->>'modality', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata->>'organism', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata->>'species', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,caption}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,caption_resources,summary}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,caption}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,summary}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,label}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,descriptor}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,extract_metadata,scientific_descriptor}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,quality_check_resources,summary}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR lower(COALESCE(r.metadata #>> '{data_agent,organize_resources,summary}', '')) LIKE '%' || lower(descriptor_filters.filter) || '%'
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata->'labels') = 'array' THEN r.metadata->'labels' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata->'descriptors') = 'array' THEN r.metadata->'descriptors' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata->'scientific_descriptors') = 'array' THEN r.metadata->'scientific_descriptors' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata->'diagnoses') = 'array' THEN r.metadata->'diagnoses' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata #> '{data_agent,extract_metadata,labels}') = 'array' THEN r.metadata #> '{data_agent,extract_metadata,labels}' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata #> '{data_agent,extract_metadata,descriptors}') = 'array' THEN r.metadata #> '{data_agent,extract_metadata,descriptors}' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
          OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.metadata #> '{data_agent,extract_metadata,scientific_descriptors}') = 'array' THEN r.metadata #> '{data_agent,extract_metadata,scientific_descriptors}' ELSE '[]'::jsonb END) AS descriptor_values(value)
            WHERE lower(descriptor_values.value) LIKE '%' || lower(descriptor_filters.filter) || '%'
          )
        )
      )
    )
),
summarized_resources AS (
  SELECT resource_id, original_name, content_type, sha256, source_uri, project_id, source_type,
         resource_kind, resource_status, metadata,
         CASE
           WHEN active_public_grant_count > 0 THEN 'public'
           WHEN is_owner AND active_owner_grant_count > 0 THEN 'shared_by_me'
           WHEN NOT is_owner AND active_matching_grant_count > 0 THEN 'shared_with_me'
           ELSE 'private'
         END::text AS share_status
  FROM visible_resources
)
SELECT count(*)::bigint
FROM summarized_resources
WHERE (
    $13::text = ''
    OR $13::text = 'all'
    OR share_status = $13
    OR ($13::text = 'shared' AND share_status IN ('shared_by_me', 'shared_with_me', 'public'))
  );

-- name: SoftDeleteResourceForUser :one
UPDATE control_resources
SET status = 'deleted',
    deleted_at = $4,
    retention_expires_at = $5,
    updated_at = $4
WHERE resource_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
  AND status <> 'deleted'
RETURNING *;

-- name: RestoreResourceForUser :one
UPDATE control_resources
SET status = 'active',
    deleted_at = NULL,
    retention_expires_at = NULL,
    updated_at = $4
WHERE resource_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
RETURNING *;

-- name: ResourceStorageStats :one
SELECT count(*)::bigint AS total_resources, COALESCE(sum(size_bytes), 0)::bigint AS total_bytes
FROM control_resources
WHERE status = 'active';

-- name: CreateResourceEvent :one
INSERT INTO control_resource_events (
  event_id, resource_id, actor_user_id, actor_org_id, event_type, ts, metadata
)
VALUES ($1, $2, $3, $4, $5, $6, $7)
RETURNING *;

-- name: ListResourceEvents :many
SELECT *
FROM control_resource_events
WHERE resource_id = $1
ORDER BY ts DESC, event_id ASC
LIMIT $2;

-- name: ListResourceEventsForUser :many
SELECT e.*
FROM control_resource_events e
JOIN control_resources r ON r.resource_id = e.resource_id
WHERE (
    (
      r.owner_user_id = $1
      AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $2)
    )
    OR (
      r.status = 'active'
      AND EXISTS (
        SELECT 1
        FROM control_resource_share_grants g
        WHERE g.resource_id = r.resource_id
          AND g.status = 'active'
          AND (
            (
              COALESCE(g.grantee_user_id, '') <> ''
              AND g.grantee_user_id = $1
              AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $2)
            )
            OR (
              COALESCE(g.grantee_user_id, '') = ''
              AND COALESCE(g.grantee_org_id, '') <> ''
              AND g.grantee_org_id = $2
            )
          )
      )
    )
  )
  AND ($3::text = '' OR e.resource_id = $3)
  AND ($4::text = '' OR e.event_type = $4)
  AND ($5::text = '' OR e.actor_user_id = $5)
ORDER BY e.ts DESC, e.event_id ASC
LIMIT $6 OFFSET $7;

-- name: CountResourceEventsForUser :one
SELECT COUNT(*)::bigint
FROM control_resource_events e
JOIN control_resources r ON r.resource_id = e.resource_id
WHERE (
    (
      r.owner_user_id = $1
      AND (COALESCE(r.owner_org_id, '') = '' OR r.owner_org_id = $2)
    )
    OR (
      r.status = 'active'
      AND EXISTS (
        SELECT 1
        FROM control_resource_share_grants g
        WHERE g.resource_id = r.resource_id
          AND g.status = 'active'
          AND (
            (
              COALESCE(g.grantee_user_id, '') <> ''
              AND g.grantee_user_id = $1
              AND (COALESCE(g.grantee_org_id, '') = '' OR g.grantee_org_id = $2)
            )
            OR (
              COALESCE(g.grantee_user_id, '') = ''
              AND COALESCE(g.grantee_org_id, '') <> ''
              AND g.grantee_org_id = $2
            )
          )
      )
    )
  )
  AND ($3::text = '' OR e.resource_id = $3)
  AND ($4::text = '' OR e.event_type = $4)
  AND ($5::text = '' OR e.actor_user_id = $5);

-- name: AppendUploadSessionEvent :one
INSERT INTO control_upload_session_events (
  event_id, session_id, actor_user_id, actor_org_id, event_type, ts, metadata
)
VALUES ($1, $2, $3, $4, $5, $6, $7)
RETURNING *;

-- name: ListUploadSessionEvents :many
SELECT *
FROM control_upload_session_events
WHERE session_id = $1
ORDER BY ts DESC, event_id ASC
LIMIT $2;

-- name: CreateUploadSession :one
INSERT INTO control_upload_sessions (
  session_id, owner_user_id, owner_org_id, owner_role, project_id, source_type, status,
  total_bytes, bytes_received, bytes_verified, bytes_committed,
  idempotency_key, browser_fingerprint, error, created_at, updated_at, completed_at, metadata
)
VALUES (
  $1, $2, $3, $4, $5, $6, $7,
  $8, $9, $10, $11,
  $12, $13, $14, $15, $16, $17, $18
)
RETURNING *;

-- name: GetUploadSessionForUser :one
SELECT *
FROM control_upload_sessions
WHERE session_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3);

-- name: GetUploadSessionByIdempotencyKey :one
SELECT *
FROM control_upload_sessions
WHERE owner_user_id = $1
  AND COALESCE(owner_org_id, '') = COALESCE($2::text, '')
  AND idempotency_key = $3
ORDER BY created_at DESC
LIMIT 1;

-- name: UpdateUploadSession :one
UPDATE control_upload_sessions
SET status = $4,
    bytes_received = $5,
    bytes_verified = $6,
    bytes_committed = $7,
    error = $8,
    updated_at = $9,
    completed_at = $10,
    metadata = $11
WHERE session_id = $1
  AND owner_user_id = $2
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $3)
RETURNING *;

-- name: UpsertUploadSessionFile :one
INSERT INTO control_upload_session_files (
  session_id, file_token, resource_id, original_name, relative_path, content_type,
  size_bytes, declared_sha256, computed_sha256, status, error, created_at, updated_at, completed_at, metadata
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
ON CONFLICT (session_id, file_token) DO UPDATE SET
  resource_id = EXCLUDED.resource_id,
  original_name = EXCLUDED.original_name,
  relative_path = EXCLUDED.relative_path,
  content_type = EXCLUDED.content_type,
  size_bytes = EXCLUDED.size_bytes,
  declared_sha256 = EXCLUDED.declared_sha256,
  computed_sha256 = EXCLUDED.computed_sha256,
  status = EXCLUDED.status,
  error = EXCLUDED.error,
  updated_at = EXCLUDED.updated_at,
  completed_at = EXCLUDED.completed_at,
  metadata = EXCLUDED.metadata
RETURNING *;

-- name: ListUploadSessionFiles :many
SELECT *
FROM control_upload_session_files
WHERE session_id = $1
ORDER BY file_token ASC;

-- name: GetUploadSessionFile :one
SELECT *
FROM control_upload_session_files
WHERE session_id = $1
  AND file_token = $2;

-- name: UpsertUploadChunk :one
INSERT INTO control_upload_chunks (
  session_id, file_token, chunk_index, byte_offset, size_bytes, sha256, status,
  storage_uri, received_at, verified_at, error, metadata
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
ON CONFLICT (session_id, file_token, chunk_index) DO UPDATE SET
  byte_offset = EXCLUDED.byte_offset,
  size_bytes = EXCLUDED.size_bytes,
  sha256 = EXCLUDED.sha256,
  status = EXCLUDED.status,
  storage_uri = EXCLUDED.storage_uri,
  received_at = EXCLUDED.received_at,
  verified_at = EXCLUDED.verified_at,
  error = EXCLUDED.error,
  metadata = EXCLUDED.metadata
WHERE control_upload_chunks.status <> 'verified'
   OR (
    control_upload_chunks.byte_offset = EXCLUDED.byte_offset
    AND control_upload_chunks.size_bytes = EXCLUDED.size_bytes
    AND lower(control_upload_chunks.sha256) = lower(EXCLUDED.sha256)
  )
RETURNING *;

-- name: ListUploadChunks :many
SELECT *
FROM control_upload_chunks
WHERE session_id = $1
  AND file_token = $2
ORDER BY chunk_index ASC;

-- name: ListUploadSessionChunks :many
SELECT *
FROM control_upload_chunks
WHERE session_id = $1
ORDER BY file_token ASC, chunk_index ASC;

-- name: GetUploadSessionTotals :one
SELECT
  bytes_received,
  bytes_verified,
  bytes_committed,
  (
    (total_bytes > 0 AND bytes_committed >= total_bytes)
    OR (
      total_bytes = 0
      AND EXISTS (
        SELECT 1 FROM control_upload_session_files
        WHERE control_upload_session_files.session_id = s.session_id
      )
      AND NOT EXISTS (
        SELECT 1 FROM control_upload_session_files
        WHERE control_upload_session_files.session_id = s.session_id
          AND control_upload_session_files.status <> 'completed'
      )
    )
  ) AS all_complete
FROM control_upload_sessions AS s
WHERE s.session_id = $1;

-- name: UploadSessionOperationalMetrics :one
SELECT
  COUNT(*)::bigint AS total,
  COUNT(*) FILTER (WHERE status = 'active')::bigint AS active,
  COUNT(*) FILTER (WHERE status = 'paused')::bigint AS paused,
  COUNT(*) FILTER (WHERE status = 'completed')::bigint AS completed,
  COUNT(*) FILTER (WHERE status = 'failed')::bigint AS failed,
  COUNT(*) FILTER (WHERE status IN ('canceled', 'cancelled'))::bigint AS canceled,
  COUNT(*) FILTER (
    WHERE status NOT IN ('active', 'paused', 'completed', 'failed', 'canceled', 'cancelled')
  )::bigint AS other,
  COALESCE(SUM(total_bytes), 0)::bigint AS bytes_total,
  COALESCE(SUM(bytes_received), 0)::bigint AS bytes_received,
  COALESCE(SUM(bytes_verified), 0)::bigint AS bytes_verified,
  COALESCE(SUM(bytes_committed), 0)::bigint AS bytes_committed
FROM control_upload_sessions;

-- name: FindActiveResourceByShaForUser :one
SELECT *
FROM control_resources
WHERE owner_user_id = $1
  AND (COALESCE(owner_org_id, '') = '' OR owner_org_id = $2)
  AND sha256 = $3
  AND size_bytes = $4
  AND status = 'active'
ORDER BY created_at DESC
LIMIT 1;
