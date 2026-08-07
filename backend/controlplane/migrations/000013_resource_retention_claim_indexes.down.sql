DROP INDEX IF EXISTS control_resources_lifecycle_fence_idx;
DROP INDEX IF EXISTS control_resources_retention_blocked_idx;
DROP INDEX IF EXISTS control_resources_purging_claim_idx;
DROP INDEX IF EXISTS control_resources_retention_claim_idx;
DROP INDEX IF EXISTS control_resources_purging_lease_idx;
DROP INDEX IF EXISTS control_resources_retention_expiry_idx;

-- Intentionally preserve control_resource_purge_tombstones. Resource IDs are
-- globally single-use after physical purge; deleting this evidence during a
-- rollback would allow stale publishers to resurrect already-deleted storage.
