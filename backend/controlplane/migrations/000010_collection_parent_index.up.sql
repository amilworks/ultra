-- Nested folders: children-by-parent lookups (browsing strips, delete guards)
-- need an index on the self-referencing parent column.
CREATE INDEX IF NOT EXISTS control_resource_collections_parent_idx ON control_resource_collections(parent_collection_id, status);
