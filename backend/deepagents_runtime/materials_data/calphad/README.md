# Curated CALPHAD databases

This directory contains immutable thermodynamic database revisions that Ultra may expose inside
the production scientific sandbox. Each stored file is content-addressed by `manifest.json`;
runtime code must verify the stored hash and byte size before parsing it. The manifest separately
records the retrieved source hash and the declared newline-only normalization so provenance does
not falsely imply byte identity.

The embedded Al-Co-W database is a published, critically reassessed TDB from the NIST Materials
Data Repository. NIST labels the repository item CC0 1.0 Universal. The assessment and its TDB are
the scientific source; inclusion here does not expand the authors' validity claims or make every
pycalphad solve scientifically valid.

User-supplied TDB files stay in the existing tenant-scoped Resource store. PostgreSQL supplies ACLs
and a server-authored resource ID/hash/size binding that is snapshotted onto each run; the worker
rehashes the staged copy before use. The generic Resource row is mutable and is not an append-only
scientific revision ledger, so modified bytes require a new resource ID. The separate CALPHAD ledger
stores immutable resource revisions, parent lineage, server-authorized runtime identity, and retained
inspection/equilibrium evidence bytes; it does not store or reinterpret thermodynamic functions.
Its `input_validated`, `equilibrium_completed`, and technical `promotable` fields are governance
facts, not independent scientific acceptance of an assessment. The TDB remains the authoritative
Gibbs-energy model. Never translate thermodynamic functions into ad hoc relational rows or merge
independently assessed databases without reference-state and model-compatibility review.
Distinct retained equilibrium artifacts append distinct events; exact callback retries are
deduplicated by evidence hash, while a separate request hash groups the scientific request without
claiming deterministic solver replay. Equilibrium lineage additionally requires the same
selection-independent inventory fingerprint as its retained inspection. Shared/read-granted TDBs
remain analyzable but produce explicitly non-promoting read-only artifacts rather than writing to
the owner's validation ledger.

Required provenance for a user database is: source URI or citation, license/use authorization,
content hash, byte size, supplied database/version name, element and phase scope, temperature or
pressure limits, reference-state convention, and any documented extrapolation limits.
