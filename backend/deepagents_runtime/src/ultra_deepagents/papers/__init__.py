"""Paper ingestion, retrieval, rendering, and typed evidence contracts."""

from .table_evidence import (
    PAPER_TABLE_EVIDENCE_SCHEMA,
    PAPER_TABLE_EVIDENCE_SCHEMA_V1,
    PROMPT_INJECTION_NEUTRALITY,
    ObservationStatus,
    PaperTableEvidence,
    PaperTableEvidenceValidationError,
    seal_paper_table_evidence,
    validate_paper_table_evidence,
)

__all__ = [
    "PAPER_TABLE_EVIDENCE_SCHEMA",
    "PAPER_TABLE_EVIDENCE_SCHEMA_V1",
    "PROMPT_INJECTION_NEUTRALITY",
    "ObservationStatus",
    "PaperTableEvidence",
    "PaperTableEvidenceValidationError",
    "seal_paper_table_evidence",
    "validate_paper_table_evidence",
]
