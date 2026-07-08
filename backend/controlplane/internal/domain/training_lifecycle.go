package domain

import (
	"strings"
	"time"
)

// M4 lifecycle inputs (plan sections 8.1-8.3): the promote path re-validates
// the gate INSIDE the store transaction - the server veto - and rollback is
// deliberately ungated.

type PromoteTrainingModelVersionInput struct {
	VersionID      string
	ActorUserID    string
	OverrideReason string
	Now            time.Time
	// RequiredMetricKeysCheck is the cheap Go half of the two-layer metrics
	// enforcement (section 7.4), injected so the store stays schema-agnostic.
	// Nil skips the check (tests only).
	RequiredMetricKeysCheck func(JSONMap) error
}

type RollbackTrainingModelVersionInput struct {
	// VersionID is the CURRENT ACTIVE version being rolled back.
	VersionID string
	// TargetVersionID defaults to the most recently retired version.
	TargetVersionID string
	ActorUserID     string
	Reason          string
}

// TrainingGateFailedError carries the server-veto reasons; handlers map it to
// 409 gate_failed.
type TrainingGateFailedError struct {
	Reasons []string
}

func (e *TrainingGateFailedError) Error() string {
	return "training gate failed: " + strings.Join(e.Reasons, "; ")
}
