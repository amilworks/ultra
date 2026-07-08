package store

import (
	"context"
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

// M4 lifecycle transactions, MemoryStore side. Semantics (plan 8.1-8.3):
// candidate->canary and canary->active each re-validate the gate inside the
// mutation; canary->active additionally enforces the held-out override rule,
// demotes the previous active to retired, stamps activated_at, and re-points
// the lineage + status row atomically (one lock). Rollback is ungated.

func (s *MemoryStore) PromoteTrainingModelVersion(ctx context.Context, input domain.PromoteTrainingModelVersionInput) (domain.TrainingModelVersionRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	versionID := strings.TrimSpace(input.VersionID)
	version, ok := s.training.versions[versionID]
	if !ok {
		return domain.TrainingModelVersionRecord{}, ErrNotFound
	}
	var fromStatus, toStatus string
	switch version.Status {
	case "candidate":
		fromStatus, toStatus = "candidate", "canary"
	case "canary":
		fromStatus, toStatus = "canary", "active"
	default:
		return domain.TrainingModelVersionRecord{}, ErrConflict
	}

	gold, goldOK := s.currentFrozenGoldLocked(version.ModelKey)
	reasons := s.promoteGateReasonsLocked(version, gold, goldOK, input)
	heldOutPending := goldOK && goldHeldOutState(gold) == "pending_new_survey"
	if toStatus == "active" && heldOutPending && strings.TrimSpace(input.OverrideReason) == "" {
		reasons = append(reasons,
			"held-out slice is pending new survey data - promoting to active requires an audited override_reason")
	}
	if len(reasons) > 0 {
		return domain.TrainingModelVersionRecord{}, &domain.TrainingGateFailedError{Reasons: reasons}
	}

	now := input.Now
	if now.IsZero() {
		now = domain.Now()
	}
	if toStatus == "active" {
		for id, row := range s.training.versions {
			if row.ModelKey == version.ModelKey && row.Status == "active" && id != versionID {
				row.Status = "retired"
				row.UpdatedAt = now
				s.training.versions[id] = row
				s.appendTrainingVersionEventLocked(id, "retired", input.ActorUserID, "active", "retired", "", now)
			}
		}
		version.ActivatedAt = &now
		s.repointTrainingLineageLocked(version.ModelKey, versionID, now)
	}
	version.Status = toStatus
	version.UpdatedAt = now
	s.training.versions[versionID] = version

	eventType := "promoted_" + toStatus
	if toStatus == "active" && heldOutPending {
		eventType = "promote_with_pending_held_out"
	}
	goldHash := ""
	if goldOK {
		goldHash = gold.ContentHash
	}
	s.appendTrainingVersionEventLocked(versionID, eventType, input.ActorUserID, fromStatus, toStatus, strings.TrimSpace(input.OverrideReason), now)
	_ = goldHash
	return cloneTrainingVersion(version), nil
}

func (s *MemoryStore) RollbackTrainingModelVersion(ctx context.Context, input domain.RollbackTrainingModelVersionInput) (domain.TrainingModelVersionRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	activeID := strings.TrimSpace(input.VersionID)
	active, ok := s.training.versions[activeID]
	if !ok {
		return domain.TrainingModelVersionRecord{}, ErrNotFound
	}
	if active.Status != "active" {
		return domain.TrainingModelVersionRecord{}, ErrConflict
	}
	targetID := strings.TrimSpace(input.TargetVersionID)
	if targetID == "" {
		var newest domain.TrainingModelVersionRecord
		for _, row := range s.training.versions {
			if row.ModelKey == active.ModelKey && row.Status == "retired" {
				if newest.VersionID == "" || row.UpdatedAt.After(newest.UpdatedAt) {
					newest = row
				}
			}
		}
		targetID = newest.VersionID
	}
	if targetID == "" {
		return domain.TrainingModelVersionRecord{}, ErrNotFound
	}
	target, ok := s.training.versions[targetID]
	if !ok || target.ModelKey != active.ModelKey {
		return domain.TrainingModelVersionRecord{}, ErrNotFound
	}

	now := domain.Now()
	active.Status = "retired"
	active.UpdatedAt = now
	s.training.versions[activeID] = active
	target.Status = "active"
	target.ActivatedAt = &now
	target.UpdatedAt = now
	s.training.versions[targetID] = target
	s.repointTrainingLineageLocked(target.ModelKey, targetID, now)
	// NO benchmark required: returning to a known-good pinned baseline. The
	// gate catches forgetting; rollback undoes it; rollback is never blocked.
	s.appendTrainingVersionEventLocked(activeID, "rolled_back", input.ActorUserID, "active", "retired", strings.TrimSpace(input.Reason), now)
	s.appendTrainingVersionEventLocked(targetID, "restored_active", input.ActorUserID, "retired", "active", strings.TrimSpace(input.Reason), now)
	return cloneTrainingVersion(target), nil
}

// --- locked helpers ----------------------------------------------------------

func (s *MemoryStore) currentFrozenGoldLocked(modelKey string) (domain.TrainingGoldSetRecord, bool) {
	var current domain.TrainingGoldSetRecord
	found := false
	for _, gold := range s.training.goldSets {
		if gold.ModelKey != modelKey || gold.Status != "frozen" {
			continue
		}
		if !found || gold.Version > current.Version {
			current = gold
			found = true
		}
	}
	return current, found
}

func goldHeldOutState(gold domain.TrainingGoldSetRecord) string {
	if gold.Provenance == nil {
		return ""
	}
	state, _ := gold.Provenance["held_out_state"].(string)
	return state
}

// promoteGateReasonsLocked is the server veto (section 8.1): a passing
// benchmark run on the CURRENT gold hash + the version's own
// promotion_benchmark_ready stamp + the Go required-keys check. Re-checking
// the HASH inside the mutation defends against a stale-passing candidate
// sneaking through after gold changed.
func (s *MemoryStore) promoteGateReasonsLocked(
	version domain.TrainingModelVersionRecord,
	gold domain.TrainingGoldSetRecord,
	goldOK bool,
	input domain.PromoteTrainingModelVersionInput,
) []string {
	reasons := []string{}
	if !goldOK {
		return append(reasons, "no frozen gold set exists - nothing can be promoted without the gate")
	}
	var latest domain.TrainingBenchmarkRunRecord
	found := false
	for _, run := range s.training.benchmarkRuns[version.VersionID] {
		if run.GoldSetContentHash != gold.ContentHash {
			continue
		}
		if !found || !run.CreatedAt.Before(latest.CreatedAt) {
			latest = run
			found = true
		}
	}
	if !found {
		return append(reasons, "no benchmark run on the current gold content hash - benchmark the version first (section 7.5)")
	}
	if !latest.GuardrailsPassed {
		reasons = append(reasons, "the latest benchmark on the current gold hash did not pass the gate")
		reasons = append(reasons, latest.GuardrailsReasons...)
	}
	if ready, _ := version.Metrics["promotion_benchmark_ready"].(bool); !ready {
		reasons = append(reasons, "promotion_benchmark_ready is not set on the version metrics")
	}
	if input.RequiredMetricKeysCheck != nil {
		if err := input.RequiredMetricKeysCheck(latest.Metrics); err != nil {
			reasons = append(reasons, "metrics blob failed the required-keys check: "+err.Error())
		}
	}
	return reasons
}

func (s *MemoryStore) appendTrainingVersionEventLocked(versionID, eventType, actor, fromStatus, toStatus, reason string, now time.Time) {
	s.training.versionEvents[versionID] = append(s.training.versionEvents[versionID], domain.TrainingModelVersionEventRecord{
		EventID:     domain.NewID("training_version_event"),
		VersionID:   versionID,
		EventType:   eventType,
		ActorUserID: actor,
		FromStatus:  fromStatus,
		ToStatus:    toStatus,
		Reason:      reason,
		CreatedAt:   now,
	})
}

func (s *MemoryStore) repointTrainingLineageLocked(modelKey, versionID string, now time.Time) {
	for id, lineage := range s.training.lineages {
		if lineage.ModelKey == modelKey && lineage.Scope == "shared" {
			lineage.ActiveVersionID = versionID
			lineage.UpdatedAt = now
			s.training.lineages[id] = lineage
		}
	}
	if status, ok := s.training.statuses[modelKey]; ok {
		status.ActiveModelVersion = versionID
		s.training.statuses[modelKey] = status
	}
}
