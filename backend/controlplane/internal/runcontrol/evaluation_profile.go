package runcontrol

import (
	"errors"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

var ErrInvalidEvaluationProfile = errors.New("invalid evaluation profile")

func storedEvaluationProfile(run domain.RunRecord) domain.EvaluationProfile {
	profile, _ := domain.EvaluationProfileFromMetadata(run.Metadata)
	return profile
}

func storedEvaluationProfileMatches(run domain.RunRecord, requested domain.EvaluationProfile) bool {
	return storedEvaluationProfile(run) == requested
}

// metadataWithStoredEvaluationProfile makes the immutable run record the only
// authority for a protected profile on queued and retried jobs.
func metadataWithStoredEvaluationProfile(run domain.RunRecord, metadata domain.JSONMap) domain.JSONMap {
	metadata = cloneMap(metadata)
	delete(metadata, domain.EvaluationProfileMetadataKey)
	if profile := storedEvaluationProfile(run); profile != "" {
		metadata[domain.EvaluationProfileMetadataKey] = string(profile)
	}
	return metadata
}

func attestEvaluationProfile(payload domain.JSONMap, run domain.RunRecord) {
	delete(payload, domain.EvaluationProfileMetadataKey)
	if profile := storedEvaluationProfile(run); profile != "" {
		payload[domain.EvaluationProfileMetadataKey] = string(profile)
	}
}
