package domain

import "strings"

const EvaluationProfileMetadataKey = "evaluation_profile"

// EvaluationProfile is a server-authorized execution contract attached to a
// run. Keep this enum closed: workers may use it to enable capabilities that
// ordinary user runs must never inherit from free-form metadata.
type EvaluationProfile string

const EvaluationProfileMaterialsCleanroomV1 EvaluationProfile = "materials_cleanroom_v1"

// ParseEvaluationProfile accepts the empty value (no protected profile) and
// the explicitly supported profiles only.
func ParseEvaluationProfile(value string) (EvaluationProfile, bool) {
	switch profile := EvaluationProfile(strings.TrimSpace(value)); profile {
	case "":
		return "", true
	case EvaluationProfileMaterialsCleanroomV1:
		return profile, true
	default:
		return "", false
	}
}

// EvaluationProfileFromMetadata returns only a recognized persisted profile.
// Unknown metadata never becomes a worker capability.
func EvaluationProfileFromMetadata(metadata JSONMap) (EvaluationProfile, bool) {
	value, ok := metadata[EvaluationProfileMetadataKey].(string)
	if !ok {
		return "", false
	}
	profile, valid := ParseEvaluationProfile(value)
	return profile, valid && profile != ""
}
