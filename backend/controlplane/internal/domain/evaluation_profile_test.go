package domain

import "testing"

func TestParseEvaluationProfileRejectsUnknownProfiles(t *testing.T) {
	t.Parallel()
	if observed, valid := ParseEvaluationProfile(""); !valid || observed != "" {
		t.Fatalf(`ParseEvaluationProfile("") = (%q, %t), want ("", true)`, observed, valid)
	}
	// No profile is supported. materials_cleanroom_v1 is listed explicitly: it
	// was the only member before the materials platform was removed and must
	// not be silently accepted again.
	for _, value := range []string{
		"materials_cleanroom_v1", "materials_cleanroom_v2", "free_form", "knowledge_arm",
	} {
		if observed, valid := ParseEvaluationProfile(value); valid || observed != "" {
			t.Fatalf("ParseEvaluationProfile(%q) = (%q, %t), want rejected", value, observed, valid)
		}
	}
}

func TestEvaluationProfileFromMetadataRejectsUnknownProfiles(t *testing.T) {
	t.Parallel()
	for _, value := range []any{"materials_cleanroom_v1", "free_form", "", 7} {
		metadata := JSONMap{EvaluationProfileMetadataKey: value}
		if profile, ok := EvaluationProfileFromMetadata(metadata); ok || profile != "" {
			t.Fatalf("EvaluationProfileFromMetadata(%v) = (%q, %t), want rejected", value, profile, ok)
		}
	}
}
