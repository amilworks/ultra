package domain

import "testing"

func TestParseEvaluationProfileAcceptsOnlyMaterialsCleanroom(t *testing.T) {
	t.Parallel()
	for _, expected := range []EvaluationProfile{"", EvaluationProfileMaterialsCleanroomV1} {
		observed, valid := ParseEvaluationProfile(string(expected))
		if !valid || observed != expected {
			t.Fatalf("ParseEvaluationProfile(%q) = (%q, %t)", expected, observed, valid)
		}
	}
	for _, value := range []string{"materials_cleanroom_v2", "free_form", "knowledge_arm"} {
		if observed, valid := ParseEvaluationProfile(value); valid || observed != "" {
			t.Fatalf("ParseEvaluationProfile(%q) = (%q, %t), want rejected", value, observed, valid)
		}
	}
}
