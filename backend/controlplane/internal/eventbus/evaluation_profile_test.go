package eventbus

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func TestJobJSONCarriesMaterialsEvaluationAndRemoteCapabilities(t *testing.T) {
	t.Parallel()
	job := Job{
		RunID:                 "run-profiled",
		ThreadID:              "thread-profiled",
		UserID:                "evaluator",
		Goal:                  "evaluate scientific analysis",
		EvaluationProfile:     domain.EvaluationProfileMaterialsCleanroomV1,
		RemoteMutationIntents: []domain.RemoteMutationIntent{domain.RemoteMutationIntentBisqueUpload},
	}
	payload, err := json.Marshal(job)
	if err != nil {
		t.Fatalf("marshal job: %v", err)
	}
	for _, expected := range []string{
		"\"evaluation_profile\":\"materials_cleanroom_v1\"",
		"\"remote_mutation_intents\":[\"bisque.upload\"]",
	} {
		if !strings.Contains(string(payload), expected) {
			t.Fatalf("job JSON missing %s: %s", expected, payload)
		}
	}
	var decoded Job
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("unmarshal job: %v", err)
	}
	if decoded.EvaluationProfile != job.EvaluationProfile ||
		len(decoded.RemoteMutationIntents) != 1 ||
		decoded.RemoteMutationIntents[0] != domain.RemoteMutationIntentBisqueUpload {
		t.Fatalf("decoded job capabilities = %#v", decoded)
	}
}
