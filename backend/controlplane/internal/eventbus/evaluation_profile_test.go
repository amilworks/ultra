package eventbus

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func TestJobJSONCarriesEvaluationAndRemoteCapabilities(t *testing.T) {
	t.Parallel()
	// No evaluation profile is currently supported, so a local literal keeps the
	// field's wire shape covered without depending on a named profile.
	profile := domain.EvaluationProfile("probe_profile_v1")
	job := Job{
		RunID:                 "run-profiled",
		ThreadID:              "thread-profiled",
		UserID:                "evaluator",
		Goal:                  "evaluate scientific analysis",
		EvaluationProfile:     profile,
		RemoteMutationIntents: []domain.RemoteMutationIntent{domain.RemoteMutationIntentBisqueUpload},
	}
	payload, err := json.Marshal(job)
	if err != nil {
		t.Fatalf("marshal job: %v", err)
	}
	for _, expected := range []string{
		"\"evaluation_profile\":\"probe_profile_v1\"",
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

func TestJobJSONOmitsEmptyEvaluationProfile(t *testing.T) {
	t.Parallel()
	payload, err := json.Marshal(Job{RunID: "run-plain", ThreadID: "thread-plain", UserID: "user"})
	if err != nil {
		t.Fatalf("marshal job: %v", err)
	}
	if strings.Contains(string(payload), "evaluation_profile") {
		t.Fatalf("job JSON should omit an empty evaluation profile: %s", payload)
	}
}
