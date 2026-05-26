package latency

import (
	"testing"
	"time"
)

func TestTargetsMatchDesignSpec(t *testing.T) {
	t.Parallel()
	if HealthConfigP95 != 50*time.Millisecond {
		t.Fatalf("HealthConfigP95 = %s", HealthConfigP95)
	}
	if CreateRunAcceptedP95 != 200*time.Millisecond {
		t.Fatalf("CreateRunAcceptedP95 = %s", CreateRunAcceptedP95)
	}
	if FirstVisibleRunEventP95 != 300*time.Millisecond {
		t.Fatalf("FirstVisibleRunEventP95 = %s", FirstVisibleRunEventP95)
	}
	if EventFanoutAfterIngestP95 != 100*time.Millisecond {
		t.Fatalf("EventFanoutAfterIngestP95 = %s", EventFanoutAfterIngestP95)
	}
}
