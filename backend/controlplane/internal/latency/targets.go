package latency

import "time"

const (
	HealthConfigP95               = 50 * time.Millisecond
	AuthenticatedLightGETP95      = 100 * time.Millisecond
	CreateThreadP95               = 150 * time.Millisecond
	CreateRunAcceptedP95          = 200 * time.Millisecond
	SSEStreamOpenP95              = 250 * time.Millisecond
	FirstVisibleRunEventP95       = 300 * time.Millisecond
	PythonSupervisorDispatchP95   = 500 * time.Millisecond
	WarmSandboxCommandStartP95    = 250 * time.Millisecond
	WarmSandboxCommandOverheadP95 = 100 * time.Millisecond
	ColdSandboxLeaseP95           = 3 * time.Second
	ArtifactMetadataWriteP95      = 150 * time.Millisecond
	SmallArtifactSignedURLP95     = 100 * time.Millisecond
	CancelSignalAcceptedP95       = 200 * time.Millisecond
	CancelPropagatedP95           = 1 * time.Second
	EventFanoutAfterIngestP95     = 100 * time.Millisecond
	RunListSearchP95              = 200 * time.Millisecond
)
