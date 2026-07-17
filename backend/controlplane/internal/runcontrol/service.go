package runcontrol

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

type Store interface {
	CreateThread(context.Context, domain.CreateThreadInput) (domain.ThreadRecord, error)
	UpdateThreadForUser(context.Context, domain.UpdateThreadInput) (domain.ThreadRecord, error)
	SoftDeleteThreadForUser(context.Context, string, string, time.Time) (domain.ThreadRecord, error)
	ApplyGeneratedThreadTitle(context.Context, domain.ApplyGeneratedThreadTitleInput) (domain.ThreadRecord, error)
	GetThread(context.Context, string) (domain.ThreadRecord, error)
	GetThreadForUser(context.Context, string, string) (domain.ThreadRecord, error)
	ListThreads(context.Context, int, int, string) (domain.ThreadListPage, error)
	ListThreadsForUser(context.Context, string, int, int, string) (domain.ThreadListPage, error)
	ListThreadMessages(context.Context, string) ([]domain.ThreadMessage, error)
	ListThreadMessagesForUser(context.Context, string, string) ([]domain.ThreadMessage, error)
	ListThreadMessagePageForUser(context.Context, string, string, string, int) ([]domain.ThreadMessage, bool, error)
	AppendThreadMessage(context.Context, domain.ThreadMessage) (domain.ThreadMessage, error)
	CreateRun(context.Context, domain.CreateRunInput) (domain.RunRecord, error)
	FindRunByIdempotencyKey(context.Context, string, string, string) (domain.RunRecord, bool, error)
	GetRun(context.Context, string) (domain.RunRecord, error)
	GetRunForUser(context.Context, string, string) (domain.RunRecord, error)
	ListRuns(context.Context, string, string, int, int) ([]domain.RunRecord, error)
	ListRunsForUser(context.Context, string, string, string, int, int) ([]domain.RunRecord, error)
	UpdateRunStatus(context.Context, string, domain.RunStatus, string, string) (domain.RunRecord, error)
	CompleteRun(context.Context, domain.CompleteRunInput) (domain.RunRecord, error)
	RecordUserTokenUsage(context.Context, domain.RecordUserTokenUsageInput) error
	RecordRunTokenUsage(context.Context, domain.RecordRunTokenUsageInput) (domain.RunTokenUsageRecord, bool, error)
	FinalizeRunTokenUsage(context.Context, domain.FinalizeRunTokenUsageInput) (domain.RunTokenUsageSummary, bool, error)
	GetRunLease(context.Context, string) (domain.RunLeaseRecord, bool, error)
	AcquireRunLease(context.Context, domain.AcquireRunLeaseInput) (domain.RunLeaseRecord, error)
	RenewRunLease(context.Context, domain.RenewRunLeaseInput) (domain.RunLeaseRecord, error)
	ReleaseRunLease(context.Context, domain.ReleaseRunLeaseInput) error
	ClearRunLease(context.Context, string) (domain.RunLeaseRecord, bool, error)
	AppendRunEvent(context.Context, domain.AppendRunEventInput) (domain.RunEventRecord, error)
	GetRunEvent(context.Context, string) (domain.RunEventRecord, bool, error)
	ListRunEvents(context.Context, string, int) ([]domain.RunEventRecord, error)
	ListRunEventsForUser(context.Context, string, string, int) ([]domain.RunEventRecord, error)
	ListRunEventsAfter(context.Context, string, int64, int) ([]domain.RunEventRecord, error)
	ListRunEventsAfterForUser(context.Context, string, string, int64, int) ([]domain.RunEventRecord, error)
	CreateArtifact(context.Context, domain.CreateArtifactInput) (domain.ArtifactRecord, error)
	ListRunArtifacts(context.Context, string, int) ([]domain.ArtifactRecord, error)
	ListRunArtifactsForUser(context.Context, string, string, int) ([]domain.ArtifactRecord, error)
	GetArtifact(context.Context, string) (domain.ArtifactRecord, error)
	GetArtifactForUser(context.Context, string, string) (domain.ArtifactRecord, error)
}

type runDispatchMarker interface {
	MarkRunDispatched(context.Context, string, time.Time) (domain.RunRecord, error)
}

// activeRunEventAppender is the single-statement ingest fast path: dedupe by
// event ID (returning the stored record), append when the run is live, or
// drop when it is missing or terminal. Stores that do not implement it fall
// back to the read-then-append legacy path.
type activeRunEventAppender interface {
	AppendRunEventIfRunActive(context.Context, domain.AppendRunEventInput) (domain.RunEventRecord, store.RunEventAppendOutcome, error)
}

type runEventSourceSequenceReader interface {
	GetRunEventBySourceSequence(context.Context, string, int64) (domain.RunEventRecord, bool, error)
}

var ErrRunEventPredecessorPending = errors.New("run event predecessor pending")

// ErrRunEventFanoutUnavailable wraps a failure to publish an already-stored
// event to the local fanout bus. The event is durably in the store, so the
// retry is cheap (dedup replays it) and the failure is transient by nature.
var ErrRunEventFanoutUnavailable = errors.New("run event fanout unavailable")

type workerHeartbeatReader interface {
	GetWorkerHeartbeat(context.Context, string) (domain.WorkerHeartbeatRecord, bool, error)
}

// heartbeatStatusWriteInterval bounds how often a run.heartbeat event is
// allowed to write run status to the store. Heartbeats only reassert
// "running"; worker liveness is tracked by leases, so coalescing these
// writes loses nothing except sub-15s freshness of runs.updated_at.
const heartbeatStatusWriteInterval = 15 * time.Second

type Service struct {
	store            Store
	bus              eventbus.Bus
	now              func() time.Time
	runtimeFacts     RuntimeFactsConfig
	idempotencyLocks [64]sync.Mutex
	eventIDLocks     [128]sync.Mutex

	heartbeatMu           sync.Mutex
	heartbeatStatusWrites map[string]time.Time
}

type ServiceOptions struct {
	Now          func() time.Time
	RuntimeFacts RuntimeFactsConfig
}

type RuntimeFactsConfig struct {
	ProductName         string
	AppName             string
	AppVersion          string
	Environment         string
	PublicURL           string
	DefaultUserTimezone string
}

type CreateThreadRequest struct {
	UserID          string
	Title           string
	Metadata        domain.JSONMap
	InitialMessages []domain.ThreadMessage
}

type CreateRunRequest struct {
	ThreadID              string
	UserID                string
	Goal                  string
	EvaluationProfile     domain.EvaluationProfile
	RemoteMutationIntents []domain.RemoteMutationIntent
	Messages              []domain.ThreadMessage
	FileIDs               []string
	ResourceURIs          []string
	DatasetURIs           []string
	SelectedToolNames     []string
	KnowledgeContext      domain.JSONMap
	WorkflowHint          domain.JSONMap
	SelectionContext      domain.JSONMap
	ReasoningMode         string
	Budgets               domain.JSONMap
	Benchmark             domain.JSONMap
	ResourceDescriptors   []domain.JSONMap
	IdempotencyKey        string
	Metadata              domain.JSONMap
	JobMetadata           domain.JSONMap
}

type CancelRunRequest struct {
	RunID    string
	Reason   string
	Metadata domain.JSONMap
}

type RequeueRunRequest struct {
	RunID    string
	Reason   string
	Metadata domain.JSONMap
}

type RecoverExpiredRunLeasesRequest struct {
	Now    time.Time
	Reason string
	Limit  int
	// RedispatchQueuedAfter bounds how long a queued run may sit without any
	// lease before its job is considered lost and re-dispatched. Zero applies
	// the default.
	RedispatchQueuedAfter time.Duration
	// WorkerHeartbeatStaleAfter bounds how long an unexpired run lease may go
	// without both lease renewal and owner heartbeat before recovery treats the
	// owner as gone. Zero applies the default.
	WorkerHeartbeatStaleAfter time.Duration
}

// defaultRedispatchQueuedAfter is how long a queued run may wait for a worker
// claim before recovery re-publishes its job. It must comfortably exceed the
// worst-case queue wait of a healthy worker pool to avoid duplicate dispatch.
const defaultRedispatchQueuedAfter = 2 * time.Minute

// defaultWorkerHeartbeatStaleAfter is intentionally longer than the Deep
// Agents busy heartbeat/lease-renewal cadence, but much shorter than the
// control-plane lease TTL so crashed workers do not pin runs for ten minutes.
const defaultWorkerHeartbeatStaleAfter = 2 * time.Minute

// zombieRunRequeueAfter is how long a running/waiting run may exist WITHOUT a
// lease before recovery reclaims it. Generous on purpose: it must exceed the
// job redelivery horizon (worker ack_wait 300s plus several delayed naks), and
// the recent-progress guard already vetoes any run whose worker is emitting.
// Too-short a grace double-executes expensive GPU runs.
const zombieRunRequeueAfter = 15 * time.Minute

type RecoverExpiredRunLeasesResult struct {
	Checked      int
	RequeuedRuns []domain.RunRecord
}

type AcquireRunLeaseRequest struct {
	RunID    string
	WorkerID string
	TTL      time.Duration
}

type RenewRunLeaseRequest struct {
	RunID      string
	LeaseToken string
	TTL        time.Duration
	// Now overrides the clock for lease-expiry evaluation. Zero uses the
	// store's real clock. Threaded (like AcquireRunLease) so lease renewal is
	// deterministically testable.
	Now time.Time
}

type ReleaseRunLeaseRequest struct {
	RunID      string
	LeaseToken string
}

func NewService(store Store, bus eventbus.Bus) *Service {
	return NewServiceWithOptions(store, bus, ServiceOptions{})
}

func NewServiceWithOptions(store Store, bus eventbus.Bus, opts ServiceOptions) *Service {
	now := opts.Now
	if now == nil {
		now = domain.Now
	}
	return &Service{
		store:                 store,
		bus:                   bus,
		now:                   now,
		runtimeFacts:          normalizeRuntimeFactsConfig(opts.RuntimeFacts),
		heartbeatStatusWrites: map[string]time.Time{},
	}
}

func (s *Service) CreateThread(ctx context.Context, req CreateThreadRequest) (domain.ThreadRecord, error) {
	return s.store.CreateThread(ctx, domain.CreateThreadInput{
		UserID:          req.UserID,
		Title:           req.Title,
		Metadata:        req.Metadata,
		InitialMessages: req.InitialMessages,
	})
}

func (s *Service) CreateRun(ctx context.Context, req CreateRunRequest) (domain.RunRecord, error) {
	evaluationProfile, valid := domain.ParseEvaluationProfile(string(req.EvaluationProfile))
	if !valid {
		return domain.RunRecord{}, ErrInvalidEvaluationProfile
	}
	req.EvaluationProfile = evaluationProfile
	remoteMutationIntents, valid := domain.ParseRemoteMutationIntents(
		domain.RemoteMutationIntentStrings(req.RemoteMutationIntents),
	)
	if !valid {
		return domain.RunRecord{}, ErrInvalidRemoteMutationIntent
	}
	if evaluationProfile != "" && len(remoteMutationIntents) > 0 {
		return domain.RunRecord{}, ErrEvaluationProfileMutation
	}
	req.RemoteMutationIntents = remoteMutationIntents
	// Artifact descriptors are capabilities over the shared artifact store. No
	// caller (including an internal caller that bypasses HTTP) may mint one in a
	// CreateRunRequest. Server-resolved prior artifacts are merged only after the
	// thread and owning user have been established below. A missing type is also
	// rejected because older runtimes interpreted it as an implicit artifact.
	req.ResourceDescriptors = withoutCallerArtifactDescriptors(req.ResourceDescriptors)
	metadata := buildRunMetadata(req)
	idempotencyKey := normalizedIdempotencyKey(req, metadata)
	if idempotencyKey != "" {
		lock := s.idempotencyLock(req.ThreadID, req.UserID, idempotencyKey)
		lock.Lock()
		defer lock.Unlock()
		existing, found, err := s.store.FindRunByIdempotencyKey(ctx, req.ThreadID, req.UserID, idempotencyKey)
		if err != nil {
			return domain.RunRecord{}, err
		}
		if found {
			if !storedEvaluationProfileMatches(existing, req.EvaluationProfile) {
				return domain.RunRecord{}, store.ErrConflict
			}
			if !storedRemoteMutationIntentsMatch(existing, req.RemoteMutationIntents) {
				return domain.RunRecord{}, store.ErrConflict
			}
			reconciled, err := s.reconcileStoredTerminalEvent(ctx, existing)
			if err != nil {
				return domain.RunRecord{}, err
			}
			if reconciled.Status == domain.RunStatusQueued {
				return s.recoverQueuedRunDispatch(ctx, reconciled)
			}
			return reconciled, nil
		}
		metadata["idempotency_key"] = idempotencyKey
	}
	s.stampRuntimeFacts(metadata)
	workflowKind := workflowKindForRun(req, metadata)
	thread, err := s.store.GetThread(ctx, req.ThreadID)
	if err != nil {
		return domain.RunRecord{}, err
	}
	existingMessages, err := s.store.ListThreadMessages(ctx, req.ThreadID)
	if err != nil {
		return domain.RunRecord{}, err
	}
	priorResourceDescriptors, err := s.priorArtifactResourceDescriptors(
		ctx, req.UserID, existingMessages,
	)
	if err != nil {
		return domain.RunRecord{}, err
	}
	resourceDescriptors := mergeResourceDescriptors(req.ResourceDescriptors, priorResourceDescriptors)
	if len(resourceDescriptors) > 0 {
		metadata["resource_descriptors"] = copyJSONMaps(resourceDescriptors)
	}
	internalRun := isInternalToolRunRequest(req)
	if internalRun {
		metadata["internal"] = true
		metadata["visible_in_thread"] = false
	}
	messagesToAppend := assignPriorAssistantRunIDs(
		transcriptSuffixToAppend(existingMessages, req.Messages),
		thread.LatestRunID,
	)
	if internalRun {
		messagesToAppend = nil
	}
	run, err := s.store.CreateRun(ctx, domain.CreateRunInput{
		ThreadID:     req.ThreadID,
		UserID:       req.UserID,
		Goal:         req.Goal,
		WorkflowKind: workflowKind,
		Mode:         "durable",
		Messages:     messagesToAppend,
		Metadata:     metadata,
		Internal:     internalRun,
	})
	if err != nil {
		if idempotencyKey != "" && errors.Is(err, store.ErrConflict) {
			return s.recoverConflictingIdempotentRun(ctx, req, idempotencyKey)
		}
		return domain.RunRecord{}, err
	}
	event, err := s.appendAcceptedRunEvent(ctx, run, false)
	if err != nil {
		return domain.RunRecord{}, err
	}
	job := jobForRun(run, req, resourceDescriptors)
	if err := s.bus.PublishJob(ctx, job); err != nil {
		return s.markRunDispatchFailed(ctx, run, err)
	}
	run = s.markRunJobDispatched(ctx, run)
	_ = s.bus.PublishRunEvent(ctx, event)
	return run, nil
}

func jobForRun(run domain.RunRecord, req CreateRunRequest, resourceDescriptors []domain.JSONMap) eventbus.Job {
	return eventbus.Job{
		RunID:                 run.RunID,
		ThreadID:              run.ThreadID,
		UserID:                run.UserID,
		Goal:                  run.Goal,
		WorkflowKind:          run.WorkflowKind,
		EvaluationProfile:     storedEvaluationProfile(run),
		RemoteMutationIntents: append([]domain.RemoteMutationIntent(nil), req.RemoteMutationIntents...),
		Messages:              copyMessages(req.Messages),
		FileIDs:               copyStrings(req.FileIDs),
		ResourceURIs:          copyStrings(req.ResourceURIs),
		DatasetURIs:           copyStrings(req.DatasetURIs),
		SelectedToolNames:     copyStrings(req.SelectedToolNames),
		KnowledgeContext:      cloneMap(req.KnowledgeContext),
		WorkflowHint:          cloneMap(req.WorkflowHint),
		SelectionContext:      cloneMap(req.SelectionContext),
		ReasoningMode:         req.ReasoningMode,
		Budgets:               cloneMap(req.Budgets),
		Benchmark:             cloneMap(req.Benchmark),
		ResourceDescriptors:   copyJSONMaps(resourceDescriptors),
		Metadata:              mergeJobMetadata(metadataWithStoredEvaluationProfile(run, run.Metadata), req.JobMetadata),
	}
}

func (s *Service) recoverConflictingIdempotentRun(ctx context.Context, req CreateRunRequest, idempotencyKey string) (domain.RunRecord, error) {
	existing, found, err := s.store.FindRunByIdempotencyKey(ctx, req.ThreadID, req.UserID, idempotencyKey)
	if err != nil {
		return domain.RunRecord{}, err
	}
	if !found {
		return domain.RunRecord{}, store.ErrConflict
	}
	if !storedEvaluationProfileMatches(existing, req.EvaluationProfile) {
		return domain.RunRecord{}, store.ErrConflict
	}
	if !storedRemoteMutationIntentsMatch(existing, req.RemoteMutationIntents) {
		return domain.RunRecord{}, store.ErrConflict
	}
	reconciled, err := s.reconcileStoredTerminalEvent(ctx, existing)
	if err != nil {
		return domain.RunRecord{}, err
	}
	if reconciled.Status == domain.RunStatusQueued {
		return s.recoverQueuedRunDispatch(ctx, reconciled)
	}
	return reconciled, nil
}

func jobForStoredRun(run domain.RunRecord, messages []domain.ThreadMessage, metadata domain.JSONMap, dispatchID string) eventbus.Job {
	metadata = metadataWithStoredEvaluationProfile(run, metadata)
	return eventbus.Job{
		RunID:                 run.RunID,
		DispatchID:            dispatchID,
		ThreadID:              run.ThreadID,
		UserID:                run.UserID,
		Goal:                  run.Goal,
		WorkflowKind:          run.WorkflowKind,
		EvaluationProfile:     storedEvaluationProfile(run),
		RemoteMutationIntents: storedRemoteMutationIntents(run),
		Messages:              copyMessages(messages),
		FileIDs:               metadataStringSlice(metadata["file_ids"]),
		ResourceURIs:          metadataStringSlice(metadata["resource_uris"]),
		DatasetURIs:           metadataStringSlice(metadata["dataset_uris"]),
		SelectedToolNames:     metadataStringSlice(metadata["selected_tool_names"]),
		KnowledgeContext:      metadataJSONMap(metadata["knowledge_context"]),
		WorkflowHint:          metadataJSONMap(metadata["workflow_hint"]),
		SelectionContext:      metadataJSONMap(metadata["selection_context"]),
		ReasoningMode:         strings.TrimSpace(anyString(metadata["reasoning_mode"])),
		Budgets:               metadataJSONMap(metadata["budgets"]),
		Benchmark:             metadataJSONMap(metadata["benchmark"]),
		ResourceDescriptors:   metadataResourceDescriptors(metadata),
		Metadata:              metadata,
	}
}

func (s *Service) recoverQueuedRunDispatch(ctx context.Context, run domain.RunRecord) (domain.RunRecord, error) {
	if runJobDispatched(run) {
		return run, nil
	}
	acceptedEvent, found, err := s.store.GetRunEvent(ctx, acceptedRunEventID(run.RunID))
	if err != nil {
		return domain.RunRecord{}, err
	}
	if !found {
		events, err := s.store.ListRunEvents(ctx, run.RunID, 1000)
		if err != nil {
			return domain.RunRecord{}, err
		}
		for _, event := range events {
			if event.EventKind == "run.accepted" {
				acceptedEvent = event
				found = true
				break
			}
		}
	}
	if !found {
		acceptedEvent, err = s.appendAcceptedRunEvent(ctx, run, true)
		if err != nil {
			return domain.RunRecord{}, err
		}
	}
	messages, err := s.store.ListThreadMessages(ctx, run.ThreadID)
	if err != nil {
		return domain.RunRecord{}, err
	}
	job := jobForStoredRun(
		run,
		messagesForRunRequeue(messages, run),
		cloneMap(run.Metadata),
		"",
	)
	if err := s.bus.PublishJob(ctx, job); err != nil {
		return s.markRunDispatchFailed(ctx, run, err)
	}
	run = s.markRunJobDispatched(ctx, run)
	_ = s.bus.PublishRunEvent(ctx, acceptedEvent)
	return run, nil
}

func mergeJobMetadata(metadata domain.JSONMap, jobMetadata domain.JSONMap) domain.JSONMap {
	merged := cloneMap(metadata)
	for key, value := range jobMetadata {
		if key == "runtime_facts" || key == domain.EvaluationProfileMetadataKey ||
			key == domain.RemoteMutationIntentsMetadataKey ||
			key == domain.BisqueAccountBindingMetadataKey {
			continue
		}
		merged[key] = value
	}
	return merged
}

func (s *Service) appendAcceptedRunEvent(ctx context.Context, run domain.RunRecord, recoveredDispatch bool) (domain.RunEventRecord, error) {
	eventID := acceptedRunEventID(run.RunID)
	if existing, found, err := s.store.GetRunEvent(ctx, eventID); err != nil {
		return domain.RunEventRecord{}, err
	} else if found {
		return existing, nil
	}
	payload := domain.JSONMap{"status": string(run.Status), "workflow_kind": run.WorkflowKind}
	attestEvaluationProfile(payload, run)
	if recoveredDispatch {
		payload["recovered_dispatch"] = true
	}
	event, err := s.store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   eventID,
		RunID:     run.RunID,
		ThreadID:  run.ThreadID,
		EventKind: "run.accepted",
		Message:   "Run accepted.",
		Payload:   payload,
	})
	if err == nil {
		return event, nil
	}
	if errors.Is(err, store.ErrConflict) {
		if existing, found, getErr := s.store.GetRunEvent(ctx, eventID); getErr != nil {
			return domain.RunEventRecord{}, getErr
		} else if found {
			return existing, nil
		}
	}
	return domain.RunEventRecord{}, err
}

func acceptedRunEventID(runID string) string {
	return "evt_" + strings.TrimSpace(runID) + "_accepted"
}

func runJobDispatched(run domain.RunRecord) bool {
	if run.Metadata == nil {
		return false
	}
	switch value := run.Metadata["job_dispatched_at"].(type) {
	case string:
		return strings.TrimSpace(value) != ""
	case time.Time:
		return !value.IsZero()
	case bool:
		return value
	default:
		return value != nil
	}
}

func (s *Service) markRunJobDispatched(ctx context.Context, run domain.RunRecord) domain.RunRecord {
	marker, ok := s.store.(runDispatchMarker)
	if !ok {
		return run
	}
	marked, err := marker.MarkRunDispatched(ctx, run.RunID, domain.Now())
	if err != nil {
		return run
	}
	return marked
}

func (s *Service) CancelRun(ctx context.Context, req CancelRunRequest) (domain.RunRecord, error) {
	existing, err := s.store.GetRun(ctx, req.RunID)
	if err != nil {
		return domain.RunRecord{}, err
	}
	if isTerminalRunStatus(existing.Status) {
		return existing, nil
	}

	cancelEventInput := domain.AppendRunEventInput{
		EventID:   canceledRunEventID(existing.RunID),
		RunID:     existing.RunID,
		ThreadID:  existing.ThreadID,
		EventKind: "run.canceled",
		EventType: "run",
		Level:     "info",
		Message:   "Run canceled.",
		Payload:   domain.JSONMap{"reason": req.Reason},
	}
	event, found, err := s.store.GetRunEvent(ctx, cancelEventInput.EventID)
	if err != nil {
		return domain.RunRecord{}, err
	}
	if !found {
		event, err = s.store.AppendRunEvent(ctx, cancelEventInput)
		if err != nil {
			return domain.RunRecord{}, err
		}
	}
	if err := s.applyRunEventSideEffects(ctx, appendInputFromEventRecord(event)); err != nil {
		return domain.RunRecord{}, err
	}
	run, err := s.store.GetRun(ctx, req.RunID)
	if err != nil {
		return domain.RunRecord{}, err
	}
	_ = s.bus.PublishRunEvent(ctx, event)
	_ = s.bus.PublishCancel(ctx, eventbus.CancelSignal{
		RunID:    existing.RunID,
		ThreadID: existing.ThreadID,
		UserID:   existing.UserID,
		Reason:   req.Reason,
		Metadata: cloneMap(req.Metadata),
	})
	return run, nil
}

func (s *Service) RequeueRun(ctx context.Context, req RequeueRunRequest) (domain.RunRecord, error) {
	run, err := s.store.GetRun(ctx, strings.TrimSpace(req.RunID))
	if err != nil {
		return domain.RunRecord{}, err
	}
	if isTerminalRunStatus(run.Status) {
		return domain.RunRecord{}, store.ErrConflict
	}
	messages, err := s.store.ListThreadMessages(ctx, run.ThreadID)
	if err != nil {
		return domain.RunRecord{}, err
	}
	dispatchID := domain.NewID("dispatch")
	reason := strings.TrimSpace(req.Reason)
	if reason == "" {
		reason = "operator requeue"
	}
	metadata := cloneMap(run.Metadata)
	metadata["requeue_reason"] = reason
	metadata["requeue_dispatch_id"] = dispatchID
	for key, value := range req.Metadata {
		if key == domain.EvaluationProfileMetadataKey || key == domain.RemoteMutationIntentsMetadataKey ||
			key == domain.BisqueAccountBindingMetadataKey {
			continue
		}
		metadata[key] = value
	}
	job := jobForStoredRun(
		run,
		messagesForRunRequeue(messages, run),
		metadata,
		dispatchID,
	)
	if err := s.bus.PublishJob(ctx, job); err != nil {
		return run, fmt.Errorf("publish requeued run job: %w", err)
	}
	evictedLease, leaseEvicted, err := s.store.ClearRunLease(ctx, run.RunID)
	if err != nil {
		return run, fmt.Errorf("clear requeued run lease: %w", err)
	}
	run = s.markRunJobDispatched(ctx, run)
	payload := cloneMap(req.Metadata)
	delete(payload, domain.EvaluationProfileMetadataKey)
	delete(payload, domain.RemoteMutationIntentsMetadataKey)
	delete(payload, domain.BisqueAccountBindingMetadataKey)
	attestEvaluationProfile(payload, run)
	payload["reason"] = reason
	payload["dispatch_id"] = dispatchID
	if leaseEvicted {
		payload["evicted_lease_worker_id"] = evictedLease.WorkerID
		payload["evicted_lease_expires_at"] = evictedLease.LeaseExpiresAt.Format(time.RFC3339Nano)
	}
	event, err := s.store.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     run.RunID,
		ThreadID:  run.ThreadID,
		EventKind: "run.requeued",
		EventType: "run",
		Level:     "info",
		Message:   "Run requeued.",
		Payload:   payload,
	})
	if err != nil {
		return run, err
	}
	_ = s.bus.PublishRunEvent(ctx, event)
	return run, nil
}

func (s *Service) RecoverExpiredRunLeases(ctx context.Context, req RecoverExpiredRunLeasesRequest) (RecoverExpiredRunLeasesResult, error) {
	now := req.Now
	if now.IsZero() {
		now = domain.Now()
	}
	limit := req.Limit
	if limit <= 0 {
		limit = 1000
	}
	reason := strings.TrimSpace(req.Reason)
	if reason == "" {
		reason = "automatic expired run lease recovery"
	}
	runs, err := s.listRecoverableRuns(ctx, limit)
	if err != nil {
		return RecoverExpiredRunLeasesResult{}, err
	}
	redispatchQueuedAfter := req.RedispatchQueuedAfter
	if redispatchQueuedAfter <= 0 {
		redispatchQueuedAfter = defaultRedispatchQueuedAfter
	}
	workerHeartbeatStaleAfter := req.WorkerHeartbeatStaleAfter
	if workerHeartbeatStaleAfter <= 0 {
		workerHeartbeatStaleAfter = defaultWorkerHeartbeatStaleAfter
	}
	heartbeatReader, canReadWorkerHeartbeats := s.store.(workerHeartbeatReader)
	result := RecoverExpiredRunLeasesResult{Checked: len(runs)}
	for _, run := range runs {
		if !isRecoverableRunStatus(run.Status) {
			continue
		}
		lease, found, err := s.store.GetRunLease(ctx, run.RunID)
		if err != nil {
			return result, err
		}
		if !found {
			if run.Status != domain.RunStatusQueued {
				// Zombie check: a running/waiting run with NO lease is
				// invisible to the other recovery branches (they all require
				// a lease row), so a worker that died after releasing or
				// never re-acquiring its lease would leave the run stuck
				// forever. Reclaim it only after a generous grace since
				// dispatch AND with no recent worker-progress events —
				// long silent GPU phases keep emitting heartbeat events, so
				// live runs are vetoed by the progress guard.
				dispatchedAt := runJobDispatchedAt(run)
				if dispatchedAt.IsZero() || now.Sub(dispatchedAt) < zombieRunRequeueAfter {
					continue
				}
				if s.runHasRecentWorkerProgress(ctx, run.RunID, domain.WorkerHeartbeatRecord{}, now, workerHeartbeatStaleAfter) {
					continue
				}
				requeued, err := s.RequeueRun(ctx, RequeueRunRequest{
					RunID:  run.RunID,
					Reason: reason,
					Metadata: domain.JSONMap{
						"recovery":           "leaseless_run_without_progress",
						"last_dispatched_at": dispatchedAt.UTC().Format(time.RFC3339Nano),
						"run_status":         string(run.Status),
					},
				})
				if err != nil {
					return result, err
				}
				result.RequeuedRuns = append(result.RequeuedRuns, requeued)
				continue
			}
			// A queued run with no lease and a stale dispatch means its job
			// was lost (consumed and dropped, or never delivered). Without
			// re-dispatch the run would stay queued forever.
			dispatchedAt := runJobDispatchedAt(run)
			if dispatchedAt.IsZero() || now.Sub(dispatchedAt) < redispatchQueuedAfter {
				continue
			}
			requeued, err := s.RequeueRun(ctx, RequeueRunRequest{
				RunID:  run.RunID,
				Reason: reason,
				Metadata: domain.JSONMap{
					"recovery":           "stale_queued_run_without_lease",
					"last_dispatched_at": dispatchedAt.UTC().Format(time.RFC3339Nano),
				},
			})
			if err != nil {
				return result, err
			}
			result.RequeuedRuns = append(result.RequeuedRuns, requeued)
			continue
		}
		if lease.LeaseExpiresAt.After(now) {
			if !canReadWorkerHeartbeats {
				continue
			}
			heartbeat, stale, err := staleRunLeaseOwnerHeartbeat(
				ctx,
				heartbeatReader,
				lease,
				now,
				workerHeartbeatStaleAfter,
			)
			if err != nil {
				return result, err
			}
			if !stale {
				continue
			}
			if s.runHasRecentWorkerProgress(ctx, run.RunID, heartbeat, now, workerHeartbeatStaleAfter) {
				continue
			}
			metadata := domain.JSONMap{
				"recovery":         "stale_run_lease_worker_heartbeat",
				"lease_worker_id":  lease.WorkerID,
				"lease_expires_at": lease.LeaseExpiresAt.UTC().Format(time.RFC3339Nano),
				"lease_updated_at": lease.UpdatedAt.UTC().Format(time.RFC3339Nano),
			}
			if heartbeat.LastHeartbeatAt.IsZero() {
				metadata["worker_heartbeat_missing"] = true
			} else {
				metadata["worker_last_heartbeat_at"] = heartbeat.LastHeartbeatAt.UTC().Format(time.RFC3339Nano)
				metadata["worker_status"] = heartbeat.Status
				metadata["worker_current_run_id"] = heartbeat.CurrentRunID
			}
			requeued, err := s.RequeueRun(ctx, RequeueRunRequest{
				RunID:    run.RunID,
				Reason:   reason,
				Metadata: metadata,
			})
			if err != nil {
				return result, err
			}
			result.RequeuedRuns = append(result.RequeuedRuns, requeued)
			continue
		}
		// An expired lease alone does not prove the worker is dead: it may be
		// mid-renewal against a briefly unreachable control plane while its
		// events still flow. Fresh worker-progress events veto the requeue;
		// a truly dead worker stops emitting and is requeued next pass.
		if s.runHasRecentWorkerProgress(ctx, run.RunID, domain.WorkerHeartbeatRecord{}, now, workerHeartbeatStaleAfter) {
			continue
		}
		requeued, err := s.RequeueRun(ctx, RequeueRunRequest{
			RunID:  run.RunID,
			Reason: reason,
			Metadata: domain.JSONMap{
				"recovery":         "expired_run_lease",
				"lease_worker_id":  lease.WorkerID,
				"lease_expires_at": lease.LeaseExpiresAt.UTC().Format(time.RFC3339Nano),
			},
		})
		if err != nil {
			return result, err
		}
		result.RequeuedRuns = append(result.RequeuedRuns, requeued)
	}
	return result, nil
}

func (s *Service) listRecoverableRuns(ctx context.Context, limit int) ([]domain.RunRecord, error) {
	statuses := []domain.RunStatus{
		domain.RunStatusQueued,
		domain.RunStatusRunning,
		domain.RunStatusWaitingForTask,
	}
	runs := make([]domain.RunRecord, 0, len(statuses)*limit)
	for _, status := range statuses {
		batch, err := s.store.ListRuns(ctx, "", string(status), limit, 0)
		if err != nil {
			return nil, err
		}
		runs = append(runs, batch...)
	}
	return runs, nil
}

func (s *Service) runHasRecentWorkerProgress(ctx context.Context, runID string, heartbeat domain.WorkerHeartbeatRecord, now time.Time, staleAfter time.Duration) bool {
	if staleAfter <= 0 {
		return false
	}
	events, err := s.store.ListRunEvents(ctx, runID, 32)
	if err != nil {
		return false
	}
	for index := len(events) - 1; index >= 0; index-- {
		event := events[index]
		if !isWorkerProgressRunEvent(event.EventKind) {
			continue
		}
		if event.TS.IsZero() {
			continue
		}
		eventTS := event.TS.UTC()
		if !heartbeat.LastHeartbeatAt.IsZero() && !eventTS.After(heartbeat.LastHeartbeatAt.UTC()) {
			continue
		}
		if now.Sub(eventTS) <= staleAfter {
			return true
		}
	}
	return false
}

func isWorkerProgressRunEvent(eventKind string) bool {
	switch strings.TrimSpace(eventKind) {
	case "", "run.accepted", "run.requeued", "run.completed", "run.failed", "run.canceled":
		return false
	default:
		return true
	}
}

func staleRunLeaseOwnerHeartbeat(
	ctx context.Context,
	reader workerHeartbeatReader,
	lease domain.RunLeaseRecord,
	now time.Time,
	staleAfter time.Duration,
) (domain.WorkerHeartbeatRecord, bool, error) {
	if staleAfter <= 0 {
		staleAfter = defaultWorkerHeartbeatStaleAfter
	}
	if now.Sub(lease.UpdatedAt) < staleAfter {
		return domain.WorkerHeartbeatRecord{}, false, nil
	}
	heartbeat, found, err := reader.GetWorkerHeartbeat(ctx, lease.WorkerID)
	if err != nil {
		return domain.WorkerHeartbeatRecord{}, false, err
	}
	if !found {
		return domain.WorkerHeartbeatRecord{}, true, nil
	}
	if now.Sub(heartbeat.LastHeartbeatAt) < staleAfter {
		return heartbeat, false, nil
	}
	return heartbeat, true, nil
}

// runJobDispatchedAt extracts the last time a job for this run was handed to
// the queue, falling back to the run's UpdatedAt when the metadata is absent.
func runJobDispatchedAt(run domain.RunRecord) time.Time {
	if run.Metadata != nil {
		if raw, ok := run.Metadata["job_dispatched_at"]; ok {
			if text, ok := raw.(string); ok {
				if parsed, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(text)); err == nil {
					return parsed
				}
			}
		}
	}
	return run.UpdatedAt
}

func isRecoverableRunStatus(status domain.RunStatus) bool {
	switch status {
	case domain.RunStatusQueued, domain.RunStatusRunning, domain.RunStatusWaitingForTask:
		return true
	default:
		return false
	}
}

func (s *Service) markRunDispatchFailed(ctx context.Context, run domain.RunRecord, dispatchErr error) (domain.RunRecord, error) {
	failureText := fmt.Sprintf("failed to enqueue run job: %v", dispatchErr)
	failureEventInput := domain.AppendRunEventInput{
		EventID:   failedDispatchRunEventID(run.RunID),
		RunID:     run.RunID,
		ThreadID:  run.ThreadID,
		EventKind: "run.failed",
		EventType: "run",
		Level:     "error",
		Message:   "Run failed before worker dispatch.",
		Payload: domain.JSONMap{
			"error": failureText,
			"stage": "job_enqueue",
		},
	}
	event, found, err := s.store.GetRunEvent(ctx, failureEventInput.EventID)
	if err != nil {
		return run, fmt.Errorf("publish run job: %w; additionally failed to check failure event: %v", dispatchErr, err)
	}
	if !found {
		event, err = s.store.AppendRunEvent(ctx, failureEventInput)
		if err != nil {
			return run, fmt.Errorf("publish run job: %w; additionally failed to append failure event: %v", dispatchErr, err)
		}
	}
	if err := s.applyRunEventSideEffects(ctx, appendInputFromEventRecord(event)); err != nil {
		return run, fmt.Errorf("publish run job: %w; additionally failed to mark run failed: %v", dispatchErr, err)
	}
	failedRun, err := s.store.GetRun(ctx, run.RunID)
	if err != nil {
		return run, fmt.Errorf("publish run job: %w; additionally failed to load failed run: %v", dispatchErr, err)
	}
	if publishErr := s.bus.PublishRunEvent(ctx, event); publishErr != nil {
		return failedRun, fmt.Errorf("publish run job: %w; additionally failed to fan out failure event: %v", dispatchErr, publishErr)
	}
	return failedRun, fmt.Errorf("publish run job: %w", dispatchErr)
}

func (s *Service) AcquireRunLease(ctx context.Context, req AcquireRunLeaseRequest) (domain.RunLeaseRecord, error) {
	return s.store.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    strings.TrimSpace(req.RunID),
		WorkerID: strings.TrimSpace(req.WorkerID),
		TTL:      req.TTL,
	})
}

func (s *Service) RenewRunLease(ctx context.Context, req RenewRunLeaseRequest) (domain.RunLeaseRecord, error) {
	return s.store.RenewRunLease(ctx, domain.RenewRunLeaseInput{
		RunID:      strings.TrimSpace(req.RunID),
		LeaseToken: strings.TrimSpace(req.LeaseToken),
		TTL:        req.TTL,
		Now:        req.Now,
	})
}

func (s *Service) ReleaseRunLease(ctx context.Context, req ReleaseRunLeaseRequest) error {
	return s.store.ReleaseRunLease(ctx, domain.ReleaseRunLeaseInput{
		RunID:      strings.TrimSpace(req.RunID),
		LeaseToken: strings.TrimSpace(req.LeaseToken),
	})
}

func (s *Service) IngestRunEvent(ctx context.Context, input domain.AppendRunEventInput) (domain.RunEventRecord, error) {
	if err := s.ensureRunEventSourcePredecessor(ctx, input); err != nil {
		return domain.RunEventRecord{}, err
	}
	if appender, ok := s.store.(activeRunEventAppender); ok {
		return s.ingestRunEventFast(ctx, appender, input)
	}
	return s.ingestRunEventLegacy(ctx, input)
}

func (s *Service) ensureRunEventSourcePredecessor(ctx context.Context, input domain.AppendRunEventInput) error {
	// The ingest worker sets the bypass after a bounded wait proves the
	// predecessor is a permanent producer-side gap: storing this event with a
	// gap beats losing it (readers order by sequence_number).
	if domain.RunEventGateBypassed(ctx) {
		return nil
	}
	if input.SourceSequence <= 1 || strings.TrimSpace(input.RunID) == "" {
		return nil
	}
	if input.EventID != "" {
		if _, found, err := s.store.GetRunEvent(ctx, input.EventID); err != nil {
			return err
		} else if found {
			return nil
		}
	}
	reader, ok := s.store.(runEventSourceSequenceReader)
	if !ok {
		return nil
	}
	_, found, err := reader.GetRunEventBySourceSequence(ctx, input.RunID, input.SourceSequence-1)
	if err != nil {
		return err
	}
	if found {
		return nil
	}
	// The predecessor is absent. Only make the caller wait (retry) for it while
	// the run is still live: a terminal or missing run will never receive the
	// predecessor, so returning ErrRunEventPredecessorPending would wedge the
	// strict partition forever. Fall through so the normal ingest path drops
	// this late event instead. GetRun runs only on this rare miss, not the
	// happy path.
	run, err := s.store.GetRun(ctx, input.RunID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil
		}
		return err
	}
	if isTerminalRunStatus(run.Status) {
		return nil
	}
	return ErrRunEventPredecessorPending
}

// ingestRunEventFast is one store round trip on the happy path: the append
// statement itself deduplicates by event ID (returning the stored record so
// a mutated redelivery can never win), enforces "run exists and is live"
// (dropping late events), and assigns the sequence. Duplicates replay side
// effects and fanout, because a crash between append and side effects
// redelivers the message.
func (s *Service) ingestRunEventFast(ctx context.Context, appender activeRunEventAppender, input domain.AppendRunEventInput) (domain.RunEventRecord, error) {
	if input.EventID != "" {
		lock := s.eventIDLock(input.EventID)
		lock.Lock()
		defer lock.Unlock()
	}
	event, outcome, err := appender.AppendRunEventIfRunActive(ctx, input)
	if err != nil {
		resolved, resolvedOutcome, resolveErr := s.resolveIngestConflict(ctx, input, err)
		if resolveErr != nil {
			return domain.RunEventRecord{}, resolveErr
		}
		event, outcome = resolved, resolvedOutcome
	}
	switch outcome {
	case store.RunEventAppendOutcomeAppended:
		if err := s.applyRunEventSideEffects(ctx, input); err != nil {
			return domain.RunEventRecord{}, err
		}
	case store.RunEventAppendOutcomeDuplicate:
		if err := s.applyRunEventSideEffects(ctx, appendInputFromEventRecord(event)); err != nil {
			return domain.RunEventRecord{}, err
		}
	default:
		return droppedRunEvent(input), nil
	}
	if err := s.bus.PublishRunEvent(ctx, event); err != nil {
		return domain.RunEventRecord{}, fmt.Errorf("%w: %v", ErrRunEventFanoutUnavailable, err)
	}
	return event, nil
}

// resolveIngestConflict turns a conflicting append into a definite outcome so
// the caller never has to retry a conflict forever (which, on a MaxAckPending=1
// partition, would wedge every run on that partition).
//   - event_id primary-key conflict (cross-replica race): the event now exists;
//     replay it as a duplicate so its side effects and fanout still run.
//   - source_sequence unique conflict: a DIFFERENT event already claimed this
//     (run_id, source_sequence) slot (e.g. an uncoordinated producer counter).
//     This event cannot be stored, so drop it (ack) rather than stall. The
//     producer-side coordination fix prevents the collision at the source; this
//     is the defensive backstop.
//   - anything else: propagate so the caller retries (now bounded).
func (s *Service) resolveIngestConflict(ctx context.Context, input domain.AppendRunEventInput, cause error) (domain.RunEventRecord, store.RunEventAppendOutcome, error) {
	if !errors.Is(cause, store.ErrConflict) {
		return domain.RunEventRecord{}, store.RunEventAppendOutcomeDropped, cause
	}
	if input.EventID != "" {
		existing, found, err := s.store.GetRunEvent(ctx, input.EventID)
		if err != nil {
			return domain.RunEventRecord{}, store.RunEventAppendOutcomeDropped, err
		}
		if found {
			return existing, store.RunEventAppendOutcomeDuplicate, nil
		}
	}
	if reader, ok := s.store.(runEventSourceSequenceReader); ok && input.SourceSequence > 0 {
		existing, found, err := reader.GetRunEventBySourceSequence(ctx, input.RunID, input.SourceSequence)
		if err != nil {
			return domain.RunEventRecord{}, store.RunEventAppendOutcomeDropped, err
		}
		if found {
			slog.Warn("dropping run event whose source_sequence is already claimed by a different event",
				"run_id", input.RunID,
				"event_id", input.EventID,
				"source_sequence", input.SourceSequence,
				"stored_event_id", existing.EventID)
			return domain.RunEventRecord{}, store.RunEventAppendOutcomeDropped, nil
		}
	}
	return domain.RunEventRecord{}, store.RunEventAppendOutcomeDropped, cause
}

func (s *Service) ingestRunEventLegacy(ctx context.Context, input domain.AppendRunEventInput) (domain.RunEventRecord, error) {
	run, err := s.store.GetRun(ctx, input.RunID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return droppedRunEvent(input), nil
		}
		return domain.RunEventRecord{}, err
	}
	if input.EventID != "" {
		existing, found, err := s.store.GetRunEvent(ctx, input.EventID)
		if err != nil {
			return domain.RunEventRecord{}, err
		}
		if found {
			if err := s.applyRunEventSideEffects(ctx, appendInputFromEventRecord(existing)); err != nil {
				return domain.RunEventRecord{}, err
			}
			if err := s.bus.PublishRunEvent(ctx, existing); err != nil {
				return domain.RunEventRecord{}, err
			}
			return existing, nil
		}
		lock := s.eventIDLock(input.EventID)
		lock.Lock()
		defer lock.Unlock()
		existing, found, err = s.store.GetRunEvent(ctx, input.EventID)
		if err != nil {
			return domain.RunEventRecord{}, err
		}
		if found {
			if err := s.applyRunEventSideEffects(ctx, appendInputFromEventRecord(existing)); err != nil {
				return domain.RunEventRecord{}, err
			}
			if err := s.bus.PublishRunEvent(ctx, existing); err != nil {
				return domain.RunEventRecord{}, err
			}
			return existing, nil
		}
		run, err = s.store.GetRun(ctx, input.RunID)
		if err != nil {
			if errors.Is(err, store.ErrNotFound) {
				return droppedRunEvent(input), nil
			}
			return domain.RunEventRecord{}, err
		}
	}
	if isTerminalRunStatus(run.Status) {
		return droppedRunEvent(input), nil
	}
	event, err := s.store.AppendRunEvent(ctx, input)
	if err != nil {
		return domain.RunEventRecord{}, err
	}
	if err := s.applyRunEventSideEffects(ctx, input); err != nil {
		return domain.RunEventRecord{}, err
	}
	if err := s.bus.PublishRunEvent(ctx, event); err != nil {
		return domain.RunEventRecord{}, err
	}
	return event, nil
}

func (s *Service) applyRunEventSideEffects(ctx context.Context, input domain.AppendRunEventInput) error {
	switch input.EventKind {
	case "run.started":
		if _, err := s.store.UpdateRunStatus(ctx, input.RunID, domain.RunStatusRunning, "", ""); err != nil {
			return err
		}
	case "run.heartbeat":
		if !s.shouldWriteHeartbeatStatus(input.RunID) {
			return nil
		}
		if _, err := s.store.UpdateRunStatus(ctx, input.RunID, domain.RunStatusRunning, "", ""); err != nil {
			return err
		}
	case "run.token_usage":
		run, err := s.store.GetRun(ctx, input.RunID)
		if err != nil {
			if errors.Is(err, store.ErrNotFound) {
				return nil
			}
			return err
		}
		if err := s.recordRunTokenUsageDelta(ctx, run, input, false); err != nil {
			return err
		}
	case "run.completed":
		responseText := stringFromPayload(input.Payload, "response_text")
		if responseText == "" {
			responseText = input.Message
		}
		completedRun, err := s.store.CompleteRun(ctx, domain.CompleteRunInput{
			RunID:        input.RunID,
			ResponseText: responseText,
		})
		if err != nil {
			return err
		}
		if err := s.ensureTerminalRunTokenUsageFinalized(ctx, completedRun, input); err != nil {
			return err
		}
		conversationTitle := stringFromPayload(input.Payload, "conversation_title")
		if conversationTitle != "" && input.ThreadID != "" {
			if _, err := s.store.ApplyGeneratedThreadTitle(ctx, domain.ApplyGeneratedThreadTitleInput{
				ThreadID:   input.ThreadID,
				RunID:      input.RunID,
				Title:      conversationTitle,
				Generation: jsonMapFromPayload(input.Payload, "title_generation"),
			}); err != nil {
				return err
			}
		}
	case "run.failed":
		errorText := stringFromPayload(input.Payload, "error")
		if errorText == "" {
			errorText = input.Message
		}
		if _, err := s.store.UpdateRunStatus(ctx, input.RunID, domain.RunStatusFailed, "", errorText); err != nil {
			return err
		}
	case "run.canceled":
		reason := stringFromPayload(input.Payload, "reason")
		if reason == "" {
			reason = input.Message
		}
		if _, err := s.store.UpdateRunStatus(ctx, input.RunID, domain.RunStatusCanceled, "", reason); err != nil {
			return err
		}
	case "artifact.created":
		if _, err := s.store.CreateArtifact(ctx, artifactInputFromEvent(input)); err != nil {
			return err
		}
	}
	return nil
}

// shouldWriteHeartbeatStatus rate-limits heartbeat-driven status writes per
// run. The first heartbeat for a run always writes (it may transition the
// run to running); later ones write at most once per interval.
func (s *Service) shouldWriteHeartbeatStatus(runID string) bool {
	now := domain.Now()
	s.heartbeatMu.Lock()
	defer s.heartbeatMu.Unlock()
	if last, ok := s.heartbeatStatusWrites[runID]; ok && now.Sub(last) < heartbeatStatusWriteInterval {
		return false
	}
	if len(s.heartbeatStatusWrites) > 4096 {
		cutoff := now.Add(-4 * heartbeatStatusWriteInterval)
		for staleRunID, last := range s.heartbeatStatusWrites {
			if last.Before(cutoff) {
				delete(s.heartbeatStatusWrites, staleRunID)
			}
		}
	}
	s.heartbeatStatusWrites[runID] = now
	return true
}

func (s *Service) recordRunTokenUsageDelta(ctx context.Context, run domain.RunRecord, input domain.AppendRunEventInput, nested bool) error {
	userID := strings.TrimSpace(run.UserID)
	if userID == "" {
		return nil
	}
	usage := input.Payload
	if nested {
		usage = jsonMapFromPayload(input.Payload, "usage")
	}
	if len(usage) == 0 {
		return nil
	}
	usageEventID := stringFromPayload(usage, "usage_event_id")
	if usageEventID == "" {
		usageEventID = stringFromPayload(input.Payload, "usage_event_id")
	}
	if usageEventID == "" {
		usageEventID = input.EventID
	}
	if usageEventID == "" {
		usageEventID = input.RunID + ":terminal_usage"
	}
	inputTokens := int64FromPayload(usage, "input_tokens")
	outputTokens := int64FromPayload(usage, "output_tokens")
	totalTokens := int64FromPayload(usage, "total_tokens")
	if totalTokens <= 0 {
		totalTokens = inputTokens + outputTokens
	}
	if totalTokens <= 0 && inputTokens <= 0 && outputTokens <= 0 {
		return nil
	}
	day := input.TS
	if day.IsZero() {
		day = domain.Now()
	}
	if nested && run.CompletedAt != nil && !run.CompletedAt.IsZero() {
		day = *run.CompletedAt
	}
	_, _, err := s.store.RecordRunTokenUsage(ctx, domain.RecordRunTokenUsageInput{
		RunID:        run.RunID,
		UsageEventID: usageEventID,
		UserID:       userID,
		Model:        stringFromPayload(usage, "model"),
		Day:          day,
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		TotalTokens:  totalTokens,
		OccurredAt:   domain.Now(),
	})
	return err
}

func (s *Service) ensureTerminalRunTokenUsageFinalized(ctx context.Context, run domain.RunRecord, input domain.AppendRunEventInput) error {
	completedAt := domain.Now()
	if run.CompletedAt != nil && !run.CompletedAt.IsZero() {
		completedAt = *run.CompletedAt
	}
	summary, _, err := s.store.FinalizeRunTokenUsage(ctx, domain.FinalizeRunTokenUsageInput{
		RunID:       input.RunID,
		CompletedAt: completedAt,
	})
	if err != nil {
		return err
	}
	if summary.TotalTokens > 0 || summary.InputTokens > 0 || summary.OutputTokens > 0 {
		return nil
	}
	if err := s.recordRunTokenUsageDelta(ctx, run, input, true); err != nil {
		return err
	}
	_, _, err = s.store.FinalizeRunTokenUsage(ctx, domain.FinalizeRunTokenUsageInput{
		RunID:       input.RunID,
		CompletedAt: completedAt,
	})
	return err
}

func (s *Service) reconcileStoredTerminalEvent(ctx context.Context, run domain.RunRecord) (domain.RunRecord, error) {
	if isTerminalRunStatus(run.Status) {
		return run, nil
	}
	events, err := s.store.ListRunEvents(ctx, run.RunID, 1000)
	if err != nil {
		return domain.RunRecord{}, err
	}
	for _, event := range events {
		if !isTerminalRunEventKind(event.EventKind) {
			continue
		}
		if err := s.applyRunEventSideEffects(ctx, appendInputFromEventRecord(event)); err != nil {
			return domain.RunRecord{}, err
		}
		return s.store.GetRun(ctx, run.RunID)
	}
	return run, nil
}

func appendInputFromEventRecord(event domain.RunEventRecord) domain.AppendRunEventInput {
	return domain.AppendRunEventInput{
		EventID:        event.EventID,
		SourceSequence: event.SourceSequence,
		RunID:          event.RunID,
		ThreadID:       event.ThreadID,
		EventKind:      event.EventKind,
		EventType:      event.EventType,
		NodeName:       event.NodeName,
		TaskID:         event.TaskID,
		CheckpointID:   event.CheckpointID,
		ScopeID:        event.ScopeID,
		AgentRole:      event.AgentRole,
		Level:          event.Level,
		TS:             event.TS,
		Message:        event.Message,
		Payload:        cloneMap(event.Payload),
	}
}

func droppedRunEvent(input domain.AppendRunEventInput) domain.RunEventRecord {
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	eventID := input.EventID
	if eventID == "" {
		eventID = domain.NewID("event")
	}
	return domain.RunEventRecord{
		EventID:        eventID,
		SourceSequence: input.SourceSequence,
		RunID:          input.RunID,
		ThreadID:       input.ThreadID,
		EventKind:      input.EventKind,
		EventType:      input.EventType,
		NodeName:       input.NodeName,
		TaskID:         input.TaskID,
		CheckpointID:   input.CheckpointID,
		ScopeID:        input.ScopeID,
		AgentRole:      input.AgentRole,
		Level:          input.Level,
		TS:             ts,
		Message:        input.Message,
		Payload:        cloneMap(input.Payload),
	}
}

func artifactInputFromEvent(input domain.AppendRunEventInput) domain.CreateArtifactInput {
	return domain.CreateArtifactInput{
		ArtifactID:    artifactIDFromEvent(input),
		RunID:         input.RunID,
		ThreadID:      input.ThreadID,
		Kind:          fallbackString(stringFromPayload(input.Payload, "kind"), "artifact"),
		Path:          fallbackString(stringFromPayload(input.Payload, "relative_path"), fallbackString(stringFromPayload(input.Payload, "path"), stringFromPayload(input.Payload, "source_path"))),
		SourcePath:    stringFromPayload(input.Payload, "source_path"),
		PreviewPath:   stringFromPayload(input.Payload, "preview_path"),
		Title:         fallbackString(stringFromPayload(input.Payload, "title"), input.Message),
		ResultGroupID: stringFromPayload(input.Payload, "result_group_id"),
		MimeType:      stringFromPayload(input.Payload, "mime_type"),
		SizeBytes:     int64FromPayload(input.Payload, "size_bytes"),
		SHA256:        stringFromPayload(input.Payload, "sha256"),
		StorageURI:    stringFromPayload(input.Payload, "storage_uri"),
		ToolName:      stringFromPayload(input.Payload, "tool_name"),
		Category:      stringFromPayload(input.Payload, "category"),
		Metadata:      cloneMap(input.Payload),
	}
}

func artifactIDFromEvent(input domain.AppendRunEventInput) string {
	if artifactID := stringFromPayload(input.Payload, "artifact_id"); artifactID != "" {
		return artifactID
	}
	if eventID := strings.TrimSpace(input.EventID); eventID != "" {
		return "artifact_" + eventID
	}
	return ""
}

func normalizedIdempotencyKey(req CreateRunRequest, metadata domain.JSONMap) string {
	if token := strings.TrimSpace(req.IdempotencyKey); token != "" {
		return token
	}
	return strings.TrimSpace(anyString(metadata["idempotency_key"]))
}

func (s *Service) idempotencyLock(threadID string, userID string, idempotencyKey string) *sync.Mutex {
	hash := fnv.New32a()
	_, _ = hash.Write([]byte(threadID))
	_, _ = hash.Write([]byte{0})
	_, _ = hash.Write([]byte(userID))
	_, _ = hash.Write([]byte{0})
	_, _ = hash.Write([]byte(idempotencyKey))
	return &s.idempotencyLocks[int(hash.Sum32())%len(s.idempotencyLocks)]
}

func (s *Service) eventIDLock(eventID string) *sync.Mutex {
	hash := fnv.New32a()
	_, _ = hash.Write([]byte(eventID))
	return &s.eventIDLocks[int(hash.Sum32())%len(s.eventIDLocks)]
}

func canceledRunEventID(runID string) string {
	return "evt_" + runID + "_canceled"
}

func failedDispatchRunEventID(runID string) string {
	return "evt_" + runID + "_dispatch_failed"
}

func (s *Service) priorArtifactResourceDescriptors(
	ctx context.Context,
	userID string,
	existingMessages []domain.ThreadMessage,
) ([]domain.JSONMap, error) {
	runIDs := priorRunIDsFromMessages(existingMessages)
	descriptors := make([]domain.JSONMap, 0)
	seen := map[string]bool{}
	for _, runID := range runIDs {
		artifacts, err := s.store.ListRunArtifactsForUser(ctx, runID, userID, 100)
		if errors.Is(err, store.ErrNotFound) {
			// A transcript can contain caller-authored run ids. Treat an unreadable
			// id exactly like an absent prior run instead of leaking its artifacts or
			// failing the new run based on whether the guessed id exists.
			continue
		}
		if err != nil {
			return nil, err
		}
		for _, artifact := range artifacts {
			descriptor := artifactResourceDescriptor(artifact)
			key := resourceDescriptorKey(descriptor)
			if key == "" || seen[key] {
				continue
			}
			seen[key] = true
			descriptors = append(descriptors, descriptor)
		}
	}
	return descriptors, nil
}

func withoutCallerArtifactDescriptors(descriptors []domain.JSONMap) []domain.JSONMap {
	filtered := make([]domain.JSONMap, 0, len(descriptors))
	for _, descriptor := range descriptors {
		descriptorType := strings.TrimSpace(anyString(descriptor["type"]))
		if descriptorType == "" || descriptorType == "artifact" {
			continue
		}
		filtered = append(filtered, cloneMap(descriptor))
	}
	return filtered
}

func priorRunIDsFromMessages(messages []domain.ThreadMessage) []string {
	seen := map[string]bool{}
	runIDs := make([]string, 0)
	for index := len(messages) - 1; index >= 0; index-- {
		runID := strings.TrimSpace(messages[index].RunID)
		if runID == "" || seen[runID] {
			continue
		}
		seen[runID] = true
		runIDs = append(runIDs, runID)
	}
	return runIDs
}

func artifactResourceDescriptor(artifact domain.ArtifactRecord) domain.JSONMap {
	descriptor := domain.JSONMap{
		"type":        "artifact",
		"artifact_id": artifact.ArtifactID,
		"run_id":      artifact.RunID,
		"thread_id":   artifact.ThreadID,
		"kind":        artifact.Kind,
		"path":        artifact.Path,
	}
	if artifact.Title != "" {
		descriptor["title"] = artifact.Title
	}
	if artifact.SourcePath != "" {
		descriptor["source_path"] = artifact.SourcePath
	}
	if artifact.PreviewPath != "" {
		descriptor["preview_path"] = artifact.PreviewPath
	}
	if artifact.ResultGroupID != "" {
		descriptor["result_group_id"] = artifact.ResultGroupID
	}
	if artifact.MimeType != "" {
		descriptor["mime_type"] = artifact.MimeType
	}
	if artifact.SizeBytes > 0 {
		descriptor["size_bytes"] = artifact.SizeBytes
	}
	if artifact.SHA256 != "" {
		descriptor["sha256"] = artifact.SHA256
	}
	if artifact.StorageURI != "" {
		descriptor["storage_uri"] = artifact.StorageURI
	}
	if artifact.ToolName != "" {
		descriptor["tool_name"] = artifact.ToolName
	}
	if artifact.Category != "" {
		descriptor["category"] = artifact.Category
	}
	if outputID := anyString(artifact.Metadata["output_id"]); strings.TrimSpace(outputID) != "" {
		descriptor["output_id"] = strings.TrimSpace(outputID)
	}
	return descriptor
}

func mergeResourceDescriptors(primary []domain.JSONMap, secondary []domain.JSONMap) []domain.JSONMap {
	merged := make([]domain.JSONMap, 0, len(primary)+len(secondary))
	seen := map[string]bool{}
	for _, descriptors := range [][]domain.JSONMap{primary, secondary} {
		for _, descriptor := range descriptors {
			copied := cloneMap(descriptor)
			key := resourceDescriptorKey(copied)
			if key == "" {
				key = fmt.Sprint(copied)
			}
			if seen[key] {
				continue
			}
			seen[key] = true
			merged = append(merged, copied)
		}
	}
	return merged
}

func resourceDescriptorKey(descriptor domain.JSONMap) string {
	if value := strings.TrimSpace(anyString(descriptor["artifact_id"])); value != "" {
		return "artifact:" + value
	}
	if value := strings.TrimSpace(anyString(descriptor["output_id"])); value != "" {
		return "output:" + value
	}
	runID := strings.TrimSpace(anyString(descriptor["run_id"]))
	path := strings.TrimSpace(anyString(descriptor["path"]))
	if runID != "" && path != "" {
		return "run-path:" + runID + ":" + path
	}
	return ""
}

func buildRunMetadata(req CreateRunRequest) domain.JSONMap {
	metadata := domain.JSONMap{}
	for key, value := range req.Metadata {
		metadata[key] = value
	}
	delete(metadata, domain.EvaluationProfileMetadataKey)
	delete(metadata, domain.RemoteMutationIntentsMetadataKey)
	if req.EvaluationProfile != "" {
		metadata[domain.EvaluationProfileMetadataKey] = string(req.EvaluationProfile)
	}
	if len(req.RemoteMutationIntents) > 0 {
		metadata[domain.RemoteMutationIntentsMetadataKey] =
			domain.RemoteMutationIntentStrings(req.RemoteMutationIntents)
	}
	if len(req.FileIDs) > 0 {
		metadata["file_ids"] = copyStrings(req.FileIDs)
	}
	if len(req.ResourceURIs) > 0 {
		metadata["resource_uris"] = copyStrings(req.ResourceURIs)
	}
	if len(req.DatasetURIs) > 0 {
		metadata["dataset_uris"] = copyStrings(req.DatasetURIs)
	}
	if len(req.SelectedToolNames) > 0 {
		metadata["selected_tool_names"] = copyStrings(req.SelectedToolNames)
	}
	if len(req.KnowledgeContext) > 0 {
		metadata["knowledge_context"] = cloneMap(req.KnowledgeContext)
	}
	if len(req.WorkflowHint) > 0 {
		metadata["workflow_hint"] = cloneMap(req.WorkflowHint)
	}
	if len(req.SelectionContext) > 0 {
		metadata["selection_context"] = cloneMap(req.SelectionContext)
	}
	if strings.TrimSpace(req.ReasoningMode) != "" {
		metadata["reasoning_mode"] = strings.TrimSpace(req.ReasoningMode)
	}
	if len(req.Budgets) > 0 {
		metadata["budgets"] = cloneMap(req.Budgets)
	}
	if len(req.Benchmark) > 0 {
		metadata["benchmark"] = cloneMap(req.Benchmark)
	}
	if len(req.ResourceDescriptors) > 0 {
		metadata["resource_descriptors"] = copyJSONMaps(req.ResourceDescriptors)
	}
	return metadata
}

func storedRemoteMutationIntents(run domain.RunRecord) []domain.RemoteMutationIntent {
	intents, valid := domain.RemoteMutationIntentsFromMetadata(run.Metadata)
	if !valid {
		return nil
	}
	return intents
}

func storedRemoteMutationIntentsMatch(run domain.RunRecord, requested []domain.RemoteMutationIntent) bool {
	stored, valid := domain.RemoteMutationIntentsFromMetadata(run.Metadata)
	if !valid || len(stored) != len(requested) {
		return false
	}
	for index := range stored {
		if stored[index] != requested[index] {
			return false
		}
	}
	return true
}

func normalizeRuntimeFactsConfig(config RuntimeFactsConfig) RuntimeFactsConfig {
	config.ProductName = strings.TrimSpace(config.ProductName)
	if config.ProductName == "" {
		config.ProductName = "Ultra"
	}
	config.AppName = strings.TrimSpace(config.AppName)
	if config.AppName == "" {
		config.AppName = "BisQue Ultra Control Plane"
	}
	config.AppVersion = strings.TrimSpace(config.AppVersion)
	if config.AppVersion == "" {
		config.AppVersion = "dev"
	}
	config.Environment = strings.TrimSpace(config.Environment)
	if config.Environment == "" {
		config.Environment = "development"
	}
	config.PublicURL = strings.TrimRight(strings.TrimSpace(config.PublicURL), "/")
	config.DefaultUserTimezone = strings.TrimSpace(config.DefaultUserTimezone)
	if config.DefaultUserTimezone == "" {
		config.DefaultUserTimezone = "UTC"
	}
	return config
}

func (s *Service) stampRuntimeFacts(metadata domain.JSONMap) {
	now := s.now()
	if now.IsZero() {
		now = domain.Now()
	}
	now = now.UTC()
	timezone, location := runtimeFactsTimezone(metadata, s.runtimeFacts.DefaultUserTimezone)
	metadata["runtime_facts"] = domain.JSONMap{
		"run_started_at":         now.Format(time.RFC3339Nano),
		"current_datetime_utc":   now.Format(time.RFC3339Nano),
		"current_date_utc":       now.Format("Monday, January 2, 2006"),
		"user_timezone":          timezone,
		"local_datetime":         now.In(location).Format("Monday, January 2, 2006 15:04:05 MST"),
		"product_name":           s.runtimeFacts.ProductName,
		"app_name":               s.runtimeFacts.AppName,
		"app_version":            s.runtimeFacts.AppVersion,
		"deployment_environment": s.runtimeFacts.Environment,
		"public_url":             s.runtimeFacts.PublicURL,
	}
}

func runtimeFactsTimezone(metadata domain.JSONMap, fallback string) (string, *time.Location) {
	candidates := []string{
		strings.TrimSpace(anyString(metadata["user_timezone"])),
		strings.TrimSpace(anyString(metadata["timezone"])),
	}
	if nested, ok := metadata["runtime_facts"].(domain.JSONMap); ok {
		candidates = append(candidates, strings.TrimSpace(anyString(nested["user_timezone"])))
	} else if nested, ok := metadata["runtime_facts"].(map[string]any); ok {
		candidates = append(candidates, strings.TrimSpace(anyString(nested["user_timezone"])))
	}
	candidates = append(candidates, strings.TrimSpace(fallback), "UTC")
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		location, err := time.LoadLocation(candidate)
		if err == nil {
			return candidate, location
		}
	}
	return "UTC", time.UTC
}

func workflowKindForRun(req CreateRunRequest, metadata domain.JSONMap) string {
	// RareSpot prairie-dog detection is now the prairie-dog-detection Skill, run by
	// the normal Deep Agents worker in the code sandbox — there is no separate
	// rarespot dispatch worker/subject, so every run takes the deep_agents path.
	// (A stale client that still requests the old tool now falls through to the
	// agent + Skill instead of hanging on an unconsumed rarespot queue.)
	_ = req
	_ = metadata
	return "deep_agents"
}

func isInternalToolRunRequest(req CreateRunRequest) bool {
	if len(req.Messages) == 0 {
		return false
	}
	for _, message := range req.Messages {
		if !strings.EqualFold(strings.TrimSpace(message.Role), "tool") {
			return false
		}
	}
	return true
}

func isInternalRunRecord(run domain.RunRecord) bool {
	if value, ok := run.Metadata["internal"].(bool); ok && value {
		return true
	}
	if value, ok := run.Metadata["visible_in_thread"].(bool); ok && !value {
		return true
	}
	return false
}

func metadataStringSlice(value any) []string {
	switch typed := value.(type) {
	case []string:
		return typed
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if token := strings.TrimSpace(anyString(item)); token != "" {
				out = append(out, token)
			}
		}
		return out
	default:
		return nil
	}
}

func metadataResourceDescriptors(metadata domain.JSONMap) []domain.JSONMap {
	switch typed := metadata["resource_descriptors"].(type) {
	case []domain.JSONMap:
		return copyJSONMaps(typed)
	case []map[string]any:
		out := make([]domain.JSONMap, 0, len(typed))
		for _, item := range typed {
			out = append(out, cloneMap(domain.JSONMap(item)))
		}
		return out
	case []any:
		out := make([]domain.JSONMap, 0, len(typed))
		for _, item := range typed {
			switch descriptor := item.(type) {
			case domain.JSONMap:
				out = append(out, cloneMap(descriptor))
			case map[string]any:
				out = append(out, cloneMap(domain.JSONMap(descriptor)))
			}
		}
		return out
	default:
		return nil
	}
}

func metadataJSONMap(value any) domain.JSONMap {
	switch typed := value.(type) {
	case domain.JSONMap:
		return cloneMap(typed)
	case map[string]any:
		return cloneMap(domain.JSONMap(typed))
	default:
		return domain.JSONMap{}
	}
}

func messagesForRunRequeue(messages []domain.ThreadMessage, run domain.RunRecord) []domain.ThreadMessage {
	if len(messages) == 0 {
		return nil
	}
	lastRunMessage := -1
	for index, message := range messages {
		if strings.TrimSpace(message.RunID) == run.RunID {
			lastRunMessage = index
		}
	}
	if lastRunMessage >= 0 {
		return copyMessages(messages[:lastRunMessage+1])
	}
	return copyMessages(messages)
}

func anyString(value any) string {
	if value == nil {
		return ""
	}
	if text, ok := value.(string); ok {
		return text
	}
	return fmt.Sprint(value)
}

func copyStrings(values []string) []string {
	return append([]string(nil), values...)
}

func copyMessages(values []domain.ThreadMessage) []domain.ThreadMessage {
	return append([]domain.ThreadMessage(nil), values...)
}

func copyJSONMaps(values []domain.JSONMap) []domain.JSONMap {
	out := make([]domain.JSONMap, 0, len(values))
	for _, value := range values {
		out = append(out, cloneMap(value))
	}
	return out
}

func transcriptSuffixToAppend(existing []domain.ThreadMessage, incoming []domain.ThreadMessage) []domain.ThreadMessage {
	if len(incoming) == 0 || len(existing) == 0 {
		return copyMessages(incoming)
	}
	prefixLen := commonTranscriptPrefixLen(existing, incoming)
	if prefixLen == len(incoming) {
		return nil
	}
	if prefixLen == 0 {
		return copyMessages(incoming)
	}
	return copyMessages(incoming[prefixLen:])
}

func assignPriorAssistantRunIDs(messages []domain.ThreadMessage, priorRunID string) []domain.ThreadMessage {
	if strings.TrimSpace(priorRunID) == "" {
		return messages
	}
	for index := range messages {
		if messages[index].RunID != "" {
			continue
		}
		if strings.EqualFold(strings.TrimSpace(messages[index].Role), "assistant") {
			messages[index].RunID = priorRunID
		}
	}
	return messages
}

func commonTranscriptPrefixLen(existing []domain.ThreadMessage, incoming []domain.ThreadMessage) int {
	limit := len(existing)
	if len(incoming) < limit {
		limit = len(incoming)
	}
	index := 0
	for index < limit && sameTranscriptMessage(existing[index], incoming[index]) {
		index++
	}
	return index
}

func sameTranscriptMessage(left domain.ThreadMessage, right domain.ThreadMessage) bool {
	leftRole := strings.TrimSpace(left.Role)
	rightRole := strings.TrimSpace(right.Role)
	if !strings.EqualFold(leftRole, rightRole) {
		return false
	}
	if strings.EqualFold(leftRole, "assistant") {
		return true
	}
	return left.Content == right.Content
}

func cloneMap(value domain.JSONMap) domain.JSONMap {
	out := domain.JSONMap{}
	for key, item := range value {
		out[key] = item
	}
	return out
}

func isTerminalRunStatus(status domain.RunStatus) bool {
	return status == domain.RunStatusSucceeded ||
		status == domain.RunStatusFailed ||
		status == domain.RunStatusCanceled
}

func isTerminalRunEventKind(eventKind string) bool {
	return eventKind == "run.completed" ||
		eventKind == "run.failed" ||
		eventKind == "run.canceled"
}

func stringFromPayload(payload domain.JSONMap, key string) string {
	return strings.TrimSpace(anyString(payload[key]))
}

func jsonMapFromPayload(payload domain.JSONMap, key string) domain.JSONMap {
	switch value := payload[key].(type) {
	case domain.JSONMap:
		return value
	case map[string]any:
		return domain.JSONMap(value)
	default:
		return domain.JSONMap{}
	}
}

func int64FromPayload(payload domain.JSONMap, key string) int64 {
	switch value := payload[key].(type) {
	case int:
		return int64(value)
	case int64:
		return value
	case float64:
		return int64(value)
	case string:
		var parsed int64
		if _, err := fmt.Sscan(strings.TrimSpace(value), &parsed); err == nil {
			return parsed
		}
	}
	return 0
}

func fallbackString(value string, fallback string) string {
	if strings.TrimSpace(value) != "" {
		return value
	}
	return fallback
}
