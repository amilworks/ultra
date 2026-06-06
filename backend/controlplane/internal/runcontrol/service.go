package runcontrol

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
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
	ApplyGeneratedThreadTitle(context.Context, domain.ApplyGeneratedThreadTitleInput) (domain.ThreadRecord, error)
	GetThread(context.Context, string) (domain.ThreadRecord, error)
	GetThreadForUser(context.Context, string, string) (domain.ThreadRecord, error)
	ListThreads(context.Context, int, int, string) (domain.ThreadListPage, error)
	ListThreadsForUser(context.Context, string, int, int, string) (domain.ThreadListPage, error)
	ListThreadMessages(context.Context, string) ([]domain.ThreadMessage, error)
	ListThreadMessagesForUser(context.Context, string, string) ([]domain.ThreadMessage, error)
	AppendThreadMessage(context.Context, domain.ThreadMessage) (domain.ThreadMessage, error)
	CreateRun(context.Context, domain.CreateRunInput) (domain.RunRecord, error)
	FindRunByIdempotencyKey(context.Context, string, string, string) (domain.RunRecord, bool, error)
	GetRun(context.Context, string) (domain.RunRecord, error)
	GetRunForUser(context.Context, string, string) (domain.RunRecord, error)
	ListRuns(context.Context, string, string, int, int) ([]domain.RunRecord, error)
	ListRunsForUser(context.Context, string, string, string, int, int) ([]domain.RunRecord, error)
	UpdateRunStatus(context.Context, string, domain.RunStatus, string, string) (domain.RunRecord, error)
	CompleteRun(context.Context, domain.CompleteRunInput) (domain.RunRecord, error)
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

type Service struct {
	store            Store
	bus              eventbus.Bus
	idempotencyLocks [64]sync.Mutex
	eventIDLocks     [128]sync.Mutex
}

type CreateThreadRequest struct {
	UserID          string
	Title           string
	Metadata        domain.JSONMap
	InitialMessages []domain.ThreadMessage
}

type CreateRunRequest struct {
	ThreadID            string
	UserID              string
	Goal                string
	Messages            []domain.ThreadMessage
	FileIDs             []string
	ResourceURIs        []string
	DatasetURIs         []string
	SelectedToolNames   []string
	KnowledgeContext    domain.JSONMap
	WorkflowHint        domain.JSONMap
	SelectionContext    domain.JSONMap
	ReasoningMode       string
	Budgets             domain.JSONMap
	Benchmark           domain.JSONMap
	ResourceDescriptors []domain.JSONMap
	IdempotencyKey      string
	Metadata            domain.JSONMap
	JobMetadata         domain.JSONMap
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
}

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
}

type ReleaseRunLeaseRequest struct {
	RunID      string
	LeaseToken string
}

func NewService(store Store, bus eventbus.Bus) *Service {
	return &Service{store: store, bus: bus}
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
			reconciled, err := s.reconcileStoredTerminalEvent(ctx, existing)
			if err != nil {
				return domain.RunRecord{}, err
			}
			if reconciled.Status == domain.RunStatusQueued {
				return s.recoverQueuedRunDispatch(ctx, reconciled, req)
			}
			return reconciled, nil
		}
		metadata["idempotency_key"] = idempotencyKey
	}
	workflowKind := workflowKindForRun(req, metadata)
	thread, err := s.store.GetThread(ctx, req.ThreadID)
	if err != nil {
		return domain.RunRecord{}, err
	}
	existingMessages, err := s.store.ListThreadMessages(ctx, req.ThreadID)
	if err != nil {
		return domain.RunRecord{}, err
	}
	priorResourceDescriptors, err := s.priorArtifactResourceDescriptors(ctx, existingMessages)
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
	job := jobForRun(run, req, resourceDescriptors, metadata)
	if err := s.bus.PublishJob(ctx, job); err != nil {
		return s.markRunDispatchFailed(ctx, run, err)
	}
	run = s.markRunJobDispatched(ctx, run)
	_ = s.bus.PublishRunEvent(ctx, event)
	return run, nil
}

func jobForRun(run domain.RunRecord, req CreateRunRequest, resourceDescriptors []domain.JSONMap, metadata domain.JSONMap) eventbus.Job {
	return eventbus.Job{
		RunID:               run.RunID,
		ThreadID:            run.ThreadID,
		UserID:              run.UserID,
		Goal:                run.Goal,
		WorkflowKind:        run.WorkflowKind,
		Messages:            copyMessages(req.Messages),
		FileIDs:             copyStrings(req.FileIDs),
		ResourceURIs:        copyStrings(req.ResourceURIs),
		DatasetURIs:         copyStrings(req.DatasetURIs),
		SelectedToolNames:   copyStrings(req.SelectedToolNames),
		KnowledgeContext:    cloneMap(req.KnowledgeContext),
		WorkflowHint:        cloneMap(req.WorkflowHint),
		SelectionContext:    cloneMap(req.SelectionContext),
		ReasoningMode:       req.ReasoningMode,
		Budgets:             cloneMap(req.Budgets),
		Benchmark:           cloneMap(req.Benchmark),
		ResourceDescriptors: copyJSONMaps(resourceDescriptors),
		Metadata:            mergeJobMetadata(metadata, req.JobMetadata),
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
	reconciled, err := s.reconcileStoredTerminalEvent(ctx, existing)
	if err != nil {
		return domain.RunRecord{}, err
	}
	if reconciled.Status == domain.RunStatusQueued {
		return s.recoverQueuedRunDispatch(ctx, reconciled, req)
	}
	return reconciled, nil
}

func jobForStoredRun(run domain.RunRecord, messages []domain.ThreadMessage, metadata domain.JSONMap, dispatchID string) eventbus.Job {
	return eventbus.Job{
		RunID:               run.RunID,
		DispatchID:          dispatchID,
		ThreadID:            run.ThreadID,
		UserID:              run.UserID,
		Goal:                run.Goal,
		WorkflowKind:        run.WorkflowKind,
		Messages:            copyMessages(messages),
		FileIDs:             metadataStringSlice(metadata["file_ids"]),
		ResourceURIs:        metadataStringSlice(metadata["resource_uris"]),
		DatasetURIs:         metadataStringSlice(metadata["dataset_uris"]),
		SelectedToolNames:   metadataStringSlice(metadata["selected_tool_names"]),
		KnowledgeContext:    metadataJSONMap(metadata["knowledge_context"]),
		WorkflowHint:        metadataJSONMap(metadata["workflow_hint"]),
		SelectionContext:    metadataJSONMap(metadata["selection_context"]),
		ReasoningMode:       strings.TrimSpace(anyString(metadata["reasoning_mode"])),
		Budgets:             metadataJSONMap(metadata["budgets"]),
		Benchmark:           metadataJSONMap(metadata["benchmark"]),
		ResourceDescriptors: metadataResourceDescriptors(metadata),
		Metadata:            metadata,
	}
}

func (s *Service) recoverQueuedRunDispatch(ctx context.Context, run domain.RunRecord, req CreateRunRequest) (domain.RunRecord, error) {
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
	job := jobForRun(run, req, metadataResourceDescriptors(run.Metadata), cloneMap(run.Metadata))
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
	if err := s.bus.PublishCancel(ctx, eventbus.CancelSignal{
		RunID:    existing.RunID,
		ThreadID: existing.ThreadID,
		UserID:   existing.UserID,
		Reason:   req.Reason,
		Metadata: cloneMap(req.Metadata),
	}); err != nil {
		return existing, fmt.Errorf("publish cancel signal: %w", err)
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
	if err := s.bus.PublishRunEvent(ctx, event); err != nil {
		return domain.RunRecord{}, err
	}
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
	payload := domain.JSONMap{
		"reason":      reason,
		"dispatch_id": dispatchID,
	}
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
	runs, err := s.store.ListRuns(ctx, "", "", limit, 0)
	if err != nil {
		return RecoverExpiredRunLeasesResult{}, err
	}
	result := RecoverExpiredRunLeasesResult{Checked: len(runs)}
	for _, run := range runs {
		if !isRecoverableRunStatus(run.Status) {
			continue
		}
		lease, found, err := s.store.GetRunLease(ctx, run.RunID)
		if err != nil {
			return result, err
		}
		if !found || lease.LeaseExpiresAt.After(now) {
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
	})
}

func (s *Service) ReleaseRunLease(ctx context.Context, req ReleaseRunLeaseRequest) error {
	return s.store.ReleaseRunLease(ctx, domain.ReleaseRunLeaseInput{
		RunID:      strings.TrimSpace(req.RunID),
		LeaseToken: strings.TrimSpace(req.LeaseToken),
	})
}

func (s *Service) IngestRunEvent(ctx context.Context, input domain.AppendRunEventInput) (domain.RunEventRecord, error) {
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
		if _, err := s.store.UpdateRunStatus(ctx, input.RunID, domain.RunStatusRunning, "", ""); err != nil {
			return err
		}
	case "run.completed":
		responseText := stringFromPayload(input.Payload, "response_text")
		if responseText == "" {
			responseText = input.Message
		}
		if _, err := s.store.CompleteRun(ctx, domain.CompleteRunInput{
			RunID:        input.RunID,
			ResponseText: responseText,
		}); err != nil {
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
		EventID:      event.EventID,
		RunID:        event.RunID,
		ThreadID:     event.ThreadID,
		EventKind:    event.EventKind,
		EventType:    event.EventType,
		NodeName:     event.NodeName,
		TaskID:       event.TaskID,
		CheckpointID: event.CheckpointID,
		ScopeID:      event.ScopeID,
		AgentRole:    event.AgentRole,
		Level:        event.Level,
		TS:           event.TS,
		Message:      event.Message,
		Payload:      cloneMap(event.Payload),
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
		EventID:      eventID,
		RunID:        input.RunID,
		ThreadID:     input.ThreadID,
		EventKind:    input.EventKind,
		EventType:    input.EventType,
		NodeName:     input.NodeName,
		TaskID:       input.TaskID,
		CheckpointID: input.CheckpointID,
		ScopeID:      input.ScopeID,
		AgentRole:    input.AgentRole,
		Level:        input.Level,
		TS:           ts,
		Message:      input.Message,
		Payload:      cloneMap(input.Payload),
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

func (s *Service) priorArtifactResourceDescriptors(ctx context.Context, existingMessages []domain.ThreadMessage) ([]domain.JSONMap, error) {
	runIDs := priorRunIDsFromMessages(existingMessages)
	descriptors := make([]domain.JSONMap, 0)
	seen := map[string]bool{}
	for _, runID := range runIDs {
		artifacts, err := s.store.ListRunArtifacts(ctx, runID, 100)
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

func workflowKindForRun(req CreateRunRequest, metadata domain.JSONMap) string {
	if containsRareSpotTool(req.SelectedToolNames) || metadataWorkflowHintIsRareSpot(req.WorkflowHint) {
		return "rarespot_ecology"
	}
	if containsRareSpotTool(metadataStringSlice(metadata["selected_tool_names"])) {
		return "rarespot_ecology"
	}
	if workflow, ok := metadata["workflow_hint"].(domain.JSONMap); ok && metadataWorkflowHintIsRareSpot(workflow) {
		return "rarespot_ecology"
	}
	if workflow, ok := metadata["workflow_hint"].(map[string]any); ok && metadataWorkflowHintIsRareSpot(domain.JSONMap(workflow)) {
		return "rarespot_ecology"
	}
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

func containsRareSpotTool(values []string) bool {
	for _, value := range values {
		token := strings.ToLower(strings.TrimSpace(value))
		if token == "rarespot_ecology" || token == "rarespot_ecology_inference" {
			return true
		}
	}
	return false
}

func metadataWorkflowHintIsRareSpot(workflow domain.JSONMap) bool {
	for _, key := range []string{"id", "name", "workflow", "workflow_kind"} {
		token := strings.ToLower(strings.TrimSpace(anyString(workflow[key])))
		if token == "rarespot_ecology" || token == "rarespot_ecology_inference" {
			return true
		}
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
