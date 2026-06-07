package store

import (
	"context"
	"errors"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

var (
	ErrNotFound = errors.New("not found")
	ErrConflict = errors.New("already exists")
)

const defaultResourceRetention = 30 * 24 * time.Hour

type MemoryStore struct {
	mu             sync.RWMutex
	threads        map[string]domain.ThreadRecord
	messages       map[string][]domain.ThreadMessage
	runs           map[string]domain.RunRecord
	events         map[string][]domain.RunEventRecord
	artifacts      map[string]domain.ArtifactRecord
	resources      map[string]domain.ResourceRecord
	resourceEvents []domain.ResourceEventRecord
	users          map[string]domain.UserAccount
	orgs           map[string]domain.Organization
	bisque         map[string]domain.BisqueCredentialRecord
	leases         map[string]domain.RunLeaseRecord
	workers        map[string]domain.WorkerHeartbeatRecord
}

func NewMemoryStore() *MemoryStore {
	store := &MemoryStore{
		threads:        map[string]domain.ThreadRecord{},
		messages:       map[string][]domain.ThreadMessage{},
		runs:           map[string]domain.RunRecord{},
		events:         map[string][]domain.RunEventRecord{},
		artifacts:      map[string]domain.ArtifactRecord{},
		resources:      map[string]domain.ResourceRecord{},
		resourceEvents: []domain.ResourceEventRecord{},
		users:          map[string]domain.UserAccount{},
		orgs:           map[string]domain.Organization{},
		bisque:         map[string]domain.BisqueCredentialRecord{},
		leases:         map[string]domain.RunLeaseRecord{},
		workers:        map[string]domain.WorkerHeartbeatRecord{},
	}
	store.orgs["local-org"] = defaultLocalOrganization(domain.Now())
	return store
}

func defaultLocalOrganization(now time.Time) domain.Organization {
	return domain.Organization{
		OrgID:     "local-org",
		Name:      "Local Organization",
		Status:    "active",
		CreatedAt: now,
		UpdatedAt: now,
		Metadata:  domain.JSONMap{"source": "dev_default"},
	}
}

func userMatchesQuery(user domain.UserAccount, query string) bool {
	return strings.Contains(strings.ToLower(user.UserID), query) ||
		strings.Contains(strings.ToLower(user.Email), query) ||
		strings.Contains(strings.ToLower(user.DisplayName), query) ||
		strings.Contains(strings.ToLower(user.Role), query) ||
		strings.Contains(strings.ToLower(user.OrgID), query)
}

func (s *MemoryStore) CreateUser(ctx context.Context, input domain.CreateUserInput) (domain.UserAccount, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	now := domain.Now()
	userID := strings.TrimSpace(input.UserID)
	if userID == "" {
		userID = domain.NewID("user")
	}
	if _, exists := s.users[userID]; exists {
		return domain.UserAccount{}, ErrConflict
	}
	email := normalizeEmail(input.Email)
	if email != "" {
		for _, existing := range s.users {
			if normalizeEmail(existing.Email) == email {
				return domain.UserAccount{}, ErrConflict
			}
		}
	}
	role := strings.TrimSpace(input.Role)
	if role == "" {
		role = "researcher"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	user := domain.UserAccount{
		UserID:      userID,
		Email:       email,
		DisplayName: strings.TrimSpace(input.DisplayName),
		Role:        role,
		Status:      status,
		OrgID:       strings.TrimSpace(input.OrgID),
		CreatedAt:   now,
		UpdatedAt:   now,
		Metadata:    mapOrEmpty(input.Metadata),
	}
	s.users[user.UserID] = user
	return user, nil
}

func (s *MemoryStore) UpsertBisqueCredential(ctx context.Context, input domain.UpsertBisqueCredentialInput) (domain.BisqueCredentialRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	now := domain.Now()
	sessionID := strings.TrimSpace(input.SessionID)
	if sessionID == "" {
		sessionID = domain.NewID("bisque_session")
	}
	userID := strings.TrimSpace(input.UserID)
	if userID == "" {
		userID = "local-user"
	}
	orgID := strings.TrimSpace(input.OrgID)
	if orgID == "" {
		orgID = "local-org"
	}
	rootURL := strings.TrimRight(strings.TrimSpace(input.RootURL), "/")
	username := strings.TrimSpace(input.Username)
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	if existingSessionID := s.bisqueSessionIDForOwnerLocked(userID, orgID, rootURL); existingSessionID != "" {
		sessionID = existingSessionID
	}
	existing := s.bisque[sessionID]
	createdAt := existing.CreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	record := domain.BisqueCredentialRecord{
		SessionID:          sessionID,
		UserID:             userID,
		OrgID:              orgID,
		RootURL:            rootURL,
		Username:           username,
		PasswordCiphertext: strings.TrimSpace(input.PasswordCiphertext),
		PasswordNonce:      strings.TrimSpace(input.PasswordNonce),
		PasswordKeyID:      strings.TrimSpace(input.PasswordKeyID),
		PasswordAlgorithm:  strings.TrimSpace(input.PasswordAlgorithm),
		Status:             status,
		LastVerifiedAt:     input.LastVerifiedAt,
		CreatedAt:          createdAt,
		UpdatedAt:          now,
		Metadata:           mapOrEmpty(input.Metadata),
	}
	s.bisque[sessionID] = record
	return record, nil
}

func (s *MemoryStore) GetBisqueCredentialBySessionID(ctx context.Context, sessionID string) (domain.BisqueCredentialRecord, bool, error) {
	_ = ctx
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return domain.BisqueCredentialRecord{}, false, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	record, ok := s.bisque[sessionID]
	if !ok || strings.TrimSpace(record.Status) == "deleted" {
		return domain.BisqueCredentialRecord{}, false, nil
	}
	return record, true, nil
}

func (s *MemoryStore) DeleteBisqueCredentialBySessionID(ctx context.Context, sessionID string) error {
	_ = ctx
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.bisque, sessionID)
	return nil
}

func (s *MemoryStore) bisqueSessionIDForOwnerLocked(userID string, orgID string, rootURL string) string {
	for sessionID, record := range s.bisque {
		if strings.EqualFold(record.UserID, userID) &&
			strings.EqualFold(record.OrgID, orgID) &&
			strings.EqualFold(strings.TrimRight(record.RootURL, "/"), rootURL) &&
			strings.TrimSpace(record.Status) != "deleted" {
			return sessionID
		}
	}
	return ""
}

func orgMatchesQuery(org domain.Organization, query string) bool {
	return strings.Contains(strings.ToLower(org.OrgID), query) ||
		strings.Contains(strings.ToLower(org.Name), query) ||
		strings.Contains(strings.ToLower(org.Status), query)
}

func normalizeOrgID(orgID string) string {
	return strings.ToLower(strings.TrimSpace(orgID))
}

func (s *MemoryStore) CreateOrganization(ctx context.Context, input domain.CreateOrganizationInput) (domain.Organization, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	now := domain.Now()
	orgID := normalizeOrgID(input.OrgID)
	if orgID == "" {
		orgID = domain.NewID("org")
	}
	if _, exists := s.orgs[orgID]; exists {
		return domain.Organization{}, ErrConflict
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	name := strings.TrimSpace(input.Name)
	if name == "" {
		name = orgID
	}
	org := domain.Organization{
		OrgID:     orgID,
		Name:      name,
		Status:    status,
		CreatedAt: now,
		UpdatedAt: now,
		Metadata:  mapOrEmpty(input.Metadata),
	}
	s.orgs[org.OrgID] = org
	return org, nil
}

func (s *MemoryStore) ListOrganizations(ctx context.Context, limit int, query string) ([]domain.Organization, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	query = strings.ToLower(strings.TrimSpace(query))
	orgs := make([]domain.Organization, 0, len(s.orgs))
	for _, org := range s.orgs {
		if query != "" && !orgMatchesQuery(org, query) {
			continue
		}
		orgs = append(orgs, org)
	}
	sort.Slice(orgs, func(i, j int) bool {
		return orgs[i].CreatedAt.After(orgs[j].CreatedAt)
	})
	return take(orgs, limit), nil
}

func normalizeEmail(email string) string {
	return strings.ToLower(strings.TrimSpace(email))
}

func (s *MemoryStore) ListUsers(ctx context.Context, limit int, query string) ([]domain.UserAccount, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	query = strings.ToLower(strings.TrimSpace(query))
	users := make([]domain.UserAccount, 0, len(s.users))
	for _, user := range s.users {
		if query != "" && !userMatchesQuery(user, query) {
			continue
		}
		users = append(users, user)
	}
	sort.Slice(users, func(i, j int) bool {
		return users[i].CreatedAt.After(users[j].CreatedAt)
	})
	return take(users, limit), nil
}

func (s *MemoryStore) GetUserByID(ctx context.Context, userID string) (domain.UserAccount, bool, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return domain.UserAccount{}, false, nil
	}
	user, ok := s.users[userID]
	return user, ok, nil
}

func (s *MemoryStore) GetUserByEmail(ctx context.Context, email string) (domain.UserAccount, bool, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	email = normalizeEmail(email)
	if email == "" {
		return domain.UserAccount{}, false, nil
	}
	for _, user := range s.users {
		if normalizeEmail(user.Email) == email {
			return user, true, nil
		}
	}
	return domain.UserAccount{}, false, nil
}

func (s *MemoryStore) UpdateUserStatus(ctx context.Context, userID string, status string) (domain.UserAccount, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	userID = strings.TrimSpace(userID)
	status = strings.TrimSpace(status)
	if status == "" {
		status = "disabled"
	}
	user, ok := s.users[userID]
	if !ok {
		return domain.UserAccount{}, ErrNotFound
	}
	now := domain.Now()
	if !now.After(user.UpdatedAt) {
		now = user.UpdatedAt.Add(time.Nanosecond)
	}
	user.Status = status
	user.UpdatedAt = now
	s.users[user.UserID] = user
	return user, nil
}

func (s *MemoryStore) UpsertWorkerHeartbeat(ctx context.Context, input domain.UpsertWorkerHeartbeatInput) (domain.WorkerHeartbeatRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	now := domain.Now()
	workerID := strings.TrimSpace(input.WorkerID)
	if workerID == "" {
		return domain.WorkerHeartbeatRecord{}, ErrConflict
	}
	workerKind := strings.TrimSpace(input.WorkerKind)
	if workerKind == "" {
		workerKind = "worker"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "alive"
	}
	heartbeatAt := input.LastHeartbeatAt
	if heartbeatAt.IsZero() {
		heartbeatAt = now
	}
	heartbeatAt = heartbeatAt.UTC()
	startedAt := input.StartedAt
	if startedAt.IsZero() {
		if existing, ok := s.workers[workerID]; ok && !existing.StartedAt.IsZero() {
			startedAt = existing.StartedAt
		} else {
			startedAt = heartbeatAt
		}
	}
	startedAt = startedAt.UTC()
	worker := domain.WorkerHeartbeatRecord{
		WorkerID:        workerID,
		WorkerKind:      workerKind,
		Status:          status,
		CurrentRunID:    strings.TrimSpace(input.CurrentRunID),
		Hostname:        strings.TrimSpace(input.Hostname),
		Version:         strings.TrimSpace(input.Version),
		StartedAt:       startedAt,
		LastHeartbeatAt: heartbeatAt,
		UpdatedAt:       now,
		Metadata:        mapOrEmpty(input.Metadata),
	}
	s.workers[worker.WorkerID] = worker
	return worker, nil
}

func (s *MemoryStore) ListWorkerHeartbeats(ctx context.Context, limit int) ([]domain.WorkerHeartbeatRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	workers := make([]domain.WorkerHeartbeatRecord, 0, len(s.workers))
	for _, worker := range s.workers {
		workers = append(workers, worker)
	}
	sort.Slice(workers, func(i, j int) bool {
		return workers[i].LastHeartbeatAt.After(workers[j].LastHeartbeatAt)
	})
	return take(workers, limit), nil
}

func (s *MemoryStore) CreateThread(ctx context.Context, input domain.CreateThreadInput) (domain.ThreadRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()

	now := domain.Now()
	thread := domain.ThreadRecord{
		ThreadID:  domain.NewID("thread"),
		UserID:    input.UserID,
		Title:     input.Title,
		Status:    domain.ThreadStatusActive,
		CreatedAt: now,
		UpdatedAt: now,
		Metadata:  mapOrEmpty(input.Metadata),
	}
	s.threads[thread.ThreadID] = thread
	for _, msg := range input.InitialMessages {
		msg.MessageID = domain.NewID("msg")
		msg.ThreadID = thread.ThreadID
		msg.CreatedAt = now
		msg.Metadata = mapOrEmpty(msg.Metadata)
		s.messages[thread.ThreadID] = append(s.messages[thread.ThreadID], msg)
	}
	return thread, nil
}

func (s *MemoryStore) GetThread(ctx context.Context, threadID string) (domain.ThreadRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	thread, ok := s.threads[threadID]
	if !ok {
		return domain.ThreadRecord{}, ErrNotFound
	}
	return thread, nil
}

func (s *MemoryStore) GetThreadForUser(ctx context.Context, threadID string, userID string) (domain.ThreadRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	thread, ok := s.threads[threadID]
	if !ok || thread.UserID != userID {
		return domain.ThreadRecord{}, ErrNotFound
	}
	return thread, nil
}

func (s *MemoryStore) UpdateThreadForUser(ctx context.Context, input domain.UpdateThreadInput) (domain.ThreadRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	thread, ok := s.threads[input.ThreadID]
	if !ok || thread.UserID != input.UserID {
		return domain.ThreadRecord{}, ErrNotFound
	}
	if title := normalizedThreadTitle(input.Title); title != "" {
		thread.Title = title
	}
	thread.Metadata = mergeThreadMetadata(thread.Metadata, mapOrEmpty(input.Metadata))
	thread.UpdatedAt = domain.Now()
	s.threads[input.ThreadID] = thread
	return thread, nil
}

func (s *MemoryStore) ApplyGeneratedThreadTitle(ctx context.Context, input domain.ApplyGeneratedThreadTitleInput) (domain.ThreadRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	thread, ok := s.threads[input.ThreadID]
	if !ok {
		return domain.ThreadRecord{}, ErrNotFound
	}
	title := normalizedThreadTitle(input.Title)
	if title == "" || !generatedThreadTitleEligible(thread) {
		return thread, nil
	}
	now := domain.Now()
	previousTitle := thread.Title
	thread.Title = title
	thread.Metadata = generatedThreadTitleMetadata(thread.Metadata, input, previousTitle, now)
	thread.UpdatedAt = now
	s.threads[input.ThreadID] = thread
	return thread, nil
}

func (s *MemoryStore) ListThreads(ctx context.Context, limit int, offset int, status string) (domain.ThreadListPage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	threads := make([]domain.ThreadRecord, 0, len(s.threads))
	status = strings.TrimSpace(status)
	for _, thread := range s.threads {
		if status != "" && string(thread.Status) != status {
			continue
		}
		threads = append(threads, thread)
	}
	sort.Slice(threads, func(i, j int) bool {
		return threads[i].UpdatedAt.After(threads[j].UpdatedAt)
	})
	totalCount := len(threads)
	if offset < 0 {
		offset = 0
	}
	if limit <= 0 {
		limit = totalCount
	}
	if offset >= len(threads) {
		threads = []domain.ThreadRecord{}
	} else {
		end := offset + limit
		if end > len(threads) {
			end = len(threads)
		}
		threads = threads[offset:end]
	}
	return domain.ThreadListPage{
		Threads:    threads,
		TotalCount: totalCount,
		Limit:      limit,
		Offset:     offset,
	}, nil
}

func (s *MemoryStore) ListThreadsForUser(ctx context.Context, userID string, limit int, offset int, status string) (domain.ThreadListPage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	threads := make([]domain.ThreadRecord, 0, len(s.threads))
	status = strings.TrimSpace(status)
	for _, thread := range s.threads {
		if thread.UserID != userID {
			continue
		}
		if status != "" && string(thread.Status) != status {
			continue
		}
		threads = append(threads, thread)
	}
	sort.Slice(threads, func(i, j int) bool {
		return threads[i].UpdatedAt.After(threads[j].UpdatedAt)
	})
	totalCount := len(threads)
	if offset < 0 {
		offset = 0
	}
	if limit <= 0 {
		limit = totalCount
	}
	if offset >= len(threads) {
		threads = []domain.ThreadRecord{}
	} else {
		end := offset + limit
		if end > len(threads) {
			end = len(threads)
		}
		threads = threads[offset:end]
	}
	return domain.ThreadListPage{
		Threads:    threads,
		TotalCount: totalCount,
		Limit:      limit,
		Offset:     offset,
	}, nil
}

func (s *MemoryStore) ListThreadMessages(ctx context.Context, threadID string) ([]domain.ThreadMessage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	messages := append([]domain.ThreadMessage(nil), s.messages[threadID]...)
	return messages, nil
}

func (s *MemoryStore) ListThreadMessagesForUser(ctx context.Context, threadID string, userID string) ([]domain.ThreadMessage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	thread, ok := s.threads[threadID]
	if !ok || thread.UserID != userID {
		return nil, ErrNotFound
	}
	messages := append([]domain.ThreadMessage(nil), s.messages[threadID]...)
	return messages, nil
}

func (s *MemoryStore) AppendThreadMessage(ctx context.Context, message domain.ThreadMessage) (domain.ThreadMessage, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.threads[message.ThreadID]; !ok {
		return domain.ThreadMessage{}, ErrNotFound
	}
	now := domain.Now()
	if message.MessageID == "" {
		message.MessageID = domain.NewID("msg")
	}
	if message.CreatedAt.IsZero() {
		message.CreatedAt = now
	}
	message.Metadata = mapOrEmpty(message.Metadata)
	s.messages[message.ThreadID] = append(s.messages[message.ThreadID], message)
	return message, nil
}

func (s *MemoryStore) CreateRun(ctx context.Context, input domain.CreateRunInput) (domain.RunRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.threads[input.ThreadID]; !ok {
		return domain.RunRecord{}, ErrNotFound
	}
	now := domain.Now()
	workflowKind := input.WorkflowKind
	if workflowKind == "" {
		workflowKind = "deep_agents"
	}
	mode := input.Mode
	if mode == "" {
		mode = "durable"
	}
	run := domain.RunRecord{
		RunID:        domain.NewID("run"),
		ThreadID:     input.ThreadID,
		UserID:       input.UserID,
		Goal:         input.Goal,
		Status:       domain.RunStatusQueued,
		WorkflowKind: workflowKind,
		Mode:         mode,
		CreatedAt:    now,
		UpdatedAt:    now,
		Metadata:     mapOrEmpty(input.Metadata),
	}
	s.runs[run.RunID] = run
	if !input.Internal {
		thread := s.threads[input.ThreadID]
		thread.LatestRunID = run.RunID
		thread.UpdatedAt = now
		s.threads[input.ThreadID] = thread
		for _, msg := range input.Messages {
			msg.MessageID = domain.NewID("msg")
			msg.ThreadID = input.ThreadID
			msg.RunID = threadMessageRunID(msg, run.RunID)
			msg.CreatedAt = now
			msg.Metadata = mapOrEmpty(msg.Metadata)
			s.messages[input.ThreadID] = append(s.messages[input.ThreadID], msg)
		}
	}
	return run, nil
}

func (s *MemoryStore) FindRunByIdempotencyKey(ctx context.Context, threadID string, userID string, idempotencyKey string) (domain.RunRecord, bool, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, run := range s.runs {
		if run.ThreadID != threadID || run.UserID != userID {
			continue
		}
		if token, ok := run.Metadata["idempotency_key"].(string); ok && token == idempotencyKey {
			return run, true, nil
		}
	}
	return domain.RunRecord{}, false, nil
}

func (s *MemoryStore) MarkRunDispatched(ctx context.Context, runID string, dispatchedAt time.Time) (domain.RunRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	run, ok := s.runs[runID]
	if !ok {
		return domain.RunRecord{}, ErrNotFound
	}
	if dispatchedAt.IsZero() {
		dispatchedAt = domain.Now()
	}
	if run.Metadata == nil {
		run.Metadata = domain.JSONMap{}
	}
	run.Metadata["job_dispatched_at"] = dispatchedAt.UTC().Format(time.RFC3339Nano)
	run.UpdatedAt = domain.Now()
	s.runs[runID] = run
	return run, nil
}

func (s *MemoryStore) GetRun(ctx context.Context, runID string) (domain.RunRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	run, ok := s.runs[runID]
	if !ok {
		return domain.RunRecord{}, ErrNotFound
	}
	return run, nil
}

func (s *MemoryStore) GetRunForUser(ctx context.Context, runID string, userID string) (domain.RunRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	run, ok := s.runs[runID]
	if !ok || run.UserID != userID {
		return domain.RunRecord{}, ErrNotFound
	}
	return run, nil
}

func (s *MemoryStore) ListRuns(ctx context.Context, threadID string, status string, limit int, offset int) ([]domain.RunRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	runs := make([]domain.RunRecord, 0, len(s.runs))
	for _, run := range s.runs {
		if threadID != "" && run.ThreadID != threadID {
			continue
		}
		if status != "" && string(run.Status) != status {
			continue
		}
		runs = append(runs, run)
	}
	sort.Slice(runs, func(i, j int) bool {
		return runs[i].UpdatedAt.After(runs[j].UpdatedAt)
	})
	if offset < 0 {
		offset = 0
	}
	if offset >= len(runs) {
		return []domain.RunRecord{}, nil
	}
	runs = runs[offset:]
	return take(runs, limit), nil
}

func (s *MemoryStore) ListRunsForUser(ctx context.Context, userID string, threadID string, status string, limit int, offset int) ([]domain.RunRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	if threadID != "" {
		thread, ok := s.threads[threadID]
		if !ok || thread.UserID != userID {
			return nil, ErrNotFound
		}
	}
	runs := make([]domain.RunRecord, 0, len(s.runs))
	for _, run := range s.runs {
		if run.UserID != userID {
			continue
		}
		if threadID != "" && run.ThreadID != threadID {
			continue
		}
		if status != "" && string(run.Status) != status {
			continue
		}
		runs = append(runs, run)
	}
	sort.Slice(runs, func(i, j int) bool {
		return runs[i].UpdatedAt.After(runs[j].UpdatedAt)
	})
	if offset < 0 {
		offset = 0
	}
	if offset >= len(runs) {
		return []domain.RunRecord{}, nil
	}
	runs = runs[offset:]
	return take(runs, limit), nil
}

func (s *MemoryStore) GetRunLease(ctx context.Context, runID string) (domain.RunLeaseRecord, bool, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	if _, ok := s.runs[runID]; !ok {
		return domain.RunLeaseRecord{}, false, ErrNotFound
	}
	lease, ok := s.leases[runID]
	return lease, ok, nil
}

func (s *MemoryStore) UpdateRunStatus(ctx context.Context, runID string, status domain.RunStatus, responseText string, errorText string) (domain.RunRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	run, ok := s.runs[runID]
	if !ok {
		return domain.RunRecord{}, ErrNotFound
	}
	if isTerminalRunStatus(run.Status) {
		return run, nil
	}
	now := domain.Now()
	run.Status = status
	run.ResponseText = responseText
	run.Error = errorText
	run.UpdatedAt = now
	if status == domain.RunStatusRunning && run.StartedAt == nil {
		run.StartedAt = &now
	}
	if isTerminalRunStatus(status) {
		run.CompletedAt = &now
	}
	s.runs[runID] = run
	return run, nil
}

func (s *MemoryStore) CompleteRun(ctx context.Context, input domain.CompleteRunInput) (domain.RunRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	run, ok := s.runs[input.RunID]
	if !ok {
		return domain.RunRecord{}, ErrNotFound
	}
	responseText := strings.TrimSpace(input.ResponseText)
	if run.Status == domain.RunStatusSucceeded {
		if strings.TrimSpace(run.ResponseText) == "" && responseText != "" {
			run.ResponseText = responseText
			run.UpdatedAt = domain.Now()
			s.runs[input.RunID] = run
		}
		s.appendCompletedAssistantMessageLocked(run, responseText)
		return s.runs[input.RunID], nil
	}
	if isTerminalRunStatus(run.Status) {
		return run, nil
	}
	now := domain.Now()
	run.Status = domain.RunStatusSucceeded
	run.ResponseText = responseText
	run.Error = ""
	run.UpdatedAt = now
	run.CompletedAt = &now
	s.runs[input.RunID] = run
	s.appendCompletedAssistantMessageLocked(run, responseText)
	return s.runs[input.RunID], nil
}

func (s *MemoryStore) appendCompletedAssistantMessageLocked(run domain.RunRecord, responseText string) {
	if responseText == "" || isInternalRunMetadata(run.Metadata) {
		return
	}
	for _, message := range s.messages[run.ThreadID] {
		if !strings.EqualFold(strings.TrimSpace(message.Role), "assistant") {
			continue
		}
		if strings.TrimSpace(message.RunID) == run.RunID && message.Content == responseText {
			return
		}
	}
	now := domain.Now()
	s.messages[run.ThreadID] = append(s.messages[run.ThreadID], domain.ThreadMessage{
		MessageID: domain.NewID("msg"),
		ThreadID:  run.ThreadID,
		Role:      "assistant",
		Content:   responseText,
		CreatedAt: now,
		Metadata:  domain.JSONMap{},
		RunID:     run.RunID,
	})
}

func isInternalRunMetadata(metadata domain.JSONMap) bool {
	if metadata == nil {
		return false
	}
	if internal, ok := metadata["internal"].(bool); ok && internal {
		return true
	}
	if visible, ok := metadata["visible_in_thread"].(bool); ok && !visible {
		return true
	}
	return false
}

func (s *MemoryStore) AcquireRunLease(ctx context.Context, input domain.AcquireRunLeaseInput) (domain.RunLeaseRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	run, ok := s.runs[input.RunID]
	if !ok {
		return domain.RunLeaseRecord{}, ErrNotFound
	}
	if isTerminalRunStatus(run.Status) {
		return domain.RunLeaseRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	ttl := positiveLeaseTTL(input.TTL)
	if existing, ok := s.leases[input.RunID]; ok && existing.LeaseExpiresAt.After(now) {
		return domain.RunLeaseRecord{}, ErrConflict
	}
	lease := domain.RunLeaseRecord{
		RunID:          input.RunID,
		WorkerID:       strings.TrimSpace(input.WorkerID),
		LeaseToken:     domain.NewID("lease"),
		LeaseExpiresAt: now.Add(ttl),
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	s.leases[input.RunID] = lease
	run.Status = domain.RunStatusRunning
	run.UpdatedAt = now
	if run.StartedAt == nil {
		run.StartedAt = &now
	}
	s.runs[input.RunID] = run
	return lease, nil
}

func (s *MemoryStore) RenewRunLease(ctx context.Context, input domain.RenewRunLeaseInput) (domain.RunLeaseRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	run, ok := s.runs[input.RunID]
	if !ok {
		return domain.RunLeaseRecord{}, ErrNotFound
	}
	if isTerminalRunStatus(run.Status) {
		return domain.RunLeaseRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	lease, ok := s.leases[input.RunID]
	if !ok || lease.LeaseToken != strings.TrimSpace(input.LeaseToken) || !lease.LeaseExpiresAt.After(now) {
		return domain.RunLeaseRecord{}, ErrConflict
	}
	lease.LeaseExpiresAt = now.Add(positiveLeaseTTL(input.TTL))
	lease.UpdatedAt = now
	s.leases[input.RunID] = lease
	run.UpdatedAt = now
	s.runs[input.RunID] = run
	return lease, nil
}

func (s *MemoryStore) ReleaseRunLease(ctx context.Context, input domain.ReleaseRunLeaseInput) error {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.runs[input.RunID]; !ok {
		return ErrNotFound
	}
	lease, ok := s.leases[input.RunID]
	if !ok {
		return nil
	}
	if lease.LeaseToken != strings.TrimSpace(input.LeaseToken) {
		return ErrConflict
	}
	delete(s.leases, input.RunID)
	return nil
}

func (s *MemoryStore) ClearRunLease(ctx context.Context, runID string) (domain.RunLeaseRecord, bool, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.runs[runID]; !ok {
		return domain.RunLeaseRecord{}, false, ErrNotFound
	}
	lease, ok := s.leases[runID]
	if !ok {
		return domain.RunLeaseRecord{}, false, nil
	}
	delete(s.leases, runID)
	return lease, true, nil
}

func (s *MemoryStore) AppendRunEvent(ctx context.Context, input domain.AppendRunEventInput) (domain.RunEventRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.runs[input.RunID]; !ok {
		return domain.RunEventRecord{}, ErrNotFound
	}
	seq := int64(len(s.events[input.RunID]) + 1)
	eventID := input.EventID
	if eventID == "" {
		eventID = domain.NewID("event")
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	event := domain.RunEventRecord{
		EventID:      eventID,
		Sequence:     seq,
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
		Payload:      mapOrEmpty(input.Payload),
	}
	s.events[input.RunID] = append(s.events[input.RunID], event)
	return event, nil
}

func (s *MemoryStore) GetRunEvent(ctx context.Context, eventID string) (domain.RunEventRecord, bool, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	if eventID == "" {
		return domain.RunEventRecord{}, false, nil
	}
	for _, events := range s.events {
		for _, event := range events {
			if event.EventID == eventID {
				return event, true, nil
			}
		}
	}
	return domain.RunEventRecord{}, false, nil
}

func (s *MemoryStore) ListRunEvents(ctx context.Context, runID string, limit int) ([]domain.RunEventRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneLatestRunEventsPage(s.events[runID], limit), nil
}

func (s *MemoryStore) ListRunEventsForUser(ctx context.Context, runID string, userID string, limit int) ([]domain.RunEventRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	run, ok := s.runs[runID]
	if !ok || run.UserID != userID {
		return nil, ErrNotFound
	}
	return cloneLatestRunEventsPage(s.events[runID], limit), nil
}

func (s *MemoryStore) ListRunEventsAfter(ctx context.Context, runID string, afterSequence int64, limit int) ([]domain.RunEventRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneRunEventsPageAfter(s.events[runID], afterSequence, limit), nil
}

func (s *MemoryStore) ListRunEventsAfterForUser(ctx context.Context, runID string, userID string, afterSequence int64, limit int) ([]domain.RunEventRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	run, ok := s.runs[runID]
	if !ok || run.UserID != userID {
		return nil, ErrNotFound
	}
	return cloneRunEventsPageAfter(s.events[runID], afterSequence, limit), nil
}

func cloneLatestRunEventsPage(source []domain.RunEventRecord, limit int) []domain.RunEventRecord {
	if limit > 0 && len(source) > limit {
		source = source[len(source)-limit:]
	}
	return append([]domain.RunEventRecord(nil), source...)
}

func cloneRunEventsPageAfter(source []domain.RunEventRecord, afterSequence int64, limit int) []domain.RunEventRecord {
	start := sort.Search(len(source), func(index int) bool {
		return source[index].Sequence > afterSequence
	})
	if start >= len(source) {
		return []domain.RunEventRecord{}
	}
	end := len(source)
	if limit > 0 && start+limit < end {
		end = start + limit
	}
	return append([]domain.RunEventRecord(nil), source[start:end]...)
}

func (s *MemoryStore) CreateArtifact(ctx context.Context, input domain.CreateArtifactInput) (domain.ArtifactRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.runs[input.RunID]; !ok {
		return domain.ArtifactRecord{}, ErrNotFound
	}
	artifactID := input.ArtifactID
	if artifactID == "" {
		artifactID = domain.NewID("artifact")
	}
	if existing, ok := s.artifacts[artifactID]; ok {
		return existing, nil
	}
	now := domain.Now()
	artifact := domain.ArtifactRecord{
		ArtifactID:    artifactID,
		RunID:         input.RunID,
		ThreadID:      input.ThreadID,
		Kind:          input.Kind,
		Path:          input.Path,
		SourcePath:    input.SourcePath,
		PreviewPath:   input.PreviewPath,
		Title:         input.Title,
		ResultGroupID: input.ResultGroupID,
		MimeType:      input.MimeType,
		SizeBytes:     input.SizeBytes,
		SHA256:        input.SHA256,
		StorageURI:    input.StorageURI,
		ToolName:      input.ToolName,
		Category:      input.Category,
		CreatedAt:     now,
		UpdatedAt:     now,
		Metadata:      mapOrEmpty(input.Metadata),
	}
	s.artifacts[artifact.ArtifactID] = artifact
	return artifact, nil
}

func (s *MemoryStore) ListRunArtifacts(ctx context.Context, runID string, limit int) ([]domain.ArtifactRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	artifacts := []domain.ArtifactRecord{}
	for _, artifact := range s.artifacts {
		if artifact.RunID == runID {
			artifacts = append(artifacts, artifact)
		}
	}
	sort.Slice(artifacts, func(i, j int) bool {
		return artifacts[i].CreatedAt.After(artifacts[j].CreatedAt)
	})
	return take(artifacts, limit), nil
}

func (s *MemoryStore) ListRunArtifactsForUser(ctx context.Context, runID string, userID string, limit int) ([]domain.ArtifactRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	run, ok := s.runs[runID]
	if !ok || run.UserID != userID {
		return nil, ErrNotFound
	}
	artifacts := []domain.ArtifactRecord{}
	for _, artifact := range s.artifacts {
		if artifact.RunID == runID {
			artifacts = append(artifacts, artifact)
		}
	}
	sort.Slice(artifacts, func(i, j int) bool {
		return artifacts[i].CreatedAt.After(artifacts[j].CreatedAt)
	})
	return take(artifacts, limit), nil
}

func (s *MemoryStore) GetArtifact(ctx context.Context, artifactID string) (domain.ArtifactRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	artifact, ok := s.artifacts[artifactID]
	if !ok {
		return domain.ArtifactRecord{}, ErrNotFound
	}
	return artifact, nil
}

func (s *MemoryStore) GetArtifactForUser(ctx context.Context, artifactID string, userID string) (domain.ArtifactRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	artifact, ok := s.artifacts[artifactID]
	if !ok {
		return domain.ArtifactRecord{}, ErrNotFound
	}
	run, ok := s.runs[artifact.RunID]
	if !ok || run.UserID != userID {
		return domain.ArtifactRecord{}, ErrNotFound
	}
	return artifact, nil
}

func (s *MemoryStore) UpsertResource(ctx context.Context, input domain.UpsertResourceInput) (domain.ResourceRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	resourceID := strings.TrimSpace(input.ResourceID)
	if resourceID == "" {
		resourceID = domain.NewID("file")
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	if ownerUserID == "" {
		ownerUserID = "local-user"
	}
	sourceType := strings.TrimSpace(input.SourceType)
	if sourceType == "" {
		sourceType = "upload"
	}
	resourceKind := strings.TrimSpace(input.ResourceKind)
	if resourceKind == "" {
		resourceKind = "file"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	now := domain.Now()
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		if existing, ok := s.resources[resourceID]; ok && !existing.CreatedAt.IsZero() {
			createdAt = existing.CreatedAt
		} else {
			createdAt = now
		}
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = now
	}
	resource := domain.ResourceRecord{
		ResourceID:         resourceID,
		OriginalName:       strings.TrimSpace(input.OriginalName),
		ContentType:        strings.TrimSpace(input.ContentType),
		SizeBytes:          input.SizeBytes,
		SHA256:             strings.TrimSpace(input.SHA256),
		StorageURI:         strings.TrimSpace(input.StorageURI),
		StoragePath:        strings.TrimSpace(input.StoragePath),
		SourceType:         sourceType,
		ResourceKind:       resourceKind,
		SourceURI:          strings.TrimSpace(input.SourceURI),
		ProjectID:          strings.TrimSpace(input.ProjectID),
		OwnerUserID:        ownerUserID,
		OwnerOrgID:         strings.TrimSpace(input.OwnerOrgID),
		OwnerRole:          strings.TrimSpace(input.OwnerRole),
		Status:             status,
		CreatedAt:          createdAt.UTC(),
		UpdatedAt:          updatedAt.UTC(),
		DeletedAt:          input.DeletedAt,
		RetentionExpiresAt: input.RetentionExpiresAt,
		Metadata:           mapOrEmpty(input.Metadata),
	}
	s.resources[resource.ResourceID] = resource
	return resource, nil
}

func (s *MemoryStore) GetResourceForUser(ctx context.Context, resourceID string, userID string, orgID string) (domain.ResourceRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	resource, ok := s.resources[strings.TrimSpace(resourceID)]
	if !ok || !resourceVisibleToOwner(resource, userID, orgID) || strings.TrimSpace(resource.Status) != "active" {
		return domain.ResourceRecord{}, ErrNotFound
	}
	return resource, nil
}

func (s *MemoryStore) ListResourcesForUser(ctx context.Context, input domain.ResourceListInput) (domain.ResourceListPage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	query := strings.ToLower(strings.TrimSpace(input.Query))
	kind := strings.TrimSpace(input.Kind)
	source := strings.TrimSpace(input.Source)
	projectID := strings.TrimSpace(input.ProjectID)
	resources := make([]domain.ResourceRecord, 0, len(s.resources))
	for _, resource := range s.resources {
		if !resourceVisibleToOwner(resource, input.UserID, input.OrgID) {
			continue
		}
		if strings.TrimSpace(resource.Status) != status {
			continue
		}
		if kind != "" && resource.ResourceKind != kind {
			continue
		}
		if source != "" && resource.SourceType != source {
			continue
		}
		if projectID != "" && resource.ProjectID != projectID {
			continue
		}
		if query != "" && !resourceRecordMatchesQuery(resource, query) {
			continue
		}
		resources = append(resources, resource)
	}
	sort.Slice(resources, func(i, j int) bool {
		if resources[i].CreatedAt.Equal(resources[j].CreatedAt) {
			return resources[i].ResourceID < resources[j].ResourceID
		}
		return resources[i].CreatedAt.After(resources[j].CreatedAt)
	})
	total := len(resources)
	offset := input.Offset
	if offset < 0 {
		offset = 0
	}
	if offset >= len(resources) {
		resources = []domain.ResourceRecord{}
	} else {
		resources = resources[offset:]
	}
	resources = take(resources, input.Limit)
	return domain.ResourceListPage{
		Resources:  resources,
		TotalCount: total,
		Limit:      input.Limit,
		Offset:     input.Offset,
	}, nil
}

func (s *MemoryStore) ListResources(ctx context.Context, limit int, offset int) ([]domain.ResourceRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	resources := make([]domain.ResourceRecord, 0, len(s.resources))
	for _, resource := range s.resources {
		resources = append(resources, resource)
	}
	sort.Slice(resources, func(i, j int) bool {
		if resources[i].CreatedAt.Equal(resources[j].CreatedAt) {
			return resources[i].ResourceID < resources[j].ResourceID
		}
		return resources[i].CreatedAt.After(resources[j].CreatedAt)
	})
	if offset < 0 {
		offset = 0
	}
	if offset >= len(resources) {
		return []domain.ResourceRecord{}, nil
	}
	return take(resources[offset:], limit), nil
}

func (s *MemoryStore) SoftDeleteResourceForUser(ctx context.Context, resourceID string, userID string, orgID string, deletedAt time.Time) (domain.ResourceRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	resource, ok := s.resources[strings.TrimSpace(resourceID)]
	if !ok || !resourceVisibleToOwner(resource, userID, orgID) || strings.TrimSpace(resource.Status) == "deleted" {
		return domain.ResourceRecord{}, ErrNotFound
	}
	if deletedAt.IsZero() {
		deletedAt = domain.Now()
	}
	resource.Status = "deleted"
	resource.DeletedAt = deletedAt.UTC()
	resource.RetentionExpiresAt = resource.DeletedAt.Add(defaultResourceRetention)
	resource.UpdatedAt = resource.DeletedAt
	s.resources[resource.ResourceID] = resource
	return resource, nil
}

func (s *MemoryStore) RestoreResourceForUser(ctx context.Context, resourceID string, userID string, orgID string, restoredAt time.Time) (domain.ResourceRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	resource, ok := s.resources[strings.TrimSpace(resourceID)]
	if !ok || !resourceVisibleToOwner(resource, userID, orgID) {
		return domain.ResourceRecord{}, ErrNotFound
	}
	if restoredAt.IsZero() {
		restoredAt = domain.Now()
	}
	resource.Status = "active"
	resource.DeletedAt = time.Time{}
	resource.RetentionExpiresAt = time.Time{}
	resource.UpdatedAt = restoredAt.UTC()
	s.resources[resource.ResourceID] = resource
	return resource, nil
}

func (s *MemoryStore) ResourceStorageStats(ctx context.Context) (domain.ResourceStorageStats, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	var stats domain.ResourceStorageStats
	for _, resource := range s.resources {
		if strings.TrimSpace(resource.Status) != "active" {
			continue
		}
		stats.TotalResources++
		stats.TotalBytes += resource.SizeBytes
	}
	return stats, nil
}

func (s *MemoryStore) CreateResourceEvent(ctx context.Context, input domain.AppendResourceEventInput) (domain.ResourceEventRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.resources[strings.TrimSpace(input.ResourceID)]; !ok {
		return domain.ResourceEventRecord{}, ErrNotFound
	}
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("resource_event")
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	record := domain.ResourceEventRecord{
		EventID:     eventID,
		ResourceID:  strings.TrimSpace(input.ResourceID),
		ActorUserID: strings.TrimSpace(input.ActorUserID),
		ActorOrgID:  strings.TrimSpace(input.ActorOrgID),
		EventType:   strings.TrimSpace(input.EventType),
		TS:          ts.UTC(),
		Metadata:    mapOrEmpty(input.Metadata),
	}
	s.resourceEvents = append(s.resourceEvents, record)
	return record, nil
}

func (s *MemoryStore) ListResourceEvents(ctx context.Context, resourceID string, limit int) ([]domain.ResourceEventRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	if _, ok := s.resources[strings.TrimSpace(resourceID)]; !ok {
		return nil, ErrNotFound
	}
	events := make([]domain.ResourceEventRecord, 0, len(s.resourceEvents))
	for _, event := range s.resourceEvents {
		if event.ResourceID == strings.TrimSpace(resourceID) {
			events = append(events, event)
		}
	}
	sort.Slice(events, func(i, j int) bool {
		if events[i].TS.Equal(events[j].TS) {
			return events[i].EventID < events[j].EventID
		}
		return events[i].TS.After(events[j].TS)
	})
	return take(events, limit), nil
}

func resourceVisibleToOwner(resource domain.ResourceRecord, userID string, orgID string) bool {
	if strings.TrimSpace(resource.OwnerUserID) != strings.TrimSpace(userID) {
		return false
	}
	ownerOrgID := strings.TrimSpace(resource.OwnerOrgID)
	return ownerOrgID == "" || ownerOrgID == strings.TrimSpace(orgID)
}

func resourceRecordMatchesQuery(resource domain.ResourceRecord, query string) bool {
	candidates := []string{
		resource.ResourceID,
		resource.OriginalName,
		resource.SourceURI,
		resource.ContentType,
		resource.ResourceKind,
		resource.SourceType,
		resource.ProjectID,
		resource.SHA256,
	}
	for _, candidate := range candidates {
		if strings.Contains(strings.ToLower(strings.TrimSpace(candidate)), query) {
			return true
		}
	}
	return false
}

func mapOrEmpty(value domain.JSONMap) domain.JSONMap {
	if value == nil {
		return domain.JSONMap{}
	}
	return value
}

func leaseNow(now time.Time) time.Time {
	if now.IsZero() {
		return domain.Now()
	}
	return now.UTC()
}

func positiveLeaseTTL(ttl time.Duration) time.Duration {
	if ttl <= 0 {
		return time.Minute
	}
	return ttl
}

func take[T any](values []T, limit int) []T {
	if limit <= 0 || len(values) <= limit {
		return values
	}
	return values[:limit]
}
