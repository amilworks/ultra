package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
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

type uploadSessionStats struct {
	FileCount          int
	CompletedFileCount int
	BytesReceived      int64
	BytesVerified      int64
	BytesCommitted     int64
}

type MemoryStore struct {
	mu                     sync.RWMutex
	threads                map[string]domain.ThreadRecord
	messages               map[string][]domain.ThreadMessage
	runs                   map[string]domain.RunRecord
	events                 map[string][]domain.RunEventRecord
	artifacts              map[string]domain.ArtifactRecord
	resources              map[string]domain.ResourceRecord
	resourceEvents         []domain.ResourceEventRecord
	resourceGrants         map[string]domain.ResourceShareGrantRecord
	collections            map[string]domain.ResourceCollectionRecord
	collectionMembers      map[string]domain.ResourceCollectionMembershipRecord
	collectionGrants       map[string]domain.ResourceCollectionShareGrantRecord
	datasetSnapshots       map[string]domain.DatasetSnapshotRecord
	datasetEntries         map[string][]domain.DatasetSnapshotResourceRecord
	datasetGrants          map[string]domain.DatasetSnapshotShareGrantRecord
	datasetEvents          []domain.DatasetSnapshotEventRecord
	dataAgentJobs          map[string]domain.DataAgentJobRecord
	dataAgentEvents        map[string][]domain.DataAgentJobEventRecord
	dataAgentLeases        map[string]domain.DataAgentJobLeaseRecord
	uploadSessions         map[string]domain.UploadSessionRecord
	uploadFiles            map[string]domain.UploadSessionFileRecord
	uploadChunks           map[string]domain.UploadChunkRecord
	uploadFilesBySession   map[string]map[string]struct{}
	uploadChunksByFile     map[string]map[string]struct{}
	uploadChunksBySession  map[string]map[string]struct{}
	uploadSessionStats     map[string]uploadSessionStats
	uploadEvents           []domain.UploadSessionEventRecord
	users                  map[string]domain.UserAccount
	runTokenUsage          map[string]domain.RunTokenUsageRecord
	runTokenUsageFinalized map[string]bool
	tokenUsageDaily        map[string]map[string]*domain.UserTokenUsageDaily
	tokenUsageLifetime     map[string]domain.UserTokenUsageStats
	orgs                   map[string]domain.Organization
	bisque                 map[string]domain.BisqueCredentialRecord
	leases                 map[string]domain.RunLeaseRecord
	workers                map[string]domain.WorkerHeartbeatRecord
}

func NewMemoryStore() *MemoryStore {
	store := &MemoryStore{
		threads:                map[string]domain.ThreadRecord{},
		messages:               map[string][]domain.ThreadMessage{},
		runs:                   map[string]domain.RunRecord{},
		events:                 map[string][]domain.RunEventRecord{},
		artifacts:              map[string]domain.ArtifactRecord{},
		resources:              map[string]domain.ResourceRecord{},
		resourceEvents:         []domain.ResourceEventRecord{},
		resourceGrants:         map[string]domain.ResourceShareGrantRecord{},
		collections:            map[string]domain.ResourceCollectionRecord{},
		collectionMembers:      map[string]domain.ResourceCollectionMembershipRecord{},
		collectionGrants:       map[string]domain.ResourceCollectionShareGrantRecord{},
		datasetSnapshots:       map[string]domain.DatasetSnapshotRecord{},
		datasetEntries:         map[string][]domain.DatasetSnapshotResourceRecord{},
		datasetGrants:          map[string]domain.DatasetSnapshotShareGrantRecord{},
		datasetEvents:          []domain.DatasetSnapshotEventRecord{},
		dataAgentJobs:          map[string]domain.DataAgentJobRecord{},
		dataAgentEvents:        map[string][]domain.DataAgentJobEventRecord{},
		dataAgentLeases:        map[string]domain.DataAgentJobLeaseRecord{},
		uploadSessions:         map[string]domain.UploadSessionRecord{},
		uploadFiles:            map[string]domain.UploadSessionFileRecord{},
		uploadChunks:           map[string]domain.UploadChunkRecord{},
		uploadFilesBySession:   map[string]map[string]struct{}{},
		uploadChunksByFile:     map[string]map[string]struct{}{},
		uploadChunksBySession:  map[string]map[string]struct{}{},
		uploadSessionStats:     map[string]uploadSessionStats{},
		uploadEvents:           []domain.UploadSessionEventRecord{},
		users:                  map[string]domain.UserAccount{},
		runTokenUsage:          map[string]domain.RunTokenUsageRecord{},
		runTokenUsageFinalized: map[string]bool{},
		tokenUsageDaily:        map[string]map[string]*domain.UserTokenUsageDaily{},
		tokenUsageLifetime:     map[string]domain.UserTokenUsageStats{},
		orgs:                   map[string]domain.Organization{},
		bisque:                 map[string]domain.BisqueCredentialRecord{},
		leases:                 map[string]domain.RunLeaseRecord{},
		workers:                map[string]domain.WorkerHeartbeatRecord{},
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

func (s *MemoryStore) GetOrganization(ctx context.Context, orgID string) (domain.Organization, bool, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	org, ok := s.orgs[normalizeOrgID(orgID)]
	if !ok {
		return domain.Organization{}, false, nil
	}
	return org, true, nil
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

func (s *MemoryStore) UpdateUserProfile(ctx context.Context, input domain.UpdateUserProfileInput) (domain.UserAccount, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	userID := strings.TrimSpace(input.UserID)
	user, ok := s.users[userID]
	if !ok {
		return domain.UserAccount{}, ErrNotFound
	}
	now := domain.Now()
	if !now.After(user.UpdatedAt) {
		now = user.UpdatedAt.Add(time.Nanosecond)
	}
	metadata := domain.JSONMap{}
	for key, value := range user.Metadata {
		metadata[key] = value
	}
	metadata["profile"] = profileToJSONMap(input.Profile)
	user.Metadata = metadata
	if displayName := strings.TrimSpace(input.Profile.DisplayName); displayName != "" {
		user.DisplayName = displayName
	}
	user.UpdatedAt = now
	s.users[user.UserID] = user
	return user, nil
}

func profileToJSONMap(profile domain.UserProfile) domain.JSONMap {
	data, err := json.Marshal(profile)
	if err != nil {
		return domain.JSONMap{}
	}
	out := domain.JSONMap{}
	if err := json.Unmarshal(data, &out); err != nil {
		return domain.JSONMap{}
	}
	return out
}

func (s *MemoryStore) RecordUserTokenUsage(ctx context.Context, input domain.RecordUserTokenUsageInput) error {
	_ = ctx
	userID := strings.TrimSpace(input.UserID)
	if userID == "" {
		return nil
	}
	now := input.OccurredAt
	if now.IsZero() {
		now = domain.Now()
	}
	day := tokenUsageBucketDay(input.Day, now)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.addUserTokenUsageLocked(userID, day, input.InputTokens, input.OutputTokens, input.TotalTokens, 1, now)
	return nil
}

func (s *MemoryStore) RecordRunTokenUsage(ctx context.Context, input domain.RecordRunTokenUsageInput) (domain.RunTokenUsageRecord, bool, error) {
	_ = ctx
	runID := strings.TrimSpace(input.RunID)
	usageEventID := strings.TrimSpace(input.UsageEventID)
	userID := strings.TrimSpace(input.UserID)
	if runID == "" || usageEventID == "" || userID == "" {
		return domain.RunTokenUsageRecord{}, false, nil
	}
	totalTokens := input.TotalTokens
	if totalTokens <= 0 {
		totalTokens = input.InputTokens + input.OutputTokens
	}
	if totalTokens <= 0 && input.InputTokens <= 0 && input.OutputTokens <= 0 {
		return domain.RunTokenUsageRecord{}, false, nil
	}
	now := input.OccurredAt
	if now.IsZero() {
		now = domain.Now()
	}
	day := tokenUsageBucketDay(input.Day, now)
	key := runTokenUsageKey(runID, usageEventID)
	s.mu.Lock()
	defer s.mu.Unlock()
	if existing, ok := s.runTokenUsage[key]; ok {
		return existing, false, nil
	}
	record := domain.RunTokenUsageRecord{
		RunID:        runID,
		UsageEventID: usageEventID,
		UserID:       userID,
		Model:        strings.TrimSpace(input.Model),
		Day:          day,
		InputTokens:  input.InputTokens,
		OutputTokens: input.OutputTokens,
		TotalTokens:  totalTokens,
		OccurredAt:   now,
		CreatedAt:    now,
	}
	s.runTokenUsage[key] = record
	s.addUserTokenUsageLocked(userID, day, input.InputTokens, input.OutputTokens, totalTokens, 0, now)
	return record, true, nil
}

func (s *MemoryStore) FinalizeRunTokenUsage(ctx context.Context, input domain.FinalizeRunTokenUsageInput) (domain.RunTokenUsageSummary, bool, error) {
	_ = ctx
	runID := strings.TrimSpace(input.RunID)
	if runID == "" {
		return domain.RunTokenUsageSummary{}, false, nil
	}
	completedAt := input.CompletedAt
	if completedAt.IsZero() {
		completedAt = domain.Now()
	}
	day := completedAt.UTC().Truncate(24 * time.Hour)
	s.mu.Lock()
	defer s.mu.Unlock()
	summary := domain.RunTokenUsageSummary{RunID: runID, Day: day}
	for _, record := range s.runTokenUsage {
		if record.RunID != runID {
			continue
		}
		if summary.UserID == "" {
			summary.UserID = record.UserID
		}
		if summary.Model == "" {
			summary.Model = record.Model
		}
		summary.InputTokens += record.InputTokens
		summary.OutputTokens += record.OutputTokens
		summary.TotalTokens += record.TotalTokens
	}
	if summary.UserID == "" || (summary.TotalTokens <= 0 && summary.InputTokens <= 0 && summary.OutputTokens <= 0) {
		return summary, false, nil
	}
	if s.runTokenUsageFinalized[runID] {
		summary.Finalized = true
		return summary, false, nil
	}
	s.runTokenUsageFinalized[runID] = true
	s.addUserTokenUsageLocked(summary.UserID, day, 0, 0, 0, 1, completedAt)
	summary.Finalized = true
	return summary, true, nil
}

func runTokenUsageKey(runID string, usageEventID string) string {
	return runID + "\x00" + usageEventID
}

func tokenUsageBucketDay(day time.Time, occurredAt time.Time) time.Time {
	if day.IsZero() {
		if occurredAt.IsZero() {
			occurredAt = domain.Now()
		}
		return occurredAt.UTC().Truncate(24 * time.Hour)
	}
	return day.UTC().Truncate(24 * time.Hour)
}

func (s *MemoryStore) addUserTokenUsageLocked(userID string, day time.Time, inputTokens int64, outputTokens int64, totalTokens int64, runCount int64, now time.Time) {
	dayKey := day.Format("2006-01-02")
	byDay, ok := s.tokenUsageDaily[userID]
	if !ok {
		byDay = map[string]*domain.UserTokenUsageDaily{}
		s.tokenUsageDaily[userID] = byDay
	}
	daily, ok := byDay[dayKey]
	if !ok {
		daily = &domain.UserTokenUsageDaily{Day: day}
		byDay[dayKey] = daily
	}
	daily.InputTokens += inputTokens
	daily.OutputTokens += outputTokens
	daily.TotalTokens += totalTokens
	daily.RunCount += runCount

	lifetime := s.tokenUsageLifetime[userID]
	lifetime.UserID = userID
	lifetime.InputTokens += inputTokens
	lifetime.OutputTokens += outputTokens
	lifetime.TotalTokens += totalTokens
	if daily.TotalTokens > lifetime.PeakDailyTotal {
		lifetime.PeakDailyTotal = daily.TotalTokens
	}
	lastActive := day
	lifetime.LastActiveDay = &lastActive
	lifetime.UpdatedAt = now
	s.tokenUsageLifetime[userID] = lifetime
}

func (s *MemoryStore) GetUserTokenUsageStats(ctx context.Context, userID string) (domain.UserTokenUsageStats, error) {
	_ = ctx
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return domain.UserTokenUsageStats{}, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	stats, ok := s.tokenUsageLifetime[userID]
	if !ok {
		return domain.UserTokenUsageStats{UserID: userID}, nil
	}
	if stats.LastActiveDay != nil {
		day := *stats.LastActiveDay
		stats.LastActiveDay = &day
	}
	return stats, nil
}

func (s *MemoryStore) ListUserTokenUsageDaily(ctx context.Context, userID string, since time.Time) ([]domain.UserTokenUsageDaily, error) {
	_ = ctx
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return []domain.UserTokenUsageDaily{}, nil
	}
	sinceDay := since.UTC().Truncate(24 * time.Hour)
	s.mu.RLock()
	defer s.mu.RUnlock()
	byDay := s.tokenUsageDaily[userID]
	daily := make([]domain.UserTokenUsageDaily, 0, len(byDay))
	for _, record := range byDay {
		if record.Day.Before(sinceDay) {
			continue
		}
		daily = append(daily, *record)
	}
	sort.Slice(daily, func(i, j int) bool {
		return daily[i].Day.Before(daily[j].Day)
	})
	return daily, nil
}

func (s *MemoryStore) GetUserLongestRunSeconds(ctx context.Context, userID string) (int64, error) {
	_ = ctx
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return 0, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	var longest int64
	for _, run := range s.runs {
		if run.UserID != userID || run.StartedAt == nil || run.CompletedAt == nil {
			continue
		}
		if run.CompletedAt.Before(*run.StartedAt) {
			continue
		}
		seconds := int64(run.CompletedAt.Sub(*run.StartedAt).Seconds())
		if seconds > longest {
			longest = seconds
		}
	}
	return longest, nil
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

func (s *MemoryStore) GetWorkerHeartbeat(ctx context.Context, workerID string) (domain.WorkerHeartbeatRecord, bool, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	worker, ok := s.workers[strings.TrimSpace(workerID)]
	if !ok {
		return domain.WorkerHeartbeatRecord{}, false, nil
	}
	return worker, true, nil
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
	if !ok || thread.UserID != userID || !threadVisibleForUserRead(thread) {
		return domain.ThreadRecord{}, ErrNotFound
	}
	return thread, nil
}

func (s *MemoryStore) UpdateThreadForUser(ctx context.Context, input domain.UpdateThreadInput) (domain.ThreadRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	thread, ok := s.threads[input.ThreadID]
	if !ok || thread.UserID != input.UserID || !threadVisibleForUserRead(thread) {
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

func (s *MemoryStore) SoftDeleteThreadForUser(ctx context.Context, threadID string, userID string, deletedAt time.Time) (domain.ThreadRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	thread, ok := s.threads[threadID]
	if !ok || thread.UserID != userID || thread.Status == domain.ThreadStatusDeleted {
		return domain.ThreadRecord{}, ErrNotFound
	}
	if deletedAt.IsZero() {
		deletedAt = domain.Now()
	}
	thread.Status = domain.ThreadStatusDeleted
	thread.UpdatedAt = deletedAt.UTC()
	s.threads[threadID] = thread
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
	if !ok || thread.UserID != userID || !threadVisibleForUserRead(thread) {
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

// AppendRunEventIfRunActive mirrors the Postgres fast path: a duplicate
// event ID returns the stored record (checked before the live-run gate so
// redelivered terminal events still replay), a missing or terminal run drops
// the event, and otherwise the event is appended.
func (s *MemoryStore) AppendRunEventIfRunActive(ctx context.Context, input domain.AppendRunEventInput) (domain.RunEventRecord, RunEventAppendOutcome, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if input.EventID != "" {
		for _, existing := range s.events[input.RunID] {
			if existing.EventID == input.EventID {
				return existing, RunEventAppendOutcomeDuplicate, nil
			}
		}
	}
	run, ok := s.runs[input.RunID]
	if !ok || isTerminalRunStatus(run.Status) {
		return domain.RunEventRecord{}, RunEventAppendOutcomeDropped, nil
	}
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
		Sequence:     int64(len(s.events[input.RunID]) + 1),
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
	return event, RunEventAppendOutcomeAppended, nil
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

func (s *MemoryStore) CreateUploadSession(ctx context.Context, input domain.CreateUploadSessionInput) (domain.UploadSessionRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	sessionID := strings.TrimSpace(input.SessionID)
	if sessionID == "" {
		sessionID = domain.NewID("upload_session")
	}
	if _, exists := s.uploadSessions[sessionID]; exists {
		return domain.UploadSessionRecord{}, ErrConflict
	}
	now := domain.Now()
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = now
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	sourceType := strings.TrimSpace(input.SourceType)
	if sourceType == "" {
		sourceType = "upload"
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	if ownerUserID == "" {
		ownerUserID = "local-user"
	}
	session := domain.UploadSessionRecord{
		SessionID:          sessionID,
		OwnerUserID:        ownerUserID,
		OwnerOrgID:         strings.TrimSpace(input.OwnerOrgID),
		OwnerRole:          strings.TrimSpace(input.OwnerRole),
		ProjectID:          strings.TrimSpace(input.ProjectID),
		SourceType:         sourceType,
		Status:             status,
		TotalBytes:         input.TotalBytes,
		BytesReceived:      input.BytesReceived,
		BytesVerified:      input.BytesVerified,
		BytesCommitted:     input.BytesCommitted,
		IdempotencyKey:     strings.TrimSpace(input.IdempotencyKey),
		BrowserFingerprint: strings.TrimSpace(input.BrowserFingerprint),
		Error:              strings.TrimSpace(input.Error),
		CreatedAt:          createdAt.UTC(),
		UpdatedAt:          updatedAt.UTC(),
		CompletedAt:        input.CompletedAt,
		Metadata:           mapOrEmpty(input.Metadata),
	}
	s.uploadSessions[session.SessionID] = session
	if _, ok := s.uploadSessionStats[session.SessionID]; !ok {
		s.uploadSessionStats[session.SessionID] = uploadSessionStats{}
	}
	return session, nil
}

func (s *MemoryStore) GetUploadSessionForUser(ctx context.Context, sessionID string, userID string, orgID string) (domain.UploadSessionRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	session, ok := s.uploadSessions[strings.TrimSpace(sessionID)]
	if !ok || !resourceVisibleToOwner(domain.ResourceRecord{OwnerUserID: session.OwnerUserID, OwnerOrgID: session.OwnerOrgID}, userID, orgID) {
		return domain.UploadSessionRecord{}, ErrNotFound
	}
	return session, nil
}

func (s *MemoryStore) GetUploadSessionByIdempotencyKeyForUser(ctx context.Context, idempotencyKey string, userID string, orgID string) (domain.UploadSessionRecord, error) {
	_ = ctx
	idempotencyKey = strings.TrimSpace(idempotencyKey)
	if idempotencyKey == "" {
		return domain.UploadSessionRecord{}, ErrNotFound
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	var matched domain.UploadSessionRecord
	for _, session := range s.uploadSessions {
		if session.IdempotencyKey != idempotencyKey {
			continue
		}
		if !resourceVisibleToOwner(domain.ResourceRecord{OwnerUserID: session.OwnerUserID, OwnerOrgID: session.OwnerOrgID}, userID, orgID) {
			continue
		}
		if matched.SessionID == "" || session.CreatedAt.After(matched.CreatedAt) {
			matched = session
		}
	}
	if matched.SessionID == "" {
		return domain.UploadSessionRecord{}, ErrNotFound
	}
	return matched, nil
}

func (s *MemoryStore) UpdateUploadSession(ctx context.Context, input domain.UpdateUploadSessionInput) (domain.UploadSessionRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	session, ok := s.uploadSessions[strings.TrimSpace(input.SessionID)]
	if !ok || !resourceVisibleToOwner(domain.ResourceRecord{OwnerUserID: session.OwnerUserID, OwnerOrgID: session.OwnerOrgID}, input.OwnerUserID, input.OwnerOrgID) {
		return domain.UploadSessionRecord{}, ErrNotFound
	}
	if strings.TrimSpace(input.Status) != "" {
		session.Status = strings.TrimSpace(input.Status)
	}
	session.BytesReceived = input.BytesReceived
	session.BytesVerified = input.BytesVerified
	session.BytesCommitted = input.BytesCommitted
	session.Error = strings.TrimSpace(input.Error)
	session.UpdatedAt = input.UpdatedAt
	if session.UpdatedAt.IsZero() {
		session.UpdatedAt = domain.Now()
	}
	if !input.CompletedAt.IsZero() {
		session.CompletedAt = input.CompletedAt
	}
	if input.Metadata != nil {
		session.Metadata = mapOrEmpty(input.Metadata)
	}
	s.uploadSessions[session.SessionID] = session
	return session, nil
}

// ClearUploadSessionIdempotencyKey frees a session's idempotency-key slot so a re-upload
// of the same content can claim a fresh session. A no-op if the session does not exist.
func (s *MemoryStore) ClearUploadSessionIdempotencyKey(ctx context.Context, sessionID string) error {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if session, ok := s.uploadSessions[strings.TrimSpace(sessionID)]; ok {
		session.IdempotencyKey = ""
		session.UpdatedAt = domain.Now()
		s.uploadSessions[session.SessionID] = session
	}
	return nil
}

func (s *MemoryStore) UpsertUploadSessionFile(ctx context.Context, input domain.UpsertUploadSessionFileInput) (domain.UploadSessionFileRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.upsertUploadSessionFileLocked(input)
}

func (s *MemoryStore) CreateUploadSessionFiles(ctx context.Context, inputs []domain.UpsertUploadSessionFileInput) ([]domain.UploadSessionFileRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	files := make([]domain.UploadSessionFileRecord, 0, len(inputs))
	for _, input := range inputs {
		file, err := s.upsertUploadSessionFileLocked(input)
		if err != nil {
			return nil, err
		}
		files = append(files, file)
	}
	return files, nil
}

func (s *MemoryStore) upsertUploadSessionFileLocked(input domain.UpsertUploadSessionFileInput) (domain.UploadSessionFileRecord, error) {
	if _, ok := s.uploadSessions[strings.TrimSpace(input.SessionID)]; !ok {
		return domain.UploadSessionFileRecord{}, ErrNotFound
	}
	now := domain.Now()
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		if existing, ok := s.uploadFiles[uploadFileKey(input.SessionID, input.FileToken)]; ok && !existing.CreatedAt.IsZero() {
			createdAt = existing.CreatedAt
		} else {
			createdAt = now
		}
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = now
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	file := domain.UploadSessionFileRecord{
		SessionID:      strings.TrimSpace(input.SessionID),
		FileToken:      strings.TrimSpace(input.FileToken),
		ResourceID:     strings.TrimSpace(input.ResourceID),
		OriginalName:   strings.TrimSpace(input.OriginalName),
		RelativePath:   strings.TrimSpace(input.RelativePath),
		ContentType:    strings.TrimSpace(input.ContentType),
		SizeBytes:      input.SizeBytes,
		DeclaredSHA256: strings.TrimSpace(input.DeclaredSHA256),
		ComputedSHA256: strings.TrimSpace(input.ComputedSHA256),
		Status:         status,
		Error:          strings.TrimSpace(input.Error),
		CreatedAt:      createdAt.UTC(),
		UpdatedAt:      updatedAt.UTC(),
		CompletedAt:    input.CompletedAt,
		Metadata:       mapOrEmpty(input.Metadata),
	}
	fileKey := uploadFileKey(file.SessionID, file.FileToken)
	stats := s.uploadSessionStats[file.SessionID]
	if existing, ok := s.uploadFiles[fileKey]; ok {
		stats = removeUploadFileFromStats(stats, existing)
	} else {
		stats.FileCount++
		if _, ok := s.uploadFilesBySession[file.SessionID]; !ok {
			s.uploadFilesBySession[file.SessionID] = map[string]struct{}{}
		}
		s.uploadFilesBySession[file.SessionID][fileKey] = struct{}{}
	}
	stats = addUploadFileToStats(stats, file)
	s.uploadSessionStats[file.SessionID] = stats
	s.uploadFiles[fileKey] = file
	s.syncUploadSessionStatsLocked(file.SessionID, stats, file.UpdatedAt)
	return file, nil
}

func (s *MemoryStore) ListUploadSessionFiles(ctx context.Context, sessionID string) ([]domain.UploadSessionFileRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	sessionID = strings.TrimSpace(sessionID)
	keys := s.uploadFilesBySession[sessionID]
	files := make([]domain.UploadSessionFileRecord, 0, len(keys))
	for key := range keys {
		if file, ok := s.uploadFiles[key]; ok {
			files = append(files, file)
		}
	}
	sort.Slice(files, func(i, j int) bool {
		return files[i].FileToken < files[j].FileToken
	})
	return files, nil
}

func (s *MemoryStore) GetUploadSessionFile(ctx context.Context, sessionID string, fileToken string) (domain.UploadSessionFileRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	file, ok := s.uploadFiles[uploadFileKey(sessionID, fileToken)]
	if !ok {
		return domain.UploadSessionFileRecord{}, ErrNotFound
	}
	return file, nil
}

func (s *MemoryStore) UpsertUploadChunk(ctx context.Context, input domain.UpsertUploadChunkInput) (domain.UploadChunkRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.uploadFiles[uploadFileKey(input.SessionID, input.FileToken)]; !ok {
		return domain.UploadChunkRecord{}, ErrNotFound
	}
	chunkKey := uploadChunkKey(input.SessionID, input.FileToken, input.ChunkIndex)
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "received"
	}
	chunk := domain.UploadChunkRecord{
		SessionID:  strings.TrimSpace(input.SessionID),
		FileToken:  strings.TrimSpace(input.FileToken),
		ChunkIndex: input.ChunkIndex,
		Offset:     input.Offset,
		SizeBytes:  input.SizeBytes,
		SHA256:     strings.TrimSpace(input.SHA256),
		Status:     status,
		StorageURI: strings.TrimSpace(input.StorageURI),
		ReceivedAt: input.ReceivedAt,
		VerifiedAt: input.VerifiedAt,
		Error:      strings.TrimSpace(input.Error),
		Metadata:   mapOrEmpty(input.Metadata),
	}
	if existing, ok := s.uploadChunks[chunkKey]; ok && existing.Status == "verified" && !sameVerifiedUploadChunk(existing, chunk) {
		return domain.UploadChunkRecord{}, fmt.Errorf("%w: verified upload chunk cannot be replaced with different bytes", ErrConflict)
	}
	stats := s.uploadSessionStats[chunk.SessionID]
	if existing, ok := s.uploadChunks[chunkKey]; ok {
		stats = removeUploadChunkFromStats(stats, existing)
	} else {
		fileKey := uploadFileKey(chunk.SessionID, chunk.FileToken)
		if _, ok := s.uploadChunksByFile[fileKey]; !ok {
			s.uploadChunksByFile[fileKey] = map[string]struct{}{}
		}
		s.uploadChunksByFile[fileKey][chunkKey] = struct{}{}
		if _, ok := s.uploadChunksBySession[chunk.SessionID]; !ok {
			s.uploadChunksBySession[chunk.SessionID] = map[string]struct{}{}
		}
		s.uploadChunksBySession[chunk.SessionID][chunkKey] = struct{}{}
	}
	stats = addUploadChunkToStats(stats, chunk)
	s.uploadSessionStats[chunk.SessionID] = stats
	s.uploadChunks[chunkKey] = chunk
	s.syncUploadSessionStatsLocked(chunk.SessionID, stats, chunk.ReceivedAt)
	return chunk, nil
}

func (s *MemoryStore) syncUploadSessionStatsLocked(sessionID string, stats uploadSessionStats, updatedAt time.Time) {
	session, ok := s.uploadSessions[strings.TrimSpace(sessionID)]
	if !ok {
		return
	}
	session.BytesReceived = stats.BytesReceived
	session.BytesVerified = stats.BytesVerified
	session.BytesCommitted = stats.BytesCommitted
	if updatedAt.IsZero() {
		updatedAt = domain.Now()
	}
	if session.UpdatedAt.IsZero() || !updatedAt.UTC().Before(session.UpdatedAt) {
		session.UpdatedAt = updatedAt.UTC()
	}
	s.uploadSessions[session.SessionID] = session
}

func sameVerifiedUploadChunk(existing domain.UploadChunkRecord, next domain.UploadChunkRecord) bool {
	return existing.Offset == next.Offset &&
		existing.SizeBytes == next.SizeBytes &&
		strings.EqualFold(strings.TrimSpace(existing.SHA256), strings.TrimSpace(next.SHA256))
}

func addUploadFileToStats(stats uploadSessionStats, file domain.UploadSessionFileRecord) uploadSessionStats {
	if file.Status == "completed" {
		stats.CompletedFileCount++
		stats.BytesCommitted += file.SizeBytes
	}
	return stats
}

func removeUploadFileFromStats(stats uploadSessionStats, file domain.UploadSessionFileRecord) uploadSessionStats {
	if file.Status == "completed" {
		stats.CompletedFileCount--
		stats.BytesCommitted -= file.SizeBytes
	}
	return stats
}

func addUploadChunkToStats(stats uploadSessionStats, chunk domain.UploadChunkRecord) uploadSessionStats {
	if chunk.Status == "verified" || chunk.Status == "received" {
		stats.BytesReceived += chunk.SizeBytes
	}
	if chunk.Status == "verified" {
		stats.BytesVerified += chunk.SizeBytes
	}
	return stats
}

func removeUploadChunkFromStats(stats uploadSessionStats, chunk domain.UploadChunkRecord) uploadSessionStats {
	if chunk.Status == "verified" || chunk.Status == "received" {
		stats.BytesReceived -= chunk.SizeBytes
	}
	if chunk.Status == "verified" {
		stats.BytesVerified -= chunk.SizeBytes
	}
	return stats
}

func (s *MemoryStore) ListUploadChunks(ctx context.Context, sessionID string, fileToken string) ([]domain.UploadChunkRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	sessionID = strings.TrimSpace(sessionID)
	fileToken = strings.TrimSpace(fileToken)
	keys := s.uploadChunksByFile[uploadFileKey(sessionID, fileToken)]
	chunks := make([]domain.UploadChunkRecord, 0, len(keys))
	for key := range keys {
		if chunk, ok := s.uploadChunks[key]; ok {
			chunks = append(chunks, chunk)
		}
	}
	sort.Slice(chunks, func(i, j int) bool {
		return chunks[i].ChunkIndex < chunks[j].ChunkIndex
	})
	return chunks, nil
}

func (s *MemoryStore) ListUploadSessionChunks(ctx context.Context, sessionID string) ([]domain.UploadChunkRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	sessionID = strings.TrimSpace(sessionID)
	keys := s.uploadChunksBySession[sessionID]
	chunks := make([]domain.UploadChunkRecord, 0, len(keys))
	for key := range keys {
		if chunk, ok := s.uploadChunks[key]; ok {
			chunks = append(chunks, chunk)
		}
	}
	sort.Slice(chunks, func(i, j int) bool {
		if chunks[i].FileToken == chunks[j].FileToken {
			return chunks[i].ChunkIndex < chunks[j].ChunkIndex
		}
		return chunks[i].FileToken < chunks[j].FileToken
	})
	return chunks, nil
}

func (s *MemoryStore) GetUploadSessionTotals(ctx context.Context, sessionID string) (domain.UploadSessionTotals, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	sessionID = strings.TrimSpace(sessionID)
	if stats, ok := s.uploadSessionStats[sessionID]; ok {
		return domain.UploadSessionTotals{
			BytesReceived:  stats.BytesReceived,
			BytesVerified:  stats.BytesVerified,
			BytesCommitted: stats.BytesCommitted,
			AllComplete:    stats.FileCount > 0 && stats.CompletedFileCount == stats.FileCount,
		}, nil
	}
	var totals domain.UploadSessionTotals
	fileCount := 0
	allComplete := true
	for _, file := range s.uploadFiles {
		if file.SessionID != sessionID {
			continue
		}
		fileCount++
		if file.Status == "completed" {
			totals.BytesCommitted += file.SizeBytes
		} else {
			allComplete = false
		}
	}
	for _, chunk := range s.uploadChunks {
		if chunk.SessionID != sessionID {
			continue
		}
		if chunk.Status == "verified" || chunk.Status == "received" {
			totals.BytesReceived += chunk.SizeBytes
		}
		if chunk.Status == "verified" {
			totals.BytesVerified += chunk.SizeBytes
		}
	}
	totals.AllComplete = fileCount > 0 && allComplete
	return totals, nil
}

func (s *MemoryStore) UploadSessionOperationalMetrics(ctx context.Context) (domain.UploadSessionOperationalMetrics, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	var metrics domain.UploadSessionOperationalMetrics
	for _, session := range s.uploadSessions {
		metrics.AddSession(session)
	}
	return metrics, nil
}

func (s *MemoryStore) AppendUploadSessionEvent(ctx context.Context, input domain.AppendUploadSessionEventInput) (domain.UploadSessionEventRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	sessionID := strings.TrimSpace(input.SessionID)
	if _, ok := s.uploadSessions[sessionID]; !ok {
		return domain.UploadSessionEventRecord{}, ErrNotFound
	}
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("upload_session_event")
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	record := domain.UploadSessionEventRecord{
		EventID:     eventID,
		SessionID:   sessionID,
		ActorUserID: strings.TrimSpace(input.ActorUserID),
		ActorOrgID:  strings.TrimSpace(input.ActorOrgID),
		EventType:   strings.TrimSpace(input.EventType),
		TS:          ts.UTC(),
		Metadata:    mapOrEmpty(input.Metadata),
	}
	s.uploadEvents = append(s.uploadEvents, record)
	return record, nil
}

func (s *MemoryStore) ListUploadSessionEvents(ctx context.Context, sessionID string, limit int) ([]domain.UploadSessionEventRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	sessionID = strings.TrimSpace(sessionID)
	if _, ok := s.uploadSessions[sessionID]; !ok {
		return nil, ErrNotFound
	}
	events := make([]domain.UploadSessionEventRecord, 0, len(s.uploadEvents))
	for _, event := range s.uploadEvents {
		if event.SessionID == sessionID {
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
		Tags:               normalizeResourceTags(input.Tags),
		Metadata:           mapOrEmpty(input.Metadata),
	}
	if len(resource.Tags) == 0 {
		resource.Tags = resourceTagsFromMetadata(resource.Metadata)
	}
	resource.Metadata = resourceMetadataWithTags(resource.Metadata, resource.Tags)
	s.resources[resource.ResourceID] = resource
	return resource, nil
}

func (s *MemoryStore) MergeResourceMetadataForUser(ctx context.Context, input domain.MergeResourceMetadataInput) (domain.ResourceRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	resourceID := strings.TrimSpace(input.ResourceID)
	resource, ok := s.resources[resourceID]
	if !ok || !resourceVisibleToOwner(resource, input.UserID, input.OrgID) {
		return domain.ResourceRecord{}, ErrNotFound
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = domain.Now()
	}
	resource.Metadata = mergeResourceMetadata(resource.Metadata, input.Patch)
	resource.UpdatedAt = updatedAt.UTC()
	s.resources[resource.ResourceID] = resource
	return resource, nil
}

func (s *MemoryStore) RenameResourceForUser(ctx context.Context, input domain.RenameResourceInput) (domain.ResourceRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	resourceID := strings.TrimSpace(input.ResourceID)
	resource, ok := s.resources[resourceID]
	if !ok || !resourceVisibleToOwner(resource, input.UserID, input.OrgID) || strings.TrimSpace(resource.Status) != "active" {
		return domain.ResourceRecord{}, ErrNotFound
	}
	name := strings.TrimSpace(input.OriginalName)
	if name == "" {
		return domain.ResourceRecord{}, ErrNotFound
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = domain.Now()
	}
	resource.OriginalName = name
	resource.UpdatedAt = updatedAt.UTC()
	s.resources[resource.ResourceID] = resource
	return resourceWithNormalizedTags(resource), nil
}

func (s *MemoryStore) GetResourceForUser(ctx context.Context, resourceID string, userID string, orgID string) (domain.ResourceRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	resource, ok := s.resources[strings.TrimSpace(resourceID)]
	if !ok || !s.resourceReadableByUserLocked(resource, userID, orgID) || strings.TrimSpace(resource.Status) != "active" {
		return domain.ResourceRecord{}, ErrNotFound
	}
	resource = resourceWithNormalizedTags(resource)
	resource.ShareSummary = s.resourceShareSummaryLocked(resource, userID, orgID)
	return resource, nil
}

func (s *MemoryStore) GetResourceForOwner(ctx context.Context, resourceID string, userID string, orgID string) (domain.ResourceRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	resource, ok := s.resources[strings.TrimSpace(resourceID)]
	if !ok || !resourceVisibleToOwner(resource, userID, orgID) {
		return domain.ResourceRecord{}, ErrNotFound
	}
	resource = resourceWithNormalizedTags(resource)
	resource.ShareSummary = s.resourceShareSummaryLocked(resource, userID, orgID)
	return resource, nil
}

func (s *MemoryStore) ListResourceIDsForOwner(ctx context.Context, userID string, orgID string, resourceIDs []string) (map[string]bool, error) {
	_ = ctx
	resourceIDs = uniqueTrimmedStrings(resourceIDs)
	existing := make(map[string]bool, len(resourceIDs))
	if len(resourceIDs) == 0 {
		return existing, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, resourceID := range resourceIDs {
		resource, ok := s.resources[resourceID]
		if ok && resourceVisibleToOwner(resource, userID, orgID) {
			existing[resourceID] = true
		}
	}
	return existing, nil
}

func (s *MemoryStore) FindActiveResourceByShaForUser(ctx context.Context, userID string, orgID string, sha256 string, sizeBytes int64) (domain.ResourceRecord, error) {
	_ = ctx
	sha256 = strings.TrimSpace(sha256)
	if sha256 == "" {
		return domain.ResourceRecord{}, ErrNotFound
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	var matched domain.ResourceRecord
	for _, resource := range s.resources {
		if strings.TrimSpace(resource.Status) != "active" {
			continue
		}
		if !resourceVisibleToOwner(resource, userID, orgID) {
			continue
		}
		if strings.TrimSpace(resource.SHA256) != sha256 || resource.SizeBytes != sizeBytes {
			continue
		}
		if matched.ResourceID == "" || resource.CreatedAt.Before(matched.CreatedAt) {
			matched = resource
		}
	}
	if matched.ResourceID == "" {
		return domain.ResourceRecord{}, ErrNotFound
	}
	return matched, nil
}

func (s *MemoryStore) CreateResourceCollection(ctx context.Context, input domain.CreateResourceCollectionInput) (domain.ResourceCollectionRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	collectionID := strings.TrimSpace(input.CollectionID)
	if collectionID == "" {
		collectionID = domain.NewID("collection")
	}
	if _, exists := s.collections[collectionID]; exists {
		return domain.ResourceCollectionRecord{}, ErrConflict
	}
	now := domain.Now()
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	if ownerUserID == "" {
		ownerUserID = "local-user"
	}
	collectionType := strings.ToLower(strings.TrimSpace(input.CollectionType))
	if collectionType == "" {
		collectionType = "collection"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	collection := domain.ResourceCollectionRecord{
		CollectionID:       collectionID,
		OwnerUserID:        ownerUserID,
		OwnerOrgID:         strings.TrimSpace(input.OwnerOrgID),
		OwnerRole:          strings.TrimSpace(input.OwnerRole),
		ProjectID:          strings.TrimSpace(input.ProjectID),
		ParentCollectionID: strings.TrimSpace(input.ParentCollectionID),
		Name:               strings.TrimSpace(input.Name),
		Description:        strings.TrimSpace(input.Description),
		CollectionType:     collectionType,
		Status:             status,
		ResourceCount:      0,
		CreatedAt:          createdAt.UTC(),
		UpdatedAt:          updatedAt.UTC(),
		Metadata:           mapOrEmpty(input.Metadata),
	}
	s.collections[collection.CollectionID] = collection
	return collection, nil
}

func (s *MemoryStore) GetResourceCollectionForUser(ctx context.Context, collectionID string, userID string, orgID string) (domain.ResourceCollectionRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	collection, ok := s.collections[strings.TrimSpace(collectionID)]
	if !ok || !s.collectionReadableByUserLocked(collection, userID, orgID) || strings.TrimSpace(collection.Status) != "active" {
		return domain.ResourceCollectionRecord{}, ErrNotFound
	}
	collection.ResourceCount = s.collectionResourceCountLocked(collection.CollectionID)
	return collection, nil
}

func (s *MemoryStore) RenameResourceCollectionForUser(ctx context.Context, input domain.RenameResourceCollectionInput) (domain.ResourceCollectionRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	collectionID := strings.TrimSpace(input.CollectionID)
	collection, ok := s.collections[collectionID]
	if !ok || !collectionVisibleToOwner(collection, input.UserID, input.OrgID) || strings.TrimSpace(collection.Status) != "active" {
		return domain.ResourceCollectionRecord{}, ErrNotFound
	}
	name := strings.TrimSpace(input.Name)
	if name == "" {
		return domain.ResourceCollectionRecord{}, ErrNotFound
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = domain.Now()
	}
	collection.Name = name
	collection.UpdatedAt = updatedAt.UTC()
	collection.ResourceCount = s.collectionResourceCountLocked(collection.CollectionID)
	s.collections[collection.CollectionID] = collection
	return collection, nil
}

func (s *MemoryStore) ListResourceCollectionsForUser(ctx context.Context, input domain.ResourceCollectionListInput) (domain.ResourceCollectionListPage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	query := strings.ToLower(strings.TrimSpace(input.Query))
	collectionType := strings.ToLower(strings.TrimSpace(input.Type))
	projectID := strings.TrimSpace(input.ProjectID)
	collections := make([]domain.ResourceCollectionRecord, 0, len(s.collections))
	for _, collection := range s.collections {
		if !s.collectionReadableByUserLocked(collection, input.UserID, input.OrgID) {
			continue
		}
		if strings.TrimSpace(collection.Status) != status {
			continue
		}
		if collectionType != "" && collection.CollectionType != collectionType {
			continue
		}
		if projectID != "" && collection.ProjectID != projectID {
			continue
		}
		if query != "" && !resourceCollectionMatchesQuery(collection, query) {
			continue
		}
		collection.ResourceCount = s.collectionResourceCountLocked(collection.CollectionID)
		collections = append(collections, collection)
	}
	sort.Slice(collections, func(i, j int) bool {
		if collections[i].UpdatedAt.Equal(collections[j].UpdatedAt) {
			return collections[i].CollectionID < collections[j].CollectionID
		}
		return collections[i].UpdatedAt.After(collections[j].UpdatedAt)
	})
	total := len(collections)
	offset := input.Offset
	if offset < 0 {
		offset = 0
	}
	if offset >= len(collections) {
		collections = []domain.ResourceCollectionRecord{}
	} else {
		collections = collections[offset:]
	}
	collections = take(collections, input.Limit)
	return domain.ResourceCollectionListPage{
		Collections: collections,
		TotalCount:  total,
		Limit:       input.Limit,
		Offset:      input.Offset,
	}, nil
}

func (s *MemoryStore) SoftDeleteResourceCollectionForUser(ctx context.Context, collectionID string, userID string, orgID string, deletedAt time.Time) (domain.ResourceCollectionRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	collection, ok := s.collections[strings.TrimSpace(collectionID)]
	if !ok || !collectionVisibleToOwner(collection, userID, orgID) || strings.TrimSpace(collection.Status) == "deleted" {
		return domain.ResourceCollectionRecord{}, ErrNotFound
	}
	if deletedAt.IsZero() {
		deletedAt = domain.Now()
	}
	collection.Status = "deleted"
	collection.UpdatedAt = deletedAt.UTC()
	collection.ResourceCount = s.collectionResourceCountLocked(collection.CollectionID)
	s.collections[collection.CollectionID] = collection
	return collection, nil
}

func (s *MemoryStore) RestoreResourceCollectionForUser(ctx context.Context, collectionID string, userID string, orgID string, restoredAt time.Time) (domain.ResourceCollectionRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	collection, ok := s.collections[strings.TrimSpace(collectionID)]
	if !ok || !collectionVisibleToOwner(collection, userID, orgID) {
		return domain.ResourceCollectionRecord{}, ErrNotFound
	}
	if restoredAt.IsZero() {
		restoredAt = domain.Now()
	}
	collection.Status = "active"
	collection.UpdatedAt = restoredAt.UTC()
	collection.ResourceCount = s.collectionResourceCountLocked(collection.CollectionID)
	s.collections[collection.CollectionID] = collection
	return collection, nil
}

func (s *MemoryStore) AddResourcesToCollection(ctx context.Context, input domain.AddResourcesToCollectionInput) (domain.AddResourcesToCollectionResult, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	collectionID := strings.TrimSpace(input.CollectionID)
	collection, ok := s.collections[collectionID]
	if !ok || !collectionVisibleToOwner(collection, input.OwnerUserID, input.OwnerOrgID) || strings.TrimSpace(collection.Status) != "active" {
		return domain.AddResourcesToCollectionResult{}, ErrNotFound
	}
	resourceIDs := uniqueTrimmedStrings(input.ResourceIDs)
	if len(resourceIDs) == 0 {
		collection.ResourceCount = s.collectionResourceCountLocked(collection.CollectionID)
		return domain.AddResourcesToCollectionResult{Collection: collection}, nil
	}
	for _, resourceID := range resourceIDs {
		resource, ok := s.resources[resourceID]
		if !ok || !resourceVisibleToOwner(resource, input.OwnerUserID, input.OwnerOrgID) || strings.TrimSpace(resource.Status) != "active" {
			return domain.AddResourcesToCollectionResult{}, ErrNotFound
		}
	}
	addedAt := input.AddedAt
	if addedAt.IsZero() {
		addedAt = domain.Now()
	}
	nextPosition := s.nextCollectionPositionLocked(collectionID)
	added := make([]domain.ResourceCollectionMembershipRecord, 0, len(resourceIDs))
	inheritedShareGrants := make([]domain.ResourceShareGrantRecord, 0)
	activeCollectionGrants := s.activeCollectionShareGrantsLocked(collectionID)
	addedCount := 0
	for _, resourceID := range resourceIDs {
		key := collectionMemberKey(collectionID, resourceID)
		if existing, ok := s.collectionMembers[key]; ok {
			added = append(added, existing)
			continue
		}
		member := domain.ResourceCollectionMembershipRecord{
			CollectionID:  collectionID,
			ResourceID:    resourceID,
			Position:      nextPosition,
			AddedByUserID: strings.TrimSpace(input.AddedByUserID),
			AddedAt:       addedAt.UTC(),
			Metadata:      mapOrEmpty(input.Metadata),
		}
		nextPosition++
		addedCount++
		s.collectionMembers[key] = member
		added = append(added, member)
		for _, collectionGrant := range activeCollectionGrants {
			grant, err := s.createInheritedResourceShareGrantLocked(resourceID, collectionGrant, addedAt, "resource_collection_share_inherited")
			if err != nil {
				return domain.AddResourcesToCollectionResult{}, err
			}
			inheritedShareGrants = append(inheritedShareGrants, grant)
		}
	}
	collection.UpdatedAt = addedAt.UTC()
	collection.ResourceCount = s.collectionResourceCountLocked(collectionID)
	s.collections[collectionID] = collection
	return domain.AddResourcesToCollectionResult{
		Collection:           collection,
		AddedCount:           addedCount,
		Memberships:          added,
		InheritedShareGrants: inheritedShareGrants,
	}, nil
}

func (s *MemoryStore) RemoveResourcesFromCollection(ctx context.Context, input domain.RemoveResourcesFromCollectionInput) (domain.RemoveResourcesFromCollectionResult, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	collectionID := strings.TrimSpace(input.CollectionID)
	collection, ok := s.collections[collectionID]
	if !ok || !collectionVisibleToOwner(collection, input.OwnerUserID, input.OwnerOrgID) || strings.TrimSpace(collection.Status) != "active" {
		return domain.RemoveResourcesFromCollectionResult{}, ErrNotFound
	}
	resourceIDs := uniqueTrimmedStrings(input.ResourceIDs)
	if len(resourceIDs) == 0 {
		collection.ResourceCount = s.collectionResourceCountLocked(collection.CollectionID)
		return domain.RemoveResourcesFromCollectionResult{Collection: collection}, nil
	}
	for _, resourceID := range resourceIDs {
		resource, ok := s.resources[resourceID]
		if !ok || !resourceVisibleToOwner(resource, input.OwnerUserID, input.OwnerOrgID) {
			return domain.RemoveResourcesFromCollectionResult{}, ErrNotFound
		}
	}
	removedAt := input.RemovedAt
	if removedAt.IsZero() {
		removedAt = domain.Now()
	}
	removed := make([]domain.ResourceCollectionMembershipRecord, 0, len(resourceIDs))
	for _, resourceID := range resourceIDs {
		key := collectionMemberKey(collectionID, resourceID)
		member, ok := s.collectionMembers[key]
		if !ok {
			continue
		}
		delete(s.collectionMembers, key)
		removed = append(removed, member)
	}
	revokedShareGrants := make([]domain.ResourceShareGrantRecord, 0)
	if len(removed) > 0 {
		collectionGrantIDs := s.collectionShareGrantIDsLocked(collectionID)
		removedIDs := map[string]struct{}{}
		for _, member := range removed {
			removedIDs[member.ResourceID] = struct{}{}
		}
		for grantID, grant := range s.resourceGrants {
			if strings.TrimSpace(grant.Status) != "active" {
				continue
			}
			if _, ok := removedIDs[strings.TrimSpace(grant.ResourceID)]; !ok {
				continue
			}
			collectionGrantID := strings.TrimSpace(fmt.Sprint(grant.Metadata["collection_share_grant_id"]))
			if _, ok := collectionGrantIDs[collectionGrantID]; !ok {
				continue
			}
			grant.Status = "revoked"
			grant.UpdatedAt = removedAt.UTC()
			grant.RevokedAt = removedAt.UTC()
			s.resourceGrants[grantID] = grant
			revokedShareGrants = append(revokedShareGrants, grant)
		}
	}
	collection.UpdatedAt = removedAt.UTC()
	collection.ResourceCount = s.collectionResourceCountLocked(collectionID)
	s.collections[collectionID] = collection
	return domain.RemoveResourcesFromCollectionResult{
		Collection:                  collection,
		RemovedCount:                len(removed),
		Memberships:                 removed,
		RevokedInheritedShareGrants: revokedShareGrants,
	}, nil
}

func (s *MemoryStore) CreateResourceCollectionShareGrant(ctx context.Context, input domain.CreateResourceCollectionShareGrantInput) (domain.CreateResourceCollectionShareGrantResult, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	collectionID := strings.TrimSpace(input.CollectionID)
	collection, ok := s.collections[collectionID]
	if !ok || !collectionVisibleToOwner(collection, input.OwnerUserID, input.OwnerOrgID) || strings.TrimSpace(collection.Status) != "active" {
		return domain.CreateResourceCollectionShareGrantResult{}, ErrNotFound
	}
	granteeUserID := strings.TrimSpace(input.GranteeUserID)
	granteeOrgID := strings.TrimSpace(input.GranteeOrgID)
	if input.Public {
		granteeUserID = domain.PublicResourceGranteeUserID
		granteeOrgID = ""
	}
	if granteeUserID == "" && granteeOrgID == "" {
		return domain.CreateResourceCollectionShareGrantResult{}, ErrNotFound
	}
	grantID := strings.TrimSpace(input.GrantID)
	if grantID == "" {
		grantID = domain.NewID("collection_grant")
	}
	if _, exists := s.collectionGrants[grantID]; exists {
		return domain.CreateResourceCollectionShareGrantResult{}, ErrConflict
	}
	role := strings.TrimSpace(input.Role)
	if role == "" {
		role = "read"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = domain.Now()
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}
	createdByUserID := strings.TrimSpace(input.CreatedByUserID)
	if createdByUserID == "" {
		createdByUserID = strings.TrimSpace(input.OwnerUserID)
	}
	grant := domain.ResourceCollectionShareGrantRecord{
		GrantID:         grantID,
		CollectionID:    collection.CollectionID,
		OwnerUserID:     collection.OwnerUserID,
		OwnerOrgID:      collection.OwnerOrgID,
		OwnerRole:       collection.OwnerRole,
		GranteeUserID:   granteeUserID,
		GranteeOrgID:    granteeOrgID,
		Role:            role,
		Status:          status,
		CreatedByUserID: createdByUserID,
		CreatedAt:       createdAt.UTC(),
		UpdatedAt:       updatedAt.UTC(),
		Metadata:        mapOrEmpty(input.Metadata),
	}
	s.collectionGrants[grant.GrantID] = grant
	members := s.collectionMembersForCollectionLocked(collection.CollectionID)
	resourceGrants := make([]domain.ResourceShareGrantRecord, 0, len(members))
	for _, member := range members {
		resource, ok := s.resources[member.ResourceID]
		if !ok || !resourceVisibleToOwner(resource, collection.OwnerUserID, collection.OwnerOrgID) || strings.TrimSpace(resource.Status) != "active" {
			continue
		}
		resourceGrant, err := s.createInheritedResourceShareGrantLocked(resource.ResourceID, grant, createdAt, "resource_collection_share")
		if err != nil {
			return domain.CreateResourceCollectionShareGrantResult{}, err
		}
		resourceGrants = append(resourceGrants, resourceGrant)
	}
	return domain.CreateResourceCollectionShareGrantResult{
		Grant:          grant,
		ResourceGrants: resourceGrants,
	}, nil
}

func (s *MemoryStore) ListResourcesForCollectionForUser(ctx context.Context, input domain.ResourceCollectionResourceListInput) (domain.ResourceListPage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	collection, ok := s.collections[strings.TrimSpace(input.CollectionID)]
	if !ok || !s.collectionReadableByUserLocked(collection, input.UserID, input.OrgID) || strings.TrimSpace(collection.Status) != "active" {
		return domain.ResourceListPage{}, ErrNotFound
	}
	query := strings.ToLower(strings.TrimSpace(input.Query))
	kind := strings.TrimSpace(input.Kind)
	source := strings.TrimSpace(input.Source)
	projectID := strings.TrimSpace(input.ProjectID)
	sharing := strings.ToLower(strings.TrimSpace(input.Sharing))
	members := s.collectionMembersForCollectionLocked(collection.CollectionID)
	resources := make([]domain.ResourceRecord, 0, len(members))
	for _, member := range members {
		resource, ok := s.resources[member.ResourceID]
		if !ok || !s.resourceReadableByUserLocked(resource, input.UserID, input.OrgID) || strings.TrimSpace(resource.Status) != "active" {
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
		if !resourceMatchesTags(resource, input.Tags) {
			continue
		}
		if !resourceMatchesDescriptors(resource, input.Descriptors) {
			continue
		}
		if !resourceMatchesMetadataFilters(resource, input.MetadataFilters) {
			continue
		}
		if !resourceMatchesCreatedRange(resource, input.CreatedAfter, input.CreatedBefore) {
			continue
		}
		if !resourceMatchesProcessingStatus(resource, input.ProcessingStatus) {
			continue
		}
		shareSummary := s.resourceShareSummaryLocked(resource, input.UserID, input.OrgID)
		if !resourceMatchesSharingFilter(shareSummary, sharing) {
			continue
		}
		resource = resourceWithNormalizedTags(resource)
		resource.ShareSummary = shareSummary
		resources = append(resources, resource)
	}
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

func (s *MemoryStore) CreateDatasetSnapshot(ctx context.Context, input domain.CreateDatasetSnapshotInput) (domain.DatasetSnapshotRecord, []domain.DatasetSnapshotResourceRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	snapshotID := strings.TrimSpace(input.SnapshotID)
	if snapshotID == "" {
		snapshotID = domain.NewID("dataset_snapshot")
	}
	if _, exists := s.datasetSnapshots[snapshotID]; exists {
		return domain.DatasetSnapshotRecord{}, nil, ErrConflict
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	if ownerUserID == "" {
		ownerUserID = "local-user"
	}
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	resourceIDs := uniqueTrimmedStrings(input.ResourceIDs)
	if len(resourceIDs) == 0 && input.ResourceQuery != nil {
		resourceIDs = s.datasetSnapshotResourceIDsForQueryLocked(input)
	}
	if len(resourceIDs) == 0 {
		return domain.DatasetSnapshotRecord{}, nil, ErrNotFound
	}
	now := domain.Now()
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	createdByUserID := strings.TrimSpace(input.CreatedByUserID)
	if createdByUserID == "" {
		createdByUserID = ownerUserID
	}
	entries := make([]domain.DatasetSnapshotResourceRecord, 0, len(resourceIDs))
	var totalBytes int64
	for position, resourceID := range resourceIDs {
		resource, ok := s.resources[resourceID]
		if !ok || !resourceVisibleToOwner(resource, ownerUserID, ownerOrgID) || strings.TrimSpace(resource.Status) != "active" {
			return domain.DatasetSnapshotRecord{}, nil, ErrNotFound
		}
		totalBytes += resource.SizeBytes
		entries = append(entries, domain.DatasetSnapshotResourceRecord{
			SnapshotID:        snapshotID,
			ResourceID:        resource.ResourceID,
			Position:          int64(position),
			OriginalName:      resource.OriginalName,
			ContentType:       resource.ContentType,
			SizeBytes:         resource.SizeBytes,
			SHA256:            resource.SHA256,
			SourceType:        resource.SourceType,
			ResourceKind:      resource.ResourceKind,
			StorageURI:        resource.StorageURI,
			SourceURI:         resource.SourceURI,
			ProjectID:         resource.ProjectID,
			ResourceCreatedAt: resource.CreatedAt.UTC(),
			Metadata:          cloneJSONMap(resource.Metadata),
		})
	}
	name := strings.TrimSpace(input.Name)
	if name == "" {
		name = snapshotID
	}
	projectID := strings.TrimSpace(input.ProjectID)
	if projectID == "" && input.ResourceQuery != nil {
		projectID = strings.TrimSpace(input.ResourceQuery.ProjectID)
	}
	snapshot := domain.DatasetSnapshotRecord{
		SnapshotID:         snapshotID,
		OwnerUserID:        ownerUserID,
		OwnerOrgID:         ownerOrgID,
		OwnerRole:          strings.TrimSpace(input.OwnerRole),
		ProjectID:          projectID,
		SourceCollectionID: strings.TrimSpace(input.SourceCollectionID),
		Name:               name,
		Description:        strings.TrimSpace(input.Description),
		Status:             "active",
		ResourceCount:      len(entries),
		TotalBytes:         totalBytes,
		CreatedByUserID:    createdByUserID,
		CreatedAt:          createdAt.UTC(),
		Metadata:           cloneJSONMap(input.Metadata),
	}
	s.datasetSnapshots[snapshotID] = snapshot
	s.datasetEntries[snapshotID] = append([]domain.DatasetSnapshotResourceRecord(nil), entries...)
	s.appendDatasetSnapshotEventLocked(domain.DatasetSnapshotEventRecord{
		SnapshotID:  snapshot.SnapshotID,
		ActorUserID: createdByUserID,
		ActorOrgID:  ownerOrgID,
		EventType:   "dataset_snapshot.created",
		TS:          snapshot.CreatedAt,
		Metadata: domain.JSONMap{
			"snapshot_name":        snapshot.Name,
			"resource_count":       snapshot.ResourceCount,
			"total_bytes":          snapshot.TotalBytes,
			"project_id":           snapshot.ProjectID,
			"source_collection_id": snapshot.SourceCollectionID,
			"source":               snapshot.Metadata["source"],
		},
	})
	return snapshot, append([]domain.DatasetSnapshotResourceRecord(nil), entries...), nil
}

func (s *MemoryStore) datasetSnapshotResourceIDsForQueryLocked(input domain.CreateDatasetSnapshotInput) []string {
	if input.ResourceQuery == nil {
		return nil
	}
	querySelector := input.ResourceQuery
	query := strings.ToLower(strings.TrimSpace(querySelector.Query))
	kind := strings.ToLower(strings.TrimSpace(querySelector.Kind))
	source := strings.ToLower(strings.TrimSpace(querySelector.Source))
	projectID := strings.TrimSpace(querySelector.ProjectID)
	if projectID == "" {
		projectID = strings.TrimSpace(input.ProjectID)
	}
	sharing := strings.ToLower(strings.TrimSpace(querySelector.Sharing))
	sourceCollectionID := strings.TrimSpace(input.SourceCollectionID)
	type candidate struct {
		resource domain.ResourceRecord
		member   domain.ResourceCollectionMembershipRecord
	}
	candidates := make([]candidate, 0, len(s.resources))
	if sourceCollectionID != "" {
		collection, ok := s.collections[sourceCollectionID]
		if !ok || !collectionVisibleToOwner(collection, input.OwnerUserID, input.OwnerOrgID) || strings.TrimSpace(collection.Status) != "active" {
			return nil
		}
		for _, member := range s.collectionMembers {
			if member.CollectionID != sourceCollectionID {
				continue
			}
			resource, ok := s.resources[member.ResourceID]
			if ok {
				candidates = append(candidates, candidate{resource: resource, member: member})
			}
		}
		sort.Slice(candidates, func(i, j int) bool {
			if candidates[i].member.Position == candidates[j].member.Position {
				return candidates[i].member.ResourceID < candidates[j].member.ResourceID
			}
			return candidates[i].member.Position < candidates[j].member.Position
		})
	} else {
		for _, resource := range s.resources {
			candidates = append(candidates, candidate{resource: resource})
		}
		sort.Slice(candidates, func(i, j int) bool {
			if candidates[i].resource.CreatedAt.Equal(candidates[j].resource.CreatedAt) {
				return candidates[i].resource.ResourceID < candidates[j].resource.ResourceID
			}
			return candidates[i].resource.CreatedAt.After(candidates[j].resource.CreatedAt)
		})
	}
	resourceIDs := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		resource := candidate.resource
		if !resourceVisibleToOwner(resource, input.OwnerUserID, input.OwnerOrgID) {
			continue
		}
		if strings.TrimSpace(resource.Status) != "active" {
			continue
		}
		if kind != "" && strings.ToLower(strings.TrimSpace(resource.ResourceKind)) != kind {
			continue
		}
		if source != "" && strings.ToLower(strings.TrimSpace(resource.SourceType)) != source {
			continue
		}
		if projectID != "" && strings.TrimSpace(resource.ProjectID) != projectID {
			continue
		}
		if query != "" && !resourceRecordMatchesQuery(resource, query) {
			continue
		}
		if !resourceMatchesTags(resource, querySelector.Tags) {
			continue
		}
		if !resourceMatchesDescriptors(resource, querySelector.Descriptors) {
			continue
		}
		if !resourceMatchesMetadataFilters(resource, querySelector.MetadataFilters) {
			continue
		}
		if !resourceMatchesCreatedRange(resource, querySelector.CreatedAfter, querySelector.CreatedBefore) {
			continue
		}
		if !resourceMatchesProcessingStatus(resource, querySelector.ProcessingStatus) {
			continue
		}
		shareSummary := s.resourceShareSummaryLocked(resource, input.OwnerUserID, input.OwnerOrgID)
		if !resourceMatchesSharingFilter(shareSummary, sharing) {
			continue
		}
		resourceIDs = append(resourceIDs, resource.ResourceID)
	}
	return resourceIDs
}

func (s *MemoryStore) GetDatasetSnapshotForUser(ctx context.Context, snapshotID string, userID string, orgID string) (domain.DatasetSnapshotRecord, []domain.DatasetSnapshotResourceRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	snapshot, ok := s.datasetSnapshots[strings.TrimSpace(snapshotID)]
	if !ok || !s.datasetSnapshotVisibleToUserLocked(snapshot, userID, orgID) || strings.TrimSpace(snapshot.Status) != "active" {
		return domain.DatasetSnapshotRecord{}, nil, ErrNotFound
	}
	entries := append([]domain.DatasetSnapshotResourceRecord(nil), s.datasetEntries[snapshot.SnapshotID]...)
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Position == entries[j].Position {
			return entries[i].ResourceID < entries[j].ResourceID
		}
		return entries[i].Position < entries[j].Position
	})
	return snapshot, entries, nil
}

func (s *MemoryStore) ListDatasetSnapshotsForUser(ctx context.Context, input domain.DatasetSnapshotListInput) (domain.DatasetSnapshotListPage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	query := strings.ToLower(strings.TrimSpace(input.Query))
	projectID := strings.TrimSpace(input.ProjectID)
	sourceCollectionID := strings.TrimSpace(input.SourceCollectionID)
	snapshots := make([]domain.DatasetSnapshotRecord, 0, len(s.datasetSnapshots))
	for _, snapshot := range s.datasetSnapshots {
		visible := s.datasetSnapshotVisibleToUserLocked(snapshot, input.UserID, input.OrgID)
		if status != "active" {
			visible = datasetSnapshotVisibleToOwner(snapshot, input.UserID, input.OrgID)
		}
		if !visible {
			continue
		}
		if strings.TrimSpace(snapshot.Status) != status {
			continue
		}
		if projectID != "" && strings.TrimSpace(snapshot.ProjectID) != projectID {
			continue
		}
		if sourceCollectionID != "" && strings.TrimSpace(snapshot.SourceCollectionID) != sourceCollectionID {
			continue
		}
		if query != "" && !datasetSnapshotMatchesQuery(snapshot, query) {
			continue
		}
		snapshots = append(snapshots, snapshot)
	}
	sort.Slice(snapshots, func(i, j int) bool {
		if snapshots[i].CreatedAt.Equal(snapshots[j].CreatedAt) {
			return snapshots[i].SnapshotID < snapshots[j].SnapshotID
		}
		return snapshots[i].CreatedAt.After(snapshots[j].CreatedAt)
	})
	total := len(snapshots)
	offset := input.Offset
	if offset < 0 {
		offset = 0
	}
	if offset >= len(snapshots) {
		snapshots = []domain.DatasetSnapshotRecord{}
	} else {
		snapshots = snapshots[offset:]
	}
	snapshots = take(snapshots, input.Limit)
	return domain.DatasetSnapshotListPage{
		Snapshots:  snapshots,
		TotalCount: total,
		Limit:      input.Limit,
		Offset:     input.Offset,
	}, nil
}

func (s *MemoryStore) SoftDeleteDatasetSnapshotForUser(ctx context.Context, snapshotID string, userID string, orgID string, deletedAt time.Time) (domain.DatasetSnapshotRecord, []domain.DatasetSnapshotResourceRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	snapshot, ok := s.datasetSnapshots[strings.TrimSpace(snapshotID)]
	if !ok || !datasetSnapshotVisibleToOwner(snapshot, userID, orgID) || strings.TrimSpace(snapshot.Status) == "deleted" {
		return domain.DatasetSnapshotRecord{}, nil, ErrNotFound
	}
	if deletedAt.IsZero() {
		deletedAt = domain.Now()
	}
	snapshot.Status = "deleted"
	s.datasetSnapshots[snapshot.SnapshotID] = snapshot
	entries := append([]domain.DatasetSnapshotResourceRecord(nil), s.datasetEntries[snapshot.SnapshotID]...)
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Position == entries[j].Position {
			return entries[i].ResourceID < entries[j].ResourceID
		}
		return entries[i].Position < entries[j].Position
	})
	s.appendDatasetSnapshotEventLocked(domain.DatasetSnapshotEventRecord{
		SnapshotID:  snapshot.SnapshotID,
		ActorUserID: strings.TrimSpace(userID),
		ActorOrgID:  strings.TrimSpace(orgID),
		EventType:   "dataset_snapshot.deleted",
		TS:          afterEventTime(s.latestDatasetSnapshotEventTimeLocked(snapshot.SnapshotID, snapshot.CreatedAt), deletedAt),
		Metadata: domain.JSONMap{
			"snapshot_id":          snapshot.SnapshotID,
			"snapshot_name":        snapshot.Name,
			"resource_count":       snapshot.ResourceCount,
			"total_bytes":          snapshot.TotalBytes,
			"project_id":           snapshot.ProjectID,
			"source_collection_id": snapshot.SourceCollectionID,
			"source":               "dataset_snapshot_lifecycle",
			"deleted_at":           deletedAt.UTC().Format(time.RFC3339Nano),
		},
	})
	return snapshot, entries, nil
}

func (s *MemoryStore) RestoreDatasetSnapshotForUser(ctx context.Context, snapshotID string, userID string, orgID string, restoredAt time.Time) (domain.DatasetSnapshotRecord, []domain.DatasetSnapshotResourceRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	snapshot, ok := s.datasetSnapshots[strings.TrimSpace(snapshotID)]
	if !ok || !datasetSnapshotVisibleToOwner(snapshot, userID, orgID) {
		return domain.DatasetSnapshotRecord{}, nil, ErrNotFound
	}
	if restoredAt.IsZero() {
		restoredAt = domain.Now()
	}
	snapshot.Status = "active"
	s.datasetSnapshots[snapshot.SnapshotID] = snapshot
	entries := append([]domain.DatasetSnapshotResourceRecord(nil), s.datasetEntries[snapshot.SnapshotID]...)
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Position == entries[j].Position {
			return entries[i].ResourceID < entries[j].ResourceID
		}
		return entries[i].Position < entries[j].Position
	})
	s.appendDatasetSnapshotEventLocked(domain.DatasetSnapshotEventRecord{
		SnapshotID:  snapshot.SnapshotID,
		ActorUserID: strings.TrimSpace(userID),
		ActorOrgID:  strings.TrimSpace(orgID),
		EventType:   "dataset_snapshot.restored",
		TS:          afterEventTime(s.latestDatasetSnapshotEventTimeLocked(snapshot.SnapshotID, snapshot.CreatedAt), restoredAt),
		Metadata: domain.JSONMap{
			"snapshot_id":          snapshot.SnapshotID,
			"snapshot_name":        snapshot.Name,
			"resource_count":       snapshot.ResourceCount,
			"total_bytes":          snapshot.TotalBytes,
			"project_id":           snapshot.ProjectID,
			"source_collection_id": snapshot.SourceCollectionID,
			"source":               "dataset_snapshot_lifecycle",
			"restored_at":          restoredAt.UTC().Format(time.RFC3339Nano),
		},
	})
	return snapshot, entries, nil
}

func (s *MemoryStore) CreateDatasetSnapshotShareGrant(ctx context.Context, input domain.CreateDatasetSnapshotShareGrantInput) (domain.DatasetSnapshotShareGrantRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	snapshotID := strings.TrimSpace(input.SnapshotID)
	snapshot, ok := s.datasetSnapshots[snapshotID]
	if !ok || !datasetSnapshotVisibleToOwner(snapshot, input.OwnerUserID, input.OwnerOrgID) || strings.TrimSpace(snapshot.Status) != "active" {
		return domain.DatasetSnapshotShareGrantRecord{}, ErrNotFound
	}
	granteeUserID := strings.TrimSpace(input.GranteeUserID)
	granteeOrgID := strings.TrimSpace(input.GranteeOrgID)
	if granteeUserID == "" && granteeOrgID == "" {
		return domain.DatasetSnapshotShareGrantRecord{}, ErrNotFound
	}
	grantID := strings.TrimSpace(input.GrantID)
	if grantID == "" {
		grantID = domain.NewID("dataset_snapshot_grant")
	}
	if _, exists := s.datasetGrants[grantID]; exists {
		return domain.DatasetSnapshotShareGrantRecord{}, ErrConflict
	}
	role := strings.TrimSpace(input.Role)
	if role == "" {
		role = "read"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = domain.Now()
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}
	createdByUserID := strings.TrimSpace(input.CreatedByUserID)
	if createdByUserID == "" {
		createdByUserID = strings.TrimSpace(input.OwnerUserID)
	}
	grant := domain.DatasetSnapshotShareGrantRecord{
		GrantID:         grantID,
		SnapshotID:      snapshot.SnapshotID,
		OwnerUserID:     snapshot.OwnerUserID,
		OwnerOrgID:      snapshot.OwnerOrgID,
		OwnerRole:       snapshot.OwnerRole,
		GranteeUserID:   granteeUserID,
		GranteeOrgID:    granteeOrgID,
		Role:            role,
		Status:          status,
		CreatedByUserID: createdByUserID,
		CreatedAt:       createdAt.UTC(),
		UpdatedAt:       updatedAt.UTC(),
		Metadata:        mapOrEmpty(input.Metadata),
	}
	s.datasetGrants[grant.GrantID] = grant
	s.appendDatasetSnapshotEventLocked(domain.DatasetSnapshotEventRecord{
		SnapshotID:  snapshot.SnapshotID,
		ActorUserID: createdByUserID,
		ActorOrgID:  snapshot.OwnerOrgID,
		EventType:   "dataset_snapshot.shared",
		TS:          afterEventTime(s.latestDatasetSnapshotEventTimeLocked(snapshot.SnapshotID, snapshot.CreatedAt), grant.CreatedAt),
		Metadata: domain.JSONMap{
			"grant_id":        grant.GrantID,
			"grantee_user_id": grant.GranteeUserID,
			"grantee_org_id":  grant.GranteeOrgID,
			"role":            grant.Role,
		},
	})
	return grant, nil
}

func (s *MemoryStore) ListDatasetSnapshotShareGrants(ctx context.Context, input domain.ListDatasetSnapshotShareGrantsInput) ([]domain.DatasetSnapshotShareGrantRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	snapshotID := strings.TrimSpace(input.SnapshotID)
	snapshot, ok := s.datasetSnapshots[snapshotID]
	if !ok || !datasetSnapshotVisibleToOwner(snapshot, input.OwnerUserID, input.OwnerOrgID) || strings.TrimSpace(snapshot.Status) != "active" {
		return nil, ErrNotFound
	}
	status := strings.TrimSpace(input.Status)
	grants := make([]domain.DatasetSnapshotShareGrantRecord, 0)
	for _, grant := range s.datasetGrants {
		if strings.TrimSpace(grant.SnapshotID) != snapshotID {
			continue
		}
		if status != "" && strings.TrimSpace(grant.Status) != status {
			continue
		}
		grants = append(grants, grant)
	}
	sort.Slice(grants, func(i, j int) bool {
		if grants[i].CreatedAt.Equal(grants[j].CreatedAt) {
			return grants[i].GrantID < grants[j].GrantID
		}
		return grants[i].CreatedAt.After(grants[j].CreatedAt)
	})
	return take(grants, input.Limit), nil
}

func (s *MemoryStore) RevokeDatasetSnapshotShareGrant(ctx context.Context, input domain.RevokeDatasetSnapshotShareGrantInput) (domain.DatasetSnapshotShareGrantRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	snapshotID := strings.TrimSpace(input.SnapshotID)
	snapshot, ok := s.datasetSnapshots[snapshotID]
	if !ok || !datasetSnapshotVisibleToOwner(snapshot, input.OwnerUserID, input.OwnerOrgID) || strings.TrimSpace(snapshot.Status) != "active" {
		return domain.DatasetSnapshotShareGrantRecord{}, ErrNotFound
	}
	grantID := strings.TrimSpace(input.GrantID)
	grant, ok := s.datasetGrants[grantID]
	if !ok || strings.TrimSpace(grant.SnapshotID) != snapshotID || strings.TrimSpace(grant.Status) != "active" {
		return domain.DatasetSnapshotShareGrantRecord{}, ErrNotFound
	}
	revokedAt := input.RevokedAt
	if revokedAt.IsZero() {
		revokedAt = domain.Now()
	}
	metadata := cloneJSONMap(grant.Metadata)
	if revokedBy := strings.TrimSpace(input.RevokedByUserID); revokedBy != "" {
		metadata["revoked_by_user_id"] = revokedBy
	}
	if reason := strings.TrimSpace(input.RevocationReason); reason != "" {
		metadata["revocation_reason"] = reason
	}
	grant.Status = "revoked"
	grant.UpdatedAt = revokedAt.UTC()
	grant.RevokedAt = revokedAt.UTC()
	grant.Metadata = metadata
	s.datasetGrants[grant.GrantID] = grant
	actorUserID := strings.TrimSpace(input.RevokedByUserID)
	if actorUserID == "" {
		actorUserID = strings.TrimSpace(input.OwnerUserID)
	}
	s.appendDatasetSnapshotEventLocked(domain.DatasetSnapshotEventRecord{
		SnapshotID:  snapshot.SnapshotID,
		ActorUserID: actorUserID,
		ActorOrgID:  snapshot.OwnerOrgID,
		EventType:   "dataset_snapshot.share_revoked",
		TS:          afterEventTime(s.latestDatasetSnapshotEventTimeLocked(snapshot.SnapshotID, grant.CreatedAt), grant.RevokedAt),
		Metadata: domain.JSONMap{
			"grant_id":        grant.GrantID,
			"grantee_user_id": grant.GranteeUserID,
			"grantee_org_id":  grant.GranteeOrgID,
			"role":            grant.Role,
		},
	})
	return grant, nil
}

func (s *MemoryStore) ListDatasetSnapshotEventsForUser(ctx context.Context, input domain.DatasetSnapshotEventListInput) (domain.DatasetSnapshotEventListPage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	snapshotID := strings.TrimSpace(input.SnapshotID)
	if snapshotID == "" {
		return domain.DatasetSnapshotEventListPage{}, ErrNotFound
	}
	snapshot, ok := s.datasetSnapshots[snapshotID]
	if !ok {
		return domain.DatasetSnapshotEventListPage{}, ErrNotFound
	}
	visible := false
	if strings.TrimSpace(snapshot.Status) == "active" {
		visible = s.datasetSnapshotVisibleToUserLocked(snapshot, input.UserID, input.OrgID)
	} else {
		visible = datasetSnapshotVisibleToOwner(snapshot, input.UserID, input.OrgID)
	}
	if !visible {
		return domain.DatasetSnapshotEventListPage{}, ErrNotFound
	}
	eventType := strings.TrimSpace(input.EventType)
	actorUserID := strings.TrimSpace(input.ActorUserID)
	events := make([]domain.DatasetSnapshotEventRecord, 0, len(s.datasetEvents))
	for _, event := range s.datasetEvents {
		if strings.TrimSpace(event.SnapshotID) != snapshotID {
			continue
		}
		if eventType != "" && strings.TrimSpace(event.EventType) != eventType {
			continue
		}
		if actorUserID != "" && strings.TrimSpace(event.ActorUserID) != actorUserID {
			continue
		}
		events = append(events, event)
	}
	sort.Slice(events, func(i, j int) bool {
		if events[i].TS.Equal(events[j].TS) {
			return events[i].EventID < events[j].EventID
		}
		return events[i].TS.After(events[j].TS)
	})
	total := len(events)
	offset := input.Offset
	if offset < 0 {
		offset = 0
	}
	if offset >= len(events) {
		events = []domain.DatasetSnapshotEventRecord{}
	} else {
		events = events[offset:]
	}
	limit := input.Limit
	if limit <= 0 {
		limit = 200
	}
	events = take(events, limit)
	return domain.DatasetSnapshotEventListPage{
		Events:     events,
		TotalCount: total,
		Limit:      limit,
		Offset:     offset,
	}, nil
}

func (s *MemoryStore) CreateDataAgentJob(ctx context.Context, input domain.CreateDataAgentJobInput) (domain.DataAgentJobRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	jobID := strings.TrimSpace(input.JobID)
	if jobID == "" {
		jobID = domain.NewID("data_agent_job")
	}
	if _, exists := s.dataAgentJobs[jobID]; exists {
		return domain.DataAgentJobRecord{}, ErrConflict
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	if ownerUserID == "" {
		ownerUserID = "local-user"
	}
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	resourceIDs := uniqueTrimmedStrings(input.ResourceIDs)
	for _, resourceID := range resourceIDs {
		resource, ok := s.resources[resourceID]
		if !ok || !resourceVisibleToOwner(resource, ownerUserID, ownerOrgID) || strings.TrimSpace(resource.Status) != "active" {
			return domain.DataAgentJobRecord{}, ErrNotFound
		}
	}
	now := domain.Now()
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}
	status := strings.ToLower(strings.TrimSpace(input.Status))
	if status == "" {
		status = "queued"
	}
	jobType := strings.ToLower(strings.TrimSpace(input.JobType))
	resourceCount := input.ResourceCount
	if len(resourceIDs) > 0 || resourceCount == 0 {
		resourceCount = len(resourceIDs)
	}
	progressTotal := input.ProgressTotal
	if progressTotal == 0 {
		progressTotal = resourceCount
	}
	createdByUserID := strings.TrimSpace(input.CreatedByUserID)
	if createdByUserID == "" {
		createdByUserID = ownerUserID
	}
	inputSelector := cloneJSONMap(input.InputSelector)
	if len(inputSelector) == 0 && len(resourceIDs) > 0 {
		inputSelector["resource_ids"] = append([]string(nil), resourceIDs...)
	}
	job := domain.DataAgentJobRecord{
		JobID:             jobID,
		OwnerUserID:       ownerUserID,
		OwnerOrgID:        ownerOrgID,
		OwnerRole:         strings.TrimSpace(input.OwnerRole),
		ProjectID:         strings.TrimSpace(input.ProjectID),
		JobType:           jobType,
		Status:            status,
		ResourceCount:     resourceCount,
		ProgressCompleted: input.ProgressCompleted,
		ProgressTotal:     progressTotal,
		Error:             strings.TrimSpace(input.Error),
		CreatedByUserID:   createdByUserID,
		CreatedAt:         createdAt.UTC(),
		UpdatedAt:         updatedAt.UTC(),
		StartedAt:         input.StartedAt,
		CompletedAt:       input.CompletedAt,
		InputSelector:     inputSelector,
		OutputSummary:     cloneJSONMap(input.OutputSummary),
		Metadata:          cloneJSONMap(input.Metadata),
	}
	s.dataAgentJobs[job.JobID] = job
	_, err := s.appendDataAgentJobEventLocked(domain.AppendDataAgentJobEventInput{
		JobID:       job.JobID,
		EventType:   "data_agent.job.created",
		ActorUserID: createdByUserID,
		ActorOrgID:  ownerOrgID,
		TS:          job.CreatedAt,
		Message:     "Data Agent job queued.",
		Metadata: domain.JSONMap{
			"job_type":       job.JobType,
			"resource_count": job.ResourceCount,
		},
	})
	if err != nil {
		return domain.DataAgentJobRecord{}, err
	}
	return job, nil
}

func (s *MemoryStore) GetDataAgentJobForUser(ctx context.Context, jobID string, userID string, orgID string) (domain.DataAgentJobRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	job, ok := s.dataAgentJobs[strings.TrimSpace(jobID)]
	if !ok || !dataAgentJobVisibleToOwner(job, userID, orgID) {
		return domain.DataAgentJobRecord{}, ErrNotFound
	}
	return job, nil
}

func (s *MemoryStore) ListDataAgentJobsForUser(ctx context.Context, input domain.DataAgentJobListInput) (domain.DataAgentJobListPage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	jobType := strings.ToLower(strings.TrimSpace(input.JobType))
	status := strings.ToLower(strings.TrimSpace(input.Status))
	projectID := strings.TrimSpace(input.ProjectID)
	jobs := make([]domain.DataAgentJobRecord, 0, len(s.dataAgentJobs))
	for _, job := range s.dataAgentJobs {
		if !dataAgentJobVisibleToOwner(job, input.UserID, input.OrgID) {
			continue
		}
		if jobType != "" && strings.TrimSpace(job.JobType) != jobType {
			continue
		}
		if status != "" && strings.TrimSpace(job.Status) != status {
			continue
		}
		if projectID != "" && strings.TrimSpace(job.ProjectID) != projectID {
			continue
		}
		jobs = append(jobs, job)
	}
	sort.Slice(jobs, func(i, j int) bool {
		if jobs[i].UpdatedAt.Equal(jobs[j].UpdatedAt) {
			return jobs[i].JobID < jobs[j].JobID
		}
		return jobs[i].UpdatedAt.After(jobs[j].UpdatedAt)
	})
	total := len(jobs)
	offset := input.Offset
	if offset < 0 {
		offset = 0
	}
	if offset >= len(jobs) {
		jobs = []domain.DataAgentJobRecord{}
	} else {
		jobs = jobs[offset:]
	}
	jobs = take(jobs, input.Limit)
	return domain.DataAgentJobListPage{
		Jobs:       jobs,
		TotalCount: total,
		Limit:      input.Limit,
		Offset:     input.Offset,
	}, nil
}

func (s *MemoryStore) UpdateDataAgentJob(ctx context.Context, input domain.UpdateDataAgentJobInput) (domain.DataAgentJobRecord, domain.DataAgentJobEventRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.dataAgentJobs[strings.TrimSpace(input.JobID)]
	if !ok || !dataAgentJobVisibleToOwner(job, input.OwnerUserID, input.OwnerOrgID) {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, ErrNotFound
	}
	status := strings.ToLower(strings.TrimSpace(input.Status))
	if status == "" {
		status = job.Status
	}
	if !validDataAgentJobStatus(status) {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, errors.New("invalid data agent job status")
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = domain.Now()
	}
	job.Status = status
	job.ProgressCompleted = clampDataAgentProgress(input.ProgressCompleted, input.ProgressTotal)
	if input.ProgressTotal > 0 {
		job.ProgressTotal = input.ProgressTotal
	}
	if job.ProgressTotal < job.ProgressCompleted {
		job.ProgressTotal = job.ProgressCompleted
	}
	job.Error = strings.TrimSpace(input.Error)
	job.UpdatedAt = updatedAt.UTC()
	if !input.StartedAt.IsZero() {
		job.StartedAt = input.StartedAt.UTC()
	} else if status == "running" && job.StartedAt.IsZero() {
		job.StartedAt = updatedAt.UTC()
	}
	if !input.CompletedAt.IsZero() {
		job.CompletedAt = input.CompletedAt.UTC()
	} else if dataAgentJobStatusIsTerminal(status) {
		job.CompletedAt = updatedAt.UTC()
	} else if !dataAgentJobStatusIsTerminal(status) {
		job.CompletedAt = time.Time{}
	}
	if input.OutputSummary != nil {
		job.OutputSummary = cloneJSONMap(input.OutputSummary)
	}
	if input.Metadata != nil {
		job.Metadata = cloneJSONMap(input.Metadata)
	}
	s.dataAgentJobs[job.JobID] = job
	eventMetadata := cloneJSONMap(input.EventMetadata)
	if len(eventMetadata) == 0 {
		eventMetadata["status"] = status
	}
	event, err := s.appendDataAgentJobEventLocked(domain.AppendDataAgentJobEventInput{
		JobID:       job.JobID,
		EventType:   dataAgentJobStatusEventType(status),
		ActorUserID: strings.TrimSpace(input.ActorUserID),
		ActorOrgID:  strings.TrimSpace(input.ActorOrgID),
		TS:          updatedAt,
		Message:     strings.TrimSpace(input.Message),
		Metadata:    eventMetadata,
	})
	if err != nil {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, err
	}
	return job, event, nil
}

func (s *MemoryStore) ControlDataAgentJob(ctx context.Context, input domain.ControlDataAgentJobInput) (domain.DataAgentJobRecord, domain.DataAgentJobEventRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.dataAgentJobs[strings.TrimSpace(input.JobID)]
	if !ok || !dataAgentJobVisibleToOwner(job, input.OwnerUserID, input.OwnerOrgID) {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, ErrNotFound
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	action := strings.ToLower(strings.TrimSpace(input.Action))
	reason := strings.TrimSpace(input.Reason)
	eventType := ""
	switch action {
	case "cancel":
		job.Status = "canceled"
		job.Error = reason
		job.CompletedAt = ts.UTC()
		eventType = "data_agent.job.canceled"
	case "retry":
		job.Status = "queued"
		job.ProgressCompleted = 0
		if job.ProgressTotal == 0 {
			job.ProgressTotal = job.ResourceCount
		}
		job.Error = ""
		job.StartedAt = time.Time{}
		job.CompletedAt = time.Time{}
		eventType = "data_agent.job.retried"
	default:
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, errors.New("data agent job control action must be cancel or retry")
	}
	job.UpdatedAt = ts.UTC()
	s.dataAgentJobs[job.JobID] = job
	event, err := s.appendDataAgentJobEventLocked(domain.AppendDataAgentJobEventInput{
		JobID:       job.JobID,
		EventType:   eventType,
		ActorUserID: strings.TrimSpace(input.ActorUserID),
		ActorOrgID:  strings.TrimSpace(input.ActorOrgID),
		TS:          ts,
		Message:     reason,
		Metadata:    cloneJSONMap(input.Metadata),
	})
	if err != nil {
		return domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, err
	}
	return job, event, nil
}

func (s *MemoryStore) AcquireDataAgentJobLease(ctx context.Context, input domain.AcquireDataAgentJobLeaseInput) (domain.DataAgentJobLeaseRecord, domain.DataAgentJobRecord, domain.DataAgentJobEventRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.dataAgentJobs[strings.TrimSpace(input.JobID)]
	if !ok || !dataAgentJobVisibleToOwner(job, input.OwnerUserID, input.OwnerOrgID) {
		return domain.DataAgentJobLeaseRecord{}, domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, ErrNotFound
	}
	if dataAgentJobStatusIsTerminal(job.Status) {
		return domain.DataAgentJobLeaseRecord{}, domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	if existing, ok := s.dataAgentLeases[job.JobID]; ok && existing.LeaseExpiresAt.After(now) {
		return domain.DataAgentJobLeaseRecord{}, domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, ErrConflict
	}
	lease := domain.DataAgentJobLeaseRecord{
		JobID:          job.JobID,
		WorkerID:       strings.TrimSpace(input.WorkerID),
		LeaseToken:     domain.NewID("lease"),
		LeaseExpiresAt: now.Add(positiveLeaseTTL(input.TTL)),
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	s.dataAgentLeases[job.JobID] = lease
	job.Status = "running"
	job.UpdatedAt = now
	if job.StartedAt.IsZero() {
		job.StartedAt = now
	}
	s.dataAgentJobs[job.JobID] = job
	event, err := s.appendDataAgentJobEventLocked(domain.AppendDataAgentJobEventInput{
		JobID:       job.JobID,
		EventType:   "data_agent.job.leased",
		ActorUserID: strings.TrimSpace(input.OwnerUserID),
		ActorOrgID:  strings.TrimSpace(input.OwnerOrgID),
		TS:          now,
		Message:     "Data Agent job leased.",
		Metadata: domain.JSONMap{
			"worker_id":        lease.WorkerID,
			"lease_expires_at": lease.LeaseExpiresAt.UTC().Format(time.RFC3339Nano),
		},
	})
	if err != nil {
		return domain.DataAgentJobLeaseRecord{}, domain.DataAgentJobRecord{}, domain.DataAgentJobEventRecord{}, err
	}
	return lease, job, event, nil
}

func (s *MemoryStore) RenewDataAgentJobLease(ctx context.Context, input domain.RenewDataAgentJobLeaseInput) (domain.DataAgentJobLeaseRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.dataAgentJobs[strings.TrimSpace(input.JobID)]
	if !ok || !dataAgentJobVisibleToOwner(job, input.OwnerUserID, input.OwnerOrgID) {
		return domain.DataAgentJobLeaseRecord{}, ErrNotFound
	}
	if dataAgentJobStatusIsTerminal(job.Status) {
		return domain.DataAgentJobLeaseRecord{}, ErrConflict
	}
	now := leaseNow(input.Now)
	lease, ok := s.dataAgentLeases[job.JobID]
	if !ok || lease.LeaseToken != strings.TrimSpace(input.LeaseToken) || !lease.LeaseExpiresAt.After(now) {
		return domain.DataAgentJobLeaseRecord{}, ErrConflict
	}
	lease.LeaseExpiresAt = now.Add(positiveLeaseTTL(input.TTL))
	lease.UpdatedAt = now
	s.dataAgentLeases[job.JobID] = lease
	job.UpdatedAt = now
	s.dataAgentJobs[job.JobID] = job
	return lease, nil
}

func (s *MemoryStore) ReleaseDataAgentJobLease(ctx context.Context, input domain.ReleaseDataAgentJobLeaseInput) error {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.dataAgentJobs[strings.TrimSpace(input.JobID)]
	if !ok || !dataAgentJobVisibleToOwner(job, input.OwnerUserID, input.OwnerOrgID) {
		return ErrNotFound
	}
	lease, ok := s.dataAgentLeases[job.JobID]
	if !ok {
		return nil
	}
	if lease.LeaseToken != strings.TrimSpace(input.LeaseToken) {
		return ErrConflict
	}
	delete(s.dataAgentLeases, job.JobID)
	return nil
}

func (s *MemoryStore) RecoverExpiredDataAgentJobLeases(ctx context.Context, input domain.RecoverExpiredDataAgentJobLeasesInput) (domain.RecoverExpiredDataAgentJobLeasesResult, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	now := input.Now
	if now.IsZero() {
		now = domain.Now()
	}
	reason := strings.TrimSpace(input.Reason)
	if reason == "" {
		reason = "automatic expired data-agent lease recovery"
	}
	limit := input.Limit
	if limit <= 0 {
		limit = 1000
	}
	candidates := make([]domain.DataAgentJobRecord, 0, len(s.dataAgentJobs))
	for _, job := range s.dataAgentJobs {
		if !dataAgentJobRecoverableStatus(job.Status) {
			continue
		}
		if _, ok := s.dataAgentLeases[job.JobID]; !ok {
			continue
		}
		candidates = append(candidates, job)
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].UpdatedAt.Equal(candidates[j].UpdatedAt) {
			return candidates[i].JobID < candidates[j].JobID
		}
		return candidates[i].UpdatedAt.Before(candidates[j].UpdatedAt)
	})
	if len(candidates) > limit {
		candidates = candidates[:limit]
	}
	result := domain.RecoverExpiredDataAgentJobLeasesResult{Checked: len(candidates)}
	for _, job := range candidates {
		lease, ok := s.dataAgentLeases[job.JobID]
		if !ok || lease.LeaseExpiresAt.After(now) {
			continue
		}
		delete(s.dataAgentLeases, job.JobID)
		job.Status = "queued"
		job.ProgressCompleted = 0
		if job.ProgressTotal <= 0 {
			job.ProgressTotal = job.ResourceCount
		}
		job.Error = ""
		job.StartedAt = time.Time{}
		job.CompletedAt = time.Time{}
		job.UpdatedAt = now.UTC()
		s.dataAgentJobs[job.JobID] = job
		if _, err := s.appendDataAgentJobEventLocked(domain.AppendDataAgentJobEventInput{
			JobID:       job.JobID,
			EventType:   "data_agent.job.requeued",
			ActorUserID: job.OwnerUserID,
			ActorOrgID:  job.OwnerOrgID,
			TS:          now,
			Message:     reason,
			Metadata: domain.JSONMap{
				"reason":           reason,
				"recovery":         "expired_data_agent_job_lease",
				"lease_worker_id":  lease.WorkerID,
				"lease_expires_at": lease.LeaseExpiresAt.UTC().Format(time.RFC3339Nano),
			},
		}); err != nil {
			return result, err
		}
		result.RequeuedJobs = append(result.RequeuedJobs, job)
	}
	return result, nil
}

func (s *MemoryStore) AppendDataAgentJobEvent(ctx context.Context, input domain.AppendDataAgentJobEventInput) (domain.DataAgentJobEventRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.appendDataAgentJobEventLocked(input)
}

func (s *MemoryStore) ListDataAgentJobEvents(ctx context.Context, jobID string, userID string, orgID string, limit int) ([]domain.DataAgentJobEventRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	job, ok := s.dataAgentJobs[strings.TrimSpace(jobID)]
	if !ok || !dataAgentJobVisibleToOwner(job, userID, orgID) {
		return nil, ErrNotFound
	}
	events := append([]domain.DataAgentJobEventRecord(nil), s.dataAgentEvents[job.JobID]...)
	sort.Slice(events, func(i, j int) bool {
		if events[i].Sequence == events[j].Sequence {
			return events[i].EventID < events[j].EventID
		}
		return events[i].Sequence < events[j].Sequence
	})
	return take(events, limit), nil
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
	sharing := strings.ToLower(strings.TrimSpace(input.Sharing))
	resources := make([]domain.ResourceRecord, 0, len(s.resources))
	for _, resource := range s.resources {
		if strings.TrimSpace(resource.Status) != status {
			continue
		}
		if status == "active" {
			if !s.resourceReadableByUserLocked(resource, input.UserID, input.OrgID) {
				continue
			}
		} else if !resourceVisibleToOwner(resource, input.UserID, input.OrgID) {
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
		if !resourceMatchesTags(resource, input.Tags) {
			continue
		}
		if !resourceMatchesDescriptors(resource, input.Descriptors) {
			continue
		}
		if !resourceMatchesMetadataFilters(resource, input.MetadataFilters) {
			continue
		}
		if !resourceMatchesCreatedRange(resource, input.CreatedAfter, input.CreatedBefore) {
			continue
		}
		if !resourceMatchesProcessingStatus(resource, input.ProcessingStatus) {
			continue
		}
		shareSummary := s.resourceShareSummaryLocked(resource, input.UserID, input.OrgID)
		if !resourceMatchesSharingFilter(shareSummary, sharing) {
			continue
		}
		resource = resourceWithNormalizedTags(resource)
		resource.ShareSummary = shareSummary
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

func (s *MemoryStore) BulkTagResourcesForUser(ctx context.Context, input domain.BulkTagResourcesInput) (domain.BulkTagResourcesResult, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	resourceIDs := uniqueTrimmedStrings(input.ResourceIDs)
	tags := normalizeResourceTags(input.Tags)
	if len(resourceIDs) == 0 || len(tags) == 0 {
		return domain.BulkTagResourcesResult{}, ErrNotFound
	}
	ownerUserID := strings.TrimSpace(input.OwnerUserID)
	ownerOrgID := strings.TrimSpace(input.OwnerOrgID)
	for _, resourceID := range resourceIDs {
		resource, ok := s.resources[resourceID]
		if !ok || !resourceVisibleToOwner(resource, ownerUserID, ownerOrgID) || strings.TrimSpace(resource.Status) != "active" {
			return domain.BulkTagResourcesResult{}, ErrNotFound
		}
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	actorUserID := strings.TrimSpace(input.ActorUserID)
	if actorUserID == "" {
		actorUserID = ownerUserID
	}
	actorOrgID := strings.TrimSpace(input.ActorOrgID)
	if actorOrgID == "" {
		actorOrgID = ownerOrgID
	}
	resources := make([]domain.ResourceRecord, 0, len(resourceIDs))
	events := make([]domain.ResourceEventRecord, 0, len(resourceIDs))
	for _, resourceID := range resourceIDs {
		resource := s.resources[resourceID]
		existingTags := tagsForResource(resource)
		resource.Tags = mergeResourceTags(existingTags, tags)
		resource.Metadata = resourceMetadataWithTags(resource.Metadata, resource.Tags)
		resource.UpdatedAt = ts.UTC()
		s.resources[resourceID] = resource
		event := domain.ResourceEventRecord{
			EventID:     domain.NewID("resource_event"),
			ResourceID:  resourceID,
			ActorUserID: actorUserID,
			ActorOrgID:  actorOrgID,
			EventType:   "resource.tagged",
			TS:          ts.UTC(),
			Metadata: domain.JSONMap{
				"tags_added": append([]string(nil), tags...),
				"tags":       append([]string(nil), resource.Tags...),
			},
		}
		if len(input.Metadata) > 0 {
			event.Metadata["request"] = cloneResourceMetadataValue(input.Metadata)
		}
		s.resourceEvents = append(s.resourceEvents, event)
		resources = append(resources, resourceWithNormalizedTags(resource))
		events = append(events, event)
	}
	return domain.BulkTagResourcesResult{
		UpdatedCount: len(resources),
		Resources:    resources,
		Events:       events,
	}, nil
}

func (s *MemoryStore) ListResources(ctx context.Context, limit int, offset int) ([]domain.ResourceRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	resources := make([]domain.ResourceRecord, 0, len(s.resources))
	for _, resource := range s.resources {
		resources = append(resources, resourceWithNormalizedTags(resource))
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

// resourcePastRetention reports whether a resource is soft-deleted AND its undelete
// window has elapsed — i.e. it can no longer be restored, so it is safe to reclaim.
func resourcePastRetention(resource domain.ResourceRecord, now time.Time) bool {
	return strings.TrimSpace(resource.Status) == "deleted" &&
		!resource.RetentionExpiresAt.IsZero() &&
		resource.RetentionExpiresAt.Before(now)
}

func (s *MemoryStore) RetentionBacklog(ctx context.Context, now time.Time) (domain.ResourceRetentionBacklog, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	var backlog domain.ResourceRetentionBacklog
	for _, resource := range s.resources {
		if resourcePastRetention(resource, now) {
			backlog.Count++
			backlog.Bytes += resource.SizeBytes
		}
	}
	return backlog, nil
}

func (s *MemoryStore) ListResourcesPastRetention(ctx context.Context, now time.Time, limit int) ([]domain.ResourceRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := []domain.ResourceRecord{}
	for _, resource := range s.resources {
		if !resourcePastRetention(resource, now) {
			continue
		}
		out = append(out, resource)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out, nil
}

func (s *MemoryStore) PurgeResource(ctx context.Context, resourceID string) error {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.resources, strings.TrimSpace(resourceID))
	return nil
}

// resourceUsageBy aggregates active-resource count + bytes for resources matching key(r).
// An aggregate so quota checks never load the whole catalog into memory.
func (s *MemoryStore) resourceUsageBy(want string, key func(domain.ResourceRecord) string) (int, int64) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if strings.TrimSpace(want) == "" {
		return 0, 0
	}
	var count int
	var bytes int64
	for _, resource := range s.resources {
		if strings.TrimSpace(resource.Status) == "active" && strings.TrimSpace(key(resource)) == want {
			count++
			bytes += resource.SizeBytes
		}
	}
	return count, bytes
}

func (s *MemoryStore) ResourceUsageForOwner(ctx context.Context, userID string) (int, int64, error) {
	_ = ctx
	count, bytes := s.resourceUsageBy(strings.TrimSpace(userID), func(r domain.ResourceRecord) string { return r.OwnerUserID })
	return count, bytes, nil
}

func (s *MemoryStore) ResourceUsageForOrg(ctx context.Context, orgID string) (int, int64, error) {
	_ = ctx
	count, bytes := s.resourceUsageBy(strings.TrimSpace(orgID), func(r domain.ResourceRecord) string { return r.OwnerOrgID })
	return count, bytes, nil
}

func (s *MemoryStore) ResourceUsageForProject(ctx context.Context, projectID string) (int, int64, error) {
	_ = ctx
	count, bytes := s.resourceUsageBy(strings.TrimSpace(projectID), func(r domain.ResourceRecord) string { return r.ProjectID })
	return count, bytes, nil
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

func (s *MemoryStore) CreateResourceShareGrant(ctx context.Context, input domain.CreateResourceShareGrantInput) (domain.ResourceShareGrantRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.createResourceShareGrantLocked(input)
}

func (s *MemoryStore) createResourceShareGrantLocked(input domain.CreateResourceShareGrantInput) (domain.ResourceShareGrantRecord, error) {
	resourceID := strings.TrimSpace(input.ResourceID)
	resource, ok := s.resources[resourceID]
	if !ok || !resourceVisibleToOwner(resource, input.OwnerUserID, input.OwnerOrgID) || strings.TrimSpace(resource.Status) != "active" {
		return domain.ResourceShareGrantRecord{}, ErrNotFound
	}
	granteeUserID := strings.TrimSpace(input.GranteeUserID)
	granteeOrgID := strings.TrimSpace(input.GranteeOrgID)
	if input.Public {
		granteeUserID = domain.PublicResourceGranteeUserID
		granteeOrgID = ""
	}
	if granteeUserID == "" && granteeOrgID == "" {
		return domain.ResourceShareGrantRecord{}, ErrNotFound
	}
	grantID := strings.TrimSpace(input.GrantID)
	if grantID == "" {
		grantID = domain.NewID("resource_grant")
	}
	if _, exists := s.resourceGrants[grantID]; exists {
		return domain.ResourceShareGrantRecord{}, ErrConflict
	}
	role := strings.TrimSpace(input.Role)
	if role == "" {
		role = "read"
	}
	status := strings.TrimSpace(input.Status)
	if status == "" {
		status = "active"
	}
	createdAt := input.CreatedAt
	if createdAt.IsZero() {
		createdAt = domain.Now()
	}
	updatedAt := input.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}
	grant := domain.ResourceShareGrantRecord{
		GrantID:         grantID,
		ResourceID:      resource.ResourceID,
		OwnerUserID:     resource.OwnerUserID,
		OwnerOrgID:      resource.OwnerOrgID,
		OwnerRole:       resource.OwnerRole,
		GranteeUserID:   granteeUserID,
		GranteeOrgID:    granteeOrgID,
		Role:            role,
		Status:          status,
		CreatedByUserID: strings.TrimSpace(input.CreatedByUserID),
		CreatedAt:       createdAt.UTC(),
		UpdatedAt:       updatedAt.UTC(),
		Metadata:        mapOrEmpty(input.Metadata),
	}
	s.resourceGrants[grant.GrantID] = grant
	return grant, nil
}

func (s *MemoryStore) createInheritedResourceShareGrantLocked(resourceID string, collectionGrant domain.ResourceCollectionShareGrantRecord, createdAt time.Time, source string) (domain.ResourceShareGrantRecord, error) {
	metadata := cloneJSONMap(collectionGrant.Metadata)
	metadata["collection_id"] = collectionGrant.CollectionID
	metadata["collection_share_grant_id"] = collectionGrant.GrantID
	metadata["source"] = source
	return s.createResourceShareGrantLocked(domain.CreateResourceShareGrantInput{
		ResourceID:      resourceID,
		OwnerUserID:     collectionGrant.OwnerUserID,
		OwnerOrgID:      collectionGrant.OwnerOrgID,
		OwnerRole:       collectionGrant.OwnerRole,
		GranteeUserID:   collectionGrant.GranteeUserID,
		GranteeOrgID:    collectionGrant.GranteeOrgID,
		Role:            collectionGrant.Role,
		Status:          "active",
		CreatedByUserID: collectionGrant.CreatedByUserID,
		CreatedAt:       createdAt,
		Metadata:        metadata,
	})
}

func (s *MemoryStore) activeCollectionShareGrantsLocked(collectionID string) []domain.ResourceCollectionShareGrantRecord {
	collectionID = strings.TrimSpace(collectionID)
	grants := make([]domain.ResourceCollectionShareGrantRecord, 0)
	for _, grant := range s.collectionGrants {
		if strings.TrimSpace(grant.CollectionID) != collectionID || strings.TrimSpace(grant.Status) != "active" {
			continue
		}
		grants = append(grants, grant)
	}
	sort.Slice(grants, func(i, j int) bool {
		if grants[i].CreatedAt.Equal(grants[j].CreatedAt) {
			return grants[i].GrantID < grants[j].GrantID
		}
		return grants[i].CreatedAt.Before(grants[j].CreatedAt)
	})
	return grants
}

func (s *MemoryStore) collectionShareGrantIDsLocked(collectionID string) map[string]struct{} {
	collectionID = strings.TrimSpace(collectionID)
	ids := map[string]struct{}{}
	for _, grant := range s.collectionGrants {
		if strings.TrimSpace(grant.CollectionID) != collectionID {
			continue
		}
		ids[strings.TrimSpace(grant.GrantID)] = struct{}{}
	}
	return ids
}

func (s *MemoryStore) collectionMembersForCollectionLocked(collectionID string) []domain.ResourceCollectionMembershipRecord {
	collectionID = strings.TrimSpace(collectionID)
	members := make([]domain.ResourceCollectionMembershipRecord, 0)
	for _, member := range s.collectionMembers {
		if member.CollectionID == collectionID {
			members = append(members, member)
		}
	}
	sort.Slice(members, func(i, j int) bool {
		if members[i].Position == members[j].Position {
			if members[i].AddedAt.Equal(members[j].AddedAt) {
				return members[i].ResourceID < members[j].ResourceID
			}
			return members[i].AddedAt.Before(members[j].AddedAt)
		}
		return members[i].Position < members[j].Position
	})
	return members
}

func (s *MemoryStore) ListResourceShareGrantsForResource(ctx context.Context, input domain.ListResourceShareGrantsInput) ([]domain.ResourceShareGrantRecord, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	resourceID := strings.TrimSpace(input.ResourceID)
	resource, ok := s.resources[resourceID]
	if !ok || !resourceVisibleToOwner(resource, input.OwnerUserID, input.OwnerOrgID) || strings.TrimSpace(resource.Status) != "active" {
		return nil, ErrNotFound
	}
	status := strings.TrimSpace(input.Status)
	grants := make([]domain.ResourceShareGrantRecord, 0)
	for _, grant := range s.resourceGrants {
		if strings.TrimSpace(grant.ResourceID) != resourceID {
			continue
		}
		if status != "" && strings.TrimSpace(grant.Status) != status {
			continue
		}
		grants = append(grants, grant)
	}
	sort.Slice(grants, func(i, j int) bool {
		if grants[i].CreatedAt.Equal(grants[j].CreatedAt) {
			return grants[i].GrantID < grants[j].GrantID
		}
		return grants[i].CreatedAt.After(grants[j].CreatedAt)
	})
	return take(grants, input.Limit), nil
}

func (s *MemoryStore) RevokeResourceShareGrant(ctx context.Context, input domain.RevokeResourceShareGrantInput) (domain.ResourceShareGrantRecord, error) {
	_ = ctx
	s.mu.Lock()
	defer s.mu.Unlock()
	resourceID := strings.TrimSpace(input.ResourceID)
	resource, ok := s.resources[resourceID]
	if !ok || !resourceVisibleToOwner(resource, input.OwnerUserID, input.OwnerOrgID) || strings.TrimSpace(resource.Status) != "active" {
		return domain.ResourceShareGrantRecord{}, ErrNotFound
	}
	grantID := strings.TrimSpace(input.GrantID)
	grant, ok := s.resourceGrants[grantID]
	if !ok || strings.TrimSpace(grant.ResourceID) != resourceID || strings.TrimSpace(grant.Status) != "active" {
		return domain.ResourceShareGrantRecord{}, ErrNotFound
	}
	revokedAt := input.RevokedAt
	if revokedAt.IsZero() {
		revokedAt = domain.Now()
	}
	grant.Status = "revoked"
	grant.UpdatedAt = revokedAt.UTC()
	grant.RevokedAt = revokedAt.UTC()
	s.resourceGrants[grant.GrantID] = grant
	return grant, nil
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

func (s *MemoryStore) ListResourceEventsForUser(ctx context.Context, input domain.ResourceEventListInput) (domain.ResourceEventListPage, error) {
	_ = ctx
	s.mu.RLock()
	defer s.mu.RUnlock()
	resourceID := strings.TrimSpace(input.ResourceID)
	eventType := strings.TrimSpace(input.EventType)
	actorUserID := strings.TrimSpace(input.ActorUserID)
	events := make([]domain.ResourceEventRecord, 0, len(s.resourceEvents))
	for _, event := range s.resourceEvents {
		if resourceID != "" && strings.TrimSpace(event.ResourceID) != resourceID {
			continue
		}
		if eventType != "" && strings.TrimSpace(event.EventType) != eventType {
			continue
		}
		if actorUserID != "" && strings.TrimSpace(event.ActorUserID) != actorUserID {
			continue
		}
		resource, ok := s.resources[strings.TrimSpace(event.ResourceID)]
		if !ok || !s.resourceAuditVisibleToUserLocked(resource, input.UserID, input.OrgID) {
			continue
		}
		events = append(events, event)
	}
	sort.Slice(events, func(i, j int) bool {
		if events[i].TS.Equal(events[j].TS) {
			return events[i].EventID < events[j].EventID
		}
		return events[i].TS.After(events[j].TS)
	})
	total := len(events)
	offset := input.Offset
	if offset < 0 {
		offset = 0
	}
	if offset >= len(events) {
		events = []domain.ResourceEventRecord{}
	} else {
		events = events[offset:]
	}
	limit := input.Limit
	if limit <= 0 {
		limit = 200
	}
	events = take(events, limit)
	return domain.ResourceEventListPage{
		Events:     events,
		TotalCount: total,
		Limit:      limit,
		Offset:     offset,
	}, nil
}

func resourceVisibleToOwner(resource domain.ResourceRecord, userID string, orgID string) bool {
	if strings.TrimSpace(resource.OwnerUserID) != strings.TrimSpace(userID) {
		return false
	}
	ownerOrgID := strings.TrimSpace(resource.OwnerOrgID)
	return ownerOrgID == "" || ownerOrgID == strings.TrimSpace(orgID)
}

func (s *MemoryStore) resourceAuditVisibleToUserLocked(resource domain.ResourceRecord, userID string, orgID string) bool {
	if resourceVisibleToOwner(resource, userID, orgID) {
		return true
	}
	if strings.TrimSpace(resource.Status) != "active" {
		return false
	}
	resourceID := strings.TrimSpace(resource.ResourceID)
	for _, grant := range s.resourceGrants {
		if strings.TrimSpace(grant.ResourceID) != resourceID || strings.TrimSpace(grant.Status) != "active" {
			continue
		}
		if resourceGrantMatchesPrincipal(grant, userID, orgID) {
			return true
		}
	}
	return false
}

func (s *MemoryStore) resourceReadableByUserLocked(resource domain.ResourceRecord, userID string, orgID string) bool {
	if resourceVisibleToOwner(resource, userID, orgID) {
		return true
	}
	resourceID := strings.TrimSpace(resource.ResourceID)
	for _, grant := range s.resourceGrants {
		if strings.TrimSpace(grant.ResourceID) != resourceID || strings.TrimSpace(grant.Status) != "active" {
			continue
		}
		if resourceGrantMatchesPrincipal(grant, userID, orgID) {
			return true
		}
	}
	return false
}

func (s *MemoryStore) resourceShareSummaryLocked(resource domain.ResourceRecord, userID string, orgID string) domain.ResourceShareSummary {
	resourceID := strings.TrimSpace(resource.ResourceID)
	isOwner := resourceVisibleToOwner(resource, userID, orgID)
	activeGrantCount := 0
	matchingGrantCount := 0
	publicGrantCount := 0
	for _, grant := range s.resourceGrants {
		if strings.TrimSpace(grant.ResourceID) != resourceID || strings.TrimSpace(grant.Status) != "active" {
			continue
		}
		if strings.TrimSpace(grant.GranteeUserID) == domain.PublicResourceGranteeUserID {
			publicGrantCount++
		}
		if isOwner {
			activeGrantCount++
		}
		if resourceGrantMatchesPrincipal(grant, userID, orgID) {
			matchingGrantCount++
		}
	}
	if isOwner && activeGrantCount > 0 {
		if publicGrantCount > 0 {
			return domain.ResourceShareSummary{
				ShareStatus:      "public",
				ActiveGrantCount: publicGrantCount,
				Public:           true,
			}
		}
		return domain.ResourceShareSummary{
			ShareStatus:      "shared_by_me",
			ActiveGrantCount: activeGrantCount,
			SharedByMe:       true,
		}
	}
	if publicGrantCount > 0 {
		return domain.ResourceShareSummary{
			ShareStatus:      "public",
			ActiveGrantCount: publicGrantCount,
			Public:           true,
		}
	}
	if !isOwner && matchingGrantCount > 0 {
		return domain.ResourceShareSummary{
			ShareStatus:      "shared_with_me",
			ActiveGrantCount: matchingGrantCount,
			SharedWithMe:     true,
		}
	}
	return domain.ResourceShareSummary{ShareStatus: "private"}
}

func resourceGrantMatchesPrincipal(grant domain.ResourceShareGrantRecord, userID string, orgID string) bool {
	userID = strings.TrimSpace(userID)
	orgID = strings.TrimSpace(orgID)
	granteeUserID := strings.TrimSpace(grant.GranteeUserID)
	granteeOrgID := strings.TrimSpace(grant.GranteeOrgID)
	if granteeUserID == domain.PublicResourceGranteeUserID {
		return true
	}
	if granteeUserID != "" && granteeUserID == userID {
		return granteeOrgID == "" || granteeOrgID == orgID
	}
	return granteeUserID == "" && granteeOrgID != "" && granteeOrgID == orgID
}

func resourceMatchesSharingFilter(summary domain.ResourceShareSummary, sharing string) bool {
	switch strings.ToLower(strings.TrimSpace(sharing)) {
	case "", "all":
		return true
	case "shared":
		return summary.ShareStatus == "shared_by_me" || summary.ShareStatus == "shared_with_me" || summary.ShareStatus == "public"
	case "private", "shared_by_me", "shared_with_me", "public":
		return summary.ShareStatus == sharing
	default:
		return false
	}
}

func collectionVisibleToOwner(collection domain.ResourceCollectionRecord, userID string, orgID string) bool {
	if strings.TrimSpace(collection.OwnerUserID) != strings.TrimSpace(userID) {
		return false
	}
	ownerOrgID := strings.TrimSpace(collection.OwnerOrgID)
	return ownerOrgID == "" || ownerOrgID == strings.TrimSpace(orgID)
}

func (s *MemoryStore) collectionReadableByUserLocked(collection domain.ResourceCollectionRecord, userID string, orgID string) bool {
	if collectionVisibleToOwner(collection, userID, orgID) {
		return true
	}
	collectionID := strings.TrimSpace(collection.CollectionID)
	for _, grant := range s.collectionGrants {
		if strings.TrimSpace(grant.CollectionID) != collectionID || strings.TrimSpace(grant.Status) != "active" || strings.TrimSpace(grant.Role) != "read" {
			continue
		}
		if collectionGrantMatchesPrincipal(grant, userID, orgID) {
			return true
		}
	}
	return false
}

func collectionGrantMatchesPrincipal(grant domain.ResourceCollectionShareGrantRecord, userID string, orgID string) bool {
	userID = strings.TrimSpace(userID)
	orgID = strings.TrimSpace(orgID)
	granteeUserID := strings.TrimSpace(grant.GranteeUserID)
	granteeOrgID := strings.TrimSpace(grant.GranteeOrgID)
	if granteeUserID == domain.PublicResourceGranteeUserID {
		return true
	}
	if granteeUserID != "" && granteeUserID == userID {
		return granteeOrgID == "" || granteeOrgID == orgID
	}
	return granteeUserID == "" && granteeOrgID != "" && granteeOrgID == orgID
}

func datasetSnapshotVisibleToOwner(snapshot domain.DatasetSnapshotRecord, userID string, orgID string) bool {
	if strings.TrimSpace(snapshot.OwnerUserID) != strings.TrimSpace(userID) {
		return false
	}
	ownerOrgID := strings.TrimSpace(snapshot.OwnerOrgID)
	return ownerOrgID == "" || ownerOrgID == strings.TrimSpace(orgID)
}

func (s *MemoryStore) datasetSnapshotVisibleToUserLocked(snapshot domain.DatasetSnapshotRecord, userID string, orgID string) bool {
	if datasetSnapshotVisibleToOwner(snapshot, userID, orgID) {
		return true
	}
	for _, grant := range s.datasetGrants {
		if strings.TrimSpace(grant.SnapshotID) != strings.TrimSpace(snapshot.SnapshotID) {
			continue
		}
		if strings.TrimSpace(grant.Status) != "active" || strings.TrimSpace(grant.Role) == "" {
			continue
		}
		if datasetSnapshotGrantMatchesPrincipal(grant, userID, orgID) {
			return true
		}
	}
	return false
}

func datasetSnapshotGrantMatchesPrincipal(grant domain.DatasetSnapshotShareGrantRecord, userID string, orgID string) bool {
	userID = strings.TrimSpace(userID)
	orgID = strings.TrimSpace(orgID)
	granteeUserID := strings.TrimSpace(grant.GranteeUserID)
	granteeOrgID := strings.TrimSpace(grant.GranteeOrgID)
	if granteeUserID != "" && granteeUserID == userID {
		return granteeOrgID == "" || granteeOrgID == orgID
	}
	return granteeUserID == "" && granteeOrgID != "" && granteeOrgID == orgID
}

func (s *MemoryStore) appendDatasetSnapshotEventLocked(event domain.DatasetSnapshotEventRecord) domain.DatasetSnapshotEventRecord {
	event.SnapshotID = strings.TrimSpace(event.SnapshotID)
	event.ActorUserID = strings.TrimSpace(event.ActorUserID)
	event.ActorOrgID = strings.TrimSpace(event.ActorOrgID)
	event.EventType = strings.TrimSpace(event.EventType)
	if event.EventID == "" {
		event.EventID = domain.NewID("dataset_snapshot_event")
	}
	if event.TS.IsZero() {
		event.TS = domain.Now()
	}
	event.TS = event.TS.UTC()
	event.Metadata = mapOrEmpty(event.Metadata)
	s.datasetEvents = append(s.datasetEvents, event)
	return event
}

func (s *MemoryStore) latestDatasetSnapshotEventTimeLocked(snapshotID string, fallback time.Time) time.Time {
	latest := fallback
	for _, event := range s.datasetEvents {
		if strings.TrimSpace(event.SnapshotID) != strings.TrimSpace(snapshotID) {
			continue
		}
		if latest.IsZero() || event.TS.After(latest) {
			latest = event.TS
		}
	}
	return latest
}

func afterEventTime(previous time.Time, candidate time.Time) time.Time {
	if previous.IsZero() {
		if candidate.IsZero() {
			return domain.Now()
		}
		return candidate
	}
	if candidate.After(previous) {
		return candidate
	}
	return previous.Add(time.Nanosecond)
}

func dataAgentJobVisibleToOwner(job domain.DataAgentJobRecord, userID string, orgID string) bool {
	if strings.TrimSpace(job.OwnerUserID) != strings.TrimSpace(userID) {
		return false
	}
	ownerOrgID := strings.TrimSpace(job.OwnerOrgID)
	return ownerOrgID == "" || ownerOrgID == strings.TrimSpace(orgID)
}

func validDataAgentJobStatus(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "queued", "running", "succeeded", "failed", "canceled":
		return true
	default:
		return false
	}
}

func dataAgentJobStatusIsTerminal(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "succeeded", "failed", "canceled":
		return true
	default:
		return false
	}
}

func dataAgentJobRecoverableStatus(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "queued", "running":
		return true
	default:
		return false
	}
}

func dataAgentJobStatusEventType(status string) string {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "running":
		return "data_agent.job.progressed"
	case "succeeded":
		return "data_agent.job.completed"
	case "failed":
		return "data_agent.job.failed"
	case "canceled":
		return "data_agent.job.canceled"
	default:
		return "data_agent.job.updated"
	}
}

func clampDataAgentProgress(completed int, total int) int {
	if completed < 0 {
		return 0
	}
	if total > 0 && completed > total {
		return total
	}
	return completed
}

func (s *MemoryStore) appendDataAgentJobEventLocked(input domain.AppendDataAgentJobEventInput) (domain.DataAgentJobEventRecord, error) {
	jobID := strings.TrimSpace(input.JobID)
	if _, ok := s.dataAgentJobs[jobID]; !ok {
		return domain.DataAgentJobEventRecord{}, ErrNotFound
	}
	eventID := strings.TrimSpace(input.EventID)
	if eventID == "" {
		eventID = domain.NewID("data_agent_job_event")
	}
	for _, existing := range s.dataAgentEvents[jobID] {
		if strings.TrimSpace(existing.EventID) == eventID {
			return domain.DataAgentJobEventRecord{}, ErrConflict
		}
	}
	sequence := input.Sequence
	if sequence <= 0 {
		sequence = int64(len(s.dataAgentEvents[jobID]) + 1)
	}
	ts := input.TS
	if ts.IsZero() {
		ts = domain.Now()
	}
	record := domain.DataAgentJobEventRecord{
		EventID:     eventID,
		JobID:       jobID,
		Sequence:    sequence,
		EventType:   strings.TrimSpace(input.EventType),
		ActorUserID: strings.TrimSpace(input.ActorUserID),
		ActorOrgID:  strings.TrimSpace(input.ActorOrgID),
		TS:          ts.UTC(),
		Message:     strings.TrimSpace(input.Message),
		Metadata:    cloneJSONMap(input.Metadata),
	}
	s.dataAgentEvents[jobID] = append(s.dataAgentEvents[jobID], record)
	return record, nil
}

func uploadFileKey(sessionID string, fileToken string) string {
	return strings.TrimSpace(sessionID) + "\x00" + strings.TrimSpace(fileToken)
}

func uploadChunkKey(sessionID string, fileToken string, chunkIndex int) string {
	return uploadFileKey(sessionID, fileToken) + "\x00" + strconv.Itoa(chunkIndex)
}

func collectionMemberKey(collectionID string, resourceID string) string {
	return strings.TrimSpace(collectionID) + "\x00" + strings.TrimSpace(resourceID)
}

func resourceRecordMatchesQuery(resource domain.ResourceRecord, query string) bool {
	return strings.Contains(resourceSearchDocument(resource), query)
}

func resourceCollectionMatchesQuery(collection domain.ResourceCollectionRecord, query string) bool {
	candidates := []string{
		collection.CollectionID,
		collection.Name,
		collection.Description,
		collection.CollectionType,
		collection.ProjectID,
	}
	for _, candidate := range candidates {
		if strings.Contains(strings.ToLower(strings.TrimSpace(candidate)), query) {
			return true
		}
	}
	return false
}

func datasetSnapshotMatchesQuery(snapshot domain.DatasetSnapshotRecord, query string) bool {
	candidates := []string{
		snapshot.SnapshotID,
		snapshot.Name,
		snapshot.Description,
		snapshot.ProjectID,
		snapshot.SourceCollectionID,
		snapshot.CreatedByUserID,
	}
	for _, candidate := range candidates {
		if strings.Contains(strings.ToLower(strings.TrimSpace(candidate)), query) {
			return true
		}
	}
	if len(snapshot.Metadata) > 0 {
		metadataBytes, err := json.Marshal(snapshot.Metadata)
		if err == nil && strings.Contains(strings.ToLower(string(metadataBytes)), query) {
			return true
		}
	}
	return false
}

func (s *MemoryStore) collectionResourceCountLocked(collectionID string) int {
	count := 0
	for _, member := range s.collectionMembers {
		if member.CollectionID != collectionID {
			continue
		}
		if resource, ok := s.resources[member.ResourceID]; ok && strings.TrimSpace(resource.Status) == "active" {
			count++
		}
	}
	return count
}

func (s *MemoryStore) nextCollectionPositionLocked(collectionID string) int64 {
	var next int64
	for _, member := range s.collectionMembers {
		if member.CollectionID == collectionID && member.Position >= next {
			next = member.Position + 1
		}
	}
	return next
}

func uniqueTrimmedStrings(values []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func mapOrEmpty(value domain.JSONMap) domain.JSONMap {
	if value == nil {
		return domain.JSONMap{}
	}
	return value
}

func cloneJSONMap(value domain.JSONMap) domain.JSONMap {
	if value == nil {
		return domain.JSONMap{}
	}
	out := make(domain.JSONMap, len(value))
	for key, item := range value {
		out[key] = item
	}
	return out
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
