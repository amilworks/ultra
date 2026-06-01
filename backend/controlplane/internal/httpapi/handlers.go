package httpapi

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/go-chi/chi/v5"
)

type ServerDeps struct {
	Version          string
	Runs             *runcontrol.Service
	Store            runcontrol.Store
	Bus              runEventSource
	ArtifactRoot     string
	UploadRoot       string
	DevAdminEnabled  bool
	Runtime          RuntimeSummary
	QueueDiagnostics eventbus.QueueDiagnosticsProvider
}

type runEventSource interface {
	SubscribeRunEvents(ctx context.Context, runID string) (<-chan domain.RunEventRecord, func())
}

type accountStore interface {
	CreateUser(context.Context, domain.CreateUserInput) (domain.UserAccount, error)
	ListUsers(context.Context, int, string) ([]domain.UserAccount, error)
	UpdateUserStatus(context.Context, string, string) (domain.UserAccount, error)
}

type organizationStore interface {
	CreateOrganization(context.Context, domain.CreateOrganizationInput) (domain.Organization, error)
	ListOrganizations(context.Context, int, string) ([]domain.Organization, error)
}

type runLeaseReader interface {
	GetRunLease(context.Context, string) (domain.RunLeaseRecord, bool, error)
}

type workerHeartbeatStore interface {
	UpsertWorkerHeartbeat(context.Context, domain.UpsertWorkerHeartbeatInput) (domain.WorkerHeartbeatRecord, error)
	ListWorkerHeartbeats(context.Context, int) ([]domain.WorkerHeartbeatRecord, error)
}

const (
	adminStaleRunThreshold       = 5 * time.Minute
	adminWorkerStaleThreshold    = 3 * time.Minute
	runEventMaxPageLimit         = 1000
	runEventStreamHeartbeatEvery = 15 * time.Second
	runEventStreamCatchupEvery   = time.Second
)

func NewRouter(deps ServerDeps) http.Handler {
	r := chi.NewRouter()
	r.Get("/v1/health", handleHealth)
	r.Get("/v1/config/public", handlePublicConfig(deps))
	r.Get("/v1/auth/session", handleAuthSession(deps))
	r.Post("/v1/auth/guest", handleAuthGuest(deps))
	r.Post("/v1/auth/login", handleAuthLogin(deps))
	r.Post("/v1/auth/logout", handleAuthLogout)
	r.Route("/v2", func(r chi.Router) {
		r.Get("/health", handleHealth)
		r.Get("/config/public", handlePublicConfig(deps))
		r.Get("/auth/session", handleAuthSession(deps))
		r.Post("/auth/guest", handleAuthGuest(deps))
		r.Post("/auth/login", handleAuthLogin(deps))
		r.Post("/auth/logout", handleAuthLogout)
		r.Get("/threads", deps.handleListThreads)
		r.Post("/threads", deps.handleCreateThread)
		r.Get("/threads/{thread_id}", deps.handleGetThread)
		r.Get("/threads/{thread_id}/messages", deps.handleListThreadMessages)
		r.Post("/threads/{thread_id}/runs", deps.handleCreateRun)
		r.Post("/uploads", deps.handleUploadFiles)
		r.Get("/uploads/{file_id}/viewer", deps.handleGetUploadViewer)
		r.Get("/uploads/{file_id}/preview", deps.handleServeUpload)
		r.Get("/uploads/{file_id}/display", deps.handleServeUpload)
		r.Get("/uploads/{file_id}/slice", deps.handleServeUploadSlice)
		r.Get("/uploads/{file_id}/caption", deps.handleGetUploadCaption)
		r.Post("/uploads/from-bisque", deps.handleNotConfigured("BisQue resource import is not configured in the Go control plane yet; use selected V2 resources or upload files directly"))
		r.Get("/resources", deps.handleListResources)
		r.Get("/resources/{file_id}", deps.handleGetResource)
		r.Delete("/resources/{file_id}", deps.handleDeleteResource)
		r.Get("/resources/{file_id}/thumbnail", deps.handleServeUpload)
		r.Get("/runs", deps.handleListRuns)
		r.Get("/runs/{run_id}", deps.handleGetRun)
		r.Post("/runs/{run_id}/lease", deps.handleAcquireRunLease)
		r.Patch("/runs/{run_id}/lease", deps.handleRenewRunLease)
		r.Delete("/runs/{run_id}/lease", deps.handleReleaseRunLease)
		r.Post("/runs/{run_id}/cancel", deps.handleCancelRun)
		r.Get("/runs/{run_id}/events", deps.handleListRunEvents)
		r.Get("/runs/{run_id}/artifacts", deps.handleListRunArtifacts)
		r.Get("/runs/{run_id}/artifacts/download", deps.handleDownloadRunArtifactByPath)
		r.Get("/artifacts/{artifact_id}", deps.handleGetArtifact)
		r.Get("/artifacts/{artifact_id}/download", deps.handleDownloadArtifact)
		r.Post("/workers/heartbeat", deps.handleWorkerHeartbeat)
		r.Get("/admin/overview", deps.handleAdminOverview)
		r.Get("/admin/orgs", deps.handleAdminOrganizations)
		r.Post("/admin/orgs", deps.handleAdminCreateOrganization)
		r.Get("/admin/users", deps.handleAdminUsers)
		r.Post("/admin/users", deps.handleAdminCreateUser)
		r.Delete("/admin/users/{user_id}", deps.handleAdminDeleteUser)
		r.Get("/admin/runs", deps.handleAdminRuns)
		r.Get("/admin/issues", deps.handleAdminIssues)
		r.Post("/admin/runs/{run_id}/cancel", deps.handleAdminCancelRun)
		r.Post("/admin/runs/{run_id}/requeue", deps.handleAdminRequeueRun)
		r.Delete("/admin/conversations/{conversation_id}", deps.handleNotConfigured("admin conversation deletion is not configured in the Go control plane yet"))
		r.Get("/training/models", deps.handleTrainingModels)
		r.Get("/training/prairie/status", deps.handlePrairieStatus)
		r.Get("/training/prairie/retrain-requests", deps.handleEmptyPrairieRetrainRequests)
		r.Post("/training/prairie/sync", deps.handleNotConfigured("prairie active-learning sync is not configured in the Go control plane yet"))
		r.Post("/training/prairie/benchmark/run", deps.handleNotConfigured("prairie benchmark jobs are not configured in the Go control plane yet"))
		r.Post("/training/prairie/retrain-request", deps.handleNotConfigured("prairie retraining jobs are not configured in the Go control plane yet"))
		r.Get("/training/datasets", deps.handleEmptyTrainingDatasets)
		r.Post("/training/datasets", deps.handleNotConfigured("training dataset creation is not configured in the Go control plane yet"))
		r.Get("/training/datasets/{dataset_id}", deps.handleNotConfigured("training datasets are not configured in the Go control plane yet"))
		r.Post("/training/datasets/{dataset_id}/items", deps.handleNotConfigured("training dataset item assignment is not configured in the Go control plane yet"))
		r.Post("/training/jobs", deps.handleNotConfigured("training jobs are not configured in the Go control plane yet"))
		r.Get("/training/jobs/{job_id}", deps.handleNotConfigured("training jobs are not configured in the Go control plane yet"))
		r.Post("/training/jobs/{job_id}/control", deps.handleNotConfigured("training job control is not configured in the Go control plane yet"))
		r.Post("/training/preflight", deps.handleNotConfigured("training preflight is not configured in the Go control plane yet"))
		r.Post("/inference/jobs", deps.handleNotConfigured("standalone inference jobs are not configured in the Go control plane yet; use V2 chat tools for RareSpot inference"))
		r.Get("/inference/jobs/{job_id}/result", deps.handleNotConfigured("standalone inference jobs are not configured in the Go control plane yet; use V2 chat tools for RareSpot inference"))
		r.Post("/segment/sam3/interactive", deps.handleNotConfigured("SAM3 interactive segmentation is not configured in the Go control plane yet; use V2 chat tools for segmentation workflows"))
		r.Get("/model-health", deps.handleModelHealth)
		r.Get("/admin/model-health", deps.handleModelHealth)
		r.Get("/training/domains", deps.handleEmptyTrainingDomains)
		r.Post("/training/domains", deps.handleNotConfigured("training domain creation is not configured in the Go control plane yet"))
		r.Get("/training/domains/{domain_id}/lineages", deps.handleEmptyTrainingLineages)
		r.Post("/training/lineages/{lineage_id}/fork", deps.handleNotConfigured("training lineage forks are not configured in the Go control plane yet"))
		r.Get("/training/lineages/{lineage_id}/versions", deps.handleEmptyTrainingVersions)
		r.Post("/training/update-proposals/preview", deps.handleNotConfigured("training update proposals are not configured in the Go control plane yet"))
		r.Get("/training/update-proposals", deps.handleEmptyTrainingUpdateProposals)
		r.Post("/training/update-proposals/{proposal_id}/approve", deps.handleNotConfigured("training update proposal decisions are not configured in the Go control plane yet"))
		r.Post("/training/update-proposals/{proposal_id}/reject", deps.handleNotConfigured("training update proposal decisions are not configured in the Go control plane yet"))
		r.Post("/training/model-versions/{version_id}/promote", deps.handleNotConfigured("training model promotion is not configured in the Go control plane yet"))
		r.Post("/training/model-versions/{version_id}/rollback", deps.handleNotConfigured("training model rollback is not configured in the Go control plane yet"))
		r.Get("/training/merge-requests", deps.handleEmptyTrainingMergeRequests)
		r.Post("/training/merge-requests", deps.handleNotConfigured("training merge requests are not configured in the Go control plane yet"))
		r.Post("/training/merge-requests/{merge_id}/approve", deps.handleNotConfigured("training merge request decisions are not configured in the Go control plane yet"))
		r.Post("/training/merge-requests/{merge_id}/reject", deps.handleNotConfigured("training merge request decisions are not configured in the Go control plane yet"))
	})
	return r
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status": "ok",
		"ts":     time.Now().UTC().Format(time.RFC3339Nano),
	})
}

func handlePublicConfig(deps ServerDeps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"app_name":      "BisQue Ultra",
			"app_version":   deps.Version,
			"admin_enabled": deps.DevAdminEnabled,
			"features": map[string]bool{
				"v2_runs": true,
			},
		})
	}
}

func handleAuthSession(deps ServerDeps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		session := devAuthSessionFromRequest(r, deps.DevAdminEnabled)
		writeJSON(w, http.StatusOK, session)
	}
}

func handleAuthGuest(deps ServerDeps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Name        string `json:"name"`
			Email       string `json:"email"`
			Affiliation string `json:"affiliation"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil && !errors.Is(err, io.EOF) {
			writeError(w, http.StatusBadRequest, errors.New("invalid guest auth payload"))
			return
		}
		name := strings.TrimSpace(payload.Name)
		if name == "" {
			name = "Guest"
		}
		email := strings.TrimSpace(payload.Email)
		affiliation := strings.TrimSpace(payload.Affiliation)
		setDevAuthCookie(w, "guest:"+name)
		writeJSON(w, http.StatusOK, devAuthSession(name, "guest", map[string]any{
			"name":        name,
			"email":       email,
			"affiliation": affiliation,
		}, deps.DevAdminEnabled))
	}
}

func handleAuthLogin(deps ServerDeps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Username string `json:"username"`
			Password string `json:"password"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			writeError(w, http.StatusBadRequest, errors.New("invalid login payload"))
			return
		}
		username := strings.TrimSpace(payload.Username)
		if username == "" {
			username = "local-user"
		}
		setDevAuthCookie(w, "bisque:"+username)
		writeJSON(w, http.StatusOK, devAuthSession(username, "bisque", nil, deps.DevAdminEnabled))
	}
}

func handleAuthLogout(w http.ResponseWriter, r *http.Request) {
	setDevAuthCookie(w, "signed_out")
	writeJSON(w, http.StatusOK, map[string]any{
		"authenticated": false,
		"user":          nil,
	})
}

func devAuthSessionFromRequest(r *http.Request, adminEnabled bool) map[string]any {
	cookie, err := r.Cookie("ultra_dev_auth")
	if err == nil {
		value := strings.TrimSpace(cookie.Value)
		if decoded, decodeErr := url.QueryUnescape(value); decodeErr == nil {
			value = decoded
		}
		if value == "signed_out" {
			return map[string]any{
				"authenticated": false,
				"user":          nil,
			}
		}
		if username, ok := strings.CutPrefix(value, "guest:"); ok {
			username = strings.TrimSpace(username)
			if username == "" {
				username = "Guest"
			}
			return devAuthSession(username, "guest", map[string]any{
				"name":        username,
				"email":       "",
				"affiliation": "",
			}, adminEnabled)
		}
		if username, ok := strings.CutPrefix(value, "bisque:"); ok {
			username = strings.TrimSpace(username)
			if username == "" {
				username = "local-user"
			}
			return devAuthSession(username, "bisque", nil, adminEnabled)
		}
	}
	return devAuthSession("Guest", "guest", map[string]any{
		"name":        "Guest",
		"email":       "",
		"affiliation": "",
	}, adminEnabled)
}

func devAuthSession(username string, mode string, guestProfile map[string]any, adminEnabled bool) map[string]any {
	role := "researcher"
	if adminEnabled {
		role = "admin"
	}
	session := map[string]any{
		"authenticated": true,
		"username":      username,
		"user": map[string]any{
			"id":       "local-user",
			"username": username,
			"org_id":   "local-org",
			"role":     role,
		},
		"mode":     mode,
		"is_admin": adminEnabled,
	}
	if guestProfile != nil {
		session["guest_profile"] = guestProfile
	}
	return session
}

type requestPrincipal struct {
	UserID string
	OrgID  string
	Role   string
}

func (p requestPrincipal) record() principalRecord {
	return principalRecord{
		UserID: p.UserID,
		OrgID:  p.OrgID,
		Role:   p.Role,
	}
}

func principalFromRequest(r *http.Request, fallbackUserID string) requestPrincipal {
	userID := firstNonEmpty(
		strings.TrimSpace(r.Header.Get("X-Ultra-User-Id")),
		strings.TrimSpace(fallbackUserID),
		"local-user",
	)
	orgID := firstNonEmpty(
		strings.TrimSpace(r.Header.Get("X-Ultra-Org-Id")),
		"local-org",
	)
	role := firstNonEmpty(
		strings.TrimSpace(r.Header.Get("X-Ultra-Role")),
		"researcher",
	)
	return requestPrincipal{UserID: userID, OrgID: orgID, Role: role}
}

func metadataWithPrincipal(metadata domain.JSONMap, principal requestPrincipal) domain.JSONMap {
	merged := domain.JSONMap{}
	for key, value := range metadata {
		merged[key] = value
	}
	merged["principal"] = domain.JSONMap{
		"user_id": principal.UserID,
		"org_id":  principal.OrgID,
		"role":    principal.Role,
	}
	merged["principal_user_id"] = principal.UserID
	merged["org_id"] = principal.OrgID
	merged["principal_role"] = principal.Role
	return merged
}

func setDevAuthCookie(w http.ResponseWriter, value string) {
	http.SetCookie(w, &http.Cookie{
		Name:     "ultra_dev_auth",
		Value:    url.QueryEscape(value),
		Path:     "/",
		SameSite: http.SameSiteLaxMode,
		MaxAge:   int((24 * time.Hour).Seconds()),
	})
}

type createThreadRequest struct {
	UserID          string                 `json:"user_id"`
	Title           string                 `json:"title"`
	Metadata        map[string]any         `json:"metadata"`
	InitialMessages []domain.ThreadMessage `json:"initial_messages"`
}

type createRunRequest struct {
	UserID              string                 `json:"user_id"`
	Goal                string                 `json:"goal"`
	Messages            []domain.ThreadMessage `json:"messages"`
	FileIDs             []string               `json:"file_ids"`
	ResourceURIs        []string               `json:"resource_uris"`
	DatasetURIs         []string               `json:"dataset_uris"`
	SelectedToolNames   []string               `json:"selected_tool_names"`
	KnowledgeContext    map[string]any         `json:"knowledge_context"`
	WorkflowHint        map[string]any         `json:"workflow_hint"`
	SelectionContext    map[string]any         `json:"selection_context"`
	ReasoningMode       string                 `json:"reasoning_mode"`
	Budgets             map[string]any         `json:"budgets"`
	Benchmark           map[string]any         `json:"benchmark"`
	ResourceDescriptors []domain.JSONMap       `json:"resource_descriptors"`
	IdempotencyKey      string                 `json:"idempotency_key"`
	Metadata            map[string]any         `json:"metadata"`
}

type cancelRunRequest struct {
	Reason   string         `json:"reason"`
	Metadata map[string]any `json:"metadata"`
}

type runLeaseRequest struct {
	WorkerID        string  `json:"worker_id"`
	LeaseToken      string  `json:"lease_token"`
	TTLSeconds      float64 `json:"ttl_seconds"`
	LeaseTTLSeconds float64 `json:"lease_ttl_seconds"`
}

type listThreadsResponse struct {
	Count   int                   `json:"count"`
	Threads []domain.ThreadRecord `json:"threads"`
}

type threadMessagesResponse struct {
	ThreadID string                 `json:"thread_id"`
	Count    int                    `json:"count"`
	Messages []domain.ThreadMessage `json:"messages"`
}

type runEventsResponse struct {
	RunID  string                  `json:"run_id"`
	Count  int                     `json:"count"`
	Events []domain.RunEventRecord `json:"events"`
}

type runArtifactsResponse struct {
	RunID     string                  `json:"run_id"`
	Count     int                     `json:"count"`
	Artifacts []domain.ArtifactRecord `json:"artifacts"`
}

type runsResponse struct {
	Count int                `json:"count"`
	Runs  []domain.RunRecord `json:"runs"`
}

type artifactResponse struct {
	Artifact domain.ArtifactRecord `json:"artifact"`
}

type uploadedFileRecord struct {
	FileID       string          `json:"file_id"`
	OriginalName string          `json:"original_name"`
	ContentType  string          `json:"content_type,omitempty"`
	SizeBytes    int64           `json:"size_bytes"`
	SHA256       string          `json:"sha256"`
	CreatedAt    string          `json:"created_at"`
	PreviewURL   string          `json:"preview_url,omitempty"`
	Principal    principalRecord `json:"principal,omitempty"`
}

type uploadFilesResponse struct {
	FileCount int                  `json:"file_count"`
	Uploaded  []uploadedFileRecord `json:"uploaded"`
}

type resourceRecord struct {
	FileID        string          `json:"file_id"`
	OriginalName  string          `json:"original_name"`
	ContentType   string          `json:"content_type,omitempty"`
	SizeBytes     int64           `json:"size_bytes"`
	SHA256        string          `json:"sha256"`
	CreatedAt     string          `json:"created_at"`
	SourceType    string          `json:"source_type"`
	ResourceKind  string          `json:"resource_kind"`
	SourceURI     string          `json:"source_uri,omitempty"`
	HasThumbnail  bool            `json:"has_thumbnail"`
	ThumbnailURL  string          `json:"thumbnail_url,omitempty"`
	PreviewURL    string          `json:"preview_url,omitempty"`
	CacheReady    bool            `json:"cache_ready"`
	StagedLocally bool            `json:"staged_locally"`
	Principal     principalRecord `json:"principal,omitempty"`
}

type principalRecord struct {
	UserID string `json:"user_id,omitempty"`
	OrgID  string `json:"org_id,omitempty"`
	Role   string `json:"role,omitempty"`
}

type resourcesResponse struct {
	Count     int              `json:"count"`
	Resources []resourceRecord `json:"resources"`
}

type resourceResponse struct {
	Resource resourceRecord `json:"resource"`
}

type adminPlatformKPIs struct {
	TotalUsers                 int     `json:"total_users"`
	ActiveUsers24h             int     `json:"active_users_24h"`
	TotalConversations         int     `json:"total_conversations"`
	ConversationsStarted24h    int     `json:"conversations_started_24h"`
	TotalMessages              int     `json:"total_messages"`
	MessagesLast24h            int     `json:"messages_last_24h"`
	UserMessagesLast24h        int     `json:"user_messages_last_24h"`
	AssistantMessagesLast24h   int     `json:"assistant_messages_last_24h"`
	TotalRuns                  int     `json:"total_runs"`
	RunsLast24h                int     `json:"runs_last_24h"`
	SuccessRateLast24h         float64 `json:"success_rate_last_24h"`
	RunningRuns                int     `json:"running_runs"`
	StaleRunningRuns           int     `json:"stale_running_runs"`
	FailedRuns24h              int     `json:"failed_runs_24h"`
	TotalUploads               int     `json:"total_uploads"`
	SoftDeletedUploads         int     `json:"soft_deleted_uploads"`
	TotalStorageBytes          int64   `json:"total_storage_bytes"`
	AvgMessagesPerConversation float64 `json:"avg_messages_per_conversation"`
}

type RuntimeSummary struct {
	AppVersion              string  `json:"app_version,omitempty"`
	StoreBackend            string  `json:"store_backend"`
	DispatchMode            string  `json:"dispatch_mode"`
	JobTransport            string  `json:"job_transport"`
	EventTransport          string  `json:"event_transport"`
	StubWorkerEnabled       bool    `json:"stub_worker_enabled"`
	NATSConfigured          bool    `json:"nats_configured"`
	NATSStream              string  `json:"nats_stream,omitempty"`
	NATSJobsSubject         string  `json:"nats_jobs_subject,omitempty"`
	NATSRareSpotJobsSubject string  `json:"nats_rarespot_jobs_subject,omitempty"`
	NATSEventsSubject       string  `json:"nats_events_subject,omitempty"`
	NATSCancelSubject       string  `json:"nats_cancel_subject,omitempty"`
	NATSEventConsumer       string  `json:"nats_event_consumer,omitempty"`
	ArtifactRoot            string  `json:"artifact_root,omitempty"`
	UploadRoot              string  `json:"upload_root,omitempty"`
	RunRecoveryEnabled      bool    `json:"run_recovery_enabled"`
	RunRecoveryIntervalSecs float64 `json:"run_recovery_interval_seconds,omitempty"`
	RunRecoveryBatchLimit   int     `json:"run_recovery_batch_limit,omitempty"`
}

type adminQueueDiagnostics struct {
	Available      bool                           `json:"available"`
	Mode           string                         `json:"mode"`
	Stream         string                         `json:"stream,omitempty"`
	StreamSubjects []string                       `json:"stream_subjects,omitempty"`
	StreamMessages uint64                         `json:"stream_messages"`
	StreamBytes    uint64                         `json:"stream_bytes"`
	FirstSequence  uint64                         `json:"first_sequence"`
	LastSequence   uint64                         `json:"last_sequence"`
	ConsumerCount  int                            `json:"consumer_count"`
	Consumers      []adminQueueConsumerDiagnostic `json:"consumers"`
	Error          string                         `json:"error,omitempty"`
}

type adminQueueConsumerDiagnostic struct {
	Name                    string  `json:"name"`
	Role                    string  `json:"role,omitempty"`
	Subject                 string  `json:"subject,omitempty"`
	Active                  bool    `json:"active"`
	AckWaitSeconds          float64 `json:"ack_wait_seconds,omitempty"`
	MaxDeliver              int     `json:"max_deliver,omitempty"`
	PendingMessages         uint64  `json:"pending_messages"`
	InFlightMessages        int     `json:"in_flight_messages"`
	RedeliveredMessages     int     `json:"redelivered_messages"`
	WaitingPullRequests     int     `json:"waiting_pull_requests"`
	DeliveredStreamSequence uint64  `json:"delivered_stream_sequence,omitempty"`
	AckFloorStreamSequence  uint64  `json:"ack_floor_stream_sequence,omitempty"`
	Error                   string  `json:"error,omitempty"`
}

type adminUsageBucket struct {
	BucketStart   string `json:"bucket_start"`
	RunsTotal     int    `json:"runs_total"`
	RunsSucceeded int    `json:"runs_succeeded"`
	RunsFailed    int    `json:"runs_failed"`
	Uploads       int    `json:"uploads"`
	NewUsers      int    `json:"new_users"`
}

type adminToolUsageRecord struct {
	ToolName  string `json:"tool_name"`
	Count     int    `json:"count"`
	Succeeded int    `json:"succeeded"`
	Failed    int    `json:"failed"`
}

type adminUserSummary struct {
	UserID         string  `json:"user_id"`
	Email          string  `json:"email,omitempty"`
	DisplayName    string  `json:"display_name,omitempty"`
	Role           string  `json:"role,omitempty"`
	Status         string  `json:"status,omitempty"`
	OrgID          string  `json:"org_id,omitempty"`
	CreatedAt      string  `json:"created_at,omitempty"`
	Conversations  int     `json:"conversations"`
	Messages       int     `json:"messages"`
	RunsTotal      int     `json:"runs_total"`
	RunsRunning    int     `json:"runs_running"`
	RunsFailed     int     `json:"runs_failed"`
	RunsSucceeded  int     `json:"runs_succeeded"`
	Uploads        int     `json:"uploads"`
	StorageBytes   int64   `json:"storage_bytes"`
	LastActivityAt *string `json:"last_activity_at,omitempty"`
}

type adminRunRecord struct {
	RunID                      string   `json:"run_id"`
	UserID                     string   `json:"user_id,omitempty"`
	ConversationID             string   `json:"conversation_id,omitempty"`
	Goal                       string   `json:"goal"`
	Status                     string   `json:"status"`
	CreatedAt                  string   `json:"created_at"`
	UpdatedAt                  string   `json:"updated_at"`
	Error                      string   `json:"error,omitempty"`
	DurationSeconds            *float64 `json:"duration_seconds,omitempty"`
	ToolNames                  []string `json:"tool_names"`
	LastEventKind              *string  `json:"last_event_kind,omitempty"`
	LastEventAt                *string  `json:"last_event_at,omitempty"`
	LastEventSequence          *int64   `json:"last_event_sequence,omitempty"`
	LastActivityAgeSeconds     *float64 `json:"last_activity_age_seconds,omitempty"`
	EventCount                 int      `json:"event_count"`
	MessageDeltaCount          int      `json:"message_delta_count"`
	ToolCallCount              int      `json:"tool_call_count"`
	ArtifactCount              int      `json:"artifact_count"`
	HeartbeatCount             int      `json:"heartbeat_count"`
	LastToolName               *string  `json:"last_tool_name,omitempty"`
	LastToolAt                 *string  `json:"last_tool_at,omitempty"`
	FirstDeltaLatency          *float64 `json:"first_delta_latency_seconds,omitempty"`
	FirstToolLatency           *float64 `json:"first_tool_latency_seconds,omitempty"`
	FirstArtifactLatency       *float64 `json:"first_artifact_latency_seconds,omitempty"`
	LeaseWorkerID              *string  `json:"lease_worker_id,omitempty"`
	LeaseExpiresAt             *string  `json:"lease_expires_at,omitempty"`
	LeaseActive                bool     `json:"lease_active"`
	LeaseExpired               bool     `json:"lease_expired"`
	LeaseSecondsRemaining      *float64 `json:"lease_seconds_remaining,omitempty"`
	LeaseLastRenewedAt         *string  `json:"lease_last_renewed_at,omitempty"`
	LeaseLastRenewedAgeSeconds *float64 `json:"lease_last_renewed_age_seconds,omitempty"`
	Stale                      bool     `json:"stale"`
	StaleReason                *string  `json:"stale_reason,omitempty"`
}

type adminIssueRecord struct {
	IssueType      string         `json:"issue_type"`
	Severity       string         `json:"severity"`
	UserID         string         `json:"user_id,omitempty"`
	RunID          string         `json:"run_id,omitempty"`
	UploadID       string         `json:"upload_id,omitempty"`
	ConversationID string         `json:"conversation_id,omitempty"`
	Message        string         `json:"message"`
	OccurredAt     string         `json:"occurred_at"`
	Metadata       domain.JSONMap `json:"metadata"`
}

type workerHeartbeatRequest struct {
	WorkerID        string         `json:"worker_id"`
	WorkerKind      string         `json:"worker_kind"`
	Status          string         `json:"status"`
	CurrentRunID    string         `json:"current_run_id"`
	Hostname        string         `json:"hostname"`
	Version         string         `json:"version"`
	StartedAt       string         `json:"started_at"`
	LastHeartbeatAt string         `json:"last_heartbeat_at"`
	Metadata        domain.JSONMap `json:"metadata"`
}

type adminWorkerRecord struct {
	WorkerID            string         `json:"worker_id"`
	WorkerKind          string         `json:"worker_kind"`
	Status              string         `json:"status"`
	CurrentRunID        *string        `json:"current_run_id,omitempty"`
	Hostname            *string        `json:"hostname,omitempty"`
	Version             *string        `json:"version,omitempty"`
	StartedAt           string         `json:"started_at"`
	LastHeartbeatAt     string         `json:"last_heartbeat_at"`
	UpdatedAt           string         `json:"updated_at"`
	HeartbeatAgeSeconds *float64       `json:"heartbeat_age_seconds,omitempty"`
	Active              bool           `json:"active"`
	Stale               bool           `json:"stale"`
	Metadata            domain.JSONMap `json:"metadata"`
}

type adminOverviewResponse struct {
	GeneratedAt  string                 `json:"generated_at"`
	Runtime      RuntimeSummary         `json:"runtime"`
	Queue        adminQueueDiagnostics  `json:"queue"`
	KPIs         adminPlatformKPIs      `json:"kpis"`
	UsageLast24h []adminUsageBucket     `json:"usage_last_24h"`
	ToolUsage7d  []adminToolUsageRecord `json:"tool_usage_7d"`
	Workers      []adminWorkerRecord    `json:"workers"`
	TopUsers     []adminUserSummary     `json:"top_users"`
	RecentIssues []adminIssueRecord     `json:"recent_issues"`
}

type adminUserListResponse struct {
	Count int                `json:"count"`
	Users []adminUserSummary `json:"users"`
}

type adminOrganizationListResponse struct {
	Count         int                   `json:"count"`
	Organizations []domain.Organization `json:"organizations"`
}

type adminCreateOrganizationRequest struct {
	OrgID    string         `json:"org_id"`
	Name     string         `json:"name"`
	Status   string         `json:"status"`
	Metadata domain.JSONMap `json:"metadata"`
}

type adminCreateUserRequest struct {
	UserID      string         `json:"user_id"`
	Email       string         `json:"email"`
	DisplayName string         `json:"display_name"`
	Role        string         `json:"role"`
	Status      string         `json:"status"`
	OrgID       string         `json:"org_id"`
	Metadata    domain.JSONMap `json:"metadata"`
}

type adminRunListResponse struct {
	Count int              `json:"count"`
	Runs  []adminRunRecord `json:"runs"`
}

type adminIssueListResponse struct {
	Count  int                `json:"count"`
	Issues []adminIssueRecord `json:"issues"`
}

type adminRunActionResponse struct {
	RunID          string `json:"run_id"`
	PreviousStatus string `json:"previous_status"`
	Status         string `json:"status"`
	Updated        bool   `json:"updated"`
}

type trainingModelRecord struct {
	Key               string         `json:"key"`
	Name              string         `json:"name"`
	Framework         string         `json:"framework"`
	TaskType          string         `json:"task_type"`
	Description       string         `json:"description"`
	SupportsTraining  bool           `json:"supports_training"`
	SupportsFinetune  bool           `json:"supports_finetune"`
	SupportsInference bool           `json:"supports_inference"`
	Dimensions        []string       `json:"dimensions"`
	DefaultConfig     domain.JSONMap `json:"default_config"`
}

type trainingModelsResponse struct {
	Count  int                   `json:"count"`
	Models []trainingModelRecord `json:"models"`
}

func (deps ServerDeps) handleListThreads(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	threads, err := deps.Store.ListThreads(r.Context(), parseLimit(r, 100))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, listThreadsResponse{Count: len(threads), Threads: threads})
}

func (deps ServerDeps) handleCreateThread(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req createThreadRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	principal := principalFromRequest(r, req.UserID)
	thread, err := deps.Runs.CreateThread(r.Context(), runcontrol.CreateThreadRequest{
		UserID:          principal.UserID,
		Title:           req.Title,
		Metadata:        metadataWithPrincipal(domain.JSONMap(req.Metadata), principal),
		InitialMessages: req.InitialMessages,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, thread)
}

func (deps ServerDeps) handleGetThread(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	thread, err := deps.Store.GetThread(r.Context(), chi.URLParam(r, "thread_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, thread)
}

func (deps ServerDeps) handleListThreadMessages(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	threadID := chi.URLParam(r, "thread_id")
	messages, err := deps.Store.ListThreadMessages(r.Context(), threadID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, threadMessagesResponse{ThreadID: threadID, Count: len(messages), Messages: messages})
}

func (deps ServerDeps) handleCreateRun(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req createRunRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	principal := principalFromRequest(r, req.UserID)
	run, err := deps.Runs.CreateRun(r.Context(), runcontrol.CreateRunRequest{
		ThreadID:            chi.URLParam(r, "thread_id"),
		UserID:              principal.UserID,
		Goal:                req.Goal,
		Messages:            req.Messages,
		FileIDs:             req.FileIDs,
		ResourceURIs:        req.ResourceURIs,
		DatasetURIs:         req.DatasetURIs,
		SelectedToolNames:   req.SelectedToolNames,
		KnowledgeContext:    domain.JSONMap(req.KnowledgeContext),
		WorkflowHint:        domain.JSONMap(req.WorkflowHint),
		SelectionContext:    domain.JSONMap(req.SelectionContext),
		ReasoningMode:       req.ReasoningMode,
		Budgets:             domain.JSONMap(req.Budgets),
		Benchmark:           domain.JSONMap(req.Benchmark),
		ResourceDescriptors: req.ResourceDescriptors,
		IdempotencyKey:      idempotencyKeyFromRequest(r, req.IdempotencyKey),
		Metadata:            metadataWithPrincipal(domain.JSONMap(req.Metadata), principal),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, run)
}

func idempotencyKeyFromRequest(r *http.Request, bodyValue string) string {
	if token := strings.TrimSpace(r.Header.Get("Idempotency-Key")); token != "" {
		return token
	}
	return strings.TrimSpace(bodyValue)
}

func (deps ServerDeps) handleUploadFiles(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if r.MultipartForm == nil || len(r.MultipartForm.File["files"]) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("multipart upload must include at least one files entry"))
		return
	}

	principal := principalFromRequest(r, "")
	uploaded := make([]uploadedFileRecord, 0, len(r.MultipartForm.File["files"]))
	for _, header := range r.MultipartForm.File["files"] {
		record, err := saveUploadedFile(root, header, principal)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		uploaded = append(uploaded, record)
	}
	writeJSON(w, http.StatusOK, uploadFilesResponse{FileCount: len(uploaded), Uploaded: uploaded})
}

func (deps ServerDeps) handleListResources(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	resources, err := listUploadResources(root)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	query := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("q")))
	kind := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("kind")))
	filtered := resources[:0]
	for _, resource := range resources {
		if query != "" && !strings.Contains(strings.ToLower(resource.OriginalName), query) && !strings.Contains(strings.ToLower(resource.FileID), query) {
			continue
		}
		if kind != "" && resource.ResourceKind != kind {
			continue
		}
		filtered = append(filtered, resource)
	}
	offset := parseOffset(r)
	limit := parseLimit(r, 200)
	paged := []resourceRecord{}
	if offset < len(filtered) {
		end := offset + limit
		if end > len(filtered) {
			end = len(filtered)
		}
		paged = filtered[offset:end]
	}
	writeJSON(w, http.StatusOK, resourcesResponse{Count: len(filtered), Resources: paged})
}

func (deps ServerDeps) handleGetResource(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	record, _, err := findUploadResource(root, chi.URLParam(r, "file_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, resourceResponse{Resource: record})
}

func (deps ServerDeps) handleDeleteResource(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	fileID := chi.URLParam(r, "file_id")
	_, path, err := findUploadResource(root, fileID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if err := os.Remove(path); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	_ = os.Remove(uploadMetadataPath(root, fileID))
	writeJSON(w, http.StatusOK, map[string]any{"deleted": true, "file_id": fileID})
}

func (deps ServerDeps) handleServeUpload(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	record, path, err := findUploadResource(root, chi.URLParam(r, "file_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if record.ContentType != "" {
		w.Header().Set("Content-Type", record.ContentType)
	}
	http.ServeFile(w, r, path)
}

func (deps ServerDeps) handleServeUploadSlice(w http.ResponseWriter, r *http.Request) {
	// V2 direct-image uploads do not need server-side slicing; the viewer can
	// request the original image bytes for any 2D slice plane.
	deps.handleServeUpload(w, r)
}

func (deps ServerDeps) handleGetUploadCaption(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	record, path, err := findUploadResource(root, chi.URLParam(r, "file_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	width, height, warnings := uploadImageDimensions(path)
	contentType := strings.TrimSpace(record.ContentType)
	caption := fmt.Sprintf("Uploaded file %s", record.OriginalName)
	if strings.HasPrefix(contentType, "image/") {
		if len(warnings) == 0 && width > 0 && height > 0 {
			caption = fmt.Sprintf(
				"Uploaded image %s (%d x %d pixels, %s).",
				record.OriginalName,
				width,
				height,
				contentType,
			)
		} else {
			caption = fmt.Sprintf("Uploaded image %s (%s); detailed dimensions are unavailable.", record.OriginalName, contentType)
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"file_id": record.FileID,
		"caption": caption,
		"source":  "fallback",
	})
}

func (deps ServerDeps) handleGetUploadViewer(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	record, path, err := findUploadResource(root, chi.URLParam(r, "file_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	width, height, warnings := uploadImageDimensions(path)
	fileIDSegment := url.PathEscape(record.FileID)
	channelCount := uploadChannelCount(record.ContentType)
	serviceURLs := map[string]any{
		"preview": "/v2/uploads/" + fileIDSegment + "/preview",
		"display": "/v2/uploads/" + fileIDSegment + "/display",
		"slice":   "/v2/uploads/" + fileIDSegment + "/slice",
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"kind":          "image",
		"file_id":       record.FileID,
		"original_name": record.OriginalName,
		"modality":      "image",
		"backend_mode":  "direct",
		"dims_order":    "YXC",
		"axis_sizes": map[string]int{
			"T": 1,
			"C": channelCount,
			"Z": 1,
			"Y": height,
			"X": width,
		},
		"selected_indices": map[string]int{"T": 0, "C": 0, "Z": 0},
		"is_volume":        false,
		"is_timeseries":    false,
		"is_multichannel":  channelCount > 1,
		"service_urls":     serviceURLs,
		"metadata": map[string]any{
			"reader":       "go-image",
			"dims_order":   "YXC",
			"array_shape":  []int{height, width, channelCount},
			"array_dtype":  "uint8",
			"scene_count":  1,
			"warnings":     warnings,
			"content_type": record.ContentType,
			"size_bytes":   record.SizeBytes,
			"sha256":       record.SHA256,
		},
		"viewer": map[string]any{
			"status":             "ready",
			"warmup_mode":        "lazy",
			"backend_mode":       "direct",
			"default_surface":    "2d",
			"available_surfaces": []string{"2d", "metadata"},
			"service_urls":       serviceURLs,
			"asset_preparation": map[string]any{
				"status":                "ready",
				"native_supported":      true,
				"tile_pyramid":          "none",
				"volume_representation": "none",
			},
		},
	})
}

func (deps ServerDeps) handleListRuns(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	threadID := strings.TrimSpace(r.URL.Query().Get("thread_id"))
	status := strings.TrimSpace(r.URL.Query().Get("status"))
	runs, err := deps.Store.ListRuns(r.Context(), threadID, status, parseLimit(r, 100), parseOffset(r))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, runsResponse{Count: len(runs), Runs: runs})
}

func (deps ServerDeps) handleGetRun(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	run, err := deps.Store.GetRun(r.Context(), chi.URLParam(r, "run_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, run)
}

func (deps ServerDeps) handleAdminOverview(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	data, err := deps.loadAdminSnapshot(r.Context())
	if err != nil {
		writeStoreError(w, err)
		return
	}
	topUsers := take(data.Users, parseLimitParam(r, "top_users", 8))
	recentIssues := take(data.Issues, parseLimitParam(r, "issue_limit", 12))
	writeJSON(w, http.StatusOK, adminOverviewResponse{
		GeneratedAt:  data.GeneratedAt,
		Runtime:      deps.adminRuntimeSummary(),
		Queue:        deps.adminQueueDiagnostics(r.Context()),
		KPIs:         data.KPIs,
		UsageLast24h: data.UsageLast24h,
		ToolUsage7d:  data.ToolUsage7d,
		Workers:      data.Workers,
		TopUsers:     topUsers,
		RecentIssues: recentIssues,
	})
}

func (deps ServerDeps) handleWorkerHeartbeat(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	workers, ok := deps.Store.(workerHeartbeatStore)
	if !ok {
		writeError(w, http.StatusServiceUnavailable, errors.New("worker heartbeat store is not configured"))
		return
	}
	var req workerHeartbeatRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	workerID := strings.TrimSpace(req.WorkerID)
	if workerID == "" {
		writeError(w, http.StatusBadRequest, errors.New("worker_id is required"))
		return
	}
	startedAt, err := parseOptionalTime(req.StartedAt)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid started_at: %w", err))
		return
	}
	lastHeartbeatAt, err := parseOptionalTime(req.LastHeartbeatAt)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid last_heartbeat_at: %w", err))
		return
	}
	worker, err := workers.UpsertWorkerHeartbeat(r.Context(), domain.UpsertWorkerHeartbeatInput{
		WorkerID:        workerID,
		WorkerKind:      req.WorkerKind,
		Status:          req.Status,
		CurrentRunID:    req.CurrentRunID,
		Hostname:        req.Hostname,
		Version:         req.Version,
		StartedAt:       startedAt,
		LastHeartbeatAt: lastHeartbeatAt,
		Metadata:        domain.JSONMap(req.Metadata),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, worker)
}

func (deps ServerDeps) adminQueueDiagnostics(ctx context.Context) adminQueueDiagnostics {
	runtime := deps.adminRuntimeSummary()
	if deps.QueueDiagnostics == nil {
		return adminQueueDiagnostics{
			Available: false,
			Mode:      runtime.DispatchMode,
			Stream:    runtime.NATSStream,
			Error:     "queue diagnostics are not configured",
		}
	}
	diagnostics, err := deps.QueueDiagnostics.QueueDiagnostics(ctx)
	if err != nil && diagnostics.Error == "" {
		diagnostics.Error = err.Error()
	}
	return adminQueueDiagnosticsFromEventBus(diagnostics)
}

func adminQueueDiagnosticsFromEventBus(diagnostics eventbus.QueueDiagnostics) adminQueueDiagnostics {
	consumers := make([]adminQueueConsumerDiagnostic, 0, len(diagnostics.Consumers))
	for _, consumer := range diagnostics.Consumers {
		consumers = append(consumers, adminQueueConsumerDiagnostic{
			Name:                    consumer.Name,
			Role:                    consumer.Role,
			Subject:                 consumer.Subject,
			Active:                  consumer.Active,
			AckWaitSeconds:          consumer.AckWaitSeconds,
			MaxDeliver:              consumer.MaxDeliver,
			PendingMessages:         consumer.PendingMessages,
			InFlightMessages:        consumer.InFlightMessages,
			RedeliveredMessages:     consumer.RedeliveredMessages,
			WaitingPullRequests:     consumer.WaitingPullRequests,
			DeliveredStreamSequence: consumer.DeliveredStreamSequence,
			AckFloorStreamSequence:  consumer.AckFloorStreamSequence,
			Error:                   consumer.Error,
		})
	}
	return adminQueueDiagnostics{
		Available:      diagnostics.Available,
		Mode:           diagnostics.Mode,
		Stream:         diagnostics.Stream,
		StreamSubjects: append([]string(nil), diagnostics.StreamSubjects...),
		StreamMessages: diagnostics.StreamMessages,
		StreamBytes:    diagnostics.StreamBytes,
		FirstSequence:  diagnostics.FirstSequence,
		LastSequence:   diagnostics.LastSequence,
		ConsumerCount:  diagnostics.ConsumerCount,
		Consumers:      consumers,
		Error:          diagnostics.Error,
	}
}

func (deps ServerDeps) adminRuntimeSummary() RuntimeSummary {
	runtime := deps.Runtime
	if strings.TrimSpace(runtime.AppVersion) == "" {
		runtime.AppVersion = deps.Version
	}
	if strings.TrimSpace(runtime.StoreBackend) == "" {
		runtime.StoreBackend = "memory"
	}
	if strings.TrimSpace(runtime.DispatchMode) == "" {
		runtime.DispatchMode = "local_memory"
	}
	if strings.TrimSpace(runtime.JobTransport) == "" {
		runtime.JobTransport = runtime.DispatchMode
	}
	if strings.TrimSpace(runtime.EventTransport) == "" {
		runtime.EventTransport = runtime.DispatchMode
	}
	if strings.TrimSpace(runtime.ArtifactRoot) == "" {
		runtime.ArtifactRoot = deps.ArtifactRoot
	}
	if strings.TrimSpace(runtime.UploadRoot) == "" {
		runtime.UploadRoot = deps.UploadRoot
	}
	return runtime
}

func (deps ServerDeps) handleAdminUsers(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	data, err := deps.loadAdminSnapshot(r.Context())
	if err != nil {
		writeStoreError(w, err)
		return
	}
	query := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("q")))
	users := make([]adminUserSummary, 0, len(data.Users))
	for _, user := range data.Users {
		if query != "" && !adminUserMatchesQuery(user, query) {
			continue
		}
		users = append(users, user)
	}
	writeJSON(w, http.StatusOK, adminUserListResponse{Count: len(users), Users: take(users, parseLimit(r, 250))})
}

func (deps ServerDeps) handleAdminOrganizations(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	orgs, ok := deps.Store.(organizationStore)
	if !ok {
		writeJSON(w, http.StatusNotImplemented, map[string]any{
			"status":  "not_configured",
			"service": "ultra-control-v2",
			"detail":  "admin organization storage is not configured",
		})
		return
	}
	records, err := orgs.ListOrganizations(r.Context(), parseLimit(r, 250), r.URL.Query().Get("q"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, adminOrganizationListResponse{Count: len(records), Organizations: records})
}

func (deps ServerDeps) handleAdminCreateOrganization(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	orgs, ok := deps.Store.(organizationStore)
	if !ok {
		writeJSON(w, http.StatusNotImplemented, map[string]any{
			"status":  "not_configured",
			"service": "ultra-control-v2",
			"detail":  "admin organization storage is not configured",
		})
		return
	}
	var req adminCreateOrganizationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}
	if strings.TrimSpace(req.Name) == "" && strings.TrimSpace(req.OrgID) == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "name or org_id is required"})
		return
	}
	org, err := orgs.CreateOrganization(r.Context(), domain.CreateOrganizationInput{
		OrgID:    req.OrgID,
		Name:     req.Name,
		Status:   req.Status,
		Metadata: req.Metadata,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, org)
}

func (deps ServerDeps) handleAdminCreateUser(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	accounts, ok := deps.Store.(accountStore)
	if !ok {
		writeJSON(w, http.StatusNotImplemented, map[string]any{
			"status":  "not_configured",
			"service": "ultra-control-v2",
			"detail":  "admin account storage is not configured",
		})
		return
	}
	var req adminCreateUserRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}
	if strings.TrimSpace(req.Email) == "" && strings.TrimSpace(req.UserID) == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "email or user_id is required"})
		return
	}
	user, err := accounts.CreateUser(r.Context(), domain.CreateUserInput{
		UserID:      req.UserID,
		Email:       req.Email,
		DisplayName: req.DisplayName,
		Role:        req.Role,
		Status:      req.Status,
		OrgID:       req.OrgID,
		Metadata:    req.Metadata,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, user)
}

func (deps ServerDeps) handleAdminDeleteUser(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	accounts, ok := deps.Store.(accountStore)
	if !ok {
		writeJSON(w, http.StatusNotImplemented, map[string]any{
			"status":  "not_configured",
			"service": "ultra-control-v2",
			"detail":  "admin account storage is not configured",
		})
		return
	}
	userID := strings.TrimSpace(chi.URLParam(r, "user_id"))
	if userID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "user_id is required"})
		return
	}
	user, err := accounts.UpdateUserStatus(r.Context(), userID, "disabled")
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, user)
}

func (deps ServerDeps) handleAdminRuns(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	status := strings.TrimSpace(r.URL.Query().Get("status"))
	runs, err := deps.Store.ListRuns(r.Context(), "", status, parseLimit(r, 200), parseOffset(r))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	userID := strings.TrimSpace(r.URL.Query().Get("user_id"))
	query := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("q")))
	records := make([]adminRunRecord, 0, len(runs))
	for _, run := range runs {
		if userID != "" && run.UserID != userID {
			continue
		}
		if query != "" && !strings.Contains(strings.ToLower(run.Goal), query) && !strings.Contains(strings.ToLower(run.RunID), query) {
			continue
		}
		diagnostic, err := deps.adminRunDiagnostic(r.Context(), run, domain.Now())
		if err != nil {
			writeStoreError(w, err)
			return
		}
		records = append(records, adminRunFromRun(run, diagnostic))
	}
	writeJSON(w, http.StatusOK, adminRunListResponse{Count: len(records), Runs: records})
}

func (deps ServerDeps) handleAdminIssues(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	data, err := deps.loadAdminSnapshot(r.Context())
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, adminIssueListResponse{Count: len(data.Issues), Issues: take(data.Issues, parseLimit(r, 25))})
}

func (deps ServerDeps) handleAdminCancelRun(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	runID := chi.URLParam(r, "run_id")
	before, err := deps.Store.GetRun(r.Context(), runID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	after, err := deps.Runs.CancelRun(r.Context(), runcontrol.CancelRunRequest{RunID: runID, Reason: "admin cancel"})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, adminRunActionResponse{
		RunID:          runID,
		PreviousStatus: string(before.Status),
		Status:         string(after.Status),
		Updated:        before.Status != after.Status,
	})
}

func (deps ServerDeps) handleAdminRequeueRun(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req cancelRunRequest
	if r.Body != nil && r.ContentLength != 0 {
		if !decodeJSON(w, r, &req) {
			return
		}
	}
	runID := chi.URLParam(r, "run_id")
	before, err := deps.Store.GetRun(r.Context(), runID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	reason := strings.TrimSpace(req.Reason)
	if reason == "" {
		reason = "admin requeue"
	}
	after, err := deps.Runs.RequeueRun(r.Context(), runcontrol.RequeueRunRequest{
		RunID:    runID,
		Reason:   reason,
		Metadata: domain.JSONMap(req.Metadata),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, adminRunActionResponse{
		RunID:          runID,
		PreviousStatus: string(before.Status),
		Status:         string(after.Status),
		Updated:        true,
	})
}

type adminSnapshot struct {
	GeneratedAt  string
	KPIs         adminPlatformKPIs
	UsageLast24h []adminUsageBucket
	ToolUsage7d  []adminToolUsageRecord
	Workers      []adminWorkerRecord
	Users        []adminUserSummary
	Issues       []adminIssueRecord
}

func (deps ServerDeps) loadAdminSnapshot(ctx context.Context) (adminSnapshot, error) {
	now := domain.Now()
	since := now.Add(-24 * time.Hour)
	threads, err := deps.Store.ListThreads(ctx, 10000)
	if err != nil {
		return adminSnapshot{}, err
	}
	runs, err := deps.Store.ListRuns(ctx, "", "", 10000, 0)
	if err != nil {
		return adminSnapshot{}, err
	}
	uploads, storageBytes := deps.uploadStats()
	workers := []adminWorkerRecord{}
	if workerStore, ok := deps.Store.(workerHeartbeatStore); ok {
		records, err := workerStore.ListWorkerHeartbeats(ctx, 250)
		if err != nil {
			return adminSnapshot{}, err
		}
		workers = adminWorkerRecords(records, now)
	}

	users := map[string]*adminUserSummary{}
	if accounts, ok := deps.Store.(accountStore); ok {
		records, err := accounts.ListUsers(ctx, 10000, "")
		if err != nil {
			return adminSnapshot{}, err
		}
		for _, record := range records {
			summary := adminUser(users, record.UserID)
			applyAccountToAdminUser(summary, record)
		}
	}
	userSeen24h := map[string]bool{}
	totalMessages := 0
	messages24h := 0
	userMessages24h := 0
	assistantMessages24h := 0
	conversationsStarted24h := 0
	for _, thread := range threads {
		user := adminUser(users, thread.UserID)
		user.Conversations++
		updateLastActivity(user, thread.UpdatedAt)
		if thread.CreatedAt.After(since) {
			conversationsStarted24h++
			userSeen24h[user.UserID] = true
		}
		messages, err := deps.Store.ListThreadMessages(ctx, thread.ThreadID)
		if err != nil {
			return adminSnapshot{}, err
		}
		user.Messages += len(messages)
		totalMessages += len(messages)
		for _, message := range messages {
			if message.CreatedAt.Before(since) {
				continue
			}
			messages24h++
			switch message.Role {
			case "assistant":
				assistantMessages24h++
			case "user":
				userMessages24h++
			}
			userSeen24h[user.UserID] = true
		}
	}

	runs24h := 0
	runsSucceeded24h := 0
	runsFailed24h := 0
	runningRuns := 0
	staleRunningRuns := 0
	toolUsage := map[string]*adminToolUsageRecord{}
	issues := []adminIssueRecord{}
	for _, run := range runs {
		diagnostic, err := deps.adminRunDiagnostic(ctx, run, now)
		if err != nil {
			return adminSnapshot{}, err
		}
		user := adminUser(users, run.UserID)
		user.RunsTotal++
		updateLastActivity(user, run.UpdatedAt)
		if run.UpdatedAt.After(since) {
			userSeen24h[user.UserID] = true
		}
		switch run.Status {
		case domain.RunStatusRunning, domain.RunStatusQueued, domain.RunStatusWaitingForInput, domain.RunStatusWaitingForTask:
			user.RunsRunning++
			runningRuns++
			if diagnostic.Stale {
				staleRunningRuns++
			}
		case domain.RunStatusFailed:
			user.RunsFailed++
		case domain.RunStatusSucceeded:
			user.RunsSucceeded++
		}
		if run.CreatedAt.After(since) {
			runs24h++
			if run.Status == domain.RunStatusSucceeded {
				runsSucceeded24h++
			}
			if run.Status == domain.RunStatusFailed {
				runsFailed24h++
			}
		}
		if run.Status == domain.RunStatusFailed {
			issues = append(issues, adminIssueRecord{
				IssueType:      "failed_run",
				Severity:       "high",
				UserID:         run.UserID,
				RunID:          run.RunID,
				ConversationID: run.ThreadID,
				Message:        firstNonEmpty(run.Error, "Run failed."),
				OccurredAt:     run.UpdatedAt.UTC().Format(time.RFC3339Nano),
				Metadata:       domain.JSONMap{"status": string(run.Status)},
			})
		}
		if diagnostic.Stale {
			leaseWorkerID, leaseExpiresAt, _ := leaseTimeFields(diagnostic.Lease)
			issues = append(issues, adminIssueRecord{
				IssueType:      "stalled_run",
				Severity:       "high",
				UserID:         run.UserID,
				RunID:          run.RunID,
				ConversationID: run.ThreadID,
				Message:        firstNonEmpty(diagnostic.StaleReason, "Run has not emitted worker progress recently."),
				OccurredAt:     diagnostic.LastActivityAt.UTC().Format(time.RFC3339Nano),
				Metadata: domain.JSONMap{
					"status":                    string(run.Status),
					"last_event_kind":           valueOrEmpty(diagnostic.LastEventKind),
					"last_event_at":             timePtrString(diagnostic.LastEventAt),
					"last_event_sequence":       int64PtrValue(diagnostic.LastEventSequence),
					"last_activity_age_seconds": diagnostic.LastActivityAgeSeconds,
					"stale_after_seconds":       adminStaleRunThreshold.Seconds(),
					"lease_worker_id":           valueOrEmpty(leaseWorkerID),
					"lease_expires_at":          valueOrEmpty(leaseExpiresAt),
					"lease_active":              diagnostic.LeaseActive,
					"lease_expired":             diagnostic.LeaseExpired,
				},
			})
		}
		for _, toolName := range metadataStringSlice(run.Metadata["selected_tool_names"]) {
			record := toolUsage[toolName]
			if record == nil {
				record = &adminToolUsageRecord{ToolName: toolName}
				toolUsage[toolName] = record
			}
			record.Count++
			if run.Status == domain.RunStatusSucceeded {
				record.Succeeded++
			}
			if run.Status == domain.RunStatusFailed {
				record.Failed++
			}
		}
	}

	userList := make([]adminUserSummary, 0, len(users))
	for _, user := range users {
		userList = append(userList, *user)
	}
	sort.Slice(userList, func(i, j int) bool {
		return userList[i].RunsTotal > userList[j].RunsTotal
	})
	issueList := issues
	sort.Slice(issueList, func(i, j int) bool {
		return issueList[i].OccurredAt > issueList[j].OccurredAt
	})
	toolList := make([]adminToolUsageRecord, 0, len(toolUsage))
	for _, record := range toolUsage {
		toolList = append(toolList, *record)
	}
	sort.Slice(toolList, func(i, j int) bool {
		return toolList[i].Count > toolList[j].Count
	})
	successRate := 0.0
	if runs24h > 0 {
		successRate = float64(runsSucceeded24h) / float64(runs24h)
	}
	avgMessages := 0.0
	if len(threads) > 0 {
		avgMessages = float64(totalMessages) / float64(len(threads))
	}
	return adminSnapshot{
		GeneratedAt: now.UTC().Format(time.RFC3339Nano),
		KPIs: adminPlatformKPIs{
			TotalUsers:                 len(users),
			ActiveUsers24h:             len(userSeen24h),
			TotalConversations:         len(threads),
			ConversationsStarted24h:    conversationsStarted24h,
			TotalMessages:              totalMessages,
			MessagesLast24h:            messages24h,
			UserMessagesLast24h:        userMessages24h,
			AssistantMessagesLast24h:   assistantMessages24h,
			TotalRuns:                  len(runs),
			RunsLast24h:                runs24h,
			SuccessRateLast24h:         successRate,
			RunningRuns:                runningRuns,
			StaleRunningRuns:           staleRunningRuns,
			FailedRuns24h:              runsFailed24h,
			TotalUploads:               uploads,
			SoftDeletedUploads:         0,
			TotalStorageBytes:          storageBytes,
			AvgMessagesPerConversation: avgMessages,
		},
		UsageLast24h: []adminUsageBucket{{
			BucketStart:   since.UTC().Format(time.RFC3339Nano),
			RunsTotal:     runs24h,
			RunsSucceeded: runsSucceeded24h,
			RunsFailed:    runsFailed24h,
			Uploads:       uploads,
			NewUsers:      len(userSeen24h),
		}},
		ToolUsage7d: toolList,
		Workers:     workers,
		Users:       userList,
		Issues:      issueList,
	}, nil
}

func adminWorkerRecords(workers []domain.WorkerHeartbeatRecord, now time.Time) []adminWorkerRecord {
	records := make([]adminWorkerRecord, 0, len(workers))
	for _, worker := range workers {
		lastHeartbeatAt := worker.LastHeartbeatAt
		if lastHeartbeatAt.IsZero() {
			lastHeartbeatAt = worker.UpdatedAt
		}
		age := now.Sub(lastHeartbeatAt)
		if age < 0 {
			age = 0
		}
		ageSeconds := age.Seconds()
		stale := age >= adminWorkerStaleThreshold
		records = append(records, adminWorkerRecord{
			WorkerID:            worker.WorkerID,
			WorkerKind:          worker.WorkerKind,
			Status:              worker.Status,
			CurrentRunID:        stringPtr(worker.CurrentRunID),
			Hostname:            stringPtr(worker.Hostname),
			Version:             stringPtr(worker.Version),
			StartedAt:           worker.StartedAt.UTC().Format(time.RFC3339Nano),
			LastHeartbeatAt:     lastHeartbeatAt.UTC().Format(time.RFC3339Nano),
			UpdatedAt:           worker.UpdatedAt.UTC().Format(time.RFC3339Nano),
			HeartbeatAgeSeconds: &ageSeconds,
			Active:              !stale,
			Stale:               stale,
			Metadata:            worker.Metadata,
		})
	}
	return records
}

type adminRunDiagnostic struct {
	LastEventKind              *string
	LastEventAt                *time.Time
	LastEventSequence          *int64
	LastActivityAt             time.Time
	LastActivityAgeSeconds     float64
	EventCount                 int
	MessageDeltaCount          int
	ToolCallCount              int
	ArtifactCount              int
	HeartbeatCount             int
	LastToolName               *string
	LastToolAt                 *time.Time
	FirstDeltaLatency          *float64
	FirstToolLatency           *float64
	FirstArtifactLatency       *float64
	Lease                      *domain.RunLeaseRecord
	LeaseActive                bool
	LeaseExpired               bool
	LeaseSecondsRemaining      *float64
	LeaseLastRenewedAgeSeconds *float64
	Stale                      bool
	StaleReason                string
}

func (deps ServerDeps) adminRunDiagnostic(ctx context.Context, run domain.RunRecord, now time.Time) (adminRunDiagnostic, error) {
	events, err := deps.Store.ListRunEvents(ctx, run.RunID, 10000)
	if err != nil {
		return adminRunDiagnostic{}, err
	}
	diagnostic := adminRunDiagnostic{
		LastActivityAt: run.UpdatedAt,
		EventCount:     len(events),
	}
	if diagnostic.LastActivityAt.IsZero() {
		diagnostic.LastActivityAt = run.CreatedAt
	}
	if len(events) > 0 {
		latest := events[len(events)-1]
		latestTS := latest.TS
		if latestTS.IsZero() {
			latestTS = diagnostic.LastActivityAt
		}
		diagnostic.LastEventKind = stringPtr(latest.EventKind)
		diagnostic.LastEventAt = &latestTS
		diagnostic.LastEventSequence = int64Ptr(latest.Sequence)
		diagnostic.LastActivityAt = latestTS
	}
	for _, event := range events {
		eventTS := event.TS
		if eventTS.IsZero() {
			eventTS = run.CreatedAt
		}
		switch event.EventKind {
		case "message.delta":
			diagnostic.MessageDeltaCount++
			if diagnostic.FirstDeltaLatency == nil {
				diagnostic.FirstDeltaLatency = runLatencySeconds(run.CreatedAt, eventTS)
			}
		case "tool_call.started":
			diagnostic.ToolCallCount++
			if diagnostic.FirstToolLatency == nil {
				diagnostic.FirstToolLatency = runLatencySeconds(run.CreatedAt, eventTS)
			}
			if toolName := eventToolName(event.Payload); toolName != "" {
				diagnostic.LastToolName = stringPtr(toolName)
				diagnostic.LastToolAt = timePtr(eventTS)
			}
		case "artifact.created":
			diagnostic.ArtifactCount++
			if diagnostic.FirstArtifactLatency == nil {
				diagnostic.FirstArtifactLatency = runLatencySeconds(run.CreatedAt, eventTS)
			}
		case "run.heartbeat":
			diagnostic.HeartbeatCount++
		}
		if strings.HasPrefix(event.EventKind, "tool_call.") && event.EventKind != "tool_call.started" {
			if toolName := eventToolName(event.Payload); toolName != "" {
				diagnostic.LastToolName = stringPtr(toolName)
				diagnostic.LastToolAt = timePtr(eventTS)
			}
		}
	}
	if diagnostic.LastActivityAt.IsZero() {
		diagnostic.LastActivityAt = now
	}
	if leases, ok := deps.Store.(runLeaseReader); ok {
		lease, found, err := leases.GetRunLease(ctx, run.RunID)
		if err != nil {
			return adminRunDiagnostic{}, err
		}
		if found {
			diagnostic.Lease = &lease
			remaining := lease.LeaseExpiresAt.Sub(now).Seconds()
			if remaining < 0 {
				remaining = 0
			}
			diagnostic.LeaseSecondsRemaining = &remaining
			diagnostic.LeaseActive = lease.LeaseExpiresAt.After(now)
			diagnostic.LeaseExpired = !diagnostic.LeaseActive
			if !lease.UpdatedAt.IsZero() {
				age := now.Sub(lease.UpdatedAt).Seconds()
				if age < 0 {
					age = 0
				}
				diagnostic.LeaseLastRenewedAgeSeconds = &age
			}
		}
	}
	age := now.Sub(diagnostic.LastActivityAt)
	if age < 0 {
		age = 0
	}
	diagnostic.LastActivityAgeSeconds = age.Seconds()
	if isAdminWatchableRunStatus(run.Status) {
		if diagnostic.LeaseExpired && diagnostic.Lease != nil {
			diagnostic.Stale = true
			diagnostic.StaleReason = fmt.Sprintf(
				"Run lease expired %s ago for worker %s.",
				formatAdminDuration(now.Sub(diagnostic.Lease.LeaseExpiresAt)),
				diagnostic.Lease.WorkerID,
			)
		} else if age >= adminStaleRunThreshold {
			diagnostic.Stale = true
			if diagnostic.LastEventKind != nil {
				diagnostic.StaleReason = fmt.Sprintf("No worker event for %s; latest event was %s.", formatAdminDuration(age), *diagnostic.LastEventKind)
			} else {
				diagnostic.StaleReason = fmt.Sprintf("No worker event for %s.", formatAdminDuration(age))
			}
		}
	}
	return diagnostic, nil
}

func leaseTimeFields(lease *domain.RunLeaseRecord) (workerID *string, expiresAt *string, lastRenewedAt *string) {
	if lease == nil {
		return nil, nil, nil
	}
	workerID = stringPtr(lease.WorkerID)
	if !lease.LeaseExpiresAt.IsZero() {
		formatted := lease.LeaseExpiresAt.UTC().Format(time.RFC3339Nano)
		expiresAt = &formatted
	}
	if !lease.UpdatedAt.IsZero() {
		formatted := lease.UpdatedAt.UTC().Format(time.RFC3339Nano)
		lastRenewedAt = &formatted
	}
	return workerID, expiresAt, lastRenewedAt
}

func isAdminWatchableRunStatus(status domain.RunStatus) bool {
	switch status {
	case domain.RunStatusQueued, domain.RunStatusRunning, domain.RunStatusWaitingForInput, domain.RunStatusWaitingForTask:
		return true
	default:
		return false
	}
}

func formatAdminDuration(duration time.Duration) string {
	if duration < time.Minute {
		return fmt.Sprintf("%.0fs", duration.Seconds())
	}
	if duration < time.Hour {
		return fmt.Sprintf("%.1fm", duration.Minutes())
	}
	return fmt.Sprintf("%.1fh", duration.Hours())
}

func stringPtr(value string) *string {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return &value
}

func int64Ptr(value int64) *int64 {
	return &value
}

func timePtr(value time.Time) *time.Time {
	return &value
}

func runLatencySeconds(start time.Time, event time.Time) *float64 {
	if start.IsZero() || event.IsZero() {
		return nil
	}
	latency := event.Sub(start).Seconds()
	if latency < 0 {
		latency = 0
	}
	return &latency
}

func eventToolName(payload domain.JSONMap) string {
	for _, key := range []string{"tool_name", "name", "tool"} {
		if value := strings.TrimSpace(toString(payload[key])); value != "" {
			return value
		}
	}
	return ""
}

func valueOrEmpty(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func timePtrString(value *time.Time) string {
	if value == nil || value.IsZero() {
		return ""
	}
	return value.UTC().Format(time.RFC3339Nano)
}

func int64PtrValue(value *int64) int64 {
	if value == nil {
		return 0
	}
	return *value
}

func adminUser(users map[string]*adminUserSummary, userID string) *adminUserSummary {
	userID = strings.TrimSpace(userID)
	if userID == "" {
		userID = "local-user"
	}
	if existing := users[userID]; existing != nil {
		return existing
	}
	user := &adminUserSummary{UserID: userID}
	users[userID] = user
	return user
}

func applyAccountToAdminUser(user *adminUserSummary, account domain.UserAccount) {
	user.Email = account.Email
	user.DisplayName = account.DisplayName
	user.Role = account.Role
	user.Status = account.Status
	user.OrgID = account.OrgID
	if !account.CreatedAt.IsZero() {
		user.CreatedAt = account.CreatedAt.UTC().Format(time.RFC3339Nano)
	}
}

func adminUserMatchesQuery(user adminUserSummary, query string) bool {
	return strings.Contains(strings.ToLower(user.UserID), query) ||
		strings.Contains(strings.ToLower(user.Email), query) ||
		strings.Contains(strings.ToLower(user.DisplayName), query) ||
		strings.Contains(strings.ToLower(user.Role), query) ||
		strings.Contains(strings.ToLower(user.Status), query) ||
		strings.Contains(strings.ToLower(user.OrgID), query)
}

func updateLastActivity(user *adminUserSummary, ts time.Time) {
	if ts.IsZero() {
		return
	}
	if user.LastActivityAt != nil {
		parsed, err := time.Parse(time.RFC3339Nano, *user.LastActivityAt)
		if err == nil && !ts.After(parsed) {
			return
		}
	}
	formatted := ts.UTC().Format(time.RFC3339Nano)
	user.LastActivityAt = &formatted
}

func adminRunFromRun(run domain.RunRecord, diagnostic adminRunDiagnostic) adminRunRecord {
	var lastEventAt *string
	if diagnostic.LastEventAt != nil && !diagnostic.LastEventAt.IsZero() {
		formatted := diagnostic.LastEventAt.UTC().Format(time.RFC3339Nano)
		lastEventAt = &formatted
	}
	var staleReason *string
	if strings.TrimSpace(diagnostic.StaleReason) != "" {
		reason := diagnostic.StaleReason
		staleReason = &reason
	}
	var lastToolAt *string
	if diagnostic.LastToolAt != nil && !diagnostic.LastToolAt.IsZero() {
		formatted := diagnostic.LastToolAt.UTC().Format(time.RFC3339Nano)
		lastToolAt = &formatted
	}
	leaseWorkerID, leaseExpiresAt, leaseLastRenewedAt := leaseTimeFields(diagnostic.Lease)
	ageSeconds := diagnostic.LastActivityAgeSeconds
	return adminRunRecord{
		RunID:                      run.RunID,
		UserID:                     run.UserID,
		ConversationID:             run.ThreadID,
		Goal:                       run.Goal,
		Status:                     string(run.Status),
		CreatedAt:                  run.CreatedAt.UTC().Format(time.RFC3339Nano),
		UpdatedAt:                  run.UpdatedAt.UTC().Format(time.RFC3339Nano),
		Error:                      run.Error,
		DurationSeconds:            runDurationSeconds(run),
		ToolNames:                  metadataStringSlice(run.Metadata["selected_tool_names"]),
		LastEventKind:              diagnostic.LastEventKind,
		LastEventAt:                lastEventAt,
		LastEventSequence:          diagnostic.LastEventSequence,
		LastActivityAgeSeconds:     &ageSeconds,
		EventCount:                 diagnostic.EventCount,
		MessageDeltaCount:          diagnostic.MessageDeltaCount,
		ToolCallCount:              diagnostic.ToolCallCount,
		ArtifactCount:              diagnostic.ArtifactCount,
		HeartbeatCount:             diagnostic.HeartbeatCount,
		LastToolName:               diagnostic.LastToolName,
		LastToolAt:                 lastToolAt,
		FirstDeltaLatency:          diagnostic.FirstDeltaLatency,
		FirstToolLatency:           diagnostic.FirstToolLatency,
		FirstArtifactLatency:       diagnostic.FirstArtifactLatency,
		LeaseWorkerID:              leaseWorkerID,
		LeaseExpiresAt:             leaseExpiresAt,
		LeaseActive:                diagnostic.LeaseActive,
		LeaseExpired:               diagnostic.LeaseExpired,
		LeaseSecondsRemaining:      diagnostic.LeaseSecondsRemaining,
		LeaseLastRenewedAt:         leaseLastRenewedAt,
		LeaseLastRenewedAgeSeconds: diagnostic.LeaseLastRenewedAgeSeconds,
		Stale:                      diagnostic.Stale,
		StaleReason:                staleReason,
	}
}

func runDurationSeconds(run domain.RunRecord) *float64 {
	if run.StartedAt == nil || run.CompletedAt == nil {
		return nil
	}
	seconds := run.CompletedAt.Sub(*run.StartedAt).Seconds()
	return &seconds
}

func metadataStringSlice(value any) []string {
	switch typed := value.(type) {
	case []string:
		return append([]string(nil), typed...)
	case []any:
		items := make([]string, 0, len(typed))
		for _, item := range typed {
			text := strings.TrimSpace(toString(item))
			if text != "" {
				items = append(items, text)
			}
		}
		return items
	default:
		return nil
	}
}

func (deps ServerDeps) uploadStats() (int, int64) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		return 0, 0
	}
	resources, err := listUploadResources(root)
	if err != nil {
		return 0, 0
	}
	var size int64
	for _, resource := range resources {
		size += resource.SizeBytes
	}
	return len(resources), size
}

func (deps ServerDeps) handleTrainingModels(w http.ResponseWriter, r *http.Request) {
	_ = deps
	writeJSON(w, http.StatusOK, trainingModelsResponse{
		Count: 1,
		Models: []trainingModelRecord{{
			Key:               "rarespot-prairie-yolo",
			Name:              "RareSpot Prairie Detector",
			Framework:         "PyTorch/YOLOv5",
			TaskType:          "object_detection",
			Description:       "Prairie dog and burrow detection through the V2 Deep Agents chat tool path. Training services are not configured in the Go control plane yet.",
			SupportsTraining:  false,
			SupportsFinetune:  false,
			SupportsInference: true,
			Dimensions:        []string{"2d"},
			DefaultConfig: domain.JSONMap{
				"workflow":       "rarespot_ecology",
				"training_state": "not_configured",
			},
		}},
	})
}

func (deps ServerDeps) handlePrairieStatus(w http.ResponseWriter, r *http.Request) {
	_ = deps
	writeJSON(w, http.StatusOK, map[string]any{
		"dataset_name":               "Prairie Active Learning",
		"dataset_id":                 nil,
		"last_sync_at":               nil,
		"next_sync_at":               nil,
		"active_model_version":       "rarespot-prairie-yolo",
		"model_health":               "Watch",
		"reviewed_images":            0,
		"unreviewed_images":          0,
		"class_counts":               map[string]int{},
		"unsupported_class_counts":   map[string]int{},
		"detection_counts":           map[string]int{},
		"latest_metrics":             map[string]any{},
		"benchmark_baseline":         map[string]any{},
		"benchmark_latest_candidate": map[string]any{},
		"last_benchmark_at":          nil,
		"benchmark_ready":            false,
		"canonical_benchmark_ready":  false,
		"promotion_benchmark_ready":  false,
		"retrain_gate":               false,
		"retrain_gate_reasons": []string{
			"Training services are not configured in the Go control plane yet.",
			"Use V2 chat with the RareSpot detector for local inference workflows.",
		},
		"retrain_gate_counts": map[string]int{},
	})
}

func (deps ServerDeps) handleEmptyPrairieRetrainRequests(w http.ResponseWriter, r *http.Request) {
	_ = deps
	writeJSON(w, http.StatusOK, map[string]any{"count": 0, "requests": []any{}})
}

func (deps ServerDeps) handleEmptyTrainingDatasets(w http.ResponseWriter, r *http.Request) {
	_ = deps
	writeJSON(w, http.StatusOK, map[string]any{"count": 0, "datasets": []any{}})
}

func (deps ServerDeps) handleEmptyTrainingDomains(w http.ResponseWriter, r *http.Request) {
	_ = deps
	writeJSON(w, http.StatusOK, map[string]any{"count": 0, "domains": []any{}})
}

func (deps ServerDeps) handleEmptyTrainingLineages(w http.ResponseWriter, r *http.Request) {
	_ = deps
	writeJSON(w, http.StatusOK, map[string]any{"count": 0, "lineages": []any{}})
}

func (deps ServerDeps) handleEmptyTrainingVersions(w http.ResponseWriter, r *http.Request) {
	_ = deps
	writeJSON(w, http.StatusOK, map[string]any{"count": 0, "versions": []any{}})
}

func (deps ServerDeps) handleEmptyTrainingUpdateProposals(w http.ResponseWriter, r *http.Request) {
	_ = deps
	writeJSON(w, http.StatusOK, map[string]any{"count": 0, "proposals": []any{}})
}

func (deps ServerDeps) handleEmptyTrainingMergeRequests(w http.ResponseWriter, r *http.Request) {
	_ = deps
	writeJSON(w, http.StatusOK, map[string]any{"count": 0, "merge_requests": []any{}})
}

func (deps ServerDeps) handleModelHealth(w http.ResponseWriter, r *http.Request) {
	_ = deps
	writeJSON(w, http.StatusOK, map[string]any{"count": 0, "models": []any{}})
}

func (deps ServerDeps) handleNotConfigured(message string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		_ = deps
		writeJSON(w, http.StatusNotImplemented, map[string]any{
			"error":   message,
			"status":  "not_configured",
			"service": "ultra-control-v2",
		})
	}
}

func (deps ServerDeps) handleCancelRun(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req cancelRunRequest
	if r.Body != nil && r.ContentLength != 0 {
		if !decodeJSON(w, r, &req) {
			return
		}
	}
	runID := chi.URLParam(r, "run_id")
	run, err := deps.Runs.CancelRun(r.Context(), runcontrol.CancelRunRequest{
		RunID:    runID,
		Reason:   req.Reason,
		Metadata: domain.JSONMap(req.Metadata),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, run)
}

func (deps ServerDeps) handleAcquireRunLease(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req runLeaseRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if strings.TrimSpace(req.WorkerID) == "" {
		writeError(w, http.StatusBadRequest, errors.New("worker_id is required"))
		return
	}
	lease, err := deps.Runs.AcquireRunLease(r.Context(), runcontrol.AcquireRunLeaseRequest{
		RunID:    chi.URLParam(r, "run_id"),
		WorkerID: req.WorkerID,
		TTL:      leaseTTL(req),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, lease)
}

func (deps ServerDeps) handleRenewRunLease(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req runLeaseRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if strings.TrimSpace(req.LeaseToken) == "" {
		writeError(w, http.StatusBadRequest, errors.New("lease_token is required"))
		return
	}
	lease, err := deps.Runs.RenewRunLease(r.Context(), runcontrol.RenewRunLeaseRequest{
		RunID:      chi.URLParam(r, "run_id"),
		LeaseToken: req.LeaseToken,
		TTL:        leaseTTL(req),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, lease)
}

func (deps ServerDeps) handleReleaseRunLease(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req runLeaseRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if strings.TrimSpace(req.LeaseToken) == "" {
		writeError(w, http.StatusBadRequest, errors.New("lease_token is required"))
		return
	}
	if err := deps.Runs.ReleaseRunLease(r.Context(), runcontrol.ReleaseRunLeaseRequest{
		RunID:      chi.URLParam(r, "run_id"),
		LeaseToken: req.LeaseToken,
	}); err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"released": true})
}

func (deps ServerDeps) handleListRunEvents(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	runID := chi.URLParam(r, "run_id")
	limit := clampLimit(parseLimit(r, 500), runEventMaxPageLimit)
	afterSequence, hasAfterSequence := parseAfterSequence(r)
	if r.URL.Query().Get("stream") == "true" {
		deps.streamRunEvents(w, r, runID, afterSequence, hasAfterSequence, limit)
		return
	}
	var events []domain.RunEventRecord
	var err error
	if hasAfterSequence {
		events, err = deps.Store.ListRunEventsAfter(r.Context(), runID, afterSequence, limit)
	} else {
		events, err = deps.Store.ListRunEvents(r.Context(), runID, limit)
	}
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, runEventsResponse{RunID: runID, Count: len(events), Events: events})
}

func (deps ServerDeps) streamRunEvents(w http.ResponseWriter, r *http.Request, runID string, afterSequence int64, hasAfterSequence bool, limit int) {
	if deps.Bus == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "run event stream is not configured"})
		return
	}

	events, unsubscribe := deps.Bus.SubscribeRunEvents(r.Context(), runID)
	defer unsubscribe()

	replay := []domain.RunEventRecord{}
	if hasAfterSequence {
		cursor := afterSequence
		for {
			page, err := deps.Store.ListRunEventsAfter(r.Context(), runID, cursor, limit)
			if err != nil {
				writeStoreError(w, err)
				return
			}
			nextCursor := cursor
			for _, event := range page {
				replay = append(replay, event)
				if event.Sequence > nextCursor {
					nextCursor = event.Sequence
				}
			}
			if len(page) < limit || nextCursor <= cursor {
				break
			}
			cursor = nextCursor
		}
	} else {
		var err error
		replay, err = deps.Store.ListRunEvents(r.Context(), runID, limit)
		if err != nil {
			writeStoreError(w, err)
			return
		}
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	sent := map[string]struct{}{}
	cursor := afterSequence
	deliver := func(event domain.RunEventRecord) bool {
		if event.RunID != runID {
			return true
		}
		key := runEventDeliveryKey(event)
		if _, ok := sent[key]; ok {
			if event.Sequence > cursor {
				cursor = event.Sequence
			}
			return true
		}
		if err := writeSSE(w, "run_event", event); err != nil {
			return false
		}
		sent[key] = struct{}{}
		if event.Sequence > cursor {
			cursor = event.Sequence
		}
		return true
	}
	catchUpFromStore := func() bool {
		for {
			pageStartCursor := cursor
			page, err := deps.Store.ListRunEventsAfter(r.Context(), runID, pageStartCursor, limit)
			if err != nil {
				_ = writeSSE(w, "error", map[string]string{"error": err.Error()})
				return false
			}
			nextCursor := pageStartCursor
			for _, event := range page {
				if !deliver(event) {
					return false
				}
				if event.Sequence > nextCursor {
					nextCursor = event.Sequence
				}
			}
			if len(page) < limit || nextCursor <= pageStartCursor {
				return true
			}
			cursor = nextCursor
		}
	}
	for _, event := range replay {
		if !deliver(event) {
			return
		}
	}

	heartbeat := time.NewTicker(runEventStreamHeartbeatEvery)
	defer heartbeat.Stop()
	catchup := time.NewTicker(runEventStreamCatchupEvery)
	defer catchup.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case event, ok := <-events:
			if !ok {
				return
			}
			if !deliver(event) {
				return
			}
		case <-catchup.C:
			if !catchUpFromStore() {
				return
			}
		case <-heartbeat.C:
			if err := writeSSE(w, "heartbeat", map[string]string{"status": "ok"}); err != nil {
				return
			}
		}
	}
}

func runEventDeliveryKey(event domain.RunEventRecord) string {
	if event.EventID != "" {
		return "event_id:" + event.EventID
	}
	if event.Sequence > 0 {
		return fmt.Sprintf("sequence:%s:%d", event.RunID, event.Sequence)
	}
	return fmt.Sprintf("fallback:%s:%s:%s:%s", event.RunID, event.EventKind, event.Message, event.TS.Format(time.RFC3339Nano))
}

func (deps ServerDeps) handleListRunArtifacts(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	runID := chi.URLParam(r, "run_id")
	artifacts, err := deps.Store.ListRunArtifacts(r.Context(), runID, parseLimit(r, 500))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, runArtifactsResponse{RunID: runID, Count: len(artifacts), Artifacts: artifacts})
}

func (deps ServerDeps) handleGetArtifact(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	artifact, err := deps.Store.GetArtifact(r.Context(), chi.URLParam(r, "artifact_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, artifactResponse{Artifact: artifact})
}

func (deps ServerDeps) handleDownloadRunArtifactByPath(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	if strings.TrimSpace(deps.ArtifactRoot) == "" {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "artifact root is not configured"})
		return
	}
	runID := chi.URLParam(r, "run_id")
	path := strings.TrimSpace(r.URL.Query().Get("path"))
	if path == "" {
		writeError(w, http.StatusBadRequest, errors.New("path query parameter is required"))
		return
	}
	artifacts, err := deps.Store.ListRunArtifacts(r.Context(), runID, 5000)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	for _, artifact := range artifacts {
		if artifactPathMatches(artifact, path) {
			deps.serveArtifactRecord(w, r, artifact)
			return
		}
	}
	writeError(w, http.StatusNotFound, errors.New("artifact path was not found for run"))
}

func (deps ServerDeps) handleDownloadArtifact(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	if strings.TrimSpace(deps.ArtifactRoot) == "" {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "artifact root is not configured"})
		return
	}
	artifact, err := deps.Store.GetArtifact(r.Context(), chi.URLParam(r, "artifact_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	deps.serveArtifactRecord(w, r, artifact)
}

func (deps ServerDeps) serveArtifactRecord(w http.ResponseWriter, r *http.Request, artifact domain.ArtifactRecord) {
	path, err := resolveArtifactDownloadPath(deps.ArtifactRoot, artifact)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if _, err := os.Stat(path); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			writeError(w, http.StatusNotFound, err)
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if artifact.MimeType != "" {
		w.Header().Set("Content-Type", artifact.MimeType)
	}
	http.ServeFile(w, r, path)
}

func artifactPathMatches(artifact domain.ArtifactRecord, raw string) bool {
	clean := strings.TrimSpace(raw)
	if clean == "" {
		return false
	}
	candidates := []string{
		artifact.ArtifactID,
		artifact.Path,
		artifact.SourcePath,
		artifact.PreviewPath,
		fileStoragePath(artifact.StorageURI),
	}
	for _, candidate := range candidates {
		candidate = strings.TrimSpace(candidate)
		if candidate == "" {
			continue
		}
		if candidate == clean || filepath.Clean(candidate) == filepath.Clean(clean) {
			return true
		}
	}
	return false
}

func resolveArtifactDownloadPath(artifactRoot string, artifact domain.ArtifactRecord) (string, error) {
	root, err := filepath.Abs(filepath.Clean(artifactRoot))
	if err != nil {
		return "", err
	}
	candidates := []string{}
	if storagePath := fileStoragePath(artifact.StorageURI); storagePath != "" {
		candidates = append(candidates, storagePath)
	}
	if artifact.SourcePath != "" {
		candidates = append(candidates, artifact.SourcePath)
	}
	if artifact.Path != "" {
		candidates = append(candidates, artifact.Path)
	}
	if len(candidates) == 0 {
		return "", errors.New("artifact does not reference a downloadable file")
	}
	var unsafeErr error
	for _, candidate := range candidates {
		resolved, err := resolveArtifactPathCandidate(root, artifact.RunID, candidate)
		if err == nil {
			return resolved, nil
		}
		if errors.Is(err, errUnsafeArtifactPath) {
			unsafeErr = err
		}
	}
	if unsafeErr != nil {
		return "", unsafeErr
	}
	return "", errors.New("artifact path could not be resolved under artifact root")
}

var errUnsafeArtifactPath = errors.New("artifact path escapes artifact root")

func fileStoragePath(storageURI string) string {
	raw := strings.TrimSpace(storageURI)
	if raw == "" || !strings.HasPrefix(raw, "file://") {
		return ""
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return parsed.Path
}

func resolveArtifactPathCandidate(root string, runID string, candidate string) (string, error) {
	raw := strings.TrimSpace(candidate)
	if raw == "" {
		return "", errors.New("empty artifact path")
	}
	var resolved string
	if filepath.IsAbs(raw) {
		resolved = filepath.Clean(raw)
	} else {
		clean := filepath.Clean(raw)
		if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return "", errUnsafeArtifactPath
		}
		resolved = filepath.Join(root, runID, clean)
	}
	if !pathIsUnderRoot(root, resolved) {
		return "", errUnsafeArtifactPath
	}
	return resolved, nil
}

func pathIsUnderRoot(root string, candidate string) bool {
	rel, err := filepath.Rel(root, candidate)
	if err != nil {
		return false
	}
	return rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)))
}

func (deps ServerDeps) resolvedUploadRoot() (string, error) {
	root := strings.TrimSpace(deps.UploadRoot)
	if root == "" {
		root = "data/uploads"
	}
	return filepath.Abs(filepath.Clean(root))
}

func saveUploadedFile(root string, header *multipart.FileHeader, principal requestPrincipal) (uploadedFileRecord, error) {
	source, err := header.Open()
	if err != nil {
		return uploadedFileRecord{}, err
	}
	defer source.Close()

	fileID := domain.NewID("file")
	originalName := safeOriginalFilename(header.Filename)
	target := filepath.Join(root, fileID+"__"+originalName)
	if !pathIsUnderRoot(root, target) {
		return uploadedFileRecord{}, errUnsafeArtifactPath
	}
	destination, err := os.Create(target)
	if err != nil {
		return uploadedFileRecord{}, err
	}
	hasher := sha256.New()
	size, copyErr := io.Copy(io.MultiWriter(destination, hasher), source)
	closeErr := destination.Close()
	if copyErr != nil {
		_ = os.Remove(target)
		return uploadedFileRecord{}, copyErr
	}
	if closeErr != nil {
		_ = os.Remove(target)
		return uploadedFileRecord{}, closeErr
	}
	info, err := os.Stat(target)
	if err != nil {
		return uploadedFileRecord{}, err
	}
	if info.Size() > 0 {
		size = info.Size()
	}
	if err := writeUploadMetadata(root, fileID, principal); err != nil {
		_ = os.Remove(target)
		return uploadedFileRecord{}, err
	}
	return uploadedFileRecord{
		FileID:       fileID,
		OriginalName: originalName,
		ContentType:  contentTypeForUpload(originalName, header.Header.Get("Content-Type")),
		SizeBytes:    size,
		SHA256:       hex.EncodeToString(hasher.Sum(nil)),
		CreatedAt:    info.ModTime().UTC().Format(time.RFC3339Nano),
		PreviewURL:   "/v2/uploads/" + url.PathEscape(fileID) + "/preview",
		Principal:    principal.record(),
	}, nil
}

func uploadMetadataPath(root string, fileID string) string {
	return filepath.Join(root, ".meta", fileID+".json")
}

func writeUploadMetadata(root string, fileID string, principal requestPrincipal) error {
	if !safeUploadID(fileID) {
		return errors.New("unsafe upload metadata id")
	}
	metaDir := filepath.Join(root, ".meta")
	if !pathIsUnderRoot(root, metaDir) {
		return errUnsafeArtifactPath
	}
	if err := os.MkdirAll(metaDir, 0o755); err != nil {
		return err
	}
	path := uploadMetadataPath(root, fileID)
	if !pathIsUnderRoot(root, path) {
		return errUnsafeArtifactPath
	}
	payload := struct {
		Principal principalRecord `json:"principal"`
	}{
		Principal: principal.record(),
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func readUploadPrincipal(root string, fileID string) principalRecord {
	if !safeUploadID(fileID) {
		return principalRecord{}
	}
	path := uploadMetadataPath(root, fileID)
	if !pathIsUnderRoot(root, path) {
		return principalRecord{}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return principalRecord{}
	}
	var payload struct {
		Principal principalRecord `json:"principal"`
	}
	if err := json.Unmarshal(data, &payload); err != nil {
		return principalRecord{}
	}
	return payload.Principal
}

func listUploadResources(root string) ([]resourceRecord, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []resourceRecord{}, nil
		}
		return nil, err
	}
	resources := make([]resourceRecord, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		path := filepath.Join(root, entry.Name())
		record, err := uploadResourceFromPath(root, path)
		if err != nil {
			continue
		}
		resources = append(resources, record)
	}
	sort.Slice(resources, func(i, j int) bool {
		if resources[i].CreatedAt == resources[j].CreatedAt {
			return resources[i].FileID < resources[j].FileID
		}
		return resources[i].CreatedAt > resources[j].CreatedAt
	})
	return resources, nil
}

func findUploadResource(root string, fileID string) (resourceRecord, string, error) {
	fileID = strings.TrimSpace(fileID)
	if !safeUploadID(fileID) {
		return resourceRecord{}, "", store.ErrNotFound
	}
	patterns := []string{
		filepath.Join(root, fileID+"__*"),
		filepath.Join(root, fileID),
		filepath.Join(root, fileID+".*"),
	}
	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			continue
		}
		sort.Strings(matches)
		for _, path := range matches {
			resolved := filepath.Clean(path)
			if !pathIsUnderRoot(root, resolved) {
				continue
			}
			record, err := uploadResourceFromPath(root, resolved)
			if err != nil {
				continue
			}
			if record.FileID == fileID {
				return record, resolved, nil
			}
		}
	}
	return resourceRecord{}, "", store.ErrNotFound
}

func uploadResourceFromPath(root string, path string) (resourceRecord, error) {
	resolved := filepath.Clean(path)
	if !pathIsUnderRoot(root, resolved) {
		return resourceRecord{}, errUnsafeArtifactPath
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return resourceRecord{}, err
	}
	if info.IsDir() {
		return resourceRecord{}, errors.New("upload resource is a directory")
	}
	fileID, originalName := uploadNameParts(filepath.Base(resolved))
	if !safeUploadID(fileID) {
		return resourceRecord{}, errors.New("unsafe upload id")
	}
	sha, err := sha256File(resolved)
	if err != nil {
		return resourceRecord{}, err
	}
	contentType := contentTypeForUpload(originalName, "")
	previewURL := "/v2/uploads/" + url.PathEscape(fileID) + "/preview"
	return resourceRecord{
		FileID:        fileID,
		OriginalName:  originalName,
		ContentType:   contentType,
		SizeBytes:     info.Size(),
		SHA256:        sha,
		CreatedAt:     info.ModTime().UTC().Format(time.RFC3339Nano),
		SourceType:    "upload",
		ResourceKind:  resourceKindForContent(originalName, contentType),
		HasThumbnail:  strings.HasPrefix(contentType, "image/"),
		ThumbnailURL:  previewURL,
		PreviewURL:    previewURL,
		CacheReady:    true,
		StagedLocally: true,
		Principal:     readUploadPrincipal(root, fileID),
	}, nil
}

func uploadImageDimensions(path string) (int, int, []string) {
	file, err := os.Open(path)
	if err != nil {
		return 1, 1, []string{"image metadata could not be opened"}
	}
	defer func() {
		_ = file.Close()
	}()
	config, _, err := image.DecodeConfig(file)
	if err != nil {
		return 1, 1, []string{"image dimensions could not be decoded"}
	}
	width := config.Width
	if width < 1 {
		width = 1
	}
	height := config.Height
	if height < 1 {
		height = 1
	}
	return width, height, []string{}
}

func uploadChannelCount(contentType string) int {
	normalized := strings.ToLower(strings.TrimSpace(contentType))
	switch normalized {
	case "image/gif", "image/jpeg", "image/png", "image/webp", "image/bmp":
		return 3
	default:
		if strings.HasPrefix(normalized, "image/") {
			return 3
		}
		return 1
	}
}

func uploadNameParts(base string) (string, string) {
	if fileID, originalName, ok := strings.Cut(base, "__"); ok {
		return fileID, originalName
	}
	ext := filepath.Ext(base)
	return strings.TrimSuffix(base, ext), base
}

func sha256File(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func safeOriginalFilename(value string) string {
	name := filepath.Base(strings.TrimSpace(value))
	if name == "." || name == "/" || name == "" {
		name = "upload.bin"
	}
	var builder strings.Builder
	for _, char := range name {
		switch {
		case char >= 'a' && char <= 'z':
			builder.WriteRune(char)
		case char >= 'A' && char <= 'Z':
			builder.WriteRune(char)
		case char >= '0' && char <= '9':
			builder.WriteRune(char)
		case char == '.' || char == '_' || char == '-':
			builder.WriteRune(char)
		default:
			builder.WriteRune('_')
		}
	}
	cleaned := strings.Trim(builder.String(), ".")
	if cleaned == "" {
		return "upload.bin"
	}
	return cleaned
}

func safeUploadID(value string) bool {
	if value == "" {
		return false
	}
	for _, char := range value {
		switch {
		case char >= 'a' && char <= 'z':
		case char >= 'A' && char <= 'Z':
		case char >= '0' && char <= '9':
		case char == '_' || char == '-' || char == '.' || char == ':':
		default:
			return false
		}
	}
	return true
}

func contentTypeForUpload(originalName string, hint string) string {
	extensionType := mime.TypeByExtension(strings.ToLower(filepath.Ext(originalName)))
	hint = strings.TrimSpace(hint)
	if hint == "" || hint == "application/octet-stream" {
		if extensionType != "" {
			return extensionType
		}
	}
	if hint != "" {
		return hint
	}
	return "application/octet-stream"
}

func resourceKindForContent(originalName string, contentType string) string {
	switch {
	case strings.HasPrefix(contentType, "image/"):
		return "image"
	case strings.HasPrefix(contentType, "video/"):
		return "video"
	case strings.Contains(contentType, "csv") || strings.EqualFold(filepath.Ext(originalName), ".csv"):
		return "table"
	default:
		return "file"
	}
}

func (deps ServerDeps) ready(w http.ResponseWriter) bool {
	if deps.Runs == nil || deps.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "run control service is not configured"})
		return false
	}
	return true
}

func decodeJSON(w http.ResponseWriter, r *http.Request, target any) bool {
	defer r.Body.Close()
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(target); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return false
	}
	return true
}

func parseLimit(r *http.Request, fallback int) int {
	raw := r.URL.Query().Get("limit")
	if raw == "" {
		return fallback
	}
	limit, err := strconv.Atoi(raw)
	if err != nil || limit < 1 {
		return fallback
	}
	return limit
}

func clampLimit(limit, max int) int {
	if max > 0 && limit > max {
		return max
	}
	return limit
}

func leaseTTL(req runLeaseRequest) time.Duration {
	seconds := req.TTLSeconds
	if seconds <= 0 {
		seconds = req.LeaseTTLSeconds
	}
	if seconds <= 0 {
		return time.Minute
	}
	return time.Duration(seconds * float64(time.Second))
}

func parseLimitParam(r *http.Request, name string, fallback int) int {
	raw := strings.TrimSpace(r.URL.Query().Get(name))
	if raw == "" {
		return fallback
	}
	limit, err := strconv.Atoi(raw)
	if err != nil || limit < 1 {
		return fallback
	}
	return limit
}

func parseOffset(r *http.Request) int {
	raw := r.URL.Query().Get("offset")
	if raw == "" {
		return 0
	}
	offset, err := strconv.Atoi(raw)
	if err != nil || offset < 0 {
		return 0
	}
	return offset
}

func parseOptionalTime(raw string) (time.Time, error) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return time.Time{}, nil
	}
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return time.Time{}, err
	}
	return parsed.UTC(), nil
}

func parseAfterSequence(r *http.Request) (int64, bool) {
	raw := strings.TrimSpace(r.URL.Query().Get("after_sequence"))
	if raw == "" {
		return 0, false
	}
	sequence, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || sequence < 0 {
		return 0, false
	}
	return sequence, true
}

func take[T any](values []T, limit int) []T {
	if limit <= 0 || len(values) <= limit {
		return values
	}
	return values[:limit]
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}

func toString(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case fmt.Stringer:
		return typed.String()
	default:
		return fmt.Sprint(value)
	}
}

func writeStoreError(w http.ResponseWriter, err error) {
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, err)
		return
	}
	if errors.Is(err, store.ErrConflict) {
		writeError(w, http.StatusConflict, err)
		return
	}
	writeError(w, http.StatusInternalServerError, err)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
