package httpapi

import (
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/go-chi/chi/v5"
	"golang.org/x/image/tiff"
	"golang.org/x/sync/singleflight"
)

type ServerDeps struct {
	Version           string
	Runs              *runcontrol.Service
	Store             runcontrol.Store
	Bus               runEventSource
	ArtifactRoot      string
	UploadRoot        string
	ImageServiceURL   string
	NgffServiceURL    string
	DevAdminEnabled   bool
	Runtime           RuntimeSummary
	QueueDiagnostics  eventbus.QueueDiagnosticsProvider
	DataAgentJobs     eventbus.DataAgentJobPublisher
	TrainingJobs      eventbus.TrainingJobPublisher
	Bisque            *BisqueService
	BisqueCredentials *BisqueCredentialStore
	WorkOS            *WorkOSAuth
	// WorkerToken authenticates trusted workers (Deep Agents, RareSpot) on the
	// run-status, run-events, run-lease, and worker-heartbeat endpoints. Empty
	// disables worker-token auth.
	WorkerToken string
	// ModelPrices maps a (normalized) model id to its per-million-token price so
	// the admin metrics cost panel can turn the token ledger into currency. Set
	// by NewRouter from ULTRA_CONTROL_MODEL_PRICES; empty reports tokens only.
	ModelPrices map[string]ModelPrice
	// DatabaseDiagnostics reports read-only Postgres pool and pg_stat_statements
	// health for the admin overview. Nil is expected for the in-memory store.
	DatabaseDiagnostics DatabaseDiagnosticsProvider
	// adminSnapshots collapses concurrent admin snapshot computations into
	// one; set by NewRouter.
	adminSnapshots *singleflight.Group
	// imageCache serves repeatable image reads (tile/atlas/thumbnail) from a bounded
	// LRU so pan/zoom + 3D re-loads skip the engine. Set by NewRouter from
	// ULTRA_CONTROL_IMAGE_CACHE_BYTES; nil disables (plain streaming proxy).
	imageCache *imageResponseCache
	// sliceCache is a separate, smaller LRU for /slice responses so a z-scrub burst
	// of one-shot slices can't evict the viewer's tile/atlas working set.
	sliceCache *imageResponseCache
	// captioner lazily generates + disk-caches academic-style captions for run-output
	// figures via a grounded VLM. nil/disabled = a graceful no-op (no captions).
	captioner *imageCaptioner
}

type workerAuthState int

const (
	workerAuthAbsent workerAuthState = iota
	workerAuthValid
	workerAuthInvalid
)

const maxJSONBodyBytes int64 = 16 << 20

var (
	workerRunPathPattern         = regexp.MustCompile(`^/v[12]/runs/[^/]+$`)
	workerRunEventsPathPattern   = regexp.MustCompile(`^/v[12]/runs/[^/]+/events$`)
	workerLeasePathPattern       = regexp.MustCompile(`^/v[12]/runs/[^/]+/lease$`)
	workerRunUserProfilePattern  = regexp.MustCompile(`^/v[12]/runs/[^/]+/user-profile$`)
	workerRunSteerListPattern    = regexp.MustCompile(`^/v[12]/runs/[^/]+/steer$`)
	workerRunSteerBarrierPattern = regexp.MustCompile(`^/v[12]/runs/[^/]+/steer/barrier$`)
	workerRunSteerAckPattern     = regexp.MustCompile(`^/v[12]/runs/[^/]+/steer/[^/]+/ack$`)
	workerEpisodicSearchPattern  = regexp.MustCompile(`^/v[12]/runs/[^/]+/episodic-search$`)
	workerResourceSearchPattern  = regexp.MustCompile(`^/v[12]/runs/[^/]+/resource-search$`)
	workerResourceResolvePattern = regexp.MustCompile(`^/v[12]/runs/[^/]+/resource-resolve$`)
	workerDataAgentJobPattern    = regexp.MustCompile(`^/v[12]/data-agent/jobs/[^/]+$`)
	workerDataAgentLeasePattern  = regexp.MustCompile(`^/v[12]/data-agent/jobs/[^/]+/lease$`)
	workerDataAgentStatusPattern = regexp.MustCompile(`^/v[12]/data-agent/jobs/[^/]+/status$`)
	workerDataAgentEventsPattern = regexp.MustCompile(`^/v[12]/data-agent/jobs/[^/]+/events$`)
	workerDataAgentOutputPattern = regexp.MustCompile(`^/v[12]/data-agent/jobs/[^/]+/outputs$`)
)

func workerTokenFromRequest(r *http.Request) string {
	if token := strings.TrimSpace(r.Header.Get("X-Ultra-Worker-Token")); token != "" {
		return token
	}
	authorization := strings.TrimSpace(r.Header.Get("Authorization"))
	const bearerPrefix = "bearer "
	if len(authorization) > len(bearerPrefix) && strings.EqualFold(authorization[:len(bearerPrefix)], bearerPrefix) {
		return strings.TrimSpace(authorization[len(bearerPrefix):])
	}
	return ""
}

// workerRequestAuth classifies the worker credential on a request. A presented
// token never falls through to user scoping: it either matches the configured
// worker token or the request is rejected.
func (deps ServerDeps) workerRequestAuth(r *http.Request) workerAuthState {
	token := workerTokenFromRequest(r)
	if token == "" {
		return workerAuthAbsent
	}
	if strings.TrimSpace(deps.WorkerToken) == "" {
		return workerAuthInvalid
	}
	if subtle.ConstantTimeCompare([]byte(token), []byte(deps.WorkerToken)) == 1 {
		return workerAuthValid
	}
	return workerAuthInvalid
}

// isWorkerScopedEndpoint reports whether the request targets one of the
// endpoints workers are allowed to reach with a worker token.
func isWorkerScopedEndpoint(r *http.Request) bool {
	path := r.URL.Path
	switch {
	case r.Method == http.MethodGet && workerRunPathPattern.MatchString(path):
		return true
	case r.Method == http.MethodGet && workerRunEventsPathPattern.MatchString(path):
		return true
	case r.Method == http.MethodGet && workerRunUserProfilePattern.MatchString(path):
		return true
	case r.Method == http.MethodGet && workerRunSteerListPattern.MatchString(path):
		return true
	case r.Method == http.MethodPost && workerRunSteerBarrierPattern.MatchString(path):
		return true
	case r.Method == http.MethodPost && workerRunSteerAckPattern.MatchString(path):
		return true
	case r.Method == http.MethodPost && workerEpisodicSearchPattern.MatchString(path):
		return true
	case r.Method == http.MethodPost && workerResourceSearchPattern.MatchString(path):
		return true
	case r.Method == http.MethodPost && workerResourceResolvePattern.MatchString(path):
		return true
	case workerLeasePathPattern.MatchString(path):
		return true
	case isWorkerDataAgentEndpoint(r):
		return true
	case r.Method == http.MethodPost && (path == "/v1/workers/heartbeat" || path == "/v2/workers/heartbeat"):
		return true
	case r.Method == http.MethodPost && isWorkerBisqueEndpointPath(path):
		// Deep Agents workers proxy BisQue tool calls for a run; the bisque
		// session header is validated against that run's metadata before any
		// linked credentials are used.
		return strings.TrimSpace(r.Header.Get("X-Ultra-Run-Id")) != ""
	}
	return false
}

func isWorkerDataAgentEndpoint(r *http.Request) bool {
	path := r.URL.Path
	switch {
	case r.Method == http.MethodGet && workerDataAgentJobPattern.MatchString(path):
		return true
	case workerDataAgentLeasePattern.MatchString(path):
		return true
	case r.Method == http.MethodPatch && workerDataAgentStatusPattern.MatchString(path):
		return true
	case r.Method == http.MethodPost && workerDataAgentEventsPattern.MatchString(path):
		return true
	case r.Method == http.MethodPost && workerDataAgentOutputPattern.MatchString(path):
		return true
	}
	return false
}

func isWorkerBisqueEndpointPath(path string) bool {
	switch path {
	case "/v2/bisque/search", "/v2/bisque/dataset-members", "/v2/bisque/image-annotations",
		"/v2/bisque/dataset-annotations", "/v2/bisque/module-run", "/v2/bisque/download",
		"/v2/bisque/upload", "/v2/bisque/datasets", "/v2/uploads":
		return true
	default:
		return false
	}
}

// runForWorkerOrUser resolves a run for worker-token requests without user
// scoping, otherwise falls back to the request principal's scope. It writes
// the HTTP error response itself when resolution fails.
func (deps ServerDeps) runForWorkerOrUser(w http.ResponseWriter, r *http.Request, runID string) (domain.RunRecord, bool) {
	switch deps.workerRequestAuth(r) {
	case workerAuthValid:
		run, err := deps.Store.GetRun(r.Context(), runID)
		if err != nil {
			writeStoreError(w, err)
			return domain.RunRecord{}, false
		}
		return run, true
	case workerAuthInvalid:
		writeError(w, http.StatusUnauthorized, errors.New("invalid worker token"))
		return domain.RunRecord{}, false
	default:
		principal := deps.principalFromRequest(r, "")
		run, err := deps.Store.GetRunForUser(r.Context(), runID, principal.UserID)
		if err != nil {
			writeStoreError(w, err)
			return domain.RunRecord{}, false
		}
		return run, true
	}
}

type runEventSource interface {
	SubscribeRunEvents(ctx context.Context, runID string) (<-chan domain.RunEventRecord, func())
}

type accountStore interface {
	CreateUser(context.Context, domain.CreateUserInput) (domain.UserAccount, error)
	GetUserByID(context.Context, string) (domain.UserAccount, bool, error)
	GetUserByEmail(context.Context, string) (domain.UserAccount, bool, error)
	ListUsers(context.Context, int, string) ([]domain.UserAccount, error)
	UpdateUserStatus(context.Context, string, string) (domain.UserAccount, error)
	UpdateUserProfile(context.Context, domain.UpdateUserProfileInput) (domain.UserAccount, error)
}

// runHistorySearchStore is the optional capability backing episodic memory.
type runHistorySearchStore interface {
	SearchRunHistoryForUser(context.Context, string, domain.RunHistorySearchOptions) ([]domain.RunHistoryHit, error)
}

type episodicSearchRequest struct {
	Query     string `json:"query"`
	SinceDays int    `json:"since_days,omitempty"`
	Limit     int    `json:"limit,omitempty"`
}

type episodicSearchResponse struct {
	Hits []domain.RunHistoryHit `json:"hits"`
}

// runResourceHit is one resource surfaced to a Deep Agents run. It is a
// model-safe projection of domain.ResourceRecord: it deliberately omits host
// filesystem paths (StorageURI/StoragePath) so the agent only ever sees the
// resource id, name, and metadata — never where the file lives on disk.
type runResourceHit struct {
	ResourceID   string         `json:"resource_id"`
	OriginalName string         `json:"original_name"`
	ContentType  string         `json:"content_type,omitempty"`
	ResourceKind string         `json:"resource_kind,omitempty"`
	SourceType   string         `json:"source_type,omitempty"`
	SizeBytes    int64          `json:"size_bytes,omitempty"`
	SHA256       string         `json:"sha256,omitempty"`
	ProjectID    string         `json:"project_id,omitempty"`
	Status       string         `json:"status,omitempty"`
	Tags         []string       `json:"tags,omitempty"`
	Metadata     domain.JSONMap `json:"metadata,omitempty"`
	TreeIdentity domain.JSONMap `json:"tree_identity,omitempty"`
	SensorFormat domain.JSONMap `json:"sensor_format,omitempty"`
	CreatedAt    *time.Time     `json:"created_at,omitempty"`
}

func runResourceHitFromRecord(resource domain.ResourceRecord) runResourceHit {
	hit := runResourceHit{
		ResourceID:   resource.ResourceID,
		OriginalName: resource.OriginalName,
		ContentType:  resource.ContentType,
		ResourceKind: resource.ResourceKind,
		SourceType:   resource.SourceType,
		SizeBytes:    resource.SizeBytes,
		SHA256:       resource.SHA256,
		ProjectID:    resource.ProjectID,
		Status:       resource.Status,
		Tags:         resource.Tags,
		Metadata:     projectRunResourceMetadata(resource.Metadata),
		TreeIdentity: projectCatalogTreeIdentity(resource),
		SensorFormat: projectCatalogSensorFormat(resource),
	}
	if !resource.CreatedAt.IsZero() {
		created := resource.CreatedAt
		hit.CreatedAt = &created
	}
	return hit
}

type runResourceSearchRequest struct {
	Query  string   `json:"query"`
	Kind   string   `json:"kind,omitempty"`
	Source string   `json:"source,omitempty"`
	Tags   []string `json:"tags,omitempty"`
	Limit  int      `json:"limit,omitempty"`
}

type runResourceSearchResponse struct {
	Resources  []runResourceHit `json:"resources"`
	TotalCount int              `json:"total_count"`
}

type runResourceResolveRequest struct {
	ResourceIDs []string `json:"resource_ids"`
}

type runResourceResolveResponse struct {
	Resources []runResourceHit `json:"resources"`
	Missing   []string         `json:"missing"`
}

type usageStatsStore interface {
	GetUserTokenUsageStats(context.Context, string) (domain.UserTokenUsageStats, error)
	ListUserTokenUsageDaily(context.Context, string, time.Time) ([]domain.UserTokenUsageDaily, error)
	GetUserLongestRunSeconds(context.Context, string) (int64, error)
}

type localBootstrapAccount struct {
	Username    string
	Password    string
	UserID      string
	DisplayName string
	Role        string
	Status      string
	OrgID       string
	Metadata    domain.JSONMap
}

type organizationStore interface {
	CreateOrganization(context.Context, domain.CreateOrganizationInput) (domain.Organization, error)
	ListOrganizations(context.Context, int, string) ([]domain.Organization, error)
}

type organizationLookupStore interface {
	GetOrganization(context.Context, string) (domain.Organization, bool, error)
}

type runLeaseReader interface {
	GetRunLease(context.Context, string) (domain.RunLeaseRecord, bool, error)
}

type workerHeartbeatStore interface {
	UpsertWorkerHeartbeat(context.Context, domain.UpsertWorkerHeartbeatInput) (domain.WorkerHeartbeatRecord, error)
	ListWorkerHeartbeats(context.Context, int) ([]domain.WorkerHeartbeatRecord, error)
}

type resourceCatalogStore interface {
	UpsertResource(context.Context, domain.UpsertResourceInput) (domain.ResourceRecord, error)
	GetResourceForUser(context.Context, string, string, string) (domain.ResourceRecord, error)
	ListResourcesForUser(context.Context, domain.ResourceListInput) (domain.ResourceListPage, error)
	SoftDeleteResourceForUser(context.Context, string, string, string, time.Time) (domain.ResourceRecord, error)
	RestoreResourceForUser(context.Context, string, string, string, time.Time) (domain.ResourceRecord, error)
	ResourceStorageStats(context.Context) (domain.ResourceStorageStats, error)
}

type resourceOwnerLookupStore interface {
	GetResourceForOwner(context.Context, string, string, string) (domain.ResourceRecord, error)
}

type resourceOwnerBatchLookupStore interface {
	ListResourceIDsForOwner(context.Context, string, string, []string) (map[string]bool, error)
}

type resourceCollectionStore interface {
	CreateResourceCollection(context.Context, domain.CreateResourceCollectionInput) (domain.ResourceCollectionRecord, error)
	GetResourceCollectionForUser(context.Context, string, string, string) (domain.ResourceCollectionRecord, error)
	RenameResourceCollectionForUser(context.Context, domain.RenameResourceCollectionInput) (domain.ResourceCollectionRecord, error)
	ListResourceCollectionsForUser(context.Context, domain.ResourceCollectionListInput) (domain.ResourceCollectionListPage, error)
	SoftDeleteResourceCollectionForUser(context.Context, string, string, string, time.Time) (domain.ResourceCollectionRecord, error)
	RestoreResourceCollectionForUser(context.Context, string, string, string, time.Time) (domain.ResourceCollectionRecord, error)
	AddResourcesToCollection(context.Context, domain.AddResourcesToCollectionInput) (domain.AddResourcesToCollectionResult, error)
	RemoveResourcesFromCollection(context.Context, domain.RemoveResourcesFromCollectionInput) (domain.RemoveResourcesFromCollectionResult, error)
	ListResourcesForCollectionForUser(context.Context, domain.ResourceCollectionResourceListInput) (domain.ResourceListPage, error)
}

type datasetSnapshotStore interface {
	CreateDatasetSnapshot(context.Context, domain.CreateDatasetSnapshotInput) (domain.DatasetSnapshotRecord, []domain.DatasetSnapshotResourceRecord, error)
	GetDatasetSnapshotForUser(context.Context, string, string, string) (domain.DatasetSnapshotRecord, []domain.DatasetSnapshotResourceRecord, error)
	ListDatasetSnapshotsForUser(context.Context, domain.DatasetSnapshotListInput) (domain.DatasetSnapshotListPage, error)
	SoftDeleteDatasetSnapshotForUser(context.Context, string, string, string, time.Time) (domain.DatasetSnapshotRecord, []domain.DatasetSnapshotResourceRecord, error)
	RestoreDatasetSnapshotForUser(context.Context, string, string, string, time.Time) (domain.DatasetSnapshotRecord, []domain.DatasetSnapshotResourceRecord, error)
}

type dataAgentJobStore interface {
	CreateDataAgentJob(context.Context, domain.CreateDataAgentJobInput) (domain.DataAgentJobRecord, error)
	GetDataAgentJobForUser(context.Context, string, string, string) (domain.DataAgentJobRecord, error)
	ListDataAgentJobsForUser(context.Context, domain.DataAgentJobListInput) (domain.DataAgentJobListPage, error)
	UpdateDataAgentJob(context.Context, domain.UpdateDataAgentJobInput) (domain.DataAgentJobRecord, domain.DataAgentJobEventRecord, error)
	ControlDataAgentJob(context.Context, domain.ControlDataAgentJobInput) (domain.DataAgentJobRecord, domain.DataAgentJobEventRecord, error)
	AcquireDataAgentJobLease(context.Context, domain.AcquireDataAgentJobLeaseInput) (domain.DataAgentJobLeaseRecord, domain.DataAgentJobRecord, domain.DataAgentJobEventRecord, error)
	RenewDataAgentJobLease(context.Context, domain.RenewDataAgentJobLeaseInput) (domain.DataAgentJobLeaseRecord, error)
	ReleaseDataAgentJobLease(context.Context, domain.ReleaseDataAgentJobLeaseInput) error
	AppendDataAgentJobEvent(context.Context, domain.AppendDataAgentJobEventInput) (domain.DataAgentJobEventRecord, error)
	ListDataAgentJobEvents(context.Context, string, string, string, int) ([]domain.DataAgentJobEventRecord, error)
}

type resourceCatalogAdminStore interface {
	ListResources(context.Context, int, int) ([]domain.ResourceRecord, error)
}

// resourceQuotaUsageStore returns active-resource usage scoped to one owner/project so a
// quota check is an indexed aggregate instead of loading (and capping) the whole catalog.
type resourceQuotaUsageStore interface {
	ResourceUsageForOwner(context.Context, string) (int, int64, error)
	ResourceUsageForOrg(context.Context, string) (int, int64, error)
	ResourceUsageForProject(context.Context, string) (int, int64, error)
}

type resourceEventStore interface {
	CreateResourceEvent(context.Context, domain.AppendResourceEventInput) (domain.ResourceEventRecord, error)
}

type resourceEventLogStore interface {
	ListResourceEvents(context.Context, string, int) ([]domain.ResourceEventRecord, error)
	ListResourceEventsForUser(context.Context, domain.ResourceEventListInput) (domain.ResourceEventListPage, error)
}

type resourceMetadataPatchStore interface {
	MergeResourceMetadataForUser(context.Context, domain.MergeResourceMetadataInput) (domain.ResourceRecord, error)
}

type resourceRenameStore interface {
	RenameResourceForUser(context.Context, domain.RenameResourceInput) (domain.ResourceRecord, error)
}

type resourceTagStore interface {
	BulkTagResourcesForUser(context.Context, domain.BulkTagResourcesInput) (domain.BulkTagResourcesResult, error)
}

type resourceShareGrantStore interface {
	CreateResourceShareGrant(context.Context, domain.CreateResourceShareGrantInput) (domain.ResourceShareGrantRecord, error)
	ListResourceShareGrantsForResource(context.Context, domain.ListResourceShareGrantsInput) ([]domain.ResourceShareGrantRecord, error)
	RevokeResourceShareGrant(context.Context, domain.RevokeResourceShareGrantInput) (domain.ResourceShareGrantRecord, error)
}

type resourceCollectionShareGrantStore interface {
	CreateResourceCollectionShareGrant(context.Context, domain.CreateResourceCollectionShareGrantInput) (domain.CreateResourceCollectionShareGrantResult, error)
	ListResourceCollectionShareGrantsForCollection(ctx context.Context, collectionID string, ownerUserID string, ownerOrgID string) ([]domain.ResourceCollectionShareGrantRecord, error)
	RevokeResourceCollectionShareGrant(ctx context.Context, collectionID string, grantID string, ownerUserID string, ownerOrgID string, revokedAt time.Time) (domain.ResourceCollectionShareGrantRecord, error)
}

type datasetSnapshotShareGrantStore interface {
	CreateDatasetSnapshotShareGrant(context.Context, domain.CreateDatasetSnapshotShareGrantInput) (domain.DatasetSnapshotShareGrantRecord, error)
	ListDatasetSnapshotShareGrants(context.Context, domain.ListDatasetSnapshotShareGrantsInput) ([]domain.DatasetSnapshotShareGrantRecord, error)
	RevokeDatasetSnapshotShareGrant(context.Context, domain.RevokeDatasetSnapshotShareGrantInput) (domain.DatasetSnapshotShareGrantRecord, error)
}

type datasetSnapshotEventStore interface {
	ListDatasetSnapshotEventsForUser(context.Context, domain.DatasetSnapshotEventListInput) (domain.DatasetSnapshotEventListPage, error)
}

type resourceDedupeStore interface {
	FindActiveResourceByShaForUser(context.Context, string, string, string, int64) (domain.ResourceRecord, error)
}

type uploadSessionStore interface {
	CreateUploadSession(context.Context, domain.CreateUploadSessionInput) (domain.UploadSessionRecord, error)
	GetUploadSessionForUser(context.Context, string, string, string) (domain.UploadSessionRecord, error)
	GetUploadSessionByIdempotencyKeyForUser(context.Context, string, string, string) (domain.UploadSessionRecord, error)
	// ClearUploadSessionIdempotencyKey frees a session's idempotency-key slot (set to
	// NULL/empty) so a re-upload of the same content can take a fresh session. Used to
	// supersede a terminal session whose committed result is no longer live.
	ClearUploadSessionIdempotencyKey(context.Context, string) error
	UpdateUploadSession(context.Context, domain.UpdateUploadSessionInput) (domain.UploadSessionRecord, error)
	UpsertUploadSessionFile(context.Context, domain.UpsertUploadSessionFileInput) (domain.UploadSessionFileRecord, error)
	ListUploadSessionFiles(context.Context, string) ([]domain.UploadSessionFileRecord, error)
	UpsertUploadChunk(context.Context, domain.UpsertUploadChunkInput) (domain.UploadChunkRecord, error)
	ListUploadChunks(context.Context, string, string) ([]domain.UploadChunkRecord, error)
}

type uploadSessionFileBatchStore interface {
	CreateUploadSessionFiles(context.Context, []domain.UpsertUploadSessionFileInput) ([]domain.UploadSessionFileRecord, error)
}

type uploadSessionFileLookupStore interface {
	GetUploadSessionFile(context.Context, string, string) (domain.UploadSessionFileRecord, error)
}

type uploadSessionChunkStore interface {
	ListUploadSessionChunks(context.Context, string) ([]domain.UploadChunkRecord, error)
}

type uploadSessionTotalsStore interface {
	GetUploadSessionTotals(context.Context, string) (domain.UploadSessionTotals, error)
}

type uploadSessionEventStore interface {
	AppendUploadSessionEvent(context.Context, domain.AppendUploadSessionEventInput) (domain.UploadSessionEventRecord, error)
	ListUploadSessionEvents(context.Context, string, int) ([]domain.UploadSessionEventRecord, error)
}

type uploadSessionOperationalMetricsStore interface {
	UploadSessionOperationalMetrics(context.Context) (domain.UploadSessionOperationalMetrics, error)
}

type uploadCatalogMigrationState struct {
	mu   sync.Mutex
	done bool
}

var uploadCatalogMigrations sync.Map

const (
	adminStaleRunThreshold         = 5 * time.Minute
	adminWorkerStaleThreshold      = 3 * time.Minute
	bisqueSessionCookieName        = "ultra_bisque_session"
	runEventMaxPageLimit           = 1000
	runEventStreamHeartbeatEvery   = 15 * time.Second
	runEventStreamCatchupEvery     = time.Second
	uploadSessionMaxParallelFiles  = 4
	uploadSessionMaxParallelChunks = 8 // more in-flight chunks better saturate the link for large files
	uploadSessionMaxFilesPerBatch  = 10_000
	directUploadMaxBodyBytes       = int64(5) << 30
)

// 512KiB buffers cut syscall count ~16x versus 32KiB on the upload hot paths — chunk
// receipt and the multi-GB /complete reassembly each stream the whole file through this.
// Pooled, so memory stays bounded by upload concurrency.
var uploadCopyBufferPool = sync.Pool{
	New: func() any {
		buffer := make([]byte, 512*1024)
		return &buffer
	},
}

func NewRouter(deps ServerDeps) http.Handler {
	if deps.BisqueCredentials == nil {
		deps.BisqueCredentials = NewBisqueCredentialStore()
	}
	if deps.adminSnapshots == nil {
		deps.adminSnapshots = &singleflight.Group{}
	}
	if deps.imageCache == nil {
		deps.imageCache = newImageResponseCacheFromEnv()
	}
	if deps.ModelPrices == nil {
		deps.ModelPrices = loadModelPricesFromEnv()
	}
	if deps.sliceCache == nil {
		deps.sliceCache = newSliceResponseCacheFromEnv()
	}
	if deps.captioner == nil {
		deps.captioner = newImageCaptionerFromEnv(deps.ArtifactRoot)
	}
	if !deps.WorkOS.Enabled() {
		_ = deps.ensureLocalBootstrapAccounts(context.Background())
	}
	r := chi.NewRouter()
	// Outermost middleware: a handler panic becomes a clean 500 for that one request
	// (structured-logged), never a dropped connection or a stderr stack-trace dump.
	r.Use(recoverPanics)
	// Make a duplicate WriteHeader a no-op instead of a superfluous-header log +
	// possible response corruption (an error path reached after the response was
	// already started). Inside recoverPanics so the recovery fallback is guarded too.
	r.Use(guardResponseWriter)
	r.Get("/v1/health", handleHealth)
	r.Get("/v1/config/public", handlePublicConfig(deps))
	r.Get("/v1/auth/session", handleAuthSession(deps))
	r.Post("/v1/auth/guest", handleAuthGuest(deps))
	r.Post("/v1/auth/login", handleAuthLogin(deps))
	r.Post("/v1/auth/logout", handleAuthLogout(deps))
	r.Post("/v1/bisque/unlink", deps.handleBisqueUnlink)
	r.Route("/v2", func(r chi.Router) {
		r.Get("/health", handleHealth)
		r.Get("/config/public", handlePublicConfig(deps))
		r.Get("/auth/session", handleAuthSession(deps))
		r.Post("/auth/request-account", deps.handleAccountRequest)
		r.Post("/auth/guest", handleAuthGuest(deps))
		r.Post("/auth/login", handleAuthLogin(deps))
		r.Post("/auth/logout", handleAuthLogout(deps))
		r.Get("/auth/workos/callback", deps.handleWorkOSCallback)
		r.Group(func(r chi.Router) {
			if deps.WorkOS.Enabled() {
				r.Use(deps.requireWorkOSAccount)
			}
			r.Get("/me", deps.handleGetCurrentUser)
			r.Patch("/me", deps.handleUpdateCurrentUser)
			r.Get("/me/token-usage", deps.handleGetTokenUsage)
			r.Get("/threads", deps.handleListThreads)
			r.Post("/threads", deps.handleCreateThread)
			r.Get("/threads/{thread_id}", deps.handleGetThread)
			r.Put("/threads/{thread_id}", deps.handleUpsertThread)
			r.Delete("/threads/{thread_id}", deps.handleDeleteThread)
			r.Get("/threads/{thread_id}/messages", deps.handleListThreadMessages)
			r.Post("/threads/{thread_id}/runs", deps.handleCreateRun)
			r.Post("/uploads", deps.handleUploadFiles)
			r.Post("/upload-sessions", deps.handleCreateUploadSession)
			r.Get("/upload-sessions/{session_id}", deps.handleGetUploadSession)
			r.Put("/upload-sessions/{session_id}/files/{file_token}/chunks/{chunk_index}", deps.handleUploadSessionChunk)
			r.Post("/upload-sessions/{session_id}/files/{file_token}/complete", deps.handleCompleteUploadSessionFile)
			r.Post("/upload-sessions/{session_id}/finalize-bundle", deps.handleFinalizeUploadBundle)
			r.Post("/upload-sessions/{session_id}/pause", deps.handlePauseUploadSession)
			r.Post("/upload-sessions/{session_id}/resume", deps.handleResumeUploadSession)
			r.Post("/upload-sessions/{session_id}/cancel", deps.handleCancelUploadSession)
			r.Get("/uploads/{file_id}/viewer", deps.handleGetUploadViewerService)
			r.Get("/uploads/{file_id}/preview", deps.handleServeUpload)
			r.Get("/uploads/{file_id}/display", deps.handleServeUpload)
			r.Get("/uploads/{file_id}/slice", deps.handleServeUploadSliceService)
			r.Get("/uploads/{file_id}/scalar-volume", deps.handleGetUploadScalarVolumeService)
			r.Get("/uploads/{file_id}/cifti/carpet", deps.handleGetUploadCiftiCarpet)
			r.Get("/uploads/{file_id}/cifti/connectivity", deps.handleGetUploadCiftiConnectivity)
			r.Get("/uploads/{file_id}/tiles/{axis}/{level}/{tile_x}/{tile_y}", deps.handleServeUploadTiles)
			r.Get("/uploads/{file_id}/atlas", deps.handleServeUploadAtlas)
			r.Get("/uploads/{file_id}/histogram", deps.handleGetUploadHistogramService)
			r.Post("/uploads/{file_id}/derive-pyramid", deps.handleDeriveUploadPyramid)
			r.Get("/uploads/{file_id}/caption", deps.handleGetUploadCaption)
			r.Get("/uploads/{file_id}/hdf5/dataset", deps.handleGetUploadHdf5Dataset)
			r.Get("/uploads/{file_id}/hdf5/preview/slice", deps.handleServeUploadHdf5Slice)
			r.Get("/uploads/{file_id}/hdf5/preview/atlas", deps.handleServeUploadHdf5Atlas)
			r.Get("/uploads/{file_id}/hdf5/preview/scalar-volume", deps.handleGetUploadHdf5ScalarVolume)
			r.Get("/uploads/{file_id}/hdf5/preview/histogram", deps.handleGetUploadHdf5Histogram)
			r.Get("/uploads/{file_id}/hdf5/preview/table", deps.handleGetUploadHdf5Table)
			r.Get("/uploads/{file_id}/hdf5/materials/dashboard", deps.handleGetUploadHdf5MaterialsDashboard)
			r.Post("/uploads/from-bisque", deps.handleImportBisqueResources)
			r.Post("/bisque/search", deps.handleBisqueSearch)
			r.Post("/bisque/dataset-members", deps.handleBisqueDatasetMembers)
			r.Post("/bisque/image-annotations", deps.handleBisqueImageAnnotations)
			r.Post("/bisque/dataset-annotations", deps.handleBisqueDatasetAnnotations)
			r.Post("/bisque/module-run", deps.handleBisqueModuleRun)
			r.Post("/bisque/download", deps.handleImportBisqueResources)
			r.Post("/bisque/upload", deps.handleBisqueUpload)
			r.Post("/bisque/datasets", deps.handleBisqueCreateDataset)
			r.Post("/bisque/push", deps.handleBisquePush)
			r.Post("/bisque/unlink", deps.handleBisqueUnlink)
			r.Get("/resource-events", deps.handleListResourceEventLog)
			r.Get("/resources", deps.handleListResources)
			r.Post("/resources/delete/bulk", deps.handleBulkDeleteResources)
			r.Post("/resources/restore/bulk", deps.handleBulkRestoreResources)
			r.Post("/resources/tags/bulk", deps.handleBulkTagResources)
			r.Post("/resources/shares/bulk", deps.handleCreateResourceShareGrants)
			r.Get("/resources/{file_id}/download", deps.handleDownloadResource)
			r.Get("/resources/{file_id}/text-head", deps.handleResourceTextHead)
			r.Get("/resources/{file_id}/csv/rows", deps.handleResourceCsvRows)
			r.Get("/resources/{file_id}", deps.handleGetResource)
			r.Patch("/resources/{file_id}", deps.handlePatchResource)
			r.Delete("/resources/{file_id}", deps.handleDeleteResource)
			r.Post("/resources/{file_id}/restore", deps.handleRestoreResource)
			r.Get("/resources/{file_id}/events", deps.handleListResourceEvents)
			r.Get("/resources/{file_id}/thumbnail", deps.handleServeResourceThumbnail)
			r.Get("/resources/{file_id}/shares", deps.handleListResourceShareGrants)
			r.Post("/resources/{file_id}/shares", deps.handleCreateResourceShareGrant)
			r.Delete("/resources/{file_id}/shares/{grant_id}", deps.handleRevokeResourceShareGrant)
			r.Get("/share-targets", deps.handleListShareTargets)
			r.Post("/resource-collections", deps.handleCreateResourceCollection)
			r.Get("/resource-collections", deps.handleListResourceCollections)
			r.Patch("/resource-collections/{collection_id}", deps.handlePatchResourceCollection)
			r.Delete("/resource-collections/{collection_id}", deps.handleDeleteResourceCollection)
			r.Post("/resource-collections/{collection_id}/restore", deps.handleRestoreResourceCollection)
			r.Post("/resource-collections/{collection_id}/shares", deps.handleCreateResourceCollectionShareGrants)
			r.Get("/resource-collections/{collection_id}/shares", deps.handleListResourceCollectionShareGrants)
			r.Delete("/resource-collections/{collection_id}/shares/{grant_id}", deps.handleRevokeResourceCollectionShareGrant)
			r.Post("/resource-collections/{collection_id}/resources", deps.handleAddResourcesToCollection)
			r.Get("/resource-collections/{collection_id}/resources", deps.handleListResourceCollectionResources)
			r.Get("/resource-collections/{collection_id}/download", deps.handleDownloadResourceCollection)
			r.Delete("/resource-collections/{collection_id}/resources/{file_id}", deps.handleRemoveResourceFromCollection)
			r.Post("/dataset-snapshots", deps.handleCreateDatasetSnapshot)
			r.Get("/dataset-snapshots", deps.handleListDatasetSnapshots)
			r.Delete("/dataset-snapshots/{snapshot_id}", deps.handleDeleteDatasetSnapshot)
			r.Post("/dataset-snapshots/{snapshot_id}/restore", deps.handleRestoreDatasetSnapshot)
			r.Get("/dataset-snapshots/{snapshot_id}/shares", deps.handleListDatasetSnapshotShareGrants)
			r.Post("/dataset-snapshots/{snapshot_id}/shares", deps.handleCreateDatasetSnapshotShareGrant)
			r.Delete("/dataset-snapshots/{snapshot_id}/shares/{grant_id}", deps.handleRevokeDatasetSnapshotShareGrant)
			r.Get("/dataset-snapshots/{snapshot_id}/events", deps.handleListDatasetSnapshotEvents)
			r.Get("/dataset-snapshots/{snapshot_id}", deps.handleGetDatasetSnapshot)
			r.Post("/data-agent/jobs", deps.handleCreateDataAgentJob)
			r.Get("/data-agent/jobs", deps.handleListDataAgentJobs)
			r.Get("/data-agent/jobs/{job_id}", deps.handleGetDataAgentJob)
			r.Post("/data-agent/jobs/{job_id}/events", deps.handleAppendDataAgentJobEvent)
			r.Patch("/data-agent/jobs/{job_id}/status", deps.handleUpdateDataAgentJobStatus)
			r.Post("/data-agent/jobs/{job_id}/control", deps.handleControlDataAgentJob)
			r.Post("/data-agent/jobs/{job_id}/lease", deps.handleAcquireDataAgentJobLease)
			r.Patch("/data-agent/jobs/{job_id}/lease", deps.handleRenewDataAgentJobLease)
			r.Delete("/data-agent/jobs/{job_id}/lease", deps.handleReleaseDataAgentJobLease)
			// Batch model inference (MegaSeg / RareSpot): submit rides the data-agent backbone;
			// the worker registers produced results via the outputs endpoint. Progress + cancel
			// reuse GET /data-agent/jobs/{id} and POST /data-agent/jobs/{id}/control.
			r.Post("/analysis/batch", deps.handleCreateBatchAnalysisJob)
			r.Post("/data-agent/jobs/{job_id}/outputs", deps.handleRegisterAnalysisOutputs)
			r.Get("/runs", deps.handleListRuns)
			r.Get("/runs/{run_id}", deps.handleGetRun)
			r.Get("/runs/{run_id}/user-profile", deps.handleGetRunUserProfile)
			r.Post("/runs/{run_id}/episodic-search", deps.handleEpisodicSearch)
			r.Post("/runs/{run_id}/resource-search", deps.handleRunResourceSearch)
			r.Post("/runs/{run_id}/resource-resolve", deps.handleRunResourceResolve)
			r.Post("/runs/{run_id}/lease", deps.handleAcquireRunLease)
			r.Patch("/runs/{run_id}/lease", deps.handleRenewRunLease)
			r.Delete("/runs/{run_id}/lease", deps.handleReleaseRunLease)
			r.Post("/runs/{run_id}/cancel", deps.handleCancelRun)
			r.Post("/runs/{run_id}/steer", deps.handleSteerRun)
			r.Get("/runs/{run_id}/steer", deps.handleListRunSteerMessages)
			r.Post("/runs/{run_id}/steer/barrier", deps.handleCloseRunSteerBarrier)
			r.Post("/runs/{run_id}/steer/{steer_id}/ack", deps.handleAckRunSteerMessage)
			r.Get("/runs/{run_id}/events", deps.handleListRunEvents)
			r.Get("/runs/{run_id}/artifacts", deps.handleListRunArtifacts)
			r.Get("/runs/{run_id}/artifacts/download", deps.handleDownloadRunArtifactByPath)
			r.Get("/runs/{run_id}/artifacts/caption", deps.handleRunArtifactCaption)
			r.Get("/artifacts/{artifact_id}", deps.handleGetArtifact)
			r.Get("/artifacts/{artifact_id}/download", deps.handleDownloadArtifact)
			r.Post("/artifacts/{artifact_id}/promote-resource", deps.handlePromoteArtifactResource)
			r.Post("/workers/heartbeat", deps.handleWorkerHeartbeat)
			r.Group(func(r chi.Router) {
				if deps.WorkOS.Enabled() {
					r.Use(deps.requireWorkOSAdmin)
				}
				r.Get("/admin/overview", deps.handleAdminOverview)
				r.Get("/admin/metrics", deps.handleAdminMetrics)
				r.Post("/admin/resources/reconcile", deps.handleAdminReconcileResources)
				r.Get("/admin/orgs", deps.handleAdminOrganizations)
				r.Post("/admin/orgs", deps.handleAdminCreateOrganization)
				r.Get("/admin/users", deps.handleAdminUsers)
				r.Post("/admin/users", deps.handleAdminCreateUser)
				r.Patch("/admin/users/{user_id}/status", deps.handleAdminUpdateUserStatus)
				r.Delete("/admin/users/{user_id}", deps.handleAdminDeleteUser)
				r.Get("/admin/runs", deps.handleAdminRuns)
				r.Get("/admin/issues", deps.handleAdminIssues)
				r.Post("/admin/runs/{run_id}/cancel", deps.handleAdminCancelRun)
				r.Post("/admin/runs/{run_id}/requeue", deps.handleAdminRequeueRun)
				r.Delete("/admin/conversations/{conversation_id}", deps.handleNotConfigured("admin conversation deletion is not configured in the Go control plane yet"))
			})
			r.Get("/training/models", deps.handleTrainingModels)
			r.Get("/training/models/{model_key}/status", deps.handleTrainingModelStatus)
			r.Post("/training/models/{model_key}/sync", deps.handleDispatchTrainingSync)
			r.Post("/training/models/{model_key}/gold-sets", deps.handleCreateTrainingGoldSetDraft)
			r.Post("/training/models/{model_key}/gold-sets/{gold_set_id}/freeze", deps.handleFreezeTrainingGoldSet)
			r.Post("/training/models/{model_key}/benchmark/run", deps.handleDispatchTrainingBenchmark)
			r.Post("/training/models/{model_key}/retrain-request", deps.handleDispatchTrainingRetrain)
			r.Post("/training/models/{model_key}/versions", deps.handleRegisterTrainingModelVersion)
			r.Post("/training/model-versions/{version_id}/reject", deps.handleRejectTrainingModelVersion)
			r.Post("/training/jobs/{job_id}/lease", deps.handleAcquireTrainingJobLease)
			r.Patch("/training/jobs/{job_id}/lease", deps.handleRenewTrainingJobLease)
			r.Delete("/training/jobs/{job_id}/lease", deps.handleReleaseTrainingJobLease)
			r.Post("/training/jobs/{job_id}/events", deps.handleAppendTrainingJobEventHTTP)
			r.Patch("/training/jobs/{job_id}/status", deps.handleUpdateTrainingJobStatusHTTP)
			r.Post("/training/jobs/{job_id}/benchmark-result", deps.handleTrainingBenchmarkResult)
			r.Post("/training/jobs/{job_id}/gold-result", deps.handleTrainingGoldResult)
			r.Post("/training/jobs/{job_id}/status-result", deps.handleTrainingStatusResult)
			r.Get("/training/models/{model_key}/retrain-requests", deps.handleTrainingRetrainRequests)
			r.Get("/training/models/{model_key}/resolve", deps.handleResolveTrainingServingWeights)
			r.Post("/training/models/{model_key}/canary-observations", deps.handleInsertTrainingCanaryObservation)
			r.Get("/training/models/{model_key}/canary-observations", deps.handleListTrainingCanaryObservations)
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
			r.Get("/training/domains", deps.handleTrainingDomains)
			r.Post("/training/domains", deps.handleNotConfigured("training domain creation is not configured in the Go control plane yet"))
			r.Get("/training/domains/{domain_id}/lineages", deps.handleTrainingLineages)
			r.Post("/training/lineages/{lineage_id}/fork", deps.handleNotConfigured("training lineage forks are not configured in the Go control plane yet"))
			r.Get("/training/lineages/{lineage_id}/versions", deps.handleTrainingVersions)
			r.Post("/training/update-proposals/preview", deps.handleNotConfigured("training update proposals are not configured in the Go control plane yet"))
			r.Get("/training/update-proposals", deps.handleEmptyTrainingUpdateProposals)
			r.Post("/training/update-proposals/{proposal_id}/approve", deps.handleNotConfigured("training update proposal decisions are not configured in the Go control plane yet"))
			r.Post("/training/update-proposals/{proposal_id}/reject", deps.handleNotConfigured("training update proposal decisions are not configured in the Go control plane yet"))
			r.Post("/training/model-versions/{version_id}/promote", deps.handlePromoteTrainingModelVersionReal)
			r.Post("/training/model-versions/{version_id}/rollback", deps.handleRollbackTrainingModelVersionReal)
			r.Get("/training/merge-requests", deps.handleEmptyTrainingMergeRequests)
			r.Post("/training/merge-requests", deps.handleNotConfigured("training merge requests are not configured in the Go control plane yet"))
			r.Post("/training/merge-requests/{merge_id}/approve", deps.handleNotConfigured("training merge request decisions are not configured in the Go control plane yet"))
			r.Post("/training/merge-requests/{merge_id}/reject", deps.handleNotConfigured("training merge request decisions are not configured in the Go control plane yet"))
		})
	})
	return r
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status": "ok",
		"ts":     time.Now().UTC().Format(time.RFC3339Nano),
	})
}

func publicBisqueNavLinks(rootURL string) (string, map[string]string) {
	rootURL = strings.TrimSpace(rootURL)
	if rootURL == "" {
		return "", nil
	}

	base := ""
	if parsed, err := url.Parse(rootURL); err == nil && parsed.Scheme != "" && parsed.Host != "" {
		path := strings.TrimRight(parsed.Path, "/")
		clientServiceIndex := strings.Index(strings.ToLower(path), "/client_service")
		switch {
		case clientServiceIndex >= 0:
			parsed.Path = path[:clientServiceIndex+len("/client_service")]
		case path == "" || path == "/":
			parsed.Path = "/client_service"
		default:
			parsed.Path = path + "/client_service"
		}
		parsed.RawQuery = ""
		parsed.Fragment = ""
		base = strings.TrimRight(parsed.String(), "/")
	} else {
		withoutQuery := strings.TrimRight(strings.Split(rootURL, "?")[0], "/")
		clientServiceIndex := strings.Index(strings.ToLower(withoutQuery), "/client_service")
		if clientServiceIndex >= 0 {
			base = withoutQuery[:clientServiceIndex+len("/client_service")]
		} else {
			base = withoutQuery + "/client_service"
		}
	}
	if base == "" {
		return "", nil
	}
	return strings.TrimRight(rootURL, "/"), map[string]string{
		"home":     base + "/",
		"datasets": base + "/browser?resource=/data_service/dataset",
		"images":   base + "/browser?resource=/data_service/image",
		"tables":   base + "/browser?resource=/data_service/table",
	}
}

func handlePublicConfig(deps ServerDeps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		payload := map[string]any{
			"app_name":      "BisQue Ultra",
			"app_version":   deps.Version,
			"admin_enabled": deps.DevAdminEnabled,
			"features": map[string]bool{
				"v2_runs": true,
			},
		}
		if rootURL, links := publicBisqueNavLinks(deps.bisqueRootURL()); rootURL != "" && links != nil {
			payload["bisque_root"] = rootURL
			payload["bisque_browser_url"] = links["home"]
			payload["bisque_urls"] = links
		}
		writeJSON(w, http.StatusOK, payload)
	}
}

func handleAuthSession(deps ServerDeps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if deps.WorkOS.Enabled() {
			session, err := deps.workOSSessionResponseForRequest(w, r)
			if err != nil {
				writeStoreError(w, err)
				return
			}
			if session["authenticated"] == true {
				session["bisque_linked"] = deps.hasLinkedBisqueSession(r.Context(), r)
			}
			writeJSON(w, http.StatusOK, session)
			return
		}
		session := devAuthSessionFromRequest(r, deps.DevAdminEnabled, deps.BisqueCredentials)
		writeJSON(w, http.StatusOK, session)
	}
}

func handleAuthGuest(deps ServerDeps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if deps.WorkOS.Enabled() {
			writeError(w, http.StatusForbidden, errors.New("guest auth is disabled when WorkOS auth is enabled"))
			return
		}
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

func (deps ServerDeps) handleAccountRequest(w http.ResponseWriter, r *http.Request) {
	accounts, ok := deps.Store.(accountStore)
	if !ok {
		writeJSON(w, http.StatusNotImplemented, map[string]any{
			"authenticated": false,
			"status":        "not_configured",
			"service":       "ultra-control-v2",
			"detail":        "account request storage is not configured",
		})
		return
	}
	var payload struct {
		Name        string `json:"name"`
		Email       string `json:"email"`
		Affiliation string `json:"affiliation"`
	}
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		writeError(w, http.StatusBadRequest, errors.New("invalid account request payload"))
		return
	}
	name := strings.TrimSpace(payload.Name)
	email := normalizeAuthEmail(payload.Email)
	affiliation := strings.TrimSpace(payload.Affiliation)
	if name == "" || email == "" || affiliation == "" {
		writeError(w, http.StatusBadRequest, errors.New("name, email, and affiliation are required"))
		return
	}
	user, found, err := accounts.GetUserByEmail(r.Context(), email)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if !found {
		user, err = accounts.CreateUser(r.Context(), domain.CreateUserInput{
			Email:       email,
			DisplayName: name,
			Role:        "researcher",
			Status:      "pending",
			OrgID:       "local-org",
			Metadata: domain.JSONMap{
				"source":      "account_request",
				"affiliation": affiliation,
			},
		})
		if err != nil {
			if errors.Is(err, store.ErrConflict) {
				user, found, err = accounts.GetUserByEmail(r.Context(), email)
				if err != nil {
					writeStoreError(w, err)
					return
				}
			}
			if err != nil {
				writeStoreError(w, err)
				return
			}
		}
	}
	status := normalizeAccountStatus(user.Status)
	if status == "" {
		status = "pending"
	}
	setDevAuthCookie(w, "signed_out")
	writeJSON(w, http.StatusAccepted, map[string]any{
		"authenticated":   false,
		"provider":        deps.authProviderName(),
		"mode":            deps.authProviderName(),
		"account_status":  status,
		"account_email":   email,
		"account_user_id": user.UserID,
		"message":         accountStatusMessage(status),
	})
}

func handleAuthLogin(deps ServerDeps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if deps.WorkOS.Enabled() {
			var payload struct {
				Username string `json:"username"`
				Password string `json:"password"`
			}
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil && !errors.Is(err, io.EOF) {
				writeError(w, http.StatusBadRequest, errors.New("invalid login payload"))
				return
			}
			if strings.TrimSpace(payload.Username) != "" || strings.TrimSpace(payload.Password) != "" {
				deps.handleWorkOSBisqueLink(w, r, payload.Username, payload.Password)
				return
			}
			deps.WorkOS.handleLogin(w, r)
			return
		}
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
		if bootstrap, ok := localBootstrapAccountForCredential(username, payload.Password); ok {
			account := localBootstrapAccountRecord(bootstrap)
			if accounts, storeOK := deps.Store.(accountStore); storeOK {
				ensured, err := ensureLocalBootstrapAccount(r.Context(), accounts, bootstrap)
				if err != nil {
					writeStoreError(w, err)
					return
				}
				if strings.TrimSpace(ensured.UserID) != "" {
					account = ensured
				}
			}
			setDevAuthCookie(w, "bisque:"+bootstrap.Username)
			session := devAuthSession(bootstrap.Username, "bisque", nil, deps.DevAdminEnabled)
			applyLocalAccountToDevSession(session, account)
			session["bisque_linked"] = false
			writeJSON(w, http.StatusOK, session)
			return
		}
		if deps.BisqueCredentials != nil && strings.TrimSpace(payload.Password) != "" {
			if deps.Bisque == nil {
				writeBisqueNotConfigured(w)
				return
			}
			credentials := BisqueCredentials{
				Username: username,
				Password: payload.Password,
			}
			if err := deps.Bisque.VerifyCredentials(r.Context(), credentials); err != nil {
				setDevAuthCookie(w, "signed_out")
				writeBisqueError(w, err)
				return
			}
			account, ok := deps.enforceLocalLoginApproval(w, r, username)
			if !ok {
				return
			}
			principal := principalFromRequest(r, devPrincipalUserID(username, "bisque"))
			sessionID, err := deps.BisqueCredentials.PutLinked(r.Context(), BisqueCredentialLinkInput{
				Credentials:    credentials,
				UserID:         principal.UserID,
				OrgID:          principal.OrgID,
				RootURL:        deps.bisqueRootURL(),
				LastVerifiedAt: domain.Now(),
				Metadata: domain.JSONMap{
					"source": "settings_link_account",
				},
			})
			if err != nil {
				writeError(w, http.StatusInternalServerError, err)
				return
			}
			setDevAuthCookie(w, "bisque_session:"+sessionID)
			session := devAuthSession(username, "bisque", nil, deps.DevAdminEnabled)
			if strings.TrimSpace(account.UserID) != "" {
				applyLocalAccountToDevSession(session, account)
			}
			session["bisque_linked"] = true
			writeJSON(w, http.StatusOK, session)
			return
		} else {
			setDevAuthCookie(w, "bisque:"+username)
		}
		writeJSON(w, http.StatusOK, devAuthSession(username, "bisque", nil, deps.DevAdminEnabled))
	}
}

func (deps ServerDeps) handleBisqueUnlink(w http.ResponseWriter, r *http.Request) {
	if deps.BisqueCredentials != nil {
		if sessionID := bisqueSessionIDFromRequest(r); sessionID != "" {
			if err := deps.BisqueCredentials.Unlink(r.Context(), sessionID); err != nil {
				writeError(w, http.StatusInternalServerError, err)
				return
			}
		}
	}
	if deps.WorkOS.Enabled() {
		clearBisqueSessionCookie(w, deps.WorkOS.cookieSecure)
		session, err := deps.workOSSessionResponseForRequest(w, r)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		if session["authenticated"] == true {
			session["bisque_linked"] = false
			writeJSON(w, http.StatusOK, session)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"authenticated": false,
			"user":          nil,
			"mode":          "workos",
			"provider":      "workos",
			"bisque_linked": false,
		})
		return
	}
	setDevAuthCookie(w, "signed_out")
	writeJSON(w, http.StatusOK, map[string]any{
		"authenticated": false,
		"user":          nil,
		"bisque_linked": false,
	})
}

func handleAuthLogout(deps ServerDeps) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if deps.WorkOS.Enabled() {
			clearBisqueSessionCookie(w, deps.WorkOS.cookieSecure)
			deps.WorkOS.handleLogout(w, r)
			return
		}
		if deps.BisqueCredentials != nil {
			if sessionID := bisqueSessionIDFromRequest(r); sessionID != "" {
				deps.BisqueCredentials.Delete(sessionID)
			}
		}
		setDevAuthCookie(w, "signed_out")
		writeJSON(w, http.StatusOK, map[string]any{
			"authenticated": false,
			"user":          nil,
			"bisque_linked": false,
		})
	}
}

func (deps ServerDeps) handleWorkOSCallback(w http.ResponseWriter, r *http.Request) {
	if !deps.WorkOS.Enabled() {
		writeError(w, http.StatusNotFound, errors.New("WorkOS auth is not configured"))
		return
	}
	deps.WorkOS.handleCallback(w, r)
}

func (deps ServerDeps) handleWorkOSBisqueLink(w http.ResponseWriter, r *http.Request, username string, password string) {
	username = strings.TrimSpace(username)
	if username == "" || password == "" {
		writeError(w, http.StatusBadRequest, errors.New("BisQue username and password are required"))
		return
	}
	snapshot, ok := deps.WorkOS.authenticateRequest(w, r)
	if !ok {
		writeError(w, http.StatusUnauthorized, errors.New("authentication required"))
		return
	}
	resolvedSnapshot, account, approved, err := deps.resolveWorkOSAccount(r.Context(), snapshot)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if !approved {
		writeJSON(w, http.StatusForbidden, workOSAccountDeniedResponse(snapshot, account))
		return
	}
	if deps.BisqueCredentials == nil {
		writeError(w, http.StatusInternalServerError, errors.New("BisQue credential store is not configured"))
		return
	}
	credentials := BisqueCredentials{Username: username, Password: password}
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	if err := deps.Bisque.VerifyCredentials(r.Context(), credentials); err != nil {
		writeBisqueError(w, err)
		return
	}
	sessionID, err := deps.BisqueCredentials.PutLinked(r.Context(), BisqueCredentialLinkInput{
		Credentials:    credentials,
		UserID:         resolvedSnapshot.Principal.UserID,
		OrgID:          resolvedSnapshot.Principal.OrgID,
		RootURL:        deps.bisqueRootURL(),
		LastVerifiedAt: domain.Now(),
		Metadata: domain.JSONMap{
			"source": "settings_link_account",
		},
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	setBisqueSessionCookie(w, sessionID, deps.WorkOS.cookieSecure)
	session := resolvedSnapshot.sessionResponse()
	session["bisque_linked"] = true
	writeJSON(w, http.StatusOK, session)
}

// localBootstrapAccounts returns the built-in dev-auth accounts. The admin
// password is read from ULTRA_CONTROL_DEV_ADMIN_PASSWORD so beta/production
// deployments can set a real credential without a code change; it defaults to
// "admin" to preserve local-dev behavior. Only the admin account carries a
// password, so it is the sole credential-loginable bootstrap account.
func localBootstrapAccounts() []localBootstrapAccount {
	adminPassword := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_DEV_ADMIN_PASSWORD"))
	if adminPassword == "" {
		adminPassword = "admin"
	}
	return []localBootstrapAccount{
		{
			Username:    "admin",
			Password:    adminPassword,
			UserID:      "bisque:admin",
			DisplayName: "Admin",
			Role:        "admin",
			Status:      "active",
			OrgID:       "local-org",
			Metadata: domain.JSONMap{
				"source": "local_bootstrap",
			},
		},
		{
			Username:    "amil",
			UserID:      "bisque:amil",
			DisplayName: "amil",
			Role:        "researcher",
			Status:      "active",
			OrgID:       "local-org",
			Metadata: domain.JSONMap{
				"source": "local_bootstrap",
			},
		},
	}
}

func localBootstrapAccountForCredential(username string, password string) (localBootstrapAccount, bool) {
	username = strings.ToLower(strings.TrimSpace(username))
	password = strings.TrimSpace(password)
	if username == "" || password == "" {
		return localBootstrapAccount{}, false
	}
	for _, account := range localBootstrapAccounts() {
		if strings.EqualFold(strings.TrimSpace(account.Username), username) &&
			account.Password != "" &&
			account.Password == password {
			return account, true
		}
	}
	return localBootstrapAccount{}, false
}

func (deps ServerDeps) ensureLocalBootstrapAccounts(ctx context.Context) error {
	accounts, ok := deps.Store.(accountStore)
	if !ok {
		return nil
	}
	for _, bootstrap := range localBootstrapAccounts() {
		if _, err := ensureLocalBootstrapAccount(ctx, accounts, bootstrap); err != nil {
			return err
		}
	}
	return nil
}

func ensureLocalBootstrapAccount(ctx context.Context, accounts accountStore, bootstrap localBootstrapAccount) (domain.UserAccount, error) {
	if account, found, err := accounts.GetUserByID(ctx, bootstrap.UserID); err != nil || found {
		return account, err
	}
	account, err := accounts.CreateUser(ctx, domain.CreateUserInput{
		UserID:      bootstrap.UserID,
		DisplayName: bootstrap.DisplayName,
		Role:        bootstrap.Role,
		Status:      bootstrap.Status,
		OrgID:       bootstrap.OrgID,
		Metadata:    bootstrap.Metadata,
	})
	if err != nil {
		if errors.Is(err, store.ErrConflict) {
			if account, found, getErr := accounts.GetUserByID(ctx, bootstrap.UserID); getErr != nil || found {
				return account, getErr
			}
		}
		return domain.UserAccount{}, err
	}
	return account, nil
}

func localBootstrapAccountRecord(bootstrap localBootstrapAccount) domain.UserAccount {
	now := domain.Now()
	return domain.UserAccount{
		UserID:      bootstrap.UserID,
		DisplayName: bootstrap.DisplayName,
		Role:        bootstrap.Role,
		Status:      bootstrap.Status,
		OrgID:       bootstrap.OrgID,
		CreatedAt:   now,
		UpdatedAt:   now,
		Metadata:    bootstrap.Metadata,
	}
}

func (deps ServerDeps) enforceLocalLoginApproval(w http.ResponseWriter, r *http.Request, username string) (domain.UserAccount, bool) {
	accounts, ok := deps.Store.(accountStore)
	if !ok {
		return domain.UserAccount{}, true
	}
	account, found, err := deps.lookupOrCreateLocalLoginAccount(r.Context(), accounts, username)
	if err != nil {
		writeStoreError(w, err)
		return domain.UserAccount{}, false
	}
	status := normalizeAccountStatus(account.Status)
	if status == "" {
		status = "pending"
	}
	if isActiveAccount(account) {
		return account, true
	}
	setDevAuthCookie(w, "signed_out")
	writeJSON(w, http.StatusForbidden, map[string]any{
		"authenticated":   false,
		"provider":        "local",
		"mode":            "local",
		"account_status":  status,
		"account_email":   account.Email,
		"account_user_id": account.UserID,
		"message":         accountStatusMessage(status),
		"pending_created": !found,
	})
	return domain.UserAccount{}, false
}

func (deps ServerDeps) lookupOrCreateLocalLoginAccount(ctx context.Context, accounts accountStore, username string) (domain.UserAccount, bool, error) {
	normalizedUsername := strings.TrimSpace(username)
	email := normalizeAuthEmail(normalizedUsername)
	if email != "" {
		if account, found, err := accounts.GetUserByEmail(ctx, email); err != nil || found {
			return account, found, err
		}
	}
	userID := devPrincipalUserID(normalizedUsername, "bisque")
	if account, found, err := accounts.GetUserByID(ctx, userID); err != nil || found {
		return account, found, err
	}
	input := domain.CreateUserInput{
		UserID:      userID,
		DisplayName: normalizedUsername,
		Role:        "researcher",
		Status:      "pending",
		OrgID:       "local-org",
		Metadata: domain.JSONMap{
			"source": "bisque_login",
		},
	}
	if email != "" {
		input.Email = email
	}
	account, err := accounts.CreateUser(ctx, input)
	if err != nil {
		if errors.Is(err, store.ErrConflict) {
			if email != "" {
				if account, found, getErr := accounts.GetUserByEmail(ctx, email); getErr != nil || found {
					return account, found, getErr
				}
			}
			return accounts.GetUserByID(ctx, userID)
		}
		return domain.UserAccount{}, false, err
	}
	return account, false, nil
}

func (deps ServerDeps) workOSSessionResponseForRequest(w http.ResponseWriter, r *http.Request) (map[string]any, error) {
	snapshot, authenticated := deps.WorkOS.authenticateRequest(w, r)
	if !authenticated {
		return map[string]any{
			"authenticated": false,
			"user":          nil,
			"mode":          "workos",
			"provider":      "workos",
			"bisque_linked": false,
		}, nil
	}
	resolvedSnapshot, account, approved, err := deps.resolveWorkOSAccount(r.Context(), snapshot)
	if err != nil {
		return nil, err
	}
	if !approved {
		return workOSAccountDeniedResponse(snapshot, account), nil
	}
	return resolvedSnapshot.sessionResponse(), nil
}

func (deps ServerDeps) requireWorkOSAccount(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if isWorkerScopedEndpoint(r) && deps.workerRequestAuth(r) == workerAuthValid {
			next.ServeHTTP(w, r)
			return
		}
		snapshot, authenticated := deps.WorkOS.authenticateRequest(w, r)
		if !authenticated {
			writeError(w, http.StatusUnauthorized, errors.New("authentication required"))
			return
		}
		resolvedSnapshot, account, approved, err := deps.resolveWorkOSAccount(r.Context(), snapshot)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		if !approved {
			writeJSON(w, http.StatusForbidden, workOSAccountDeniedResponse(snapshot, account))
			return
		}
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), workOSPrincipalContextKey{}, resolvedSnapshot)))
	})
}

func (deps ServerDeps) requireWorkOSAdmin(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		snapshot, ok := workOSSnapshotFromContext(r.Context())
		if !ok || !strings.EqualFold(snapshot.Principal.Role, "admin") {
			writeError(w, http.StatusForbidden, errors.New("admin role required"))
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (deps ServerDeps) resolveWorkOSAccount(ctx context.Context, snapshot workOSSessionSnapshot) (workOSSessionSnapshot, domain.UserAccount, bool, error) {
	accounts, ok := deps.Store.(accountStore)
	if !ok {
		return snapshot, domain.UserAccount{Status: "not_configured"}, false, errors.New("Ultra account storage is not configured")
	}
	account, found, err := accounts.GetUserByID(ctx, snapshot.Principal.UserID)
	if err != nil {
		return snapshot, domain.UserAccount{}, false, err
	}
	if !found && strings.TrimSpace(snapshot.Email) != "" {
		account, found, err = accounts.GetUserByEmail(ctx, snapshot.Email)
		if err != nil {
			return snapshot, domain.UserAccount{}, false, err
		}
	}
	if !found {
		account, err = accounts.CreateUser(ctx, domain.CreateUserInput{
			UserID:      snapshot.Principal.UserID,
			Email:       snapshot.Email,
			DisplayName: workOSDisplayName(snapshot),
			Role:        "researcher",
			Status:      "active",
			OrgID:       snapshot.Principal.OrgID,
			Metadata: domain.JSONMap{
				"source":       "workos_authkit",
				"workos_id":    snapshot.UserID,
				"workos_email": snapshot.Email,
			},
		})
		if err != nil {
			if errors.Is(err, store.ErrConflict) {
				account, found, err = accounts.GetUserByID(ctx, snapshot.Principal.UserID)
				if err == nil && !found && strings.TrimSpace(snapshot.Email) != "" {
					account, found, err = accounts.GetUserByEmail(ctx, snapshot.Email)
				}
				if err == nil && found {
					return deps.applyAdminEmailOverride(applyUltraAccountToWorkOSSnapshot(snapshot, account)), account, isActiveAccount(account), nil
				}
			}
			return snapshot, domain.UserAccount{}, false, err
		}
	}
	resolved := deps.applyAdminEmailOverride(applyUltraAccountToWorkOSSnapshot(snapshot, account))
	return resolved, account, isActiveAccount(account), nil
}

// applyAdminEmailOverride makes the ULTRA_CONTROL_ADMIN_EMAILS allowlist the
// final authority on the admin role, winning over the stored account role. This
// is the bootstrap path: a fresh WorkOS account is created with role
// "researcher" and there is no in-app role editor, so without this an operator
// could never get the first admin into the console.
func (deps ServerDeps) applyAdminEmailOverride(snapshot workOSSessionSnapshot) workOSSessionSnapshot {
	if deps.WorkOS.isAdminEmail(snapshot.Email) {
		snapshot.Principal.Role = "admin"
	}
	return snapshot
}

func applyUltraAccountToWorkOSSnapshot(snapshot workOSSessionSnapshot, account domain.UserAccount) workOSSessionSnapshot {
	snapshot.Principal.UserID = firstNonEmpty(strings.TrimSpace(account.UserID), snapshot.Principal.UserID)
	snapshot.Principal.OrgID = firstNonEmpty(strings.TrimSpace(account.OrgID), snapshot.Principal.OrgID)
	snapshot.Principal.Role = firstNonEmpty(strings.TrimSpace(account.Role), snapshot.Principal.Role, "researcher")
	snapshot.AccountStatus = normalizeAccountStatus(account.Status)
	if email := normalizeAuthEmail(account.Email); email != "" {
		snapshot.Email = email
	}
	if displayName := strings.TrimSpace(account.DisplayName); displayName != "" && snapshot.FirstName == "" && snapshot.LastName == "" {
		snapshot.FirstName = displayName
	}
	return snapshot
}

func workOSAccountDeniedResponse(snapshot workOSSessionSnapshot, account domain.UserAccount) map[string]any {
	status := normalizeAccountStatus(account.Status)
	if status == "" {
		status = "pending"
	}
	email := firstNonEmpty(normalizeAuthEmail(account.Email), normalizeAuthEmail(snapshot.Email))
	userID := firstNonEmpty(strings.TrimSpace(account.UserID), snapshot.Principal.UserID)
	return map[string]any{
		"authenticated":   false,
		"user":            nil,
		"mode":            "workos",
		"provider":        "workos",
		"bisque_linked":   false,
		"account_status":  status,
		"account_email":   email,
		"account_user_id": userID,
		"message":         accountStatusMessage(status),
	}
}

func isActiveAccount(account domain.UserAccount) bool {
	return normalizeAccountStatus(account.Status) == "active"
}

func normalizeAccountStatus(status string) string {
	return strings.ToLower(strings.TrimSpace(status))
}

func normalizeAuthEmail(email string) string {
	return strings.ToLower(strings.TrimSpace(email))
}

func accountStatusMessage(status string) string {
	switch normalizeAccountStatus(status) {
	case "active":
		return "Your account is approved."
	case "disabled":
		return "Your account has been disabled. Contact an administrator for access."
	case "rejected":
		return "Your account request was not approved. Contact an administrator for details."
	case "not_configured":
		return "Account approval storage is not configured."
	default:
		return "Your account request is pending administrator approval."
	}
}

func workOSDisplayName(snapshot workOSSessionSnapshot) string {
	return strings.TrimSpace(strings.Join([]string{snapshot.FirstName, snapshot.LastName}, " "))
}

func (deps ServerDeps) authProviderName() string {
	if deps.WorkOS.Enabled() {
		return "workos"
	}
	return "local"
}

func devAuthSessionFromRequest(r *http.Request, adminEnabled bool, credentials *BisqueCredentialStore) map[string]any {
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
				"bisque_linked": false,
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
		if sessionID, ok := strings.CutPrefix(value, "bisque_session:"); ok {
			sessionID = strings.TrimSpace(sessionID)
			if credentials != nil {
				if linked, found, _ := credentials.GetWithContext(r.Context(), sessionID); found {
					username := strings.TrimSpace(linked.Username)
					if username == "" {
						username = "local-user"
					}
					session := devAuthSession(username, "bisque", nil, adminEnabled)
					session["bisque_linked"] = true
					return session
				}
			}
			return map[string]any{
				"authenticated": false,
				"user":          nil,
				"bisque_linked": false,
			}
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

func bisqueSessionIDFromRequest(r *http.Request) string {
	if sessionID := cookieValueFromRequest(r, bisqueSessionCookieName); sessionID != "" {
		return sessionID
	}
	cookie, err := r.Cookie("ultra_dev_auth")
	if err != nil {
		return ""
	}
	value := strings.TrimSpace(cookie.Value)
	if decoded, decodeErr := url.QueryUnescape(value); decodeErr == nil {
		value = decoded
	}
	sessionID, ok := strings.CutPrefix(value, "bisque_session:")
	if !ok {
		return ""
	}
	return strings.TrimSpace(sessionID)
}

func (deps ServerDeps) hasLinkedBisqueSession(ctx context.Context, r *http.Request) bool {
	if deps.BisqueCredentials == nil {
		return false
	}
	if sessionID := bisqueSessionIDFromRequest(r); sessionID != "" {
		if _, found, err := deps.BisqueCredentials.GetWithContext(ctx, sessionID); found && err == nil {
			return true
		}
	}
	// Linked status is a durable property of the account, not the session
	// cookie: a user who linked BisQue stays linked after the cookie expires.
	principal := deps.principalFromRequest(r, "")
	if strings.TrimSpace(principal.UserID) == "" {
		return false
	}
	_, _, ok := deps.BisqueCredentials.ResolveLinkedSessionForUser(ctx, principal.UserID, principal.OrgID)
	return ok
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
			"id":       devPrincipalUserID(username, mode),
			"username": username,
			"org_id":   "local-org",
			"role":     role,
		},
		"mode":          mode,
		"is_admin":      adminEnabled,
		"bisque_linked": false,
	}
	if guestProfile != nil {
		session["guest_profile"] = guestProfile
	}
	return session
}

func applyLocalAccountToDevSession(session map[string]any, account domain.UserAccount) {
	role := strings.TrimSpace(account.Role)
	if role == "" {
		role = "researcher"
	}
	session["is_admin"] = strings.EqualFold(role, "admin")
	if user, ok := session["user"].(map[string]any); ok {
		if userID := strings.TrimSpace(account.UserID); userID != "" {
			user["id"] = userID
		}
		if orgID := strings.TrimSpace(account.OrgID); orgID != "" {
			user["org_id"] = orgID
		}
		user["role"] = role
	}
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

func (deps ServerDeps) principalFromRequest(r *http.Request, fallbackUserID string) requestPrincipal {
	if principal, ok := deps.workerRunPrincipal(r); ok {
		return principal
	}
	if principal, ok := deps.workerDataAgentPrincipal(r); ok {
		return principal
	}
	if deps.WorkOS.Enabled() {
		if principal, ok := deps.WorkOS.principalFromRequest(r); ok {
			return principal
		}
		return requestPrincipal{UserID: "unauthenticated", OrgID: "workos-org", Role: "anonymous"}
	}
	userIDHeader := strings.TrimSpace(r.Header.Get("X-Ultra-User-Id"))
	orgID := firstNonEmpty(
		strings.TrimSpace(r.Header.Get("X-Ultra-Org-Id")),
		"local-org",
	)
	role := firstNonEmpty(
		strings.TrimSpace(r.Header.Get("X-Ultra-Role")),
		"researcher",
	)
	if userIDHeader != "" {
		return requestPrincipal{UserID: userIDHeader, OrgID: orgID, Role: role}
	}
	if userID, ok := deps.devCookiePrincipalUserID(r); ok {
		return requestPrincipal{UserID: userID, OrgID: orgID, Role: role}
	}
	return requestPrincipal{
		UserID: firstNonEmpty(strings.TrimSpace(fallbackUserID), "local-user"),
		OrgID:  orgID,
		Role:   role,
	}
}

// workerRunPrincipal anchors trusted-worker requests to the user that owns the
// run named in X-Ultra-Run-Id, so files staged or pushed by agent tools land in
// that user's catalog instead of an "unauthenticated" principal.
func (deps ServerDeps) workerRunPrincipal(r *http.Request) (requestPrincipal, bool) {
	if deps.Store == nil || deps.workerRequestAuth(r) != workerAuthValid || !isWorkerScopedEndpoint(r) {
		return requestPrincipal{}, false
	}
	runID := strings.TrimSpace(r.Header.Get("X-Ultra-Run-Id"))
	if runID == "" {
		return requestPrincipal{}, false
	}
	run, err := deps.Store.GetRun(r.Context(), runID)
	if err != nil || strings.TrimSpace(run.UserID) == "" {
		return requestPrincipal{}, false
	}
	orgID := trustedRunOrgID(run)
	principalMetadata, _ := jsonMapValue(run.Metadata["principal"])
	role, _ := safeMetadataString(principalMetadata["role"], 128)
	if role == "" {
		role, _ = safeMetadataString(run.Metadata["principal_role"], 128)
	}
	if orgID == "" || role == "" {
		return requestPrincipal{}, false
	}
	return requestPrincipal{UserID: strings.TrimSpace(run.UserID), OrgID: orgID, Role: role}, true
}

func (deps ServerDeps) workerDataAgentPrincipal(r *http.Request) (requestPrincipal, bool) {
	if deps.workerRequestAuth(r) != workerAuthValid || !isWorkerDataAgentEndpoint(r) {
		return requestPrincipal{}, false
	}
	userID := strings.TrimSpace(r.Header.Get("X-Ultra-User-Id"))
	if userID == "" {
		return requestPrincipal{}, false
	}
	orgID := firstNonEmpty(strings.TrimSpace(r.Header.Get("X-Ultra-Org-Id")), "local-org")
	role := firstNonEmpty(strings.TrimSpace(r.Header.Get("X-Ultra-Role")), "researcher")
	return requestPrincipal{UserID: userID, OrgID: orgID, Role: role}, true
}

func (deps ServerDeps) devCookiePrincipalUserID(r *http.Request) (string, bool) {
	cookie, err := r.Cookie("ultra_dev_auth")
	if err != nil {
		return "", false
	}
	value := strings.TrimSpace(cookie.Value)
	if decoded, decodeErr := url.QueryUnescape(value); decodeErr == nil {
		value = decoded
	}
	if value == "" || value == "signed_out" {
		return "", false
	}
	if username, ok := strings.CutPrefix(value, "guest:"); ok {
		return devPrincipalUserID(username, "guest"), true
	}
	if username, ok := strings.CutPrefix(value, "bisque:"); ok {
		return devPrincipalUserID(username, "bisque"), true
	}
	if sessionID, ok := strings.CutPrefix(value, "bisque_session:"); ok {
		sessionID = strings.TrimSpace(sessionID)
		if deps.BisqueCredentials == nil || sessionID == "" {
			return "", false
		}
		credentials, found, err := deps.BisqueCredentials.GetWithContext(r.Context(), sessionID)
		if err != nil || !found {
			return "", false
		}
		return devPrincipalUserID(credentials.Username, "bisque"), true
	}
	return "", false
}

func devPrincipalUserID(username string, mode string) string {
	mode = strings.ToLower(strings.TrimSpace(mode))
	switch mode {
	case "bisque":
		return "bisque:" + devPrincipalIDSegment(username, "local-user")
	case "guest":
		segment := devPrincipalIDSegment(username, "guest")
		if segment == "guest" {
			return "local-user"
		}
		return "guest:" + segment
	default:
		return "local-user"
	}
}

func devPrincipalIDSegment(value string, fallback string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		value = strings.ToLower(strings.TrimSpace(fallback))
	}
	var b strings.Builder
	lastDash := false
	for _, r := range value {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r), r == '_', r == '.', r == '@':
			b.WriteRune(r)
			lastDash = false
		case r == '-' || unicode.IsSpace(r):
			if !lastDash && b.Len() > 0 {
				b.WriteRune('-')
				lastDash = true
			}
		}
	}
	segment := strings.Trim(b.String(), "-")
	if segment == "" {
		return "local-user"
	}
	return segment
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

func setBisqueSessionCookie(w http.ResponseWriter, sessionID string, secure bool) {
	http.SetCookie(w, &http.Cookie{
		Name:     bisqueSessionCookieName,
		Value:    url.QueryEscape(strings.TrimSpace(sessionID)),
		Path:     "/",
		HttpOnly: true,
		Secure:   secure,
		SameSite: http.SameSiteLaxMode,
		MaxAge:   int((30 * 24 * time.Hour).Seconds()),
	})
}

func clearBisqueSessionCookie(w http.ResponseWriter, secure bool) {
	http.SetCookie(w, &http.Cookie{
		Name:     bisqueSessionCookieName,
		Value:    "",
		Path:     "/",
		HttpOnly: true,
		Secure:   secure,
		SameSite: http.SameSiteLaxMode,
		MaxAge:   -1,
	})
}

func cookieValueFromRequest(r *http.Request, name string) string {
	cookie, err := r.Cookie(name)
	if err != nil {
		return ""
	}
	value := strings.TrimSpace(cookie.Value)
	if decoded, decodeErr := url.QueryUnescape(value); decodeErr == nil {
		value = decoded
	}
	return strings.TrimSpace(value)
}

type createThreadRequest struct {
	UserID          string                 `json:"user_id"`
	Title           string                 `json:"title"`
	Metadata        map[string]any         `json:"metadata"`
	InitialMessages []domain.ThreadMessage `json:"initial_messages"`
}

type upsertThreadRequest struct {
	Title    string         `json:"title"`
	Metadata map[string]any `json:"metadata"`
}

type createRunRequest struct {
	UserID                string                 `json:"user_id"`
	Goal                  string                 `json:"goal"`
	EvaluationProfile     string                 `json:"evaluation_profile"`
	RemoteMutationIntents []string               `json:"remote_mutation_intents"`
	Messages              []domain.ThreadMessage `json:"messages"`
	FileIDs               []string               `json:"file_ids"`
	ResourceURIs          []string               `json:"resource_uris"`
	DatasetURIs           []string               `json:"dataset_uris"`
	SelectedToolNames     []string               `json:"selected_tool_names"`
	KnowledgeContext      map[string]any         `json:"knowledge_context"`
	WorkflowHint          map[string]any         `json:"workflow_hint"`
	SelectionContext      map[string]any         `json:"selection_context"`
	ReasoningMode         string                 `json:"reasoning_mode"`
	Budgets               map[string]any         `json:"budgets"`
	Benchmark             map[string]any         `json:"benchmark"`
	ResourceDescriptors   []domain.JSONMap       `json:"resource_descriptors"`
	IdempotencyKey        string                 `json:"idempotency_key"`
	Metadata              map[string]any         `json:"metadata"`
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
	Count      int                   `json:"count"`
	TotalCount int                   `json:"total_count"`
	Limit      int                   `json:"limit"`
	Offset     int                   `json:"offset"`
	HasMore    bool                  `json:"has_more"`
	Threads    []domain.ThreadRecord `json:"threads"`
}

type threadMessagesResponse struct {
	ThreadID string                 `json:"thread_id"`
	Count    int                    `json:"count"`
	Messages []domain.ThreadMessage `json:"messages"`
	// Pagination (present only when ?limit/?before is used): NextCursor is the oldest message id in
	// this page — pass it back as ?before to load the previous (older) page. HasMore is false on the
	// first/oldest page. Absent when the endpoint returns the full thread (no pagination params).
	NextCursor string `json:"next_cursor,omitempty"`
	HasMore    bool   `json:"has_more,omitempty"`
}

type runEventsResponse struct {
	RunID  string                  `json:"run_id"`
	Count  int                     `json:"count"`
	Events []domain.RunEventRecord `json:"events"`
	// NextCursor/HasMore implement keyset pagination: when a full page is returned, NextCursor is
	// an opaque token the client passes back via ?cursor= to fetch the next page. Omitted on the
	// last page so callers can drain a large trace deterministically without hand-tracking sequence.
	NextCursor string `json:"next_cursor,omitempty"`
	HasMore    bool   `json:"has_more,omitempty"`
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
	FileID             string          `json:"file_id"`
	OriginalName       string          `json:"original_name"`
	ContentType        string          `json:"content_type,omitempty"`
	SizeBytes          int64           `json:"size_bytes"`
	SHA256             string          `json:"sha256"`
	CreatedAt          string          `json:"created_at"`
	SourceURI          string          `json:"source_uri,omitempty"`
	ProjectID          string          `json:"project_id,omitempty"`
	PreviewURL         string          `json:"preview_url,omitempty"`
	Principal          principalRecord `json:"principal,omitempty"`
	TrustedSourceRunID string          `json:"-"`
}

type uploadFilesResponse struct {
	FileCount int                  `json:"file_count"`
	Uploaded  []uploadedFileRecord `json:"uploaded"`
}

type createUploadSessionRequest struct {
	IdempotencyKey     string                           `json:"idempotency_key"`
	BrowserFingerprint string                           `json:"browser_fingerprint"`
	ProjectID          string                           `json:"project_id"`
	TotalBytes         int64                            `json:"total_bytes"`
	Files              []createUploadSessionFileRequest `json:"files"`
}

type createUploadSessionFileRequest struct {
	FileToken      string `json:"file_token"`
	OriginalName   string `json:"original_name"`
	RelativePath   string `json:"relative_path"`
	ContentType    string `json:"content_type"`
	SizeBytes      int64  `json:"size_bytes"`
	DeclaredSHA256 string `json:"declared_sha256"`
}

type uploadSessionResponse struct {
	Session domain.UploadSessionRecord        `json:"session"`
	Files   []domain.UploadSessionFileRecord  `json:"files"`
	Chunks  []domain.UploadChunkRecord        `json:"chunks,omitempty"`
	Events  []domain.UploadSessionEventRecord `json:"events"`
	Limits  uploadSessionLimits               `json:"limits"`
}

type uploadSessionLimits struct {
	MaxParallelFiles   int `json:"max_parallel_files"`
	MaxParallelChunks  int `json:"max_parallel_chunks"`
	MaxFilesPerSession int `json:"max_files_per_session"`
}

type uploadChunkResponse struct {
	Session domain.UploadSessionRecord     `json:"session"`
	File    domain.UploadSessionFileRecord `json:"file"`
	Chunk   domain.UploadChunkRecord       `json:"chunk"`
}

type uploadSessionFileCompleteResponse struct {
	Session  domain.UploadSessionRecord     `json:"session"`
	File     domain.UploadSessionFileRecord `json:"file"`
	Resource uploadedFileRecord             `json:"resource"`
}

type uploadHistogramResponse struct {
	FileID      string                 `json:"file_id"`
	Bins        int                    `json:"bins"`
	DType       string                 `json:"dtype"`
	Channels    []int                  `json:"channels"`
	Source      string                 `json:"source"`
	SampleCount int                    `json:"sample_count"`
	Histogram   uploadHistogramPayload `json:"histogram"`
}

type uploadHistogramPayload struct {
	Bins           []int     `json:"bins"`
	Edges          []float64 `json:"edges"`
	Min            float64   `json:"min"`
	Max            float64   `json:"max"`
	ChannelIndices []int     `json:"channel_indices"`
	TimeIndex      int       `json:"time_index"`
}

type resourceRecord struct {
	FileID             string                      `json:"file_id"`
	OriginalName       string                      `json:"original_name"`
	ContentType        string                      `json:"content_type,omitempty"`
	SizeBytes          int64                       `json:"size_bytes"`
	SHA256             string                      `json:"sha256"`
	CreatedAt          string                      `json:"created_at"`
	Status             string                      `json:"status"`
	SourceType         string                      `json:"source_type"`
	ResourceKind       string                      `json:"resource_kind"`
	SourceURI          string                      `json:"source_uri,omitempty"`
	ProjectID          string                      `json:"project_id,omitempty"`
	HasThumbnail       bool                        `json:"has_thumbnail"`
	ThumbnailURL       string                      `json:"thumbnail_url,omitempty"`
	PreviewURL         string                      `json:"preview_url,omitempty"`
	CacheReady         bool                        `json:"cache_ready"`
	StagedLocally      bool                        `json:"staged_locally"`
	Principal          principalRecord             `json:"principal,omitempty"`
	Tags               []string                    `json:"tags,omitempty"`
	Metadata           domain.JSONMap              `json:"metadata,omitempty"`
	ShareSummary       domain.ResourceShareSummary `json:"share_summary,omitempty"`
	TrustedSourceRunID string                      `json:"-"`
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

type patchResourceRequest struct {
	OriginalName                          string         `json:"original_name"`
	Metadata                              domain.JSONMap `json:"metadata"`
	CalibrationExpectedSourceSHA256       string         `json:"-"`
	CalibrationSelectionExpectedRevisions map[string]int `json:"-"`
}

type bulkLifecycleResourcesRequest struct {
	ResourceIDs []string `json:"resource_ids"`
}

type bulkLifecycleResourcesResponse struct {
	Count     int                          `json:"count"`
	Resources []resourceRecord             `json:"resources"`
	Events    []domain.ResourceEventRecord `json:"events"`
}

type bulkTagResourcesRequest struct {
	ResourceIDs []string       `json:"resource_ids"`
	Tags        []string       `json:"tags"`
	Metadata    domain.JSONMap `json:"metadata"`
}

type bulkTagResourcesResponse struct {
	Count     int                          `json:"count"`
	Resources []resourceRecord             `json:"resources"`
	Events    []domain.ResourceEventRecord `json:"events"`
}

type resourceEventsResponse struct {
	ResourceID string                       `json:"resource_id"`
	Count      int                          `json:"count"`
	Events     []domain.ResourceEventRecord `json:"events"`
}

type resourceEventListResponse struct {
	Count      int                          `json:"count"`
	TotalCount int                          `json:"total_count"`
	Limit      int                          `json:"limit"`
	Offset     int                          `json:"offset"`
	Events     []domain.ResourceEventRecord `json:"events"`
}

type createResourceShareGrantRequest struct {
	GranteeUserID string         `json:"grantee_user_id"`
	GranteeOrgID  string         `json:"grantee_org_id"`
	Public        bool           `json:"public"`
	Role          string         `json:"role"`
	Metadata      domain.JSONMap `json:"metadata"`
}

type createResourceShareGrantsRequest struct {
	ResourceIDs   []string       `json:"resource_ids"`
	GranteeUserID string         `json:"grantee_user_id"`
	GranteeOrgID  string         `json:"grantee_org_id"`
	Public        bool           `json:"public"`
	Role          string         `json:"role"`
	Metadata      domain.JSONMap `json:"metadata"`
}

type resourceShareGrantResponse struct {
	Grant domain.ResourceShareGrantRecord `json:"grant"`
}

type resourceShareGrantsCreateResponse struct {
	Count  int                               `json:"count"`
	Grants []domain.ResourceShareGrantRecord `json:"grants"`
}

type resourceCollectionShareGrantsCreateResponse struct {
	Count      int                               `json:"count"`
	Collection domain.ResourceCollectionRecord   `json:"collection"`
	Grants     []domain.ResourceShareGrantRecord `json:"grants"`
}

type resourceShareGrantsResponse struct {
	ResourceID string                            `json:"resource_id"`
	Count      int                               `json:"count"`
	Grants     []domain.ResourceShareGrantRecord `json:"grants"`
}

type createResourceCollectionRequest struct {
	Name               string         `json:"name"`
	Description        string         `json:"description"`
	CollectionType     string         `json:"collection_type"`
	ProjectID          string         `json:"project_id"`
	ParentCollectionID string         `json:"parent_collection_id"`
	Metadata           domain.JSONMap `json:"metadata"`
}

type patchResourceCollectionRequest struct {
	Name string `json:"name"`
}

type resourceCollectionResponse struct {
	Collection domain.ResourceCollectionRecord `json:"collection"`
}

type resourceCollectionsResponse struct {
	Count       int                               `json:"count"`
	Collections []domain.ResourceCollectionRecord `json:"collections"`
}

type addResourcesToCollectionRequest struct {
	ResourceIDs []string       `json:"resource_ids"`
	Metadata    domain.JSONMap `json:"metadata"`
}

type addResourcesToCollectionResponse struct {
	Collection  domain.ResourceCollectionRecord             `json:"collection"`
	AddedCount  int                                         `json:"added_count"`
	Memberships []domain.ResourceCollectionMembershipRecord `json:"memberships"`
}

type removeResourcesFromCollectionResponse struct {
	Collection   domain.ResourceCollectionRecord             `json:"collection"`
	RemovedCount int                                         `json:"removed_count"`
	Memberships  []domain.ResourceCollectionMembershipRecord `json:"memberships"`
}

type createDatasetSnapshotRequest struct {
	Name               string                               `json:"name"`
	Description        string                               `json:"description"`
	SourceCollectionID string                               `json:"source_collection_id"`
	ResourceIDs        []string                             `json:"resource_ids"`
	ResourceQuery      *datasetSnapshotResourceQueryRequest `json:"resource_query"`
	ProjectID          string                               `json:"project_id"`
	Metadata           domain.JSONMap                       `json:"metadata"`
}

type datasetSnapshotResourceQueryRequest struct {
	Query            string                          `json:"q"`
	Kind             string                          `json:"kind"`
	Source           string                          `json:"source"`
	ProjectID        string                          `json:"project_id"`
	Sharing          string                          `json:"sharing"`
	Tags             []string                        `json:"tags"`
	Descriptors      []string                        `json:"descriptors"`
	MetadataFilters  []domain.ResourceMetadataFilter `json:"metadata_filters"`
	CreatedAfter     string                          `json:"created_after"`
	CreatedBefore    string                          `json:"created_before"`
	ProcessingStatus string                          `json:"processing_status"`
}

type datasetSnapshotResponse struct {
	Snapshot  domain.DatasetSnapshotRecord           `json:"snapshot"`
	Resources []domain.DatasetSnapshotResourceRecord `json:"resources"`
}

type datasetSnapshotsResponse struct {
	Count     int                            `json:"count"`
	Snapshots []domain.DatasetSnapshotRecord `json:"snapshots"`
}

type datasetSnapshotEventsResponse struct {
	SnapshotID string                              `json:"snapshot_id"`
	Count      int                                 `json:"count"`
	TotalCount int                                 `json:"total_count"`
	Limit      int                                 `json:"limit"`
	Offset     int                                 `json:"offset"`
	Events     []domain.DatasetSnapshotEventRecord `json:"events"`
}

type datasetSnapshotShareGrantResponse struct {
	Grant domain.DatasetSnapshotShareGrantRecord `json:"grant"`
}

type datasetSnapshotShareGrantsResponse struct {
	Count  int                                      `json:"count"`
	Grants []domain.DatasetSnapshotShareGrantRecord `json:"grants"`
}

type createDataAgentJobRequest struct {
	JobType            string                               `json:"job_type"`
	ResourceIDs        []string                             `json:"resource_ids"`
	SourceCollectionID string                               `json:"source_collection_id"`
	ProjectID          string                               `json:"project_id"`
	ResourceQuery      *datasetSnapshotResourceQueryRequest `json:"resource_query"`
	InputSelector      domain.JSONMap                       `json:"input_selector"`
	Metadata           domain.JSONMap                       `json:"metadata"`
}

type updateDataAgentJobStatusRequest struct {
	Status            string         `json:"status"`
	ProgressCompleted int            `json:"progress_completed"`
	ProgressTotal     int            `json:"progress_total"`
	Error             string         `json:"error"`
	Message           string         `json:"message"`
	OutputSummary     domain.JSONMap `json:"output_summary"`
	Metadata          domain.JSONMap `json:"metadata"`
	EventMetadata     domain.JSONMap `json:"event_metadata"`
}

type appendDataAgentJobEventRequest struct {
	EventID   string         `json:"event_id"`
	EventType string         `json:"event_type"`
	Message   string         `json:"message"`
	Metadata  domain.JSONMap `json:"metadata"`
}

type controlDataAgentJobRequest struct {
	Action   string         `json:"action"`
	Reason   string         `json:"reason"`
	Metadata domain.JSONMap `json:"metadata"`
}

type dataAgentJobResponse struct {
	Job    domain.DataAgentJobRecord        `json:"job"`
	Events []domain.DataAgentJobEventRecord `json:"events"`
}

type dataAgentJobsResponse struct {
	Count int                         `json:"count"`
	Jobs  []domain.DataAgentJobRecord `json:"jobs"`
}

type promoteArtifactResourceRequest struct {
	OriginalName string `json:"original_name"`
	ProjectID    string `json:"project_id"`
}

type resourceReconcileIssue struct {
	IssueType   string `json:"issue_type"`
	Severity    string `json:"severity"`
	ResourceID  string `json:"resource_id,omitempty"`
	Path        string `json:"path,omitempty"`
	Message     string `json:"message"`
	ExpectedSHA string `json:"expected_sha,omitempty"`
	ActualSHA   string `json:"actual_sha,omitempty"`
}

type resourceReconcileResponse struct {
	CheckedAt  string                   `json:"checked_at"`
	IssueCount int                      `json:"issue_count"`
	Summary    map[string]int           `json:"summary"`
	Issues     []resourceReconcileIssue `json:"issues"`
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
	AppVersion               string  `json:"app_version,omitempty"`
	StoreBackend             string  `json:"store_backend"`
	DispatchMode             string  `json:"dispatch_mode"`
	JobTransport             string  `json:"job_transport"`
	EventTransport           string  `json:"event_transport"`
	StubWorkerEnabled        bool    `json:"stub_worker_enabled"`
	NATSConfigured           bool    `json:"nats_configured"`
	NATSStream               string  `json:"nats_stream,omitempty"`
	NATSJobsSubject          string  `json:"nats_jobs_subject,omitempty"`
	NATSDataAgentJobsSubject string  `json:"nats_data_agent_jobs_subject,omitempty"`
	NATSEventsSubject        string  `json:"nats_events_subject,omitempty"`
	NATSCancelSubject        string  `json:"nats_cancel_subject,omitempty"`
	NATSEventConsumer        string  `json:"nats_event_consumer,omitempty"`
	ArtifactRoot             string  `json:"artifact_root,omitempty"`
	UploadRoot               string  `json:"upload_root,omitempty"`
	RunRecoveryEnabled       bool    `json:"run_recovery_enabled"`
	RunRecoveryIntervalSecs  float64 `json:"run_recovery_interval_seconds,omitempty"`
	RunRecoveryBatchLimit    int     `json:"run_recovery_batch_limit,omitempty"`
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

type adminActivityPeriod struct {
	Label             string `json:"label"`
	Window            string `json:"window"`
	Messages          int    `json:"messages"`
	UserMessages      int    `json:"user_messages"`
	AssistantMessages int    `json:"assistant_messages"`
	ToolCalls         int    `json:"tool_calls"`
	ActiveUsers       int    `json:"active_users"`
	Runs              int    `json:"runs"`
	FailedRuns        int    `json:"failed_runs"`
	Artifacts         int    `json:"artifacts"`
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

type resourceAccountingSummary struct {
	ActiveResources      int
	SoftDeletedResources int
	ActiveBytes          int64
	Users                map[string]resourceOwnerAccounting
	Orgs                 map[string]resourceOwnerAccounting
	Projects             map[string]resourceOwnerAccounting
}

type resourceOwnerAccounting struct {
	Uploads      int
	StorageBytes int64
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

type adminImageCacheStats struct {
	Enabled  bool    `json:"enabled"`
	Hits     uint64  `json:"hits"`
	Misses   uint64  `json:"misses"`
	HitRate  float64 `json:"hit_rate"`
	Entries  int     `json:"entries"`
	Bytes    int64   `json:"bytes"`
	MaxBytes int64   `json:"max_bytes"`
}

// adminRetentionBacklog surfaces how much storage is held by soft-deleted resources past
// their undelete window — the unbounded-growth backlog a retention GC reclaims.
type adminRetentionBacklog struct {
	ExpiredResources int64 `json:"expired_resources"`
	ReclaimableBytes int64 `json:"reclaimable_bytes"`
}

type retentionBacklogStore interface {
	RetentionBacklog(context.Context, time.Time) (domain.ResourceRetentionBacklog, error)
}

type adminOverviewResponse struct {
	GeneratedAt      string                                 `json:"generated_at"`
	Runtime          RuntimeSummary                         `json:"runtime"`
	Queue            adminQueueDiagnostics                  `json:"queue"`
	Database         adminDatabaseDiagnostics               `json:"database"`
	ImageCache       adminImageCacheStats                   `json:"image_cache"`
	RetentionBacklog adminRetentionBacklog                  `json:"retention_backlog"`
	KPIs             adminPlatformKPIs                      `json:"kpis"`
	UploadSessions   domain.UploadSessionOperationalMetrics `json:"upload_sessions"`
	Activity         []adminActivityPeriod                  `json:"activity"`
	UsageLast24h     []adminUsageBucket                     `json:"usage_last_24h"`
	ToolUsage7d      []adminToolUsageRecord                 `json:"tool_usage_7d"`
	Workers          []adminWorkerRecord                    `json:"workers"`
	TopUsers         []adminUserSummary                     `json:"top_users"`
	ResourceProjects []adminResourceOwnerSummary            `json:"resource_projects"`
	RecentIssues     []adminIssueRecord                     `json:"recent_issues"`
}

type adminResourceOwnerSummary struct {
	ID           string `json:"id"`
	Uploads      int    `json:"uploads"`
	StorageBytes int64  `json:"storage_bytes"`
}

type adminUserListResponse struct {
	Count int                `json:"count"`
	Users []adminUserSummary `json:"users"`
}

type adminOrganizationListResponse struct {
	Count         int                        `json:"count"`
	Organizations []adminOrganizationSummary `json:"organizations"`
}

type adminOrganizationSummary struct {
	OrgID        string         `json:"org_id"`
	Name         string         `json:"name,omitempty"`
	Status       string         `json:"status,omitempty"`
	CreatedAt    string         `json:"created_at,omitempty"`
	UpdatedAt    string         `json:"updated_at,omitempty"`
	Metadata     domain.JSONMap `json:"metadata"`
	Uploads      int            `json:"uploads"`
	StorageBytes int64          `json:"storage_bytes"`
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

type adminUpdateUserStatusRequest struct {
	Status string `json:"status"`
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

type meUser struct {
	UserID      string `json:"user_id"`
	Email       string `json:"email,omitempty"`
	DisplayName string `json:"display_name,omitempty"`
	Role        string `json:"role,omitempty"`
	OrgID       string `json:"org_id,omitempty"`
}

type meResponse struct {
	User    meUser             `json:"user"`
	Profile domain.UserProfile `json:"profile"`
}

type updateProfileRequest struct {
	DisplayName       *string `json:"display_name"`
	Title             *string `json:"title"`
	Institution       *string `json:"institution"`
	ResearchInterests *string `json:"research_interests"`
	Bio               *string `json:"bio"`
}

func (req updateProfileRequest) apply(profile *domain.UserProfile) {
	if req.DisplayName != nil {
		profile.DisplayName = clampProfileField(*req.DisplayName, 200)
	}
	if req.Title != nil {
		profile.Title = clampProfileField(*req.Title, 200)
	}
	if req.Institution != nil {
		profile.Institution = clampProfileField(*req.Institution, 200)
	}
	if req.ResearchInterests != nil {
		profile.ResearchInterests = clampProfileField(*req.ResearchInterests, 1000)
	}
	if req.Bio != nil {
		profile.Bio = clampProfileField(*req.Bio, 4000)
	}
}

func clampProfileField(value string, max int) string {
	value = strings.TrimSpace(value)
	if len(value) > max {
		return strings.TrimSpace(value[:max])
	}
	return value
}

func userProfileFromAccount(account domain.UserAccount) domain.UserProfile {
	profile := domain.UserProfile{}
	if account.Metadata == nil {
		return profile
	}
	raw, ok := account.Metadata["profile"]
	if !ok {
		return profile
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return profile
	}
	_ = json.Unmarshal(data, &profile)
	return profile
}

func buildMeResponse(account domain.UserAccount) meResponse {
	profile := userProfileFromAccount(account)
	if strings.TrimSpace(profile.DisplayName) == "" {
		profile.DisplayName = account.DisplayName
	}
	return meResponse{
		User: meUser{
			UserID:      account.UserID,
			Email:       account.Email,
			DisplayName: firstNonEmpty(account.DisplayName, profile.DisplayName),
			Role:        account.Role,
			OrgID:       account.OrgID,
		},
		Profile: profile,
	}
}

func (deps ServerDeps) handleGetCurrentUser(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	accounts, ok := deps.Store.(accountStore)
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("account profiles are not supported by this store"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	account, found, err := accounts.GetUserByID(r.Context(), principal.UserID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if !found {
		account = domain.UserAccount{
			UserID: principal.UserID,
			Role:   principal.Role,
			OrgID:  principal.OrgID,
			Status: "active",
		}
	}
	writeJSON(w, http.StatusOK, buildMeResponse(account))
}

func (deps ServerDeps) handleUpdateCurrentUser(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	accounts, ok := deps.Store.(accountStore)
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("account profiles are not supported by this store"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	var req updateProfileRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	ctx := r.Context()
	existing, found, err := accounts.GetUserByID(ctx, principal.UserID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if !found {
		created, createErr := accounts.CreateUser(ctx, domain.CreateUserInput{
			UserID: principal.UserID,
			Role:   principal.Role,
			OrgID:  principal.OrgID,
			Status: "active",
		})
		if createErr != nil {
			writeError(w, http.StatusInternalServerError, createErr)
			return
		}
		existing = created
	}
	profile := userProfileFromAccount(existing)
	req.apply(&profile)
	account, err := accounts.UpdateUserProfile(ctx, domain.UpdateUserProfileInput{
		UserID:  principal.UserID,
		Profile: profile,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, buildMeResponse(account))
}

// handleGetRunUserProfile lets a trusted worker fetch the profile of a run's
// owner (derived from the run record, authorized by the worker token). This is
// the worker-safe equivalent of GET /v2/me, which is bound to the browser's
// WorkOS session and cannot be read with a worker token.
func (deps ServerDeps) handleGetRunUserProfile(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	accounts, hasAccounts := deps.Store.(accountStore)
	if !hasAccounts {
		writeError(w, http.StatusNotImplemented, errors.New("account profiles are not supported by this store"))
		return
	}
	runID := strings.TrimSpace(chi.URLParam(r, "run_id"))
	run, resolved := deps.runForWorkerOrUser(w, r, runID)
	if !resolved {
		return
	}
	account, found, err := accounts.GetUserByID(r.Context(), run.UserID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if !found {
		writeJSON(w, http.StatusOK, meResponse{
			User:    meUser{UserID: run.UserID},
			Profile: domain.UserProfile{},
		})
		return
	}
	writeJSON(w, http.StatusOK, buildMeResponse(account))
}

// handleEpisodicSearch powers episodic memory: the Deep Agents worker (run-anchored
// worker token) asks for the run owner's own past conversations. The owner is
// resolved server-side from the run, so one user can never read another's history.
func (deps ServerDeps) handleEpisodicSearch(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	history, ok := deps.Store.(runHistorySearchStore)
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("run history search is not supported by this store"))
		return
	}
	runID := strings.TrimSpace(chi.URLParam(r, "run_id"))
	run, resolved := deps.runForWorkerOrUser(w, r, runID)
	if !resolved {
		return
	}
	var req episodicSearchRequest
	if r.Body != nil {
		// An empty body is valid (recency-only search); ignore decode EOF.
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
			writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
			return
		}
	}
	opts := domain.RunHistorySearchOptions{
		Query:        strings.TrimSpace(req.Query),
		Limit:        req.Limit,
		ExcludeRunID: run.RunID,
	}
	if req.SinceDays > 0 {
		since := domain.Now().AddDate(0, 0, -req.SinceDays)
		opts.Since = &since
	}
	hits, err := history.SearchRunHistoryForUser(r.Context(), run.UserID, opts)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if hits == nil {
		hits = []domain.RunHistoryHit{}
	}
	writeJSON(w, http.StatusOK, episodicSearchResponse{Hits: hits})
}

// handleRunResourceSearch lets a Deep Agents run discover the run owner's
// Resources library so the agent can pull a prior dataset/image into its
// workspace and analyze it. The run OWNER is resolved server-side from the run
// (run-anchored worker token) and is the access boundary: the agent sees exactly
// the owner's READABLE catalog — the owner's own resources PLUS any shared with
// them or their org via active share grants, the SAME set the owner sees in
// their Resources page — and never another user's PRIVATE (un-shared) resources.
// The agent acts as the owner, so this is the owner's own access, not a
// cross-tenant escalation. The org from the trusted worker header only narrows
// within that readable set; it can never widen access across owners.
func (deps ServerDeps) handleRunResourceSearch(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("resource catalog is not supported by this store"))
		return
	}
	runID := strings.TrimSpace(chi.URLParam(r, "run_id"))
	run, resolved := deps.runForWorkerOrUser(w, r, runID)
	if !resolved {
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var req runResourceSearchRequest
	if r.Body != nil {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
			writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
			return
		}
	}
	limit := req.Limit
	if limit <= 0 {
		limit = 50
	}
	page, err := catalog.ListResourcesForUser(r.Context(), domain.ResourceListInput{
		UserID: run.UserID,
		OrgID:  trustedRunOrgID(run),
		Query:  strings.TrimSpace(req.Query),
		Kind:   strings.ToLower(strings.TrimSpace(req.Kind)),
		Source: strings.ToLower(strings.TrimSpace(req.Source)),
		Tags:   req.Tags,
		Status: "active",
		Limit:  clampLimit(limit, 200),
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	hits := make([]runResourceHit, 0, len(page.Resources))
	for _, resource := range page.Resources {
		hits = append(hits, runResourceHitFromRecord(resource))
	}
	writeJSON(w, http.StatusOK, runResourceSearchResponse{Resources: hits, TotalCount: page.TotalCount})
}

// handleRunResourceResolve verifies that each requested resource id is READABLE
// by the run owner (defense-in-depth before the worker copies a file into the
// sandbox) and returns its model-safe metadata. Readability is enforced via
// GetResourceForUser against the run-derived user id — the owner's own resources
// plus any shared with them — so the worker only ever stages files the owner can
// access. Ids the owner cannot read come back in "missing", never as an error
// that would leak existence.
func (deps ServerDeps) handleRunResourceResolve(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("resource catalog is not supported by this store"))
		return
	}
	runID := strings.TrimSpace(chi.URLParam(r, "run_id"))
	run, resolved := deps.runForWorkerOrUser(w, r, runID)
	if !resolved {
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var req runResourceResolveRequest
	if r.Body != nil {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
			writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
			return
		}
	}
	orgID := trustedRunOrgID(run)
	resources := []runResourceHit{}
	missing := []string{}
	seen := map[string]bool{}
	for _, raw := range req.ResourceIDs {
		id := strings.TrimSpace(raw)
		if id == "" || seen[id] {
			continue
		}
		seen[id] = true
		resource, err := catalog.GetResourceForUser(r.Context(), id, run.UserID, orgID)
		if err != nil {
			missing = append(missing, id)
			continue
		}
		resources = append(resources, runResourceHitFromRecord(resource))
	}
	writeJSON(w, http.StatusOK, runResourceResolveResponse{Resources: resources, Missing: missing})
}

type tokenUsageSummaryResponse struct {
	LifetimeInputTokens  int64  `json:"lifetime_input_tokens"`
	LifetimeOutputTokens int64  `json:"lifetime_output_tokens"`
	LifetimeTotalTokens  int64  `json:"lifetime_total_tokens"`
	PeakDailyTotal       int64  `json:"peak_daily_total"`
	LongestTaskSeconds   int64  `json:"longest_task_seconds"`
	CurrentStreakDays    int    `json:"current_streak_days"`
	LongestStreakDays    int    `json:"longest_streak_days"`
	ActiveDays           int    `json:"active_days"`
	LastActiveDay        string `json:"last_active_day,omitempty"`
}

type tokenUsageDailyPoint struct {
	Day          string `json:"day"`
	InputTokens  int64  `json:"input_tokens"`
	OutputTokens int64  `json:"output_tokens"`
	TotalTokens  int64  `json:"total_tokens"`
	RunCount     int64  `json:"run_count"`
}

type tokenUsageResponse struct {
	Days    int                       `json:"days"`
	Summary tokenUsageSummaryResponse `json:"summary"`
	Daily   []tokenUsageDailyPoint    `json:"daily"`
}

func (deps ServerDeps) handleGetTokenUsage(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	usageStore, ok := deps.Store.(usageStatsStore)
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("token usage tracking is not supported by this store"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	days := parseDaysQuery(r, 365, 730)
	ctx := r.Context()
	today := domain.Now().UTC().Truncate(24 * time.Hour)
	since := today.AddDate(0, 0, -(days - 1))

	stats, err := usageStore.GetUserTokenUsageStats(ctx, principal.UserID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	daily, err := usageStore.ListUserTokenUsageDaily(ctx, principal.UserID, since)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	longestSeconds, err := usageStore.GetUserLongestRunSeconds(ctx, principal.UserID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, buildTokenUsageResponse(stats, daily, longestSeconds, days, today))
}

func parseDaysQuery(r *http.Request, fallback int, max int) int {
	raw := strings.TrimSpace(r.URL.Query().Get("days"))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		return fallback
	}
	if value > max {
		return max
	}
	return value
}

func buildTokenUsageResponse(
	stats domain.UserTokenUsageStats,
	daily []domain.UserTokenUsageDaily,
	longestSeconds int64,
	days int,
	today time.Time,
) tokenUsageResponse {
	points := make([]tokenUsageDailyPoint, 0, len(daily))
	active := make(map[string]bool, len(daily))
	for _, record := range daily {
		key := record.Day.UTC().Format("2006-01-02")
		points = append(points, tokenUsageDailyPoint{
			Day:          key,
			InputTokens:  record.InputTokens,
			OutputTokens: record.OutputTokens,
			TotalTokens:  record.TotalTokens,
			RunCount:     record.RunCount,
		})
		if record.TotalTokens > 0 || record.RunCount > 0 {
			active[key] = true
		}
	}
	current, longest := computeActivityStreaks(active, today)
	summary := tokenUsageSummaryResponse{
		LifetimeInputTokens:  stats.InputTokens,
		LifetimeOutputTokens: stats.OutputTokens,
		LifetimeTotalTokens:  stats.TotalTokens,
		PeakDailyTotal:       stats.PeakDailyTotal,
		LongestTaskSeconds:   longestSeconds,
		CurrentStreakDays:    current,
		LongestStreakDays:    longest,
		ActiveDays:           len(active),
	}
	if stats.LastActiveDay != nil && !stats.LastActiveDay.IsZero() {
		summary.LastActiveDay = stats.LastActiveDay.UTC().Format("2006-01-02")
	}
	return tokenUsageResponse{
		Days:    days,
		Summary: summary,
		Daily:   points,
	}
}

// computeActivityStreaks derives the current and longest run of consecutive
// active days from the set of active day keys (formatted YYYY-MM-DD, UTC).
func computeActivityStreaks(active map[string]bool, today time.Time) (current int, longest int) {
	if len(active) == 0 {
		return 0, 0
	}
	today = today.UTC().Truncate(24 * time.Hour)
	dayKey := func(t time.Time) string { return t.Format("2006-01-02") }

	// The current streak ends today, or yesterday when today has no activity yet.
	anchor := today
	if !active[dayKey(anchor)] {
		anchor = anchor.AddDate(0, 0, -1)
	}
	for active[dayKey(anchor)] {
		current++
		anchor = anchor.AddDate(0, 0, -1)
	}

	parsedDays := make([]time.Time, 0, len(active))
	for key := range active {
		parsed, err := time.Parse("2006-01-02", key)
		if err != nil {
			continue
		}
		parsedDays = append(parsedDays, parsed)
	}
	sort.Slice(parsedDays, func(i, j int) bool { return parsedDays[i].Before(parsedDays[j]) })
	run := 0
	var prev time.Time
	for i, day := range parsedDays {
		if i > 0 && day.Equal(prev.AddDate(0, 0, 1)) {
			run++
		} else {
			run = 1
		}
		if run > longest {
			longest = run
		}
		prev = day
	}
	return current, longest
}

func (deps ServerDeps) handleListThreads(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	principal := deps.principalFromRequest(r, "")
	limit := clampLimit(parseLimit(r, 100), 500)
	offset := parseOffset(r)
	status := strings.TrimSpace(r.URL.Query().Get("status"))
	if status == "" {
		status = string(domain.ThreadStatusActive)
	}
	page, err := deps.Store.ListThreadsForUser(r.Context(), principal.UserID, limit, offset, status)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, listThreadsResponse{
		Count:      len(page.Threads),
		TotalCount: page.TotalCount,
		Limit:      page.Limit,
		Offset:     page.Offset,
		HasMore:    page.Offset+len(page.Threads) < page.TotalCount,
		Threads:    page.Threads,
	})
}

func (deps ServerDeps) handleCreateThread(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req createThreadRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	principal := deps.principalFromRequest(r, req.UserID)
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
	principal := deps.principalFromRequest(r, "")
	thread, err := deps.Store.GetThreadForUser(r.Context(), chi.URLParam(r, "thread_id"), principal.UserID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, thread)
}

func (deps ServerDeps) handleUpsertThread(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req upsertThreadRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	principal := deps.principalFromRequest(r, "")
	thread, err := deps.Store.UpdateThreadForUser(r.Context(), domain.UpdateThreadInput{
		ThreadID: chi.URLParam(r, "thread_id"),
		UserID:   principal.UserID,
		Title:    req.Title,
		Metadata: metadataWithPrincipal(domain.JSONMap(req.Metadata), principal),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, thread)
}

func (deps ServerDeps) handleDeleteThread(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	principal := deps.principalFromRequest(r, "")
	// True erasure, not concealment. This used to soft delete, which removed no
	// rows and left the whole transcript readable in metadata.frontend_state
	// while the dialog told the user it had been removed from storage.
	storageURIs, err := deps.Store.HardDeleteThreadForUser(r.Context(), chi.URLParam(r, "thread_id"), principal.UserID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	// Blobs are unlinked after the row deletion has committed, and failures are
	// logged rather than surfaced: the conversation is already gone, and a
	// blob-store hiccup must not tell the user their delete failed. Orphaned
	// bytes are a storage-reclamation problem, not a correctness one.
	if len(storageURIs) > 0 {
		deps.deleteArtifactBlobs(r.Context(), storageURIs)
	}
	w.WriteHeader(http.StatusNoContent)
}

func (deps ServerDeps) handleListThreadMessages(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	threadID := chi.URLParam(r, "thread_id")
	principal := deps.principalFromRequest(r, "")
	limit := parseLimitParam(r, "limit", 0)
	before := strings.TrimSpace(r.URL.Query().Get("before"))
	// Back-compat: with no pagination params, return the full thread (current behavior). With ?limit
	// (and optional ?before) the client gets a "load earlier" page + a cursor for infinite scroll-up.
	if limit <= 0 && before == "" {
		messages, err := deps.Store.ListThreadMessagesForUser(r.Context(), threadID, principal.UserID)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, threadMessagesResponse{ThreadID: threadID, Count: len(messages), Messages: messages})
		return
	}
	page, hasMore, err := deps.Store.ListThreadMessagePageForUser(r.Context(), threadID, principal.UserID, before, limit)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	resp := threadMessagesResponse{ThreadID: threadID, Count: len(page), Messages: page, HasMore: hasMore}
	if hasMore && len(page) > 0 {
		resp.NextCursor = page[0].MessageID // oldest in this page → pass as ?before for the next-older page
	}
	writeJSON(w, http.StatusOK, resp)
}

func (deps ServerDeps) handleCreateRun(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req createRunRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	principal := deps.principalFromRequest(r, req.UserID)
	evaluationProfile, valid := domain.ParseEvaluationProfile(req.EvaluationProfile)
	if !valid {
		writeError(w, http.StatusBadRequest, errors.New("evaluation_profile is not a supported protected profile"))
		return
	}
	remoteMutationIntents, valid := domain.ParseRemoteMutationIntents(req.RemoteMutationIntents)
	if !valid {
		writeError(w, http.StatusBadRequest, errors.New("remote_mutation_intents contains an unsupported operation"))
		return
	}
	if evaluationProfile != "" && len(remoteMutationIntents) > 0 {
		writeError(w, http.StatusBadRequest, errors.New("protected evaluation profiles forbid remote mutations"))
		return
	}
	if evaluationProfile != "" && !strings.EqualFold(strings.TrimSpace(principal.Role), "admin") {
		writeError(w, http.StatusForbidden, errors.New("admin role required for protected evaluation profile"))
		return
	}
	threadID := chi.URLParam(r, "thread_id")
	if _, err := deps.Store.GetThreadForUser(r.Context(), threadID, principal.UserID); err != nil {
		writeStoreError(w, err)
		return
	}
	req.FileIDs = uniqueTrimmedStringValues(req.FileIDs)
	var selectedResources []domain.ResourceRecord
	if len(req.FileIDs) > 0 {
		catalog, ok := deps.resourceCatalogStore()
		if !ok {
			writeError(w, http.StatusNotImplemented, errors.New("resource catalog is not supported by this store"))
			return
		}
		authorized, resources, err := authorizeRunFileIDs(r.Context(), catalog, principal, req.FileIDs)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		req.FileIDs = authorized
		selectedResources = resources
	}
	req.ResourceDescriptors = withAuthorizedSelectedResourceDescriptors(
		req.ResourceDescriptors, selectedResources,
	)
	runMetadata := domain.JSONMap(req.Metadata)
	if runMetadata == nil {
		runMetadata = domain.JSONMap{}
	}
	// Selected-resource capabilities are derived from the catalog above.
	// Free-form request metadata cannot mint alternate file bindings.
	delete(runMetadata, "file_ids")
	delete(runMetadata, "resource_descriptors")
	delete(runMetadata, domain.EvaluationProfileMetadataKey)
	delete(runMetadata, domain.RemoteMutationIntentsMetadataKey)
	delete(runMetadata, domain.BisqueAccountBindingMetadataKey)
	jobMetadata := domain.JSONMap{}
	if sessionID, binding, bound := deps.bisqueRunBindingFromRequest(r, principal); bound {
		runMetadata[domain.BisqueAccountBindingMetadataKey] = binding
		jobMetadata["bisque_session_id"] = sessionID
	} else if len(remoteMutationIntents) > 0 {
		// No durable BisQue account is linked, so a remote mutation cannot be
		// authorized for this run. Start the run WITHOUT the mutation capability
		// instead of failing the whole chat turn at creation — a mutating BisQue
		// tool call then returns a clean, actionable "link an account" error at
		// tool time, exactly as it did before run-create gained this gate.
		remoteMutationIntents = nil
	}
	run, err := deps.Runs.CreateRun(r.Context(), runcontrol.CreateRunRequest{
		ThreadID:              threadID,
		UserID:                principal.UserID,
		Goal:                  req.Goal,
		EvaluationProfile:     evaluationProfile,
		RemoteMutationIntents: remoteMutationIntents,
		Messages:              req.Messages,
		FileIDs:               req.FileIDs,
		ResourceURIs:          req.ResourceURIs,
		DatasetURIs:           req.DatasetURIs,
		SelectedToolNames:     req.SelectedToolNames,
		KnowledgeContext:      domain.JSONMap(req.KnowledgeContext),
		WorkflowHint:          domain.JSONMap(req.WorkflowHint),
		SelectionContext:      domain.JSONMap(req.SelectionContext),
		ReasoningMode:         req.ReasoningMode,
		Budgets:               domain.JSONMap(req.Budgets),
		Benchmark:             domain.JSONMap(req.Benchmark),
		ResourceDescriptors:   req.ResourceDescriptors,
		IdempotencyKey:        idempotencyKeyFromRequest(r, req.IdempotencyKey),
		Metadata:              metadataWithPrincipal(runMetadata, principal),
		JobMetadata:           jobMetadata,
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
	authority, authorized := deps.authorizeBisqueRequest(w, r, domain.RemoteMutationIntentBisqueUpload)
	if !authorized {
		return
	}
	if r.ContentLength > directUploadMaxBodyBytes {
		writeDirectUploadTooLarge(w)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, directUploadMaxBodyBytes)
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
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			writeDirectUploadTooLarge(w)
			return
		}
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if r.MultipartForm == nil || len(r.MultipartForm.File["files"]) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("multipart upload must include at least one files entry"))
		return
	}

	principal := authority.Principal
	projectID := strings.TrimSpace(r.FormValue("project_id"))
	uploaded := make([]uploadedFileRecord, 0, len(r.MultipartForm.File["files"]))
	for _, header := range r.MultipartForm.File["files"] {
		record, err := saveUploadedFile(root, header, principal, projectID)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		if authority.Worker {
			record.TrustedSourceRunID = authority.Run.RunID
		}
		if err := deps.enforceResourceQuota(r.Context(), principal, record.ProjectID, record.SizeBytes); err != nil {
			_ = removeUploadedFile(root, record.FileID)
			writeResourceQuotaError(w, err)
			return
		}
		if err := deps.catalogUploadedFile(r.Context(), root, record, "resource.uploaded"); err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		uploaded = append(uploaded, record)
	}
	writeJSON(w, http.StatusOK, uploadFilesResponse{FileCount: len(uploaded), Uploaded: uploaded})
}

func (deps ServerDeps) handleCreateUploadSession(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	sessions, ok := deps.uploadSessionStore()
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("upload sessions are not configured"))
		return
	}
	var req createUploadSessionRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if len(req.Files) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("upload session must include at least one file"))
		return
	}
	limits := defaultUploadSessionLimits()
	if len(req.Files) > limits.MaxFilesPerSession {
		writeError(w, http.StatusBadRequest, fmt.Errorf("upload session cannot include more than %d files", limits.MaxFilesPerSession))
		return
	}
	if req.TotalBytes < 0 {
		writeError(w, http.StatusBadRequest, errors.New("total_bytes cannot be negative"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	now := domain.Now()
	totalBytes := req.TotalBytes
	computedTotal := int64(0)
	seenTokens := map[string]bool{}
	fileInputs := make([]domain.UpsertUploadSessionFileInput, 0, len(req.Files))
	for _, fileReq := range req.Files {
		fileToken := safePathToken(fileReq.FileToken)
		if fileToken == "" {
			writeError(w, http.StatusBadRequest, errors.New("file_token is required"))
			return
		}
		if seenTokens[fileToken] {
			writeError(w, http.StatusBadRequest, fmt.Errorf("duplicate file_token %q", fileToken))
			return
		}
		seenTokens[fileToken] = true
		if fileReq.SizeBytes < 0 {
			writeError(w, http.StatusBadRequest, errors.New("file size cannot be negative"))
			return
		}
		declaredSHA := strings.ToLower(strings.TrimSpace(fileReq.DeclaredSHA256))
		if declaredSHA != "" && !isSHA256Hex(declaredSHA) {
			writeError(w, http.StatusBadRequest, errors.New("declared_sha256 must be a sha256 hex digest"))
			return
		}
		originalName := safeOriginalFilename(fileReq.OriginalName)
		computedTotal += fileReq.SizeBytes
		fileInputs = append(fileInputs, domain.UpsertUploadSessionFileInput{
			FileToken:      fileToken,
			OriginalName:   originalName,
			RelativePath:   strings.TrimSpace(fileReq.RelativePath),
			ContentType:    contentTypeForUpload(originalName, fileReq.ContentType),
			SizeBytes:      fileReq.SizeBytes,
			DeclaredSHA256: declaredSHA,
			Status:         "pending",
			CreatedAt:      now,
			UpdatedAt:      now,
			Metadata: domain.JSONMap{
				"source": "resumable_upload_v2",
			},
		})
	}
	if totalBytes == 0 {
		totalBytes = computedTotal
	}
	if computedTotal > 0 && totalBytes < computedTotal {
		writeError(w, http.StatusBadRequest, errors.New("total_bytes cannot be smaller than declared file sizes"))
		return
	}
	// Detect directory-format bundles (OME-Zarr) from the file relative paths: a group of
	// files sharing a special-format top segment (scan.ome.zarr/...) is committed as ONE
	// resource by finalize-bundle, not one per member file.
	sessionMetadata := domain.JSONMap{
		"source":     "resumable_upload_v2",
		"file_count": len(fileInputs),
	}
	bundles := detectSessionBundles(fileInputs, func() string { return domain.NewID("file") })
	if len(bundles) > 0 {
		sessionMetadata["bundles"] = bundleMetadataValue(bundles)
	}
	idempotencyKey := idempotencyKeyFromRequest(r, req.IdempotencyKey)
	if idempotencyKey != "" {
		existing, err := sessions.GetUploadSessionByIdempotencyKeyForUser(r.Context(), idempotencyKey, principal.UserID, principal.OrgID)
		if err == nil {
			// A still-in-flight session (active/paused) replays so the client resumes it.
			// A TERMINAL session (completed/canceled) replays ONLY if its committed result
			// is still live; otherwise the user is re-uploading a file they finished and
			// then deleted (or a canceled attempt), which must start a FRESH upload — not
			// 409 against the dead session (the reported "upload session is completed" bug).
			if !uploadSessionTerminal(existing.Status) || deps.terminalUploadSessionReplayable(r.Context(), sessions, existing) {
				existingFiles, err := sessions.ListUploadSessionFiles(r.Context(), existing.SessionID)
				if err != nil {
					writeStoreError(w, err)
					return
				}
				if !uploadSessionManifestMatches(existing, existingFiles, fileInputs, totalBytes, strings.TrimSpace(req.ProjectID)) {
					writeStoreError(w, fmt.Errorf("%w: upload session idempotency replay does not match original manifest", store.ErrConflict))
					return
				}
				response, err := uploadSessionState(r.Context(), sessions, existing)
				if err != nil {
					writeStoreError(w, err)
					return
				}
				writeJSON(w, http.StatusOK, response)
				return
			}
			// Supersede the dead terminal session: free its idempotency-key slot so the
			// fresh session below can claim it (the partial unique index ignores empty keys).
			if err := sessions.ClearUploadSessionIdempotencyKey(r.Context(), existing.SessionID); err != nil {
				writeStoreError(w, err)
				return
			}
		} else if !errors.Is(err, store.ErrNotFound) {
			writeStoreError(w, err)
			return
		}
	}
	session, err := sessions.CreateUploadSession(r.Context(), domain.CreateUploadSessionInput{
		OwnerUserID:        principal.UserID,
		OwnerOrgID:         principal.OrgID,
		OwnerRole:          principal.Role,
		ProjectID:          strings.TrimSpace(req.ProjectID),
		SourceType:         "upload",
		Status:             "active",
		TotalBytes:         totalBytes,
		IdempotencyKey:     idempotencyKey,
		BrowserFingerprint: strings.TrimSpace(req.BrowserFingerprint),
		CreatedAt:          now,
		UpdatedAt:          now,
		Metadata:           sessionMetadata,
	})
	if err != nil {
		// Collision-tolerant: a concurrent create with the same idempotency key (two
		// tabs, a double-submit, a retry after a lost response) may have won the partial
		// unique-index race. Resolve to that winner and replay it instead of failing the
		// loser with a 409 — re-uploading a file must succeed, not error.
		if idempotencyKey != "" && errors.Is(err, store.ErrConflict) {
			if winner, lookupErr := sessions.GetUploadSessionByIdempotencyKeyForUser(r.Context(), idempotencyKey, principal.UserID, principal.OrgID); lookupErr == nil {
				response, stateErr := uploadSessionState(r.Context(), sessions, winner)
				if stateErr != nil {
					writeStoreError(w, stateErr)
					return
				}
				writeJSON(w, http.StatusOK, response)
				return
			}
		}
		writeStoreError(w, err)
		return
	}
	for index := range fileInputs {
		fileInputs[index].SessionID = session.SessionID
	}
	files, err := createUploadSessionFiles(r.Context(), sessions, fileInputs)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	deps.recordUploadSessionEvent(r.Context(), session, principal, "upload_session.created", domain.JSONMap{
		"file_count": len(files),
	})
	writeJSON(w, http.StatusCreated, uploadSessionResponse{
		Session: session,
		Files:   files,
		Chunks:  []domain.UploadChunkRecord{},
		Events:  []domain.UploadSessionEventRecord{},
		Limits:  defaultUploadSessionLimits(),
	})
}

func (deps ServerDeps) handleGetUploadSession(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	sessions, ok := deps.uploadSessionStore()
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("upload sessions are not configured"))
		return
	}
	session, err := deps.uploadSessionForRequest(r, sessions)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	response, err := uploadSessionState(r.Context(), sessions, session)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func (deps ServerDeps) handleUploadSessionChunk(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	sessions, ok := deps.uploadSessionStore()
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("upload sessions are not configured"))
		return
	}
	session, err := deps.uploadSessionForRequest(r, sessions)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if uploadSessionWriteBlocked(session.Status) {
		writeError(w, http.StatusConflict, fmt.Errorf("upload session is %s", session.Status))
		return
	}
	fileToken := safePathToken(chi.URLParam(r, "file_token"))
	chunkIndex, err := parseUploadChunkIndex(chi.URLParam(r, "chunk_index"))
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	offset, err := parseUploadOffsetHeader(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	declaredChunkSHA := strings.ToLower(strings.TrimSpace(r.Header.Get("X-Upload-Chunk-Sha256")))
	if !isSHA256Hex(declaredChunkSHA) {
		writeError(w, http.StatusBadRequest, errors.New("X-Upload-Chunk-Sha256 must be a sha256 hex digest"))
		return
	}
	file, err := uploadSessionFileForToken(r.Context(), sessions, session.SessionID, fileToken)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if file.Status == "completed" {
		writeError(w, http.StatusConflict, errors.New("upload session file is already completed"))
		return
	}
	remainingBytes, err := uploadSessionRemainingChunkBytes(file, offset)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if r.ContentLength > remainingBytes {
		writeUploadChunkTooLarge(w, remainingBytes)
		return
	}

	chunkDir := uploadSessionFileStagingDir(root, session.SessionID, fileToken)
	if err := os.MkdirAll(chunkDir, 0o755); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	target := uploadSessionChunkPath(root, session.SessionID, fileToken, chunkIndex)
	if !pathIsUnderRoot(chunkDir, target) {
		writeError(w, http.StatusBadRequest, errUnsafeArtifactPath)
		return
	}
	tmp := target + ".tmp-" + safePathToken(domain.NewID("chunk"))
	destination, err := os.Create(tmp)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	hasher := sha256.New()
	chunkBody := http.MaxBytesReader(w, r.Body, remainingBytes)
	defer chunkBody.Close()
	size, copyErr := copyWithPooledBuffer(io.MultiWriter(destination, hasher), chunkBody)
	closeErr := destination.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		var maxBytesErr *http.MaxBytesError
		if errors.As(copyErr, &maxBytesErr) {
			writeUploadChunkTooLarge(w, remainingBytes)
			return
		}
		writeError(w, http.StatusInternalServerError, copyErr)
		return
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		writeError(w, http.StatusInternalServerError, closeErr)
		return
	}
	actualSHA := hex.EncodeToString(hasher.Sum(nil))
	if actualSHA != declaredChunkSHA {
		_ = os.Remove(tmp)
		if err := recordFailedUploadChunk(r.Context(), sessions, session, file, chunkIndex, offset, size, actualSHA, "chunk checksum mismatch"); err != nil {
			writeStoreError(w, err)
			return
		}
		writeError(w, http.StatusBadRequest, errors.New("chunk checksum mismatch"))
		return
	}
	if file.SizeBytes > 0 && offset+size > file.SizeBytes {
		_ = os.Remove(tmp)
		if err := recordFailedUploadChunk(r.Context(), sessions, session, file, chunkIndex, offset, size, actualSHA, "chunk exceeds declared file size"); err != nil {
			writeStoreError(w, err)
			return
		}
		writeError(w, http.StatusBadRequest, errors.New("chunk exceeds declared file size"))
		return
	}
	createdTarget, err := installUploadChunkTemp(tmp, target, actualSHA, size)
	if err != nil {
		_ = os.Remove(tmp)
		if errors.Is(err, store.ErrConflict) {
			writeStoreError(w, err)
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	now := domain.Now()
	chunk, err := sessions.UpsertUploadChunk(r.Context(), domain.UpsertUploadChunkInput{
		SessionID:  session.SessionID,
		FileToken:  file.FileToken,
		ChunkIndex: chunkIndex,
		Offset:     offset,
		SizeBytes:  size,
		SHA256:     actualSHA,
		Status:     "verified",
		StorageURI: fileStorageURI(target),
		ReceivedAt: now,
		VerifiedAt: now,
		Metadata: domain.JSONMap{
			"source": "resumable_upload_v2",
		},
	})
	if err != nil {
		if createdTarget && errors.Is(err, store.ErrConflict) {
			_ = os.Remove(target)
		}
		writeStoreError(w, err)
		return
	}
	if file.Status == "pending" {
		file.Status = "uploading"
		file.UpdatedAt = now
		updated, err := sessions.UpsertUploadSessionFile(r.Context(), uploadSessionFileInput(file))
		if err != nil {
			writeStoreError(w, err)
			return
		}
		file = updated
	}
	session, err = sessions.GetUploadSessionForUser(r.Context(), session.SessionID, session.OwnerUserID, session.OwnerOrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, uploadChunkResponse{Session: session, File: file, Chunk: chunk})
}

func (deps ServerDeps) handleCompleteUploadSessionFile(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	sessions, ok := deps.uploadSessionStore()
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("upload sessions are not configured"))
		return
	}
	session, err := deps.uploadSessionForRequest(r, sessions)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	fileToken := safePathToken(chi.URLParam(r, "file_token"))
	file, err := uploadSessionFileForToken(r.Context(), sessions, session.SessionID, fileToken)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	// A genuinely-completed file returns its committed resource (idempotent re-complete),
	// checked BEFORE the write-block — otherwise completing an already-completed file in a
	// completed session would 409 instead of returning the resource the client asked for.
	if deps.respondUploadFileAlreadyCompleted(w, r, root, session, file, principal) {
		return
	}
	if uploadSessionWriteBlocked(session.Status) {
		writeError(w, http.StatusConflict, fmt.Errorf("upload session is %s", session.Status))
		return
	}
	// Serialize concurrent completions of the SAME (session, file). A client retry or
	// double-submit would otherwise both pass the not-completed check above and each mint
	// a distinct resourceID for one upload — duplicate catalog entry, leaked bytes on disk,
	// and double-charged quota. The lock is held across the whole commit, so the loser
	// observes the winner's committed state below and returns that resource (idempotent
	// completion) instead of duplicating it. Per-file key: distinct files never contend.
	unlock := uploadCompletionLocks.Lock(session.SessionID + "\x00" + fileToken)
	defer unlock()
	if file, err = uploadSessionFileForToken(r.Context(), sessions, session.SessionID, fileToken); err != nil {
		writeStoreError(w, err)
		return
	}
	if deps.respondUploadFileAlreadyCompleted(w, r, root, session, file, principal) {
		return
	}
	chunks, err := sessions.ListUploadChunks(r.Context(), session.SessionID, file.FileToken)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if err := validateCompleteUploadChunks(file, chunks); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if err := deps.enforceResourceQuota(r.Context(), principal, session.ProjectID, file.SizeBytes); err != nil {
		writeResourceQuotaError(w, err)
		return
	}

	resourceID := domain.NewID("file")
	originalName := safeOriginalFilename(file.OriginalName)
	target := filepath.Join(root, resourceID+"__"+originalName)
	if !pathIsUnderRoot(root, target) {
		writeError(w, http.StatusInternalServerError, errUnsafeArtifactPath)
		return
	}
	tmp := target + ".tmp-" + safePathToken(domain.NewID("commit"))
	destination, err := os.Create(tmp)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	hasher := sha256.New()
	var written int64
	for _, chunk := range chunks {
		chunkPath, err := uploadSessionChunkLocalPath(root, session.SessionID, chunk)
		if err != nil {
			_ = destination.Close()
			_ = os.Remove(tmp)
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		source, err := os.Open(chunkPath)
		if err != nil {
			_ = destination.Close()
			_ = os.Remove(tmp)
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		chunkHasher := sha256.New()
		n, copyErr := copyWithPooledBuffer(io.MultiWriter(destination, hasher, chunkHasher), source)
		closeErr := source.Close()
		if copyErr != nil {
			_ = destination.Close()
			_ = os.Remove(tmp)
			writeError(w, http.StatusInternalServerError, copyErr)
			return
		}
		if closeErr != nil {
			_ = destination.Close()
			_ = os.Remove(tmp)
			writeError(w, http.StatusInternalServerError, closeErr)
			return
		}
		if n != chunk.SizeBytes {
			_ = destination.Close()
			_ = os.Remove(tmp)
			writeError(w, http.StatusBadRequest, errors.New("chunk size changed before commit"))
			return
		}
		if chunk.SHA256 != "" && !strings.EqualFold(chunk.SHA256, hex.EncodeToString(chunkHasher.Sum(nil))) {
			_ = destination.Close()
			_ = os.Remove(tmp)
			writeError(w, http.StatusBadRequest, errors.New("chunk checksum mismatch before commit"))
			return
		}
		written += n
	}
	closeErr := destination.Close()
	if closeErr != nil {
		_ = os.Remove(tmp)
		writeError(w, http.StatusInternalServerError, closeErr)
		return
	}
	computedSHA := hex.EncodeToString(hasher.Sum(nil))
	if file.DeclaredSHA256 != "" && !strings.EqualFold(file.DeclaredSHA256, computedSHA) {
		_ = os.Remove(tmp)
		writeError(w, http.StatusBadRequest, errors.New("completed file checksum mismatch"))
		return
	}
	if written != file.SizeBytes {
		_ = os.Remove(tmp)
		writeError(w, http.StatusBadRequest, errors.New("completed file size mismatch"))
		return
	}
	// Bundle member (a file inside an OME-Zarr folder upload): move it into the bundle tree
	// at {root}/bundles/{bundleID}/{relative_path} and DO NOT catalog/dedup it — the whole
	// bundle is committed as one resource by finalize-bundle. The session file is marked
	// completed against the shared bundle id so finalize knows its members are in.
	if dest, bundle, isBundle := bundleMemberTarget(root, session, file); isBundle {
		// Finalization snapshots the complete member set and authors one tree
		// identity. Serialize the move + durable completed state with that
		// snapshot so a late member can never mutate an already-cataloged tree.
		bundleUnlock := uploadBundleLocks.Lock(session.SessionID + "\x00" + bundle.ID)
		defer bundleUnlock()
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			_ = os.Remove(tmp)
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		if err := os.Rename(tmp, dest); err != nil {
			_ = os.Remove(tmp)
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		session, completedFile, err := completeUploadSessionStoreState(r.Context(), sessions, session, file, bundle.ID, computedSHA)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		deps.recordUploadSessionFileCompleted(r.Context(), session, completedFile, principal)
		writeJSON(w, http.StatusOK, uploadSessionFileCompleteResponse{
			Session: session,
			File:    completedFile,
			Resource: uploadedFileRecord{
				FileID:       bundle.ID,
				OriginalName: bundle.Name,
				ContentType:  "application/octet-stream",
				SizeBytes:    file.SizeBytes,
				SHA256:       computedSHA,
				ProjectID:    session.ProjectID,
				Principal:    principal.record(),
			},
		})
		return
	}
	// Serialize the dedup-check-then-commit for identical content across DIFFERENT
	// sessions. The per-(session,file) lock above does not cover this: two re-uploads of
	// the same bytes use different session IDs, so without a content lock both could miss
	// dedup and mint duplicate resources. Keyed on owner+sha+size, held through catalog.
	contentUnlock := uploadContentLocks.Lock(principal.UserID + "\x00" + principal.OrgID + "\x00" + computedSHA + "\x00" + strconv.FormatInt(file.SizeBytes, 10))
	defer contentUnlock()
	if dedupe, ok := deps.Store.(resourceDedupeStore); ok {
		existing, err := dedupe.FindActiveResourceByShaForUser(r.Context(), principal.UserID, principal.OrgID, computedSHA, file.SizeBytes)
		if err == nil {
			_ = os.Remove(tmp)
			record := deps.uploadedFileRecordFromCatalog(root, existing)
			session, completedFile, err := completeUploadSessionStoreState(r.Context(), sessions, session, file, record.FileID, computedSHA)
			if err != nil {
				writeStoreError(w, err)
				return
			}
			deps.recordUploadSessionFileCompleted(r.Context(), session, completedFile, principal)
			deps.recordResourceEvent(r.Context(), record.FileID, principal, "resource.upload_deduplicated", domain.JSONMap{
				"upload_session_id": session.SessionID,
				"file_token":        file.FileToken,
			})
			writeJSON(w, http.StatusOK, uploadSessionFileCompleteResponse{Session: session, File: completedFile, Resource: record})
			return
		}
		if !errors.Is(err, store.ErrNotFound) {
			_ = os.Remove(tmp)
			writeStoreError(w, err)
			return
		}
	}
	if err := os.Rename(tmp, target); err != nil {
		_ = os.Remove(tmp)
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	metadata := uploadMetadataRecord{
		Principal:  principal.record(),
		SourceURI:  uploadSessionSourceURI(session.SessionID, file.FileToken),
		SourceType: "upload",
		ProjectID:  session.ProjectID,
	}
	if err := writeUploadMetadataRecord(root, resourceID, metadata); err != nil {
		_ = os.Remove(target)
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	info, err := os.Stat(target)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	record := uploadedFileRecord{
		FileID:       resourceID,
		OriginalName: originalName,
		ContentType:  contentTypeForUpload(originalName, file.ContentType),
		SizeBytes:    file.SizeBytes,
		SHA256:       computedSHA,
		CreatedAt:    info.ModTime().UTC().Format(time.RFC3339Nano),
		SourceURI:    metadata.SourceURI,
		ProjectID:    session.ProjectID,
		PreviewURL:   "/v2/uploads/" + url.PathEscape(resourceID) + "/preview",
		Principal:    principal.record(),
	}
	if err := deps.catalogUploadedFileAtPath(r.Context(), root, target, record, "resource.uploaded"); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	session, completedFile, err := completeUploadSessionStoreState(r.Context(), sessions, session, file, resourceID, computedSHA)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	deps.recordUploadSessionFileCompleted(r.Context(), session, completedFile, principal)
	writeJSON(w, http.StatusOK, uploadSessionFileCompleteResponse{Session: session, File: completedFile, Resource: record})
}

// handleFinalizeUploadBundle commits the directory-format bundle(s) of an upload session
// (OME-Zarr folder uploads) as catalog resources — ONE resource per bundle, pointing at its
// reconstructed tree under {root}/bundles/{id}/{name}. Members were already written there by
// the per-file complete handler. Idempotent (UpsertResource on a stable bundle id).
func (deps ServerDeps) handleFinalizeUploadBundle(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	sessions, ok := deps.uploadSessionStore()
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("upload sessions are not configured"))
		return
	}
	session, err := deps.uploadSessionForRequest(r, sessions)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	bundles := sessionBundles(session)
	if len(bundles) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("upload session has no directory-format bundles"))
		return
	}
	// Acquire every bundle lock in deterministic order. Member completion holds
	// exactly one of these locks, so this cannot deadlock and makes the following
	// all-bundle preflight atomic with respect to final member installation.
	bundleTops := make([]string, 0, len(bundles))
	for top := range bundles {
		bundleTops = append(bundleTops, top)
	}
	sort.Strings(bundleTops)
	unlockBundles := make([]func(), 0, len(bundleTops))
	for _, top := range bundleTops {
		unlockBundles = append(unlockBundles, uploadBundleLocks.Lock(session.SessionID+"\x00"+bundles[top].ID))
	}
	defer func() {
		for index := len(unlockBundles) - 1; index >= 0; index-- {
			unlockBundles[index]()
		}
	}()
	files, err := sessions.ListUploadSessionFiles(r.Context(), session.SessionID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	type preparedBundle struct {
		info     bundleInfo
		dir      string
		identity bundleTreeIdentity
	}
	prepared := make([]preparedBundle, 0, len(bundleTops))
	for _, top := range bundleTops {
		bundle := bundles[top]
		dir := bundleDirPath(root, bundle)
		identity, identityErr := finalizedBundleTreeIdentity(dir, top, bundle, files)
		if identityErr != nil {
			writeError(w, http.StatusConflict, fmt.Errorf("directory-format bundle %q is not finalizable: %w", bundle.Name, identityErr))
			return
		}
		prepared = append(prepared, preparedBundle{info: bundle, dir: dir, identity: identity})
	}

	now := domain.Now().Format(time.RFC3339Nano)
	results := make([]uploadedFileRecord, 0, len(prepared))
	newlyCataloged := 0
	catalog, hasCatalog := deps.resourceCatalogStore()
	for _, bundle := range prepared {
		b := bundle.info
		if hasCatalog {
			existing, lookupErr := catalog.GetResourceForUser(
				r.Context(), b.ID, principal.UserID, principal.OrgID,
			)
			if lookupErr == nil {
				existingPath, pathErr := resolveCatalogResourcePath(root, existing)
				if pathErr != nil || filepath.Clean(existingPath) != filepath.Clean(bundle.dir) ||
					existing.Status != "active" || existing.OriginalName != b.Name ||
					existing.SizeBytes != bundle.identity.SizeBytes ||
					!strings.EqualFold(existing.SHA256, bundle.identity.ManifestSHA256) {
					writeError(w, http.StatusConflict, fmt.Errorf(
						"directory-format bundle %q is already cataloged with a different immutable identity", b.Name,
					))
					return
				}
				results = append(results, deps.uploadedFileRecordFromCatalog(root, existing))
				continue
			}
			if !errors.Is(lookupErr, store.ErrNotFound) {
				writeStoreError(w, lookupErr)
				return
			}
		}
		// Bundle members are retained before they become one catalog resource.
		// Enforce quota again on the final physical tree size so the generated
		// manifest overhead and the aggregate member bytes are both charged.
		if err := deps.enforceResourceQuota(
			r.Context(), principal, session.ProjectID, bundle.identity.SizeBytes,
		); err != nil {
			writeResourceQuotaError(w, err)
			return
		}
		record := uploadedFileRecord{
			FileID:       b.ID,
			OriginalName: b.Name,
			ContentType:  "application/octet-stream",
			SizeBytes:    bundle.identity.SizeBytes,
			SHA256:       bundle.identity.ManifestSHA256,
			CreatedAt:    now,
			SourceURI:    uploadSessionSourceURI(session.SessionID, b.ID),
			ProjectID:    session.ProjectID,
			PreviewURL:   "/v2/uploads/" + url.PathEscape(b.ID) + "/preview",
			Principal:    principal.record(),
		}
		if err := deps.catalogUploadedFileAtPath(r.Context(), root, bundle.dir, record, "resource.uploaded"); err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		results = append(results, record)
		newlyCataloged++
	}
	if len(results) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("no bundle contents found to finalize"))
		return
	}
	if newlyCataloged > 0 {
		deps.recordUploadSessionEvent(r.Context(), session, principal, "upload_session.bundle_finalized", domain.JSONMap{
			"bundle_count":         newlyCataloged,
			"tree_identity_schema": resourceTreeIdentitySchema,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"bundles": results})
}

// keyedMutex is a set of mutexes addressed by string key, used to serialize work on
// the same logical resource without serializing unrelated resources. Map entries are
// reference-counted and removed when idle, so a high-cardinality key space (one per
// upload file) does not leak memory.
type keyedMutex struct {
	mu sync.Mutex
	m  map[string]*keyedMutexEntry
}

type keyedMutexEntry struct {
	mu   sync.Mutex
	refs int
}

// Lock blocks until the mutex for key is held and returns the unlock function. The
// returned func must be called exactly once (typically via defer).
func (k *keyedMutex) Lock(key string) func() {
	k.mu.Lock()
	if k.m == nil {
		k.m = make(map[string]*keyedMutexEntry)
	}
	e := k.m[key]
	if e == nil {
		e = &keyedMutexEntry{}
		k.m[key] = e
	}
	e.refs++
	k.mu.Unlock()

	e.mu.Lock()
	return func() {
		e.mu.Unlock()
		k.mu.Lock()
		e.refs--
		if e.refs == 0 {
			delete(k.m, key)
		}
		k.mu.Unlock()
	}
}

// uploadCompletionLocks serializes /complete calls per (session, file) so a client
// retry or double-submit can't mint two resources for one uploaded file.
var uploadCompletionLocks keyedMutex

// uploadBundleLocks serialize member installation with the final closure/identity
// snapshot for one directory-format resource.
var uploadBundleLocks keyedMutex

// uploadContentLocks serializes the dedup-check-then-commit for identical content
// (owner+sha+size) across different upload sessions, so two concurrent re-uploads of the
// same bytes cannot both miss dedup and create duplicate resources.
var uploadContentLocks keyedMutex

// respondUploadFileAlreadyCompleted writes the already-committed resource for a
// completed upload file and reports whether it handled the response. Used both on
// the lock-free fast path (a client polling a finished file) and again under the
// completion lock (the loser of a completion race observes the committed state).
func (deps ServerDeps) respondUploadFileAlreadyCompleted(w http.ResponseWriter, r *http.Request, root string, session domain.UploadSessionRecord, file domain.UploadSessionFileRecord, principal requestPrincipal) bool {
	if file.Status != "completed" {
		return false
	}
	if strings.TrimSpace(file.ResourceID) == "" {
		writeError(w, http.StatusConflict, errors.New("upload session file is already completed without a committed resource"))
		return true
	}
	catalog, ok := deps.Store.(resourceCatalogStore)
	if !ok {
		writeError(w, http.StatusConflict, errors.New("upload session file is already completed"))
		return true
	}
	resource, err := catalog.GetResourceForUser(r.Context(), file.ResourceID, principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return true
	}
	writeJSON(w, http.StatusOK, uploadSessionFileCompleteResponse{
		Session:  session,
		File:     file,
		Resource: deps.uploadedFileRecordFromCatalog(root, resource),
	})
	return true
}

func (deps ServerDeps) handleCancelUploadSession(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	sessions, ok := deps.uploadSessionStore()
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("upload sessions are not configured"))
		return
	}
	session, err := deps.uploadSessionForRequest(r, sessions)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if session.Status == "canceled" || session.Status == "completed" {
		writeError(w, http.StatusConflict, fmt.Errorf("upload session is %s", session.Status))
		return
	}
	received, verified, committed, _, err := uploadSessionTotals(r.Context(), sessions, session.SessionID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	now := domain.Now()
	session, err = sessions.UpdateUploadSession(r.Context(), domain.UpdateUploadSessionInput{
		SessionID:      session.SessionID,
		OwnerUserID:    session.OwnerUserID,
		OwnerOrgID:     session.OwnerOrgID,
		Status:         "canceled",
		BytesReceived:  received,
		BytesVerified:  verified,
		BytesCommitted: committed,
		Error:          "canceled by user",
		UpdatedAt:      now,
		Metadata:       session.Metadata,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	deps.recordUploadSessionEvent(r.Context(), session, deps.principalFromRequest(r, ""), "upload_session.canceled", nil)
	_ = os.RemoveAll(uploadSessionStagingRoot(root, session.SessionID))
	response, err := uploadSessionState(r.Context(), sessions, session)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func (deps ServerDeps) handlePauseUploadSession(w http.ResponseWriter, r *http.Request) {
	deps.handleUploadSessionLifecycle(w, r, "paused", "paused by user", "upload_session.paused")
}

func (deps ServerDeps) handleResumeUploadSession(w http.ResponseWriter, r *http.Request) {
	deps.handleUploadSessionLifecycle(w, r, "active", "", "upload_session.resumed")
}

func (deps ServerDeps) handleUploadSessionLifecycle(w http.ResponseWriter, r *http.Request, status string, errorText string, eventType string) {
	if !deps.ready(w) {
		return
	}
	sessions, ok := deps.uploadSessionStore()
	if !ok {
		writeError(w, http.StatusNotImplemented, errors.New("upload sessions are not configured"))
		return
	}
	session, err := deps.uploadSessionForRequest(r, sessions)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if session.Status == "canceled" || session.Status == "completed" {
		writeError(w, http.StatusConflict, fmt.Errorf("upload session is %s", session.Status))
		return
	}
	received, verified, committed, _, err := uploadSessionTotals(r.Context(), sessions, session.SessionID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	session, err = sessions.UpdateUploadSession(r.Context(), domain.UpdateUploadSessionInput{
		SessionID:      session.SessionID,
		OwnerUserID:    session.OwnerUserID,
		OwnerOrgID:     session.OwnerOrgID,
		Status:         status,
		BytesReceived:  received,
		BytesVerified:  verified,
		BytesCommitted: committed,
		Error:          errorText,
		UpdatedAt:      domain.Now(),
		Metadata:       session.Metadata,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	deps.recordUploadSessionEvent(r.Context(), session, deps.principalFromRequest(r, ""), eventType, nil)
	response, err := uploadSessionState(r.Context(), sessions, session)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func uploadSessionWriteBlocked(status string) bool {
	switch status {
	case "paused", "canceled", "completed":
		return true
	default:
		return false
	}
}

// uploadSessionTerminal reports whether a session has reached a final state. Paused is
// NOT terminal — it is resumable, so its idempotency key must stay findable for resume.
func uploadSessionTerminal(status string) bool {
	switch status {
	case "completed", "canceled":
		return true
	default:
		return false
	}
}

// terminalUploadSessionReplayable reports whether a terminal session's committed result
// is still live, so an idempotency-key replay should return it (idempotent retry / dedup)
// rather than start a fresh upload. True only for a fully-completed session whose every
// committed file still resolves to an active resource; a canceled session, or one whose
// resource was soft-deleted, is NOT replayable and must be superseded by a fresh upload.
func (deps ServerDeps) terminalUploadSessionReplayable(ctx context.Context, sessions uploadSessionStore, session domain.UploadSessionRecord) bool {
	if session.Status != "completed" {
		return false // canceled has no committed result to replay
	}
	catalog, ok := deps.Store.(resourceCatalogStore)
	if !ok {
		return false
	}
	files, err := sessions.ListUploadSessionFiles(ctx, session.SessionID)
	if err != nil || len(files) == 0 {
		return false
	}
	for _, file := range files {
		if file.Status != "completed" || strings.TrimSpace(file.ResourceID) == "" {
			return false
		}
		if _, err := catalog.GetResourceForUser(ctx, file.ResourceID, session.OwnerUserID, session.OwnerOrgID); err != nil {
			return false // the committed resource is gone / soft-deleted -> supersede
		}
	}
	return true
}

func (deps ServerDeps) uploadSessionForRequest(r *http.Request, sessions uploadSessionStore) (domain.UploadSessionRecord, error) {
	principal := deps.principalFromRequest(r, "")
	return sessions.GetUploadSessionForUser(r.Context(), chi.URLParam(r, "session_id"), principal.UserID, principal.OrgID)
}

func uploadSessionState(ctx context.Context, sessions uploadSessionStore, session domain.UploadSessionRecord) (uploadSessionResponse, error) {
	files, err := sessions.ListUploadSessionFiles(ctx, session.SessionID)
	if err != nil {
		return uploadSessionResponse{}, err
	}
	chunks, err := uploadSessionChunksForFiles(ctx, sessions, session.SessionID, files)
	if err != nil {
		return uploadSessionResponse{}, err
	}
	events := []domain.UploadSessionEventRecord{}
	if eventStore, ok := sessions.(uploadSessionEventStore); ok {
		events, err = eventStore.ListUploadSessionEvents(ctx, session.SessionID, 200)
		if err != nil {
			return uploadSessionResponse{}, err
		}
	}
	return uploadSessionResponse{Session: session, Files: files, Chunks: chunks, Events: events, Limits: defaultUploadSessionLimits()}, nil
}

func createUploadSessionFiles(ctx context.Context, sessions uploadSessionStore, inputs []domain.UpsertUploadSessionFileInput) ([]domain.UploadSessionFileRecord, error) {
	if batchStore, ok := sessions.(uploadSessionFileBatchStore); ok {
		return batchStore.CreateUploadSessionFiles(ctx, inputs)
	}
	files := make([]domain.UploadSessionFileRecord, 0, len(inputs))
	for _, input := range inputs {
		file, err := sessions.UpsertUploadSessionFile(ctx, input)
		if err != nil {
			return nil, err
		}
		files = append(files, file)
	}
	return files, nil
}

func defaultUploadSessionLimits() uploadSessionLimits {
	return uploadSessionLimits{
		MaxParallelFiles:   uploadSessionMaxParallelFiles,
		MaxParallelChunks:  uploadSessionMaxParallelChunks,
		MaxFilesPerSession: uploadSessionMaxFilesPerBatch,
	}
}

func uploadSessionChunksForFiles(ctx context.Context, sessions uploadSessionStore, sessionID string, files []domain.UploadSessionFileRecord) ([]domain.UploadChunkRecord, error) {
	if aggregate, ok := sessions.(uploadSessionChunkStore); ok {
		chunks, err := aggregate.ListUploadSessionChunks(ctx, sessionID)
		if err != nil {
			return nil, err
		}
		sortUploadChunks(chunks)
		return chunks, nil
	}
	chunks := make([]domain.UploadChunkRecord, 0)
	for _, file := range files {
		fileChunks, err := sessions.ListUploadChunks(ctx, sessionID, file.FileToken)
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, fileChunks...)
	}
	sortUploadChunks(chunks)
	return chunks, nil
}

func sortUploadChunks(chunks []domain.UploadChunkRecord) {
	sort.Slice(chunks, func(i, j int) bool {
		if chunks[i].FileToken == chunks[j].FileToken {
			return chunks[i].ChunkIndex < chunks[j].ChunkIndex
		}
		return chunks[i].FileToken < chunks[j].FileToken
	})
}

func copyWithPooledBuffer(dst io.Writer, src io.Reader) (int64, error) {
	bufferPtr := uploadCopyBufferPool.Get().(*[]byte)
	defer uploadCopyBufferPool.Put(bufferPtr)
	return io.CopyBuffer(pooledCopyWriter{Writer: dst}, pooledCopyReader{Reader: src}, *bufferPtr)
}

type pooledCopyReader struct {
	io.Reader
}

type pooledCopyWriter struct {
	io.Writer
}

func uploadSessionFileByToken(files []domain.UploadSessionFileRecord, fileToken string) (domain.UploadSessionFileRecord, bool) {
	fileToken = safePathToken(fileToken)
	for _, file := range files {
		if file.FileToken == fileToken {
			return file, true
		}
	}
	return domain.UploadSessionFileRecord{}, false
}

func uploadSessionFileForToken(ctx context.Context, sessions uploadSessionStore, sessionID string, fileToken string) (domain.UploadSessionFileRecord, error) {
	fileToken = safePathToken(fileToken)
	if lookup, ok := sessions.(uploadSessionFileLookupStore); ok {
		return lookup.GetUploadSessionFile(ctx, sessionID, fileToken)
	}
	files, err := sessions.ListUploadSessionFiles(ctx, sessionID)
	if err != nil {
		return domain.UploadSessionFileRecord{}, err
	}
	file, ok := uploadSessionFileByToken(files, fileToken)
	if !ok {
		return domain.UploadSessionFileRecord{}, store.ErrNotFound
	}
	return file, nil
}

func uploadSessionManifestMatches(session domain.UploadSessionRecord, existingFiles []domain.UploadSessionFileRecord, requestedFiles []domain.UpsertUploadSessionFileInput, totalBytes int64, projectID string) bool {
	if session.TotalBytes != totalBytes || strings.TrimSpace(session.ProjectID) != strings.TrimSpace(projectID) {
		return false
	}
	if len(existingFiles) != len(requestedFiles) {
		return false
	}
	existingByToken := make(map[string]domain.UploadSessionFileRecord, len(existingFiles))
	for _, file := range existingFiles {
		existingByToken[file.FileToken] = file
	}
	for _, requested := range requestedFiles {
		existing, ok := existingByToken[requested.FileToken]
		if !ok {
			return false
		}
		if strings.TrimSpace(existing.OriginalName) != strings.TrimSpace(requested.OriginalName) ||
			strings.TrimSpace(existing.RelativePath) != strings.TrimSpace(requested.RelativePath) ||
			strings.TrimSpace(existing.ContentType) != strings.TrimSpace(requested.ContentType) ||
			existing.SizeBytes != requested.SizeBytes ||
			!strings.EqualFold(strings.TrimSpace(existing.DeclaredSHA256), strings.TrimSpace(requested.DeclaredSHA256)) {
			return false
		}
	}
	return true
}

func uploadSessionRemainingChunkBytes(file domain.UploadSessionFileRecord, offset int64) (int64, error) {
	if offset > file.SizeBytes {
		return 0, errors.New("chunk offset exceeds declared file size")
	}
	return file.SizeBytes - offset, nil
}

func writeUploadChunkTooLarge(w http.ResponseWriter, remainingBytes int64) {
	writeError(w, http.StatusRequestEntityTooLarge, fmt.Errorf("upload chunk exceeds remaining declared file size of %d bytes", remainingBytes))
}

func writeDirectUploadTooLarge(w http.ResponseWriter) {
	writeError(w, http.StatusRequestEntityTooLarge, fmt.Errorf("direct upload body exceeds %d bytes; use resumable upload sessions for larger files", directUploadMaxBodyBytes))
}

func recordFailedUploadChunk(ctx context.Context, sessions uploadSessionStore, session domain.UploadSessionRecord, file domain.UploadSessionFileRecord, chunkIndex int, offset int64, size int64, actualSHA string, message string) error {
	existingChunks, err := sessions.ListUploadChunks(ctx, session.SessionID, file.FileToken)
	if err != nil {
		return err
	}
	for _, existing := range existingChunks {
		if existing.ChunkIndex != chunkIndex {
			continue
		}
		if existing.Status == "verified" || existing.Status == "received" {
			return nil
		}
	}
	now := domain.Now()
	_, err = sessions.UpsertUploadChunk(ctx, domain.UpsertUploadChunkInput{
		SessionID:  session.SessionID,
		FileToken:  file.FileToken,
		ChunkIndex: chunkIndex,
		Offset:     offset,
		SizeBytes:  size,
		SHA256:     strings.TrimSpace(actualSHA),
		Status:     "failed",
		ReceivedAt: now,
		Error:      message,
		Metadata: domain.JSONMap{
			"source":        "resumable_upload_v2",
			"failure_stage": "chunk_validation",
		},
	})
	return err
}

func installUploadChunkTemp(tmp string, target string, actualSHA string, size int64) (bool, error) {
	if err := os.Link(tmp, target); err == nil {
		_ = os.Remove(tmp)
		return true, nil
	} else if !errors.Is(err, os.ErrExist) {
		return false, err
	}
	existing, err := os.Stat(target)
	if err != nil {
		return false, err
	}
	existingSHA, err := sha256File(target)
	if err != nil {
		return false, err
	}
	_ = os.Remove(tmp)
	if existing.Size() == size && strings.EqualFold(existingSHA, actualSHA) {
		return false, nil
	}
	return false, fmt.Errorf("%w: verified upload chunk cannot be replaced with different bytes", store.ErrConflict)
}

func uploadSessionFileInput(file domain.UploadSessionFileRecord) domain.UpsertUploadSessionFileInput {
	return domain.UpsertUploadSessionFileInput{
		SessionID:      file.SessionID,
		FileToken:      file.FileToken,
		ResourceID:     file.ResourceID,
		OriginalName:   file.OriginalName,
		RelativePath:   file.RelativePath,
		ContentType:    file.ContentType,
		SizeBytes:      file.SizeBytes,
		DeclaredSHA256: file.DeclaredSHA256,
		ComputedSHA256: file.ComputedSHA256,
		Status:         file.Status,
		Error:          file.Error,
		CreatedAt:      file.CreatedAt,
		UpdatedAt:      file.UpdatedAt,
		CompletedAt:    file.CompletedAt,
		Metadata:       file.Metadata,
	}
}

func completeUploadSessionStoreState(ctx context.Context, sessions uploadSessionStore, session domain.UploadSessionRecord, file domain.UploadSessionFileRecord, resourceID string, computedSHA string) (domain.UploadSessionRecord, domain.UploadSessionFileRecord, error) {
	now := domain.Now()
	file.ResourceID = resourceID
	file.ComputedSHA256 = computedSHA
	file.Status = "completed"
	file.Error = ""
	file.UpdatedAt = now
	file.CompletedAt = now
	completedFile, err := sessions.UpsertUploadSessionFile(ctx, uploadSessionFileInput(file))
	if err != nil {
		return domain.UploadSessionRecord{}, domain.UploadSessionFileRecord{}, err
	}
	updatedSession, err := sessions.GetUploadSessionForUser(ctx, session.SessionID, session.OwnerUserID, session.OwnerOrgID)
	if err != nil {
		return domain.UploadSessionRecord{}, domain.UploadSessionFileRecord{}, err
	}
	allComplete := updatedSession.TotalBytes > 0 && updatedSession.BytesCommitted >= updatedSession.TotalBytes
	if updatedSession.TotalBytes == 0 {
		_, _, _, allComplete, err = uploadSessionTotals(ctx, sessions, session.SessionID)
		if err != nil {
			return domain.UploadSessionRecord{}, domain.UploadSessionFileRecord{}, err
		}
	}
	if !allComplete {
		return updatedSession, completedFile, nil
	}
	updatedSession, err = sessions.UpdateUploadSession(ctx, domain.UpdateUploadSessionInput{
		SessionID:      session.SessionID,
		OwnerUserID:    session.OwnerUserID,
		OwnerOrgID:     session.OwnerOrgID,
		Status:         "completed",
		BytesReceived:  updatedSession.BytesReceived,
		BytesVerified:  updatedSession.BytesVerified,
		BytesCommitted: updatedSession.BytesCommitted,
		UpdatedAt:      now,
		CompletedAt:    now,
		Metadata:       updatedSession.Metadata,
	})
	if err != nil {
		return domain.UploadSessionRecord{}, domain.UploadSessionFileRecord{}, err
	}
	return updatedSession, completedFile, nil
}

func uploadSessionTotals(ctx context.Context, sessions uploadSessionStore, sessionID string) (int64, int64, int64, bool, error) {
	if totalsStore, ok := sessions.(uploadSessionTotalsStore); ok {
		totals, err := totalsStore.GetUploadSessionTotals(ctx, sessionID)
		if err != nil {
			return 0, 0, 0, false, err
		}
		return totals.BytesReceived, totals.BytesVerified, totals.BytesCommitted, totals.AllComplete, nil
	}
	files, err := sessions.ListUploadSessionFiles(ctx, sessionID)
	if err != nil {
		return 0, 0, 0, false, err
	}
	chunks, err := uploadSessionChunksForFiles(ctx, sessions, sessionID, files)
	if err != nil {
		return 0, 0, 0, false, err
	}
	var received int64
	var verified int64
	var committed int64
	allComplete := len(files) > 0
	for _, file := range files {
		if file.Status == "completed" {
			committed += file.SizeBytes
		} else {
			allComplete = false
		}
	}
	for _, chunk := range chunks {
		if chunk.Status == "verified" || chunk.Status == "received" {
			received += chunk.SizeBytes
		}
		if chunk.Status == "verified" {
			verified += chunk.SizeBytes
		}
	}
	return received, verified, committed, allComplete, nil
}

func validateCompleteUploadChunks(file domain.UploadSessionFileRecord, chunks []domain.UploadChunkRecord) error {
	if file.SizeBytes == 0 && len(chunks) == 0 {
		return nil
	}
	if len(chunks) == 0 {
		return errors.New("upload chunks are incomplete")
	}
	sort.Slice(chunks, func(i, j int) bool {
		return chunks[i].ChunkIndex < chunks[j].ChunkIndex
	})
	expectedOffset := int64(0)
	for expectedIndex, chunk := range chunks {
		if chunk.ChunkIndex != expectedIndex || chunk.Status != "verified" || chunk.Offset != expectedOffset || chunk.SizeBytes <= 0 {
			return errors.New("upload chunks are incomplete")
		}
		expectedOffset += chunk.SizeBytes
	}
	if expectedOffset != file.SizeBytes {
		return errors.New("upload chunks are incomplete")
	}
	return nil
}

func parseUploadChunkIndex(value string) (int, error) {
	index, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || index < 0 {
		return 0, errors.New("chunk_index must be a non-negative integer")
	}
	return index, nil
}

func parseUploadOffsetHeader(r *http.Request) (int64, error) {
	value := strings.TrimSpace(r.Header.Get("X-Upload-Offset"))
	if value == "" {
		value = strings.TrimSpace(r.URL.Query().Get("offset"))
	}
	if value == "" {
		return 0, errors.New("X-Upload-Offset is required")
	}
	offset, err := strconv.ParseInt(value, 10, 64)
	if err != nil || offset < 0 {
		return 0, errors.New("X-Upload-Offset must be a non-negative integer")
	}
	return offset, nil
}

func isSHA256Hex(value string) bool {
	value = strings.TrimSpace(value)
	if len(value) != 64 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func parseResourceLifecycleStatus(r *http.Request) (string, error) {
	status := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("status")))
	switch status {
	case "", "active":
		return "active", nil
	case "deleted":
		return "deleted", nil
	default:
		return "", errors.New("status must be active or deleted")
	}
}

func (deps ServerDeps) handleListResources(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	status, err := parseResourceLifecycleStatus(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	metadataFilters, err := parseResourceMetadataFilters(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	createdAfter, createdBefore, err := parseResourceCreatedRange(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	processingStatus, err := parseResourceProcessingStatus(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	if catalog, ok := deps.resourceCatalogStore(); ok {
		if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		page, err := catalog.ListResourcesForUser(r.Context(), domain.ResourceListInput{
			UserID:           principal.UserID,
			OrgID:            principal.OrgID,
			Query:            strings.TrimSpace(r.URL.Query().Get("q")),
			Kind:             strings.ToLower(strings.TrimSpace(r.URL.Query().Get("kind"))),
			Source:           strings.ToLower(strings.TrimSpace(r.URL.Query().Get("source"))),
			ProjectID:        strings.TrimSpace(r.URL.Query().Get("project_id")),
			Sharing:          strings.ToLower(strings.TrimSpace(r.URL.Query().Get("sharing"))),
			Tags:             parseResourceTagFilters(r),
			Descriptors:      parseResourceDescriptorFilters(r),
			MetadataFilters:  metadataFilters,
			CreatedAfter:     createdAfter,
			CreatedBefore:    createdBefore,
			ProcessingStatus: processingStatus,
			Status:           status,
			Offset:           parseOffset(r),
			Limit:            clampLimit(parseLimit(r, 200), 1000),
		})
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		records := make([]resourceRecord, 0, len(page.Resources))
		for _, resource := range page.Resources {
			records = append(records, deps.resourceRecordFromCatalog(root, resource))
		}
		writeJSON(w, http.StatusOK, resourcesResponse{Count: page.TotalCount, Resources: records})
		return
	}
	if status != "active" {
		writeJSON(w, http.StatusOK, resourcesResponse{Count: 0, Resources: []resourceRecord{}})
		return
	}
	resources, err := listUploadResources(root)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	query := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("q")))
	kind := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("kind")))
	source := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("source")))
	projectID := strings.TrimSpace(r.URL.Query().Get("project_id"))
	sharing := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("sharing")))
	tagFilters := parseResourceTagFilters(r)
	filtered := resources[:0]
	for _, resource := range resources {
		if !resourceVisibleToPrincipal(resource, principal) {
			continue
		}
		if sharing != "" && sharing != "all" && sharing != "private" {
			continue
		}
		if query != "" && !resourceMatchesQuery(resource, query) {
			continue
		}
		if kind != "" && resource.ResourceKind != kind {
			continue
		}
		if source != "" && strings.ToLower(strings.TrimSpace(resource.SourceType)) != source {
			continue
		}
		if projectID != "" && resource.ProjectID != projectID {
			continue
		}
		if !resourceRecordMatchesTags(resource, tagFilters) {
			continue
		}
		if len(metadataFilters) > 0 {
			continue
		}
		if !resourceRecordMatchesCreatedRange(resource, createdAfter, createdBefore) {
			continue
		}
		filtered = append(filtered, resource)
	}
	offset := parseOffset(r)
	limit := clampLimit(parseLimit(r, 200), 1000)
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

func (deps ServerDeps) handleCreateResourceCollection(w http.ResponseWriter, r *http.Request) {
	collections, ok := deps.resourceCollectionStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
		return
	}
	var req createResourceCollectionRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	name := strings.TrimSpace(req.Name)
	if name == "" {
		writeError(w, http.StatusBadRequest, errors.New("collection name is required"))
		return
	}
	collectionType, err := normalizeResourceCollectionType(req.CollectionType)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	parentCollectionID := strings.TrimSpace(req.ParentCollectionID)
	if parentCollectionID != "" {
		// Owner-scoped parent validation. Without it any caller could nest a
		// collection under another user's folder by guessing its id (the FK
		// checks existence only), and a bad id surfaced as a raw 500.
		parent, parentErr := collections.GetResourceCollectionForUser(
			r.Context(), parentCollectionID, principal.UserID, principal.OrgID)
		if parentErr != nil {
			if errors.Is(parentErr, store.ErrNotFound) {
				writeError(w, http.StatusNotFound, errors.New("parent collection not found"))
				return
			}
			writeStoreError(w, parentErr)
			return
		}
		if parent.Status == "deleted" {
			writeError(w, http.StatusConflict, errors.New("parent collection is deleted"))
			return
		}
		if parent.CollectionType != "folder" {
			writeError(w, http.StatusBadRequest, errors.New("parent collection must be a folder"))
			return
		}
	}
	now := domain.Now()
	collection, err := collections.CreateResourceCollection(r.Context(), domain.CreateResourceCollectionInput{
		OwnerUserID:        principal.UserID,
		OwnerOrgID:         principal.OrgID,
		OwnerRole:          principal.Role,
		ProjectID:          strings.TrimSpace(req.ProjectID),
		ParentCollectionID: parentCollectionID,
		Name:               name,
		Description:        strings.TrimSpace(req.Description),
		CollectionType:     collectionType,
		Status:             "active",
		CreatedAt:          now,
		UpdatedAt:          now,
		Metadata:           mapOrEmptyJSON(req.Metadata),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, resourceCollectionResponse{Collection: collection})
}

func (deps ServerDeps) handleListResourceCollections(w http.ResponseWriter, r *http.Request) {
	collections, ok := deps.resourceCollectionStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
		return
	}
	status, err := parseResourceLifecycleStatus(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	page, err := collections.ListResourceCollectionsForUser(r.Context(), domain.ResourceCollectionListInput{
		UserID:    principal.UserID,
		OrgID:     principal.OrgID,
		Query:     strings.TrimSpace(r.URL.Query().Get("q")),
		Type:      strings.ToLower(strings.TrimSpace(r.URL.Query().Get("collection_type"))),
		ProjectID: strings.TrimSpace(r.URL.Query().Get("project_id")),
		Status:    status,
		Limit:     clampLimit(parseLimit(r, 200), 1000),
		Offset:    parseOffset(r),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, resourceCollectionsResponse{Count: page.TotalCount, Collections: page.Collections})
}

func (deps ServerDeps) handlePatchResourceCollection(w http.ResponseWriter, r *http.Request) {
	collections, ok := deps.resourceCollectionStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
		return
	}
	var req patchResourceCollectionRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	name := strings.TrimSpace(req.Name)
	if name == "" {
		writeError(w, http.StatusBadRequest, errors.New("collection name is required"))
		return
	}
	collectionID := strings.TrimSpace(chi.URLParam(r, "collection_id"))
	if collectionID == "" {
		writeError(w, http.StatusBadRequest, errors.New("collection id is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	existing, err := collections.GetResourceCollectionForUser(r.Context(), collectionID, principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if existing.Name == name {
		writeJSON(w, http.StatusOK, resourceCollectionResponse{Collection: existing})
		return
	}
	now := domain.Now()
	collection, err := collections.RenameResourceCollectionForUser(r.Context(), domain.RenameResourceCollectionInput{
		CollectionID: collectionID,
		UserID:       principal.UserID,
		OrgID:        principal.OrgID,
		Name:         name,
		UpdatedAt:    now,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	members, err := collections.ListResourcesForCollectionForUser(r.Context(), domain.ResourceCollectionResourceListInput{
		CollectionID: collection.CollectionID,
		UserID:       principal.UserID,
		OrgID:        principal.OrgID,
		Limit:        uploadSessionMaxFilesPerBatch,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	for _, resource := range members.Resources {
		deps.recordResourceEvent(r.Context(), resource.ResourceID, principal, "resource.collection_renamed", domain.JSONMap{
			"collection_id":   collection.CollectionID,
			"previous_name":   existing.Name,
			"collection_name": collection.Name,
			"collection_type": collection.CollectionType,
			"updated_at":      now.UTC().Format(time.RFC3339Nano),
			"source":          "resource_collection_patch",
		})
	}
	writeJSON(w, http.StatusOK, resourceCollectionResponse{Collection: collection})
}

func (deps ServerDeps) handleDeleteResourceCollection(w http.ResponseWriter, r *http.Request) {
	collections, ok := deps.resourceCollectionStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
		return
	}
	principal := deps.principalFromRequest(r, "")
	collectionID := strings.TrimSpace(chi.URLParam(r, "collection_id"))
	if collectionID == "" {
		writeError(w, http.StatusBadRequest, errors.New("collection id is required"))
		return
	}
	members, err := collections.ListResourcesForCollectionForUser(r.Context(), domain.ResourceCollectionResourceListInput{
		CollectionID: collectionID,
		UserID:       principal.UserID,
		OrgID:        principal.OrgID,
		Limit:        uploadSessionMaxFilesPerBatch,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	deletedAt := domain.Now()
	collection, err := collections.SoftDeleteResourceCollectionForUser(r.Context(), collectionID, principal.UserID, principal.OrgID, deletedAt)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	for _, resource := range members.Resources {
		deps.recordResourceEvent(r.Context(), resource.ResourceID, principal, "resource.collection_deleted", domain.JSONMap{
			"collection_id":   collection.CollectionID,
			"collection_name": collection.Name,
			"collection_type": collection.CollectionType,
			"deleted_at":      deletedAt.UTC().Format(time.RFC3339Nano),
			"resource_count":  collection.ResourceCount,
			"audited_count":   len(members.Resources),
			"source":          "resource_collection_lifecycle",
		})
	}
	writeJSON(w, http.StatusOK, resourceCollectionResponse{Collection: collection})
}

func (deps ServerDeps) handleRestoreResourceCollection(w http.ResponseWriter, r *http.Request) {
	collections, ok := deps.resourceCollectionStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
		return
	}
	principal := deps.principalFromRequest(r, "")
	collectionID := strings.TrimSpace(chi.URLParam(r, "collection_id"))
	if collectionID == "" {
		writeError(w, http.StatusBadRequest, errors.New("collection id is required"))
		return
	}
	restoredAt := domain.Now()
	collection, err := collections.RestoreResourceCollectionForUser(r.Context(), collectionID, principal.UserID, principal.OrgID, restoredAt)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	members, err := collections.ListResourcesForCollectionForUser(r.Context(), domain.ResourceCollectionResourceListInput{
		CollectionID: collection.CollectionID,
		UserID:       principal.UserID,
		OrgID:        principal.OrgID,
		Limit:        uploadSessionMaxFilesPerBatch,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	for _, resource := range members.Resources {
		deps.recordResourceEvent(r.Context(), resource.ResourceID, principal, "resource.collection_restored", domain.JSONMap{
			"collection_id":   collection.CollectionID,
			"collection_name": collection.Name,
			"collection_type": collection.CollectionType,
			"restored_at":     restoredAt.UTC().Format(time.RFC3339Nano),
			"resource_count":  collection.ResourceCount,
			"audited_count":   len(members.Resources),
			"source":          "resource_collection_lifecycle",
		})
	}
	writeJSON(w, http.StatusOK, resourceCollectionResponse{Collection: collection})
}

// handleListResourceCollectionShareGrants: owner-only view of a folder's
// collection-level grants — the "People with access" list for folders.
func (deps ServerDeps) handleListResourceCollectionShareGrants(w http.ResponseWriter, r *http.Request) {
	collections, ok := deps.Store.(resourceCollectionShareGrantStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "collection sharing is not configured"})
		return
	}
	principal := deps.principalFromRequest(r, "")
	collectionID := strings.TrimSpace(chi.URLParam(r, "collection_id"))
	if collectionID == "" {
		writeError(w, http.StatusBadRequest, errors.New("collection id is required"))
		return
	}
	grants, err := collections.ListResourceCollectionShareGrantsForCollection(r.Context(), collectionID, principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"grants": grants})
}

// handleRevokeResourceCollectionShareGrant un-shares a folder in one call:
// the collection grant flips to revoked and every inherited per-resource
// grant cascades with it (store-side, transactional in Postgres).
func (deps ServerDeps) handleRevokeResourceCollectionShareGrant(w http.ResponseWriter, r *http.Request) {
	collections, ok := deps.Store.(resourceCollectionShareGrantStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "collection sharing is not configured"})
		return
	}
	principal := deps.principalFromRequest(r, "")
	collectionID := strings.TrimSpace(chi.URLParam(r, "collection_id"))
	grantID := strings.TrimSpace(chi.URLParam(r, "grant_id"))
	if collectionID == "" || grantID == "" {
		writeError(w, http.StatusBadRequest, errors.New("collection id and grant id are required"))
		return
	}
	grant, err := collections.RevokeResourceCollectionShareGrant(r.Context(), collectionID, grantID, principal.UserID, principal.OrgID, domain.Now())
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"grant": grant})
}

func (deps ServerDeps) handleCreateResourceCollectionShareGrants(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	collections, ok := deps.resourceCollectionStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
		return
	}
	collectionShareGrants, ok := deps.Store.(resourceCollectionShareGrantStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collection sharing is not configured"})
		return
	}
	var request createResourceShareGrantRequest
	if !decodeJSON(w, r, &request) {
		return
	}
	role := strings.TrimSpace(request.Role)
	if role == "" {
		role = "read"
	}
	if role != "read" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("unsupported resource share role %q", role))
		return
	}
	granteeUserID := strings.TrimSpace(request.GranteeUserID)
	granteeOrgID := strings.TrimSpace(request.GranteeOrgID)
	if granteeUserID == "" && granteeOrgID == "" {
		writeError(w, http.StatusBadRequest, errors.New("grantee_user_id or grantee_org_id is required"))
		return
	}
	collectionID := strings.TrimSpace(chi.URLParam(r, "collection_id"))
	if collectionID == "" {
		writeError(w, http.StatusBadRequest, errors.New("collection id is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	collection, err := collections.GetResourceCollectionForUser(r.Context(), collectionID, principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	result, err := collectionShareGrants.CreateResourceCollectionShareGrant(r.Context(), domain.CreateResourceCollectionShareGrantInput{
		CollectionID:    collection.CollectionID,
		OwnerUserID:     principal.UserID,
		OwnerOrgID:      principal.OrgID,
		OwnerRole:       principal.Role,
		GranteeUserID:   granteeUserID,
		GranteeOrgID:    granteeOrgID,
		Role:            role,
		Status:          "active",
		CreatedByUserID: principal.UserID,
		Metadata:        request.Metadata,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	for _, grant := range result.ResourceGrants {
		deps.recordResourceEvent(r.Context(), grant.ResourceID, principal, "resource.shared", domain.JSONMap{
			"grant_id":                  grant.GrantID,
			"collection_share_grant_id": result.Grant.GrantID,
			"grantee_user_id":           grant.GranteeUserID,
			"grantee_org_id":            grant.GranteeOrgID,
			"public":                    request.Public,
			"role":                      grant.Role,
			"collection_id":             collection.CollectionID,
			"collection_name":           collection.Name,
			"collection_type":           collection.CollectionType,
			"source":                    "resource_collection_share",
		})
	}
	writeJSON(w, http.StatusCreated, resourceCollectionShareGrantsCreateResponse{
		Count:      len(result.ResourceGrants),
		Collection: collection,
		Grants:     result.ResourceGrants,
	})
}

func (deps ServerDeps) handleAddResourcesToCollection(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	collections, ok := deps.resourceCollectionStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
		return
	}
	var req addResourcesToCollectionRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	resourceIDs := uniqueTrimmedStringValues(req.ResourceIDs)
	if len(resourceIDs) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("resource_ids must include at least one resource"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	result, err := collections.AddResourcesToCollection(r.Context(), domain.AddResourcesToCollectionInput{
		CollectionID:  chi.URLParam(r, "collection_id"),
		OwnerUserID:   principal.UserID,
		OwnerOrgID:    principal.OrgID,
		ResourceIDs:   resourceIDs,
		AddedByUserID: principal.UserID,
		AddedAt:       domain.Now(),
		Metadata:      mapOrEmptyJSON(req.Metadata),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	for _, resourceID := range resourceIDs {
		deps.recordResourceEvent(r.Context(), resourceID, principal, "resource.collection_added", domain.JSONMap{
			"collection_id":   result.Collection.CollectionID,
			"collection_name": result.Collection.Name,
			"collection_type": result.Collection.CollectionType,
		})
	}
	for _, grant := range result.InheritedShareGrants {
		deps.recordResourceEvent(r.Context(), grant.ResourceID, principal, "resource.shared", domain.JSONMap{
			"grant_id":                  grant.GrantID,
			"collection_share_grant_id": grant.Metadata["collection_share_grant_id"],
			"grantee_user_id":           grant.GranteeUserID,
			"grantee_org_id":            grant.GranteeOrgID,
			"role":                      grant.Role,
			"collection_id":             result.Collection.CollectionID,
			"collection_name":           result.Collection.Name,
			"collection_type":           result.Collection.CollectionType,
			"source":                    "resource_collection_share_inherited",
		})
	}
	writeJSON(w, http.StatusOK, addResourcesToCollectionResponse{
		Collection:  result.Collection,
		AddedCount:  result.AddedCount,
		Memberships: result.Memberships,
	})
}

func (deps ServerDeps) handleRemoveResourceFromCollection(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	collections, ok := deps.resourceCollectionStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
		return
	}
	collectionID := strings.TrimSpace(chi.URLParam(r, "collection_id"))
	resourceID := strings.TrimSpace(chi.URLParam(r, "file_id"))
	if collectionID == "" || resourceID == "" {
		writeError(w, http.StatusBadRequest, errors.New("collection id and resource id are required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	removedAt := domain.Now()
	result, err := collections.RemoveResourcesFromCollection(r.Context(), domain.RemoveResourcesFromCollectionInput{
		CollectionID:    collectionID,
		OwnerUserID:     principal.UserID,
		OwnerOrgID:      principal.OrgID,
		ResourceIDs:     []string{resourceID},
		RemovedByUserID: principal.UserID,
		RemovedAt:       removedAt,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	for _, member := range result.Memberships {
		deps.recordResourceEvent(r.Context(), member.ResourceID, principal, "resource.collection_removed", domain.JSONMap{
			"collection_id":      result.Collection.CollectionID,
			"collection_name":    result.Collection.Name,
			"collection_type":    result.Collection.CollectionType,
			"removed_at":         removedAt.UTC().Format(time.RFC3339Nano),
			"removed_by_user_id": principal.UserID,
			"source":             "resource_collection_membership",
		})
	}
	for _, grant := range result.RevokedInheritedShareGrants {
		deps.recordResourceEvent(r.Context(), grant.ResourceID, principal, "resource.share_revoked", domain.JSONMap{
			"grant_id":        grant.GrantID,
			"collection_id":   result.Collection.CollectionID,
			"collection_name": result.Collection.Name,
			"collection_type": result.Collection.CollectionType,
			"source":          "resource_collection_membership",
		})
	}
	writeJSON(w, http.StatusOK, removeResourcesFromCollectionResponse{
		Collection:   result.Collection,
		RemovedCount: result.RemovedCount,
		Memberships:  result.Memberships,
	})
}

func (deps ServerDeps) handleListResourceCollectionResources(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	collections, ok := deps.resourceCollectionStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
		return
	}
	metadataFilters, err := parseResourceMetadataFilters(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	createdAfter, createdBefore, err := parseResourceCreatedRange(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	processingStatus, err := parseResourceProcessingStatus(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	page, err := collections.ListResourcesForCollectionForUser(r.Context(), domain.ResourceCollectionResourceListInput{
		CollectionID:     chi.URLParam(r, "collection_id"),
		UserID:           principal.UserID,
		OrgID:            principal.OrgID,
		Query:            strings.TrimSpace(r.URL.Query().Get("q")),
		Kind:             strings.ToLower(strings.TrimSpace(r.URL.Query().Get("kind"))),
		Source:           strings.ToLower(strings.TrimSpace(r.URL.Query().Get("source"))),
		ProjectID:        strings.TrimSpace(r.URL.Query().Get("project_id")),
		Sharing:          strings.ToLower(strings.TrimSpace(r.URL.Query().Get("sharing"))),
		Tags:             parseResourceTagFilters(r),
		Descriptors:      parseResourceDescriptorFilters(r),
		MetadataFilters:  metadataFilters,
		CreatedAfter:     createdAfter,
		CreatedBefore:    createdBefore,
		ProcessingStatus: processingStatus,
		Limit:            clampLimit(parseLimit(r, 200), 1000),
		Offset:           parseOffset(r),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	records := make([]resourceRecord, 0, len(page.Resources))
	for _, resource := range page.Resources {
		records = append(records, deps.resourceRecordFromCatalog(root, resource))
	}
	writeJSON(w, http.StatusOK, resourcesResponse{Count: page.TotalCount, Resources: records})
}

func (deps ServerDeps) handleCreateDatasetSnapshot(w http.ResponseWriter, r *http.Request) {
	snapshots, ok := deps.datasetSnapshotStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "dataset snapshots are not configured"})
		return
	}
	var req createDatasetSnapshotRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	name := strings.TrimSpace(req.Name)
	if name == "" {
		writeError(w, http.StatusBadRequest, errors.New("snapshot name is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	sourceCollectionID := strings.TrimSpace(req.SourceCollectionID)
	resourceIDs := uniqueTrimmedStringValues(req.ResourceIDs)
	resourceQuery := datasetSnapshotResourceQueryFromRequest(req.ResourceQuery)
	if len(resourceIDs) == 0 && sourceCollectionID != "" && resourceQuery == nil {
		collections, ok := deps.resourceCollectionStore()
		if !ok {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
			return
		}
		page, err := collections.ListResourcesForCollectionForUser(r.Context(), domain.ResourceCollectionResourceListInput{
			CollectionID: sourceCollectionID,
			UserID:       principal.UserID,
			OrgID:        principal.OrgID,
			Limit:        uploadSessionMaxFilesPerBatch,
		})
		if err != nil {
			writeStoreError(w, err)
			return
		}
		if page.TotalCount > len(page.Resources) {
			writeError(w, http.StatusBadRequest, errors.New("source collection contains too many resources to snapshot in one request"))
			return
		}
		resourceIDs = make([]string, 0, len(page.Resources))
		for _, resource := range page.Resources {
			resourceIDs = append(resourceIDs, resource.ResourceID)
		}
	}
	if len(resourceIDs) == 0 && sourceCollectionID == "" && resourceQuery == nil {
		writeError(w, http.StatusBadRequest, errors.New("resource_ids, source_collection_id, or resource_query is required"))
		return
	}
	projectID := strings.TrimSpace(req.ProjectID)
	if projectID == "" && resourceQuery != nil {
		projectID = strings.TrimSpace(resourceQuery.ProjectID)
	}
	snapshot, entries, err := snapshots.CreateDatasetSnapshot(r.Context(), domain.CreateDatasetSnapshotInput{
		OwnerUserID:        principal.UserID,
		OwnerOrgID:         principal.OrgID,
		OwnerRole:          principal.Role,
		ProjectID:          projectID,
		SourceCollectionID: sourceCollectionID,
		Name:               name,
		Description:        strings.TrimSpace(req.Description),
		ResourceIDs:        resourceIDs,
		ResourceQuery:      resourceQuery,
		CreatedByUserID:    principal.UserID,
		CreatedAt:          domain.Now(),
		Metadata:           mapOrEmptyJSON(req.Metadata),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	for _, entry := range entries {
		deps.recordResourceEvent(r.Context(), entry.ResourceID, principal, "resource.dataset_snapshotted", domain.JSONMap{
			"snapshot_id":          snapshot.SnapshotID,
			"snapshot_name":        snapshot.Name,
			"source_collection_id": snapshot.SourceCollectionID,
		})
	}
	writeJSON(w, http.StatusCreated, datasetSnapshotResponse{Snapshot: snapshot, Resources: entries})
}

func (deps ServerDeps) handleListDatasetSnapshots(w http.ResponseWriter, r *http.Request) {
	snapshots, ok := deps.datasetSnapshotStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "dataset snapshots are not configured"})
		return
	}
	status, err := parseResourceLifecycleStatus(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	page, err := snapshots.ListDatasetSnapshotsForUser(r.Context(), domain.DatasetSnapshotListInput{
		UserID:             principal.UserID,
		OrgID:              principal.OrgID,
		Query:              strings.TrimSpace(r.URL.Query().Get("q")),
		ProjectID:          strings.TrimSpace(r.URL.Query().Get("project_id")),
		SourceCollectionID: strings.TrimSpace(r.URL.Query().Get("source_collection_id")),
		Status:             status,
		Limit:              clampLimit(parseLimit(r, 200), 1000),
		Offset:             parseOffset(r),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, datasetSnapshotsResponse{Count: page.TotalCount, Snapshots: page.Snapshots})
}

func (deps ServerDeps) handleDeleteDatasetSnapshot(w http.ResponseWriter, r *http.Request) {
	snapshots, ok := deps.datasetSnapshotStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "dataset snapshots are not configured"})
		return
	}
	snapshotID := strings.TrimSpace(chi.URLParam(r, "snapshot_id"))
	if snapshotID == "" {
		writeError(w, http.StatusBadRequest, errors.New("dataset snapshot id is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	deletedAt := domain.Now()
	snapshot, entries, err := snapshots.SoftDeleteDatasetSnapshotForUser(r.Context(), snapshotID, principal.UserID, principal.OrgID, deletedAt)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	for _, entry := range entries {
		deps.recordResourceEvent(r.Context(), entry.ResourceID, principal, "resource.dataset_snapshot_deleted", domain.JSONMap{
			"snapshot_id":          snapshot.SnapshotID,
			"snapshot_name":        snapshot.Name,
			"project_id":           snapshot.ProjectID,
			"source_collection_id": snapshot.SourceCollectionID,
			"deleted_at":           deletedAt.UTC().Format(time.RFC3339Nano),
			"resource_count":       snapshot.ResourceCount,
			"audited_count":        len(entries),
			"source":               "dataset_snapshot_lifecycle",
		})
	}
	writeJSON(w, http.StatusOK, datasetSnapshotResponse{Snapshot: snapshot, Resources: entries})
}

func (deps ServerDeps) handleRestoreDatasetSnapshot(w http.ResponseWriter, r *http.Request) {
	snapshots, ok := deps.datasetSnapshotStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "dataset snapshots are not configured"})
		return
	}
	snapshotID := strings.TrimSpace(chi.URLParam(r, "snapshot_id"))
	if snapshotID == "" {
		writeError(w, http.StatusBadRequest, errors.New("dataset snapshot id is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	restoredAt := domain.Now()
	snapshot, entries, err := snapshots.RestoreDatasetSnapshotForUser(r.Context(), snapshotID, principal.UserID, principal.OrgID, restoredAt)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	for _, entry := range entries {
		deps.recordResourceEvent(r.Context(), entry.ResourceID, principal, "resource.dataset_snapshot_restored", domain.JSONMap{
			"snapshot_id":          snapshot.SnapshotID,
			"snapshot_name":        snapshot.Name,
			"project_id":           snapshot.ProjectID,
			"source_collection_id": snapshot.SourceCollectionID,
			"restored_at":          restoredAt.UTC().Format(time.RFC3339Nano),
			"resource_count":       snapshot.ResourceCount,
			"audited_count":        len(entries),
			"source":               "dataset_snapshot_lifecycle",
		})
	}
	writeJSON(w, http.StatusOK, datasetSnapshotResponse{Snapshot: snapshot, Resources: entries})
}

func (deps ServerDeps) handleGetDatasetSnapshot(w http.ResponseWriter, r *http.Request) {
	snapshots, ok := deps.datasetSnapshotStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "dataset snapshots are not configured"})
		return
	}
	principal := deps.principalFromRequest(r, "")
	snapshot, entries, err := snapshots.GetDatasetSnapshotForUser(r.Context(), chi.URLParam(r, "snapshot_id"), principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, datasetSnapshotResponse{Snapshot: snapshot, Resources: entries})
}

func (deps ServerDeps) handleListDatasetSnapshotEvents(w http.ResponseWriter, r *http.Request) {
	events, ok := deps.Store.(datasetSnapshotEventStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "dataset snapshot events are not configured"})
		return
	}
	snapshotID := strings.TrimSpace(chi.URLParam(r, "snapshot_id"))
	if snapshotID == "" {
		writeError(w, http.StatusBadRequest, errors.New("dataset snapshot id is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	limit := clampLimit(parseLimit(r, 200), 1000)
	offset := parseOffset(r)
	page, err := events.ListDatasetSnapshotEventsForUser(r.Context(), domain.DatasetSnapshotEventListInput{
		SnapshotID:  snapshotID,
		UserID:      principal.UserID,
		OrgID:       principal.OrgID,
		EventType:   strings.TrimSpace(r.URL.Query().Get("event_type")),
		ActorUserID: strings.TrimSpace(r.URL.Query().Get("actor_user_id")),
		Limit:       limit,
		Offset:      offset,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, datasetSnapshotEventsResponse{
		SnapshotID: snapshotID,
		Count:      len(page.Events),
		TotalCount: page.TotalCount,
		Limit:      page.Limit,
		Offset:     page.Offset,
		Events:     page.Events,
	})
}

func (deps ServerDeps) handleCreateDatasetSnapshotShareGrant(w http.ResponseWriter, r *http.Request) {
	shareGrants, ok := deps.Store.(datasetSnapshotShareGrantStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "dataset snapshot sharing is not configured"})
		return
	}
	var request createResourceShareGrantRequest
	if !decodeJSON(w, r, &request) {
		return
	}
	snapshotID := strings.TrimSpace(chi.URLParam(r, "snapshot_id"))
	if snapshotID == "" {
		writeError(w, http.StatusBadRequest, errors.New("dataset snapshot id is required"))
		return
	}
	role := strings.TrimSpace(request.Role)
	if role == "" {
		role = "read"
	}
	if role != "read" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("unsupported dataset snapshot share role %q", role))
		return
	}
	granteeUserID := strings.TrimSpace(request.GranteeUserID)
	granteeOrgID := strings.TrimSpace(request.GranteeOrgID)
	if granteeUserID == "" && granteeOrgID == "" {
		writeError(w, http.StatusBadRequest, errors.New("grantee_user_id or grantee_org_id is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	grant, err := shareGrants.CreateDatasetSnapshotShareGrant(r.Context(), domain.CreateDatasetSnapshotShareGrantInput{
		SnapshotID:      snapshotID,
		OwnerUserID:     principal.UserID,
		OwnerOrgID:      principal.OrgID,
		OwnerRole:       principal.Role,
		GranteeUserID:   granteeUserID,
		GranteeOrgID:    granteeOrgID,
		Role:            role,
		Status:          "active",
		CreatedByUserID: principal.UserID,
		CreatedAt:       domain.Now(),
		Metadata:        request.Metadata,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, datasetSnapshotShareGrantResponse{Grant: grant})
}

func (deps ServerDeps) handleListDatasetSnapshotShareGrants(w http.ResponseWriter, r *http.Request) {
	shareGrants, ok := deps.Store.(datasetSnapshotShareGrantStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "dataset snapshot sharing is not configured"})
		return
	}
	snapshotID := strings.TrimSpace(chi.URLParam(r, "snapshot_id"))
	if snapshotID == "" {
		writeError(w, http.StatusBadRequest, errors.New("dataset snapshot id is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	grants, err := shareGrants.ListDatasetSnapshotShareGrants(r.Context(), domain.ListDatasetSnapshotShareGrantsInput{
		SnapshotID:  snapshotID,
		OwnerUserID: principal.UserID,
		OwnerOrgID:  principal.OrgID,
		Status:      strings.TrimSpace(r.URL.Query().Get("status")),
		Limit:       clampLimit(parseLimit(r, 200), 1000),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, datasetSnapshotShareGrantsResponse{Count: len(grants), Grants: grants})
}

func (deps ServerDeps) handleRevokeDatasetSnapshotShareGrant(w http.ResponseWriter, r *http.Request) {
	shareGrants, ok := deps.Store.(datasetSnapshotShareGrantStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "dataset snapshot sharing is not configured"})
		return
	}
	snapshotID := strings.TrimSpace(chi.URLParam(r, "snapshot_id"))
	grantID := strings.TrimSpace(chi.URLParam(r, "grant_id"))
	if snapshotID == "" || grantID == "" {
		writeError(w, http.StatusBadRequest, errors.New("dataset snapshot id and grant id are required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	grant, err := shareGrants.RevokeDatasetSnapshotShareGrant(r.Context(), domain.RevokeDatasetSnapshotShareGrantInput{
		SnapshotID:      snapshotID,
		GrantID:         grantID,
		OwnerUserID:     principal.UserID,
		OwnerOrgID:      principal.OrgID,
		RevokedByUserID: principal.UserID,
		RevokedAt:       domain.Now(),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, datasetSnapshotShareGrantResponse{Grant: grant})
}

func (deps ServerDeps) handleCreateDataAgentJob(w http.ResponseWriter, r *http.Request) {
	jobs, ok := deps.dataAgentJobStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "data agent jobs are not configured"})
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var req createDataAgentJobRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	jobType, err := normalizeDataAgentJobType(req.JobType)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	sourceCollectionID := strings.TrimSpace(req.SourceCollectionID)
	resourceIDs := uniqueTrimmedStringValues(req.ResourceIDs)
	resourceQuery := datasetSnapshotResourceQueryFromRequest(req.ResourceQuery)
	if len(resourceIDs) == 0 && sourceCollectionID != "" && resourceQuery == nil {
		collections, ok := deps.resourceCollectionStore()
		if !ok {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
			return
		}
		page, err := collections.ListResourcesForCollectionForUser(r.Context(), domain.ResourceCollectionResourceListInput{
			CollectionID: sourceCollectionID,
			UserID:       principal.UserID,
			OrgID:        principal.OrgID,
			Limit:        uploadSessionMaxFilesPerBatch,
		})
		if err != nil {
			writeStoreError(w, err)
			return
		}
		if page.TotalCount > len(page.Resources) {
			writeError(w, http.StatusBadRequest, errors.New("source collection contains too many resources for one data-agent job"))
			return
		}
		resourceIDs = make([]string, 0, len(page.Resources))
		for _, resource := range page.Resources {
			resourceIDs = append(resourceIDs, resource.ResourceID)
		}
	}
	if len(resourceIDs) == 0 && sourceCollectionID == "" && resourceQuery == nil {
		writeError(w, http.StatusBadRequest, errors.New("resource_ids, source_collection_id, or resource_query is required"))
		return
	}
	resourceCount := len(resourceIDs)
	if len(resourceIDs) == 0 && resourceQuery != nil {
		resourceCount, err = deps.countDataAgentQueryResources(r.Context(), principal, sourceCollectionID, resourceQuery)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		if resourceCount > domain.DataAgentQueryResourceHardLimit {
			writeError(w, http.StatusBadRequest, fmt.Errorf("resource query matched %d resources, above one data-agent job limit %d", resourceCount, domain.DataAgentQueryResourceHardLimit))
			return
		}
	}
	inputSelector := mapOrEmptyJSON(req.InputSelector)
	if len(resourceIDs) > 0 {
		inputSelector["resource_ids"] = append([]string(nil), resourceIDs...)
	}
	if sourceCollectionID != "" {
		inputSelector["source_collection_id"] = sourceCollectionID
	}
	if resourceQuery != nil {
		inputSelector["resource_query"] = datasetSnapshotResourceQuerySelector(resourceQuery)
	}
	job, err := jobs.CreateDataAgentJob(r.Context(), domain.CreateDataAgentJobInput{
		OwnerUserID:     principal.UserID,
		OwnerOrgID:      principal.OrgID,
		OwnerRole:       principal.Role,
		ProjectID:       strings.TrimSpace(req.ProjectID),
		JobType:         jobType,
		Status:          "queued",
		ResourceIDs:     resourceIDs,
		ResourceCount:   resourceCount,
		InputSelector:   inputSelector,
		CreatedByUserID: principal.UserID,
		CreatedAt:       domain.Now(),
		Metadata:        mapOrEmptyJSON(req.Metadata),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	for _, resourceID := range resourceIDs {
		deps.recordResourceEvent(r.Context(), resourceID, principal, "resource.data_agent_job_queued", domain.JSONMap{
			"job_id":               job.JobID,
			"job_type":             job.JobType,
			"source_collection_id": sourceCollectionID,
		})
	}
	if err := deps.dispatchDataAgentJob(r.Context(), jobs, job, principal); err != nil {
		writeError(w, http.StatusServiceUnavailable, err)
		return
	}
	events, err := jobs.ListDataAgentJobEvents(r.Context(), job.JobID, principal.UserID, principal.OrgID, parseLimit(r, 200))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusAccepted, dataAgentJobResponse{Job: job, Events: events})
}

func (deps ServerDeps) countDataAgentQueryResources(ctx context.Context, principal requestPrincipal, sourceCollectionID string, query *domain.DatasetSnapshotResourceQuery) (int, error) {
	if query == nil {
		return 0, nil
	}
	if sourceCollectionID != "" {
		collections, ok := deps.resourceCollectionStore()
		if !ok {
			return 0, errors.New("resource collections are not configured")
		}
		page, err := collections.ListResourcesForCollectionForUser(ctx, domain.ResourceCollectionResourceListInput{
			CollectionID:     sourceCollectionID,
			UserID:           principal.UserID,
			OrgID:            principal.OrgID,
			Query:            query.Query,
			Kind:             query.Kind,
			Source:           query.Source,
			ProjectID:        query.ProjectID,
			Sharing:          query.Sharing,
			Tags:             query.Tags,
			Descriptors:      query.Descriptors,
			MetadataFilters:  query.MetadataFilters,
			CreatedAfter:     query.CreatedAfter,
			CreatedBefore:    query.CreatedBefore,
			ProcessingStatus: query.ProcessingStatus,
			Limit:            1,
		})
		if err != nil {
			return 0, err
		}
		return page.TotalCount, nil
	}
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		return 0, errors.New("resource catalog is not configured")
	}
	page, err := catalog.ListResourcesForUser(ctx, domain.ResourceListInput{
		UserID:           principal.UserID,
		OrgID:            principal.OrgID,
		Query:            query.Query,
		Kind:             query.Kind,
		Source:           query.Source,
		ProjectID:        query.ProjectID,
		Sharing:          query.Sharing,
		Tags:             query.Tags,
		Descriptors:      query.Descriptors,
		MetadataFilters:  query.MetadataFilters,
		CreatedAfter:     query.CreatedAfter,
		CreatedBefore:    query.CreatedBefore,
		ProcessingStatus: query.ProcessingStatus,
		Limit:            1,
	})
	if err != nil {
		return 0, err
	}
	return page.TotalCount, nil
}

func (deps ServerDeps) handleListDataAgentJobs(w http.ResponseWriter, r *http.Request) {
	jobs, ok := deps.dataAgentJobStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "data agent jobs are not configured"})
		return
	}
	principal := deps.principalFromRequest(r, "")
	page, err := jobs.ListDataAgentJobsForUser(r.Context(), domain.DataAgentJobListInput{
		UserID:    principal.UserID,
		OrgID:     principal.OrgID,
		JobType:   strings.ToLower(strings.TrimSpace(r.URL.Query().Get("job_type"))),
		Status:    strings.ToLower(strings.TrimSpace(r.URL.Query().Get("status"))),
		ProjectID: strings.TrimSpace(r.URL.Query().Get("project_id")),
		Limit:     parseLimit(r, 200),
		Offset:    parseOffset(r),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, dataAgentJobsResponse{Count: page.TotalCount, Jobs: page.Jobs})
}

func (deps ServerDeps) handleGetDataAgentJob(w http.ResponseWriter, r *http.Request) {
	jobs, ok := deps.dataAgentJobStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "data agent jobs are not configured"})
		return
	}
	principal := deps.principalFromRequest(r, "")
	job, err := jobs.GetDataAgentJobForUser(r.Context(), chi.URLParam(r, "job_id"), principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	events, err := jobs.ListDataAgentJobEvents(r.Context(), job.JobID, principal.UserID, principal.OrgID, parseLimit(r, 200))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, dataAgentJobResponse{Job: job, Events: events})
}

func (deps ServerDeps) handleUpdateDataAgentJobStatus(w http.ResponseWriter, r *http.Request) {
	jobs, ok := deps.dataAgentJobStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "data agent jobs are not configured"})
		return
	}
	var req updateDataAgentJobStatusRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	status, err := normalizeDataAgentJobStatus(req.Status)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	job, _, err := jobs.UpdateDataAgentJob(r.Context(), domain.UpdateDataAgentJobInput{
		JobID:             chi.URLParam(r, "job_id"),
		OwnerUserID:       principal.UserID,
		OwnerOrgID:        principal.OrgID,
		Status:            status,
		ProgressCompleted: req.ProgressCompleted,
		ProgressTotal:     req.ProgressTotal,
		Error:             strings.TrimSpace(req.Error),
		ActorUserID:       principal.UserID,
		ActorOrgID:        principal.OrgID,
		Message:           strings.TrimSpace(req.Message),
		UpdatedAt:         domain.Now(),
		OutputSummary:     mapOrEmptyJSON(req.OutputSummary),
		Metadata:          req.Metadata,
		EventMetadata:     mapOrEmptyJSON(req.EventMetadata),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	events, err := jobs.ListDataAgentJobEvents(r.Context(), job.JobID, principal.UserID, principal.OrgID, parseLimit(r, 200))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, dataAgentJobResponse{Job: job, Events: events})
}

func (deps ServerDeps) handleAppendDataAgentJobEvent(w http.ResponseWriter, r *http.Request) {
	jobs, ok := deps.dataAgentJobStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "data agent jobs are not configured"})
		return
	}
	var req appendDataAgentJobEventRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	eventType, err := normalizeDataAgentJobAppendEventType(req.EventType)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	job, err := jobs.GetDataAgentJobForUser(r.Context(), chi.URLParam(r, "job_id"), principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	eventID := strings.TrimSpace(req.EventID)
	if _, err := jobs.AppendDataAgentJobEvent(r.Context(), domain.AppendDataAgentJobEventInput{
		EventID:     eventID,
		JobID:       job.JobID,
		EventType:   eventType,
		ActorUserID: principal.UserID,
		ActorOrgID:  principal.OrgID,
		TS:          domain.Now(),
		Message:     strings.TrimSpace(req.Message),
		Metadata:    mapOrEmptyJSON(req.Metadata),
	}); err != nil {
		if errors.Is(err, store.ErrConflict) && eventID != "" {
			events, listErr := jobs.ListDataAgentJobEvents(r.Context(), job.JobID, principal.UserID, principal.OrgID, parseLimit(r, 200))
			if listErr != nil {
				writeStoreError(w, listErr)
				return
			}
			if dataAgentJobEventsContain(events, eventID, eventType) {
				writeJSON(w, http.StatusOK, dataAgentJobResponse{Job: job, Events: events})
				return
			}
		}
		writeStoreError(w, err)
		return
	}
	events, err := jobs.ListDataAgentJobEvents(r.Context(), job.JobID, principal.UserID, principal.OrgID, parseLimit(r, 200))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, dataAgentJobResponse{Job: job, Events: events})
}

func (deps ServerDeps) handleControlDataAgentJob(w http.ResponseWriter, r *http.Request) {
	jobs, ok := deps.dataAgentJobStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "data agent jobs are not configured"})
		return
	}
	var req controlDataAgentJobRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	action, err := normalizeDataAgentJobControlAction(req.Action)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	job, _, err := jobs.ControlDataAgentJob(r.Context(), domain.ControlDataAgentJobInput{
		JobID:       chi.URLParam(r, "job_id"),
		OwnerUserID: principal.UserID,
		OwnerOrgID:  principal.OrgID,
		Action:      action,
		Reason:      strings.TrimSpace(req.Reason),
		ActorUserID: principal.UserID,
		ActorOrgID:  principal.OrgID,
		TS:          domain.Now(),
		Metadata:    mapOrEmptyJSON(req.Metadata),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if action == "retry" {
		if err := deps.dispatchDataAgentJob(r.Context(), jobs, job, principal); err != nil {
			writeError(w, http.StatusServiceUnavailable, err)
			return
		}
	}
	events, err := jobs.ListDataAgentJobEvents(r.Context(), job.JobID, principal.UserID, principal.OrgID, parseLimit(r, 200))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, dataAgentJobResponse{Job: job, Events: events})
}

func (deps ServerDeps) handleAcquireDataAgentJobLease(w http.ResponseWriter, r *http.Request) {
	jobs, ok := deps.dataAgentJobStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "data agent jobs are not configured"})
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
	principal := deps.principalFromRequest(r, "")
	lease, _, _, err := jobs.AcquireDataAgentJobLease(r.Context(), domain.AcquireDataAgentJobLeaseInput{
		JobID:       chi.URLParam(r, "job_id"),
		OwnerUserID: principal.UserID,
		OwnerOrgID:  principal.OrgID,
		WorkerID:    req.WorkerID,
		TTL:         leaseTTL(req),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, lease)
}

func (deps ServerDeps) handleRenewDataAgentJobLease(w http.ResponseWriter, r *http.Request) {
	jobs, ok := deps.dataAgentJobStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "data agent jobs are not configured"})
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
	principal := deps.principalFromRequest(r, "")
	lease, err := jobs.RenewDataAgentJobLease(r.Context(), domain.RenewDataAgentJobLeaseInput{
		JobID:       chi.URLParam(r, "job_id"),
		OwnerUserID: principal.UserID,
		OwnerOrgID:  principal.OrgID,
		LeaseToken:  req.LeaseToken,
		TTL:         leaseTTL(req),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, lease)
}

func (deps ServerDeps) handleReleaseDataAgentJobLease(w http.ResponseWriter, r *http.Request) {
	jobs, ok := deps.dataAgentJobStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "data agent jobs are not configured"})
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
	principal := deps.principalFromRequest(r, "")
	if err := jobs.ReleaseDataAgentJobLease(r.Context(), domain.ReleaseDataAgentJobLeaseInput{
		JobID:       chi.URLParam(r, "job_id"),
		OwnerUserID: principal.UserID,
		OwnerOrgID:  principal.OrgID,
		LeaseToken:  req.LeaseToken,
	}); err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"released": true})
}

func (deps ServerDeps) dispatchDataAgentJob(ctx context.Context, jobs dataAgentJobStore, job domain.DataAgentJobRecord, principal requestPrincipal) error {
	if deps.DataAgentJobs == nil {
		return nil
	}
	dispatchID := domain.NewID("dispatch")
	envelope := eventbus.DataAgentJob{
		JobID:         job.JobID,
		DispatchID:    dispatchID,
		OwnerUserID:   job.OwnerUserID,
		OwnerOrgID:    job.OwnerOrgID,
		ProjectID:     job.ProjectID,
		JobType:       job.JobType,
		ResourceIDs:   metadataStringSlice(job.InputSelector["resource_ids"]),
		ResourceCount: job.ResourceCount,
		InputSelector: cloneHTTPJSONMap(job.InputSelector),
		Metadata:      cloneHTTPJSONMap(job.Metadata),
	}
	if err := deps.DataAgentJobs.PublishDataAgentJob(ctx, envelope); err != nil {
		failureText := fmt.Sprintf("failed to enqueue data-agent job: %v", err)
		if _, _, markErr := jobs.UpdateDataAgentJob(ctx, domain.UpdateDataAgentJobInput{
			JobID:             job.JobID,
			OwnerUserID:       job.OwnerUserID,
			OwnerOrgID:        job.OwnerOrgID,
			Status:            "failed",
			ProgressCompleted: job.ProgressCompleted,
			ProgressTotal:     job.ProgressTotal,
			Error:             failureText,
			ActorUserID:       principal.UserID,
			ActorOrgID:        principal.OrgID,
			Message:           "Data Agent job failed before worker dispatch.",
			UpdatedAt:         domain.Now(),
			EventMetadata: domain.JSONMap{
				"stage": "job_enqueue",
				"error": failureText,
			},
		}); markErr != nil {
			return fmt.Errorf("publish data-agent job: %w; additionally failed to mark job failed: %v", err, markErr)
		}
		return fmt.Errorf("publish data-agent job: %w", err)
	}
	_, err := jobs.AppendDataAgentJobEvent(ctx, domain.AppendDataAgentJobEventInput{
		JobID:       job.JobID,
		EventType:   "data_agent.job.dispatched",
		ActorUserID: principal.UserID,
		ActorOrgID:  principal.OrgID,
		TS:          domain.Now(),
		Message:     "Data Agent job dispatched.",
		Metadata: domain.JSONMap{
			"dispatch_id":    dispatchID,
			"job_type":       job.JobType,
			"resource_count": job.ResourceCount,
		},
	})
	if err != nil {
		return fmt.Errorf("append data-agent dispatch event: %w", err)
	}
	return nil
}

func normalizeResourceCollectionType(value string) (string, error) {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		value = "collection"
	}
	switch value {
	case "collection", "folder", "dataset":
		return value, nil
	default:
		return "", errors.New("collection_type must be collection, folder, or dataset")
	}
}

func normalizeDataAgentJobType(value string) (string, error) {
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "caption_resources", "extract_metadata", "organize_resources", "deduplicate_resources", "quality_check_resources", "batch_tag_resources", "create_dataset_snapshot":
		return value, nil
	default:
		return "", errors.New("job_type must be caption_resources, extract_metadata, organize_resources, deduplicate_resources, quality_check_resources, batch_tag_resources, or create_dataset_snapshot")
	}
}

func normalizeDataAgentJobStatus(value string) (string, error) {
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "queued", "running", "succeeded", "failed", "canceled":
		return value, nil
	default:
		return "", errors.New("status must be queued, running, succeeded, failed, or canceled")
	}
}

func normalizeDataAgentJobAppendEventType(value string) (string, error) {
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "data_agent.job.skipped":
		return value, nil
	default:
		return "", errors.New("event_type must be data_agent.job.skipped")
	}
}

func dataAgentJobEventsContain(events []domain.DataAgentJobEventRecord, eventID string, eventType string) bool {
	eventID = strings.TrimSpace(eventID)
	eventType = strings.TrimSpace(eventType)
	for _, event := range events {
		if event.EventID == eventID && event.EventType == eventType {
			return true
		}
	}
	return false
}

func normalizeDataAgentJobControlAction(value string) (string, error) {
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "cancel", "retry":
		return value, nil
	default:
		return "", errors.New("action must be cancel or retry")
	}
}

func mapOrEmptyJSON(value domain.JSONMap) domain.JSONMap {
	if value == nil {
		return domain.JSONMap{}
	}
	return value
}

func cloneHTTPJSONMap(value domain.JSONMap) domain.JSONMap {
	if value == nil {
		return nil
	}
	cloned := domain.JSONMap{}
	for key, item := range value {
		cloned[key] = item
	}
	return cloned
}

func uniqueTrimmedStringValues(values []string) []string {
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

func parseResourceTagFilters(r *http.Request) []string {
	values := append([]string(nil), r.URL.Query()["tag"]...)
	for _, value := range r.URL.Query()["tags"] {
		for _, part := range strings.Split(value, ",") {
			values = append(values, part)
		}
	}
	return uniqueTrimmedStringValues(values)
}

func parseResourceDescriptorFilters(r *http.Request) []string {
	values := append([]string(nil), r.URL.Query()["descriptor"]...)
	for _, value := range r.URL.Query()["descriptors"] {
		for _, part := range strings.Split(value, ",") {
			values = append(values, part)
		}
	}
	return uniqueTrimmedStringValues(values)
}

func parseResourceMetadataFilters(r *http.Request) ([]domain.ResourceMetadataFilter, error) {
	values := append([]string(nil), r.URL.Query()["metadata_filter"]...)
	if len(values) == 0 {
		return nil, nil
	}
	filters := make([]domain.ResourceMetadataFilter, 0, len(values))
	for _, raw := range values {
		raw = strings.TrimSpace(raw)
		if raw == "" {
			continue
		}
		parts := strings.SplitN(raw, ":", 3)
		if len(parts) != 3 {
			return nil, fmt.Errorf("metadata_filter %q must use path:op:value", raw)
		}
		path := strings.TrimSpace(parts[0])
		operator := strings.ToLower(strings.TrimSpace(parts[1]))
		value := strings.TrimSpace(parts[2])
		if err := validateResourceMetadataFilter(path, operator, value); err != nil {
			return nil, err
		}
		filters = append(filters, domain.ResourceMetadataFilter{
			Path:     path,
			Operator: operator,
			Value:    value,
		})
	}
	return filters, nil
}

func parseResourceCreatedRange(r *http.Request) (time.Time, time.Time, error) {
	createdAfter, err := parseResourceCreatedBound(r.URL.Query().Get("created_after"), false)
	if err != nil {
		return time.Time{}, time.Time{}, err
	}
	createdBefore, err := parseResourceCreatedBound(r.URL.Query().Get("created_before"), true)
	if err != nil {
		return time.Time{}, time.Time{}, err
	}
	return createdAfter, createdBefore, nil
}

func parseResourceProcessingStatus(r *http.Request) (string, error) {
	return validateResourceProcessingStatus(r.URL.Query().Get("processing_status"))
}

func validateResourceProcessingStatus(value string) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(value))
	switch normalized {
	case "", "all":
		return "", nil
	case "caption_ready", "metadata_ready", "tags_ready", "qc_complete", "dedupe_checked", "organization_ready", "data_agent_ready", "needs_caption", "needs_metadata", "data_agent_failed":
		return normalized, nil
	default:
		return "", errors.New("processing_status must be all, caption_ready, metadata_ready, tags_ready, qc_complete, dedupe_checked, organization_ready, data_agent_ready, needs_caption, needs_metadata, or data_agent_failed")
	}
}

func parseResourceCreatedBound(value string, endOfDay bool) (time.Time, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return time.Time{}, nil
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339} {
		if ts, err := time.Parse(layout, value); err == nil {
			return ts.UTC(), nil
		}
	}
	ts, err := time.Parse("2006-01-02", value)
	if err != nil {
		return time.Time{}, fmt.Errorf("created date %q must be RFC3339 or YYYY-MM-DD", value)
	}
	if endOfDay {
		ts = ts.AddDate(0, 0, 1).Add(-time.Nanosecond)
	}
	return ts.UTC(), nil
}

func validateResourceMetadataFilter(path string, operator string, value string) error {
	if path == "" {
		return errors.New("metadata_filter path is required")
	}
	for _, part := range strings.Split(path, ".") {
		if strings.TrimSpace(part) == "" {
			return fmt.Errorf("metadata_filter path %q is invalid", path)
		}
	}
	switch operator {
	case "eq", "contains", "exists":
	case "lt", "lte", "gt", "gte":
		if _, err := strconv.ParseFloat(value, 64); err != nil {
			return fmt.Errorf("metadata_filter %s requires a numeric value", operator)
		}
	default:
		return fmt.Errorf("metadata_filter operator %q is unsupported", operator)
	}
	if operator != "exists" && value == "" {
		return errors.New("metadata_filter value is required")
	}
	return nil
}

func datasetSnapshotResourceQueryFromRequest(req *datasetSnapshotResourceQueryRequest) *domain.DatasetSnapshotResourceQuery {
	if req == nil {
		return nil
	}
	query := strings.TrimSpace(req.Query)
	kind := strings.ToLower(strings.TrimSpace(req.Kind))
	source := strings.ToLower(strings.TrimSpace(req.Source))
	projectID := strings.TrimSpace(req.ProjectID)
	sharing := strings.ToLower(strings.TrimSpace(req.Sharing))
	tags := uniqueTrimmedStringValues(req.Tags)
	descriptors := uniqueTrimmedStringValues(req.Descriptors)
	metadataFilters := normalizeResourceMetadataFilters(req.MetadataFilters)
	createdAfter := normalizeResourceQueryCreatedBound(req.CreatedAfter, false)
	createdBefore := normalizeResourceQueryCreatedBound(req.CreatedBefore, true)
	processingStatus := normalizeResourceProcessingStatus(req.ProcessingStatus)
	if query == "" && kind == "" && source == "" && projectID == "" && sharing == "" && len(tags) == 0 && len(descriptors) == 0 && len(metadataFilters) == 0 && createdAfter.IsZero() && createdBefore.IsZero() && processingStatus == "" {
		return nil
	}
	return &domain.DatasetSnapshotResourceQuery{
		Query:            query,
		Kind:             kind,
		Source:           source,
		ProjectID:        projectID,
		Sharing:          sharing,
		Tags:             tags,
		Descriptors:      descriptors,
		MetadataFilters:  metadataFilters,
		CreatedAfter:     createdAfter,
		CreatedBefore:    createdBefore,
		ProcessingStatus: processingStatus,
	}
}

func normalizeResourceQueryCreatedBound(value string, endOfDay bool) time.Time {
	ts, err := parseResourceCreatedBound(value, endOfDay)
	if err != nil {
		return time.Time{}
	}
	return ts
}

func normalizeResourceProcessingStatus(value string) string {
	processingStatus, err := validateResourceProcessingStatus(value)
	if err != nil {
		return ""
	}
	return processingStatus
}

func normalizeResourceMetadataFilters(filters []domain.ResourceMetadataFilter) []domain.ResourceMetadataFilter {
	out := make([]domain.ResourceMetadataFilter, 0, len(filters))
	for _, filter := range filters {
		path := strings.TrimSpace(filter.Path)
		operator := strings.ToLower(strings.TrimSpace(filter.Operator))
		value := strings.TrimSpace(filter.Value)
		if validateResourceMetadataFilter(path, operator, value) != nil {
			continue
		}
		out = append(out, domain.ResourceMetadataFilter{Path: path, Operator: operator, Value: value})
	}
	return out
}

func datasetSnapshotResourceQuerySelector(query *domain.DatasetSnapshotResourceQuery) domain.JSONMap {
	if query == nil {
		return nil
	}
	selector := domain.JSONMap{}
	if strings.TrimSpace(query.Query) != "" {
		selector["q"] = strings.TrimSpace(query.Query)
	}
	if strings.TrimSpace(query.Kind) != "" {
		selector["kind"] = strings.TrimSpace(query.Kind)
	}
	if strings.TrimSpace(query.Source) != "" {
		selector["source"] = strings.TrimSpace(query.Source)
	}
	if strings.TrimSpace(query.ProjectID) != "" {
		selector["project_id"] = strings.TrimSpace(query.ProjectID)
	}
	if strings.TrimSpace(query.Sharing) != "" {
		selector["sharing"] = strings.TrimSpace(query.Sharing)
	}
	if len(query.Tags) > 0 {
		selector["tags"] = append([]string(nil), query.Tags...)
	}
	if len(query.Descriptors) > 0 {
		selector["descriptors"] = append([]string(nil), query.Descriptors...)
	}
	if len(query.MetadataFilters) > 0 {
		filters := make([]domain.JSONMap, 0, len(query.MetadataFilters))
		for _, filter := range query.MetadataFilters {
			filters = append(filters, domain.JSONMap{
				"path":     filter.Path,
				"operator": filter.Operator,
				"value":    filter.Value,
			})
		}
		selector["metadata_filters"] = filters
	}
	if !query.CreatedAfter.IsZero() {
		selector["created_after"] = query.CreatedAfter.UTC().Format(time.RFC3339Nano)
	}
	if !query.CreatedBefore.IsZero() {
		selector["created_before"] = query.CreatedBefore.UTC().Format(time.RFC3339Nano)
	}
	if strings.TrimSpace(query.ProcessingStatus) != "" {
		selector["processing_status"] = strings.TrimSpace(query.ProcessingStatus)
	}
	return selector
}

func resourceVisibleToPrincipal(resource resourceRecord, principal requestPrincipal) bool {
	owner := resource.Principal
	if strings.TrimSpace(owner.UserID) == "" {
		return strings.TrimSpace(principal.UserID) == "local-user"
	}
	if strings.TrimSpace(owner.UserID) != strings.TrimSpace(principal.UserID) {
		return false
	}
	ownerOrg := strings.TrimSpace(owner.OrgID)
	if ownerOrg == "" {
		return true
	}
	return ownerOrg == strings.TrimSpace(principal.OrgID)
}

func resourceOwnedByPrincipal(resource domain.ResourceRecord, principal requestPrincipal) bool {
	if strings.TrimSpace(resource.OwnerUserID) != strings.TrimSpace(principal.UserID) {
		return false
	}
	ownerOrg := strings.TrimSpace(resource.OwnerOrgID)
	return ownerOrg == "" || ownerOrg == strings.TrimSpace(principal.OrgID)
}

func sortedJSONMapKeys(value domain.JSONMap) []string {
	keys := make([]string, 0, len(value))
	for key := range value {
		if trimmed := strings.TrimSpace(key); trimmed != "" {
			keys = append(keys, trimmed)
		}
	}
	sort.Strings(keys)
	return keys
}

func resourceMatchesQuery(resource resourceRecord, query string) bool {
	if query == "" {
		return true
	}
	candidates := []string{
		resource.OriginalName,
		resource.FileID,
		resource.SourceURI,
		resource.ContentType,
		resource.ResourceKind,
		resource.SourceType,
		resource.SHA256,
		resource.ProjectID,
	}
	for _, candidate := range candidates {
		if strings.Contains(strings.ToLower(strings.TrimSpace(candidate)), query) {
			return true
		}
	}
	for _, tag := range resource.Tags {
		if strings.Contains(strings.ToLower(strings.TrimSpace(tag)), query) {
			return true
		}
	}
	return false
}

func resourceRecordMatchesTags(resource resourceRecord, tags []string) bool {
	required := uniqueTrimmedStringValues(tags)
	if len(required) == 0 {
		return true
	}
	have := map[string]bool{}
	for _, tag := range resource.Tags {
		have[strings.ToLower(strings.TrimSpace(tag))] = true
	}
	for _, tag := range required {
		if !have[strings.ToLower(strings.TrimSpace(tag))] {
			return false
		}
	}
	return true
}

func resourceRecordMatchesCreatedRange(resource resourceRecord, createdAfter time.Time, createdBefore time.Time) bool {
	if createdAfter.IsZero() && createdBefore.IsZero() {
		return true
	}
	createdAt, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(resource.CreatedAt))
	if err != nil {
		createdAt, err = time.Parse(time.RFC3339, strings.TrimSpace(resource.CreatedAt))
	}
	if err != nil {
		return false
	}
	createdAt = createdAt.UTC()
	if !createdAfter.IsZero() && createdAt.Before(createdAfter.UTC()) {
		return false
	}
	if !createdBefore.IsZero() && createdAt.After(createdBefore.UTC()) {
		return false
	}
	return true
}

func (deps ServerDeps) handleGetResource(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	record, _, err := deps.findUploadResourceForRequest(r.Context(), root, deps.principalFromRequest(r, ""), chi.URLParam(r, "file_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, resourceResponse{Resource: record})
}

// handleDownloadResourceCollection streams a zip of a collection's member
// resources. Best-effort per member: an unreadable member is skipped and
// counted in a trailing manifest entry rather than aborting a long stream
// (write errors to the client DO abort). Bundle resources (directory trees)
// are walked into the archive under their resource name; only regular files
// are copied, so symlinks can never pull bytes from outside the upload root.
// Direct members only — child folders are separate collections.
// shareTarget is one pickable grantee: a same-org person or the org itself.
type shareTarget struct {
	Kind          string `json:"kind"`
	GranteeUserID string `json:"grantee_user_id,omitempty"`
	GranteeOrgID  string `json:"grantee_org_id,omitempty"`
	Label         string `json:"label"`
	Detail        string `json:"detail,omitempty"`
}

// handleListShareTargets resolves who a user can share with: people in their
// own organization (never a deployment-wide directory) plus the organization
// itself as a single "everyone" target. This is what makes sharing reliable —
// grantees are picked from real principals instead of typed as free text,
// which silently matched nobody (principal ids are synthetic, not emails).
func (deps ServerDeps) handleListShareTargets(w http.ResponseWriter, r *http.Request) {
	accounts, ok := deps.Store.(accountStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "share targets are not configured"})
		return
	}
	principal := deps.principalFromRequest(r, "")
	principalOrg := strings.TrimSpace(principal.OrgID)
	query := strings.TrimSpace(r.URL.Query().Get("q"))
	records, err := accounts.ListUsers(r.Context(), 500, query)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	targets := []shareTarget{}
	if principalOrg != "" && (query == "" || strings.Contains(strings.ToLower("everyone organization "+principalOrg), strings.ToLower(query))) {
		targets = append(targets, shareTarget{
			Kind:         "org",
			GranteeOrgID: principalOrg,
			Label:        "Everyone in your organization",
			Detail:       principalOrg,
		})
	}
	const maxPeople = 20
	people := 0
	for _, record := range records {
		if people >= maxPeople {
			break
		}
		if record.UserID == principal.UserID {
			continue
		}
		if principalOrg == "" || strings.TrimSpace(record.OrgID) != principalOrg {
			continue
		}
		if strings.EqualFold(strings.TrimSpace(record.Status), "rejected") {
			continue
		}
		label := strings.TrimSpace(record.DisplayName)
		if label == "" {
			label = strings.TrimSpace(record.Email)
		}
		if label == "" {
			label = record.UserID
		}
		targets = append(targets, shareTarget{
			Kind:          "user",
			GranteeUserID: record.UserID,
			Label:         label,
			Detail:        strings.TrimSpace(record.Email),
		})
		people++
	}
	writeJSON(w, http.StatusOK, map[string]any{"targets": targets})
}

func (deps ServerDeps) handleDownloadResourceCollection(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	collections, ok := deps.resourceCollectionStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
		return
	}
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		writeStoreError(w, store.ErrNotFound)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	collectionID := strings.TrimSpace(chi.URLParam(r, "collection_id"))
	collection, err := collections.GetResourceCollectionForUser(r.Context(), collectionID, principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	members, err := collections.ListResourcesForCollectionForUser(r.Context(), domain.ResourceCollectionResourceListInput{
		CollectionID: collection.CollectionID,
		UserID:       principal.UserID,
		OrgID:        principal.OrgID,
		Limit:        uploadSessionMaxFilesPerBatch,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}

	archiveName := strings.TrimSpace(collection.Name)
	if archiveName == "" {
		archiveName = collection.CollectionID
	}
	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", archiveName+".zip"))
	w.WriteHeader(http.StatusOK)
	archive := zip.NewWriter(w)
	defer archive.Close()

	usedEntryNames := map[string]int{}
	uniqueEntryName := func(name string) string {
		if strings.TrimSpace(name) == "" {
			name = "resource"
		}
		count := usedEntryNames[name]
		usedEntryNames[name] = count + 1
		if count == 0 {
			return name
		}
		extension := filepath.Ext(name)
		return fmt.Sprintf("%s (%d)%s", strings.TrimSuffix(name, extension), count, extension)
	}

	skipped := 0
	for _, member := range members.Resources {
		resource, memberErr := catalog.GetResourceForUser(r.Context(), member.ResourceID, principal.UserID, principal.OrgID)
		if memberErr != nil {
			skipped++
			continue
		}
		path, pathErr := resolveCatalogResourcePath(root, resource)
		if pathErr != nil {
			skipped++
			continue
		}
		info, statErr := os.Stat(path)
		if statErr != nil {
			skipped++
			continue
		}
		memberName := strings.TrimSpace(resource.OriginalName)
		if memberName == "" {
			memberName = resource.ResourceID
		}
		if info.IsDir() {
			base := uniqueEntryName(memberName)
			walkErr := filepath.Walk(path, func(memberPath string, fileInfo os.FileInfo, walkErr error) error {
				if walkErr != nil || fileInfo == nil || !fileInfo.Mode().IsRegular() {
					return nil
				}
				relative, relErr := filepath.Rel(path, memberPath)
				if relErr != nil {
					return nil
				}
				file, openErr := os.Open(memberPath)
				if openErr != nil {
					return nil
				}
				defer file.Close()
				entry, entryErr := archive.Create(base + "/" + filepath.ToSlash(relative))
				if entryErr != nil {
					return entryErr
				}
				_, copyErr := io.Copy(entry, file)
				return copyErr
			})
			if walkErr != nil {
				// Zip-writer/client write failure mid-stream: nothing more to send.
				return
			}
			continue
		}
		file, openErr := os.Open(path)
		if openErr != nil {
			skipped++
			continue
		}
		entry, entryErr := archive.Create(uniqueEntryName(memberName))
		if entryErr != nil {
			file.Close()
			return
		}
		if _, copyErr := io.Copy(entry, file); copyErr != nil {
			file.Close()
			return
		}
		file.Close()
	}
	if skipped > 0 {
		if entry, entryErr := archive.Create("SKIPPED.txt"); entryErr == nil {
			fmt.Fprintf(entry, "%d member resource(s) could not be read and were skipped\n", skipped)
		}
	}
}

func (deps ServerDeps) handleDownloadResource(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		writeStoreError(w, store.ErrNotFound)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	fileID := chi.URLParam(r, "file_id")
	principal := deps.principalFromRequest(r, "")
	resource, err := catalog.GetResourceForUser(r.Context(), fileID, principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	path, err := resolveCatalogResourcePath(root, resource)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			writeStoreError(w, store.ErrNotFound)
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if info.IsDir() {
		writeStoreError(w, store.ErrNotFound)
		return
	}
	filename := strings.TrimSpace(resource.OriginalName)
	if filename == "" {
		filename = strings.TrimSpace(resource.ResourceID)
	}
	if filename == "" {
		filename = "resource"
	}
	contentType := strings.TrimSpace(resource.ContentType)
	if contentType == "" || contentType == "application/octet-stream" {
		contentType = contentTypeForUpload(filename, contentType)
	}
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	w.Header().Set("Content-Type", contentType)
	// Defense-in-depth: never let the browser sniff a declared text type up into
	// active content (HTML/SVG), which matters most on the inline path below.
	w.Header().Set("X-Content-Type-Options", "nosniff")
	// Default to attachment (the Download button). The text/data viewer reads bytes
	// via fetch()+Range, so disposition is irrelevant to it; ?disposition=inline is
	// an opt-in for an "open raw" affordance, restricted to safe text-like types so
	// it can never coax the browser into rendering active content inline.
	disposition := "attachment"
	if strings.EqualFold(strings.TrimSpace(r.URL.Query().Get("disposition")), "inline") && isInlineSafeContentType(contentType) {
		disposition = "inline"
	}
	w.Header().Set("Content-Disposition", mime.FormatMediaType(disposition, map[string]string{"filename": filename}))
	// Let http.ServeContent own Content-Length: it rewrites it to the partial length
	// on a 206 Range response, so pre-setting the full size here would be wrong for
	// the bounded head-fetch the viewer relies on.
	http.ServeContent(w, r, filename, info.ModTime(), file)
}

// isInlineSafeContentType reports whether a content type is safe to serve with an
// inline Content-Disposition (plain text / data formats the browser will not treat
// as active content). HTML/SVG are deliberately excluded.
func isInlineSafeContentType(contentType string) bool {
	normalized := strings.ToLower(strings.TrimSpace(contentType))
	if idx := strings.IndexByte(normalized, ';'); idx >= 0 {
		normalized = strings.TrimSpace(normalized[:idx])
	}
	switch normalized {
	case "text/plain", "text/csv", "text/tab-separated-values", "text/markdown",
		"text/yaml", "application/json", "application/xml", "text/xml",
		"application/x-yaml", "application/yaml", "application/x-ndjson":
		return true
	}
	if strings.HasSuffix(normalized, "+json") || strings.HasSuffix(normalized, "+xml") {
		return true
	}
	return false
}

func (deps ServerDeps) handlePatchResource(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		writeStoreError(w, store.ErrNotFound)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var req patchResourceRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	renamedOriginalName := strings.TrimSpace(req.OriginalName)
	if len(req.Metadata) == 0 && renamedOriginalName == "" {
		writeError(w, http.StatusBadRequest, errors.New("resource name or metadata patch is required"))
		return
	}
	fileID := chi.URLParam(r, "file_id")
	principal := deps.principalFromRequest(r, "")
	resource, err := catalog.GetResourceForUser(r.Context(), fileID, principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if !resourceOwnedByPrincipal(resource, principal) {
		writeStoreError(w, store.ErrNotFound)
		return
	}
	rawCalibration, calibrationPatch := req.Metadata["ultra_viewer_calibration_v1"]
	if calibrationPatch {
		sourcePath, pathErr := resolveCatalogResourcePath(root, resource)
		if pathErr != nil {
			writeStoreError(w, pathErr)
			return
		}
		sourceInfo, timeCount, channelCount, _, authorityErr := deps.sourceImageServiceViewerInfo(
			r.Context(),
			sourcePath,
		)
		if authorityErr != nil {
			writeImageSourceAuthorityError(w, authorityErr)
			return
		}
		sanitizeScalarMaskCapability(
			sourceInfo,
			deps.resourceRecordFromCatalog(root, resource),
		)
		if _, capable := jsonObject(sourceInfo["scalar_mask_capability"]); !capable {
			writeError(w, http.StatusUnprocessableEntity, errors.New("mask calibration is unsupported for this source"))
			return
		}
		sanitized, expectedRevisions, validationErr := validateViewerCalibrationMetadata(
			rawCalibration,
			resource.SHA256,
			timeCount,
			channelCount,
			capabilityDtype(sourceInfo),
		)
		if validationErr != nil {
			writeError(w, http.StatusBadRequest, validationErr)
			return
		}
		req.Metadata["ultra_viewer_calibration_v1"] = sanitized
		req.CalibrationExpectedSourceSHA256 = resource.SHA256
		req.CalibrationSelectionExpectedRevisions = expectedRevisions
	}
	now := domain.Now()
	updated := resource
	if len(req.Metadata) > 0 {
		metadataStore, ok := deps.Store.(resourceMetadataPatchStore)
		if !ok {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource metadata editing is not configured"})
			return
		}
		metadataKeys := sortedJSONMapKeys(req.Metadata)
		updated, err = metadataStore.MergeResourceMetadataForUser(r.Context(), domain.MergeResourceMetadataInput{
			ResourceID:                 fileID,
			UserID:                     principal.UserID,
			OrgID:                      principal.OrgID,
			Patch:                      req.Metadata,
			ExpectedSourceSHA256:       req.CalibrationExpectedSourceSHA256,
			SelectionExpectedRevisions: req.CalibrationSelectionExpectedRevisions,
			UpdatedAt:                  now,
		})
		if err != nil {
			writeStoreError(w, err)
			return
		}
		if calibrationPatch {
			if _, recorded := deps.appendResourceEvent(
				r.Context(),
				updated.ResourceID,
				principal,
				"resource.viewer_calibration_updated",
				domain.JSONMap{
					"source_sha256": resource.SHA256,
					"calibration":   req.Metadata["ultra_viewer_calibration_v1"],
					"updated_at":    now.UTC().Format(time.RFC3339Nano),
				},
			); !recorded {
				writeError(
					w,
					http.StatusInternalServerError,
					errors.New("viewer calibration persisted but its audit event could not be recorded"),
				)
				return
			}
		}
		deps.recordResourceEvent(r.Context(), updated.ResourceID, principal, "resource.metadata_updated", domain.JSONMap{
			"metadata_keys":      metadataKeys,
			"metadata_key_count": len(metadataKeys),
			"source":             "resource_patch",
			"updated_at":         now.UTC().Format(time.RFC3339Nano),
		})
	}
	if renamedOriginalName != "" && renamedOriginalName != updated.OriginalName {
		renameStore, ok := deps.Store.(resourceRenameStore)
		if !ok {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource renaming is not configured"})
			return
		}
		previousName := updated.OriginalName
		updated, err = renameStore.RenameResourceForUser(r.Context(), domain.RenameResourceInput{
			ResourceID:   fileID,
			UserID:       principal.UserID,
			OrgID:        principal.OrgID,
			OriginalName: renamedOriginalName,
			UpdatedAt:    now,
		})
		if err != nil {
			writeStoreError(w, err)
			return
		}
		deps.recordResourceEvent(r.Context(), updated.ResourceID, principal, "resource.renamed", domain.JSONMap{
			"previous_name": previousName,
			"name":          updated.OriginalName,
			"source":        "resource_patch",
			"updated_at":    now.UTC().Format(time.RFC3339Nano),
		})
	}
	writeJSON(w, http.StatusOK, resourceResponse{Resource: deps.resourceRecordFromCatalog(root, updated)})
}

func (deps ServerDeps) handleDeleteResource(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	fileID := chi.URLParam(r, "file_id")
	principal := deps.principalFromRequest(r, "")
	if catalog, ok := deps.resourceCatalogStore(); ok {
		if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		record, err := catalog.SoftDeleteResourceForUser(r.Context(), fileID, principal.UserID, principal.OrgID, domain.Now())
		if err != nil {
			writeStoreError(w, err)
			return
		}
		deps.recordResourceEvent(r.Context(), record.ResourceID, principal, "resource.deleted", nil)
		writeJSON(w, http.StatusOK, map[string]any{"deleted": true, "soft_deleted": true, "file_id": fileID})
		return
	}
	_, path, err := deps.findUploadResourceForRequest(r.Context(), root, principal, fileID)
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

func (deps ServerDeps) handleRestoreResource(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		writeStoreError(w, store.ErrNotFound)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	principal := deps.principalFromRequest(r, "")
	resource, err := catalog.RestoreResourceForUser(r.Context(), chi.URLParam(r, "file_id"), principal.UserID, principal.OrgID, domain.Now())
	if err != nil {
		writeStoreError(w, err)
		return
	}
	deps.recordResourceEvent(r.Context(), resource.ResourceID, principal, "resource.restored", nil)
	writeJSON(w, http.StatusOK, resourceResponse{Resource: deps.resourceRecordFromCatalog(root, resource)})
}

func (deps ServerDeps) handleBulkDeleteResources(w http.ResponseWriter, r *http.Request) {
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource catalog is not configured"})
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var request bulkLifecycleResourcesRequest
	if !decodeJSON(w, r, &request) {
		return
	}
	resourceIDs := uniqueTrimmedStringValues(request.ResourceIDs)
	if len(resourceIDs) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("resource_ids must include at least one resource"))
		return
	}
	if len(resourceIDs) > uploadSessionMaxFilesPerBatch {
		writeError(w, http.StatusBadRequest, fmt.Errorf("resource_ids cannot include more than %d resources", uploadSessionMaxFilesPerBatch))
		return
	}
	principal := deps.principalFromRequest(r, "")
	for _, resourceID := range resourceIDs {
		resource, err := catalog.GetResourceForUser(r.Context(), resourceID, principal.UserID, principal.OrgID)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		if !resourceOwnedByPrincipal(resource, principal) {
			writeStoreError(w, store.ErrNotFound)
			return
		}
	}
	deletedAt := domain.Now()
	records := make([]resourceRecord, 0, len(resourceIDs))
	events := make([]domain.ResourceEventRecord, 0, len(resourceIDs))
	for _, resourceID := range resourceIDs {
		record, err := catalog.SoftDeleteResourceForUser(r.Context(), resourceID, principal.UserID, principal.OrgID, deletedAt)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		records = append(records, deps.resourceRecordFromCatalog(root, record))
		if event, ok := deps.appendResourceEvent(r.Context(), record.ResourceID, principal, "resource.deleted", domain.JSONMap{
			"source":      "resources_bulk_delete",
			"deleted_at":  deletedAt.UTC().Format(time.RFC3339Nano),
			"batch_count": len(resourceIDs),
		}); ok {
			events = append(events, event)
		}
	}
	writeJSON(w, http.StatusOK, bulkLifecycleResourcesResponse{
		Count:     len(records),
		Resources: records,
		Events:    events,
	})
}

func (deps ServerDeps) handleBulkRestoreResources(w http.ResponseWriter, r *http.Request) {
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource catalog is not configured"})
		return
	}
	ownerLookup, ok := deps.resourceOwnerLookupStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource owner lookup is not configured"})
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var request bulkLifecycleResourcesRequest
	if !decodeJSON(w, r, &request) {
		return
	}
	resourceIDs := uniqueTrimmedStringValues(request.ResourceIDs)
	if len(resourceIDs) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("resource_ids must include at least one resource"))
		return
	}
	if len(resourceIDs) > uploadSessionMaxFilesPerBatch {
		writeError(w, http.StatusBadRequest, fmt.Errorf("resource_ids cannot include more than %d resources", uploadSessionMaxFilesPerBatch))
		return
	}
	principal := deps.principalFromRequest(r, "")
	for _, resourceID := range resourceIDs {
		resource, err := ownerLookup.GetResourceForOwner(r.Context(), resourceID, principal.UserID, principal.OrgID)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		if !resourceOwnedByPrincipal(resource, principal) {
			writeStoreError(w, store.ErrNotFound)
			return
		}
	}
	restoredAt := domain.Now()
	records := make([]resourceRecord, 0, len(resourceIDs))
	events := make([]domain.ResourceEventRecord, 0, len(resourceIDs))
	for _, resourceID := range resourceIDs {
		record, err := catalog.RestoreResourceForUser(r.Context(), resourceID, principal.UserID, principal.OrgID, restoredAt)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		records = append(records, deps.resourceRecordFromCatalog(root, record))
		if event, ok := deps.appendResourceEvent(r.Context(), record.ResourceID, principal, "resource.restored", domain.JSONMap{
			"source":      "resources_bulk_restore",
			"restored_at": restoredAt.UTC().Format(time.RFC3339Nano),
			"batch_count": len(resourceIDs),
		}); ok {
			events = append(events, event)
		}
	}
	writeJSON(w, http.StatusOK, bulkLifecycleResourcesResponse{
		Count:     len(records),
		Resources: records,
		Events:    events,
	})
}

func (deps ServerDeps) handleListResourceEventLog(w http.ResponseWriter, r *http.Request) {
	eventLog, ok := deps.Store.(resourceEventLogStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource event log is not configured"})
		return
	}
	principal := deps.principalFromRequest(r, "")
	limit := clampLimit(parseLimit(r, 200), 1000)
	offset := parseOffset(r)
	page, err := eventLog.ListResourceEventsForUser(r.Context(), domain.ResourceEventListInput{
		UserID:      principal.UserID,
		OrgID:       principal.OrgID,
		ResourceID:  strings.TrimSpace(r.URL.Query().Get("resource_id")),
		EventType:   strings.TrimSpace(r.URL.Query().Get("event_type")),
		ActorUserID: strings.TrimSpace(r.URL.Query().Get("actor_user_id")),
		Limit:       limit,
		Offset:      offset,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, resourceEventListResponse{
		Count:      len(page.Events),
		TotalCount: page.TotalCount,
		Limit:      page.Limit,
		Offset:     page.Offset,
		Events:     page.Events,
	})
}

func (deps ServerDeps) handleListResourceEvents(w http.ResponseWriter, r *http.Request) {
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource catalog is not configured"})
		return
	}
	eventLog, ok := deps.Store.(resourceEventLogStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource event log is not configured"})
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	fileID := chi.URLParam(r, "file_id")
	principal := deps.principalFromRequest(r, "")
	if _, err := catalog.GetResourceForUser(r.Context(), fileID, principal.UserID, principal.OrgID); err != nil {
		writeStoreError(w, err)
		return
	}
	events, err := eventLog.ListResourceEvents(r.Context(), fileID, parseLimit(r, 200))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, resourceEventsResponse{ResourceID: fileID, Count: len(events), Events: events})
}

func (deps ServerDeps) handleBulkTagResources(w http.ResponseWriter, r *http.Request) {
	tagger, ok := deps.Store.(resourceTagStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource tags are not configured"})
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var request bulkTagResourcesRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if len(uniqueTrimmedStringValues(request.ResourceIDs)) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("resource_ids are required"))
		return
	}
	if len(uniqueTrimmedStringValues(request.Tags)) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("tags are required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	result, err := tagger.BulkTagResourcesForUser(r.Context(), domain.BulkTagResourcesInput{
		OwnerUserID: principal.UserID,
		OwnerOrgID:  principal.OrgID,
		ActorUserID: principal.UserID,
		ActorOrgID:  principal.OrgID,
		ResourceIDs: request.ResourceIDs,
		Tags:        request.Tags,
		Metadata:    request.Metadata,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	records := make([]resourceRecord, 0, len(result.Resources))
	for _, resource := range result.Resources {
		records = append(records, deps.resourceRecordFromCatalog(root, resource))
	}
	writeJSON(w, http.StatusOK, bulkTagResourcesResponse{
		Count:     result.UpdatedCount,
		Resources: records,
		Events:    result.Events,
	})
}

func (deps ServerDeps) handleCreateResourceShareGrant(w http.ResponseWriter, r *http.Request) {
	shareGrants, ok := deps.Store.(resourceShareGrantStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource sharing is not configured"})
		return
	}
	fileID := strings.TrimSpace(chi.URLParam(r, "file_id"))
	if fileID == "" {
		writeError(w, http.StatusBadRequest, errors.New("resource id is required"))
		return
	}
	var request createResourceShareGrantRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	role := strings.TrimSpace(request.Role)
	if role == "" {
		role = "read"
	}
	if role != "read" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("unsupported resource share role %q", role))
		return
	}
	granteeUserID := strings.TrimSpace(request.GranteeUserID)
	granteeOrgID := strings.TrimSpace(request.GranteeOrgID)
	if request.Public {
		granteeUserID = domain.PublicResourceGranteeUserID
		granteeOrgID = ""
	}
	if !request.Public && granteeUserID == "" && granteeOrgID == "" {
		writeError(w, http.StatusBadRequest, errors.New("grantee_user_id or grantee_org_id is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	grant, err := shareGrants.CreateResourceShareGrant(r.Context(), domain.CreateResourceShareGrantInput{
		ResourceID:      fileID,
		OwnerUserID:     principal.UserID,
		OwnerOrgID:      principal.OrgID,
		OwnerRole:       principal.Role,
		GranteeUserID:   granteeUserID,
		GranteeOrgID:    granteeOrgID,
		Public:          request.Public,
		Role:            role,
		Status:          "active",
		CreatedByUserID: principal.UserID,
		Metadata:        request.Metadata,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	deps.recordResourceEvent(r.Context(), grant.ResourceID, principal, "resource.shared", domain.JSONMap{
		"grant_id":        grant.GrantID,
		"grantee_user_id": grant.GranteeUserID,
		"grantee_org_id":  grant.GranteeOrgID,
		"public":          request.Public,
		"role":            grant.Role,
	})
	writeJSON(w, http.StatusCreated, resourceShareGrantResponse{Grant: grant})
}

func (deps ServerDeps) handleCreateResourceShareGrants(w http.ResponseWriter, r *http.Request) {
	shareGrants, ok := deps.Store.(resourceShareGrantStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource sharing is not configured"})
		return
	}
	var request createResourceShareGrantsRequest
	if !decodeJSON(w, r, &request) {
		return
	}
	resourceIDs := uniqueTrimmedStringValues(request.ResourceIDs)
	if len(resourceIDs) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("resource_ids must include at least one resource"))
		return
	}
	if len(resourceIDs) > uploadSessionMaxFilesPerBatch {
		writeError(w, http.StatusBadRequest, fmt.Errorf("resource_ids cannot include more than %d resources", uploadSessionMaxFilesPerBatch))
		return
	}
	role := strings.TrimSpace(request.Role)
	if role == "" {
		role = "read"
	}
	if role != "read" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("unsupported resource share role %q", role))
		return
	}
	granteeUserID := strings.TrimSpace(request.GranteeUserID)
	granteeOrgID := strings.TrimSpace(request.GranteeOrgID)
	if request.Public {
		granteeUserID = domain.PublicResourceGranteeUserID
		granteeOrgID = ""
	}
	if !request.Public && granteeUserID == "" && granteeOrgID == "" {
		writeError(w, http.StatusBadRequest, errors.New("grantee_user_id or grantee_org_id is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	for _, resourceID := range resourceIDs {
		if _, err := shareGrants.ListResourceShareGrantsForResource(r.Context(), domain.ListResourceShareGrantsInput{
			ResourceID:  resourceID,
			OwnerUserID: principal.UserID,
			OwnerOrgID:  principal.OrgID,
			Limit:       1,
		}); err != nil {
			writeStoreError(w, err)
			return
		}
	}
	grants := make([]domain.ResourceShareGrantRecord, 0, len(resourceIDs))
	for _, resourceID := range resourceIDs {
		grant, err := shareGrants.CreateResourceShareGrant(r.Context(), domain.CreateResourceShareGrantInput{
			ResourceID:      resourceID,
			OwnerUserID:     principal.UserID,
			OwnerOrgID:      principal.OrgID,
			OwnerRole:       principal.Role,
			GranteeUserID:   granteeUserID,
			GranteeOrgID:    granteeOrgID,
			Public:          request.Public,
			Role:            role,
			Status:          "active",
			CreatedByUserID: principal.UserID,
			Metadata:        request.Metadata,
		})
		if err != nil {
			writeStoreError(w, err)
			return
		}
		grants = append(grants, grant)
		deps.recordResourceEvent(r.Context(), grant.ResourceID, principal, "resource.shared", domain.JSONMap{
			"grant_id":        grant.GrantID,
			"grantee_user_id": grant.GranteeUserID,
			"grantee_org_id":  grant.GranteeOrgID,
			"public":          request.Public,
			"role":            grant.Role,
		})
	}
	writeJSON(w, http.StatusCreated, resourceShareGrantsCreateResponse{Count: len(grants), Grants: grants})
}

func (deps ServerDeps) handleListResourceShareGrants(w http.ResponseWriter, r *http.Request) {
	shareGrants, ok := deps.Store.(resourceShareGrantStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource sharing is not configured"})
		return
	}
	fileID := strings.TrimSpace(chi.URLParam(r, "file_id"))
	if fileID == "" {
		writeError(w, http.StatusBadRequest, errors.New("resource id is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	grants, err := shareGrants.ListResourceShareGrantsForResource(r.Context(), domain.ListResourceShareGrantsInput{
		ResourceID:  fileID,
		OwnerUserID: principal.UserID,
		OwnerOrgID:  principal.OrgID,
		Status:      strings.TrimSpace(r.URL.Query().Get("status")),
		Limit:       clampLimit(parseLimit(r, 200), 1000),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, resourceShareGrantsResponse{ResourceID: fileID, Count: len(grants), Grants: grants})
}

func (deps ServerDeps) handleRevokeResourceShareGrant(w http.ResponseWriter, r *http.Request) {
	shareGrants, ok := deps.Store.(resourceShareGrantStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource sharing is not configured"})
		return
	}
	fileID := strings.TrimSpace(chi.URLParam(r, "file_id"))
	grantID := strings.TrimSpace(chi.URLParam(r, "grant_id"))
	if fileID == "" || grantID == "" {
		writeError(w, http.StatusBadRequest, errors.New("resource id and grant id are required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	grant, err := shareGrants.RevokeResourceShareGrant(r.Context(), domain.RevokeResourceShareGrantInput{
		ResourceID:  fileID,
		GrantID:     grantID,
		OwnerUserID: principal.UserID,
		OwnerOrgID:  principal.OrgID,
		RevokedAt:   domain.Now(),
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	deps.recordResourceEvent(r.Context(), grant.ResourceID, principal, "resource.share_revoked", domain.JSONMap{
		"grant_id":        grant.GrantID,
		"grantee_user_id": grant.GranteeUserID,
		"grantee_org_id":  grant.GranteeOrgID,
		"role":            grant.Role,
	})
	writeJSON(w, http.StatusOK, resourceShareGrantResponse{Grant: grant})
}

func (deps ServerDeps) handleServeUpload(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	record, path, err := deps.findUploadResourceForRequest(r.Context(), root, deps.principalFromRequest(r, ""), chi.URLParam(r, "file_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	// OME-Zarr is a directory, not a file — http.ServeFile below would fail on it. Render a
	// display/preview PNG natively via the ngff-service (its /slice is the omero-aware plane
	// render). Covers both /display and /preview, which share this handler.
	if deps.servesViaNgff(record, path) {
		q := url.Values{"path": {path}}
		for _, key := range []string{"t", "z", "channels"} {
			if v := strings.TrimSpace(r.URL.Query().Get(key)); v != "" {
				q.Set(key, v)
			}
		}
		deps.ngffDeps().proxyImageServiceCached(w, r, "/slice", q)
		return
	}
	if isNiftiUpload(record.OriginalName, record.ContentType) {
		if err := serveNiftiSliceAsPNG(w, path, r); err != nil {
			writeError(w, http.StatusUnsupportedMediaType, err)
		}
		return
	}
	if uploadRequestNeedsBrowserPNG(record.OriginalName, record.ContentType, r) {
		if err := serveUploadAsPNG(w, path, uploadPreviewTransformFromRequest(r), r); err != nil {
			writeError(w, http.StatusUnsupportedMediaType, err)
		}
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

func (deps ServerDeps) handleGetUploadScalarVolume(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	record, path, err := deps.findUploadResourceForRequest(r.Context(), root, deps.principalFromRequest(r, ""), chi.URLParam(r, "file_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if !isNiftiUpload(record.OriginalName, record.ContentType) {
		writeError(w, http.StatusUnsupportedMediaType, errors.New("upload scalar volume is only available for NIfTI resources"))
		return
	}
	volume, err := loadNiftiScalarVolumeAt(path, parseUploadScalarTimeIndex(r), parseUploadScalarChannelIndex(r))
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, err)
		return
	}
	volumeBytes := int64(len(volume.Data))
	if !scalarVolumeInFlightBudget.tryAcquire(volumeBytes) {
		w.Header().Set("Retry-After", "1")
		writeError(
			w,
			http.StatusServiceUnavailable,
			errors.New("scalar volume in-flight byte budget is exhausted"),
		)
		return
	}
	defer scalarVolumeInFlightBudget.release(volumeBytes)
	maybeDecompressNiftiSidecar(path, volume.TimeCount)
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Cache-Control", "private, max-age=3600")
	w.Header().Set("x-volume-time", strconv.Itoa(volume.TimeIndex))
	w.Header().Set("x-volume-time-count", strconv.Itoa(volume.TimeCount))
	w.Header().Set("x-volume-width", strconv.Itoa(volume.Width))
	w.Header().Set("x-volume-height", strconv.Itoa(volume.Height))
	w.Header().Set("x-volume-depth", strconv.Itoa(volume.Depth))
	w.Header().Set("x-volume-dtype", volume.DType)
	w.Header().Set("x-volume-bytes-per-voxel", strconv.Itoa(volume.BytesPerVoxel))
	w.Header().Set("x-volume-raw-min", formatScalarHeaderFloat(volume.RawMin))
	w.Header().Set("x-volume-raw-max", formatScalarHeaderFloat(volume.RawMax))
	w.Header().Set("x-volume-channel", strconv.Itoa(volume.ChannelIndex))
	w.Header().Set("x-volume-source-width", strconv.Itoa(volume.Width))
	w.Header().Set("x-volume-source-height", strconv.Itoa(volume.Height))
	w.Header().Set("x-volume-source-depth", strconv.Itoa(volume.Depth))
	w.Header().Set("x-volume-downsample-x", "1")
	w.Header().Set("x-volume-downsample-y", "1")
	w.Header().Set("x-volume-downsample-z", "1")
	w.Header().Set("x-volume-preview-policy", "native-exact-v1")
	w.Header().Set("x-volume-sampling", "box")
	// Rescale to physical units (HU/SUV) so the client can window in true
	// intensities: physical = slope*code + inter.
	w.Header().Set("x-volume-scl-slope", formatScalarHeaderFloat(volume.SclSlope))
	w.Header().Set("x-volume-scl-inter", formatScalarHeaderFloat(volume.SclInter))
	_, _ = w.Write(volume.Data)
}

func (deps ServerDeps) handleGetUploadHistogram(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	record, path, err := deps.findUploadResourceForRequest(r.Context(), root, deps.principalFromRequest(r, ""), chi.URLParam(r, "file_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if isNiftiUpload(record.OriginalName, record.ContentType) {
		binCount := parseUploadHistogramBins(r)
		channelIndex := parseUploadScalarChannelIndex(r)
		volume, err := loadNiftiScalarVolume(path, channelIndex)
		if err != nil {
			writeError(w, http.StatusUnsupportedMediaType, err)
			return
		}
		result, err := histogramForNiftiScalarVolume(volume, binCount, parseUploadHistogramTimeIndex(r))
		if err != nil {
			writeError(w, http.StatusUnsupportedMediaType, err)
			return
		}
		result.FileID = record.FileID
		writeJSON(w, http.StatusOK, result)
		return
	}
	if !strings.HasPrefix(strings.ToLower(record.ContentType), "image/") && !isTIFFUpload(record.OriginalName, record.ContentType) {
		writeError(w, http.StatusUnsupportedMediaType, errors.New("upload histogram is only available for image resources"))
		return
	}
	img, err := decodeUploadImage(path)
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, err)
		return
	}
	binCount := parseUploadHistogramBins(r)
	channelIndices, channelsRequested := parseUploadHistogramChannels(r.URL.Query().Get("channels"), uploadHistogramDefaultChannels(img))
	timeIndex := parseUploadHistogramTimeIndex(r)
	result, err := histogramForUploadImage(img, binCount, channelIndices, channelsRequested, timeIndex)
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, err)
		return
	}
	result.FileID = record.FileID
	writeJSON(w, http.StatusOK, result)
}

func (deps ServerDeps) handleGetUploadCaption(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	record, path, err := deps.findUploadResourceForRequest(r.Context(), root, deps.principalFromRequest(r, ""), chi.URLParam(r, "file_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	imageInfo := uploadImageDescriptorForPath(path, record.ContentType)
	width, height, warnings := imageInfo.Width, imageInfo.Height, imageInfo.Warnings
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
	record, path, err := deps.findUploadResourceForRequest(r.Context(), root, deps.principalFromRequest(r, ""), chi.URLParam(r, "file_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if isNiftiUpload(record.OriginalName, record.ContentType) {
		deps.writeNiftiUploadViewer(w, record, path)
		return
	}
	imageInfo := uploadImageDescriptorForPath(path, record.ContentType)
	if imageInfo.OME != nil {
		deps.writeOMETiffUploadViewer(w, record, imageInfo)
		return
	}
	width, height, warnings := imageInfo.Width, imageInfo.Height, imageInfo.Warnings
	fileIDSegment := url.PathEscape(record.FileID)
	channelCount := imageInfo.ChannelCount
	isTIFF := isTIFFUpload(record.OriginalName, record.ContentType)
	isOME := isOmeTIFFName(record.OriginalName)
	modality := "image"
	reader := "go-image"
	arrayDType := imageInfo.ArrayDType
	viewerStatus := "ready"
	warmupMode := "lazy"
	nativeSupported := true
	tilePyramid := "none"
	if isTIFF {
		reader = "go-image+tiff"
		if strings.TrimSpace(arrayDType) == "" {
			arrayDType = "unknown"
		}
		viewerStatus = "preview-ready"
		warmupMode = "deferred"
		nativeSupported = false
		tilePyramid = "deferred"
		warnings = append(warnings, "TIFF first-plane preview is available; full multiscale tile and volume preparation is deferred.")
	}
	if isOME {
		modality = "microscopy"
		warnings = append(warnings, "OME metadata and multi-scene series extraction are not yet materialized in the native viewer.")
	}
	serviceURLs := map[string]any{
		"preview":   "/v2/uploads/" + fileIDSegment + "/preview",
		"display":   "/v2/uploads/" + fileIDSegment + "/display",
		"slice":     "/v2/uploads/" + fileIDSegment + "/slice",
		"histogram": "/v2/uploads/" + fileIDSegment + "/histogram",
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"kind":          "image",
		"file_id":       record.FileID,
		"original_name": record.OriginalName,
		"modality":      modality,
		"backend_mode":  "direct",
		"dims_order":    imageInfo.DimsOrder,
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
			"reader":       reader,
			"dims_order":   imageInfo.DimsOrder,
			"array_shape":  imageInfo.ArrayShape,
			"array_dtype":  arrayDType,
			"scene_count":  1,
			"warnings":     warnings,
			"content_type": record.ContentType,
			"size_bytes":   record.SizeBytes,
			"sha256":       record.SHA256,
		},
		"viewer": map[string]any{
			"status":             viewerStatus,
			"warmup_mode":        warmupMode,
			"backend_mode":       "direct",
			"default_surface":    "2d",
			"available_surfaces": []string{"2d", "metadata"},
			"service_urls":       serviceURLs,
			"asset_preparation": map[string]any{
				"status":                viewerStatus,
				"native_supported":      nativeSupported,
				"tile_pyramid":          tilePyramid,
				"volume_representation": "none",
			},
		},
	})
}

func (deps ServerDeps) writeOMETiffUploadViewer(w http.ResponseWriter, record resourceRecord, imageInfo uploadImageDescriptor) {
	meta := imageInfo.OME
	if meta == nil {
		writeError(w, http.StatusUnsupportedMediaType, errors.New("OME-TIFF metadata is unavailable"))
		return
	}
	fileIDSegment := url.PathEscape(record.FileID)
	serviceURLs := map[string]any{
		"preview":   "/v2/uploads/" + fileIDSegment + "/preview",
		"display":   "/v2/uploads/" + fileIDSegment + "/display",
		"slice":     "/v2/uploads/" + fileIDSegment + "/slice",
		"histogram": "/v2/uploads/" + fileIDSegment + "/histogram",
	}
	selectedZ := positiveIntOr(meta.SizeZ, 1) / 2
	selectedC := omeDefaultChannelIndex(meta)
	channelColors := omeChannelColorStrings(meta)
	// OME-TIFF is non-medical microscopy: 2D-only for now. The multichannel 3D render
	// is not shippable yet, so we lead with reliable 2D + first-class Z/T scrubbing.
	// volume_mode stays "slice_stack" below so the 2D Z scrub still knows it is a
	// stack — only the 3D surfaces are withheld. Mirrors the engine-backed viewerinfo
	// policy (3D surfaces are medical-only); medical NIfTI/DICOM keep 3D elsewhere.
	availableSurfaces := []string{"2d", "metadata"}
	displayCapabilities := []string{"slice_navigation", "intensity_window"}
	viewerCapabilities := []string{"webgl_first_paint", "direct_delivery", "linear_sampling", "slice_navigation"}
	measurementPolicy := "pixel-only"
	if omeHasPhysicalSpacing(meta) {
		displayCapabilities = append(displayCapabilities, "physical_scale")
		viewerCapabilities = append(viewerCapabilities, "physical_scale")
		measurementPolicy = "spacing-aware"
	}
	if meta.SizeC > 1 {
		displayCapabilities = append(displayCapabilities, "channel_visibility")
		viewerCapabilities = append(viewerCapabilities, "channel_selection")
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"kind":          "image",
		"file_id":       record.FileID,
		"original_name": record.OriginalName,
		"modality":      "microscopy",
		"backend_mode":  "direct",
		"dims_order":    imageInfo.DimsOrder,
		"axis_sizes": map[string]int{
			"T": positiveIntOr(meta.SizeT, 1),
			"C": positiveIntOr(meta.SizeC, 1),
			"Z": positiveIntOr(meta.SizeZ, 1),
			"Y": positiveIntOr(meta.SizeY, imageInfo.Height),
			"X": positiveIntOr(meta.SizeX, imageInfo.Width),
		},
		"selected_indices": map[string]int{"T": 0, "C": selectedC, "Z": selectedZ},
		"is_volume":        meta.SizeZ > 1,
		"is_timeseries":    meta.SizeT > 1,
		"is_multichannel":  meta.SizeC > 1,
		"phys": map[string]any{
			"resource_uniq":    record.FileID,
			"name":             record.OriginalName,
			"x":                positiveIntOr(meta.SizeX, imageInfo.Width),
			"y":                positiveIntOr(meta.SizeY, imageInfo.Height),
			"z":                positiveIntOr(meta.SizeZ, 1),
			"t":                positiveIntOr(meta.SizeT, 1),
			"ch":               positiveIntOr(meta.SizeC, 1),
			"pixel_depth":      omePixelDepth(meta),
			"pixel_format":     omePixelFormat(meta),
			"pixel_size":       []float64{positiveFloatOr(meta.PhysicalSizeX, 1), positiveFloatOr(meta.PhysicalSizeY, 1), positiveFloatOr(meta.PhysicalSizeZ, 1), 1},
			"pixel_units":      []string{nonEmptyString(meta.PhysicalUnitX, "px"), nonEmptyString(meta.PhysicalUnitY, "px"), nonEmptyString(meta.PhysicalUnitZ, "slice"), "frame"},
			"channel_names":    meta.ChannelNames,
			"display_channels": []int{selectedC},
			"channel_colors":   omeChannelColorPayload(meta),
			"units":            "physical",
		},
		"display_defaults": map[string]any{
			"enhancement":     "d",
			"negative":        false,
			"rotate":          0,
			"fusion_method":   "m",
			"channel_mode":    "single",
			"channels":        []int{selectedC},
			"channel_colors":  channelColors,
			"time_index":      0,
			"z_index":         selectedZ,
			"volume_channel":  selectedC,
			"volume_clip_min": map[string]float64{"x": 0, "y": 0, "z": 0},
			"volume_clip_max": map[string]float64{"x": 1, "y": 1, "z": 1},
		},
		"service_urls": serviceURLs,
		"metadata": map[string]any{
			"reader":           "ome-tiff+xml+go-image",
			"dims_order":       imageInfo.DimsOrder,
			"array_shape":      imageInfo.ArrayShape,
			"array_dtype":      imageInfo.ArrayDType,
			"physical_spacing": omePhysicalSpacing(meta),
			"scene":            meta.SceneName,
			"scene_count":      positiveIntOr(meta.SceneCount, 1),
			"header": map[string]string{
				"OME DimensionOrder": meta.DimensionOrder,
			},
			"microscopy": map[string]any{
				"channel_names":      meta.ChannelNames,
				"dimensions_present": imageInfo.DimsOrder,
				"current_scene":      meta.SceneName,
				"scene_names":        []string{meta.SceneName},
			},
			"warnings":     imageInfo.Warnings,
			"content_type": record.ContentType,
			"size_bytes":   record.SizeBytes,
			"sha256":       record.SHA256,
		},
		"viewer": map[string]any{
			"status":               "ready",
			"warmup_mode":          "lazy",
			"backend_mode":         "direct",
			"default_surface":      "2d",
			"available_surfaces":   availableSurfaces,
			"default_axis":         "z",
			"slice_axes":           []string{"z"},
			"channel_mode":         "single",
			"volume_mode":          "slice_stack",
			"render_policy":        "scalar",
			"delivery_mode":        "direct",
			"first_paint_mode":     "webgl",
			"measurement_policy":   measurementPolicy,
			"texture_policy":       "linear",
			"display_capabilities": displayCapabilities,
			"viewer_capabilities":  viewerCapabilities,
			"service_urls":         serviceURLs,
			"asset_preparation": map[string]any{
				"status":                "ready",
				"native_supported":      false,
				"tile_pyramid":          "deferred",
				"volume_representation": "slice_stack",
			},
		},
	})
}

func (deps ServerDeps) writeNiftiUploadViewer(w http.ResponseWriter, record resourceRecord, path string) {
	// CIFTI (.dtseries.nii et al.) is a NIfTI-2 container whose data is a
	// grayordinate/parcel matrix, not a spatial volume. Route it to the CIFTI
	// viewer (carpet + connectivity) instead of the slice loader. See cifti.go.
	if info, ok := niftiCiftiPeek(path, record.OriginalName); ok {
		deps.writeCiftiViewer(w, record, info, path)
		return
	}
	volume, err := loadNiftiScalarVolume(path)
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, err)
		return
	}
	// Warm the random-access sidecar for 4D series so the first time-scrub is
	// already fast; no-op for single-timepoint or uncompressed volumes.
	maybeDecompressNiftiSidecar(path, volume.TimeCount)
	dimsOrder := niftiScalarDimsOrder(volume)
	arrayShape := niftiScalarArrayShape(volume)
	channelColors := niftiDefaultChannelColors(volume.ChannelCount)
	displayCapabilities := []string{"slice_navigation", "histogram", "volume_context", "physical_scale", "window_level", "scalar_probe", "diagnostic_mpr"}
	viewerCapabilities := []string{"webgl_first_paint", "scalar_volume_delivery", "linear_sampling", "mpr_truth_surface", "slice_navigation", "volume_context", "physical_scale", "window_level"}
	if volume.ChannelCount > 1 {
		displayCapabilities = append(displayCapabilities, "channel_visibility")
		viewerCapabilities = append(viewerCapabilities, "channel_selection")
	}
	if volume.TimeCount > 1 {
		displayCapabilities = append(displayCapabilities, "time_navigation")
		viewerCapabilities = append(viewerCapabilities, "time_series_delivery")
	}
	fileIDSegment := url.PathEscape(record.FileID)
	serviceURLs := map[string]any{
		"preview":       "/v2/uploads/" + fileIDSegment + "/preview",
		"display":       "/v2/uploads/" + fileIDSegment + "/display",
		"slice":         "/v2/uploads/" + fileIDSegment + "/slice",
		"scalar_volume": "/v2/uploads/" + fileIDSegment + "/scalar-volume",
	}
	spacing := map[string]float64{
		"x": volume.SpacingX,
		"y": volume.SpacingY,
		"z": volume.SpacingZ,
	}
	orientationCode, axisEnds, planeAxis := niftiOrientation(volume.Affine)
	orientationKnown := volume.AffineCode > 0
	spaceUnit := volume.SpaceUnit
	if spaceUnit == "" {
		spaceUnit = "mm" // NIfTI's default spatial unit when xyzt_units is unset
	}
	effectiveSlope := volume.SclSlope
	if effectiveSlope == 0 {
		effectiveSlope = 1
	}
	orientation := map[string]any{
		"frame":         "patient",
		"convention":    "neurological", // on-screen left = patient left
		"code":          orientationCode,
		"known":         orientationKnown,
		"affine_method": niftiAffineMethodName(volume.AffineCode),
		"axis_labels": map[string]any{
			"x": map[string]string{"negative": axisEnds[0][0], "positive": axisEnds[0][1]},
			"y": map[string]string{"negative": axisEnds[1][0], "positive": axisEnds[1][1]},
			"z": map[string]string{"negative": axisEnds[2][0], "positive": axisEnds[2][1]},
		},
		"plane_axis": planeAxis,
	}
	affineRows := [][]float64{
		{volume.Affine[0], volume.Affine[1], volume.Affine[2], volume.Affine[3]},
		{volume.Affine[4], volume.Affine[5], volume.Affine[6], volume.Affine[7]},
		{volume.Affine[8], volume.Affine[9], volume.Affine[10], volume.Affine[11]},
		{0, 0, 0, 1},
	}
	if orientationKnown {
		displayCapabilities = append(displayCapabilities, "orientation_markers")
		viewerCapabilities = append(viewerCapabilities, "anatomical_orientation")
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"kind":          "image",
		"file_id":       record.FileID,
		"original_name": record.OriginalName,
		"modality":      "medical",
		"backend_mode":  "scalar",
		"dims_order":    dimsOrder,
		"axis_sizes": map[string]int{
			"T": volume.TimeCount,
			"C": volume.ChannelCount,
			"Z": volume.Depth,
			"Y": volume.Height,
			"X": volume.Width,
		},
		"selected_indices": map[string]int{"T": volume.TimeIndex, "C": volume.ChannelIndex, "Z": volume.Depth / 2},
		"is_volume":        volume.Depth > 1,
		"is_timeseries":    volume.TimeCount > 1,
		"is_multichannel":  volume.ChannelCount > 1,
		"service_urls":     serviceURLs,
		"display_defaults": niftiScalarDisplayDefaults(volume, channelColors),
		"metadata": map[string]any{
			"reader":                   "nifti-1",
			"dims_order":               dimsOrder,
			"array_shape":              arrayShape,
			"array_dtype":              volume.DType,
			"array_min":                volume.RawMin,
			"array_max":                volume.RawMax,
			"intensity_stats":          map[string]float64{"min": volume.RawMin, "max": volume.RawMax},
			"intensity_stats_physical": map[string]float64{"min": volume.physical(volume.RawMin), "max": volume.physical(volume.RawMax)},
			"rescale_slope":            effectiveSlope,
			"rescale_intercept":        volume.SclInter,
			"physical_spacing":         spacing,
			"physical_spacing_unit":    spaceUnit,
			"affine":                   affineRows,
			"orientation_code":         orientationCode,
			"scene_count":              1,
			"warnings":                 volume.Warnings,
			"content_type":             record.ContentType,
			"size_bytes":               record.SizeBytes,
			"sha256":                   record.SHA256,
		},
		"viewer": map[string]any{
			"status":                "ready",
			"warmup_mode":           "lazy",
			"backend_mode":          "scalar",
			"default_surface":       "volume",
			"available_surfaces":    []string{"2d", "mpr", "volume", "metadata"},
			"default_axis":          "z",
			"slice_axes":            []string{"z", "y", "x"},
			"orientation":           orientation,
			"physical_spacing_unit": spaceUnit,
			"channel_mode":          "single",
			"volume_mode":           "scalar",
			"render_policy":         "scalar",
			"delivery_mode":         "scalar",
			"diagnostic_surface":    "mpr",
			"first_paint_mode":      "webgl",
			"measurement_policy":    "spacing-aware",
			"texture_policy":        "linear",
			"display_capabilities":  displayCapabilities,
			"viewer_capabilities":   viewerCapabilities,
			"service_urls":          serviceURLs,
			"asset_preparation": map[string]any{
				"status":                "ready",
				"native_supported":      true,
				"tile_pyramid":          "deferred",
				"volume_representation": "scalar",
			},
		},
	})
}

func (deps ServerDeps) handleListRuns(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	principal := deps.principalFromRequest(r, "")
	threadID := strings.TrimSpace(r.URL.Query().Get("thread_id"))
	status := strings.TrimSpace(r.URL.Query().Get("status"))
	runs, err := deps.Store.ListRunsForUser(r.Context(), principal.UserID, threadID, status, parseLimit(r, 100), parseOffset(r))
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
	run, ok := deps.runForWorkerOrUser(w, r, chi.URLParam(r, "run_id"))
	if !ok {
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
		GeneratedAt:      data.GeneratedAt,
		Runtime:          deps.adminRuntimeSummary(),
		Queue:            deps.adminQueueDiagnostics(r.Context()),
		Database:         deps.adminDatabaseDiagnostics(r.Context()),
		ImageCache:       deps.adminImageCacheStats(),
		RetentionBacklog: deps.adminRetentionBacklog(r.Context()),
		KPIs:             data.KPIs,
		UploadSessions:   data.UploadSessions,
		Activity:         data.Activity,
		UsageLast24h:     data.UsageLast24h,
		ToolUsage7d:      data.ToolUsage7d,
		Workers:          data.Workers,
		TopUsers:         topUsers,
		ResourceProjects: take(data.ResourceProjects, parseLimitParam(r, "resource_project_limit", 12)),
		RecentIssues:     recentIssues,
	})
}

func (deps ServerDeps) handleAdminReconcileResources(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	catalog, ok := deps.Store.(resourceCatalogAdminStore)
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource catalog snapshot is not configured"})
		return
	}
	resources, err := catalog.ListResources(r.Context(), 10000, 0)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	uploads, err := listUploadResourceEntries(root)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	catalogByID := make(map[string]domain.ResourceRecord, len(resources))
	for _, resource := range resources {
		catalogByID[resource.ResourceID] = resource
	}
	uploadByID := make(map[string]uploadResourceEntry, len(uploads))
	for _, upload := range uploads {
		uploadByID[upload.FileID] = upload
	}
	issues := []resourceReconcileIssue{}
	addIssue := func(issue resourceReconcileIssue) {
		if issue.Severity == "" {
			issue.Severity = "warning"
		}
		issues = append(issues, issue)
	}
	for _, upload := range uploads {
		if _, ok := catalogByID[upload.FileID]; !ok {
			addIssue(resourceReconcileIssue{
				IssueType:  "missing_catalog_row",
				Severity:   "warning",
				ResourceID: upload.FileID,
				Path:       filepath.Base(upload.Path),
				Message:    "upload blob exists on NFS without a catalog row",
			})
		}
		if _, err := os.Stat(uploadMetadataPath(root, upload.FileID)); err != nil && errors.Is(err, os.ErrNotExist) {
			addIssue(resourceReconcileIssue{
				IssueType:  "missing_sidecar",
				Severity:   "warning",
				ResourceID: upload.FileID,
				Path:       filepath.Base(upload.Path),
				Message:    "upload blob is missing its legacy sidecar metadata",
			})
		}
	}
	for _, resource := range resources {
		if strings.TrimSpace(resource.Status) != "active" {
			continue
		}
		path, err := resolveCatalogResourcePath(root, resource)
		if err != nil {
			addIssue(resourceReconcileIssue{
				IssueType:  "missing_blob",
				Severity:   "error",
				ResourceID: resource.ResourceID,
				Path:       resource.StoragePath,
				Message:    "catalog row does not resolve to a blob under the upload root",
			})
			continue
		}
		info, err := os.Stat(path)
		if err != nil || info.IsDir() {
			addIssue(resourceReconcileIssue{
				IssueType:  "missing_blob",
				Severity:   "error",
				ResourceID: resource.ResourceID,
				Path:       filepath.Base(path),
				Message:    "catalog row points to a missing blob",
			})
			continue
		}
		if strings.TrimSpace(resource.SHA256) != "" {
			actualSHA := ""
			if upload, ok := uploadByID[resource.ResourceID]; ok && filepath.Clean(upload.Path) == filepath.Clean(path) {
				actualSHA = upload.SHA256
			}
			if actualSHA == "" {
				actualSHA, err = sha256File(path)
			}
			if err != nil {
				addIssue(resourceReconcileIssue{
					IssueType:  "checksum_unreadable",
					Severity:   "error",
					ResourceID: resource.ResourceID,
					Path:       filepath.Base(path),
					Message:    "blob could not be hashed",
				})
			} else if !strings.EqualFold(strings.TrimSpace(resource.SHA256), actualSHA) {
				addIssue(resourceReconcileIssue{
					IssueType:   "checksum_drift",
					Severity:    "error",
					ResourceID:  resource.ResourceID,
					Path:        filepath.Base(path),
					Message:     "blob checksum differs from catalog checksum",
					ExpectedSHA: resource.SHA256,
					ActualSHA:   actualSHA,
				})
			}
		}
		if resourceNeedsPreview(resource) {
			if err := validatePreviewSource(path); err != nil {
				addIssue(resourceReconcileIssue{
					IssueType:  "failed_preview",
					Severity:   "warning",
					ResourceID: resource.ResourceID,
					Path:       filepath.Base(path),
					Message:    err.Error(),
				})
			}
		}
	}
	summary := make(map[string]int)
	for _, issue := range issues {
		summary[issue.IssueType]++
	}
	writeJSON(w, http.StatusOK, resourceReconcileResponse{
		CheckedAt:  domain.Now().Format(time.RFC3339Nano),
		IssueCount: len(issues),
		Summary:    summary,
		Issues:     issues,
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

// adminImageCacheStats surfaces the image response cache's hit rate and saturation
// so an operator can see whether the viewer's hot path is being served from memory
// (hits) or hammering the decode engine (misses), and how full the cache is.
func (deps ServerDeps) adminImageCacheStats() adminImageCacheStats {
	// Combine the main (tile/atlas/thumbnail) cache and the dedicated /slice cache so
	// the operator sees total image-cache effectiveness across both partitions.
	caches := []*imageResponseCache{deps.imageCache, deps.sliceCache}
	var hits, misses uint64
	var entries int
	var bytes, maxBytes int64
	enabled := false
	for _, c := range caches {
		if c == nil {
			continue
		}
		enabled = true
		h, m, e, b := c.stats()
		hits += h
		misses += m
		entries += e
		bytes += b
		maxBytes += c.maxBytes
	}
	if !enabled {
		return adminImageCacheStats{Enabled: false}
	}
	rate := 0.0
	if total := hits + misses; total > 0 {
		rate = float64(hits) / float64(total)
	}
	return adminImageCacheStats{
		Enabled:  true,
		Hits:     hits,
		Misses:   misses,
		HitRate:  rate,
		Entries:  entries,
		Bytes:    bytes,
		MaxBytes: maxBytes,
	}
}

// adminRetentionBacklog reports the storage held by soft-deleted resources past their
// undelete window — the reclaimable backlog an operator watches to decide whether to
// enable the retention GC. Read-only; never deletes.
func (deps ServerDeps) adminRetentionBacklog(ctx context.Context) adminRetentionBacklog {
	store, ok := deps.Store.(retentionBacklogStore)
	if !ok {
		return adminRetentionBacklog{}
	}
	backlog, err := store.RetentionBacklog(ctx, time.Now())
	if err != nil {
		return adminRetentionBacklog{}
	}
	return adminRetentionBacklog{ExpiredResources: backlog.Count, ReclaimableBytes: backlog.Bytes}
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
	accounting := deps.resourceAccounting(r.Context())
	summaries := make([]adminOrganizationSummary, 0, len(records))
	for _, record := range records {
		usage := accounting.Orgs[record.OrgID]
		summaries = append(summaries, adminOrganizationSummary{
			OrgID:        record.OrgID,
			Name:         record.Name,
			Status:       record.Status,
			CreatedAt:    record.CreatedAt.UTC().Format(time.RFC3339Nano),
			UpdatedAt:    record.UpdatedAt.UTC().Format(time.RFC3339Nano),
			Metadata:     record.Metadata,
			Uploads:      usage.Uploads,
			StorageBytes: usage.StorageBytes,
		})
	}
	writeJSON(w, http.StatusOK, adminOrganizationListResponse{Count: len(summaries), Organizations: summaries})
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

func (deps ServerDeps) handleAdminUpdateUserStatus(w http.ResponseWriter, r *http.Request) {
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
	var req adminUpdateUserStatusRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}
	status := normalizeAccountStatus(req.Status)
	switch status {
	case "active", "pending", "disabled", "rejected":
	default:
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "status must be active, pending, disabled, or rejected"})
		return
	}
	user, err := accounts.UpdateUserStatus(r.Context(), userID, status)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, user)
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
	GeneratedAt      string
	KPIs             adminPlatformKPIs
	UploadSessions   domain.UploadSessionOperationalMetrics
	Activity         []adminActivityPeriod
	UsageLast24h     []adminUsageBucket
	ToolUsage7d      []adminToolUsageRecord
	Workers          []adminWorkerRecord
	Users            []adminUserSummary
	ResourceProjects []adminResourceOwnerSummary
	Issues           []adminIssueRecord
}

type adminActivityWindow struct {
	label  string
	window string
	since  *time.Time
}

type adminActivityCounter struct {
	label             string
	window            string
	messages          int
	userMessages      int
	assistantMessages int
	toolCalls         int
	runs              int
	failedRuns        int
	artifacts         int
	activeUsers       map[string]bool
}

type adminActivityAccumulator struct {
	windows  []adminActivityWindow
	counters []*adminActivityCounter
}

func newAdminActivityAccumulator(now time.Time) *adminActivityAccumulator {
	daily := now.Add(-24 * time.Hour)
	weekly := now.Add(-7 * 24 * time.Hour)
	monthly := now.Add(-30 * 24 * time.Hour)
	windows := []adminActivityWindow{
		{label: "Daily", window: "24h", since: &daily},
		{label: "Weekly", window: "7d", since: &weekly},
		{label: "Monthly", window: "30d", since: &monthly},
		{label: "Total", window: "all"},
	}
	counters := make([]*adminActivityCounter, 0, len(windows))
	for _, window := range windows {
		counters = append(counters, &adminActivityCounter{
			label:       window.label,
			window:      window.window,
			activeUsers: map[string]bool{},
		})
	}
	return &adminActivityAccumulator{windows: windows, counters: counters}
}

func (a *adminActivityAccumulator) addMessage(ts time.Time, userID string, role string) {
	for _, counter := range a.matchingCounters(ts) {
		counter.messages++
		switch strings.ToLower(strings.TrimSpace(role)) {
		case "user":
			counter.userMessages++
		case "assistant":
			counter.assistantMessages++
		}
		counter.addActiveUser(userID)
	}
}

func (a *adminActivityAccumulator) addRun(ts time.Time, userID string, status domain.RunStatus) {
	for _, counter := range a.matchingCounters(ts) {
		counter.runs++
		if status == domain.RunStatusFailed {
			counter.failedRuns++
		}
		counter.addActiveUser(userID)
	}
}

func (a *adminActivityAccumulator) addToolCall(ts time.Time, userID string) {
	for _, counter := range a.matchingCounters(ts) {
		counter.toolCalls++
		counter.addActiveUser(userID)
	}
}

func (a *adminActivityAccumulator) addArtifact(ts time.Time, userID string) {
	for _, counter := range a.matchingCounters(ts) {
		counter.artifacts++
		counter.addActiveUser(userID)
	}
}

// addUserMessageStats and addUserEventStats fold pre-aggregated per-user
// window counts into the accumulator. They depend on the counter order
// established in newAdminActivityAccumulator: Daily, Weekly, Monthly, Total.
func (a *adminActivityAccumulator) addUserMessageStats(stat domain.AdminUserMessageStats) {
	buckets := []struct {
		index             int
		messages          int
		userMessages      int
		assistantMessages int
	}{
		{0, stat.Messages.Last24h, stat.UserMessages.Last24h, stat.AssistantMessages.Last24h},
		{1, stat.Messages.Last7d, stat.UserMessages.Last7d, stat.AssistantMessages.Last7d},
		{2, stat.Messages.Last30d, stat.UserMessages.Last30d, stat.AssistantMessages.Last30d},
		{3, stat.Messages.Total, stat.UserMessages.Total, stat.AssistantMessages.Total},
	}
	for _, bucket := range buckets {
		counter := a.counters[bucket.index]
		counter.messages += bucket.messages
		counter.userMessages += bucket.userMessages
		counter.assistantMessages += bucket.assistantMessages
		if bucket.messages > 0 {
			counter.addActiveUser(stat.UserID)
		}
	}
}

func (a *adminActivityAccumulator) addUserEventStats(stat domain.AdminUserEventStats) {
	buckets := []struct {
		index     int
		toolCalls int
		artifacts int
	}{
		{0, stat.ToolCalls.Last24h, stat.Artifacts.Last24h},
		{1, stat.ToolCalls.Last7d, stat.Artifacts.Last7d},
		{2, stat.ToolCalls.Last30d, stat.Artifacts.Last30d},
		{3, stat.ToolCalls.Total, stat.Artifacts.Total},
	}
	for _, bucket := range buckets {
		counter := a.counters[bucket.index]
		counter.toolCalls += bucket.toolCalls
		counter.artifacts += bucket.artifacts
		if bucket.toolCalls > 0 || bucket.artifacts > 0 {
			counter.addActiveUser(stat.UserID)
		}
	}
}

func (a *adminActivityAccumulator) matchingCounters(ts time.Time) []*adminActivityCounter {
	if ts.IsZero() {
		return a.counters
	}
	matches := make([]*adminActivityCounter, 0, len(a.counters))
	for index, window := range a.windows {
		if window.since == nil || !ts.Before(*window.since) {
			matches = append(matches, a.counters[index])
		}
	}
	return matches
}

func (counter *adminActivityCounter) addActiveUser(userID string) {
	userID = strings.TrimSpace(userID)
	if userID != "" {
		counter.activeUsers[userID] = true
	}
}

func (a *adminActivityAccumulator) periods() []adminActivityPeriod {
	periods := make([]adminActivityPeriod, 0, len(a.counters))
	for _, counter := range a.counters {
		periods = append(periods, adminActivityPeriod{
			Label:             counter.label,
			Window:            counter.window,
			Messages:          counter.messages,
			UserMessages:      counter.userMessages,
			AssistantMessages: counter.assistantMessages,
			ToolCalls:         counter.toolCalls,
			ActiveUsers:       len(counter.activeUsers),
			Runs:              counter.runs,
			FailedRuns:        counter.failedRuns,
			Artifacts:         counter.artifacts,
		})
	}
	return periods
}

// adminAggregateStore is the store capability that lets the admin snapshot
// be computed from grouped queries instead of per-thread and per-run fetch
// loops. Stores without it (the in-memory dev store, test fakes) fall back
// to the legacy loop.
type adminAggregateStore interface {
	AdminUserMessageStats(ctx context.Context, since24h, since7d, since30d time.Time) ([]domain.AdminUserMessageStats, error)
	AdminUserEventStats(ctx context.Context, since24h, since7d, since30d time.Time) ([]domain.AdminUserEventStats, error)
	AdminResourceStats(ctx context.Context) (domain.AdminResourceStats, error)
}

func (deps ServerDeps) loadAdminSnapshot(ctx context.Context) (adminSnapshot, error) {
	if deps.adminSnapshots == nil {
		return deps.computeAdminSnapshot(ctx)
	}
	// Concurrent admin requests share one computation; a stampede of
	// dashboard tabs costs one snapshot. The shared call is detached from
	// the first caller's context so its cancellation does not fail the rest.
	result, err, _ := deps.adminSnapshots.Do("admin_snapshot", func() (any, error) {
		computeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
		defer cancel()
		return deps.computeAdminSnapshot(computeCtx)
	})
	if err != nil {
		return adminSnapshot{}, err
	}
	return result.(adminSnapshot), nil
}

func (deps ServerDeps) computeAdminSnapshot(ctx context.Context) (adminSnapshot, error) {
	if aggregates, ok := deps.Store.(adminAggregateStore); ok {
		return deps.loadAdminSnapshotAggregate(ctx, aggregates, domain.Now())
	}
	return deps.loadAdminSnapshotLegacy(ctx, domain.Now())
}

// loadAdminSnapshotAggregate mirrors loadAdminSnapshotLegacy exactly (the
// equivalence is covered by TestAdminSnapshotAggregateMatchesLegacy) but
// sources message totals, tool/artifact activity, and resource accounting
// from grouped store queries, and only checks staleness on watchable runs
// from their latest event, instead of fetching every thread's messages and
// every run's events.
func (deps ServerDeps) loadAdminSnapshotAggregate(ctx context.Context, aggregates adminAggregateStore, now time.Time) (adminSnapshot, error) {
	since := now.Add(-24 * time.Hour)
	since7d := now.Add(-7 * 24 * time.Hour)
	since30d := now.Add(-30 * 24 * time.Hour)
	activity := newAdminActivityAccumulator(now)
	threadPage, err := deps.Store.ListThreads(ctx, 10000, 0, "")
	if err != nil {
		return adminSnapshot{}, err
	}
	threads := threadPage.Threads
	runs, err := deps.Store.ListRuns(ctx, "", "", 10000, 0)
	if err != nil {
		return adminSnapshot{}, err
	}
	resourceStats, err := aggregates.AdminResourceStats(ctx)
	if err != nil {
		return adminSnapshot{}, err
	}
	messageStats, err := aggregates.AdminUserMessageStats(ctx, since, since7d, since30d)
	if err != nil {
		return adminSnapshot{}, err
	}
	eventStats, err := aggregates.AdminUserEventStats(ctx, since, since7d, since30d)
	if err != nil {
		return adminSnapshot{}, err
	}
	uploadSessionMetrics := domain.UploadSessionOperationalMetrics{}
	if metricsStore, ok := deps.Store.(uploadSessionOperationalMetricsStore); ok {
		uploadSessionMetrics, err = metricsStore.UploadSessionOperationalMetrics(ctx)
		if err != nil {
			return adminSnapshot{}, err
		}
	}
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
	}
	for _, stat := range messageStats {
		user := adminUser(users, stat.UserID)
		user.Messages += stat.Messages.Total
		totalMessages += stat.Messages.Total
		messages24h += stat.Messages.Last24h
		userMessages24h += stat.UserMessages.Last24h
		assistantMessages24h += stat.AssistantMessages.Last24h
		if stat.Messages.Last24h > 0 {
			userSeen24h[user.UserID] = true
		}
		activity.addUserMessageStats(stat)
	}
	for _, stat := range eventStats {
		adminUser(users, stat.UserID)
		activity.addUserEventStats(stat)
	}

	runs24h := 0
	runsSucceeded24h := 0
	runsFailed24h := 0
	runningRuns := 0
	staleRunningRuns := 0
	toolUsage := map[string]*adminToolUsageRecord{}
	issues := []adminIssueRecord{}
	for _, run := range runs {
		user := adminUser(users, run.UserID)
		user.RunsTotal++
		updateLastActivity(user, run.UpdatedAt)
		activity.addRun(run.CreatedAt, user.UserID, run.Status)
		if run.UpdatedAt.After(since) {
			userSeen24h[user.UserID] = true
		}
		switch run.Status {
		case domain.RunStatusRunning, domain.RunStatusQueued, domain.RunStatusWaitingForInput, domain.RunStatusWaitingForTask:
			diagnostic, err := deps.adminRunStaleCheck(ctx, run, now)
			if err != nil {
				return adminSnapshot{}, err
			}
			user.RunsRunning++
			runningRuns++
			if diagnostic.Stale {
				staleRunningRuns++
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

	resourceUsers := map[string]resourceOwnerAccounting{}
	for _, owner := range resourceStats.Users {
		resourceUsers[owner.Owner] = resourceOwnerAccounting{Uploads: owner.Uploads, StorageBytes: owner.StorageBytes}
	}
	resourceProjectUsage := map[string]resourceOwnerAccounting{}
	for _, owner := range resourceStats.Projects {
		resourceProjectUsage[owner.Owner] = resourceOwnerAccounting{Uploads: owner.Uploads, StorageBytes: owner.StorageBytes}
	}
	for userID, usage := range resourceUsers {
		user := adminUser(users, userID)
		user.Uploads += usage.Uploads
		user.StorageBytes += usage.StorageBytes
	}
	resourceProjects := resourceOwnerSummaries(resourceProjectUsage)
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
			TotalUploads:               resourceStats.ActiveResources,
			SoftDeletedUploads:         resourceStats.SoftDeletedResources,
			TotalStorageBytes:          resourceStats.ActiveBytes,
			AvgMessagesPerConversation: avgMessages,
		},
		UploadSessions: uploadSessionMetrics,
		Activity:       activity.periods(),
		UsageLast24h: []adminUsageBucket{{
			BucketStart:   since.UTC().Format(time.RFC3339Nano),
			RunsTotal:     runs24h,
			RunsSucceeded: runsSucceeded24h,
			RunsFailed:    runsFailed24h,
			Uploads:       resourceStats.ActiveResources,
			NewUsers:      len(userSeen24h),
		}},
		ToolUsage7d:      toolList,
		Workers:          workers,
		Users:            userList,
		ResourceProjects: resourceProjects,
		Issues:           issueList,
	}, nil
}

func (deps ServerDeps) loadAdminSnapshotLegacy(ctx context.Context, now time.Time) (adminSnapshot, error) {
	since := now.Add(-24 * time.Hour)
	activity := newAdminActivityAccumulator(now)
	threadPage, err := deps.Store.ListThreads(ctx, 10000, 0, "")
	if err != nil {
		return adminSnapshot{}, err
	}
	threads := threadPage.Threads
	runs, err := deps.Store.ListRuns(ctx, "", "", 10000, 0)
	if err != nil {
		return adminSnapshot{}, err
	}
	resourceAccounting := deps.resourceAccounting(ctx)
	uploadSessionMetrics := domain.UploadSessionOperationalMetrics{}
	if metricsStore, ok := deps.Store.(uploadSessionOperationalMetricsStore); ok {
		uploadSessionMetrics, err = metricsStore.UploadSessionOperationalMetrics(ctx)
		if err != nil {
			return adminSnapshot{}, err
		}
	}
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
			activity.addMessage(message.CreatedAt, user.UserID, message.Role)
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
		activity.addRun(run.CreatedAt, user.UserID, run.Status)
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
		for _, event := range diagnostic.Events {
			eventTS := event.TS
			if eventTS.IsZero() {
				eventTS = run.CreatedAt
			}
			switch event.EventKind {
			case "tool_call.started":
				activity.addToolCall(eventTS, user.UserID)
			case "artifact.created":
				activity.addArtifact(eventTS, user.UserID)
			}
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

	for userID, usage := range resourceAccounting.Users {
		user := adminUser(users, userID)
		user.Uploads += usage.Uploads
		user.StorageBytes += usage.StorageBytes
	}
	resourceProjects := resourceOwnerSummaries(resourceAccounting.Projects)
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
			TotalUploads:               resourceAccounting.ActiveResources,
			SoftDeletedUploads:         resourceAccounting.SoftDeletedResources,
			TotalStorageBytes:          resourceAccounting.ActiveBytes,
			AvgMessagesPerConversation: avgMessages,
		},
		UploadSessions: uploadSessionMetrics,
		Activity:       activity.periods(),
		UsageLast24h: []adminUsageBucket{{
			BucketStart:   since.UTC().Format(time.RFC3339Nano),
			RunsTotal:     runs24h,
			RunsSucceeded: runsSucceeded24h,
			RunsFailed:    runsFailed24h,
			Uploads:       resourceAccounting.ActiveResources,
			NewUsers:      len(userSeen24h),
		}},
		ToolUsage7d:      toolList,
		Workers:          workers,
		Users:            userList,
		ResourceProjects: resourceProjects,
		Issues:           issueList,
	}, nil
}

func resourceOwnerSummaries(values map[string]resourceOwnerAccounting) []adminResourceOwnerSummary {
	summaries := make([]adminResourceOwnerSummary, 0, len(values))
	for id, usage := range values {
		if strings.TrimSpace(id) == "" {
			continue
		}
		summaries = append(summaries, adminResourceOwnerSummary{
			ID:           id,
			Uploads:      usage.Uploads,
			StorageBytes: usage.StorageBytes,
		})
	}
	sort.Slice(summaries, func(i, j int) bool {
		if summaries[i].StorageBytes == summaries[j].StorageBytes {
			if summaries[i].Uploads == summaries[j].Uploads {
				return summaries[i].ID < summaries[j].ID
			}
			return summaries[i].Uploads > summaries[j].Uploads
		}
		return summaries[i].StorageBytes > summaries[j].StorageBytes
	})
	return summaries
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
	Events                     []domain.RunEventRecord
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
		Events:         events,
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
	if err := deps.applyAdminRunLeaseAndStaleness(ctx, run, now, &diagnostic); err != nil {
		return adminRunDiagnostic{}, err
	}
	return diagnostic, nil
}

// adminRunStaleCheck computes the lease and staleness portion of the run
// diagnostic from the run's latest event only. The admin snapshot uses it
// for watchable (non-terminal) runs, where staleness matters, instead of
// fetching every event of every run.
func (deps ServerDeps) adminRunStaleCheck(ctx context.Context, run domain.RunRecord, now time.Time) (adminRunDiagnostic, error) {
	events, err := deps.Store.ListRunEvents(ctx, run.RunID, 1)
	if err != nil {
		return adminRunDiagnostic{}, err
	}
	diagnostic := adminRunDiagnostic{LastActivityAt: run.UpdatedAt}
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
	if diagnostic.LastActivityAt.IsZero() {
		diagnostic.LastActivityAt = now
	}
	if err := deps.applyAdminRunLeaseAndStaleness(ctx, run, now, &diagnostic); err != nil {
		return adminRunDiagnostic{}, err
	}
	return diagnostic, nil
}

func (deps ServerDeps) applyAdminRunLeaseAndStaleness(ctx context.Context, run domain.RunRecord, now time.Time, diagnostic *adminRunDiagnostic) error {
	if leases, ok := deps.Store.(runLeaseReader); ok {
		lease, found, err := leases.GetRunLease(ctx, run.RunID)
		if err != nil {
			return err
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
	return nil
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
	if catalog, ok := deps.resourceCatalogStore(); ok {
		stats, err := catalog.ResourceStorageStats(context.Background())
		if err == nil {
			return stats.TotalResources, stats.TotalBytes
		}
	}
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

type resourceQuotaError struct {
	ScopeType        string `json:"scope_type"`
	ScopeID          string `json:"scope_id"`
	ProjectID        string `json:"project_id,omitempty"`
	LimitResources   int    `json:"limit_resources,omitempty"`
	CurrentResources int    `json:"current_resources"`
	LimitBytes       int64  `json:"limit_bytes,omitempty"`
	CurrentBytes     int64  `json:"current_bytes"`
	RequestedBytes   int64  `json:"requested_bytes"`
}

func (err *resourceQuotaError) Error() string {
	if err == nil {
		return "resource quota exceeded"
	}
	scope := strings.TrimSpace(err.ScopeType)
	if scope == "" {
		scope = "resource"
	}
	if err.ScopeID != "" {
		return fmt.Sprintf("%s resource quota exceeded for %s", scope, err.ScopeID)
	}
	return fmt.Sprintf("%s resource quota exceeded", scope)
}

func writeResourceQuotaError(w http.ResponseWriter, err error) {
	var quotaErr *resourceQuotaError
	if errors.As(err, &quotaErr) {
		writeJSON(w, http.StatusRequestEntityTooLarge, map[string]any{
			"error": "resource_quota_exceeded",
			"quota": quotaErr,
		})
		return
	}
	writeError(w, http.StatusInternalServerError, err)
}

func (deps ServerDeps) enforceResourceQuota(ctx context.Context, principal requestPrincipal, projectID string, requestedBytes int64) error {
	userID := strings.TrimSpace(principal.UserID)
	if userID == "" {
		userID = "local-user"
	}
	projectID = strings.TrimSpace(projectID)
	// Lazy full-catalog accounting is the fallback for stores without scoped aggregates;
	// the real stores implement resourceQuotaUsageStore so a quota check stays O(indexed
	// aggregate) instead of loading (and capping at 100k) the whole catalog per upload.
	var accounting *resourceAccountingSummary
	loadAccounting := func() resourceAccountingSummary {
		if accounting == nil {
			summary := deps.resourceAccounting(ctx)
			accounting = &summary
		}
		return *accounting
	}
	usageFor := func(scope, id string) (resourceOwnerAccounting, error) {
		usageStore, ok := deps.Store.(resourceQuotaUsageStore)
		if !ok {
			switch scope {
			case "user":
				return loadAccounting().Users[id], nil
			case "org":
				return loadAccounting().Orgs[id], nil
			default:
				return loadAccounting().Projects[id], nil
			}
		}
		var count int
		var bytes int64
		var err error
		switch scope {
		case "user":
			count, bytes, err = usageStore.ResourceUsageForOwner(ctx, id)
		case "org":
			count, bytes, err = usageStore.ResourceUsageForOrg(ctx, id)
		default:
			count, bytes, err = usageStore.ResourceUsageForProject(ctx, id)
		}
		if err != nil {
			return resourceOwnerAccounting{}, err
		}
		return resourceOwnerAccounting{Uploads: count, StorageBytes: bytes}, nil
	}
	checkScope := func(scope, scopeID string, quota resourceQuotaLimits) error {
		usage, err := usageFor(scope, scopeID)
		if err != nil {
			return err
		}
		return checkResourceQuota(scope, scopeID, projectID, quota, usage, requestedBytes)
	}
	if accounts, ok := deps.Store.(accountStore); ok {
		if account, found, err := accounts.GetUserByID(ctx, userID); err != nil {
			return err
		} else if found {
			if quota := resourceQuotaFromMetadata(account.Metadata); resourceQuotaConfigured(quota) {
				if err := checkScope("user", userID, quota); err != nil {
					return err
				}
			}
			if quota, ok := projectResourceQuotaFromMetadata(account.Metadata, projectID); ok && resourceQuotaConfigured(quota) {
				if err := checkScope("project", projectID, quota); err != nil {
					return err
				}
			}
		}
	}
	orgID := strings.TrimSpace(principal.OrgID)
	if orgID != "" {
		if org, found, err := deps.organizationByID(ctx, orgID); err != nil {
			return err
		} else if found {
			if quota := resourceQuotaFromMetadata(org.Metadata); resourceQuotaConfigured(quota) {
				if err := checkScope("org", orgID, quota); err != nil {
					return err
				}
			}
			if quota, ok := projectResourceQuotaFromMetadata(org.Metadata, projectID); ok && resourceQuotaConfigured(quota) {
				if err := checkScope("project", projectID, quota); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (deps ServerDeps) organizationByID(ctx context.Context, orgID string) (domain.Organization, bool, error) {
	orgID = strings.TrimSpace(orgID)
	if orgID == "" {
		return domain.Organization{}, false, nil
	}
	orgs, ok := deps.Store.(organizationStore)
	if !ok {
		return domain.Organization{}, false, nil
	}
	if lookup, ok := deps.Store.(organizationLookupStore); ok {
		return lookup.GetOrganization(ctx, orgID)
	}
	records, err := orgs.ListOrganizations(ctx, 10000, "")
	if err != nil {
		return domain.Organization{}, false, err
	}
	for _, record := range records {
		if strings.TrimSpace(record.OrgID) == orgID {
			return record, true, nil
		}
	}
	return domain.Organization{}, false, nil
}

func checkResourceQuotaMetadata(scopeType string, scopeID string, projectID string, metadata domain.JSONMap, usage resourceOwnerAccounting, requestedBytes int64) error {
	quota := resourceQuotaFromMetadata(metadata)
	if quota.limitResources <= 0 && quota.limitBytes <= 0 {
		return nil
	}
	return checkResourceQuota(scopeType, scopeID, projectID, quota, usage, requestedBytes)
}

func checkProjectResourceQuotaMetadata(scopeType string, projectID string, metadata domain.JSONMap, usage resourceOwnerAccounting, requestedBytes int64) error {
	quota, ok := projectResourceQuotaFromMetadata(metadata, projectID)
	if !ok || (quota.limitResources <= 0 && quota.limitBytes <= 0) {
		return nil
	}
	return checkResourceQuota(scopeType, projectID, projectID, quota, usage, requestedBytes)
}

type resourceQuotaLimits struct {
	limitResources int
	limitBytes     int64
}

func resourceQuotaConfigured(quota resourceQuotaLimits) bool {
	return quota.limitResources > 0 || quota.limitBytes > 0
}

func checkResourceQuota(scopeType string, scopeID string, projectID string, quota resourceQuotaLimits, usage resourceOwnerAccounting, requestedBytes int64) error {
	if quota.limitResources > 0 && usage.Uploads+1 > quota.limitResources {
		return &resourceQuotaError{
			ScopeType:        scopeType,
			ScopeID:          scopeID,
			ProjectID:        projectID,
			LimitResources:   quota.limitResources,
			CurrentResources: usage.Uploads,
			CurrentBytes:     usage.StorageBytes,
			RequestedBytes:   requestedBytes,
		}
	}
	if quota.limitBytes > 0 && usage.StorageBytes+requestedBytes > quota.limitBytes {
		return &resourceQuotaError{
			ScopeType:        scopeType,
			ScopeID:          scopeID,
			ProjectID:        projectID,
			LimitBytes:       quota.limitBytes,
			CurrentResources: usage.Uploads,
			CurrentBytes:     usage.StorageBytes,
			RequestedBytes:   requestedBytes,
		}
	}
	return nil
}

func resourceQuotaFromMetadata(metadata domain.JSONMap) resourceQuotaLimits {
	return resourceQuotaLimits{
		limitResources: metadataInt(metadata, "resource_quota_count", "resource_max_count", "max_resources"),
		limitBytes:     metadataInt64(metadata, "resource_quota_bytes", "resource_max_bytes", "max_resource_bytes"),
	}
}

func projectResourceQuotaFromMetadata(metadata domain.JSONMap, projectID string) (resourceQuotaLimits, bool) {
	projectID = strings.TrimSpace(projectID)
	if projectID == "" {
		return resourceQuotaLimits{}, false
	}
	for _, key := range []string{"resource_project_quotas", "project_resource_quotas"} {
		raw, ok := metadata[key]
		if !ok {
			continue
		}
		quotas, ok := raw.(map[string]any)
		if !ok {
			if typed, ok := raw.(domain.JSONMap); ok {
				quotas = map[string]any(typed)
			}
		}
		if quotas == nil {
			continue
		}
		payload, ok := quotas[projectID]
		if !ok {
			continue
		}
		switch typed := payload.(type) {
		case map[string]any:
			return resourceQuotaFromMetadata(domain.JSONMap(typed)), true
		case domain.JSONMap:
			return resourceQuotaFromMetadata(typed), true
		}
	}
	return resourceQuotaLimits{}, false
}

func metadataInt(metadata domain.JSONMap, keys ...string) int {
	value := metadataInt64(metadata, keys...)
	if value <= 0 {
		return 0
	}
	if value > int64(^uint(0)>>1) {
		return int(^uint(0) >> 1)
	}
	return int(value)
}

func metadataInt64(metadata domain.JSONMap, keys ...string) int64 {
	for _, key := range keys {
		raw, ok := metadata[key]
		if !ok {
			continue
		}
		switch typed := raw.(type) {
		case int:
			return int64(typed)
		case int64:
			return typed
		case float64:
			return int64(typed)
		case json.Number:
			parsed, _ := typed.Int64()
			return parsed
		case string:
			parsed, err := strconv.ParseInt(strings.TrimSpace(typed), 10, 64)
			if err == nil {
				return parsed
			}
		}
	}
	return 0
}

func (deps ServerDeps) resourceAccounting(ctx context.Context) resourceAccountingSummary {
	accounting := resourceAccountingSummary{
		Users:    map[string]resourceOwnerAccounting{},
		Orgs:     map[string]resourceOwnerAccounting{},
		Projects: map[string]resourceOwnerAccounting{},
	}
	if catalog, ok := deps.Store.(resourceCatalogAdminStore); ok {
		resources, err := catalog.ListResources(ctx, 100000, 0)
		if err == nil {
			for _, resource := range resources {
				switch strings.TrimSpace(resource.Status) {
				case "deleted":
					accounting.SoftDeletedResources++
					continue
				case "active":
				default:
					continue
				}
				accounting.ActiveResources++
				accounting.ActiveBytes += resource.SizeBytes
				userID := strings.TrimSpace(resource.OwnerUserID)
				if userID == "" {
					userID = "local-user"
				}
				owner := accounting.Users[userID]
				owner.Uploads++
				owner.StorageBytes += resource.SizeBytes
				accounting.Users[userID] = owner
				orgID := strings.TrimSpace(resource.OwnerOrgID)
				if orgID != "" {
					org := accounting.Orgs[orgID]
					org.Uploads++
					org.StorageBytes += resource.SizeBytes
					accounting.Orgs[orgID] = org
				}
				projectID := strings.TrimSpace(resource.ProjectID)
				if projectID != "" {
					project := accounting.Projects[projectID]
					project.Uploads++
					project.StorageBytes += resource.SizeBytes
					accounting.Projects[projectID] = project
				}
			}
			return accounting
		}
	}
	uploads, bytes := deps.uploadStats()
	accounting.ActiveResources = uploads
	accounting.ActiveBytes = bytes
	return accounting
}

func (deps ServerDeps) handleEmptyTrainingDatasets(w http.ResponseWriter, r *http.Request) {
	_ = deps
	writeJSON(w, http.StatusOK, map[string]any{"count": 0, "datasets": []any{}})
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
	principal := deps.principalFromRequest(r, "")
	if _, err := deps.Store.GetRunForUser(r.Context(), runID, principal.UserID); err != nil {
		writeStoreError(w, err)
		return
	}
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

type steerRunRequest struct {
	SteerID string `json:"steer_id"`
	Text    string `json:"text"`
}

type ackRunSteerRequest struct {
	WorkerID string `json:"worker_id"`
}

// handleSteerRun accepts a mid-run steering message from the run's owner.
// 409 with code "steering_closed" means the run is terminal or finalizing —
// the client falls back to Phase 0 queueing.
func (deps ServerDeps) handleSteerRun(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	var req steerRunRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	runID := chi.URLParam(r, "run_id")
	principal := deps.principalFromRequest(r, "")
	if _, err := deps.Store.GetRunForUser(r.Context(), runID, principal.UserID); err != nil {
		writeStoreError(w, err)
		return
	}
	record, err := deps.Runs.SteerRun(r.Context(), runcontrol.SteerRunRequest{
		RunID:   runID,
		UserID:  principal.UserID,
		SteerID: req.SteerID,
		Text:    req.Text,
	})
	if err != nil {
		if errors.Is(err, runcontrol.ErrInvalidSteer) {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		if errors.Is(err, store.ErrSteeringClosed) {
			writeJSON(w, http.StatusConflict, map[string]string{
				"error": err.Error(),
				"code":  "steering_closed",
			})
			return
		}
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, record)
}

func (deps ServerDeps) handleListRunSteerMessages(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	runID := chi.URLParam(r, "run_id")
	if _, ok := deps.runForWorkerOrUser(w, r, runID); !ok {
		return
	}
	records, err := deps.Runs.ListRunSteerMessages(r.Context(), runID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if records == nil {
		records = []domain.RunSteerMessageRecord{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"steer_messages": records})
}

// handleCloseRunSteerBarrier is worker-only: it atomically stops steer
// acceptance for a finalizing run and returns the still-pending steers the
// worker must apply before its terminal event.
func (deps ServerDeps) handleCloseRunSteerBarrier(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	if deps.workerRequestAuth(r) != workerAuthValid {
		writeError(w, http.StatusForbidden, errors.New("steer barrier is worker-only"))
		return
	}
	runID := chi.URLParam(r, "run_id")
	pending, err := deps.Runs.CloseRunSteerBarrier(r.Context(), runID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if pending == nil {
		pending = []domain.RunSteerMessageRecord{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"pending": pending})
}

func (deps ServerDeps) handleAckRunSteerMessage(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	if deps.workerRequestAuth(r) != workerAuthValid {
		writeError(w, http.StatusForbidden, errors.New("steer ack is worker-only"))
		return
	}
	var req ackRunSteerRequest
	if r.Body != nil && r.ContentLength != 0 {
		if !decodeJSON(w, r, &req) {
			return
		}
	}
	record, err := deps.Runs.AckRunSteerMessage(r.Context(), runcontrol.AckRunSteerRequest{
		RunID:    chi.URLParam(r, "run_id"),
		SteerID:  chi.URLParam(r, "steer_id"),
		WorkerID: req.WorkerID,
	})
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, record)
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
	runID := chi.URLParam(r, "run_id")
	if _, ok := deps.runForWorkerOrUser(w, r, runID); !ok {
		return
	}
	lease, err := deps.Runs.AcquireRunLease(r.Context(), runcontrol.AcquireRunLeaseRequest{
		RunID:    runID,
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
	runID := chi.URLParam(r, "run_id")
	if _, ok := deps.runForWorkerOrUser(w, r, runID); !ok {
		return
	}
	lease, err := deps.Runs.RenewRunLease(r.Context(), runcontrol.RenewRunLeaseRequest{
		RunID:      runID,
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
	runID := chi.URLParam(r, "run_id")
	if _, ok := deps.runForWorkerOrUser(w, r, runID); !ok {
		return
	}
	if err := deps.Runs.ReleaseRunLease(r.Context(), runcontrol.ReleaseRunLeaseRequest{
		RunID:      runID,
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
	// An opaque ?cursor= (keyset convention) takes precedence over the legacy after_sequence param;
	// a malformed cursor is ignored so a stale token degrades to "from the start" rather than 400.
	if cursor := strings.TrimSpace(r.URL.Query().Get("cursor")); cursor != "" {
		if seq, ok := decodeSeqCursor(cursor); ok {
			afterSequence, hasAfterSequence = seq, true
		}
	}
	workerAuth := deps.workerRequestAuth(r)
	if r.URL.Query().Get("stream") == "true" {
		switch workerAuth {
		case workerAuthValid:
			if _, err := deps.Store.GetRun(r.Context(), runID); err != nil {
				writeStoreError(w, err)
				return
			}
		case workerAuthInvalid:
			writeError(w, http.StatusUnauthorized, errors.New("invalid worker token"))
			return
		default:
			principal := deps.principalFromRequest(r, "")
			if _, err := deps.Store.GetRunForUser(r.Context(), runID, principal.UserID); err != nil {
				writeStoreError(w, err)
				return
			}
		}
		deps.streamRunEvents(w, r, runID, afterSequence, hasAfterSequence, limit)
		return
	}
	var events []domain.RunEventRecord
	var err error
	switch workerAuth {
	case workerAuthValid:
		if hasAfterSequence {
			events, err = deps.Store.ListRunEventsAfter(r.Context(), runID, afterSequence, limit)
		} else {
			events, err = deps.Store.ListRunEvents(r.Context(), runID, limit)
		}
	case workerAuthInvalid:
		writeError(w, http.StatusUnauthorized, errors.New("invalid worker token"))
		return
	default:
		principal := deps.principalFromRequest(r, "")
		if hasAfterSequence {
			events, err = deps.Store.ListRunEventsAfterForUser(r.Context(), runID, principal.UserID, afterSequence, limit)
		} else {
			events, err = deps.Store.ListRunEventsForUser(r.Context(), runID, principal.UserID, limit)
		}
	}
	if err != nil {
		writeStoreError(w, err)
		return
	}
	resp := runEventsResponse{RunID: runID, Count: len(events), Events: events}
	// Cursor pagination applies only to FORWARD drain (after_sequence / cursor present), where the
	// page is an ascending slice keyed on sequence — a full page means more may follow, so hand back
	// an opaque cursor at the last (max) sequence. The no-cursor call returns the newest tail (a
	// convenience snapshot, not a drain), so it carries no forward cursor. A forward drain begins at
	// ?after_sequence=0. (events[last].Sequence is the max in a forward page.)
	if hasAfterSequence && limit > 0 && len(events) == limit {
		resp.NextCursor = encodeSeqCursor(events[len(events)-1].Sequence)
		resp.HasMore = true
	}
	writeJSON(w, http.StatusOK, resp)
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
			if event.RunID != runID || event.Sequence <= cursor {
				continue
			}
			// Treat local fanout as a wake-up signal and deliver from the
			// store instead of writing the bus event directly. Another
			// replica may have ingested earlier sequences that never reached
			// this replica's bus; writing the bus event first would emit
			// deltas out of order and scramble streamed text. The store read
			// returns everything at or below this event's sequence in order.
			if !catchUpFromStore() {
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
	principal := deps.principalFromRequest(r, "")
	artifacts, err := deps.Store.ListRunArtifactsForUser(r.Context(), runID, principal.UserID, parseLimit(r, 500))
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
	principal := deps.principalFromRequest(r, "")
	artifact, err := deps.Store.GetArtifactForUser(r.Context(), chi.URLParam(r, "artifact_id"), principal.UserID)
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
	principal := deps.principalFromRequest(r, "")
	path := strings.TrimSpace(r.URL.Query().Get("path"))
	if path == "" {
		writeError(w, http.StatusBadRequest, errors.New("path query parameter is required"))
		return
	}
	artifacts, err := deps.Store.ListRunArtifactsForUser(r.Context(), runID, principal.UserID, 5000)
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
	principal := deps.principalFromRequest(r, "")
	artifact, err := deps.Store.GetArtifactForUser(r.Context(), chi.URLParam(r, "artifact_id"), principal.UserID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	deps.serveArtifactRecord(w, r, artifact)
}

func (deps ServerDeps) handlePromoteArtifactResource(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource catalog is not configured"})
		return
	}
	if strings.TrimSpace(deps.ArtifactRoot) == "" {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "artifact root is not configured"})
		return
	}
	uploadRoot, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var req promoteArtifactResourceRequest
	if r.Body != nil && r.ContentLength != 0 {
		if !decodeJSON(w, r, &req) {
			return
		}
	}
	principal := deps.principalFromRequest(r, "")
	artifact, err := deps.Store.GetArtifactForUser(r.Context(), chi.URLParam(r, "artifact_id"), principal.UserID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	artifactPath, err := resolveArtifactDownloadPath(deps.ArtifactRoot, artifact)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	originalName := safeOriginalFilename(req.OriginalName)
	if originalName == "upload.bin" && strings.TrimSpace(req.OriginalName) == "" {
		originalName = artifactName(artifact, artifactPath)
	}
	record, err := copyFileIntoUploadRoot(uploadRoot, artifactPath, originalName, artifact.MimeType, principal, uploadMetadataRecord{
		Principal:  principal.record(),
		SourceURI:  "/v2/artifacts/" + artifact.ArtifactID,
		SourceType: "artifact",
		ProjectID:  strings.TrimSpace(req.ProjectID),
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if err := deps.enforceResourceQuota(r.Context(), principal, record.ProjectID, record.SizeBytes); err != nil {
		_ = removeUploadedFile(uploadRoot, record.FileID)
		writeResourceQuotaError(w, err)
		return
	}
	resourceRecord := resourceRecord{
		FileID:        record.FileID,
		OriginalName:  record.OriginalName,
		ContentType:   record.ContentType,
		SizeBytes:     record.SizeBytes,
		SHA256:        record.SHA256,
		CreatedAt:     record.CreatedAt,
		SourceType:    "artifact",
		ResourceKind:  resourceKindForContent(record.OriginalName, record.ContentType),
		SourceURI:     record.SourceURI,
		ProjectID:     strings.TrimSpace(req.ProjectID),
		PreviewURL:    record.PreviewURL,
		Principal:     principal.record(),
		StagedLocally: true,
		CacheReady:    true,
	}
	if err := deps.catalogResourceRecord(r.Context(), uploadRoot, resourceRecord, "resource.promoted"); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	resource, err := catalog.GetResourceForUser(r.Context(), record.FileID, principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	promotionMetadata := domain.JSONMap{
		"artifact_id":            artifact.ArtifactID,
		"run_id":                 artifact.RunID,
		"artifact_kind":          artifact.Kind,
		"artifact_path":          artifact.Path,
		"artifact_title":         artifact.Title,
		"artifact_mime_type":     artifact.MimeType,
		"promoted_resource_id":   resource.ResourceID,
		"promoted_original_name": resource.OriginalName,
		"source_uri":             resource.SourceURI,
		"source_type":            resource.SourceType,
		"resource_kind":          resource.ResourceKind,
		"content_type":           resource.ContentType,
		"size_bytes":             resource.SizeBytes,
		"sha256":                 resource.SHA256,
	}
	if strings.TrimSpace(artifact.SourcePath) != "" {
		promotionMetadata["artifact_source_path"] = artifact.SourcePath
	}
	if strings.TrimSpace(artifact.PreviewPath) != "" {
		promotionMetadata["artifact_preview_path"] = artifact.PreviewPath
	}
	if strings.TrimSpace(artifact.ResultGroupID) != "" {
		promotionMetadata["artifact_result_group_id"] = artifact.ResultGroupID
	}
	if strings.TrimSpace(artifact.StorageURI) != "" {
		promotionMetadata["artifact_storage_uri"] = artifact.StorageURI
	}
	if strings.TrimSpace(artifact.SHA256) != "" {
		promotionMetadata["artifact_sha256"] = artifact.SHA256
	}
	if artifact.SizeBytes > 0 {
		promotionMetadata["artifact_size_bytes"] = artifact.SizeBytes
	}
	if strings.TrimSpace(artifact.ToolName) != "" {
		promotionMetadata["artifact_tool_name"] = artifact.ToolName
	}
	if strings.TrimSpace(artifact.Category) != "" {
		promotionMetadata["artifact_category"] = artifact.Category
	}
	if strings.TrimSpace(resource.ProjectID) != "" {
		promotionMetadata["project_id"] = resource.ProjectID
	}
	deps.recordResourceEvent(r.Context(), resource.ResourceID, principal, "resource.artifact_promoted", promotionMetadata)
	writeJSON(w, http.StatusCreated, resourceResponse{Resource: deps.resourceRecordFromCatalog(uploadRoot, resource)})
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

// deleteArtifactBlobs unlinks the files behind a deleted conversation's
// artifacts. The rows go with the cascade; the bytes have to be removed here or
// a "permanent" delete leaves the user's figures on disk.
//
// Every path is re-validated against ArtifactRoot before a single unlink. These
// URIs come out of the database, and a value that had ever been poisoned would
// otherwise turn conversation deletion into an arbitrary-file-delete primitive;
// pathIsUnderRoot is the same containment gate the download path uses. Non-file
// URIs are skipped rather than guessed at — a future object-store backend needs
// its own deliberate implementation, not a filepath heuristic.
//
// Best-effort by design, and called only after the row deletion has committed:
// the conversation is already gone, so a failure here is reclaimable storage,
// not a failed delete. Never surface it to the caller.
func (deps ServerDeps) deleteArtifactBlobs(ctx context.Context, storageURIs []string) {
	root := strings.TrimSpace(deps.ArtifactRoot)
	if root == "" {
		return
	}
	seen := make(map[string]struct{}, len(storageURIs))
	removed, skipped, failed := 0, 0, 0
	for _, uri := range storageURIs {
		path := fileStoragePath(uri)
		if path == "" {
			skipped++
			continue
		}
		resolved := filepath.Clean(path)
		if !pathIsUnderRoot(root, resolved) {
			// Outside the root we own: refuse, and say so — this is the shape a
			// poisoned storage_uri would take.
			slog.WarnContext(ctx, "refusing to unlink artifact outside artifact root",
				"path", resolved, "artifact_root", root)
			skipped++
			continue
		}
		if _, dup := seen[resolved]; dup {
			continue
		}
		seen[resolved] = struct{}{}
		if err := os.Remove(resolved); err != nil {
			if !errors.Is(err, fs.ErrNotExist) {
				failed++
				slog.WarnContext(ctx, "artifact blob unlink failed", "path", resolved, "error", err)
			}
			continue
		}
		removed++
	}
	if removed > 0 || failed > 0 || skipped > 0 {
		slog.InfoContext(ctx, "artifact blobs cleaned after conversation delete",
			"removed", removed, "skipped", skipped, "failed", failed)
	}
}

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

func safePathToken(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	var builder strings.Builder
	for _, char := range value {
		switch {
		case char >= 'a' && char <= 'z':
			builder.WriteRune(char)
		case char >= 'A' && char <= 'Z':
			builder.WriteRune(char)
		case char >= '0' && char <= '9':
			builder.WriteRune(char)
		case unicode.IsLetter(char) || unicode.IsDigit(char):
			builder.WriteRune(char)
		case char == '.' || char == '_' || char == '-':
			builder.WriteRune(char)
		default:
			builder.WriteRune('_')
		}
	}
	cleaned := strings.Trim(builder.String(), ".")
	if cleaned == "" {
		return ""
	}
	return cleaned
}

func uploadSessionStagingRoot(root string, sessionID string) string {
	return filepath.Join(root, ".upload_sessions", safePathToken(sessionID))
}

func uploadSessionFileStagingDir(root string, sessionID string, fileToken string) string {
	return filepath.Join(uploadSessionStagingRoot(root, sessionID), safePathToken(fileToken))
}

func uploadSessionChunkPath(root string, sessionID string, fileToken string, chunkIndex int) string {
	return filepath.Join(uploadSessionFileStagingDir(root, sessionID, fileToken), fmt.Sprintf("chunk-%08d.part", chunkIndex))
}

func uploadSessionChunkLocalPath(root string, sessionID string, chunk domain.UploadChunkRecord) (string, error) {
	path := ""
	if strings.TrimSpace(chunk.StorageURI) != "" {
		parsed, err := url.Parse(strings.TrimSpace(chunk.StorageURI))
		if err == nil && parsed.Scheme == "file" {
			path = parsed.Path
		}
	}
	if path == "" {
		path = uploadSessionChunkPath(root, sessionID, chunk.FileToken, chunk.ChunkIndex)
	}
	path = filepath.Clean(path)
	if !pathIsUnderRoot(uploadSessionFileStagingDir(root, sessionID, chunk.FileToken), path) {
		return "", errUnsafeArtifactPath
	}
	return path, nil
}

func uploadSessionSourceURI(sessionID string, fileToken string) string {
	return "upload-session://" + url.PathEscape(sessionID) + "/" + url.PathEscape(fileToken)
}

func (deps ServerDeps) resolvedUploadRoot() (string, error) {
	root := strings.TrimSpace(deps.UploadRoot)
	if root == "" {
		root = "data/uploads"
	}
	return filepath.Abs(filepath.Clean(root))
}

func (deps ServerDeps) resourceCatalogStore() (resourceCatalogStore, bool) {
	if deps.Store == nil {
		return nil, false
	}
	catalog, ok := deps.Store.(resourceCatalogStore)
	return catalog, ok
}

func (deps ServerDeps) resourceOwnerLookupStore() (resourceOwnerLookupStore, bool) {
	if deps.Store == nil {
		return nil, false
	}
	lookup, ok := deps.Store.(resourceOwnerLookupStore)
	return lookup, ok
}

func (deps ServerDeps) resourceCollectionStore() (resourceCollectionStore, bool) {
	if deps.Store == nil {
		return nil, false
	}
	collections, ok := deps.Store.(resourceCollectionStore)
	return collections, ok
}

func (deps ServerDeps) datasetSnapshotStore() (datasetSnapshotStore, bool) {
	if deps.Store == nil {
		return nil, false
	}
	snapshots, ok := deps.Store.(datasetSnapshotStore)
	return snapshots, ok
}

func (deps ServerDeps) dataAgentJobStore() (dataAgentJobStore, bool) {
	if deps.Store == nil {
		return nil, false
	}
	jobs, ok := deps.Store.(dataAgentJobStore)
	return jobs, ok
}

func (deps ServerDeps) uploadSessionStore() (uploadSessionStore, bool) {
	if deps.Store == nil {
		return nil, false
	}
	sessions, ok := deps.Store.(uploadSessionStore)
	return sessions, ok
}

func (deps ServerDeps) ensureUploadCatalogMigrated(ctx context.Context, root string) error {
	if _, ok := deps.resourceCatalogStore(); !ok {
		return nil
	}
	stateValue, _ := uploadCatalogMigrations.LoadOrStore(root, &uploadCatalogMigrationState{})
	state := stateValue.(*uploadCatalogMigrationState)
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.done {
		return nil
	}
	// This one-time, process-wide catalog backfill must not be tied to the
	// triggering request's lifecycle. If that request is canceled (client abort,
	// navigation, a superseded fetch) while the migration runs, a request-scoped
	// context returns context.Canceled — and the previous sync.Once cached it, so
	// EVERY later resource request returned "context canceled" until a restart.
	// Detach cancellation (keep a bounded deadline) and only mark the migration
	// done on success, so a genuine failure simply retries on the next request.
	migrateCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Minute)
	defer cancel()
	if err := deps.migrateUploadCatalog(migrateCtx, root); err != nil {
		return err
	}
	state.done = true
	return nil
}

func (deps ServerDeps) migrateUploadCatalog(ctx context.Context, root string) error {
	entries, err := os.ReadDir(root)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	ownerLookup, hasOwnerLookup := deps.Store.(resourceOwnerLookupStore)
	batchOwnerLookup, hasBatchOwnerLookup := deps.Store.(resourceOwnerBatchLookupStore)

	type uploadCatalogOwnerKey struct {
		userID string
		orgID  string
	}
	type uploadCatalogCandidate struct {
		fileID   string
		path     string
		metadata uploadMetadataRecord
	}
	candidates := make([]uploadCatalogCandidate, 0, len(entries))
	batchLookupIDs := map[uploadCatalogOwnerKey][]string{}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		path := filepath.Clean(filepath.Join(root, entry.Name()))
		if !pathIsUnderRoot(root, path) {
			continue
		}
		fileID, _ := uploadNameParts(entry.Name())
		if !safeUploadID(fileID) {
			continue
		}
		metadata := uploadMetadataRecord{}
		if hasOwnerLookup || hasBatchOwnerLookup {
			metadata = readUploadMetadata(root, fileID)
			userID := strings.TrimSpace(metadata.Principal.UserID)
			if userID != "" && hasBatchOwnerLookup {
				key := uploadCatalogOwnerKey{userID: userID, orgID: strings.TrimSpace(metadata.Principal.OrgID)}
				batchLookupIDs[key] = append(batchLookupIDs[key], fileID)
			}
			candidates = append(candidates, uploadCatalogCandidate{
				fileID:   fileID,
				path:     path,
				metadata: metadata,
			})
			continue
		}
		candidates = append(candidates, uploadCatalogCandidate{
			fileID: fileID,
			path:   path,
		})
	}

	existingByFileID := map[string]bool{}
	if hasBatchOwnerLookup {
		for key, resourceIDs := range batchLookupIDs {
			existing, err := batchOwnerLookup.ListResourceIDsForOwner(ctx, key.userID, key.orgID, resourceIDs)
			if err != nil {
				return err
			}
			for resourceID := range existing {
				existingByFileID[resourceID] = true
			}
		}
	}

	for _, candidate := range candidates {
		if existingByFileID[candidate.fileID] {
			continue
		}
		if !hasBatchOwnerLookup && hasOwnerLookup {
			userID := strings.TrimSpace(candidate.metadata.Principal.UserID)
			if userID != "" {
				if _, err := ownerLookup.GetResourceForOwner(ctx, candidate.fileID, userID, strings.TrimSpace(candidate.metadata.Principal.OrgID)); err == nil {
					continue
				} else if !errors.Is(err, store.ErrNotFound) {
					return err
				}
			}
		}
		record, err := uploadResourceFromPath(root, candidate.path)
		if err != nil {
			continue
		}
		if err := deps.catalogResourceRecordAtPathWithEventMetadata(ctx, root, candidate.path, record, "resource.migrated", nil); err != nil {
			return err
		}
	}
	return nil
}

func (deps ServerDeps) catalogUploadedFile(ctx context.Context, root string, record uploadedFileRecord, eventType string) error {
	return deps.catalogUploadedFileWithEventMetadata(ctx, root, record, eventType, nil)
}

func (deps ServerDeps) catalogUploadedFileWithEventMetadata(ctx context.Context, root string, record uploadedFileRecord, eventType string, eventMetadata domain.JSONMap) error {
	resource, _, err := findUploadResource(root, record.FileID)
	if err != nil {
		return err
	}
	if record.SourceURI != "" {
		resource.SourceURI = record.SourceURI
	}
	if record.Principal.UserID != "" {
		resource.Principal = record.Principal
	}
	// Preserve the content type resolved at upload time; re-deriving from the
	// on-disk name alone can downgrade declared types to octet-stream.
	if contentType := strings.TrimSpace(record.ContentType); contentType != "" &&
		contentType != "application/octet-stream" {
		resource.ContentType = contentType
	}
	resource.TrustedSourceRunID = record.TrustedSourceRunID
	return deps.catalogResourceRecordWithEventMetadata(ctx, root, resource, eventType, eventMetadata)
}

func (deps ServerDeps) catalogUploadedFileAtPath(ctx context.Context, root string, path string, record uploadedFileRecord, eventType string) error {
	resource := resourceRecord{
		FileID:             record.FileID,
		OriginalName:       record.OriginalName,
		ContentType:        contentTypeForUpload(record.OriginalName, record.ContentType),
		SizeBytes:          record.SizeBytes,
		SHA256:             record.SHA256,
		CreatedAt:          record.CreatedAt,
		SourceType:         "upload",
		ResourceKind:       resourceKindForContent(record.OriginalName, record.ContentType),
		SourceURI:          record.SourceURI,
		ProjectID:          record.ProjectID,
		PreviewURL:         record.PreviewURL,
		Principal:          record.Principal,
		TrustedSourceRunID: record.TrustedSourceRunID,
	}
	return deps.catalogResourceRecordAtPathWithEventMetadata(ctx, root, path, resource, eventType, nil)
}

func (deps ServerDeps) catalogResourceRecord(ctx context.Context, root string, record resourceRecord, eventType string) error {
	return deps.catalogResourceRecordWithEventMetadata(ctx, root, record, eventType, nil)
}

func (deps ServerDeps) catalogResourceRecordWithEventMetadata(ctx context.Context, root string, record resourceRecord, eventType string, eventMetadata domain.JSONMap) error {
	if _, ok := deps.resourceCatalogStore(); !ok {
		return nil
	}
	record.Principal = normalizedResourcePrincipal(record.Principal)
	_, path, err := findUploadResource(root, record.FileID)
	if err != nil {
		return err
	}
	return deps.catalogResourceRecordAtPathWithEventMetadata(ctx, root, path, record, eventType, eventMetadata)
}

func (deps ServerDeps) catalogResourceRecordAtPathWithEventMetadata(ctx context.Context, root string, path string, record resourceRecord, eventType string, eventMetadata domain.JSONMap) error {
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		return nil
	}
	record.Principal = normalizedResourcePrincipal(record.Principal)
	path = filepath.Clean(path)
	if !pathIsUnderRoot(root, path) {
		return errUnsafeArtifactPath
	}
	if _, err := os.Stat(path); err != nil {
		return err
	}
	createdAt := parseResourceCreatedAt(record.CreatedAt)
	input := domain.UpsertResourceInput{
		ResourceID:   record.FileID,
		OriginalName: record.OriginalName,
		ContentType:  record.ContentType,
		SizeBytes:    record.SizeBytes,
		SHA256:       record.SHA256,
		StorageURI:   fileStorageURI(path),
		StoragePath:  filepath.Base(path),
		SourceType:   record.SourceType,
		ResourceKind: record.ResourceKind,
		SourceURI:    record.SourceURI,
		ProjectID:    record.ProjectID,
		OwnerUserID:  record.Principal.UserID,
		OwnerOrgID:   record.Principal.OrgID,
		OwnerRole:    record.Principal.Role,
		Status:       "active",
		CreatedAt:    createdAt,
		UpdatedAt:    domain.Now(),
		Tags:         record.Tags,
		Metadata:     uploadCatalogMetadataForPath(path, record),
	}
	resource, err := catalog.UpsertResource(ctx, input)
	if err != nil {
		return err
	}
	deps.recordResourceEvent(ctx, resource.ResourceID, requestPrincipal{
		UserID: record.Principal.UserID,
		OrgID:  record.Principal.OrgID,
		Role:   record.Principal.Role,
	}, eventType, catalogResourceEventMetadata(record, eventMetadata))
	// Convert-on-upload: kick off a tiled-pyramid derivation so the Scientific
	// Viewer can serve bounded tiles for large images (best-effort).
	deps.maybeEnqueuePyramidDerivation(ctx, root, record, path, eventType)
	return nil
}

func uploadCatalogMetadataForPath(path string, record resourceRecord) domain.JSONMap {
	metadata := domain.JSONMap{
		"source": "upload_store",
	}
	if sourceRunID := strings.TrimSpace(record.TrustedSourceRunID); sourceRunID != "" {
		metadata["source_run_id"] = sourceRunID
		metadata["source_authority"] = "trusted_worker_run"
	}
	if header := uploadImageHeaderMetadataForPath(path, record); len(header) > 0 {
		metadata["image_header"] = header
	}
	if exif := uploadEXIFMetadataForPath(path, record.ContentType); len(exif) > 0 {
		metadata["exif"] = exif
	}
	return metadata
}

func uploadImageHeaderMetadataForPath(path string, record resourceRecord) domain.JSONMap {
	if isNiftiUpload(record.OriginalName, record.ContentType) {
		geom, err := readNiftiHeaderGeometry(path)
		if err != nil {
			return nil
		}
		dimsOrder := niftiGeometryDimsOrder(geom)
		return domain.JSONMap{
			"reader":                 "nifti-1",
			"width":                  float64(geom.width),
			"height":                 float64(geom.height),
			"depth":                  float64(geom.depth),
			"time_count":             float64(geom.timeCount),
			"channel_count":          float64(geom.channelCount),
			"dims_order":             dimsOrder,
			"array_shape":            niftiGeometryArrayShape(geom, dimsOrder),
			"array_dtype":            geom.dtype,
			"bytes_per_voxel":        float64(geom.bytesPerVoxel),
			"physical_spacing":       domain.JSONMap{"x": geom.spacingX, "y": geom.spacingY, "z": geom.spacingZ},
			"physical_spacing_unit":  geom.spaceUnit,
			"voxel_offset":           float64(geom.voxOffset),
			"rescale_slope":          geom.sclSlope,
			"rescale_intercept":      geom.sclInter,
			"affine_method":          niftiAffineMethodName(geom.affineCode),
			"content_type":           record.ContentType,
			"size_bytes":             float64(record.SizeBytes),
			"sha256":                 record.SHA256,
			"source_metadata_reader": "catalog_header",
		}
	}
	if record.ResourceKind != "image" && !strings.HasPrefix(strings.ToLower(strings.TrimSpace(record.ContentType)), "image/") {
		return nil
	}
	descriptor := uploadImageDescriptorForPath(path, record.ContentType)
	reader := "go-image"
	if descriptor.OME != nil {
		reader = "ome-tiff+xml+go-image"
	}
	header := domain.JSONMap{
		"reader":                 reader,
		"width":                  float64(descriptor.Width),
		"height":                 float64(descriptor.Height),
		"depth":                  float64(descriptor.Depth),
		"time_count":             float64(descriptor.TimeCount),
		"channel_count":          float64(descriptor.ChannelCount),
		"dims_order":             descriptor.DimsOrder,
		"array_shape":            intsAsJSONNumbers(descriptor.ArrayShape),
		"array_dtype":            descriptor.ArrayDType,
		"warnings":               append([]string(nil), descriptor.Warnings...),
		"content_type":           record.ContentType,
		"size_bytes":             float64(record.SizeBytes),
		"sha256":                 record.SHA256,
		"source_metadata_reader": "catalog_header",
	}
	if descriptor.OME != nil {
		header["physical_spacing"] = omePhysicalSpacing(descriptor.OME)
		header["scene"] = descriptor.OME.SceneName
		header["scene_count"] = float64(positiveIntOr(descriptor.OME.SceneCount, 1))
		header["microscopy"] = domain.JSONMap{
			"channel_names":      append([]string(nil), descriptor.OME.ChannelNames...),
			"dimensions_present": descriptor.DimsOrder,
			"current_scene":      descriptor.OME.SceneName,
			"scene_names":        []string{descriptor.OME.SceneName},
		}
	}
	return header
}

func intsAsJSONNumbers(values []int) []any {
	out := make([]any, 0, len(values))
	for _, value := range values {
		out = append(out, float64(value))
	}
	return out
}

func catalogResourceEventMetadata(record resourceRecord, extra domain.JSONMap) domain.JSONMap {
	metadata := domain.JSONMap{
		"source_type":   record.SourceType,
		"resource_kind": record.ResourceKind,
		"original_name": record.OriginalName,
		"size_bytes":    record.SizeBytes,
		"sha256":        record.SHA256,
	}
	if strings.TrimSpace(record.SourceURI) != "" {
		metadata["source_uri"] = record.SourceURI
	}
	if strings.TrimSpace(record.ProjectID) != "" {
		metadata["project_id"] = record.ProjectID
	}
	for key, value := range extra {
		metadata[key] = value
	}
	return metadata
}

func normalizedResourcePrincipal(principal principalRecord) principalRecord {
	principal.UserID = strings.TrimSpace(principal.UserID)
	principal.OrgID = strings.TrimSpace(principal.OrgID)
	principal.Role = strings.TrimSpace(principal.Role)
	if principal.UserID == "" {
		principal.UserID = "local-user"
	}
	if principal.Role == "" {
		principal.Role = "researcher"
	}
	return principal
}

func parseResourceCreatedAt(value string) time.Time {
	value = strings.TrimSpace(value)
	if value == "" {
		return domain.Now()
	}
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return domain.Now()
	}
	return parsed.UTC()
}

func fileStorageURI(path string) string {
	return (&url.URL{Scheme: "file", Path: filepath.Clean(path)}).String()
}

func (deps ServerDeps) recordResourceEvent(ctx context.Context, resourceID string, principal requestPrincipal, eventType string, metadata domain.JSONMap) {
	_, _ = deps.appendResourceEvent(ctx, resourceID, principal, eventType, metadata)
}

func (deps ServerDeps) appendResourceEvent(ctx context.Context, resourceID string, principal requestPrincipal, eventType string, metadata domain.JSONMap) (domain.ResourceEventRecord, bool) {
	events, ok := deps.Store.(resourceEventStore)
	if !ok || strings.TrimSpace(resourceID) == "" || strings.TrimSpace(eventType) == "" {
		return domain.ResourceEventRecord{}, false
	}
	event, err := events.CreateResourceEvent(ctx, domain.AppendResourceEventInput{
		ResourceID:  resourceID,
		ActorUserID: principal.UserID,
		ActorOrgID:  principal.OrgID,
		EventType:   eventType,
		Metadata:    metadata,
	})
	if err != nil {
		return domain.ResourceEventRecord{}, false
	}
	return event, true
}

func (deps ServerDeps) recordUploadSessionEvent(ctx context.Context, session domain.UploadSessionRecord, principal requestPrincipal, eventType string, metadata domain.JSONMap) {
	events, ok := deps.Store.(uploadSessionEventStore)
	if !ok || strings.TrimSpace(session.SessionID) == "" || strings.TrimSpace(eventType) == "" {
		return
	}
	eventMetadata := uploadSessionEventMetadata(session)
	for key, value := range metadata {
		eventMetadata[key] = value
	}
	_, _ = events.AppendUploadSessionEvent(ctx, domain.AppendUploadSessionEventInput{
		SessionID:   session.SessionID,
		ActorUserID: principal.UserID,
		ActorOrgID:  principal.OrgID,
		EventType:   eventType,
		Metadata:    eventMetadata,
	})
}

func (deps ServerDeps) recordUploadSessionFileCompleted(ctx context.Context, session domain.UploadSessionRecord, file domain.UploadSessionFileRecord, principal requestPrincipal) {
	metadata := domain.JSONMap{
		"file_token":      file.FileToken,
		"resource_id":     file.ResourceID,
		"original_name":   file.OriginalName,
		"relative_path":   file.RelativePath,
		"size_bytes":      file.SizeBytes,
		"computed_sha256": file.ComputedSHA256,
		"file_status":     file.Status,
	}
	deps.recordUploadSessionEvent(ctx, session, principal, "upload_session.file_completed", metadata)
	if session.Status == "completed" {
		deps.recordUploadSessionEvent(ctx, session, principal, "upload_session.completed", domain.JSONMap{
			"completed_at": session.CompletedAt.UTC().Format(time.RFC3339Nano),
		})
	}
}

func uploadSessionEventMetadata(session domain.UploadSessionRecord) domain.JSONMap {
	metadata := domain.JSONMap{
		"status":          session.Status,
		"source_type":     session.SourceType,
		"total_bytes":     session.TotalBytes,
		"bytes_received":  session.BytesReceived,
		"bytes_verified":  session.BytesVerified,
		"bytes_committed": session.BytesCommitted,
	}
	if strings.TrimSpace(session.ProjectID) != "" {
		metadata["project_id"] = session.ProjectID
	}
	if strings.TrimSpace(session.Error) != "" {
		metadata["error"] = session.Error
	}
	return metadata
}

func saveUploadedFile(root string, header *multipart.FileHeader, principal requestPrincipal, projectID string) (uploadedFileRecord, error) {
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
	if err := writeUploadMetadataRecord(root, fileID, uploadMetadataRecord{Principal: principal.record(), ProjectID: strings.TrimSpace(projectID)}); err != nil {
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
		ProjectID:    strings.TrimSpace(projectID),
		Principal:    principal.record(),
	}, nil
}

func copyFileIntoUploadRoot(root string, sourcePath string, originalName string, contentType string, principal requestPrincipal, metadata uploadMetadataRecord) (uploadedFileRecord, error) {
	sourcePath = filepath.Clean(sourcePath)
	source, err := os.Open(sourcePath)
	if err != nil {
		return uploadedFileRecord{}, err
	}
	defer source.Close()

	fileID := domain.NewID("file")
	if strings.TrimSpace(originalName) == "" {
		originalName = filepath.Base(sourcePath)
	}
	originalName = safeOriginalFilename(originalName)
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
	if metadata.Principal.UserID == "" {
		metadata.Principal = principal.record()
	}
	if metadata.SourceType == "" {
		metadata.SourceType = "upload"
	}
	if err := writeUploadMetadataRecord(root, fileID, metadata); err != nil {
		_ = os.Remove(target)
		return uploadedFileRecord{}, err
	}
	if strings.TrimSpace(contentType) == "" {
		contentType = contentTypeForUpload(originalName, "")
	}
	return uploadedFileRecord{
		FileID:       fileID,
		OriginalName: originalName,
		ContentType:  contentTypeForUpload(originalName, contentType),
		SizeBytes:    size,
		SHA256:       hex.EncodeToString(hasher.Sum(nil)),
		CreatedAt:    info.ModTime().UTC().Format(time.RFC3339Nano),
		SourceURI:    metadata.SourceURI,
		ProjectID:    strings.TrimSpace(metadata.ProjectID),
		PreviewURL:   "/v2/uploads/" + url.PathEscape(fileID) + "/preview",
		Principal:    metadata.Principal,
	}, nil
}

func removeUploadedFile(root string, fileID string) error {
	if !safeUploadID(fileID) {
		return errors.New("unsafe upload id")
	}
	var firstErr error
	patterns := []string{
		filepath.Join(root, fileID+"__*"),
		filepath.Join(root, fileID),
		filepath.Join(root, fileID+".*"),
	}
	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		for _, match := range matches {
			resolved := filepath.Clean(match)
			if !pathIsUnderRoot(root, resolved) {
				continue
			}
			if err := os.Remove(resolved); err != nil && !errors.Is(err, os.ErrNotExist) && firstErr == nil {
				firstErr = err
			}
		}
	}
	if err := os.Remove(uploadMetadataPath(root, fileID)); err != nil && !errors.Is(err, os.ErrNotExist) && firstErr == nil {
		firstErr = err
	}
	// Directory-format bundle (OME-Zarr): the whole bundles/{fileID} tree.
	if bundleDir := filepath.Join(root, bundlesDirName, fileID); pathIsUnderRoot(root, bundleDir) {
		if err := os.RemoveAll(bundleDir); err != nil && !errors.Is(err, os.ErrNotExist) && firstErr == nil {
			firstErr = err
		}
	}
	// Derived pyramid + its permanent-failure marker (for transcoded/derived formats).
	for _, p := range []string{
		filepath.Join(root, "derived", derivedPyramidName(fileID)),
		filepath.Join(root, "derived", derivedPyramidFailedName(fileID)),
	} {
		if pathIsUnderRoot(root, p) {
			if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) && firstErr == nil {
				firstErr = err
			}
		}
	}
	return firstErr
}

func uploadMetadataPath(root string, fileID string) string {
	return filepath.Join(root, ".meta", fileID+".json")
}

func writeUploadMetadata(root string, fileID string, principal requestPrincipal) error {
	return writeUploadMetadataRecord(root, fileID, uploadMetadataRecord{Principal: principal.record()})
}

type uploadMetadataRecord struct {
	Principal  principalRecord `json:"principal"`
	SourceURI  string          `json:"source_uri,omitempty"`
	SourceType string          `json:"source_type,omitempty"`
	ProjectID  string          `json:"project_id,omitempty"`
}

func writeUploadMetadataRecord(root string, fileID string, payload uploadMetadataRecord) error {
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
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func readUploadPrincipal(root string, fileID string) principalRecord {
	return readUploadMetadata(root, fileID).Principal
}

func readUploadMetadata(root string, fileID string) uploadMetadataRecord {
	if !safeUploadID(fileID) {
		return uploadMetadataRecord{}
	}
	path := uploadMetadataPath(root, fileID)
	if !pathIsUnderRoot(root, path) {
		return uploadMetadataRecord{}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return uploadMetadataRecord{}
	}
	var payload uploadMetadataRecord
	if err := json.Unmarshal(data, &payload); err != nil {
		return uploadMetadataRecord{}
	}
	return payload
}

type uploadResourceEntry struct {
	resourceRecord
	Path string
}

func listUploadResources(root string) ([]resourceRecord, error) {
	entries, err := listUploadResourceEntries(root)
	if err != nil {
		return nil, err
	}
	resources := make([]resourceRecord, 0, len(entries))
	for _, entry := range entries {
		resources = append(resources, entry.resourceRecord)
	}
	return resources, nil
}

func listUploadResourceEntries(root string) ([]uploadResourceEntry, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []uploadResourceEntry{}, nil
		}
		return nil, err
	}
	resources := make([]uploadResourceEntry, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		path := filepath.Join(root, entry.Name())
		record, err := uploadResourceFromPath(root, path)
		if err != nil {
			continue
		}
		resources = append(resources, uploadResourceEntry{
			resourceRecord: record,
			Path:           path,
		})
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

func (deps ServerDeps) findUploadResourceForRequest(ctx context.Context, root string, principal requestPrincipal, fileID string) (resourceRecord, string, error) {
	if catalog, ok := deps.resourceCatalogStore(); ok {
		if err := deps.ensureUploadCatalogMigrated(ctx, root); err != nil {
			return resourceRecord{}, "", err
		}
		resource, err := catalog.GetResourceForUser(ctx, fileID, principal.UserID, principal.OrgID)
		if err != nil {
			return resourceRecord{}, "", err
		}
		record := deps.resourceRecordFromCatalog(root, resource)
		path, err := resolveCatalogResourcePath(root, resource)
		if err != nil {
			return record, "", err
		}
		if _, err := os.Stat(path); err != nil {
			return record, "", store.ErrNotFound
		}
		return record, path, nil
	}
	record, path, err := findUploadResource(root, fileID)
	if err != nil {
		return resourceRecord{}, "", err
	}
	if !resourceVisibleToPrincipal(record, principal) {
		return resourceRecord{}, "", store.ErrNotFound
	}
	return record, path, nil
}

func (deps ServerDeps) resourceRecordFromCatalog(root string, resource domain.ResourceRecord) resourceRecord {
	path, pathErr := resolveCatalogResourcePath(root, resource)
	stagedLocally := false
	if pathErr == nil {
		if info, err := os.Stat(path); err == nil && !info.IsDir() {
			stagedLocally = true
		}
	}
	previewURL := "/v2/uploads/" + url.PathEscape(resource.ResourceID) + "/preview"
	// Read-time classification repair: chunked/bundle uploads persist an
	// "application/octet-stream" content type (handlers.go:3819,3944), so a large
	// CSV/JSON lands with a useless type and a "file" kind. Re-derive both from the
	// filename extension here (no storage mutation) so the catalog shows correct
	// icons and the viewer routes the file to the text/data viewer.
	contentType := strings.TrimSpace(resource.ContentType)
	if contentType == "" || contentType == "application/octet-stream" {
		contentType = contentTypeForUpload(resource.OriginalName, contentType)
	}
	resourceKind := strings.TrimSpace(resource.ResourceKind)
	if resourceKind == "" || resourceKind == "file" {
		resourceKind = resourceKindForContent(resource.OriginalName, contentType)
	}
	sourceType := strings.TrimSpace(resource.SourceType)
	if sourceType == "" {
		sourceType = "upload"
	}
	status := strings.TrimSpace(resource.Status)
	if status == "" {
		status = "active"
	}
	return resourceRecord{
		FileID:        resource.ResourceID,
		OriginalName:  resource.OriginalName,
		ContentType:   contentType,
		SizeBytes:     resource.SizeBytes,
		SHA256:        resource.SHA256,
		CreatedAt:     resource.CreatedAt.UTC().Format(time.RFC3339Nano),
		Status:        status,
		SourceType:    sourceType,
		ResourceKind:  resourceKind,
		SourceURI:     resource.SourceURI,
		ProjectID:     resource.ProjectID,
		HasThumbnail:  stagedLocally && strings.HasPrefix(contentType, "image/"),
		ThumbnailURL:  previewURL,
		PreviewURL:    previewURL,
		CacheReady:    stagedLocally,
		StagedLocally: stagedLocally,
		Principal: principalRecord{
			UserID: resource.OwnerUserID,
			OrgID:  resource.OwnerOrgID,
			Role:   resource.OwnerRole,
		},
		Tags:         append([]string(nil), resource.Tags...),
		Metadata:     resource.Metadata,
		ShareSummary: resource.ShareSummary,
	}
}

func (deps ServerDeps) uploadedFileRecordFromCatalog(root string, resource domain.ResourceRecord) uploadedFileRecord {
	record := deps.resourceRecordFromCatalog(root, resource)
	return uploadedFileRecord{
		FileID:       record.FileID,
		OriginalName: record.OriginalName,
		ContentType:  record.ContentType,
		SizeBytes:    record.SizeBytes,
		SHA256:       record.SHA256,
		CreatedAt:    record.CreatedAt,
		SourceURI:    record.SourceURI,
		ProjectID:    record.ProjectID,
		PreviewURL:   record.PreviewURL,
		Principal:    record.Principal,
	}
}

func resolveCatalogResourcePath(root string, resource domain.ResourceRecord) (string, error) {
	candidates := []string{}
	if strings.TrimSpace(resource.StorageURI) != "" {
		if parsed, err := url.Parse(strings.TrimSpace(resource.StorageURI)); err == nil && parsed.Scheme == "file" {
			candidates = append(candidates, parsed.Path)
		}
	}
	if storagePath := strings.TrimSpace(resource.StoragePath); storagePath != "" {
		if filepath.IsAbs(storagePath) {
			candidates = append(candidates, storagePath)
		} else {
			candidates = append(candidates, filepath.Join(root, storagePath))
		}
	}
	if strings.TrimSpace(resource.ResourceID) != "" && strings.TrimSpace(resource.OriginalName) != "" {
		candidates = append(candidates, filepath.Join(root, resource.ResourceID+"__"+safeOriginalFilename(resource.OriginalName)))
	}
	for _, candidate := range candidates {
		resolved := filepath.Clean(candidate)
		if !pathIsUnderRoot(root, resolved) {
			continue
		}
		return resolved, nil
	}
	return "", store.ErrNotFound
}

func resourceNeedsPreview(resource domain.ResourceRecord) bool {
	contentType := strings.ToLower(strings.TrimSpace(resource.ContentType))
	if strings.HasPrefix(contentType, "image/") {
		return true
	}
	kind := strings.ToLower(strings.TrimSpace(resource.ResourceKind))
	if kind == "image" {
		return true
	}
	name := strings.ToLower(strings.TrimSpace(resource.OriginalName))
	return strings.HasSuffix(name, ".png") ||
		strings.HasSuffix(name, ".jpg") ||
		strings.HasSuffix(name, ".jpeg") ||
		strings.HasSuffix(name, ".gif") ||
		strings.HasSuffix(name, ".tif") ||
		strings.HasSuffix(name, ".tiff")
}

func validatePreviewSource(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	if _, _, err := image.DecodeConfig(file); err != nil {
		return fmt.Errorf("image preview source could not be decoded: %w", err)
	}
	return nil
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
	metadata := readUploadMetadata(root, fileID)
	sourceType := strings.TrimSpace(metadata.SourceType)
	if sourceType == "" {
		sourceType = "upload"
	}
	return resourceRecord{
		FileID:        fileID,
		OriginalName:  originalName,
		ContentType:   contentType,
		SizeBytes:     info.Size(),
		SHA256:        sha,
		CreatedAt:     info.ModTime().UTC().Format(time.RFC3339Nano),
		SourceType:    sourceType,
		ResourceKind:  resourceKindForContent(originalName, contentType),
		SourceURI:     strings.TrimSpace(metadata.SourceURI),
		ProjectID:     strings.TrimSpace(metadata.ProjectID),
		HasThumbnail:  strings.HasPrefix(contentType, "image/"),
		ThumbnailURL:  previewURL,
		PreviewURL:    previewURL,
		CacheReady:    true,
		StagedLocally: true,
		Principal:     metadata.Principal,
	}, nil
}

type uploadImageDescriptor struct {
	Width        int
	Height       int
	Depth        int
	TimeCount    int
	ChannelCount int
	DimsOrder    string
	ArrayShape   []int
	ArrayDType   string
	Warnings     []string
	OME          *omeTIFFMetadata
}

func uploadImageDimensions(path string) (int, int, []string) {
	descriptor := uploadImageDescriptorForPath(path, "")
	return descriptor.Width, descriptor.Height, descriptor.Warnings
}

type omeTIFFMetadata struct {
	DimensionOrder  string
	SizeT           int
	SizeC           int
	SizeZ           int
	SizeY           int
	SizeX           int
	PixelType       string
	SignificantBits int
	PhysicalSizeX   float64
	PhysicalSizeY   float64
	PhysicalSizeZ   float64
	PhysicalUnitX   string
	PhysicalUnitY   string
	PhysicalUnitZ   string
	SceneName       string
	SceneCount      int
	ChannelNames    []string
	ChannelColors   []string
	Channels        []omeChannelMetadata
	TiffData        map[omePlaneKey]int
}

type omeChannelMetadata struct {
	Name             string
	Fluor            string
	Color            string
	IlluminationType string
	AcquisitionMode  string
}

type omePlaneKey struct {
	T int
	Z int
	C int
}

type omeXMLDocument struct {
	Images []omeXMLImage `xml:"Image"`
}

type omeXMLImage struct {
	ID              string       `xml:"ID,attr"`
	Name            string       `xml:"Name,attr"`
	AcquisitionDate string       `xml:"AcquisitionDate"`
	Pixels          omeXMLPixels `xml:"Pixels"`
}

type omeXMLPixels struct {
	DimensionOrder    string           `xml:"DimensionOrder,attr"`
	Type              string           `xml:"Type,attr"`
	SignificantBits   int              `xml:"SignificantBits,attr"`
	SizeT             int              `xml:"SizeT,attr"`
	SizeC             int              `xml:"SizeC,attr"`
	SizeZ             int              `xml:"SizeZ,attr"`
	SizeY             int              `xml:"SizeY,attr"`
	SizeX             int              `xml:"SizeX,attr"`
	PhysicalSizeX     float64          `xml:"PhysicalSizeX,attr"`
	PhysicalSizeY     float64          `xml:"PhysicalSizeY,attr"`
	PhysicalSizeZ     float64          `xml:"PhysicalSizeZ,attr"`
	PhysicalSizeXUnit string           `xml:"PhysicalSizeXUnit,attr"`
	PhysicalSizeYUnit string           `xml:"PhysicalSizeYUnit,attr"`
	PhysicalSizeZUnit string           `xml:"PhysicalSizeZUnit,attr"`
	Channels          []omeXMLChannel  `xml:"Channel"`
	TiffData          []omeXMLTiffData `xml:"TiffData"`
}

type omeXMLChannel struct {
	Name             string `xml:"Name,attr"`
	Fluor            string `xml:"Fluor,attr"`
	Color            string `xml:"Color,attr"`
	IlluminationType string `xml:"IlluminationType,attr"`
	AcquisitionMode  string `xml:"AcquisitionMode,attr"`
}

type omeXMLTiffData struct {
	IFD        int `xml:"IFD,attr"`
	FirstT     int `xml:"FirstT,attr"`
	FirstZ     int `xml:"FirstZ,attr"`
	FirstC     int `xml:"FirstC,attr"`
	PlaneCount int `xml:"PlaneCount,attr"`
}

func uploadImageDescriptorForPath(path string, contentType string) uploadImageDescriptor {
	file, err := os.Open(path)
	if err != nil {
		return uploadImageDescriptorFallback(contentType, "image metadata could not be opened")
	}
	defer func() {
		_ = file.Close()
	}()
	config, _, err := image.DecodeConfig(file)
	if err != nil {
		return uploadImageDescriptorFallback(contentType, "image dimensions could not be decoded")
	}
	width := config.Width
	if width < 1 {
		width = 1
	}
	height := config.Height
	if height < 1 {
		height = 1
	}
	channelCount, dtype := uploadImageProfileFromConfig(config, contentType)
	dimsOrder, shape := uploadArrayLayout(height, width, channelCount)
	if omeMeta, err := omeTIFFMetadataForPath(path); err == nil && omeMeta != nil {
		return uploadImageDescriptorFromOME(config, omeMeta)
	}
	return uploadImageDescriptor{
		Width:        width,
		Height:       height,
		Depth:        1,
		TimeCount:    1,
		ChannelCount: channelCount,
		DimsOrder:    dimsOrder,
		ArrayShape:   shape,
		ArrayDType:   dtype,
		Warnings:     []string{},
	}
}

func uploadImageDescriptorFallback(contentType string, warning string) uploadImageDescriptor {
	channelCount := uploadChannelCount(contentType)
	dimsOrder, shape := uploadArrayLayout(1, 1, channelCount)
	return uploadImageDescriptor{
		Width:        1,
		Height:       1,
		Depth:        1,
		TimeCount:    1,
		ChannelCount: channelCount,
		DimsOrder:    dimsOrder,
		ArrayShape:   shape,
		ArrayDType:   "unknown",
		Warnings:     []string{warning},
	}
}

func uploadImageDescriptorFromOME(config image.Config, meta *omeTIFFMetadata) uploadImageDescriptor {
	width := meta.SizeX
	if width < 1 {
		width = config.Width
	}
	if width < 1 {
		width = 1
	}
	height := meta.SizeY
	if height < 1 {
		height = config.Height
	}
	if height < 1 {
		height = 1
	}
	depth := meta.SizeZ
	if depth < 1 {
		depth = 1
	}
	timeCount := meta.SizeT
	if timeCount < 1 {
		timeCount = 1
	}
	channelCount := meta.SizeC
	if channelCount < 1 {
		channelCount = 1
	}
	dimsOrder := omeArrayDimsOrder(meta)
	return uploadImageDescriptor{
		Width:        width,
		Height:       height,
		Depth:        depth,
		TimeCount:    timeCount,
		ChannelCount: channelCount,
		DimsOrder:    dimsOrder,
		ArrayShape:   omeArrayShape(meta, dimsOrder),
		ArrayDType:   omePixelType(meta.PixelType),
		Warnings:     []string{},
		OME:          meta,
	}
}

func uploadImageProfileFromConfig(config image.Config, contentType string) (int, string) {
	channelCount := uploadChannelCount(contentType)
	arrayDType := "uint8"
	if config.ColorModel == nil {
		return channelCount, arrayDType
	}
	sample := config.ColorModel.Convert(color.RGBA{R: 17, G: 31, B: 47, A: 255})
	switch sample.(type) {
	case color.Gray:
		return 1, "uint8"
	case color.Gray16:
		return 1, "uint16"
	case color.Alpha:
		return 1, "uint8"
	case color.Alpha16:
		return 1, "uint16"
	case color.RGBA64, color.NRGBA64:
		if channelCount < 3 {
			channelCount = 3
		}
		return channelCount, "uint16"
	case color.CMYK:
		return 4, "uint8"
	case color.YCbCr:
		return 3, "uint8"
	default:
		return channelCount, arrayDType
	}
}

func uploadArrayLayout(height int, width int, channelCount int) (string, []int) {
	if height < 1 {
		height = 1
	}
	if width < 1 {
		width = 1
	}
	if channelCount <= 1 {
		return "YX", []int{height, width}
	}
	return "YXC", []int{height, width, channelCount}
}

func omeTIFFMetadataForPath(path string) (*omeTIFFMetadata, error) {
	description, err := tiffImageDescription(path)
	if err != nil {
		return nil, err
	}
	if !strings.Contains(description, "<OME") || !strings.Contains(description, "<Pixels") {
		return nil, errors.New("OME metadata is not present")
	}
	var document omeXMLDocument
	if err := xml.Unmarshal([]byte(description), &document); err != nil {
		return nil, fmt.Errorf("OME metadata could not be parsed: %w", err)
	}
	if len(document.Images) == 0 {
		return nil, errors.New("OME metadata has no images")
	}
	imageMeta := document.Images[0]
	pixels := imageMeta.Pixels
	if pixels.SizeX <= 0 || pixels.SizeY <= 0 {
		return nil, errors.New("OME metadata has invalid pixel dimensions")
	}
	meta := &omeTIFFMetadata{
		DimensionOrder:  strings.ToUpper(strings.TrimSpace(pixels.DimensionOrder)),
		SizeT:           positiveIntOr(pixels.SizeT, 1),
		SizeC:           positiveIntOr(pixels.SizeC, 1),
		SizeZ:           positiveIntOr(pixels.SizeZ, 1),
		SizeY:           positiveIntOr(pixels.SizeY, 1),
		SizeX:           positiveIntOr(pixels.SizeX, 1),
		PixelType:       strings.TrimSpace(pixels.Type),
		SignificantBits: pixels.SignificantBits,
		PhysicalSizeX:   pixels.PhysicalSizeX,
		PhysicalSizeY:   pixels.PhysicalSizeY,
		PhysicalSizeZ:   pixels.PhysicalSizeZ,
		PhysicalUnitX:   strings.TrimSpace(pixels.PhysicalSizeXUnit),
		PhysicalUnitY:   strings.TrimSpace(pixels.PhysicalSizeYUnit),
		PhysicalUnitZ:   strings.TrimSpace(pixels.PhysicalSizeZUnit),
		SceneName:       strings.TrimSpace(imageMeta.Name),
		SceneCount:      len(document.Images),
		TiffData:        map[omePlaneKey]int{},
	}
	if meta.DimensionOrder == "" {
		meta.DimensionOrder = "XYCZT"
	}
	for index, channel := range pixels.Channels {
		name := strings.TrimSpace(channel.Name)
		if name == "" {
			name = strings.TrimSpace(channel.Fluor)
		}
		if name == "" {
			name = fmt.Sprintf("Channel %d", index+1)
		}
		colorHex := omeColorToHex(channel.Color, index)
		meta.ChannelNames = append(meta.ChannelNames, name)
		meta.ChannelColors = append(meta.ChannelColors, colorHex)
		meta.Channels = append(meta.Channels, omeChannelMetadata{
			Name:             name,
			Fluor:            strings.TrimSpace(channel.Fluor),
			Color:            colorHex,
			IlluminationType: strings.TrimSpace(channel.IlluminationType),
			AcquisitionMode:  strings.TrimSpace(channel.AcquisitionMode),
		})
	}
	for len(meta.ChannelNames) < meta.SizeC {
		index := len(meta.ChannelNames)
		meta.ChannelNames = append(meta.ChannelNames, fmt.Sprintf("Channel %d", index+1))
		meta.ChannelColors = append(meta.ChannelColors, omeColorToHex("", index))
		meta.Channels = append(meta.Channels, omeChannelMetadata{
			Name:  fmt.Sprintf("Channel %d", index+1),
			Color: omeColorToHex("", index),
		})
	}
	for _, entry := range pixels.TiffData {
		planeCount := positiveIntOr(entry.PlaneCount, 1)
		for offset := 0; offset < planeCount; offset++ {
			t, z, c := omePlaneCoordinatesForOffset(meta, entry.FirstT, entry.FirstZ, entry.FirstC, offset)
			if t >= 0 && t < meta.SizeT && z >= 0 && z < meta.SizeZ && c >= 0 && c < meta.SizeC {
				meta.TiffData[omePlaneKey{T: t, Z: z, C: c}] = entry.IFD + offset
			}
		}
	}
	return meta, nil
}

func positiveIntOr(value int, fallback int) int {
	if value > 0 {
		return value
	}
	if fallback > 0 {
		return fallback
	}
	return 1
}

func omePixelType(value string) string {
	normalized := strings.ToLower(strings.TrimSpace(value))
	switch normalized {
	case "uint8", "uint16", "uint32", "int8", "int16", "int32", "float", "float32", "double", "float64":
		if normalized == "float" {
			return "float32"
		}
		if normalized == "double" {
			return "float64"
		}
		return normalized
	default:
		return "unknown"
	}
}

func omeArrayDimsOrder(meta *omeTIFFMetadata) string {
	if meta == nil {
		return "YX"
	}
	order := strings.ToUpper(strings.TrimSpace(meta.DimensionOrder))
	if order == "" {
		order = "XYCZT"
	}
	var axes []string
	for index := len(order) - 1; index >= 0; index-- {
		axis := order[index]
		if axis == 'X' || axis == 'Y' {
			continue
		}
		if omeAxisSize(meta, axis) > 1 {
			axes = append(axes, string(axis))
		}
	}
	axes = append(axes, "Y", "X")
	return strings.Join(axes, "")
}

func omeArrayShape(meta *omeTIFFMetadata, dimsOrder string) []int {
	if meta == nil {
		return []int{1, 1}
	}
	shape := make([]int, 0, len(dimsOrder))
	for _, axis := range strings.ToUpper(dimsOrder) {
		shape = append(shape, omeAxisSize(meta, byte(axis)))
	}
	return shape
}

func omeAxisSize(meta *omeTIFFMetadata, axis byte) int {
	if meta == nil {
		return 1
	}
	switch axis {
	case 'T':
		return positiveIntOr(meta.SizeT, 1)
	case 'C':
		return positiveIntOr(meta.SizeC, 1)
	case 'Z':
		return positiveIntOr(meta.SizeZ, 1)
	case 'Y':
		return positiveIntOr(meta.SizeY, 1)
	case 'X':
		return positiveIntOr(meta.SizeX, 1)
	default:
		return 1
	}
}

func omePlaneCoordinatesForOffset(meta *omeTIFFMetadata, firstT int, firstZ int, firstC int, offset int) (int, int, int) {
	t, z, c := firstT, firstZ, firstC
	order := strings.ToUpper(strings.TrimSpace(meta.DimensionOrder))
	if order == "" {
		order = "XYCZT"
	}
	for _, axis := range order {
		if axis == 'X' || axis == 'Y' {
			continue
		}
		switch axis {
		case 'C':
			c += offset
			offset = c / meta.SizeC
			c %= meta.SizeC
		case 'Z':
			z += offset
			offset = z / meta.SizeZ
			z %= meta.SizeZ
		case 'T':
			t += offset
			offset = t / meta.SizeT
			t %= meta.SizeT
		}
		if offset == 0 {
			break
		}
	}
	return t, z, c
}

func omeColorToHex(raw string, index int) string {
	if value, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 32); err == nil {
		bits := uint32(int32(value))
		return fmt.Sprintf("#%06x", bits&0x00ffffff)
	}
	palette := []string{"#ffffff", "#00ff00", "#ff00ff", "#00ffff", "#ffcc00", "#ff4d4d", "#8fc8ff"}
	return palette[index%len(palette)]
}

type tiffIFDEntry struct {
	Tag        uint16
	DataType   uint16
	Count      uint32
	ValueBytes [4]byte
}

func tiffImageDescription(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer func() {
		_ = file.Close()
	}()
	order, firstIFDOffset, err := readClassicTIFFHeader(file)
	if err != nil {
		return "", err
	}
	entries, _, err := readClassicTIFFIFD(file, order, firstIFDOffset)
	if err != nil {
		return "", err
	}
	for _, entry := range entries {
		if entry.Tag != 270 {
			continue
		}
		raw, err := readTIFFEntryBytes(file, order, entry)
		if err != nil {
			return "", err
		}
		return strings.TrimRight(string(raw), "\x00"), nil
	}
	return "", errors.New("TIFF ImageDescription is not present")
}

func uploadEXIFMetadataForPath(path string, contentType string) domain.JSONMap {
	normalizedType := strings.ToLower(strings.TrimSpace(contentType))
	lowerPath := strings.ToLower(strings.TrimSpace(path))
	if normalizedType != "image/jpeg" && normalizedType != "image/jpg" && !strings.HasSuffix(lowerPath, ".jpg") && !strings.HasSuffix(lowerPath, ".jpeg") {
		return nil
	}
	file, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer func() {
		_ = file.Close()
	}()
	data, err := io.ReadAll(io.LimitReader(file, 2<<20))
	if err != nil {
		return nil
	}
	return parseJPEGEXIFMetadata(data)
}

func parseJPEGEXIFMetadata(data []byte) domain.JSONMap {
	tiffData, ok := jpegEXIFTIFFPayload(data)
	if !ok {
		return nil
	}
	metadata := parseEXIFTIFFMetadata(tiffData)
	if len(metadata) == 0 {
		return nil
	}
	return metadata
}

func jpegEXIFTIFFPayload(data []byte) ([]byte, bool) {
	if len(data) < 4 || data[0] != 0xff || data[1] != 0xd8 {
		return nil, false
	}
	offset := 2
	for offset+4 <= len(data) {
		for offset < len(data) && data[offset] == 0xff {
			offset++
		}
		if offset >= len(data) {
			return nil, false
		}
		marker := data[offset]
		offset++
		if marker == 0xd9 || marker == 0xda {
			return nil, false
		}
		if offset+2 > len(data) {
			return nil, false
		}
		segmentLength := int(binary.BigEndian.Uint16(data[offset : offset+2]))
		if segmentLength < 2 || offset+segmentLength > len(data) {
			return nil, false
		}
		payload := data[offset+2 : offset+segmentLength]
		if marker == 0xe1 && len(payload) > 6 && string(payload[:6]) == "Exif\x00\x00" {
			return payload[6:], true
		}
		offset += segmentLength
	}
	return nil, false
}

func parseEXIFTIFFMetadata(data []byte) domain.JSONMap {
	if len(data) < 8 {
		return nil
	}
	var order binary.ByteOrder
	switch string(data[:2]) {
	case "II":
		order = binary.LittleEndian
	case "MM":
		order = binary.BigEndian
	default:
		return nil
	}
	if order.Uint16(data[2:4]) != 42 {
		return nil
	}
	ifd0Offset := int(order.Uint32(data[4:8]))
	entries, ok := readEXIFIFDEntries(data, order, ifd0Offset)
	if !ok {
		return nil
	}
	metadata := domain.JSONMap{}
	var exifOffset int
	for _, entry := range entries {
		switch entry.Tag {
		case 0x010f:
			if value, ok := exifASCIIValue(data, order, entry); ok {
				metadata["camera_make"] = value
			}
		case 0x0110:
			if value, ok := exifASCIIValue(data, order, entry); ok {
				metadata["camera_model"] = value
			}
		case 0x0112:
			if value, ok := exifUnsignedValue(data, order, entry); ok {
				metadata["orientation"] = float64(value)
			}
		case 0x0132:
			if value, ok := exifASCIIValue(data, order, entry); ok {
				metadata["datetime"] = value
			}
		case 0x8769:
			if value, ok := exifUnsignedValue(data, order, entry); ok {
				exifOffset = int(value)
			}
		}
	}
	if exifOffset > 0 {
		if exifEntries, ok := readEXIFIFDEntries(data, order, exifOffset); ok {
			for _, entry := range exifEntries {
				switch entry.Tag {
				case 0x829a:
					if value, ok := exifRationalValue(data, order, entry); ok {
						metadata["exposure_time_seconds"] = value
					}
				case 0x829d:
					if value, ok := exifRationalValue(data, order, entry); ok {
						metadata["f_number"] = value
					}
				case 0x8827:
					if value, ok := exifUnsignedValue(data, order, entry); ok {
						metadata["iso"] = float64(value)
					}
				case 0x9003:
					if value, ok := exifASCIIValue(data, order, entry); ok {
						metadata["datetime_original"] = value
					}
				case 0x920a:
					if value, ok := exifRationalValue(data, order, entry); ok {
						metadata["focal_length_mm"] = value
					}
				case 0xa434:
					if value, ok := exifASCIIValue(data, order, entry); ok {
						metadata["lens_model"] = value
					}
				}
			}
		}
	}
	return metadata
}

func readEXIFIFDEntries(data []byte, order binary.ByteOrder, offset int) ([]tiffIFDEntry, bool) {
	if offset < 0 || offset+2 > len(data) {
		return nil, false
	}
	count := int(order.Uint16(data[offset : offset+2]))
	entriesOffset := offset + 2
	if count < 0 || entriesOffset+count*12 > len(data) {
		return nil, false
	}
	entries := make([]tiffIFDEntry, 0, count)
	for index := 0; index < count; index++ {
		entryOffset := entriesOffset + index*12
		entry := tiffIFDEntry{
			Tag:      order.Uint16(data[entryOffset : entryOffset+2]),
			DataType: order.Uint16(data[entryOffset+2 : entryOffset+4]),
			Count:    order.Uint32(data[entryOffset+4 : entryOffset+8]),
		}
		copy(entry.ValueBytes[:], data[entryOffset+8:entryOffset+12])
		entries = append(entries, entry)
	}
	return entries, true
}

func exifEntryBytes(data []byte, order binary.ByteOrder, entry tiffIFDEntry) ([]byte, bool) {
	typeSize := exifTypeSize(entry.DataType)
	if typeSize <= 0 {
		return nil, false
	}
	total := int(entry.Count) * typeSize
	if total < 0 {
		return nil, false
	}
	if total <= 4 {
		return entry.ValueBytes[:total], true
	}
	offset := int(order.Uint32(entry.ValueBytes[:]))
	if offset < 0 || offset+total > len(data) {
		return nil, false
	}
	return data[offset : offset+total], true
}

func exifTypeSize(dataType uint16) int {
	switch dataType {
	case 1, 2, 7:
		return 1
	case 3:
		return 2
	case 4, 9:
		return 4
	case 5, 10:
		return 8
	default:
		return 0
	}
}

func exifASCIIValue(data []byte, order binary.ByteOrder, entry tiffIFDEntry) (string, bool) {
	if entry.DataType != 2 {
		return "", false
	}
	raw, ok := exifEntryBytes(data, order, entry)
	if !ok {
		return "", false
	}
	value := strings.TrimRight(string(raw), "\x00")
	value = strings.TrimSpace(value)
	return value, value != ""
}

func exifUnsignedValue(data []byte, order binary.ByteOrder, entry tiffIFDEntry) (uint32, bool) {
	raw, ok := exifEntryBytes(data, order, entry)
	if !ok || len(raw) == 0 {
		return 0, false
	}
	switch entry.DataType {
	case 1, 7:
		return uint32(raw[0]), true
	case 3:
		if len(raw) < 2 {
			return 0, false
		}
		return uint32(order.Uint16(raw[:2])), true
	case 4:
		if len(raw) < 4 {
			return 0, false
		}
		return order.Uint32(raw[:4]), true
	default:
		return 0, false
	}
}

func exifRationalValue(data []byte, order binary.ByteOrder, entry tiffIFDEntry) (float64, bool) {
	raw, ok := exifEntryBytes(data, order, entry)
	if !ok || len(raw) < 8 {
		return 0, false
	}
	numerator := float64(order.Uint32(raw[:4]))
	denominator := float64(order.Uint32(raw[4:8]))
	if denominator == 0 {
		return 0, false
	}
	value := numerator / denominator
	return value, numberIsFinite(value)
}

func readClassicTIFFHeader(file *os.File) (binary.ByteOrder, uint32, error) {
	var header [8]byte
	if _, err := file.ReadAt(header[:], 0); err != nil {
		return binary.LittleEndian, 0, err
	}
	var order binary.ByteOrder
	switch string(header[0:2]) {
	case "II":
		order = binary.LittleEndian
	case "MM":
		order = binary.BigEndian
	default:
		return binary.LittleEndian, 0, errors.New("not a TIFF file")
	}
	if magic := order.Uint16(header[2:4]); magic != 42 {
		if magic == 43 {
			return order, 0, errors.New("BigTIFF is not supported by the native preview reader")
		}
		return order, 0, fmt.Errorf("unsupported TIFF magic %d", magic)
	}
	return order, order.Uint32(header[4:8]), nil
}

func readClassicTIFFIFD(file *os.File, order binary.ByteOrder, offset uint32) ([]tiffIFDEntry, uint32, error) {
	if offset == 0 {
		return nil, 0, errors.New("TIFF IFD offset is empty")
	}
	var countBytes [2]byte
	if _, err := file.ReadAt(countBytes[:], int64(offset)); err != nil {
		return nil, 0, err
	}
	count := int(order.Uint16(countBytes[:]))
	if count < 0 || count > 4096 {
		return nil, 0, fmt.Errorf("TIFF IFD entry count %d is out of range", count)
	}
	raw := make([]byte, count*12+4)
	if _, err := file.ReadAt(raw, int64(offset)+2); err != nil {
		return nil, 0, err
	}
	entries := make([]tiffIFDEntry, 0, count)
	for index := 0; index < count; index++ {
		entryBytes := raw[index*12 : index*12+12]
		var valueBytes [4]byte
		copy(valueBytes[:], entryBytes[8:12])
		entries = append(entries, tiffIFDEntry{
			Tag:        order.Uint16(entryBytes[0:2]),
			DataType:   order.Uint16(entryBytes[2:4]),
			Count:      order.Uint32(entryBytes[4:8]),
			ValueBytes: valueBytes,
		})
	}
	nextOffset := order.Uint32(raw[count*12 : count*12+4])
	return entries, nextOffset, nil
}

func readTIFFEntryBytes(file *os.File, order binary.ByteOrder, entry tiffIFDEntry) ([]byte, error) {
	typeSize := tiffDataTypeSize(entry.DataType)
	if typeSize <= 0 {
		return nil, fmt.Errorf("unsupported TIFF datatype %d", entry.DataType)
	}
	length64 := uint64(typeSize) * uint64(entry.Count)
	const maxTIFFTagByteLength = 128 * 1024 * 1024
	if length64 > maxTIFFTagByteLength {
		return nil, fmt.Errorf("TIFF tag %d is too large", entry.Tag)
	}
	length := int(length64)
	if length <= 4 {
		return append([]byte(nil), entry.ValueBytes[:length]...), nil
	}
	valueOffset := order.Uint32(entry.ValueBytes[:])
	raw := make([]byte, length)
	if _, err := file.ReadAt(raw, int64(valueOffset)); err != nil {
		return nil, err
	}
	return raw, nil
}

func tiffDataTypeSize(dataType uint16) int {
	switch dataType {
	case 1, 2, 6, 7:
		return 1
	case 3, 8:
		return 2
	case 4, 9, 11:
		return 4
	case 5, 10, 12:
		return 8
	default:
		return 0
	}
}

func readClassicTIFFIFDOffsets(file *os.File, order binary.ByteOrder, firstOffset uint32, limit int) ([]uint32, error) {
	offsets := []uint32{}
	seen := map[uint32]bool{}
	offset := firstOffset
	for offset != 0 && (limit <= 0 || len(offsets) < limit) {
		if seen[offset] {
			return offsets, errors.New("TIFF IFD chain contains a cycle")
		}
		seen[offset] = true
		offsets = append(offsets, offset)
		_, nextOffset, err := readClassicTIFFIFD(file, order, offset)
		if err != nil {
			return offsets, err
		}
		offset = nextOffset
	}
	return offsets, nil
}

type redirectedTIFFReader struct {
	base           io.ReaderAt
	reader         *io.SectionReader
	order          binary.ByteOrder
	firstIFDOffset uint32
}

func (reader *redirectedTIFFReader) Read(p []byte) (int, error) {
	return reader.reader.Read(p)
}

func (reader *redirectedTIFFReader) ReadAt(p []byte, off int64) (int, error) {
	n, err := reader.base.ReadAt(p, off)
	for index := 0; index < n; index++ {
		absolute := off + int64(index)
		if absolute < 4 || absolute >= 8 {
			continue
		}
		var encoded [4]byte
		reader.order.PutUint32(encoded[:], reader.firstIFDOffset)
		p[index] = encoded[int(absolute-4)]
	}
	return n, err
}

func decodeTIFFImageAtIFD(path string, ifdIndex int) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = file.Close()
	}()
	stat, err := file.Stat()
	if err != nil {
		return nil, err
	}
	order, firstOffset, err := readClassicTIFFHeader(file)
	if err != nil {
		return nil, err
	}
	offsets, err := readClassicTIFFIFDOffsets(file, order, firstOffset, ifdIndex+1)
	if err != nil {
		return nil, err
	}
	if ifdIndex < 0 || ifdIndex >= len(offsets) {
		return nil, fmt.Errorf("TIFF IFD %d is unavailable", ifdIndex)
	}
	reader := &redirectedTIFFReader{
		base:           file,
		reader:         io.NewSectionReader(file, 0, stat.Size()),
		order:          order,
		firstIFDOffset: offsets[ifdIndex],
	}
	return tiff.Decode(reader)
}

func selectOMEPlane(meta *omeTIFFMetadata, transform uploadPreviewTransform, request ...*http.Request) omePlaneKey {
	selection := omePlaneKey{
		T: 0,
		Z: positiveIntOr(meta.SizeZ, 1) / 2,
		C: omeDefaultChannelIndex(meta),
	}
	if len(transform.Channels) > 0 {
		selection.C = transform.Channels[0]
	}
	if len(request) > 0 && request[0] != nil {
		query := request[0].URL.Query()
		selection.T = parseNonNegativeInt(query.Get("t"), selection.T)
		selection.Z = parseNonNegativeInt(query.Get("z"), selection.Z)
		if raw := strings.TrimSpace(query.Get("channel")); raw != "" {
			selection.C = parseNonNegativeInt(raw, selection.C)
		} else if raw := strings.TrimSpace(query.Get("c")); raw != "" {
			selection.C = parseNonNegativeInt(raw, selection.C)
		}
	}
	selection.T = clampInt(selection.T, 0, positiveIntOr(meta.SizeT, 1)-1)
	selection.Z = clampInt(selection.Z, 0, positiveIntOr(meta.SizeZ, 1)-1)
	selection.C = clampInt(selection.C, 0, positiveIntOr(meta.SizeC, 1)-1)
	return selection
}

func omeIFDForSelection(meta *omeTIFFMetadata, selection omePlaneKey) int {
	if meta == nil {
		return 0
	}
	if meta.TiffData != nil {
		if ifdIndex, ok := meta.TiffData[selection]; ok && ifdIndex >= 0 {
			return ifdIndex
		}
	}
	return omeLinearIFD(meta, selection)
}

func omeLinearIFD(meta *omeTIFFMetadata, selection omePlaneKey) int {
	order := strings.ToUpper(strings.TrimSpace(meta.DimensionOrder))
	if order == "" {
		order = "XYCZT"
	}
	stride := 1
	ifdIndex := 0
	for _, axis := range order {
		if axis == 'X' || axis == 'Y' {
			continue
		}
		switch axis {
		case 'C':
			ifdIndex += selection.C * stride
			stride *= positiveIntOr(meta.SizeC, 1)
		case 'Z':
			ifdIndex += selection.Z * stride
			stride *= positiveIntOr(meta.SizeZ, 1)
		case 'T':
			ifdIndex += selection.T * stride
			stride *= positiveIntOr(meta.SizeT, 1)
		}
	}
	return ifdIndex
}

func omeDefaultChannelIndex(meta *omeTIFFMetadata) int {
	if meta == nil || meta.SizeC <= 1 {
		return 0
	}
	for index, channel := range meta.Channels {
		haystack := strings.ToLower(strings.Join([]string{
			channel.Name,
			channel.Fluor,
			channel.IlluminationType,
			channel.AcquisitionMode,
		}, " "))
		if strings.Contains(haystack, "bright") || strings.Contains(haystack, "transmitted") {
			return clampInt(index, 0, meta.SizeC-1)
		}
	}
	return 0
}

func omeChannelColorStrings(meta *omeTIFFMetadata) []string {
	if meta == nil || meta.SizeC <= 0 {
		return []string{"#ffffff"}
	}
	colors := make([]string, meta.SizeC)
	for index := range colors {
		if index < len(meta.ChannelColors) && strings.TrimSpace(meta.ChannelColors[index]) != "" {
			colors[index] = meta.ChannelColors[index]
		} else {
			colors[index] = omeColorToHex("", index)
		}
	}
	return colors
}

func omeChannelColorPayload(meta *omeTIFFMetadata) []map[string]any {
	colors := omeChannelColorStrings(meta)
	payload := make([]map[string]any, len(colors))
	for index, hex := range colors {
		payload[index] = map[string]any{
			"index": index,
			"hex":   hex,
			"rgb":   hexToRGBTriplet(hex),
		}
	}
	return payload
}

func hexToRGBTriplet(hex string) []int {
	normalized := strings.TrimPrefix(strings.TrimSpace(hex), "#")
	if len(normalized) != 6 {
		return []int{255, 255, 255}
	}
	value, err := strconv.ParseUint(normalized, 16, 32)
	if err != nil {
		return []int{255, 255, 255}
	}
	return []int{int((value >> 16) & 0xff), int((value >> 8) & 0xff), int(value & 0xff)}
}

func omePhysicalSpacing(meta *omeTIFFMetadata) map[string]float64 {
	return map[string]float64{
		"x": positiveFloatOr(meta.PhysicalSizeX, 1),
		"y": positiveFloatOr(meta.PhysicalSizeY, 1),
		"z": positiveFloatOr(meta.PhysicalSizeZ, 1),
	}
}

func omeHasPhysicalSpacing(meta *omeTIFFMetadata) bool {
	if meta == nil {
		return false
	}
	return meta.PhysicalSizeX > 0 || meta.PhysicalSizeY > 0 || meta.PhysicalSizeZ > 0
}

func positiveFloatOr(value float64, fallback float64) float64 {
	if numberIsFinite(value) && value > 0 {
		return value
	}
	if numberIsFinite(fallback) && fallback > 0 {
		return fallback
	}
	return 1
}

func nonEmptyString(value string, fallback string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed != "" {
		return trimmed
	}
	return fallback
}

func omePixelDepth(meta *omeTIFFMetadata) int {
	if meta != nil && meta.SignificantBits > 0 {
		return meta.SignificantBits
	}
	pixelType := ""
	if meta != nil {
		pixelType = meta.PixelType
	}
	switch omePixelType(pixelType) {
	case "uint8", "int8":
		return 8
	case "uint16", "int16":
		return 16
	case "uint32", "int32", "float32":
		return 32
	case "float64":
		return 64
	default:
		return 16
	}
}

func omePixelFormat(meta *omeTIFFMetadata) string {
	pixelType := ""
	if meta != nil {
		pixelType = meta.PixelType
	}
	switch omePixelType(pixelType) {
	case "int8", "int16", "int32":
		return "s"
	case "float32", "float64":
		return "f"
	default:
		return "u"
	}
}

func clampInt(value int, minValue int, maxValue int) int {
	if maxValue < minValue {
		return minValue
	}
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

type uploadPreviewTransform struct {
	WindowMin    float64
	WindowMax    float64
	WindowActive bool
	// WindowIsPhysical marks the window as expressed in physical units (e.g.
	// Hounsfield from a "hounsfield:WC:WW" preset); it is converted back to stored
	// codes before comparison against raw voxel samples.
	WindowIsPhysical  bool
	Gamma             float64
	Negative          bool
	FullRange         bool
	Channels          []int
	ChannelsRequested bool
}

func serveUploadAsPNG(w http.ResponseWriter, path string, transform uploadPreviewTransform, request ...*http.Request) error {
	img, err := decodeUploadPreviewImage(path, transform, request...)
	if err != nil {
		return fmt.Errorf("image preview could not be decoded: %w", err)
	}
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "private, max-age=3600")
	return png.Encode(w, browserPreviewImage(img, transform))
}

func decodeUploadPreviewImage(path string, transform uploadPreviewTransform, request ...*http.Request) (image.Image, error) {
	if omeMeta, err := omeTIFFMetadataForPath(path); err == nil && omeMeta != nil {
		selection := selectOMEPlane(omeMeta, transform, request...)
		ifdIndex := omeIFDForSelection(omeMeta, selection)
		return decodeTIFFImageAtIFD(path, ifdIndex)
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = file.Close()
	}()
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	return img, nil
}

func decodeUploadImage(path string) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = file.Close()
	}()
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("image histogram could not be decoded: %w", err)
	}
	return img, nil
}

type niftiScalarVolume struct {
	Width         int
	Height        int
	Depth         int
	TimeCount     int
	TimeIndex     int
	ChannelCount  int
	ChannelIndex  int
	DType         string
	BytesPerVoxel int
	Data          []byte
	RawMin        float64
	RawMax        float64
	SpacingX      float64
	SpacingY      float64
	SpacingZ      float64
	// SclSlope/SclInter rescale stored codes to physical units (HU/SUV):
	// physical = SclSlope*code + SclInter. RawMin/RawMax stay in code space so
	// the raw 3D payload normalizes consistently; physical values are derived.
	SclSlope   float64
	SclInter   float64
	Affine     [12]float64
	AffineCode int
	SpaceUnit  string
	Warnings   []string
}

// physical converts a stored code value to physical units (Hounsfield/SUV/etc.).
func (v niftiScalarVolume) physical(code float64) float64 {
	slope := v.SclSlope
	if slope == 0 {
		slope = 1
	}
	return slope*code + v.SclInter
}

// codeFromPhysical inverts the rescale: physical -> stored code, so a physical
// (e.g. Hounsfield) window can be compared against raw code samples.
func (v niftiScalarVolume) codeFromPhysical(physical float64) float64 {
	slope := v.SclSlope
	if slope == 0 {
		slope = 1
	}
	return (physical - v.SclInter) / slope
}

func niftiScalarDisplayDefaults(volume niftiScalarVolume, channelColors []string) map[string]any {
	defaults := map[string]any{
		"enhancement":        "d",
		"negative":           false,
		"rotate":             0,
		"fusion_method":      "a",
		"channel_mode":       "single",
		"channels":           []int{0},
		"channel_colors":     channelColors,
		"scalar_colormap":    "grayscale",
		"time_index":         0,
		"z_index":            volume.Depth / 2,
		"volume_channel":     0,
		"volume_clip_min":    map[string]float64{"x": 0, "y": 0, "z": 0},
		"volume_clip_max":    map[string]float64{"x": 1, "y": 1, "z": 1},
		"volume_view_preset": "iso",
	}
	if niftiScalarRangeLooksCTLike(volume) {
		// Default head CT to the diagnostic brain window (WC 40 / WW 80 HU). The
		// previous 350/1800 (= window [-550, 1250] HU) is not a clinical window —
		// it washed brain soft tissue into a flat gray. Bone/soft-tissue/lung
		// windows are available as explicit presets.
		defaults["enhancement"] = "hounsfield:40.000:80.000"
		defaults["volume_signal_floor"] = 0.12
		defaults["volume_density"] = 1.75
		defaults["volume_lighting"] = true
		defaults["volume_lighting_strength"] = 0.72
		defaults["volume_camera_mode"] = "orthographic"
	}
	return defaults
}

// niftiScalarRangeLooksCTLike detects Hounsfield-scaled CT from the PHYSICAL
// (rescaled) intensity range, so a CT stored as unsigned codes with
// scl_inter=-1024 is still recognized. The window spans air (~-1000 HU) to
// soft-tissue/bone (>=300 HU).
func niftiScalarRangeLooksCTLike(volume niftiScalarVolume) bool {
	physMin := volume.physical(volume.RawMin)
	physMax := volume.physical(volume.RawMax)
	return physMin <= -300 && physMin >= -1100 && physMax >= 300
}

const (
	// A NIfTI-1 header is 348 bytes; voxel data starts at vox_offset, which the
	// single-file (n+1) format requires to be >= 352. Reading exactly this many
	// bytes is enough to parse every geometry field without touching voxels.
	niftiHeaderReadSize = 352
	// A single 3D scalar volume (one timepoint, one channel) larger than this is
	// rejected rather than risking an OOM from a malformed or hostile header. A
	// 91x109x91 fMRI timepoint is ~3.6 MB; even a 1024^3 float32 anatomical is
	// 4 GiB and is refused on purpose.
	niftiMaxSingleVolumeBytes = int64(1) << 31 // 2 GiB
)

// niftiGeometry is the dimensional shape parsed from a NIfTI-1 header. The 4th
// dimension is time and the 5th is per-voxel channels/components, per the
// NIfTI-1 spec — so a 1200-volume fMRI BOLD series is 1200 timepoints, not 1200
// "channels". Voxels are stored column-major (dim[1]/X fastest), which makes
// each 3D volume a single contiguous slab.
type niftiGeometry struct {
	order         binary.ByteOrder
	width         int
	height        int
	depth         int
	timeCount     int
	channelCount  int
	dtype         string
	bytesPerVoxel int
	voxOffset     int64
	spacingX      float64
	spacingY      float64
	spacingZ      float64
	// affine is the row-major 3x4 voxel->world (RAS+ mm) transform: rows
	// [0..3]=x, [4..7]=y, [8..11]=z; the implicit 4th row is [0 0 0 1]. affineCode
	// is the NIfTI method that produced it (3=sform, 2=qform, 0=pixdim-only).
	affine     [12]float64
	affineCode int
	// sclSlope/sclInter rescale stored codes to physical units (Hounsfield, SUV,
	// etc.): physical = sclSlope*code + sclInter. sclSlope==0 means no scaling.
	sclSlope  float64
	sclInter  float64
	spaceUnit string // "m" | "mm" | "um" | "" (from xyzt_units)
}

// volumeBytes is the byte length of one 3D scalar volume (one timepoint, one
// channel). int64 throughout so a 4 GB+ dataset can't overflow the arithmetic.
func (g niftiGeometry) volumeBytes() int64 {
	return int64(g.width) * int64(g.height) * int64(g.depth) * int64(g.bytesPerVoxel)
}

// volumeOffset is the byte offset of the (timeIndex, channelIndex) 3D volume
// within the file. Column-major layout means time varies before channel:
// flat plane index = timeIndex + timeCount*channelIndex.
func (g niftiGeometry) volumeOffset(timeIndex, channelIndex int) int64 {
	plane := int64(timeIndex) + int64(g.timeCount)*int64(channelIndex)
	return g.voxOffset + g.volumeBytes()*plane
}

func parseNiftiGeometry(header []byte) (niftiGeometry, error) {
	if len(header) < niftiHeaderReadSize {
		return niftiGeometry{}, errors.New("NIfTI file is too small")
	}
	order, version, err := niftiHeaderVersion(header)
	if err != nil {
		return niftiGeometry{}, err
	}
	if version == 2 {
		// NIfTI-2 is a distinct binary layout (64-bit dims/floats, magic at
		// offset 4); its parser produces the same niftiGeometry so the rest of
		// the pipeline is unchanged. See nifti2.go.
		return parseNifti2Geometry(order, header)
	}
	magic := string(header[344:348])
	if magic != "n+1\x00" && magic != "ni1\x00" {
		return niftiGeometry{}, fmt.Errorf("unsupported NIfTI magic %q", strings.TrimRight(magic, "\x00"))
	}
	dim0 := niftiInt16(order, header[40:42])
	if dim0 < 2 {
		return niftiGeometry{}, fmt.Errorf("unsupported NIfTI dimension count %d", dim0)
	}
	width := niftiDimension(order, header[42:44])
	height := niftiDimension(order, header[44:46])
	depth := 1
	if dim0 >= 3 {
		depth = niftiDimension(order, header[46:48])
	}
	timeCount := 1
	if dim0 >= 4 {
		timeCount = niftiDimension(order, header[48:50])
	}
	channelCount := 1
	if dim0 >= 5 {
		channelCount = niftiDimension(order, header[50:52])
	}
	if width <= 0 || height <= 0 || depth <= 0 {
		return niftiGeometry{}, fmt.Errorf("invalid NIfTI dimensions %dx%dx%d", width, height, depth)
	}
	if timeCount <= 0 {
		timeCount = 1
	}
	if channelCount <= 0 {
		channelCount = 1
	}
	datatype := niftiInt16(order, header[70:72])
	dtype, bytesPerVoxel, err := niftiScalarType(datatype)
	if err != nil {
		return niftiGeometry{}, err
	}
	voxOffset := int64(math.Round(float64(niftiFloat32(order, header[108:112]))))
	if voxOffset < niftiHeaderReadSize {
		voxOffset = niftiHeaderReadSize
	}
	spacingX := niftiSpacing(order, header[80:84])
	spacingY := niftiSpacing(order, header[84:88])
	spacingZ := niftiSpacing(order, header[88:92])
	affine, affineCode := niftiAffineFromHeader(order, header, spacingX, spacingY, spacingZ)
	sclSlope, sclInter := niftiRescaleFromHeader(order, header)
	return niftiGeometry{
		order:         order,
		width:         width,
		height:        height,
		depth:         depth,
		timeCount:     timeCount,
		channelCount:  channelCount,
		dtype:         dtype,
		bytesPerVoxel: bytesPerVoxel,
		voxOffset:     voxOffset,
		spacingX:      spacingX,
		spacingY:      spacingY,
		spacingZ:      spacingZ,
		affine:        affine,
		affineCode:    affineCode,
		sclSlope:      sclSlope,
		sclInter:      sclInter,
		spaceUnit:     niftiSpaceUnitFromHeader(header),
	}, nil
}

func readNiftiHeaderGeometry(path string) (niftiGeometry, error) {
	file, err := os.Open(path)
	if err != nil {
		return niftiGeometry{}, err
	}
	defer func() {
		_ = file.Close()
	}()
	var reader io.Reader = file
	if strings.HasSuffix(strings.ToLower(strings.TrimSpace(path)), ".gz") {
		gzipReader, err := gzip.NewReader(file)
		if err != nil {
			return niftiGeometry{}, err
		}
		defer func() {
			_ = gzipReader.Close()
		}()
		reader = gzipReader
	}
	header, _, err := readNiftiHeaderBytes(reader)
	if err != nil {
		return niftiGeometry{}, err
	}
	return parseNiftiGeometry(header)
}

func niftiGeometryDimsOrder(geom niftiGeometry) string {
	axes := make([]string, 0, 5)
	if geom.timeCount > 1 {
		axes = append(axes, "T")
	}
	if geom.channelCount > 1 {
		axes = append(axes, "C")
	}
	if geom.depth > 1 {
		axes = append(axes, "Z")
	}
	axes = append(axes, "Y", "X")
	return strings.Join(axes, "")
}

func niftiGeometryArrayShape(geom niftiGeometry, dimsOrder string) []any {
	shape := make([]any, 0, len(dimsOrder))
	for _, axis := range strings.ToUpper(dimsOrder) {
		switch axis {
		case 'T':
			shape = append(shape, float64(geom.timeCount))
		case 'C':
			shape = append(shape, float64(geom.channelCount))
		case 'Z':
			shape = append(shape, float64(geom.depth))
		case 'Y':
			shape = append(shape, float64(geom.height))
		case 'X':
			shape = append(shape, float64(geom.width))
		}
	}
	return shape
}

// niftiRescaleFromHeader reads scl_slope/scl_inter (header[112:120]). Per the
// NIfTI-1 spec a zero (or non-finite) slope means "no rescaling".
func niftiRescaleFromHeader(order binary.ByteOrder, header []byte) (float64, float64) {
	slope := float64(niftiFloat32(order, header[112:116]))
	inter := float64(niftiFloat32(order, header[116:120]))
	if !numberIsFinite(slope) || slope == 0 {
		return 1, 0
	}
	if !numberIsFinite(inter) {
		inter = 0
	}
	return slope, inter
}

// niftiSpaceUnitFromHeader decodes the spatial bits (0x07) of xyzt_units
// (header[123]): 1=meter, 2=millimeter, 3=micron.
func niftiSpaceUnitFromHeader(header []byte) string {
	if len(header) < 124 {
		return ""
	}
	switch header[123] & 0x07 {
	case 1:
		return "m"
	case 2:
		return "mm"
	case 3:
		return "um"
	default:
		return ""
	}
}

// niftiAffineFromHeader builds the voxel->RAS+ (mm) transform, preferring the
// sform (method 3, an explicit 3x4) and falling back to the qform quaternion
// (method 2) and finally to pixdim-only diagonal scaling (method 1, orientation
// unknown). Returns the row-major 3x4 matrix and the method code.
func niftiAffineFromHeader(order binary.ByteOrder, h []byte, sx, sy, sz float64) ([12]float64, int) {
	var affine [12]float64
	qformCode := int(niftiInt16(order, h[252:254]))
	sformCode := int(niftiInt16(order, h[254:256]))
	if sformCode > 0 {
		nonZero := false
		for c := 0; c < 4; c++ {
			affine[0*4+c] = float64(niftiFloat32(order, h[280+c*4:284+c*4]))
			affine[1*4+c] = float64(niftiFloat32(order, h[296+c*4:300+c*4]))
			affine[2*4+c] = float64(niftiFloat32(order, h[312+c*4:316+c*4]))
		}
		for i := 0; i < 9; i++ {
			if affine[(i/3)*4+(i%3)] != 0 {
				nonZero = true
				break
			}
		}
		if nonZero {
			return affine, 3
		}
		affine = [12]float64{}
	}
	if qformCode > 0 {
		b := float64(niftiFloat32(order, h[256:260]))
		c := float64(niftiFloat32(order, h[260:264]))
		d := float64(niftiFloat32(order, h[264:268]))
		a2 := 1.0 - (b*b + c*c + d*d)
		a := 0.0
		if a2 > 0 {
			a = math.Sqrt(a2)
		}
		qfac := float64(niftiFloat32(order, h[76:80]))
		if qfac >= 0 {
			qfac = 1
		} else {
			qfac = -1
		}
		r := [3][3]float64{
			{a*a + b*b - c*c - d*d, 2 * (b*c - a*d), 2 * (b*d + a*c)},
			{2 * (b*c + a*d), a*a + c*c - b*b - d*d, 2 * (c*d - a*b)},
			{2 * (b*d - a*c), 2 * (c*d + a*b), a*a + d*d - b*b - c*c},
		}
		off := [3]float64{
			float64(niftiFloat32(order, h[268:272])),
			float64(niftiFloat32(order, h[272:276])),
			float64(niftiFloat32(order, h[276:280])),
		}
		scale := [3]float64{sx, sy, sz * qfac}
		for i := 0; i < 3; i++ {
			affine[i*4+0] = r[i][0] * scale[0]
			affine[i*4+1] = r[i][1] * scale[1]
			affine[i*4+2] = r[i][2] * scale[2]
			affine[i*4+3] = off[i]
		}
		return affine, 2
	}
	// Method 1: pixdim scaling only; orientation is unknown (assumed RAS-ish).
	affine[0] = sx
	affine[5] = sy
	affine[10] = sz
	return affine, 0
}

func niftiAffineMethodName(code int) string {
	switch code {
	case 3:
		return "sform"
	case 2:
		return "qform"
	default:
		return "pixdim"
	}
}

var niftiAxisPositive = [3]string{"R", "A", "S"}
var niftiAxisNegative = [3]string{"L", "P", "I"}

// niftiOrientation classifies each voxel axis (i,j,k) to its dominant anatomical
// direction from the affine's 3x3, mirroring nibabel's aff2axcodes. Returns the
// 3-letter orientation code (e.g. "RAS"), the positive/negative anatomical end
// label per voxel axis, and which voxel axis is normal to each standard plane.
func niftiOrientation(affine [12]float64) (code string, ends [3][2]string, planeAxis map[string]int) {
	used := [3]bool{}
	var letters [3]byte
	for j := 0; j < 3; j++ {
		bestRow, bestAbs := -1, -1.0
		for i := 0; i < 3; i++ {
			if used[i] {
				continue
			}
			v := math.Abs(affine[i*4+j])
			if v > bestAbs {
				bestAbs = v
				bestRow = i
			}
		}
		if bestRow < 0 {
			for i := 0; i < 3; i++ {
				if !used[i] {
					bestRow = i
					break
				}
			}
		}
		used[bestRow] = true
		if affine[bestRow*4+j] >= 0 {
			letters[j] = niftiAxisPositive[bestRow][0]
			ends[j] = [2]string{niftiAxisNegative[bestRow], niftiAxisPositive[bestRow]}
		} else {
			letters[j] = niftiAxisNegative[bestRow][0]
			ends[j] = [2]string{niftiAxisPositive[bestRow], niftiAxisNegative[bestRow]}
		}
	}
	code = string(letters[:])
	planeAxis = map[string]int{}
	for axis := 0; axis < 3; axis++ {
		switch letters[axis] {
		case 'R', 'L':
			planeAxis["sagittal"] = axis
		case 'A', 'P':
			planeAxis["coronal"] = axis
		case 'S', 'I':
			planeAxis["axial"] = axis
		}
	}
	return code, ends, planeAxis
}

func clampNiftiIndex(index, count int) int {
	if count <= 0 || index < 0 {
		return 0
	}
	if index >= count {
		return count - 1
	}
	return index
}

func guardNiftiVolumeBytes(length int64) error {
	if length <= 0 {
		return errors.New("invalid NIfTI volume size")
	}
	if length > niftiMaxSingleVolumeBytes {
		return fmt.Errorf("NIfTI single-volume size %d bytes exceeds the %d byte limit", length, niftiMaxSingleVolumeBytes)
	}
	return nil
}

// loadNiftiScalarVolume reads timepoint 0 (channel 0 unless overridden) and
// preserves the historical signature for callers that don't select time.
func loadNiftiScalarVolume(path string, requestedChannel ...int) (niftiScalarVolume, error) {
	channelIndex := 0
	if len(requestedChannel) > 0 {
		channelIndex = requestedChannel[0]
	}
	return loadNiftiScalarVolumeAt(path, 0, channelIndex)
}

// loadNiftiScalarVolumeAt reads exactly one 3D scalar volume — the
// timeIndex-th timepoint, channelIndex-th channel — without materializing the
// rest of the 4D payload. For a 4D fMRI this bounds memory at a single ~MB
// timepoint instead of the whole multi-GB series, which is what previously OOM'd
// the control plane on every viewer request.
func loadNiftiScalarVolumeAt(path string, timeIndex, channelIndex int) (niftiScalarVolume, error) {
	gzipped := strings.HasSuffix(strings.ToLower(path), ".gz")
	if gzipped {
		// A one-time decompressed sidecar (when ready) turns every timepoint
		// into an O(1) random read; until then, stream the gzip and stop at the
		// requested slab.
		if sidecar := readyDecompressedNiftiSidecar(path); sidecar != "" {
			path = sidecar
			gzipped = false
		}
	}
	var (
		geom niftiGeometry
		slab []byte
		err  error
	)
	if gzipped {
		geom, slab, err = readNiftiSlabStreaming(path, timeIndex, channelIndex)
	} else {
		geom, slab, err = readNiftiSlabRandomAccess(path, timeIndex, channelIndex)
	}
	if err != nil {
		return niftiScalarVolume{}, err
	}
	if geom.bytesPerVoxel > 1 && geom.order != binary.LittleEndian {
		normalizeScalarPayloadToLittleEndian(slab, geom.bytesPerVoxel)
	}
	minValue, maxValue := niftiScalarRange(slab, geom.dtype, geom.bytesPerVoxel)
	clampedTime := clampNiftiIndex(timeIndex, geom.timeCount)
	clampedChannel := clampNiftiIndex(channelIndex, geom.channelCount)
	warnings := []string{}
	if geom.timeCount > 1 {
		warnings = append(warnings, fmt.Sprintf("NIfTI 4D time series with %d timepoints; serving timepoint %d.", geom.timeCount, clampedTime))
	}
	return niftiScalarVolume{
		Width:         geom.width,
		Height:        geom.height,
		Depth:         geom.depth,
		TimeCount:     geom.timeCount,
		TimeIndex:     clampedTime,
		ChannelCount:  geom.channelCount,
		ChannelIndex:  clampedChannel,
		DType:         geom.dtype,
		BytesPerVoxel: geom.bytesPerVoxel,
		Data:          slab,
		RawMin:        minValue,
		RawMax:        maxValue,
		SpacingX:      geom.spacingX,
		SpacingY:      geom.spacingY,
		SpacingZ:      geom.spacingZ,
		SclSlope:      geom.sclSlope,
		SclInter:      geom.sclInter,
		Affine:        geom.affine,
		AffineCode:    geom.affineCode,
		SpaceUnit:     geom.spaceUnit,
		Warnings:      warnings,
	}, nil
}

// readNiftiSlabRandomAccess serves one volume from an uncompressed .nii (or a
// decompressed sidecar) with two small reads: the header, then a single ReadAt
// of the requested slab. Any timepoint costs one ~MB disk read, regardless of
// series length.
func readNiftiSlabRandomAccess(path string, timeIndex, channelIndex int) (niftiGeometry, []byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return niftiGeometry{}, nil, err
	}
	defer func() { _ = f.Close() }()
	header, _, err := readNiftiHeaderBytes(f)
	if err != nil {
		return niftiGeometry{}, nil, fmt.Errorf("read NIfTI header: %w", err)
	}
	geom, err := parseNiftiGeometry(header)
	if err != nil {
		return niftiGeometry{}, nil, err
	}
	length := geom.volumeBytes()
	if err := guardNiftiVolumeBytes(length); err != nil {
		return niftiGeometry{}, nil, err
	}
	offset := geom.volumeOffset(clampNiftiIndex(timeIndex, geom.timeCount), clampNiftiIndex(channelIndex, geom.channelCount))
	if info, statErr := f.Stat(); statErr == nil && offset+length > info.Size() {
		return niftiGeometry{}, nil, fmt.Errorf("NIfTI voxel payload is incomplete: need %d bytes at offset %d", length, offset)
	}
	slab := make([]byte, length)
	if _, err := f.ReadAt(slab, offset); err != nil {
		return niftiGeometry{}, nil, fmt.Errorf("read NIfTI volume: %w", err)
	}
	return geom, slab, nil
}

// readNiftiSlabStreaming serves one volume from a .nii.gz by decompressing the
// stream only up to the end of the requested slab and retaining just that slab.
// Memory stays bounded to one volume; the tail of the series is never held.
func readNiftiSlabStreaming(path string, timeIndex, channelIndex int) (niftiGeometry, []byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return niftiGeometry{}, nil, err
	}
	defer func() { _ = f.Close() }()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return niftiGeometry{}, nil, err
	}
	defer func() { _ = zr.Close() }()
	header, headerConsumed, err := readNiftiHeaderBytes(zr)
	if err != nil {
		return niftiGeometry{}, nil, fmt.Errorf("read NIfTI header: %w", err)
	}
	geom, err := parseNiftiGeometry(header)
	if err != nil {
		return niftiGeometry{}, nil, err
	}
	length := geom.volumeBytes()
	if err := guardNiftiVolumeBytes(length); err != nil {
		return niftiGeometry{}, nil, err
	}
	offset := geom.volumeOffset(clampNiftiIndex(timeIndex, geom.timeCount), clampNiftiIndex(channelIndex, geom.channelCount))
	// Skip from the number of header bytes actually consumed (352 for NIfTI-1,
	// 544 for NIfTI-2) to the voxel payload.
	skip := offset - int64(headerConsumed)
	if skip < 0 {
		return niftiGeometry{}, nil, fmt.Errorf("invalid NIfTI voxel offset %d", offset)
	}
	if skip > 0 {
		if _, err := io.CopyN(io.Discard, zr, skip); err != nil {
			return niftiGeometry{}, nil, fmt.Errorf("seek NIfTI volume: %w", err)
		}
	}
	slab := make([]byte, length)
	if _, err := io.ReadFull(zr, slab); err != nil {
		return niftiGeometry{}, nil, fmt.Errorf("read NIfTI volume: %w", err)
	}
	return geom, slab, nil
}

// --- gzip random-access: decompress a 4D .nii.gz once to a sidecar -----------

var niftiDecompressInFlight sync.Map // sidecar path -> in-progress marker

// niftiDecompressSem bounds how many gzip sidecar builds run at once, so a burst of
// users opening large .nii.gz files can't saturate disk I/O / fill the disk with many
// simultaneous multi-GB decompressions. Lazily sized from env on first use.
var (
	niftiDecompressSemOnce sync.Once
	niftiDecompressSem     chan struct{}
)

func niftiDecompressSemaphore() chan struct{} {
	niftiDecompressSemOnce.Do(func() {
		n := 3
		if raw := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_NIFTI_DECOMPRESS_CONCURRENCY")); raw != "" {
			if v, err := strconv.Atoi(raw); err == nil && v > 0 {
				n = v
			}
		}
		niftiDecompressSem = make(chan struct{}, n)
	})
	return niftiDecompressSem
}

// niftiDecompressTimeout caps a single sidecar build so a corrupt/truncated gzip (or a
// stuck disk) can't leak a goroutine + open file handle forever. Generous (a 10GB+
// decompress is legitimate); only a genuine hang trips it.
func niftiDecompressTimeout() time.Duration {
	if raw := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_NIFTI_DECOMPRESS_TIMEOUT_S")); raw != "" {
		if v, err := strconv.Atoi(raw); err == nil && v > 0 {
			return time.Duration(v) * time.Second
		}
	}
	return 30 * time.Minute
}

func niftiDecompressCacheEnabled() bool {
	if raw, ok := os.LookupEnv("ULTRA_CONTROL_NIFTI_DECOMPRESS_CACHE"); ok {
		switch strings.ToLower(strings.TrimSpace(raw)) {
		case "0", "false", "no", "off":
			return false
		}
	}
	return true
}

// niftiDecompressedSidecarPath maps <root>/file_x.nii.gz to the decompressed
// <root>/derived/file_x.nii served via random access.
func niftiDecompressedSidecarPath(srcPath string) string {
	base := strings.TrimSuffix(filepath.Base(srcPath), ".gz")
	return filepath.Join(filepath.Dir(srcPath), "derived", base)
}

func readyDecompressedNiftiSidecar(srcPath string) string {
	if !niftiDecompressCacheEnabled() {
		return ""
	}
	dst := niftiDecompressedSidecarPath(srcPath)
	if info, err := os.Stat(dst); err == nil && info.Size() > 0 {
		return dst
	}
	return ""
}

// maybeDecompressNiftiSidecar kicks off a one-time background decompression of a
// gzipped time series so subsequent timepoints serve via O(1) ReadAt. It is
// best-effort: bounded-memory streaming requests already work without it, so a
// failure (disk full, etc.) never affects correctness. Concurrent callers
// dedupe on the destination path.
func maybeDecompressNiftiSidecar(srcPath string, timeCount int) {
	if timeCount <= 1 || !niftiDecompressCacheEnabled() {
		return
	}
	if !strings.HasSuffix(strings.ToLower(srcPath), ".gz") {
		return
	}
	dst := niftiDecompressedSidecarPath(srcPath)
	if info, err := os.Stat(dst); err == nil && info.Size() > 0 {
		return
	}
	if _, inFlight := niftiDecompressInFlight.LoadOrStore(dst, struct{}{}); inFlight {
		return
	}
	go func() {
		defer niftiDecompressInFlight.Delete(dst)
		// Bound concurrency: if too many sidecar builds are already running, skip —
		// the bounded streaming reader still serves correctly, and the build retries
		// on a later request. Non-blocking so we never pile up goroutines.
		sem := niftiDecompressSemaphore()
		select {
		case sem <- struct{}{}:
			defer func() { <-sem }()
		default:
			return
		}
		// Best-effort + time-bounded: a failed/hung sidecar (disk full, corrupt gzip)
		// only forgoes the random-access speedup and must not surface or leak.
		ctx, cancel := context.WithTimeout(context.Background(), niftiDecompressTimeout())
		defer cancel()
		_ = buildDecompressedNiftiSidecar(ctx, srcPath, dst)
	}()
}

func buildDecompressedNiftiSidecar(ctx context.Context, srcPath, dst string) (err error) {
	if info, statErr := os.Stat(dst); statErr == nil && info.Size() > 0 {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	in, err := os.Open(srcPath)
	if err != nil {
		return err
	}
	defer func() { _ = in.Close() }()
	// Abort a hung/slow decompress when ctx expires by closing the source, which
	// surfaces as a read error in the io.Copy below (the tmp file is then cleaned up).
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		select {
		case <-ctx.Done():
			_ = in.Close()
		case <-stop:
		}
	}()
	zr, err := gzip.NewReader(in)
	if err != nil {
		return err
	}
	defer func() { _ = zr.Close() }()
	tmp := dst + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	committed := false
	defer func() {
		_ = out.Close()
		if !committed {
			_ = os.Remove(tmp)
		}
	}()
	if _, err := io.Copy(out, zr); err != nil { // streaming copy, O(1) memory
		return err
	}
	if err := out.Sync(); err != nil {
		return err
	}
	if err := os.Rename(tmp, dst); err != nil { // atomic publish
		return err
	}
	committed = true
	return nil
}

func parseUploadScalarChannelIndex(r *http.Request) int {
	if r == nil {
		return 0
	}
	query := r.URL.Query()
	for _, key := range []string{"channel", "c"} {
		raw := strings.TrimSpace(query.Get(key))
		if raw == "" {
			continue
		}
		value, err := strconv.Atoi(raw)
		if err == nil && value >= 0 {
			return value
		}
	}
	for _, part := range strings.Split(query.Get("channels"), ",") {
		value, err := strconv.Atoi(strings.TrimSpace(part))
		if err == nil && value >= 0 {
			return value
		}
	}
	return 0
}

// parseUploadScalarTimeIndex reads the requested 4D timepoint (the NIfTI 4th
// dimension). Out-of-range values are clamped by the loader against the actual
// time count.
func parseUploadScalarTimeIndex(r *http.Request) int {
	if r == nil {
		return 0
	}
	query := r.URL.Query()
	for _, key := range []string{"t", "time", "timepoint"} {
		raw := strings.TrimSpace(query.Get(key))
		if raw == "" {
			continue
		}
		value, err := strconv.Atoi(raw)
		if err == nil && value >= 0 {
			return value
		}
	}
	return 0
}

// niftiScalarDimsOrder and niftiScalarArrayShape describe the full logical
// dataset (not the single served volume), outermost-first: channels, then time,
// then the spatial Z/Y/X. A 4D fMRI is "TZYX"; a multi-component volume "CZYX";
// both "CTZYX".
func niftiScalarDimsOrder(volume niftiScalarVolume) string {
	order := "ZYX"
	if volume.TimeCount > 1 {
		order = "T" + order
	}
	if volume.ChannelCount > 1 {
		order = "C" + order
	}
	return order
}

func niftiScalarArrayShape(volume niftiScalarVolume) []int {
	shape := []int{volume.Depth, volume.Height, volume.Width}
	if volume.TimeCount > 1 {
		shape = append([]int{volume.TimeCount}, shape...)
	}
	if volume.ChannelCount > 1 {
		shape = append([]int{volume.ChannelCount}, shape...)
	}
	return shape
}

func niftiDefaultChannelColors(channelCount int) []string {
	if channelCount <= 0 {
		channelCount = 1
	}
	palette := []string{"#ffffff", "#00ff00", "#ff00ff", "#00ffff", "#ffcc00", "#ff4d4d"}
	colors := make([]string, channelCount)
	for index := range colors {
		colors[index] = palette[index%len(palette)]
	}
	return colors
}

func niftiByteOrder(data []byte) (binary.ByteOrder, error) {
	if len(data) < 4 {
		return binary.LittleEndian, errors.New("NIfTI header is too small")
	}
	if binary.LittleEndian.Uint32(data[0:4]) == 348 {
		return binary.LittleEndian, nil
	}
	if binary.BigEndian.Uint32(data[0:4]) == 348 {
		return binary.BigEndian, nil
	}
	return binary.LittleEndian, errors.New("NIfTI header size is invalid")
}

func niftiInt16(order binary.ByteOrder, data []byte) int {
	return int(int16(order.Uint16(data)))
}

func niftiDimension(order binary.ByteOrder, data []byte) int {
	value := niftiInt16(order, data)
	if value < 0 {
		return 0
	}
	return value
}

func niftiFloat32(order binary.ByteOrder, data []byte) float32 {
	return math.Float32frombits(order.Uint32(data))
}

func niftiSpacing(order binary.ByteOrder, data []byte) float64 {
	value := float64(niftiFloat32(order, data))
	if !numberIsFinite(value) || value <= 0 {
		return 1
	}
	return value
}

func niftiScalarType(datatype int) (string, int, error) {
	switch datatype {
	case 2:
		return "uint8", 1, nil
	case 4:
		return "int16", 2, nil
	case 16:
		return "float32", 4, nil
	case 512:
		return "uint16", 2, nil
	default:
		return "", 0, fmt.Errorf("unsupported NIfTI scalar datatype %d", datatype)
	}
}

func normalizeScalarPayloadToLittleEndian(data []byte, bytesPerVoxel int) {
	switch bytesPerVoxel {
	case 2:
		for index := 0; index+1 < len(data); index += 2 {
			data[index], data[index+1] = data[index+1], data[index]
		}
	case 4:
		for index := 0; index+3 < len(data); index += 4 {
			data[index], data[index+1], data[index+2], data[index+3] = data[index+3], data[index+2], data[index+1], data[index]
		}
	}
}

func niftiScalarRange(data []byte, dtype string, bytesPerVoxel int) (float64, float64) {
	if len(data) == 0 || bytesPerVoxel <= 0 {
		return 0, 0
	}
	minValue := 0.0
	maxValue := 0.0
	seenFinite := false
	for index := 0; index+bytesPerVoxel <= len(data); index += bytesPerVoxel {
		value := niftiScalarDataValue(data, index, dtype, bytesPerVoxel)
		if !numberIsFinite(value) {
			continue
		}
		if !seenFinite || value < minValue {
			minValue = value
		}
		if !seenFinite || value > maxValue {
			maxValue = value
		}
		seenFinite = true
	}
	if !seenFinite {
		return 0, 0
	}
	return minValue, maxValue
}

func niftiScalarDataValue(data []byte, offset int, dtype string, bytesPerVoxel int) float64 {
	if offset < 0 || offset+bytesPerVoxel > len(data) {
		return 0
	}
	switch dtype {
	case "int16":
		return float64(int16(binary.LittleEndian.Uint16(data[offset : offset+2])))
	case "uint16":
		return float64(binary.LittleEndian.Uint16(data[offset : offset+2]))
	case "float32":
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(data[offset : offset+4])))
	default:
		return float64(data[offset])
	}
}

func formatScalarHeaderFloat(value float64) string {
	return strconv.FormatFloat(value, 'g', -1, 64)
}

func uploadPreviewTransformFromRequest(r *http.Request) uploadPreviewTransform {
	transform := uploadPreviewTransform{Gamma: 1}
	if r == nil {
		return transform
	}
	query := r.URL.Query()
	enhancement := strings.TrimSpace(query.Get("enhancement"))
	if strings.HasPrefix(enhancement, "hounsfield:") {
		parts := strings.Split(enhancement, ":")
		if len(parts) >= 3 {
			center, centerErr := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
			width, widthErr := strconv.ParseFloat(strings.TrimSpace(parts[2]), 64)
			if centerErr == nil && widthErr == nil && width > 0 {
				transform.WindowMin = center - width/2
				transform.WindowMax = center + width/2
				transform.WindowActive = true
				transform.WindowIsPhysical = true
			}
		}
	} else if strings.EqualFold(enhancement, "f") || strings.EqualFold(enhancement, "full") {
		transform.FullRange = true
	}
	if minRaw := strings.TrimSpace(query.Get("window_min")); minRaw != "" {
		if maxRaw := strings.TrimSpace(query.Get("window_max")); maxRaw != "" {
			minValue, minErr := strconv.ParseFloat(minRaw, 64)
			maxValue, maxErr := strconv.ParseFloat(maxRaw, 64)
			if minErr == nil && maxErr == nil && maxValue > minValue {
				transform.WindowMin = minValue
				transform.WindowMax = maxValue
				transform.WindowActive = true
				// Explicit window_min/window_max are raw sample values, not HU.
				transform.WindowIsPhysical = false
			}
		}
	}
	if gammaRaw := strings.TrimSpace(query.Get("gamma")); gammaRaw != "" {
		if gamma, err := strconv.ParseFloat(gammaRaw, 64); err == nil && gamma > 0 {
			transform.Gamma = math.Max(0.05, math.Min(8, gamma))
		}
	}
	transform.Channels, transform.ChannelsRequested = parseUploadHistogramChannels(query.Get("channels"), []int{})
	transform.Negative = parseBoolQuery(query.Get("negative"))
	return transform
}

func parseBoolQuery(raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "true", "t", "yes", "y", "on":
		return true
	default:
		return false
	}
}

func browserPreviewImage(img image.Image, transform uploadPreviewTransform) image.Image {
	if img == nil {
		return img
	}
	if gray16, ok := img.(*image.Gray16); ok {
		return windowGray16Like(gray16, resolveGray16Window(gray16, transform), transform)
	}
	if img.ColorModel() == color.Gray16Model {
		return windowGray16Like(img, resolveGray16Window(img, transform), transform)
	}
	if transform.ChannelsRequested || transform.Negative {
		return channelFilteredPreviewImage(img, transform)
	}
	return img
}

func channelFilteredPreviewImage(img image.Image, transform uploadPreviewTransform) image.Image {
	bounds := img.Bounds()
	output := image.NewRGBA(bounds)
	selected := map[int]bool{}
	for _, channel := range transform.Channels {
		if channel >= 0 && channel < 3 {
			selected[channel] = true
		}
	}
	if !transform.ChannelsRequested || len(selected) == 0 {
		selected = map[int]bool{0: true, 1: true, 2: true}
	}
	soloChannel := -1
	if len(selected) == 1 {
		for channel := range selected {
			soloChannel = channel
		}
	}
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r16, g16, b16, a16 := img.At(x, y).RGBA()
			components := [3]uint8{uint8(r16 >> 8), uint8(g16 >> 8), uint8(b16 >> 8)}
			alpha := uint8(a16 >> 8)
			if soloChannel >= 0 {
				value := components[soloChannel]
				if transform.Negative {
					value = 255 - value
				}
				output.SetRGBA(x, y, color.RGBA{R: value, G: value, B: value, A: alpha})
				continue
			}
			pixel := color.RGBA{A: alpha}
			if selected[0] {
				pixel.R = components[0]
			}
			if selected[1] {
				pixel.G = components[1]
			}
			if selected[2] {
				pixel.B = components[2]
			}
			if transform.Negative {
				if selected[0] {
					pixel.R = 255 - pixel.R
				}
				if selected[1] {
					pixel.G = 255 - pixel.G
				}
				if selected[2] {
					pixel.B = 255 - pixel.B
				}
			}
			output.SetRGBA(x, y, pixel)
		}
	}
	return output
}

func serveNiftiSliceAsPNG(w http.ResponseWriter, path string, r *http.Request) error {
	volume, err := loadNiftiScalarVolumeAt(path, parseUploadScalarTimeIndex(r), parseUploadScalarChannelIndex(r))
	if err != nil {
		return err
	}
	transform := uploadPreviewTransformFromRequest(r)
	windowMin, windowMax := scalarPreviewWindow(volume, transform)
	axis := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("axis")))
	if axis != "x" && axis != "y" && axis != "z" {
		axis = "z"
	}
	xIndex := parseNonNegativeInt(r.URL.Query().Get("x"), volume.Width/2)
	yIndex := parseNonNegativeInt(r.URL.Query().Get("y"), volume.Height/2)
	zIndex := parseNonNegativeInt(r.URL.Query().Get("z"), volume.Depth/2)
	if xIndex >= volume.Width {
		xIndex = volume.Width - 1
	}
	if yIndex >= volume.Height {
		yIndex = volume.Height - 1
	}
	if zIndex >= volume.Depth {
		zIndex = volume.Depth - 1
	}
	width, height := volume.Width, volume.Height
	if axis == "x" {
		width, height = volume.Height, volume.Depth
	} else if axis == "y" {
		width, height = volume.Width, volume.Depth
	}
	out := image.NewGray(image.Rect(0, 0, width, height))
	gamma := transform.Gamma
	if gamma <= 0 {
		gamma = 1
	}
	scale := windowMax - windowMin
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			x, y, z := col, row, zIndex
			if axis == "x" {
				x, y, z = xIndex, col, row
			} else if axis == "y" {
				x, y, z = col, yIndex, row
			}
			value := niftiScalarValue(volume, x, y, z)
			normalized := 0.0
			if scale > 0 {
				normalized = (value - windowMin) / scale
			}
			if normalized < 0 {
				normalized = 0
			} else if normalized > 1 {
				normalized = 1
			}
			if gamma != 1 {
				normalized = math.Pow(normalized, 1/gamma)
			}
			pixel := uint8(math.Round(normalized * 255))
			if transform.Negative {
				pixel = 255 - pixel
			}
			out.SetGray(col, row, color.Gray{Y: pixel})
		}
	}
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "private, max-age=3600")
	return png.Encode(w, out)
}

func scalarPreviewWindow(volume niftiScalarVolume, transform uploadPreviewTransform) (float64, float64) {
	if transform.FullRange {
		// "Full range" means the full span of the actual DATA, not the datatype.
		// Windowing an int16 MRI (values ~0..1000) to the [-32768, 32767] datatype
		// span rendered it near-black; the data range keeps it legible. uint8
		// display data spans the full 0..255 byte range.
		if volume.DType == "uint8" {
			return 0, 255
		}
		if volume.RawMax > volume.RawMin {
			return volume.RawMin, volume.RawMax
		}
		switch volume.DType {
		case "int16":
			return math.MinInt16, math.MaxInt16
		case "uint16":
			return 0, math.MaxUint16
		}
	}
	if transform.WindowActive && transform.WindowMax > transform.WindowMin {
		// niftiScalarValue samples raw stored codes, so a physical-unit (HU)
		// window must be converted back to code space to compare correctly.
		if transform.WindowIsPhysical {
			lo := volume.codeFromPhysical(transform.WindowMin)
			hi := volume.codeFromPhysical(transform.WindowMax)
			if hi < lo {
				lo, hi = hi, lo
			}
			return lo, hi
		}
		return transform.WindowMin, transform.WindowMax
	}
	return volume.RawMin, volume.RawMax
}

func niftiScalarValue(volume niftiScalarVolume, x int, y int, z int) float64 {
	if x < 0 || y < 0 || z < 0 || x >= volume.Width || y >= volume.Height || z >= volume.Depth {
		return 0
	}
	offset := ((z*volume.Height+y)*volume.Width + x) * volume.BytesPerVoxel
	return niftiScalarDataValue(volume.Data, offset, volume.DType, volume.BytesPerVoxel)
}

func parseNonNegativeInt(raw string, fallback int) int {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || value < 0 {
		return fallback
	}
	return value
}

type gray16Window struct {
	Min uint16
	Max uint16
}

func resolveGray16Window(img image.Image, transform uploadPreviewTransform) gray16Window {
	bounds := img.Bounds()
	if bounds.Empty() {
		return gray16Window{}
	}
	if transform.FullRange {
		return gray16Window{Min: 0, Max: math.MaxUint16}
	}
	if transform.WindowActive && transform.WindowMax > transform.WindowMin {
		return gray16Window{
			Min: clampFloatToUint16(transform.WindowMin),
			Max: clampFloatToUint16(transform.WindowMax),
		}
	}
	minValue := uint16(math.MaxUint16)
	maxValue := uint16(0)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			value := color.Gray16Model.Convert(img.At(x, y)).(color.Gray16).Y
			if value < minValue {
				minValue = value
			}
			if value > maxValue {
				maxValue = value
			}
		}
	}
	return gray16Window{Min: minValue, Max: maxValue}
}

func clampFloatToUint16(value float64) uint16 {
	if !numberIsFinite(value) || value <= 0 {
		return 0
	}
	if value >= math.MaxUint16 {
		return math.MaxUint16
	}
	return uint16(math.Round(value))
}

func numberIsFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func windowGray16Like(img image.Image, window gray16Window, transform uploadPreviewTransform) image.Image {
	bounds := img.Bounds()
	out := image.NewGray(bounds)
	if bounds.Empty() {
		return out
	}
	if window.Max <= window.Min {
		fill := uint8(window.Max >> 8)
		if transform.Negative {
			fill = 255 - fill
		}
		for index := range out.Pix {
			out.Pix[index] = fill
		}
		return out
	}
	scale := float64(window.Max - window.Min)
	gamma := transform.Gamma
	if gamma <= 0 {
		gamma = 1
	}
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			value := color.Gray16Model.Convert(img.At(x, y)).(color.Gray16).Y
			normalized := (float64(value) - float64(window.Min)) / scale
			if normalized < 0 {
				normalized = 0
			} else if normalized > 1 {
				normalized = 1
			}
			if gamma != 1 {
				normalized = math.Pow(normalized, 1/gamma)
			}
			windowed := uint8(math.Round(normalized * 255))
			if transform.Negative {
				windowed = 255 - windowed
			}
			out.SetGray(x, y, color.Gray{Y: windowed})
		}
	}
	return out
}

const (
	uploadHistogramDefaultBins = 256
	uploadHistogramMinBins     = 8
	uploadHistogramMaxBins     = 4096
	uploadHistogramMaxSamples  = 5_000_000
)

func histogramForUploadImage(img image.Image, binCount int, channelIndices []int, channelsRequested bool, timeIndex int) (uploadHistogramResponse, error) {
	if img == nil {
		return uploadHistogramResponse{}, errors.New("image histogram source is empty")
	}
	if binCount < uploadHistogramMinBins {
		binCount = uploadHistogramMinBins
	}
	if binCount > uploadHistogramMaxBins {
		binCount = uploadHistogramMaxBins
	}
	bounds := img.Bounds()
	if bounds.Empty() {
		return uploadHistogramResponse{}, errors.New("image histogram source has no pixels")
	}
	scalar := uploadHistogramScalarImage(img)
	if scalar {
		channelIndices = []int{0}
		channelsRequested = false
	} else if len(channelIndices) == 0 {
		channelIndices = uploadHistogramDefaultChannels(img)
	}
	dtype := uploadHistogramDType(img)
	stride := uploadHistogramStride(bounds)
	minValue := 0.0
	maxValue := 0.0
	sampleCount := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y += stride {
		for x := bounds.Min.X; x < bounds.Max.X; x += stride {
			value := uploadHistogramSampleValue(img, x, y, dtype, channelIndices, channelsRequested)
			if sampleCount == 0 || value < minValue {
				minValue = value
			}
			if sampleCount == 0 || value > maxValue {
				maxValue = value
			}
			sampleCount++
		}
	}
	if sampleCount == 0 {
		return uploadHistogramResponse{}, errors.New("image histogram source has no sampled pixels")
	}
	counts := make([]int, binCount)
	valueRange := maxValue - minValue
	for y := bounds.Min.Y; y < bounds.Max.Y; y += stride {
		for x := bounds.Min.X; x < bounds.Max.X; x += stride {
			binIndex := 0
			if valueRange > 0 {
				value := uploadHistogramSampleValue(img, x, y, dtype, channelIndices, channelsRequested)
				binIndex = int(math.Floor(((value - minValue) / valueRange) * float64(binCount)))
				if binIndex < 0 {
					binIndex = 0
				} else if binIndex >= binCount {
					binIndex = binCount - 1
				}
			}
			counts[binIndex]++
		}
	}
	return uploadHistogramResponse{
		Bins:        binCount,
		DType:       dtype,
		Channels:    append([]int(nil), channelIndices...),
		Source:      "decoded-image",
		SampleCount: sampleCount,
		Histogram: uploadHistogramPayload{
			Bins:           counts,
			Edges:          uploadHistogramEdges(minValue, maxValue, binCount),
			Min:            minValue,
			Max:            maxValue,
			ChannelIndices: append([]int(nil), channelIndices...),
			TimeIndex:      timeIndex,
		},
	}, nil
}

func histogramForNiftiScalarVolume(volume niftiScalarVolume, binCount int, timeIndex int) (uploadHistogramResponse, error) {
	if binCount < uploadHistogramMinBins {
		binCount = uploadHistogramMinBins
	}
	if binCount > uploadHistogramMaxBins {
		binCount = uploadHistogramMaxBins
	}
	if volume.BytesPerVoxel <= 0 || len(volume.Data) < volume.BytesPerVoxel {
		return uploadHistogramResponse{}, errors.New("scalar volume histogram source has no voxels")
	}
	voxelCount := len(volume.Data) / volume.BytesPerVoxel
	stride := uploadScalarVolumeHistogramStride(voxelCount)
	counts := make([]int, binCount)
	valueRange := volume.RawMax - volume.RawMin
	sampleCount := 0
	for voxelIndex := 0; voxelIndex < voxelCount; voxelIndex += stride {
		offset := voxelIndex * volume.BytesPerVoxel
		value := niftiScalarDataValue(volume.Data, offset, volume.DType, volume.BytesPerVoxel)
		if !numberIsFinite(value) {
			continue
		}
		binIndex := 0
		if valueRange > 0 {
			binIndex = int(math.Floor(((value - volume.RawMin) / valueRange) * float64(binCount)))
			if binIndex < 0 {
				binIndex = 0
			} else if binIndex >= binCount {
				binIndex = binCount - 1
			}
		}
		counts[binIndex]++
		sampleCount++
	}
	if sampleCount == 0 {
		return uploadHistogramResponse{}, errors.New("scalar volume histogram source has no sampled voxels")
	}
	channels := []int{volume.ChannelIndex}
	return uploadHistogramResponse{
		Bins:        binCount,
		DType:       volume.DType,
		Channels:    channels,
		Source:      "scalar-volume",
		SampleCount: sampleCount,
		Histogram: uploadHistogramPayload{
			Bins:           counts,
			Edges:          uploadHistogramEdges(volume.RawMin, volume.RawMax, binCount),
			Min:            volume.RawMin,
			Max:            volume.RawMax,
			ChannelIndices: channels,
			TimeIndex:      timeIndex,
		},
	}, nil
}

func uploadScalarVolumeHistogramStride(voxelCount int) int {
	if voxelCount <= uploadHistogramMaxSamples {
		return 1
	}
	stride := int(math.Ceil(float64(voxelCount) / float64(uploadHistogramMaxSamples)))
	if stride < 1 {
		return 1
	}
	return stride
}

func uploadHistogramStride(bounds image.Rectangle) int {
	pixelCount := int64(bounds.Dx()) * int64(bounds.Dy())
	if pixelCount <= uploadHistogramMaxSamples {
		return 1
	}
	stride := int(math.Ceil(math.Sqrt(float64(pixelCount) / float64(uploadHistogramMaxSamples))))
	if stride < 1 {
		return 1
	}
	return stride
}

func uploadHistogramEdges(minValue float64, maxValue float64, binCount int) []float64 {
	edges := make([]float64, binCount+1)
	rangeMax := maxValue
	if rangeMax <= minValue {
		rangeMax = minValue + 1
	}
	for index := range edges {
		edges[index] = minValue + (rangeMax-minValue)*float64(index)/float64(binCount)
	}
	return edges
}

func uploadHistogramSampleValue(img image.Image, x int, y int, dtype string, channelIndices []int, channelsRequested bool) float64 {
	model := img.ColorModel()
	switch model {
	case color.Gray16Model:
		return float64(color.Gray16Model.Convert(img.At(x, y)).(color.Gray16).Y)
	case color.Alpha16Model:
		return float64(color.Alpha16Model.Convert(img.At(x, y)).(color.Alpha16).A)
	case color.GrayModel:
		return float64(color.GrayModel.Convert(img.At(x, y)).(color.Gray).Y)
	case color.AlphaModel:
		return float64(color.AlphaModel.Convert(img.At(x, y)).(color.Alpha).A)
	}
	r, g, b, _ := img.At(x, y).RGBA()
	scale := 1.0
	if dtype != "uint16" {
		scale = 1.0 / 257.0
	}
	components := [3]float64{float64(r) * scale, float64(g) * scale, float64(b) * scale}
	if channelsRequested && len(channelIndices) > 0 {
		total := 0.0
		used := 0
		for _, channel := range channelIndices {
			if channel >= 0 && channel < len(components) {
				total += components[channel]
				used++
			}
		}
		if used > 0 {
			return total / float64(used)
		}
	}
	return components[0]*0.299 + components[1]*0.587 + components[2]*0.114
}

func uploadHistogramDefaultChannels(img image.Image) []int {
	if uploadHistogramScalarImage(img) {
		return []int{0}
	}
	return []int{0, 1, 2}
}

func uploadHistogramScalarImage(img image.Image) bool {
	if img == nil || img.ColorModel() == nil {
		return false
	}
	switch img.ColorModel() {
	case color.GrayModel, color.Gray16Model, color.AlphaModel, color.Alpha16Model:
		return true
	default:
		return false
	}
}

func uploadHistogramDType(img image.Image) string {
	if img == nil || img.ColorModel() == nil {
		return "uint8"
	}
	switch img.ColorModel() {
	case color.Gray16Model, color.Alpha16Model, color.RGBA64Model, color.NRGBA64Model:
		return "uint16"
	default:
		return "uint8"
	}
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

func uploadNeedsBrowserPNG(originalName string, contentType string) bool {
	return isTIFFUpload(originalName, contentType)
}

func uploadRequestNeedsBrowserPNG(originalName string, contentType string, r *http.Request) bool {
	if uploadNeedsBrowserPNG(originalName, contentType) {
		return true
	}
	if r == nil {
		return false
	}
	query := r.URL.Query()
	if strings.TrimSpace(query.Get("channels")) != "" ||
		strings.TrimSpace(query.Get("window_min")) != "" ||
		strings.TrimSpace(query.Get("window_max")) != "" ||
		strings.TrimSpace(query.Get("gamma")) != "" {
		return true
	}
	if parseBoolQuery(query.Get("negative")) {
		return true
	}
	enhancement := strings.TrimSpace(query.Get("enhancement"))
	return enhancement != "" && !strings.EqualFold(enhancement, "d") && !strings.EqualFold(enhancement, "dynamic")
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
	if _, err := copyWithPooledBuffer(hasher, file); err != nil {
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
	if isTIFFUpload(originalName, hint) {
		return "image/tiff"
	}
	if isNiftiUpload(originalName, hint) {
		return "application/x-nifti"
	}
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
	case detectSpecialFormatByName(originalName) != nil:
		// OME-Zarr (and other registry special formats) render as images.
		if sf := detectSpecialFormatByName(originalName); sf.ResourceKind != "" {
			return sf.ResourceKind
		}
		return "image"
	case isTIFFUpload(originalName, contentType):
		return "image"
	case strings.HasPrefix(contentType, "image/"):
		return "image"
	case strings.HasPrefix(contentType, "video/"):
		return "video"
	case isTabularUpload(originalName, contentType):
		return "table"
	case isTextDocumentUpload(originalName, contentType):
		return "document"
	default:
		return "file"
	}
}

// isTabularUpload reports whether the file is a delimited table (CSV/TSV) that
// the text/data viewer should open as a table.
func isTabularUpload(originalName string, contentType string) bool {
	if strings.Contains(strings.ToLower(contentType), "csv") || strings.Contains(strings.ToLower(contentType), "tab-separated") {
		return true
	}
	switch strings.ToLower(filepath.Ext(originalName)) {
	case ".csv", ".tsv":
		return true
	}
	return false
}

// isTextDocumentUpload reports whether the file is a human-readable text/data
// document (JSON/YAML/XML/Markdown/plain text/logs) that the text viewer renders.
// Detection leans on the extension first because chunked uploads frequently
// persist "application/octet-stream" as the content type.
func isTextDocumentUpload(originalName string, contentType string) bool {
	normalizedType := strings.ToLower(strings.TrimSpace(contentType))
	switch {
	case strings.HasPrefix(normalizedType, "text/"):
		return true
	case normalizedType == "application/json" || strings.HasSuffix(normalizedType, "+json"):
		return true
	case normalizedType == "application/xml" || strings.HasSuffix(normalizedType, "+xml"):
		return true
	case normalizedType == "application/x-yaml" || normalizedType == "application/yaml":
		return true
	}
	switch strings.ToLower(filepath.Ext(originalName)) {
	case ".json", ".jsonl", ".ndjson", ".geojson",
		".yaml", ".yml",
		".xml", ".xsd", ".xslt",
		".md", ".markdown", ".mdx",
		".txt", ".text", ".log",
		".ini", ".toml", ".cfg", ".conf", ".properties", ".env":
		return true
	}
	return false
}

func isOmeTIFFName(originalName string) bool {
	lowerName := strings.ToLower(strings.TrimSpace(originalName))
	return strings.HasSuffix(lowerName, ".ome.tif") || strings.HasSuffix(lowerName, ".ome.tiff")
}

func isTIFFUpload(originalName string, contentType string) bool {
	normalizedType := strings.ToLower(strings.TrimSpace(contentType))
	if normalizedType == "image/tiff" || normalizedType == "image/tif" {
		return true
	}
	lowerName := strings.ToLower(strings.TrimSpace(originalName))
	return strings.HasSuffix(lowerName, ".tif") || strings.HasSuffix(lowerName, ".tiff")
}

func isNiftiUpload(originalName string, contentType string) bool {
	normalizedType := strings.ToLower(strings.TrimSpace(contentType))
	if normalizedType == "application/x-nifti" || normalizedType == "image/x-nifti" {
		return true
	}
	lowerName := strings.ToLower(strings.TrimSpace(originalName))
	return strings.HasSuffix(lowerName, ".nii") || strings.HasSuffix(lowerName, ".nii.gz") || strings.HasSuffix(lowerName, ".nifti")
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
	reader := http.MaxBytesReader(w, r.Body, maxJSONBodyBytes)
	decoder := json.NewDecoder(reader)
	if err := decoder.Decode(target); err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			writeError(w, http.StatusRequestEntityTooLarge, fmt.Errorf("JSON request body exceeds %d bytes", maxJSONBodyBytes))
			return false
		}
		writeError(w, http.StatusBadRequest, err)
		return false
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			writeError(w, http.StatusRequestEntityTooLarge, fmt.Errorf("JSON request body exceeds %d bytes", maxJSONBodyBytes))
			return false
		}
		writeError(w, http.StatusBadRequest, errors.New("request body must contain a single JSON value"))
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

func parseUploadHistogramBins(r *http.Request) int {
	raw := strings.TrimSpace(r.URL.Query().Get("bins"))
	if raw == "" {
		return uploadHistogramDefaultBins
	}
	bins, err := strconv.Atoi(raw)
	if err != nil {
		return uploadHistogramDefaultBins
	}
	if bins < uploadHistogramMinBins {
		return uploadHistogramMinBins
	}
	if bins > uploadHistogramMaxBins {
		return uploadHistogramMaxBins
	}
	return bins
}

func parseUploadHistogramChannels(raw string, fallback []int) ([]int, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return append([]int(nil), fallback...), false
	}
	seen := map[int]bool{}
	channels := []int{}
	for _, part := range strings.Split(raw, ",") {
		channel, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || channel < 0 || channel > 3 || seen[channel] {
			continue
		}
		seen[channel] = true
		channels = append(channels, channel)
	}
	if len(channels) == 0 {
		return append([]int(nil), fallback...), false
	}
	return channels, true
}

func parseUploadHistogramTimeIndex(r *http.Request) int {
	raw := strings.TrimSpace(r.URL.Query().Get("t"))
	if raw == "" {
		return 0
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 0 {
		return 0
	}
	return value
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
