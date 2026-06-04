package httpapi

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
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
	"math"
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
	"unicode"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/go-chi/chi/v5"
	"golang.org/x/image/tiff"
)

type ServerDeps struct {
	Version           string
	Runs              *runcontrol.Service
	Store             runcontrol.Store
	Bus               runEventSource
	ArtifactRoot      string
	UploadRoot        string
	DevAdminEnabled   bool
	Runtime           RuntimeSummary
	QueueDiagnostics  eventbus.QueueDiagnosticsProvider
	Bisque            *BisqueService
	BisqueCredentials *BisqueCredentialStore
	WorkOS            *WorkOSAuth
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
	bisqueSessionCookieName      = "ultra_bisque_session"
	runEventMaxPageLimit         = 1000
	runEventStreamHeartbeatEvery = 15 * time.Second
	runEventStreamCatchupEvery   = time.Second
)

func NewRouter(deps ServerDeps) http.Handler {
	if deps.BisqueCredentials == nil {
		deps.BisqueCredentials = NewBisqueCredentialStore()
	}
	if !deps.WorkOS.Enabled() {
		_ = deps.ensureLocalBootstrapAccounts(context.Background())
	}
	r := chi.NewRouter()
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
			r.Get("/uploads/{file_id}/scalar-volume", deps.handleGetUploadScalarVolume)
			r.Get("/uploads/{file_id}/tiles/{axis}/{level}/{tile_x}/{tile_y}", deps.handleNotConfigured("upload tile pyramid delivery is not configured in the Go control plane yet"))
			r.Get("/uploads/{file_id}/atlas", deps.handleNotConfigured("upload atlas delivery is not configured in the Go control plane yet"))
			r.Get("/uploads/{file_id}/histogram", deps.handleGetUploadHistogram)
			r.Get("/uploads/{file_id}/caption", deps.handleGetUploadCaption)
			r.Post("/uploads/from-bisque", deps.handleImportBisqueResources)
			r.Post("/bisque/search", deps.handleBisqueSearch)
			r.Post("/bisque/download", deps.handleImportBisqueResources)
			r.Post("/bisque/upload", deps.handleBisqueUpload)
			r.Post("/bisque/unlink", deps.handleBisqueUnlink)
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
			r.Group(func(r chi.Router) {
				if deps.WorkOS.Enabled() {
					r.Use(deps.requireWorkOSAdmin)
				}
				r.Get("/admin/overview", deps.handleAdminOverview)
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

func localBootstrapAccounts() []localBootstrapAccount {
	return []localBootstrapAccount{
		{
			Username:    "admin",
			Password:    "admin",
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
	resolvedSnapshot, _, _, err := deps.resolveWorkOSAccount(r.Context(), snapshot)
	if err != nil {
		return nil, err
	}
	return resolvedSnapshot.sessionResponse(), nil
}

func (deps ServerDeps) requireWorkOSAccount(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		snapshot, authenticated := deps.WorkOS.authenticateRequest(w, r)
		if !authenticated {
			writeError(w, http.StatusUnauthorized, errors.New("authentication required"))
			return
		}
		resolvedSnapshot, _, _, err := deps.resolveWorkOSAccount(r.Context(), snapshot)
		if err != nil {
			writeStoreError(w, err)
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
					return applyUltraAccountToWorkOSSnapshot(snapshot, account), account, true, nil
				}
			}
			return snapshot, domain.UserAccount{}, false, err
		}
	}
	resolved := applyUltraAccountToWorkOSSnapshot(snapshot, account)
	return resolved, account, true, nil
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
	sessionID := bisqueSessionIDFromRequest(r)
	if sessionID == "" {
		return false
	}
	_, found, err := deps.BisqueCredentials.GetWithContext(ctx, sessionID)
	return err == nil && found
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
	SourceURI    string          `json:"source_uri,omitempty"`
	PreviewURL   string          `json:"preview_url,omitempty"`
	Principal    principalRecord `json:"principal,omitempty"`
}

type uploadFilesResponse struct {
	FileCount int                  `json:"file_count"`
	Uploaded  []uploadedFileRecord `json:"uploaded"`
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
	Activity     []adminActivityPeriod  `json:"activity"`
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

func (deps ServerDeps) handleListThreads(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	limit := clampLimit(parseLimit(r, 100), 500)
	offset := parseOffset(r)
	page, err := deps.Store.ListThreads(r.Context(), limit, offset, strings.TrimSpace(r.URL.Query().Get("status")))
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
	principal := deps.principalFromRequest(r, req.UserID)
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
		JobMetadata:         deps.bisqueJobMetadataFromRequest(r),
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

	principal := deps.principalFromRequest(r, "")
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
	principal := deps.principalFromRequest(r, "")
	query := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("q")))
	kind := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("kind")))
	source := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("source")))
	filtered := resources[:0]
	for _, resource := range resources {
		if !resourceVisibleToPrincipal(resource, principal) {
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
	}
	for _, candidate := range candidates {
		if strings.Contains(strings.ToLower(strings.TrimSpace(candidate)), query) {
			return true
		}
	}
	return false
}

func (deps ServerDeps) handleGetResource(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	record, _, err := findUploadResourceForRequest(root, deps.principalFromRequest(r, ""), chi.URLParam(r, "file_id"))
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
	_, path, err := findUploadResourceForRequest(root, deps.principalFromRequest(r, ""), fileID)
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
	record, path, err := findUploadResourceForRequest(root, deps.principalFromRequest(r, ""), chi.URLParam(r, "file_id"))
	if err != nil {
		writeStoreError(w, err)
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
	record, path, err := findUploadResourceForRequest(root, deps.principalFromRequest(r, ""), chi.URLParam(r, "file_id"))
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if !isNiftiUpload(record.OriginalName, record.ContentType) {
		writeError(w, http.StatusUnsupportedMediaType, errors.New("upload scalar volume is only available for NIfTI resources"))
		return
	}
	volume, err := loadNiftiScalarVolume(path, parseUploadScalarChannelIndex(r))
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, err)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Cache-Control", "private, max-age=3600")
	w.Header().Set("x-volume-width", strconv.Itoa(volume.Width))
	w.Header().Set("x-volume-height", strconv.Itoa(volume.Height))
	w.Header().Set("x-volume-depth", strconv.Itoa(volume.Depth))
	w.Header().Set("x-volume-dtype", volume.DType)
	w.Header().Set("x-volume-bytes-per-voxel", strconv.Itoa(volume.BytesPerVoxel))
	w.Header().Set("x-volume-raw-min", formatScalarHeaderFloat(volume.RawMin))
	w.Header().Set("x-volume-raw-max", formatScalarHeaderFloat(volume.RawMax))
	w.Header().Set("x-volume-channel", strconv.Itoa(volume.ChannelIndex))
	_, _ = w.Write(volume.Data)
}

func (deps ServerDeps) handleGetUploadHistogram(w http.ResponseWriter, r *http.Request) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	record, path, err := findUploadResourceForRequest(root, deps.principalFromRequest(r, ""), chi.URLParam(r, "file_id"))
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
	record, path, err := findUploadResourceForRequest(root, deps.principalFromRequest(r, ""), chi.URLParam(r, "file_id"))
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
	record, path, err := findUploadResourceForRequest(root, deps.principalFromRequest(r, ""), chi.URLParam(r, "file_id"))
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
			"available_surfaces":   []string{"2d", "metadata"},
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
	volume, err := loadNiftiScalarVolume(path)
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, err)
		return
	}
	dimsOrder := niftiScalarDimsOrder(volume)
	arrayShape := niftiScalarArrayShape(volume)
	channelColors := niftiDefaultChannelColors(volume.ChannelCount)
	displayCapabilities := []string{"slice_navigation", "histogram", "volume_context", "physical_scale", "window_level", "scalar_probe", "diagnostic_mpr"}
	viewerCapabilities := []string{"webgl_first_paint", "scalar_volume_delivery", "linear_sampling", "mpr_truth_surface", "slice_navigation", "volume_context", "physical_scale", "window_level"}
	if volume.ChannelCount > 1 {
		displayCapabilities = append(displayCapabilities, "channel_visibility")
		viewerCapabilities = append(viewerCapabilities, "channel_selection")
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
	writeJSON(w, http.StatusOK, map[string]any{
		"kind":          "image",
		"file_id":       record.FileID,
		"original_name": record.OriginalName,
		"modality":      "medical",
		"backend_mode":  "scalar",
		"dims_order":    dimsOrder,
		"axis_sizes": map[string]int{
			"T": 1,
			"C": volume.ChannelCount,
			"Z": volume.Depth,
			"Y": volume.Height,
			"X": volume.Width,
		},
		"selected_indices": map[string]int{"T": 0, "C": 0, "Z": volume.Depth / 2},
		"is_volume":        volume.Depth > 1,
		"is_timeseries":    false,
		"is_multichannel":  volume.ChannelCount > 1,
		"service_urls":     serviceURLs,
		"display_defaults": map[string]any{
			"enhancement":     "d",
			"negative":        false,
			"rotate":          0,
			"fusion_method":   "a",
			"channel_mode":    "single",
			"channels":        []int{0},
			"channel_colors":  channelColors,
			"time_index":      0,
			"z_index":         volume.Depth / 2,
			"volume_channel":  0,
			"volume_clip_min": map[string]float64{"x": 0, "y": 0, "z": 0},
			"volume_clip_max": map[string]float64{"x": 1, "y": 1, "z": 1},
		},
		"metadata": map[string]any{
			"reader":           "nifti-1",
			"dims_order":       dimsOrder,
			"array_shape":      arrayShape,
			"array_dtype":      volume.DType,
			"array_min":        volume.RawMin,
			"array_max":        volume.RawMax,
			"intensity_stats":  map[string]float64{"min": volume.RawMin, "max": volume.RawMax},
			"physical_spacing": spacing,
			"scene_count":      1,
			"warnings":         volume.Warnings,
			"content_type":     record.ContentType,
			"size_bytes":       record.SizeBytes,
			"sha256":           record.SHA256,
		},
		"viewer": map[string]any{
			"status":               "ready",
			"warmup_mode":          "lazy",
			"backend_mode":         "scalar",
			"default_surface":      "volume",
			"available_surfaces":   []string{"2d", "mpr", "volume", "metadata"},
			"default_axis":         "z",
			"slice_axes":           []string{"z", "y", "x"},
			"channel_mode":         "single",
			"volume_mode":          "scalar",
			"render_policy":        "scalar",
			"delivery_mode":        "scalar",
			"diagnostic_surface":   "mpr",
			"first_paint_mode":     "webgl",
			"measurement_policy":   "spacing-aware",
			"texture_policy":       "linear",
			"display_capabilities": displayCapabilities,
			"viewer_capabilities":  viewerCapabilities,
			"service_urls":         serviceURLs,
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
		Activity:     data.Activity,
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
	GeneratedAt  string
	KPIs         adminPlatformKPIs
	Activity     []adminActivityPeriod
	UsageLast24h []adminUsageBucket
	ToolUsage7d  []adminToolUsageRecord
	Workers      []adminWorkerRecord
	Users        []adminUserSummary
	Issues       []adminIssueRecord
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

func (deps ServerDeps) loadAdminSnapshot(ctx context.Context) (adminSnapshot, error) {
	now := domain.Now()
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
		Activity: activity.periods(),
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
	return writeUploadMetadataRecord(root, fileID, uploadMetadataRecord{Principal: principal.record()})
}

type uploadMetadataRecord struct {
	Principal  principalRecord `json:"principal"`
	SourceURI  string          `json:"source_uri,omitempty"`
	SourceType string          `json:"source_type,omitempty"`
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

func findUploadResourceForRequest(root string, principal requestPrincipal, fileID string) (resourceRecord, string, error) {
	record, path, err := findUploadResource(root, fileID)
	if err != nil {
		return resourceRecord{}, "", err
	}
	if !resourceVisibleToPrincipal(record, principal) {
		return resourceRecord{}, "", store.ErrNotFound
	}
	return record, path, nil
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
	WindowMin         float64
	WindowMax         float64
	WindowActive      bool
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
	Warnings      []string
}

func loadNiftiScalarVolume(path string, requestedChannel ...int) (niftiScalarVolume, error) {
	data, err := readPossiblyGzippedFile(path)
	if err != nil {
		return niftiScalarVolume{}, err
	}
	if len(data) < 352 {
		return niftiScalarVolume{}, errors.New("NIfTI file is too small")
	}
	order, err := niftiByteOrder(data)
	if err != nil {
		return niftiScalarVolume{}, err
	}
	magic := string(data[344:348])
	if magic != "n+1\x00" && magic != "ni1\x00" {
		return niftiScalarVolume{}, fmt.Errorf("unsupported NIfTI magic %q", strings.TrimRight(magic, "\x00"))
	}
	dim0 := niftiInt16(order, data[40:42])
	if dim0 < 2 {
		return niftiScalarVolume{}, fmt.Errorf("unsupported NIfTI dimension count %d", dim0)
	}
	width := niftiDimension(order, data[42:44])
	height := niftiDimension(order, data[44:46])
	depth := 1
	if dim0 >= 3 {
		depth = niftiDimension(order, data[46:48])
	}
	channelCount := 1
	if dim0 >= 4 {
		channelCount = niftiDimension(order, data[48:50])
	}
	if width <= 0 || height <= 0 || depth <= 0 {
		return niftiScalarVolume{}, fmt.Errorf("invalid NIfTI dimensions %dx%dx%d", width, height, depth)
	}
	if channelCount <= 0 {
		channelCount = 1
	}
	datatype := niftiInt16(order, data[70:72])
	dtype, bytesPerVoxel, err := niftiScalarType(datatype)
	if err != nil {
		return niftiScalarVolume{}, err
	}
	voxOffset := int(math.Round(float64(niftiFloat32(order, data[108:112]))))
	if voxOffset < 352 {
		voxOffset = 352
	}
	voxelCount := width * height * depth
	channelPayloadBytes := voxelCount * bytesPerVoxel
	totalByteCount := channelPayloadBytes * channelCount
	if voxelCount <= 0 || totalByteCount <= 0 || voxOffset+totalByteCount > len(data) {
		return niftiScalarVolume{}, fmt.Errorf("NIfTI voxel payload is incomplete: need %d bytes at offset %d", totalByteCount, voxOffset)
	}
	channelIndex := 0
	if len(requestedChannel) > 0 {
		channelIndex = requestedChannel[0]
	}
	if channelIndex < 0 {
		channelIndex = 0
	} else if channelIndex >= channelCount {
		channelIndex = channelCount - 1
	}
	channelOffset := voxOffset + channelIndex*channelPayloadBytes
	payload := append([]byte(nil), data[channelOffset:channelOffset+channelPayloadBytes]...)
	if bytesPerVoxel > 1 && order != binary.LittleEndian {
		normalizeScalarPayloadToLittleEndian(payload, bytesPerVoxel)
	}
	minValue, maxValue := niftiScalarRange(payload, dtype, bytesPerVoxel)
	warnings := []string{}
	if channelCount > 1 {
		warnings = append(warnings, "NIfTI fourth dimension is exposed as selectable scalar channels.")
	}
	return niftiScalarVolume{
		Width:         width,
		Height:        height,
		Depth:         depth,
		ChannelCount:  channelCount,
		ChannelIndex:  channelIndex,
		DType:         dtype,
		BytesPerVoxel: bytesPerVoxel,
		Data:          payload,
		RawMin:        minValue,
		RawMax:        maxValue,
		SpacingX:      niftiSpacing(order, data[80:84]),
		SpacingY:      niftiSpacing(order, data[84:88]),
		SpacingZ:      niftiSpacing(order, data[88:92]),
		Warnings:      warnings,
	}, nil
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

func niftiScalarDimsOrder(volume niftiScalarVolume) string {
	if volume.ChannelCount > 1 {
		return "CZYX"
	}
	return "ZYX"
}

func niftiScalarArrayShape(volume niftiScalarVolume) []int {
	if volume.ChannelCount > 1 {
		return []int{volume.ChannelCount, volume.Depth, volume.Height, volume.Width}
	}
	return []int{volume.Depth, volume.Height, volume.Width}
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

func readPossiblyGzippedFile(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if !strings.HasSuffix(strings.ToLower(path), ".gz") {
		return data, nil
	}
	reader, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = reader.Close()
	}()
	return io.ReadAll(reader)
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
	volume, err := loadNiftiScalarVolume(path, parseUploadScalarChannelIndex(r))
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
		switch volume.DType {
		case "uint8":
			return 0, 255
		case "int16":
			return math.MinInt16, math.MaxInt16
		case "uint16":
			return 0, math.MaxUint16
		}
	}
	if transform.WindowActive && transform.WindowMax > transform.WindowMin {
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
	case isTIFFUpload(originalName, contentType):
		return "image"
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
