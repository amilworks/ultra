package httpapi

import (
	"crypto/subtle"
	"errors"
	"net/http"
	"strings"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

type bisqueRequestAuthority struct {
	Worker      bool
	Run         domain.RunRecord
	Principal   requestPrincipal
	SessionID   string
	Credentials BisqueCredentials
}

// authorizeBisqueRequest leaves direct authenticated user actions on their
// existing principal boundary, but turns every worker call into a run-bound
// capability check. External HTTP cannot be part of the PostgreSQL lease
// transaction, so handlers must call this immediately before preflight and
// create a durable operation receipt before invoking BisQue.
func (deps ServerDeps) authorizeBisqueRequest(
	w http.ResponseWriter,
	r *http.Request,
	requiredIntents ...domain.RemoteMutationIntent,
) (bisqueRequestAuthority, bool) {
	switch deps.workerRequestAuth(r) {
	case workerAuthAbsent:
		principal := deps.principalFromRequest(r, "")
		return bisqueRequestAuthority{
			Principal:   principal,
			Credentials: deps.directBisqueCredentialsFromRequest(r.Context(), r, principal),
		}, true
	case workerAuthInvalid:
		writeError(w, http.StatusUnauthorized, errors.New("invalid worker token"))
		return bisqueRequestAuthority{}, false
	}

	if !isWorkerScopedEndpoint(r) {
		writeError(w, http.StatusUnauthorized, errors.New("worker credential is not authorized for this endpoint"))
		return bisqueRequestAuthority{}, false
	}
	runID := strings.TrimSpace(r.Header.Get("X-Ultra-Run-Id"))
	if runID == "" || deps.Store == nil {
		writeError(w, http.StatusUnauthorized, errors.New("active run lease required"))
		return bisqueRequestAuthority{}, false
	}
	run, err := deps.Store.GetRun(r.Context(), runID)
	if err != nil || run.Status != domain.RunStatusRunning {
		writeError(w, http.StatusUnauthorized, errors.New("active run lease required"))
		return bisqueRequestAuthority{}, false
	}
	orgID := trustedRunOrgID(run)
	principalMetadata, _ := jsonMapValue(run.Metadata["principal"])
	role, _ := safeMetadataString(principalMetadata["role"], 128)
	if role == "" {
		role, _ = safeMetadataString(run.Metadata["principal_role"], 128)
	}
	if strings.TrimSpace(run.UserID) == "" || orgID == "" || role == "" {
		writeError(w, http.StatusUnauthorized, errors.New("trusted run principal required"))
		return bisqueRequestAuthority{}, false
	}
	for _, required := range requiredIntents {
		if !domain.HasRemoteMutationIntent(run.Metadata, required) {
			writeError(w, http.StatusForbidden, errors.New("run is not authorized for this remote mutation"))
			return bisqueRequestAuthority{}, false
		}
	}

	lease, found, err := deps.Store.GetRunLease(r.Context(), runID)
	if err != nil || !found || !lease.LeaseExpiresAt.After(domain.Now()) {
		writeError(w, http.StatusUnauthorized, errors.New("active run lease required"))
		return bisqueRequestAuthority{}, false
	}
	requestWorkerID := strings.TrimSpace(r.Header.Get("X-Ultra-Worker-Id"))
	requestLeaseToken := strings.TrimSpace(r.Header.Get("X-Ultra-Run-Lease-Token"))
	if requestWorkerID == "" || requestWorkerID != lease.WorkerID || requestLeaseToken == "" ||
		subtle.ConstantTimeCompare([]byte(requestLeaseToken), []byte(lease.LeaseToken)) != 1 {
		writeError(w, http.StatusUnauthorized, errors.New("active run lease required"))
		return bisqueRequestAuthority{}, false
	}

	sessionID := strings.TrimSpace(r.Header.Get("X-Ultra-Bisque-Session-Id"))
	if sessionID == "" && deps.BisqueCredentials != nil {
		resolvedID, _, ok := deps.BisqueCredentials.ResolveLinkedSessionForUser(r.Context(), run.UserID, orgID)
		if ok {
			sessionID = resolvedID
		}
	}
	if deps.BisqueCredentials == nil || !bisqueSessionMatchesRunBinding(run, sessionID, deps.bisqueRootURL()) {
		writeError(w, http.StatusUnauthorized, errors.New("run-bound BisQue account required"))
		return bisqueRequestAuthority{}, false
	}
	ownerUserID, ownerOrgID, known, err := deps.BisqueCredentials.OwnerPrincipal(r.Context(), sessionID)
	if err != nil || !known || ownerUserID != run.UserID || ownerOrgID != orgID {
		writeError(w, http.StatusUnauthorized, errors.New("run-bound BisQue account required"))
		return bisqueRequestAuthority{}, false
	}
	credentials, found, err := deps.BisqueCredentials.GetWithContext(r.Context(), sessionID)
	if err != nil || !found || strings.TrimSpace(credentials.Username) == "" || strings.TrimSpace(credentials.Password) == "" {
		writeError(w, http.StatusUnauthorized, errors.New("run-bound BisQue account required"))
		return bisqueRequestAuthority{}, false
	}
	return bisqueRequestAuthority{
		Worker:      true,
		Run:         run,
		Principal:   requestPrincipal{UserID: run.UserID, OrgID: orgID, Role: role},
		SessionID:   sessionID,
		Credentials: credentials,
	}, true
}

func workerRunAllowsBisqueResource(run domain.RunRecord, resource domain.ResourceRecord) bool {
	return workerRunAllowsBisqueResourceBinding(run, resource.ResourceID, resource.Metadata)
}

func workerRunAllowsBisqueResourceRecord(run domain.RunRecord, resource resourceRecord) bool {
	return workerRunAllowsBisqueResourceBinding(run, resource.FileID, resource.Metadata)
}

func workerRunAllowsBisqueResourceBinding(run domain.RunRecord, id string, metadata domain.JSONMap) bool {
	resourceID := strings.TrimSpace(id)
	if resourceID == "" {
		return false
	}
	for _, selected := range metadataStringValues(run.Metadata["file_ids"]) {
		if strings.TrimSpace(selected) == resourceID {
			return true
		}
	}
	sourceRunID, _ := safeMetadataString(metadata["source_run_id"], 512)
	sourceAuthority, _ := safeMetadataString(metadata["source_authority"], 128)
	return sourceAuthority == "trusted_worker_run" && sourceRunID == run.RunID
}

func workerRunAllowsBisqueArtifact(run domain.RunRecord, artifact domain.ArtifactRecord) bool {
	if strings.TrimSpace(artifact.ArtifactID) == "" {
		return false
	}
	if artifact.RunID == run.RunID {
		return true
	}
	for _, descriptor := range metadataJSONMaps(run.Metadata["resource_descriptors"]) {
		artifactID, _ := safeMetadataString(descriptor["artifact_id"], 512)
		if artifactID == artifact.ArtifactID {
			return true
		}
	}
	return false
}

func metadataStringValues(value any) []string {
	values := []string{}
	switch typed := value.(type) {
	case []string:
		values = append(values, typed...)
	case []any:
		for _, item := range typed {
			if text, ok := item.(string); ok {
				values = append(values, text)
			}
		}
	}
	return values
}

func metadataJSONMaps(value any) []domain.JSONMap {
	values := []domain.JSONMap{}
	switch typed := value.(type) {
	case []domain.JSONMap:
		for _, item := range typed {
			values = append(values, item)
		}
	case []any:
		for _, item := range typed {
			if mapped, ok := jsonMapValue(item); ok {
				values = append(values, mapped)
			}
		}
	}
	return values
}
