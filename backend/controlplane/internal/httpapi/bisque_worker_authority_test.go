package httpapi

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func testRunAuthorityMetadata(
	userID string,
	orgID string,
	rootURL string,
	sessionID string,
	intents ...domain.RemoteMutationIntent,
) domain.JSONMap {
	metadata := domain.JSONMap{
		"principal": domain.JSONMap{
			"user_id": userID,
			"org_id":  orgID,
			"role":    "researcher",
		},
		"principal_user_id": userID,
		"org_id":            orgID,
		"principal_role":    "researcher",
	}
	if sessionID != "" {
		metadata[domain.BisqueAccountBindingMetadataKey] = domain.JSONMap{
			"schema_version": "ultra.bisque_account_binding.v1",
			"authority":      "control_plane",
			"session_sha256": bisqueSessionDigest(sessionID),
			"root_url":       rootURL,
			"owner_user_id":  userID,
			"owner_org_id":   orgID,
		}
	}
	if len(intents) > 0 {
		metadata[domain.RemoteMutationIntentsMetadataKey] = domain.RemoteMutationIntentStrings(intents)
	}
	return metadata
}

func startTestWorkerRunLease(t *testing.T, memory *store.MemoryStore, runID string, workerID string) domain.RunLeaseRecord {
	t.Helper()
	if _, err := memory.UpdateRunStatus(t.Context(), runID, domain.RunStatusRunning, "worker running", ""); err != nil {
		t.Fatalf("UpdateRunStatus: %v", err)
	}
	lease, err := memory.AcquireRunLease(t.Context(), domain.AcquireRunLeaseInput{
		RunID: runID, WorkerID: workerID, TTL: time.Minute,
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}
	return lease
}

func setTestWorkerRunHeaders(req *http.Request, runID string, workerID string, leaseToken string, sessionID string) {
	req.Header.Set("X-Ultra-Worker-Token", "worker-secret")
	req.Header.Set("X-Ultra-Run-Id", runID)
	req.Header.Set("X-Ultra-Worker-Id", workerID)
	req.Header.Set("X-Ultra-Run-Lease-Token", leaseToken)
	if sessionID != "" {
		req.Header.Set("X-Ultra-Bisque-Session-Id", sessionID)
	}
}

func TestWorkerBisqueMutationRequiresRunCapabilityAndReplaysReceipt(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	bisque, state := newFakeBisqueUploadServer(t)
	memory := store.NewMemoryStore()
	cipher, err := NewBisqueCredentialCipher(bytes.Repeat([]byte{8}, 32), "mutation-test-key")
	if err != nil {
		t.Fatalf("NewBisqueCredentialCipher: %v", err)
	}
	credentials := NewPersistentBisqueCredentialStore(memory, cipher, bisque.URL)
	sessionID, err := credentials.PutLinked(ctx, BisqueCredentialLinkInput{
		Credentials: BisqueCredentials{Username: "scientist", Password: "linked-secret"},
		UserID:      "scientist-1",
		OrgID:       "materials-lab",
		RootURL:     bisque.URL,
	})
	if err != nil {
		t.Fatalf("PutLinked: %v", err)
	}
	uploadRoot := t.TempDir()
	selectedID := "file_selected"
	selectedName := "selected-result.txt"
	selectedContent := []byte("run-selected scientific output\n")
	selectedPath := filepath.Join(uploadRoot, selectedID+"__"+selectedName)
	if err := os.WriteFile(selectedPath, selectedContent, 0o600); err != nil {
		t.Fatalf("write selected resource: %v", err)
	}
	if err := writeUploadMetadataRecord(uploadRoot, selectedID, uploadMetadataRecord{Principal: principalRecord{
		UserID: "scientist-1", OrgID: "materials-lab", Role: "researcher",
	}}); err != nil {
		t.Fatalf("write selected resource metadata: %v", err)
	}
	selectedDigest := sha256.Sum256(selectedContent)
	if _, err := memory.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: selectedID, OriginalName: selectedName, ContentType: "text/plain",
		SizeBytes: int64(len(selectedContent)), SHA256: hex.EncodeToString(selectedDigest[:]),
		StoragePath: filepath.Base(selectedPath), SourceType: "upload", ResourceKind: "document",
		OwnerUserID: "scientist-1", OwnerOrgID: "materials-lab", OwnerRole: "researcher",
		Status: "active", CreatedAt: domain.Now(), UpdatedAt: domain.Now(), Metadata: domain.JSONMap{},
	}); err != nil {
		t.Fatalf("UpsertResource selected: %v", err)
	}
	unselectedID := "file_unselected"
	unselectedName := "unselected-secret.txt"
	unselectedContent := []byte("unselected owned content\n")
	unselectedPath := filepath.Join(uploadRoot, unselectedID+"__"+unselectedName)
	if err := os.WriteFile(unselectedPath, unselectedContent, 0o600); err != nil {
		t.Fatalf("write unselected resource: %v", err)
	}
	if err := writeUploadMetadataRecord(uploadRoot, unselectedID, uploadMetadataRecord{Principal: principalRecord{
		UserID: "scientist-1", OrgID: "materials-lab", Role: "researcher",
	}}); err != nil {
		t.Fatalf("write unselected resource metadata: %v", err)
	}
	unselectedDigest := sha256.Sum256(unselectedContent)
	if _, err := memory.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID: unselectedID, OriginalName: unselectedName, ContentType: "text/plain",
		SizeBytes: int64(len(unselectedContent)), SHA256: hex.EncodeToString(unselectedDigest[:]),
		StoragePath: filepath.Base(unselectedPath), SourceType: "upload", ResourceKind: "document",
		OwnerUserID: "scientist-1", OwnerOrgID: "materials-lab", OwnerRole: "researcher",
		Status: "active", CreatedAt: domain.Now(), UpdatedAt: domain.Now(), Metadata: domain.JSONMap{},
	}); err != nil {
		t.Fatalf("UpsertResource unselected: %v", err)
	}

	thread, err := memory.CreateThread(ctx, domain.CreateThreadInput{UserID: "scientist-1", Title: "remote mutation"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	metadata := testRunAuthorityMetadata(
		"scientist-1", "materials-lab", bisque.URL, sessionID,
		domain.RemoteMutationIntentBisqueUpload,
		domain.RemoteMutationIntentBisqueCreateDataset,
	)
	metadata["file_ids"] = []string{selectedID}
	run, err := memory.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: thread.ThreadID, UserID: "scientist-1", Goal: "publish selected output",
		Metadata: metadata,
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	lease := startTestWorkerRunLease(t, memory, run.RunID, "worker-a")
	router := NewRouter(ServerDeps{
		Version: "test-version", Store: memory, UploadRoot: uploadRoot, ArtifactRoot: t.TempDir(),
		WorkerToken: "worker-secret", BisqueCredentials: credentials,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL: bisque.URL, AllowedRoots: []string{bisque.URL}, HTTPClient: bisque.Client(),
			UploadRoot: uploadRoot, MaxImportSize: 8 << 20,
		}),
	})
	post := func(path string, body string, workerID string, leaseToken string, session string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		setTestWorkerRunHeaders(req, run.RunID, workerID, leaseToken, session)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}

	for name, authority := range map[string][2]string{
		"missing lease": {"", ""},
		"wrong worker":  {"worker-b", lease.LeaseToken},
		"wrong token":   {"worker-a", lease.LeaseToken + "-forged"},
	} {
		t.Run(name, func(t *testing.T) {
			rec := post("/v2/bisque/upload", `{"file_ids":["`+selectedID+`"]}`, authority[0], authority[1], sessionID)
			if rec.Code != http.StatusUnauthorized {
				t.Fatalf("status = %d body=%s, want 401", rec.Code, rec.Body.String())
			}
		})
	}
	if rec := post("/v2/bisque/upload", `{"file_ids":["`+selectedID+`"]}`, "worker-a", lease.LeaseToken, "forged-session"); rec.Code != http.StatusUnauthorized {
		t.Fatalf("forged session status = %d body=%s, want 401", rec.Code, rec.Body.String())
	}

	valid := post("/v2/bisque/upload", `{"file_ids":["`+selectedID+`"]}`, "worker-a", lease.LeaseToken, sessionID)
	if valid.Code != http.StatusOK {
		t.Fatalf("valid upload status = %d body=%s", valid.Code, valid.Body.String())
	}
	var upload bisqueUploadResponse
	if err := json.Unmarshal(valid.Body.Bytes(), &upload); err != nil || len(upload.Uploads) != 1 {
		t.Fatalf("decode upload response: %+v err=%v", upload, err)
	}
	replayed := post("/v2/bisque/upload", `{"file_ids":["`+selectedID+`"]}`, "worker-a", lease.LeaseToken, sessionID)
	if replayed.Code != http.StatusOK {
		t.Fatalf("replayed upload status = %d body=%s", replayed.Code, replayed.Body.String())
	}
	state.mu.Lock()
	if state.uploadCount != 1 {
		t.Fatalf("upstream upload count = %d, want exactly one", state.uploadCount)
	}
	state.mu.Unlock()

	if rec := post("/v2/bisque/upload", `{"file_ids":["`+unselectedID+`"]}`, "worker-a", lease.LeaseToken, sessionID); rec.Code != http.StatusNotFound {
		t.Fatalf("unselected owned resource status = %d body=%s, want 404", rec.Code, rec.Body.String())
	}

	resourceURI := upload.Uploads[0].ResourceURI
	datasetBody, err := json.Marshal(bisqueCreateDatasetRequest{
		Name: "Selected outputs", ResourceURIs: []string{resourceURI},
	})
	if err != nil {
		t.Fatalf("marshal dataset request: %v", err)
	}
	dataset := post("/v2/bisque/datasets", string(datasetBody), "worker-a", lease.LeaseToken, sessionID)
	if dataset.Code != http.StatusOK {
		t.Fatalf("dataset status = %d body=%s", dataset.Code, dataset.Body.String())
	}
	datasetReplay := post("/v2/bisque/datasets", string(datasetBody), "worker-a", lease.LeaseToken, sessionID)
	if datasetReplay.Code != http.StatusOK {
		t.Fatalf("dataset replay status = %d body=%s", datasetReplay.Code, datasetReplay.Body.String())
	}
	state.mu.Lock()
	if len(state.datasetXML) != 1 {
		t.Fatalf("upstream dataset count = %d, want exactly one", len(state.datasetXML))
	}
	state.mu.Unlock()

	arbitrary := post(
		"/v2/bisque/datasets",
		`{"name":"arbitrary","resource_uris":["`+bisque.URL+`/data_service/00-OTHER"]}`,
		"worker-a", lease.LeaseToken, sessionID,
	)
	if arbitrary.Code != http.StatusNotFound {
		t.Fatalf("arbitrary dataset member status = %d body=%s, want 404", arbitrary.Code, arbitrary.Body.String())
	}

	if err := memory.ReleaseRunLease(ctx, domain.ReleaseRunLeaseInput{RunID: run.RunID, LeaseToken: lease.LeaseToken}); err != nil {
		t.Fatalf("ReleaseRunLease: %v", err)
	}
	released := post("/v2/bisque/upload", `{"file_ids":["`+selectedID+`"]}`, "worker-a", lease.LeaseToken, sessionID)
	if released.Code != http.StatusUnauthorized {
		t.Fatalf("released lease status = %d body=%s, want 401", released.Code, released.Body.String())
	}
}
