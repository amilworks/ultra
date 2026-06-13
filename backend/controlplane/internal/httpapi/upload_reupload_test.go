package httpapi

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func uploadSessionInputForTest(user, org, key string) domain.CreateUploadSessionInput {
	return domain.CreateUploadSessionInput{
		OwnerUserID:    user,
		OwnerOrgID:     org,
		Status:         "active",
		TotalBytes:     10,
		IdempotencyKey: key,
	}
}

func reuploadTestRouter(t *testing.T) (*store.MemoryStore, http.Handler) {
	t.Helper()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: t.TempDir(),
	})
	return mem, router
}

// TestReuploadAfterDeleteStartsFreshSession is the reported bug: a scientist re-uploads a
// file they previously uploaded and then deleted (EnrNE_d2_r1.tif). The prior session is
// "completed" with the same content idempotency key; the fix must NOT replay that dead
// session (which 409'd "upload session is completed") but start a FRESH upload that
// succeeds and mints a new resource.
func TestReuploadAfterDeleteStartsFreshSession(t *testing.T) {
	t.Parallel()
	mem, router := reuploadTestRouter(t)
	const user, org = "u", "o"
	payload := []byte("orthomosaic bytes that get re-uploaded")
	sum := sha256.Sum256(payload)
	sha := hex.EncodeToString(sum[:])
	body := uploadSessionCreateBody("reupload-del", "f1", "EnrNE_d2_r1.tif", "image/tiff", payload, sha)

	created := createUploadSessionForTest(t, router, body, user, org, http.StatusCreated)
	uploadChunkForTest(t, router, created.Session.SessionID, "f1", 0, 0, payload, sha, user, org)
	done := completeUploadSessionFileForTest(t, router, created.Session.SessionID, "f1", user, org, http.StatusOK)
	resID := done.Resource.FileID
	if resID == "" {
		t.Fatal("first upload produced no resource")
	}

	// The scientist deletes the resource (soft-delete), then re-uploads the same file.
	if _, err := mem.SoftDeleteResourceForUser(context.Background(), resID, user, org, time.Now()); err != nil {
		t.Fatalf("soft delete: %v", err)
	}

	// Re-create must NOT 409 / replay the dead completed session — it must return a FRESH
	// active session (201), because the committed resource is gone.
	recreated := createUploadSessionForTest(t, router, body, user, org, http.StatusCreated)
	if recreated.Session.SessionID == created.Session.SessionID {
		t.Fatalf("re-upload replayed the dead session %s; want a fresh session", created.Session.SessionID)
	}
	if recreated.Session.Status != "active" {
		t.Fatalf("re-upload session status = %s, want active", recreated.Session.Status)
	}

	// The fresh upload completes with no 409 and mints a NEW resource.
	uploadChunkForTest(t, router, recreated.Session.SessionID, "f1", 0, 0, payload, sha, user, org)
	done2 := completeUploadSessionFileForTest(t, router, recreated.Session.SessionID, "f1", user, org, http.StatusOK)
	if done2.Resource.FileID == "" {
		t.Fatal("re-upload produced no resource")
	}
	if done2.Resource.FileID == resID {
		t.Fatalf("re-upload reused the deleted resource %s; want a new resource", resID)
	}
}

// TestReuploadOfActiveFileDedups: re-uploading a file whose resource is still ACTIVE
// replays the completed session (idempotent, no re-upload) and /complete returns the SAME
// resource via dedup — exactly one resource, no 409.
func TestReuploadOfActiveFileDedups(t *testing.T) {
	t.Parallel()
	mem, router := reuploadTestRouter(t)
	const user, org = "u", "o"
	payload := []byte("a file uploaded twice while still active")
	sum := sha256.Sum256(payload)
	sha := hex.EncodeToString(sum[:])
	body := uploadSessionCreateBody("reupload-active", "f1", "scan.tif", "image/tiff", payload, sha)

	created := createUploadSessionForTest(t, router, body, user, org, http.StatusCreated)
	uploadChunkForTest(t, router, created.Session.SessionID, "f1", 0, 0, payload, sha, user, org)
	done := completeUploadSessionFileForTest(t, router, created.Session.SessionID, "f1", user, org, http.StatusOK)
	resID := done.Resource.FileID

	// Resource is active -> re-create replays the SAME completed session (HTTP 200).
	recreated := createUploadSessionForTest(t, router, body, user, org, http.StatusOK)
	if recreated.Session.SessionID != created.Session.SessionID {
		t.Fatalf("re-upload of an active file should replay the completed session; got %s want %s", recreated.Session.SessionID, created.Session.SessionID)
	}
	if recreated.Session.Status != "completed" {
		t.Fatalf("replayed session status = %s, want completed", recreated.Session.Status)
	}

	// The frontend's completed-session fast path calls /complete, which now returns the
	// committed resource (200) instead of 409 (the SECONDARY reorder).
	redone := completeUploadSessionFileForTest(t, router, recreated.Session.SessionID, "f1", user, org, http.StatusOK)
	if redone.Resource.FileID != resID {
		t.Fatalf("re-complete returned resource %s, want the original %s (dedup)", redone.Resource.FileID, resID)
	}

	records, err := mem.ListResources(context.Background(), 100, 0)
	if err != nil {
		t.Fatalf("ListResources: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("re-upload of an active file created %d resources, want exactly 1", len(records))
	}
}

// TestClearUploadSessionIdempotencyKeyFreesTheSlot: after clearing, the key no longer
// resolves a session and a new session can take the key.
func TestClearUploadSessionIdempotencyKeyFreesTheSlot(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	ctx := context.Background()
	s1, err := mem.CreateUploadSession(ctx, uploadSessionInputForTest("u", "o", "shared-key"))
	if err != nil {
		t.Fatalf("create s1: %v", err)
	}
	if got, err := mem.GetUploadSessionByIdempotencyKeyForUser(ctx, "shared-key", "u", "o"); err != nil || got.SessionID != s1.SessionID {
		t.Fatalf("lookup before clear = %+v err=%v, want %s", got, err, s1.SessionID)
	}
	if err := mem.ClearUploadSessionIdempotencyKey(ctx, s1.SessionID); err != nil {
		t.Fatalf("clear: %v", err)
	}
	if _, err := mem.GetUploadSessionByIdempotencyKeyForUser(ctx, "shared-key", "u", "o"); err == nil {
		t.Fatal("lookup after clear should miss, but found a session")
	}
	// The freed key can now back a new session.
	s2, err := mem.CreateUploadSession(ctx, uploadSessionInputForTest("u", "o", "shared-key"))
	if err != nil {
		t.Fatalf("create s2 with freed key: %v", err)
	}
	if got, err := mem.GetUploadSessionByIdempotencyKeyForUser(ctx, "shared-key", "u", "o"); err != nil || got.SessionID != s2.SessionID {
		t.Fatalf("lookup after re-create = %+v err=%v, want %s", got, err, s2.SessionID)
	}
}
