package httpapi

// Reliability contract tests for the Google Drive picker integration, run
// against a fake Google (httptest): token exchange and refresh, encrypted
// at-rest storage, integrity-verified streaming imports, 429 retry with
// Retry-After, revocation handling, and refresh singleflight.

import (
	"context"
	"crypto/md5" // #nosec G501 — mirrors Drive's integrity checksum.
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

type fakeGoogle struct {
	server        *httptest.Server
	mu            sync.Mutex
	tokenCalls    int32
	mediaCalls    int32
	refreshBroken bool
	failFirstMedia int
	corruptMedia  bool
	fileBytes     []byte
	fileName      string
	fileMime      string
}

func newFakeGoogle(t *testing.T) *fakeGoogle {
	t.Helper()
	fake := &fakeGoogle{
		fileBytes: []byte("volumetric science bytes, definitely a TIFF"),
		fileName:  "activation_volume.tif",
		fileMime:  "image/tiff",
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fake.tokenCalls, 1)
		_ = r.ParseForm()
		grant := r.Form.Get("grant_type")
		if grant == "refresh_token" && fake.refreshBroken {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": "invalid_grant"})
			return
		}
		// A small artificial delay makes the singleflight test decisive: five
		// concurrent callers all arrive while the first refresh is in flight.
		time.Sleep(50 * time.Millisecond)
		idPayload, _ := json.Marshal(map[string]string{"email": "scientist@ucsb.edu"})
		idToken := "h." + base64.RawURLEncoding.EncodeToString(idPayload) + ".s"
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":  fmt.Sprintf("access-%d", atomic.LoadInt32(&fake.tokenCalls)),
			"refresh_token": "refresh-secret-1",
			"expires_in":    3600,
			"id_token":      idToken,
		})
	})
	mux.HandleFunc("/drive/v3/files/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("alt") == "media" {
			calls := atomic.AddInt32(&fake.mediaCalls, 1)
			fake.mu.Lock()
			failFirst := fake.failFirstMedia
			corrupt := fake.corruptMedia
			fake.mu.Unlock()
			if int(calls) <= failFirst {
				w.Header().Set("Retry-After", "1")
				w.WriteHeader(http.StatusTooManyRequests)
				return
			}
			body := fake.fileBytes
			if corrupt {
				body = []byte("corrupted payload from a flaky proxy")
			}
			w.Header().Set("Content-Type", fake.fileMime)
			_, _ = w.Write(body)
			return
		}
		sum := md5.Sum(fake.fileBytes) // #nosec G401
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":          strings.TrimPrefix(r.URL.Path, "/drive/v3/files/"),
			"name":        fake.fileName,
			"mimeType":    fake.fileMime,
			"size":        fmt.Sprintf("%d", len(fake.fileBytes)),
			"md5Checksum": hex.EncodeToString(sum[:]),
		})
	})
	fake.server = httptest.NewServer(mux)
	t.Cleanup(fake.server.Close)
	return fake
}

func newTestGoogleService(t *testing.T, fake *fakeGoogle) (*GoogleDriveService, *store.MemoryStore) {
	t.Helper()
	cipher, err := NewBisqueCredentialCipher([]byte("0123456789abcdef0123456789abcdef"), "test-key")
	if err != nil {
		t.Fatalf("cipher: %v", err)
	}
	mem := store.NewMemoryStore()
	service := NewGoogleDriveService(GoogleDriveConfig{
		ClientID:       "12345-test.apps.googleusercontent.com",
		ClientSecret:   "test-secret",
		RedirectURL:    "https://ultra.example.org/v2/integrations/google/callback",
		PickerAPIKey:   "picker-key",
		MaxImportBytes: 1 << 20,
		AuthBase:       fake.server.URL + "/auth",
		TokenURL:       fake.server.URL + "/token",
		APIBase:        fake.server.URL,
	}, mem, cipher)
	return service, mem
}

func connectTestUser(t *testing.T, service *GoogleDriveService) requestPrincipal {
	t.Helper()
	principal := requestPrincipal{UserID: "user-1", OrgID: "org-1"}
	if _, err := service.CompleteConnect(context.Background(), principal, "auth-code"); err != nil {
		t.Fatalf("connect: %v", err)
	}
	return principal
}

func TestGoogleConnectStoresEncryptedRefreshToken(t *testing.T) {
	fake := newFakeGoogle(t)
	service, mem := newTestGoogleService(t, fake)
	principal := connectTestUser(t, service)

	record, found, err := mem.GetGoogleCredentialForUser(context.Background(), principal.UserID)
	if err != nil || !found {
		t.Fatalf("expected stored credential, found=%v err=%v", found, err)
	}
	if record.RefreshTokenCiphertext == "refresh-secret-1" || strings.Contains(record.RefreshTokenCiphertext, "refresh-secret") {
		t.Fatalf("refresh token stored in plaintext")
	}
	decrypted, err := service.cipher.Decrypt(record.RefreshTokenCiphertext, record.RefreshTokenNonce)
	if err != nil || decrypted != "refresh-secret-1" {
		t.Fatalf("round-trip decrypt failed: %q %v", decrypted, err)
	}
	if record.AccountEmail != "scientist@ucsb.edu" {
		t.Fatalf("email not captured for display: %q", record.AccountEmail)
	}
	if record.Status != "active" {
		t.Fatalf("expected active status, got %q", record.Status)
	}
}

func TestGoogleStateTamperAndExpiry(t *testing.T) {
	fake := newFakeGoogle(t)
	service, _ := newTestGoogleService(t, fake)

	state := service.mintState("user-9")
	userID, err := service.verifyState(state)
	if err != nil || userID != "user-9" {
		t.Fatalf("round trip: %q %v", userID, err)
	}
	if _, err := service.verifyState(state + "x"); err == nil {
		t.Fatal("tampered signature accepted")
	}
	service.now = func() time.Time { return time.Now().Add(googleStateTTL + time.Minute) }
	if _, err := service.verifyState(state); err == nil {
		t.Fatal("expired state accepted")
	}
}

func TestGoogleImportStreamsAndVerifies(t *testing.T) {
	fake := newFakeGoogle(t)
	service, _ := newTestGoogleService(t, fake)
	principal := connectTestUser(t, service)

	result, err := service.ImportFile(context.Background(), t.TempDir(), principal, GoogleImportInput{FileID: "file-abc"})
	if err != nil {
		t.Fatalf("import: %v", err)
	}
	if result.Uploaded.OriginalName != "activation_volume.tif" {
		t.Fatalf("name: %q", result.Uploaded.OriginalName)
	}
	if result.Uploaded.SizeBytes != int64(len(fake.fileBytes)) {
		t.Fatalf("size: %d != %d", result.Uploaded.SizeBytes, len(fake.fileBytes))
	}
	if result.Uploaded.SHA256 == "" {
		t.Fatal("sha256 missing — dedup chips depend on it")
	}
	if result.Uploaded.SourceURI != "gdrive://file-abc" {
		t.Fatalf("source uri: %q", result.Uploaded.SourceURI)
	}
}

func TestGoogleImportRetriesRateLimit(t *testing.T) {
	fake := newFakeGoogle(t)
	fake.failFirstMedia = 1
	service, _ := newTestGoogleService(t, fake)
	principal := connectTestUser(t, service)

	if _, err := service.ImportFile(context.Background(), t.TempDir(), principal, GoogleImportInput{FileID: "file-retry"}); err != nil {
		t.Fatalf("import should survive one 429: %v", err)
	}
	if calls := atomic.LoadInt32(&fake.mediaCalls); calls != 2 {
		t.Fatalf("expected exactly 2 media attempts, got %d", calls)
	}
}

func TestGoogleImportRejectsCorruption(t *testing.T) {
	fake := newFakeGoogle(t)
	fake.corruptMedia = true
	service, _ := newTestGoogleService(t, fake)
	principal := connectTestUser(t, service)

	_, err := service.ImportFile(context.Background(), t.TempDir(), principal, GoogleImportInput{FileID: "file-bad"})
	if err == nil || !strings.Contains(err.Error(), "integrity") {
		t.Fatalf("corrupted download must fail integrity verification, got: %v", err)
	}
}

func TestGoogleImportRejectsNativeDocsAndFolders(t *testing.T) {
	fake := newFakeGoogle(t)
	service, _ := newTestGoogleService(t, fake)
	principal := connectTestUser(t, service)

	fake.fileMime = "application/vnd.google-apps.document"
	if _, err := service.ImportFile(context.Background(), t.TempDir(), principal, GoogleImportInput{FileID: "doc"}); err == nil ||
		!strings.Contains(err.Error(), "native Google document") {
		t.Fatalf("native doc should be rejected with guidance, got: %v", err)
	}
	fake.fileMime = "application/vnd.google-apps.folder"
	if _, err := service.ImportFile(context.Background(), t.TempDir(), principal, GoogleImportInput{FileID: "folder"}); err == nil ||
		!strings.Contains(err.Error(), "folders") {
		t.Fatalf("folder should be rejected with guidance, got: %v", err)
	}
}

func TestGoogleImportSizeCap(t *testing.T) {
	fake := newFakeGoogle(t)
	service, _ := newTestGoogleService(t, fake)
	service.cfg.MaxImportBytes = 8
	principal := connectTestUser(t, service)

	_, err := service.ImportFile(context.Background(), t.TempDir(), principal, GoogleImportInput{FileID: "big"})
	if err == nil || !strings.Contains(err.Error(), "limit") {
		t.Fatalf("oversized file should be rejected before download, got: %v", err)
	}
	if atomic.LoadInt32(&fake.mediaCalls) != 0 {
		t.Fatal("oversized file must not be downloaded at all")
	}
}

func TestGoogleRevokedGrantMarksBrokenAndSurfacesReconnect(t *testing.T) {
	fake := newFakeGoogle(t)
	service, mem := newTestGoogleService(t, fake)
	principal := connectTestUser(t, service)
	service.dropCachedToken(principal.UserID)
	fake.refreshBroken = true

	_, _, err := service.AccessToken(context.Background(), principal.UserID)
	if err == nil || !strings.Contains(err.Error(), "reconnect") {
		t.Fatalf("expected reconnect-required, got: %v", err)
	}
	record, _, _ := mem.GetGoogleCredentialForUser(context.Background(), principal.UserID)
	if record.Status != "broken" {
		t.Fatalf("credential should be marked broken, got %q", record.Status)
	}
}

func TestGoogleRefreshSingleflight(t *testing.T) {
	fake := newFakeGoogle(t)
	service, _ := newTestGoogleService(t, fake)
	principal := connectTestUser(t, service)
	service.dropCachedToken(principal.UserID)
	baseline := atomic.LoadInt32(&fake.tokenCalls)

	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _, _ = service.AccessToken(context.Background(), principal.UserID)
		}()
	}
	wg.Wait()
	if refreshes := atomic.LoadInt32(&fake.tokenCalls) - baseline; refreshes != 1 {
		t.Fatalf("5 concurrent callers must share ONE refresh, got %d", refreshes)
	}
}

func TestGoogleDisconnectDeletesCredential(t *testing.T) {
	fake := newFakeGoogle(t)
	service, mem := newTestGoogleService(t, fake)
	principal := connectTestUser(t, service)

	if err := service.Disconnect(context.Background(), principal.UserID); err != nil {
		t.Fatalf("disconnect: %v", err)
	}
	if _, found, _ := mem.GetGoogleCredentialForUser(context.Background(), principal.UserID); found {
		t.Fatal("credential should be gone after disconnect")
	}
}
