package httpapi

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/jackc/pgx/v5/pgxpool"
)

type fakeQueueDiagnosticsProvider struct {
	diagnostics eventbus.QueueDiagnostics
	err         error
}

func (p fakeQueueDiagnosticsProvider) QueueDiagnostics(context.Context) (eventbus.QueueDiagnostics, error) {
	return p.diagnostics, p.err
}

type recordingDataAgentJobPublisher struct {
	jobs []eventbus.DataAgentJob
	err  error
}

func (p *recordingDataAgentJobPublisher) PublishDataAgentJob(ctx context.Context, job eventbus.DataAgentJob) error {
	_ = ctx
	if p.err != nil {
		return p.err
	}
	p.jobs = append(p.jobs, job)
	return nil
}

func TestHealthAndPublicConfig(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{
		Version: "test-version",
	})

	healthReq := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	healthRec := httptest.NewRecorder()
	router.ServeHTTP(healthRec, healthReq)

	if healthRec.Code != http.StatusOK {
		t.Fatalf("health status = %d, want 200", healthRec.Code)
	}
	var health map[string]string
	if err := json.Unmarshal(healthRec.Body.Bytes(), &health); err != nil {
		t.Fatalf("decode health: %v", err)
	}
	if health["status"] != "ok" {
		t.Fatalf("health status body = %q, want ok", health["status"])
	}
	if health["ts"] == "" {
		t.Fatalf("health response must include ts")
	}

	configReq := httptest.NewRequest(http.MethodGet, "/v1/config/public", nil)
	configRec := httptest.NewRecorder()
	router.ServeHTTP(configRec, configReq)

	if configRec.Code != http.StatusOK {
		t.Fatalf("config status = %d, want 200", configRec.Code)
	}
	var config map[string]any
	if err := json.Unmarshal(configRec.Body.Bytes(), &config); err != nil {
		t.Fatalf("decode config: %v", err)
	}
	if config["app_version"] != "test-version" {
		t.Fatalf("app_version = %v, want test-version", config["app_version"])
	}
	if config["admin_enabled"] != false {
		t.Fatalf("admin_enabled = %v, want false without explicit local admin deps", config["admin_enabled"])
	}
}

func TestPublicConfigIncludesBisqueProductionLinks(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL: "https://bisque2.ece.ucsb.edu",
		}),
	})

	for _, path := range []string{"/v1/config/public", "/v2/config/public"} {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, path, nil)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("config status = %d, want 200 body=%s", rec.Code, rec.Body.String())
			}
			var config map[string]any
			if err := json.Unmarshal(rec.Body.Bytes(), &config); err != nil {
				t.Fatalf("decode config: %v", err)
			}
			if config["bisque_root"] != "https://bisque2.ece.ucsb.edu" {
				t.Fatalf("bisque_root = %#v", config["bisque_root"])
			}
			if config["bisque_browser_url"] != "https://bisque2.ece.ucsb.edu/client_service/" {
				t.Fatalf("bisque_browser_url = %#v", config["bisque_browser_url"])
			}
			links, ok := config["bisque_urls"].(map[string]any)
			if !ok {
				t.Fatalf("bisque_urls = %#v, want object", config["bisque_urls"])
			}
			if links["home"] != "https://bisque2.ece.ucsb.edu/client_service/" {
				t.Fatalf("home link = %#v", links["home"])
			}
			if links["images"] != "https://bisque2.ece.ucsb.edu/client_service/browser?resource=/data_service/image" {
				t.Fatalf("images link = %#v", links["images"])
			}
			if links["datasets"] != "https://bisque2.ece.ucsb.edu/client_service/browser?resource=/data_service/dataset" {
				t.Fatalf("datasets link = %#v", links["datasets"])
			}
			if links["tables"] != "https://bisque2.ece.ucsb.edu/client_service/browser?resource=/data_service/table" {
				t.Fatalf("tables link = %#v", links["tables"])
			}
		})
	}
}

func TestDecodeJSONRejectsTrailingValuesAndOversizedBodies(t *testing.T) {
	t.Run("trailing json value", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/test", strings.NewReader(`{"title":"ok"} {"title":"smuggled"}`))
		rec := httptest.NewRecorder()
		var target struct {
			Title string `json:"title"`
		}

		if decodeJSON(rec, req, &target) {
			t.Fatalf("decodeJSON accepted multiple JSON values")
		}
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400 body=%s", rec.Code, rec.Body.String())
		}
	})

	t.Run("oversized json body", func(t *testing.T) {
		body := `{"title":"` + strings.Repeat("x", 17<<20) + `"}`
		req := httptest.NewRequest(http.MethodPost, "/test", strings.NewReader(body))
		rec := httptest.NewRecorder()
		var target struct {
			Title string `json:"title"`
		}

		if decodeJSON(rec, req, &target) {
			t.Fatalf("decodeJSON accepted oversized JSON body")
		}
		if rec.Code != http.StatusRequestEntityTooLarge {
			t.Fatalf("status = %d, want 413 body=%s", rec.Code, rec.Body.String())
		}
	})
}

func TestDevAuthGuestSessionLifecycle(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{Version: "test-version", DevAdminEnabled: true})

	sessionReq := httptest.NewRequest(http.MethodGet, "/v1/auth/session", nil)
	sessionRec := httptest.NewRecorder()
	router.ServeHTTP(sessionRec, sessionReq)
	if sessionRec.Code != http.StatusOK {
		t.Fatalf("default session status = %d body=%s", sessionRec.Code, sessionRec.Body.String())
	}
	var defaultSession map[string]any
	if err := json.Unmarshal(sessionRec.Body.Bytes(), &defaultSession); err != nil {
		t.Fatalf("decode default session: %v", err)
	}
	if defaultSession["authenticated"] != true || defaultSession["mode"] != "guest" {
		t.Fatalf("default session = %#v, want local guest session", defaultSession)
	}
	if defaultSession["is_admin"] != true {
		t.Fatalf("default session is_admin = %#v, want local dev admin access", defaultSession["is_admin"])
	}
	user, ok := defaultSession["user"].(map[string]any)
	if !ok || user["role"] != "admin" {
		t.Fatalf("default session user = %#v, want admin role", defaultSession["user"])
	}

	guestBody := strings.NewReader(`{"name":"Ada Lovelace","email":"ada@example.org","affiliation":"Analytical Engine Lab"}`)
	guestReq := httptest.NewRequest(http.MethodPost, "/v1/auth/guest", guestBody)
	guestReq.Header.Set("Content-Type", "application/json")
	guestRec := httptest.NewRecorder()
	router.ServeHTTP(guestRec, guestReq)
	if guestRec.Code != http.StatusOK {
		t.Fatalf("guest auth status = %d body=%s", guestRec.Code, guestRec.Body.String())
	}
	var guestSession map[string]any
	if err := json.Unmarshal(guestRec.Body.Bytes(), &guestSession); err != nil {
		t.Fatalf("decode guest session: %v", err)
	}
	if guestSession["authenticated"] != true || guestSession["username"] != "Ada Lovelace" {
		t.Fatalf("guest session = %#v, want authenticated Ada", guestSession)
	}
	if guestSession["is_admin"] != true {
		t.Fatalf("guest session is_admin = %#v, want local dev admin access", guestSession["is_admin"])
	}
	if len(guestRec.Result().Cookies()) == 0 {
		t.Fatalf("guest auth should set a dev session cookie")
	}
	cookieSessionReq := httptest.NewRequest(http.MethodGet, "/v1/auth/session", nil)
	cookieSessionReq.AddCookie(guestRec.Result().Cookies()[0])
	cookieSessionRec := httptest.NewRecorder()
	router.ServeHTTP(cookieSessionRec, cookieSessionReq)
	var cookieSession map[string]any
	if err := json.Unmarshal(cookieSessionRec.Body.Bytes(), &cookieSession); err != nil {
		t.Fatalf("decode cookie session: %v", err)
	}
	if cookieSession["username"] != "Ada Lovelace" || cookieSession["mode"] != "guest" {
		t.Fatalf("cookie session = %#v, want persisted guest", cookieSession)
	}
	if cookieSession["is_admin"] != true {
		t.Fatalf("cookie session is_admin = %#v, want local dev admin access", cookieSession["is_admin"])
	}

	logoutReq := httptest.NewRequest(http.MethodPost, "/v1/auth/logout", nil)
	logoutRec := httptest.NewRecorder()
	router.ServeHTTP(logoutRec, logoutReq)
	if logoutRec.Code != http.StatusOK {
		t.Fatalf("logout status = %d body=%s", logoutRec.Code, logoutRec.Body.String())
	}
	var logoutSession map[string]any
	if err := json.Unmarshal(logoutRec.Body.Bytes(), &logoutSession); err != nil {
		t.Fatalf("decode logout session: %v", err)
	}
	if logoutSession["authenticated"] != false {
		t.Fatalf("logout session = %#v, want unauthenticated response", logoutSession)
	}
}

func TestDevAuthSessionCanDisableLocalAdmin(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{Version: "test-version", DevAdminEnabled: false})

	sessionReq := httptest.NewRequest(http.MethodGet, "/v2/auth/session", nil)
	sessionRec := httptest.NewRecorder()
	router.ServeHTTP(sessionRec, sessionReq)
	if sessionRec.Code != http.StatusOK {
		t.Fatalf("session status = %d body=%s", sessionRec.Code, sessionRec.Body.String())
	}
	var session map[string]any
	if err := json.Unmarshal(sessionRec.Body.Bytes(), &session); err != nil {
		t.Fatalf("decode session: %v", err)
	}
	if session["is_admin"] != false {
		t.Fatalf("session is_admin = %#v, want disabled local admin", session["is_admin"])
	}
	user, ok := session["user"].(map[string]any)
	if !ok || user["role"] != "researcher" {
		t.Fatalf("session user = %#v, want researcher role", session["user"])
	}
}

func TestBisqueCredentialBackedSessionReportsLinkedStatus(t *testing.T) {
	t.Parallel()

	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		if r.URL.Path == "/auth_service/session" {
			_, _ = w.Write([]byte(`<response><tag name="user" value="amil"/></response>`))
			return
		}
		_, _ = w.Write([]byte(`<response><image uri="` + bisque.URL + `/data_service/image/linked" name="linked.jpg" resource_uniq="linked"/></response>`))
	}))
	defer bisque.Close()

	credentialStore := NewBisqueCredentialStore()
	router := NewRouter(ServerDeps{
		Version:           "test-version",
		DevAdminEnabled:   true,
		BisqueCredentials: credentialStore,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	loginReq := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"amil","password":"secret"}`))
	loginReq.Header.Set("Content-Type", "application/json")
	loginRec := httptest.NewRecorder()
	router.ServeHTTP(loginRec, loginReq)
	if loginRec.Code != http.StatusOK {
		t.Fatalf("login status = %d body=%s", loginRec.Code, loginRec.Body.String())
	}
	var loginSession map[string]any
	if err := json.Unmarshal(loginRec.Body.Bytes(), &loginSession); err != nil {
		t.Fatalf("decode login session: %v", err)
	}
	if loginSession["bisque_linked"] != true {
		t.Fatalf("login bisque_linked = %#v, want true for credential-backed session", loginSession["bisque_linked"])
	}
	if user, ok := loginSession["user"].(map[string]any); !ok || user["id"] != "bisque:amil" {
		t.Fatalf("login session user = %#v, want bisque:amil", loginSession["user"])
	}
	if len(loginRec.Result().Cookies()) == 0 {
		t.Fatalf("login should set a credential-backed dev session cookie")
	}

	sessionReq := httptest.NewRequest(http.MethodGet, "/v2/auth/session", nil)
	sessionReq.AddCookie(loginRec.Result().Cookies()[0])
	sessionRec := httptest.NewRecorder()
	router.ServeHTTP(sessionRec, sessionReq)
	if sessionRec.Code != http.StatusOK {
		t.Fatalf("session status = %d body=%s", sessionRec.Code, sessionRec.Body.String())
	}
	var session map[string]any
	if err := json.Unmarshal(sessionRec.Body.Bytes(), &session); err != nil {
		t.Fatalf("decode session: %v", err)
	}
	if session["bisque_linked"] != true {
		t.Fatalf("session bisque_linked = %#v, want true for credential-backed session", session["bisque_linked"])
	}
	if user, ok := session["user"].(map[string]any); !ok || user["id"] != "bisque:amil" {
		t.Fatalf("session user = %#v, want credential-backed bisque principal", session["user"])
	}
}

func TestLegacyBisqueIdentityCookieIsNotCredentialLinked(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{Version: "test-version", DevAdminEnabled: true})

	sessionReq := httptest.NewRequest(http.MethodGet, "/v2/auth/session", nil)
	sessionReq.AddCookie(&http.Cookie{Name: "ultra_dev_auth", Value: "bisque:amil"})
	sessionRec := httptest.NewRecorder()
	router.ServeHTTP(sessionRec, sessionReq)
	if sessionRec.Code != http.StatusOK {
		t.Fatalf("session status = %d body=%s", sessionRec.Code, sessionRec.Body.String())
	}
	var session map[string]any
	if err := json.Unmarshal(sessionRec.Body.Bytes(), &session); err != nil {
		t.Fatalf("decode session: %v", err)
	}
	if session["authenticated"] != true || session["mode"] != "bisque" {
		t.Fatalf("session = %#v, want local BisQue identity session", session)
	}
	if session["bisque_linked"] != false {
		t.Fatalf("session bisque_linked = %#v, want false for legacy identity-only cookie", session["bisque_linked"])
	}
	if user, ok := session["user"].(map[string]any); !ok || user["id"] != "bisque:amil" {
		t.Fatalf("legacy BisQue session user = %#v, want bisque:amil", session["user"])
	}
}

func TestV2HealthConfigAndAuthAliases(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{Version: "test-version", DevAdminEnabled: true})

	for _, path := range []string{"/v2/health", "/v2/config/public", "/v2/auth/session"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s status = %d body=%s", path, rec.Code, rec.Body.String())
		}
	}

	guestReq := httptest.NewRequest(http.MethodPost, "/v2/auth/guest", strings.NewReader(`{"name":"Grace Hopper"}`))
	guestReq.Header.Set("Content-Type", "application/json")
	guestRec := httptest.NewRecorder()
	router.ServeHTTP(guestRec, guestReq)
	if guestRec.Code != http.StatusOK {
		t.Fatalf("guest alias status = %d body=%s", guestRec.Code, guestRec.Body.String())
	}

	loginReq := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"local-user"}`))
	loginReq.Header.Set("Content-Type", "application/json")
	loginRec := httptest.NewRecorder()
	router.ServeHTTP(loginRec, loginReq)
	if loginRec.Code != http.StatusOK {
		t.Fatalf("login alias status = %d body=%s", loginRec.Code, loginRec.Body.String())
	}

	logoutReq := httptest.NewRequest(http.MethodPost, "/v2/auth/logout", nil)
	logoutRec := httptest.NewRecorder()
	router.ServeHTTP(logoutRec, logoutReq)
	if logoutRec.Code != http.StatusOK {
		t.Fatalf("logout alias status = %d body=%s", logoutRec.Code, logoutRec.Body.String())
	}
}

func TestV2AccountRequestCreatesPendingAccountWithoutAuthenticating(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{Version: "test-version", Store: mem, DevAdminEnabled: true})

	req := httptest.NewRequest(http.MethodPost, "/v2/auth/request-account", strings.NewReader(`{
		"name":"Ada Lovelace",
		"email":"Ada@Example.ORG",
		"affiliation":"Analytical Engine Lab"
	}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("request account status = %d body=%s, want 202", rec.Code, rec.Body.String())
	}
	var response map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode request account response: %v", err)
	}
	if response["authenticated"] != false || response["account_status"] != "pending" {
		t.Fatalf("request account response = %#v, want unauthenticated pending status", response)
	}
	cookie := findCookie(rec.Result().Cookies(), "ultra_dev_auth")
	if cookie == nil {
		t.Fatalf("request account should set signed-out dev auth cookie")
	}
	if cookie.Value != "signed_out" {
		t.Fatalf("request account dev auth cookie = %q, want signed_out", cookie.Value)
	}
	sessionReq := httptest.NewRequest(http.MethodGet, "/v2/auth/session", nil)
	sessionReq.AddCookie(cookie)
	sessionRec := httptest.NewRecorder()
	router.ServeHTTP(sessionRec, sessionReq)
	var session map[string]any
	if err := json.Unmarshal(sessionRec.Body.Bytes(), &session); err != nil {
		t.Fatalf("decode signed-out session: %v", err)
	}
	if session["authenticated"] != false {
		t.Fatalf("signed-out request account session = %#v, want unauthenticated", session)
	}

	users, err := mem.ListUsers(context.Background(), 10, "ada@example.org")
	if err != nil {
		t.Fatalf("ListUsers: %v", err)
	}
	if len(users) != 1 {
		t.Fatalf("users = %+v, want one pending request", users)
	}
	got := users[0]
	if got.Email != "ada@example.org" || got.DisplayName != "Ada Lovelace" || got.Status != "pending" {
		t.Fatalf("pending user = %+v, want normalized pending Ada account", got)
	}
	if got.Metadata["affiliation"] != "Analytical Engine Lab" || got.Metadata["source"] != "account_request" {
		t.Fatalf("pending user metadata = %#v, want affiliation/source", got.Metadata)
	}
}

func TestV2AuthLoginWithPasswordRequiresBisqueVerifier(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{
		Version:           "test-version",
		BisqueCredentials: NewBisqueCredentialStore(),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"fake-user","password":"not-real"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("login status = %d body=%s, want 501 without BisQue verifier", rec.Code, rec.Body.String())
	}
	if cookie := findCookie(rec.Result().Cookies(), "ultra_dev_auth"); cookie != nil {
		t.Fatalf("failed password login should not set auth cookie, got %#v", cookie)
	}
}

func TestV2AuthLoginAcceptsLocalBootstrapAdminAndSeedsDefinedUsers(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:           "test-version",
		Store:             mem,
		BisqueCredentials: NewBisqueCredentialStore(),
		DevAdminEnabled:   false,
	})

	users, err := mem.ListUsers(context.Background(), 20, "")
	if err != nil {
		t.Fatalf("ListUsers: %v", err)
	}
	byID := map[string]domain.UserAccount{}
	for _, user := range users {
		byID[user.UserID] = user
	}
	admin := byID["bisque:admin"]
	if admin.UserID == "" || admin.Role != "admin" || admin.Status != "active" {
		t.Fatalf("bootstrap admin = %+v, want active admin account", admin)
	}
	amil := byID["bisque:amil"]
	if amil.UserID == "" || amil.Role != "researcher" || amil.Status != "active" {
		t.Fatalf("bootstrap amil = %+v, want active researcher account", amil)
	}

	req := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"admin","password":"admin"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("bootstrap admin login status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var session map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &session); err != nil {
		t.Fatalf("decode bootstrap admin session: %v", err)
	}
	if session["authenticated"] != true || session["username"] != "admin" || session["is_admin"] != true {
		t.Fatalf("bootstrap admin session = %#v, want authenticated admin session", session)
	}
	if session["bisque_linked"] != false {
		t.Fatalf("bootstrap admin bisque_linked = %#v, want false without BisQue verifier", session["bisque_linked"])
	}
	user, ok := session["user"].(map[string]any)
	if !ok || user["id"] != "bisque:admin" || user["role"] != "admin" {
		t.Fatalf("bootstrap admin user = %#v, want bisque:admin admin role", session["user"])
	}
}

func TestV2AuthLoginRejectsBisqueAuthRedirect(t *testing.T) {
	t.Parallel()

	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/auth_service/login" {
			w.Header().Set("Content-Type", "text/html")
			_, _ = w.Write([]byte(`<html><body>login</body></html>`))
			return
		}
		if r.Header.Get("Authorization") == "Basic "+base64.StdEncoding.EncodeToString([]byte("real-user:real-secret")) {
			w.Header().Set("Content-Type", "application/xml")
			if r.URL.Path == "/auth_service/session" {
				_, _ = w.Write([]byte(`<response><tag name="user" value="real-user"/></response>`))
				return
			}
			_, _ = w.Write([]byte(`<response><image uri="` + bisque.URL + `/data_service/image/ok" name="ok.jpg" resource_uniq="ok"/></response>`))
			return
		}
		http.Redirect(w, r, bisque.URL+"/auth_service/login", http.StatusFound)
	}))
	defer bisque.Close()

	router := NewRouter(ServerDeps{
		Version:           "test-version",
		BisqueCredentials: NewBisqueCredentialStore(),
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"fake-user","password":"not-real"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("fake login status = %d body=%s, want 502 after auth redirect", rec.Code, rec.Body.String())
	}
	if cookie := findCookie(rec.Result().Cookies(), "ultra_dev_auth"); cookie == nil || cookie.Value != "signed_out" {
		t.Fatalf("fake login cookie = %#v, want signed_out", cookie)
	}

	validReq := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"real-user","password":"real-secret"}`))
	validReq.Header.Set("Content-Type", "application/json")
	validRec := httptest.NewRecorder()
	router.ServeHTTP(validRec, validReq)
	if validRec.Code != http.StatusOK {
		t.Fatalf("valid login status = %d body=%s, want 200", validRec.Code, validRec.Body.String())
	}
}

func TestV2AuthLoginRejectsApprovedUserWhenBisqueSessionDoesNotConfirmIdentity(t *testing.T) {
	t.Parallel()

	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		if r.URL.Path == "/auth_service/session" {
			_, _ = w.Write([]byte(`<response/>`))
			return
		}
		_, _ = w.Write([]byte(`<response><image uri="` + bisque.URL + `/data_service/image/public" name="public.jpg" resource_uniq="public"/></response>`))
	}))
	defer bisque.Close()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:           "test-version",
		Store:             mem,
		BisqueCredentials: NewBisqueCredentialStore(),
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"amil","password":"definitely-wrong"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code == http.StatusOK {
		t.Fatalf("unconfirmed BisQue session login status = %d body=%s, want rejection", rec.Code, rec.Body.String())
	}
	if cookie := findCookie(rec.Result().Cookies(), "ultra_dev_auth"); cookie == nil || cookie.Value != "signed_out" {
		t.Fatalf("unconfirmed BisQue session login cookie = %#v, want signed_out", cookie)
	}
}

func TestV2AuthLoginRequiresApprovedUltraAccountWhenStoreConfigured(t *testing.T) {
	t.Parallel()

	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		if r.URL.Path == "/auth_service/session" {
			_, _ = w.Write([]byte(`<response><tag name="user" value="verified"/></response>`))
			return
		}
		_, _ = w.Write([]byte(`<response><image uri="` + bisque.URL + `/data_service/image/ok" name="ok.jpg" resource_uniq="ok"/></response>`))
	}))
	defer bisque.Close()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:           "test-version",
		Store:             mem,
		BisqueCredentials: NewBisqueCredentialStore(),
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"fake-user@example.org","password":"not-real"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("unapproved login status = %d body=%s, want 403", rec.Code, rec.Body.String())
	}
	if cookie := findCookie(rec.Result().Cookies(), "ultra_dev_auth"); cookie == nil || cookie.Value != "signed_out" {
		t.Fatalf("unapproved login cookie = %#v, want signed_out", cookie)
	}
	var denied map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &denied); err != nil {
		t.Fatalf("decode denied login: %v", err)
	}
	if denied["authenticated"] != false || denied["account_status"] != "pending" {
		t.Fatalf("denied login = %#v, want unauthenticated pending", denied)
	}

	users, err := mem.ListUsers(context.Background(), 10, "fake-user@example.org")
	if err != nil {
		t.Fatalf("ListUsers: %v", err)
	}
	if len(users) != 1 || users[0].Status != "pending" {
		t.Fatalf("users = %+v, want pending account created for review", users)
	}

	if _, err := mem.CreateUser(context.Background(), domain.CreateUserInput{
		UserID:      "bisque:approved@example.org",
		Email:       "approved@example.org",
		DisplayName: "Approved Scientist",
		Role:        "researcher",
		Status:      "active",
		OrgID:       "local-org",
	}); err != nil {
		t.Fatalf("CreateUser approved: %v", err)
	}
	approvedReq := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"approved@example.org","password":"real-secret"}`))
	approvedReq.Header.Set("Content-Type", "application/json")
	approvedRec := httptest.NewRecorder()
	router.ServeHTTP(approvedRec, approvedReq)
	if approvedRec.Code != http.StatusOK {
		t.Fatalf("approved login status = %d body=%s, want 200", approvedRec.Code, approvedRec.Body.String())
	}
}

func TestV2ThreadRunArtifactHandlers(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	createThreadBody := strings.NewReader(`{"title":"Research","initial_messages":[{"role":"user","content":"hello"}]}`)
	createThreadReq := httptest.NewRequest(http.MethodPost, "/v2/threads", createThreadBody)
	createThreadReq.Header.Set("Content-Type", "application/json")
	createThreadRec := httptest.NewRecorder()
	router.ServeHTTP(createThreadRec, createThreadReq)
	if createThreadRec.Code != http.StatusOK {
		t.Fatalf("create thread status = %d body=%s", createThreadRec.Code, createThreadRec.Body.String())
	}
	var thread map[string]any
	if err := json.Unmarshal(createThreadRec.Body.Bytes(), &thread); err != nil {
		t.Fatalf("decode thread: %v", err)
	}
	threadID, ok := thread["thread_id"].(string)
	if !ok || threadID == "" {
		t.Fatalf("thread response missing thread_id: %+v", thread)
	}

	createRunBody := strings.NewReader(`{"goal":"hello","messages":[{"role":"user","content":"hello"}],"file_ids":["file-1"],"resource_uris":["bisque://resource/1"],"dataset_uris":["bisque://dataset/2"],"selected_tool_names":["rarespot_ecology_inference"],"knowledge_context":{"active_paper":"arxiv:2509.26626"},"workflow_hint":{"id":"rarespot_ecology"},"selection_context":{"source":"sidebar"},"budgets":{"max_runtime_seconds":1800},"reasoning_mode":"deep","benchmark":{"suite":"http-context"}}`)
	createRunReq := httptest.NewRequest(http.MethodPost, "/v2/threads/"+threadID+"/runs", createRunBody)
	createRunReq.Header.Set("Content-Type", "application/json")
	createRunRec := httptest.NewRecorder()
	router.ServeHTTP(createRunRec, createRunReq)
	if createRunRec.Code != http.StatusOK {
		t.Fatalf("create run status = %d body=%s", createRunRec.Code, createRunRec.Body.String())
	}
	var run map[string]any
	if err := json.Unmarshal(createRunRec.Body.Bytes(), &run); err != nil {
		t.Fatalf("decode run: %v", err)
	}
	runID, ok := run["run_id"].(string)
	if !ok || runID == "" {
		t.Fatalf("run response missing run_id: %+v", run)
	}
	if run["thread_id"] != threadID {
		t.Fatalf("run thread = %v, want %s", run["thread_id"], threadID)
	}
	if run["workflow_kind"] != "rarespot_ecology" {
		t.Fatalf("run workflow_kind = %v, want rarespot_ecology", run["workflow_kind"])
	}
	metadata, ok := run["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("run metadata missing: %+v", run)
	}
	if got := metadata["file_ids"]; !jsonArrayEquals(got, []string{"file-1"}) {
		t.Fatalf("metadata file_ids = %#v, want file-1", got)
	}
	if got := metadata["resource_uris"]; !jsonArrayEquals(got, []string{"bisque://resource/1"}) {
		t.Fatalf("metadata resource_uris = %#v, want resource URI", got)
	}
	if got := metadata["dataset_uris"]; !jsonArrayEquals(got, []string{"bisque://dataset/2"}) {
		t.Fatalf("metadata dataset_uris = %#v, want dataset URI", got)
	}
	knowledge, ok := metadata["knowledge_context"].(map[string]any)
	if !ok || knowledge["active_paper"] != "arxiv:2509.26626" {
		t.Fatalf("metadata knowledge_context = %#v, want active paper", metadata["knowledge_context"])
	}
	if metadata["reasoning_mode"] != "deep" {
		t.Fatalf("metadata reasoning_mode = %#v, want deep", metadata["reasoning_mode"])
	}
	benchmark, ok := metadata["benchmark"].(map[string]any)
	if !ok || benchmark["suite"] != "http-context" {
		t.Fatalf("metadata benchmark = %#v, want http context", metadata["benchmark"])
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/runs/"+runID+"/events?limit=10", nil)
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("events status = %d body=%s", eventsRec.Code, eventsRec.Body.String())
	}
	var events struct {
		RunID  string                  `json:"run_id"`
		Count  int                     `json:"count"`
		Events []domain.RunEventRecord `json:"events"`
	}
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode events: %v", err)
	}
	if events.Count != 1 || events.Events[0].EventKind != "run.accepted" {
		t.Fatalf("events = %+v, want run.accepted", events)
	}
}

func TestV2ThreadListHonorsOffsetAndReturnsPageMetadata(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	for _, title := range []string{"first chat", "second chat", "third chat"} {
		req := httptest.NewRequest(
			http.MethodPost,
			"/v2/threads",
			strings.NewReader(`{"title":"`+title+`"}`),
		)
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("create %q status = %d body=%s", title, rec.Code, rec.Body.String())
		}
		time.Sleep(time.Millisecond)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/threads?limit=1&offset=1", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("list status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload struct {
		Count      int                   `json:"count"`
		TotalCount int                   `json:"total_count"`
		Limit      int                   `json:"limit"`
		Offset     int                   `json:"offset"`
		HasMore    bool                  `json:"has_more"`
		Threads    []domain.ThreadRecord `json:"threads"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode list: %v", err)
	}
	if payload.Count != 1 || payload.TotalCount != 3 || payload.Limit != 1 || payload.Offset != 1 || !payload.HasMore {
		t.Fatalf("payload metadata = count %d total %d limit %d offset %d has_more %v",
			payload.Count, payload.TotalCount, payload.Limit, payload.Offset, payload.HasMore)
	}
	if len(payload.Threads) != 1 || payload.Threads[0].Title != "second chat" {
		t.Fatalf("paged threads = %+v, want second chat", payload.Threads)
	}
}

func TestV2ThreadAndRunCreationUsesDevPrincipalHeaders(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	createThreadReq := httptest.NewRequest(http.MethodPost, "/v2/threads", strings.NewReader(`{"user_id":"body-user","title":"Principal thread"}`))
	createThreadReq.Header.Set("Content-Type", "application/json")
	createThreadReq.Header.Set("X-Ultra-User-Id", "ada")
	createThreadReq.Header.Set("X-Ultra-Org-Id", "allen-institute")
	createThreadReq.Header.Set("X-Ultra-Role", "admin")
	createThreadRec := httptest.NewRecorder()
	router.ServeHTTP(createThreadRec, createThreadReq)
	if createThreadRec.Code != http.StatusOK {
		t.Fatalf("create thread status = %d body=%s", createThreadRec.Code, createThreadRec.Body.String())
	}
	var thread domain.ThreadRecord
	if err := json.Unmarshal(createThreadRec.Body.Bytes(), &thread); err != nil {
		t.Fatalf("decode thread: %v", err)
	}
	if thread.UserID != "ada" {
		t.Fatalf("thread user_id = %q, want principal header user", thread.UserID)
	}
	threadPrincipal, ok := thread.Metadata["principal"].(map[string]any)
	if !ok {
		t.Fatalf("thread metadata = %+v, want principal metadata", thread.Metadata)
	}
	if threadPrincipal["user_id"] != "ada" || threadPrincipal["org_id"] != "allen-institute" || threadPrincipal["role"] != "admin" {
		t.Fatalf("thread principal = %+v, want header principal", threadPrincipal)
	}

	createRunReq := httptest.NewRequest(http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", strings.NewReader(`{"user_id":"body-user","goal":"Run attributed work","messages":[{"role":"user","content":"Run attributed work"}],"metadata":{"existing":"kept"}}`))
	createRunReq.Header.Set("Content-Type", "application/json")
	createRunReq.Header.Set("X-Ultra-User-Id", "ada")
	createRunReq.Header.Set("X-Ultra-Org-Id", "allen-institute")
	createRunReq.Header.Set("X-Ultra-Role", "admin")
	createRunRec := httptest.NewRecorder()
	router.ServeHTTP(createRunRec, createRunReq)
	if createRunRec.Code != http.StatusOK {
		t.Fatalf("create run status = %d body=%s", createRunRec.Code, createRunRec.Body.String())
	}
	var run domain.RunRecord
	if err := json.Unmarshal(createRunRec.Body.Bytes(), &run); err != nil {
		t.Fatalf("decode run: %v", err)
	}
	if run.UserID != "ada" {
		t.Fatalf("run user_id = %q, want principal header user", run.UserID)
	}
	if run.Metadata["existing"] != "kept" {
		t.Fatalf("run metadata existing = %+v, want caller metadata preserved", run.Metadata)
	}
	runPrincipal, ok := run.Metadata["principal"].(map[string]any)
	if !ok {
		t.Fatalf("run metadata = %+v, want principal metadata", run.Metadata)
	}
	if runPrincipal["user_id"] != "ada" || runPrincipal["org_id"] != "allen-institute" || runPrincipal["role"] != "admin" {
		t.Fatalf("run principal = %+v, want header principal", runPrincipal)
	}
}

func TestV2CurrentUserProfileRoundTrip(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	service := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	// GET before any profile exists still returns the principal identity.
	getReq := httptest.NewRequest(http.MethodGet, "/v2/me", nil)
	getReq.Header.Set("X-Ultra-User-Id", "ada")
	getReq.Header.Set("X-Ultra-Role", "researcher")
	getRec := httptest.NewRecorder()
	router.ServeHTTP(getRec, getReq)
	if getRec.Code != http.StatusOK {
		t.Fatalf("GET /v2/me status = %d body=%s", getRec.Code, getRec.Body.String())
	}

	// PATCH writes a profile, creating the account on demand.
	body := `{"display_name":"Ada Lovelace","title":"Principal Investigator","institution":"Analytical Engine Lab","research_interests":"symbolic computation","bio":"Studies general-purpose computation."}`
	patchReq := httptest.NewRequest(http.MethodPatch, "/v2/me", strings.NewReader(body))
	patchReq.Header.Set("Content-Type", "application/json")
	patchReq.Header.Set("X-Ultra-User-Id", "ada")
	patchReq.Header.Set("X-Ultra-Role", "researcher")
	patchRec := httptest.NewRecorder()
	router.ServeHTTP(patchRec, patchReq)
	if patchRec.Code != http.StatusOK {
		t.Fatalf("PATCH /v2/me status = %d body=%s", patchRec.Code, patchRec.Body.String())
	}
	var patched struct {
		User struct {
			UserID      string `json:"user_id"`
			DisplayName string `json:"display_name"`
		} `json:"user"`
		Profile domain.UserProfile `json:"profile"`
	}
	if err := json.Unmarshal(patchRec.Body.Bytes(), &patched); err != nil {
		t.Fatalf("decode PATCH response: %v body=%s", err, patchRec.Body.String())
	}
	if patched.User.UserID != "ada" || patched.User.DisplayName != "Ada Lovelace" {
		t.Fatalf("patched user = %+v, want ada / Ada Lovelace", patched.User)
	}
	if patched.Profile.Title != "Principal Investigator" || patched.Profile.Institution != "Analytical Engine Lab" {
		t.Fatalf("patched profile = %+v, want PI / Analytical Engine Lab", patched.Profile)
	}
	if patched.Profile.ResearchInterests != "symbolic computation" {
		t.Fatalf("patched research interests = %q", patched.Profile.ResearchInterests)
	}

	// GET now reflects the saved profile, and a partial PATCH preserves other fields.
	partialReq := httptest.NewRequest(http.MethodPatch, "/v2/me", strings.NewReader(`{"bio":"Updated bio only."}`))
	partialReq.Header.Set("Content-Type", "application/json")
	partialReq.Header.Set("X-Ultra-User-Id", "ada")
	partialRec := httptest.NewRecorder()
	router.ServeHTTP(partialRec, partialReq)
	if partialRec.Code != http.StatusOK {
		t.Fatalf("partial PATCH status = %d body=%s", partialRec.Code, partialRec.Body.String())
	}
	var partial struct {
		Profile domain.UserProfile `json:"profile"`
	}
	if err := json.Unmarshal(partialRec.Body.Bytes(), &partial); err != nil {
		t.Fatalf("decode partial PATCH: %v", err)
	}
	if partial.Profile.Bio != "Updated bio only." {
		t.Fatalf("bio = %q, want updated", partial.Profile.Bio)
	}
	if partial.Profile.Title != "Principal Investigator" {
		t.Fatalf("title = %q, want preserved across partial patch", partial.Profile.Title)
	}

	// A different principal must not see ada's profile.
	otherReq := httptest.NewRequest(http.MethodGet, "/v2/me", nil)
	otherReq.Header.Set("X-Ultra-User-Id", "grace")
	otherRec := httptest.NewRecorder()
	router.ServeHTTP(otherRec, otherReq)
	if otherRec.Code != http.StatusOK {
		t.Fatalf("GET /v2/me other status = %d", otherRec.Code)
	}
	var other struct {
		Profile domain.UserProfile `json:"profile"`
	}
	if err := json.Unmarshal(otherRec.Body.Bytes(), &other); err != nil {
		t.Fatalf("decode other: %v", err)
	}
	if other.Profile.Title != "" || other.Profile.Bio != "" {
		t.Fatalf("other principal profile = %+v, want empty", other.Profile)
	}
}

func TestV2TokenUsageReturnsAggregatedStatsAndDailySeries(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	service := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})
	ctx := context.Background()
	today := domain.Now().UTC().Truncate(24 * time.Hour)
	yesterday := today.AddDate(0, 0, -1)

	// One run yesterday, two runs today → today is the peak day and the streak
	// spans both days.
	for _, usage := range []domain.RecordUserTokenUsageInput{
		{UserID: "ada", Day: yesterday, InputTokens: 100, OutputTokens: 20, TotalTokens: 120},
		{UserID: "ada", Day: today, InputTokens: 200, OutputTokens: 40, TotalTokens: 240},
		{UserID: "ada", Day: today, InputTokens: 50, OutputTokens: 10, TotalTokens: 60},
		// A different user's spend must never leak into ada's totals.
		{UserID: "grace", Day: today, InputTokens: 9000, OutputTokens: 9000, TotalTokens: 18000},
	} {
		if err := mem.RecordUserTokenUsage(ctx, usage); err != nil {
			t.Fatalf("RecordUserTokenUsage: %v", err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/me/token-usage?days=30", nil)
	req.Header.Set("X-Ultra-User-Id", "ada")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("token usage status = %d body=%s", rec.Code, rec.Body.String())
	}

	var resp struct {
		Days    int `json:"days"`
		Summary struct {
			LifetimeTotalTokens int64 `json:"lifetime_total_tokens"`
			PeakDailyTotal      int64 `json:"peak_daily_total"`
			CurrentStreakDays   int   `json:"current_streak_days"`
			LongestStreakDays   int   `json:"longest_streak_days"`
			ActiveDays          int   `json:"active_days"`
		} `json:"summary"`
		Daily []struct {
			Day         string `json:"day"`
			TotalTokens int64  `json:"total_tokens"`
			RunCount    int64  `json:"run_count"`
		} `json:"daily"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode token usage: %v body=%s", err, rec.Body.String())
	}
	if resp.Days != 30 {
		t.Fatalf("days = %d, want 30", resp.Days)
	}
	if resp.Summary.LifetimeTotalTokens != 420 {
		t.Fatalf("lifetime total = %d, want 420 (120+240+60)", resp.Summary.LifetimeTotalTokens)
	}
	if resp.Summary.PeakDailyTotal != 300 {
		t.Fatalf("peak daily = %d, want 300 (today 240+60)", resp.Summary.PeakDailyTotal)
	}
	if resp.Summary.CurrentStreakDays != 2 {
		t.Fatalf("current streak = %d, want 2", resp.Summary.CurrentStreakDays)
	}
	if resp.Summary.LongestStreakDays != 2 {
		t.Fatalf("longest streak = %d, want 2", resp.Summary.LongestStreakDays)
	}
	if resp.Summary.ActiveDays != 2 || len(resp.Daily) != 2 {
		t.Fatalf("active days = %d / daily rows = %d, want 2/2", resp.Summary.ActiveDays, len(resp.Daily))
	}
	todayKey := today.Format("2006-01-02")
	var sawToday bool
	for _, point := range resp.Daily {
		if point.Day == todayKey {
			sawToday = true
			if point.TotalTokens != 300 || point.RunCount != 2 {
				t.Fatalf("today point = %+v, want total 300 run_count 2", point)
			}
		}
	}
	if !sawToday {
		t.Fatalf("daily series missing today %q: %+v", todayKey, resp.Daily)
	}
}

func TestV2ThreadAndRunCreationDefaultsLocalDevPrincipal(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	createThreadReq := httptest.NewRequest(http.MethodPost, "/v2/threads", strings.NewReader(`{"title":"Default principal thread"}`))
	createThreadReq.Header.Set("Content-Type", "application/json")
	createThreadRec := httptest.NewRecorder()
	router.ServeHTTP(createThreadRec, createThreadReq)
	if createThreadRec.Code != http.StatusOK {
		t.Fatalf("create thread status = %d body=%s", createThreadRec.Code, createThreadRec.Body.String())
	}
	var thread domain.ThreadRecord
	if err := json.Unmarshal(createThreadRec.Body.Bytes(), &thread); err != nil {
		t.Fatalf("decode thread: %v", err)
	}
	if thread.UserID != "local-user" {
		t.Fatalf("thread user_id = %q, want default local-user", thread.UserID)
	}
	principal, ok := thread.Metadata["principal"].(map[string]any)
	if !ok || principal["org_id"] != "local-org" || principal["role"] != "researcher" {
		t.Fatalf("thread principal = %+v, want default local principal", thread.Metadata["principal"])
	}

	createRunReq := httptest.NewRequest(http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", strings.NewReader(`{"goal":"Run default principal work"}`))
	createRunReq.Header.Set("Content-Type", "application/json")
	createRunRec := httptest.NewRecorder()
	router.ServeHTTP(createRunRec, createRunReq)
	if createRunRec.Code != http.StatusOK {
		t.Fatalf("create run status = %d body=%s", createRunRec.Code, createRunRec.Body.String())
	}
	var run domain.RunRecord
	if err := json.Unmarshal(createRunRec.Body.Bytes(), &run); err != nil {
		t.Fatalf("decode run: %v", err)
	}
	if run.UserID != "local-user" {
		t.Fatalf("run user_id = %q, want default local-user", run.UserID)
	}
	runPrincipal, ok := run.Metadata["principal"].(map[string]any)
	if !ok || runPrincipal["user_id"] != "local-user" || runPrincipal["org_id"] != "local-org" || runPrincipal["role"] != "researcher" {
		t.Fatalf("run principal = %+v, want default local principal", run.Metadata["principal"])
	}
}

func TestV2ThreadUpsertIsTenantScopedAndPersistsManualTitleState(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "alice",
		Title:  "Initial automatic title",
		Metadata: domain.JSONMap{
			"frontend_bridge": "v2-chat",
			"conversation_id": "conversation-alice",
			"title_state": domain.JSONMap{
				"source": "auto",
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	req := httptest.NewRequest(
		http.MethodPut,
		"/v2/threads/"+thread.ThreadID,
		strings.NewReader(`{"title":"Manual ecology review","metadata":{"conversation_id":"conversation-alice","title_state":{"source":"manual","updated_by":"user"}}}`),
	)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-User-Id", "alice")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("upsert status = %d body=%s", rec.Code, rec.Body.String())
	}
	var updated domain.ThreadRecord
	if err := json.Unmarshal(rec.Body.Bytes(), &updated); err != nil {
		t.Fatalf("decode updated thread: %v", err)
	}
	if updated.Title != "Manual ecology review" {
		t.Fatalf("updated title = %q, want manual title", updated.Title)
	}
	titleState, ok := updated.Metadata["title_state"].(map[string]any)
	if !ok || titleState["source"] != "manual" || titleState["updated_by"] != "user" {
		t.Fatalf("title_state = %+v, want manual metadata", updated.Metadata["title_state"])
	}
	if updated.Metadata["frontend_bridge"] != "v2-chat" {
		t.Fatalf("metadata = %+v, want existing frontend_bridge preserved", updated.Metadata)
	}

	bobReq := httptest.NewRequest(
		http.MethodPut,
		"/v2/threads/"+thread.ThreadID,
		strings.NewReader(`{"title":"Bob should not write"}`),
	)
	bobReq.Header.Set("Content-Type", "application/json")
	bobReq.Header.Set("X-Ultra-User-Id", "bob")
	bobRec := httptest.NewRecorder()
	router.ServeHTTP(bobRec, bobReq)
	if bobRec.Code != http.StatusNotFound {
		t.Fatalf("bob upsert status = %d body=%s, want 404", bobRec.Code, bobRec.Body.String())
	}
}

func TestV2ThreadDeleteSoftDeletesAndHidesFromUserHistory(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "alice",
		Title:  "Alice delete me",
	})
	if err != nil {
		t.Fatalf("CreateThread alice: %v", err)
	}

	bobDeleteReq := httptest.NewRequest(http.MethodDelete, "/v2/threads/"+thread.ThreadID, nil)
	bobDeleteReq.Header.Set("X-Ultra-User-Id", "bob")
	bobDeleteRec := httptest.NewRecorder()
	router.ServeHTTP(bobDeleteRec, bobDeleteReq)
	if bobDeleteRec.Code != http.StatusNotFound {
		t.Fatalf("bob delete status = %d body=%s, want 404", bobDeleteRec.Code, bobDeleteRec.Body.String())
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, "/v2/threads/"+thread.ThreadID, nil)
	deleteReq.Header.Set("X-Ultra-User-Id", "alice")
	deleteRec := httptest.NewRecorder()
	router.ServeHTTP(deleteRec, deleteReq)
	if deleteRec.Code != http.StatusNoContent {
		t.Fatalf("delete status = %d body=%s, want 204", deleteRec.Code, deleteRec.Body.String())
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/threads?limit=20", nil)
	listReq.Header.Set("X-Ultra-User-Id", "alice")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list active status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var activeList listThreadsResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &activeList); err != nil {
		t.Fatalf("decode active list: %v", err)
	}
	if activeList.TotalCount != 0 || len(activeList.Threads) != 0 {
		t.Fatalf("active threads after delete = %+v, want none", activeList.Threads)
	}

	getReq := httptest.NewRequest(http.MethodGet, "/v2/threads/"+thread.ThreadID, nil)
	getReq.Header.Set("X-Ultra-User-Id", "alice")
	getRec := httptest.NewRecorder()
	router.ServeHTTP(getRec, getReq)
	if getRec.Code != http.StatusNotFound {
		t.Fatalf("get deleted status = %d body=%s, want 404", getRec.Code, getRec.Body.String())
	}

	deletedReq := httptest.NewRequest(http.MethodGet, "/v2/threads?status=deleted&limit=20", nil)
	deletedReq.Header.Set("X-Ultra-User-Id", "alice")
	deletedRec := httptest.NewRecorder()
	router.ServeHTTP(deletedRec, deletedReq)
	if deletedRec.Code != http.StatusOK {
		t.Fatalf("list deleted status = %d body=%s", deletedRec.Code, deletedRec.Body.String())
	}
	var deletedList listThreadsResponse
	if err := json.Unmarshal(deletedRec.Body.Bytes(), &deletedList); err != nil {
		t.Fatalf("decode deleted list: %v", err)
	}
	if deletedList.TotalCount != 1 || len(deletedList.Threads) != 1 || deletedList.Threads[0].Status != domain.ThreadStatusDeleted {
		t.Fatalf("deleted threads = %+v, want one deleted thread", deletedList.Threads)
	}
}

func TestV2ListRunsHandler(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "runs",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	first, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "queued run",
	})
	if err != nil {
		t.Fatalf("CreateRun first: %v", err)
	}
	second, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "running run",
	})
	if err != nil {
		t.Fatalf("CreateRun second: %v", err)
	}
	if _, err := mem.UpdateRunStatus(ctx, second.RunID, domain.RunStatusRunning, "", ""); err != nil {
		t.Fatalf("UpdateRunStatus: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/runs?limit=20&status=queued", nil)
	req.Header.Set("X-Ultra-User-Id", "user-1")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("list runs status = %d body=%s", rec.Code, rec.Body.String())
	}

	var response struct {
		Count int                `json:"count"`
		Runs  []domain.RunRecord `json:"runs"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode list runs: %v", err)
	}
	if response.Count != 1 || len(response.Runs) != 1 || response.Runs[0].RunID != first.RunID {
		t.Fatalf("runs = %+v, want queued first run only", response)
	}
}

func TestV2UserRoutesAreScopedToRequestPrincipal(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	artifactRoot := t.TempDir()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:      "test-version",
		Runs:         service,
		Store:        mem,
		Bus:          bus,
		ArtifactRoot: artifactRoot,
	})

	aliceThread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID:          "alice",
		Title:           "alice thread",
		InitialMessages: []domain.ThreadMessage{{Role: "user", Content: "secret prompt"}},
	})
	if err != nil {
		t.Fatalf("CreateThread alice: %v", err)
	}
	aliceRun, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: aliceThread.ThreadID,
		UserID:   "alice",
		Goal:     "secret run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "secret run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun alice: %v", err)
	}
	if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
		RunID:     aliceRun.RunID,
		ThreadID:  aliceThread.ThreadID,
		EventKind: "message.delta",
		Message:   "private trace",
	}); err != nil {
		t.Fatalf("AppendRunEvent alice: %v", err)
	}
	reportPath := filepath.Join(artifactRoot, aliceRun.RunID, "report.md")
	if err := os.MkdirAll(filepath.Dir(reportPath), 0o755); err != nil {
		t.Fatalf("MkdirAll artifact: %v", err)
	}
	if err := os.WriteFile(reportPath, []byte("alice report"), 0o644); err != nil {
		t.Fatalf("WriteFile artifact: %v", err)
	}
	aliceArtifact, err := mem.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:    aliceRun.RunID,
		ThreadID: aliceThread.ThreadID,
		Kind:     "report",
		Path:     "report.md",
		MimeType: "text/markdown",
	})
	if err != nil {
		t.Fatalf("CreateArtifact alice: %v", err)
	}

	bobThread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "bob", Title: "bob thread"})
	if err != nil {
		t.Fatalf("CreateThread bob: %v", err)
	}
	if _, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: bobThread.ThreadID,
		UserID:   "bob",
		Goal:     "bob run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "bob run"}},
	}); err != nil {
		t.Fatalf("CreateRun bob: %v", err)
	}

	listThreads := httptest.NewRequest(http.MethodGet, "/v2/threads?limit=20", nil)
	listThreads.Header.Set("X-Ultra-User-Id", "bob")
	listThreadsRec := httptest.NewRecorder()
	router.ServeHTTP(listThreadsRec, listThreads)
	if listThreadsRec.Code != http.StatusOK {
		t.Fatalf("bob list threads status = %d body=%s", listThreadsRec.Code, listThreadsRec.Body.String())
	}
	var threadList struct {
		TotalCount int                   `json:"total_count"`
		Threads    []domain.ThreadRecord `json:"threads"`
	}
	if err := json.Unmarshal(listThreadsRec.Body.Bytes(), &threadList); err != nil {
		t.Fatalf("decode bob thread list: %v", err)
	}
	if threadList.TotalCount != 1 || len(threadList.Threads) != 1 || threadList.Threads[0].UserID != "bob" {
		t.Fatalf("bob thread list = %+v, want only bob thread", threadList)
	}

	listRuns := httptest.NewRequest(http.MethodGet, "/v2/runs?limit=20", nil)
	listRuns.Header.Set("X-Ultra-User-Id", "bob")
	listRunsRec := httptest.NewRecorder()
	router.ServeHTTP(listRunsRec, listRuns)
	if listRunsRec.Code != http.StatusOK {
		t.Fatalf("bob list runs status = %d body=%s", listRunsRec.Code, listRunsRec.Body.String())
	}
	var runList struct {
		Runs []domain.RunRecord `json:"runs"`
	}
	if err := json.Unmarshal(listRunsRec.Body.Bytes(), &runList); err != nil {
		t.Fatalf("decode bob run list: %v", err)
	}
	if len(runList.Runs) != 1 || runList.Runs[0].UserID != "bob" {
		t.Fatalf("bob run list = %+v, want only bob run", runList)
	}

	for _, tc := range []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{name: "get thread", method: http.MethodGet, path: "/v2/threads/" + aliceThread.ThreadID},
		{name: "list messages", method: http.MethodGet, path: "/v2/threads/" + aliceThread.ThreadID + "/messages"},
		{name: "create run", method: http.MethodPost, path: "/v2/threads/" + aliceThread.ThreadID + "/runs", body: `{"goal":"steal work"}`},
		{name: "get run", method: http.MethodGet, path: "/v2/runs/" + aliceRun.RunID},
		{name: "list events", method: http.MethodGet, path: "/v2/runs/" + aliceRun.RunID + "/events?limit=10"},
		{name: "cancel run", method: http.MethodPost, path: "/v2/runs/" + aliceRun.RunID + "/cancel", body: `{"reason":"not mine"}`},
		{name: "acquire lease", method: http.MethodPost, path: "/v2/runs/" + aliceRun.RunID + "/lease", body: `{"worker_id":"not-mine","ttl_seconds":60}`},
		{name: "list artifacts", method: http.MethodGet, path: "/v2/runs/" + aliceRun.RunID + "/artifacts"},
		{name: "download artifact path", method: http.MethodGet, path: "/v2/runs/" + aliceRun.RunID + "/artifacts/download?path=report.md"},
		{name: "get artifact", method: http.MethodGet, path: "/v2/artifacts/" + aliceArtifact.ArtifactID},
		{name: "download artifact", method: http.MethodGet, path: "/v2/artifacts/" + aliceArtifact.ArtifactID + "/download"},
	} {
		body := strings.NewReader(tc.body)
		req := httptest.NewRequest(tc.method, tc.path, body)
		req.Header.Set("X-Ultra-User-Id", "bob")
		if tc.body != "" {
			req.Header.Set("Content-Type", "application/json")
		}
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusNotFound {
			t.Fatalf("%s status = %d body=%s, want 404 for another user's object", tc.name, rec.Code, rec.Body.String())
		}
	}
}

func TestV2AdminAndTrainingReadEndpointsAreOwnedByGo(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "admin",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "inspect me",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := mem.UpdateRunStatus(ctx, run.RunID, domain.RunStatusFailed, "", "synthetic failure"); err != nil {
		t.Fatalf("UpdateRunStatus: %v", err)
	}

	cases := []struct {
		path       string
		wantKeys   []string
		statusCode int
	}{
		{path: "/v2/admin/overview", wantKeys: []string{"generated_at", "runtime", "kpis", "recent_issues"}, statusCode: http.StatusOK},
		{path: "/v2/admin/orgs", wantKeys: []string{"count", "organizations"}, statusCode: http.StatusOK},
		{path: "/v2/admin/users", wantKeys: []string{"count", "users"}, statusCode: http.StatusOK},
		{path: "/v2/admin/runs?status=failed", wantKeys: []string{"count", "runs"}, statusCode: http.StatusOK},
		{path: "/v2/admin/issues", wantKeys: []string{"count", "issues"}, statusCode: http.StatusOK},
		{path: "/v2/training/models", wantKeys: []string{"count", "models"}, statusCode: http.StatusOK},
		{path: "/v2/training/prairie/status", wantKeys: []string{"dataset_name", "model_health", "retrain_gate_reasons"}, statusCode: http.StatusOK},
		{path: "/v2/training/prairie/retrain-requests", wantKeys: []string{"count", "requests"}, statusCode: http.StatusOK},
		{path: "/v2/training/domains", wantKeys: []string{"count", "domains"}, statusCode: http.StatusOK},
		{path: "/v2/training/domains/prairie/lineages", wantKeys: []string{"count", "lineages"}, statusCode: http.StatusOK},
		{path: "/v2/training/lineages/prairie-default/versions", wantKeys: []string{"count", "versions"}, statusCode: http.StatusOK},
	}
	for _, tc := range cases {
		req := httptest.NewRequest(http.MethodGet, tc.path, nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != tc.statusCode {
			t.Fatalf("%s status = %d body=%s", tc.path, rec.Code, rec.Body.String())
		}
		var payload map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
			t.Fatalf("%s decode: %v body=%s", tc.path, err, rec.Body.String())
		}
		for _, key := range tc.wantKeys {
			if _, ok := payload[key]; !ok {
				t.Fatalf("%s missing key %q in %#v", tc.path, key, payload)
			}
		}
	}
}

func TestV2AdminCreateUserPersistsFirstClassAccount(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	body := `{"email":"grace@example.org","display_name":"Grace Hopper","role":"admin","org_id":"local-org"}`
	req := httptest.NewRequest(http.MethodPost, "/v2/admin/users", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create user status = %d body=%s", rec.Code, rec.Body.String())
	}
	var created domain.UserAccount
	if err := json.Unmarshal(rec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created user: %v body=%s", err, rec.Body.String())
	}
	if created.UserID == "" || created.Email != "grace@example.org" || created.DisplayName != "Grace Hopper" {
		t.Fatalf("created user = %+v, want persisted account fields", created)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/admin/users?q=grace", nil)
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list users status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var payload adminUserListResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode list users: %v body=%s", err, listRec.Body.String())
	}
	if payload.Count != 1 {
		t.Fatalf("user count = %d, want 1 payload=%+v", payload.Count, payload)
	}
	got := payload.Users[0]
	if got.UserID != created.UserID || got.Email != "grace@example.org" || got.DisplayName != "Grace Hopper" || got.Role != "admin" || got.Status != "active" {
		t.Fatalf("listed user = %+v, want created account plus telemetry", got)
	}
}

func TestV2AdminUsersIncludeResourceCatalogAccounting(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	account, err := mem.CreateUser(ctx, domain.CreateUserInput{
		UserID:      "researcher-1",
		Email:       "researcher@example.org",
		DisplayName: "Researcher One",
		OrgID:       "local-org",
	})
	if err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_active_a",
		OriginalName: "a.png",
		ContentType:  "image/png",
		SizeBytes:    100,
		SourceType:   "upload",
		ResourceKind: "image",
		ProjectID:    "project-alpha",
		OwnerUserID:  account.UserID,
		OwnerOrgID:   "local-org",
		Status:       "active",
	}); err != nil {
		t.Fatalf("UpsertResource active a: %v", err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_active_b",
		OriginalName: "b.csv",
		ContentType:  "text/csv",
		SizeBytes:    23,
		SourceType:   "artifact",
		ResourceKind: "table",
		ProjectID:    "project-alpha",
		OwnerUserID:  account.UserID,
		OwnerOrgID:   "local-org",
		Status:       "active",
	}); err != nil {
		t.Fatalf("UpsertResource active b: %v", err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_deleted",
		OriginalName: "deleted.txt",
		SizeBytes:    999,
		SourceType:   "upload",
		ResourceKind: "file",
		OwnerUserID:  account.UserID,
		OwnerOrgID:   "local-org",
		Status:       "active",
	}); err != nil {
		t.Fatalf("UpsertResource deleted: %v", err)
	}
	if _, err := mem.SoftDeleteResourceForUser(ctx, "file_deleted", account.UserID, "local-org", time.Now()); err != nil {
		t.Fatalf("SoftDeleteResourceForUser: %v", err)
	}

	usersReq := httptest.NewRequest(http.MethodGet, "/v2/admin/users?q=researcher@example.org", nil)
	usersRec := httptest.NewRecorder()
	router.ServeHTTP(usersRec, usersReq)
	if usersRec.Code != http.StatusOK {
		t.Fatalf("list users status = %d body=%s", usersRec.Code, usersRec.Body.String())
	}
	var usersPayload adminUserListResponse
	if err := json.Unmarshal(usersRec.Body.Bytes(), &usersPayload); err != nil {
		t.Fatalf("decode users: %v", err)
	}
	if usersPayload.Count != 1 || usersPayload.Users[0].Uploads != 2 || usersPayload.Users[0].StorageBytes != 123 {
		t.Fatalf("users payload = %+v, want active catalog accounting", usersPayload)
	}

	overviewReq := httptest.NewRequest(http.MethodGet, "/v2/admin/overview", nil)
	overviewRec := httptest.NewRecorder()
	router.ServeHTTP(overviewRec, overviewReq)
	if overviewRec.Code != http.StatusOK {
		t.Fatalf("overview status = %d body=%s", overviewRec.Code, overviewRec.Body.String())
	}
	var overview adminOverviewResponse
	if err := json.Unmarshal(overviewRec.Body.Bytes(), &overview); err != nil {
		t.Fatalf("decode overview: %v", err)
	}
	if overview.KPIs.TotalUploads != 2 || overview.KPIs.SoftDeletedUploads != 1 || overview.KPIs.TotalStorageBytes != 123 {
		t.Fatalf("overview KPIs = %+v, want resource catalog accounting", overview.KPIs)
	}
	if len(overview.ResourceProjects) != 1 || overview.ResourceProjects[0].ID != "project-alpha" || overview.ResourceProjects[0].Uploads != 2 || overview.ResourceProjects[0].StorageBytes != 123 {
		t.Fatalf("overview resource projects = %+v, want project-alpha accounting", overview.ResourceProjects)
	}

	orgsReq := httptest.NewRequest(http.MethodGet, "/v2/admin/orgs?q=local", nil)
	orgsRec := httptest.NewRecorder()
	router.ServeHTTP(orgsRec, orgsReq)
	if orgsRec.Code != http.StatusOK {
		t.Fatalf("orgs status = %d body=%s", orgsRec.Code, orgsRec.Body.String())
	}
	var orgsPayload adminOrganizationListResponse
	if err := json.Unmarshal(orgsRec.Body.Bytes(), &orgsPayload); err != nil {
		t.Fatalf("decode orgs: %v", err)
	}
	if orgsPayload.Count != 1 || orgsPayload.Organizations[0].Uploads != 2 || orgsPayload.Organizations[0].StorageBytes != 123 {
		t.Fatalf("orgs payload = %+v, want org resource accounting", orgsPayload)
	}
}

func TestV2AdminCreateUserDuplicateEmailReturnsConflict(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	first := httptest.NewRequest(http.MethodPost, "/v2/admin/users", strings.NewReader(`{"email":"ada@example.org"}`))
	first.Header.Set("Content-Type", "application/json")
	firstRec := httptest.NewRecorder()
	router.ServeHTTP(firstRec, first)
	if firstRec.Code != http.StatusCreated {
		t.Fatalf("first create status = %d body=%s", firstRec.Code, firstRec.Body.String())
	}

	duplicate := httptest.NewRequest(http.MethodPost, "/v2/admin/users", strings.NewReader(`{"email":"ADA@example.org"}`))
	duplicate.Header.Set("Content-Type", "application/json")
	duplicateRec := httptest.NewRecorder()
	router.ServeHTTP(duplicateRec, duplicate)
	if duplicateRec.Code != http.StatusConflict {
		t.Fatalf("duplicate create status = %d body=%s", duplicateRec.Code, duplicateRec.Body.String())
	}
	if !strings.Contains(strings.ToLower(duplicateRec.Body.String()), "already exists") {
		t.Fatalf("duplicate response should explain conflict: %s", duplicateRec.Body.String())
	}
}

func TestV2AdminDeleteUserSoftDisablesAccount(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	create := httptest.NewRequest(http.MethodPost, "/v2/admin/users", strings.NewReader(`{"email":"remove-me@example.org","display_name":"Remove Me"}`))
	create.Header.Set("Content-Type", "application/json")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, create)
	if createRec.Code != http.StatusCreated {
		t.Fatalf("create user status = %d body=%s", createRec.Code, createRec.Body.String())
	}
	var created domain.UserAccount
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created user: %v", err)
	}

	req := httptest.NewRequest(http.MethodDelete, "/v2/admin/users/"+created.UserID, nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("delete user status = %d body=%s", rec.Code, rec.Body.String())
	}
	var disabled domain.UserAccount
	if err := json.Unmarshal(rec.Body.Bytes(), &disabled); err != nil {
		t.Fatalf("decode disabled user: %v body=%s", err, rec.Body.String())
	}
	if disabled.UserID != created.UserID || disabled.Status != "disabled" {
		t.Fatalf("disabled user = %+v, want same user with disabled status", disabled)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/admin/users?q=remove-me", nil)
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list users status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var payload adminUserListResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode list users: %v body=%s", err, listRec.Body.String())
	}
	if payload.Count != 1 || payload.Users[0].Status != "disabled" {
		t.Fatalf("listed users = %+v, want disabled account retained", payload.Users)
	}
}

func TestV2AdminCreateOrganizationPersistsFirstClassOrg(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	body := `{"org_id":"allen-institute","name":"Allen Institute","status":"active","metadata":{"source":"admin_console"}}`
	req := httptest.NewRequest(http.MethodPost, "/v2/admin/orgs", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create org status = %d body=%s", rec.Code, rec.Body.String())
	}
	var created domain.Organization
	if err := json.Unmarshal(rec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created org: %v body=%s", err, rec.Body.String())
	}
	if created.OrgID != "allen-institute" || created.Name != "Allen Institute" || created.Status != "active" {
		t.Fatalf("created org = %+v, want persisted organization fields", created)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/admin/orgs?q=allen", nil)
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list orgs status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var payload adminOrganizationListResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode list orgs: %v body=%s", err, listRec.Body.String())
	}
	if payload.Count != 1 || payload.Organizations[0].OrgID != created.OrgID {
		t.Fatalf("org list = %+v, want created organization", payload)
	}
}

func TestV2AdminCreateOrganizationDuplicateIDReturnsConflict(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	first := httptest.NewRequest(http.MethodPost, "/v2/admin/orgs", strings.NewReader(`{"org_id":"smithsonian","name":"Smithsonian"}`))
	first.Header.Set("Content-Type", "application/json")
	firstRec := httptest.NewRecorder()
	router.ServeHTTP(firstRec, first)
	if firstRec.Code != http.StatusCreated {
		t.Fatalf("first create status = %d body=%s", firstRec.Code, firstRec.Body.String())
	}

	duplicate := httptest.NewRequest(http.MethodPost, "/v2/admin/orgs", strings.NewReader(`{"org_id":"smithsonian","name":"Smithsonian duplicate"}`))
	duplicate.Header.Set("Content-Type", "application/json")
	duplicateRec := httptest.NewRecorder()
	router.ServeHTTP(duplicateRec, duplicate)
	if duplicateRec.Code != http.StatusConflict {
		t.Fatalf("duplicate create status = %d body=%s", duplicateRec.Code, duplicateRec.Body.String())
	}
}

func TestV2RunLeaseClaimRenewAndRelease(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(context.Background(), runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "lease",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(context.Background(), runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long worker run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long worker run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	claim := httptest.NewRequest(http.MethodPost, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(`{"worker_id":"worker-a","ttl_seconds":60}`))
	claim.Header.Set("Content-Type", "application/json")
	claim.Header.Set("X-Ultra-User-Id", "user-1")
	claimRec := httptest.NewRecorder()
	router.ServeHTTP(claimRec, claim)
	if claimRec.Code != http.StatusOK {
		t.Fatalf("claim status = %d body=%s", claimRec.Code, claimRec.Body.String())
	}
	var lease domain.RunLeaseRecord
	if err := json.Unmarshal(claimRec.Body.Bytes(), &lease); err != nil {
		t.Fatalf("decode lease: %v body=%s", err, claimRec.Body.String())
	}
	if lease.RunID != run.RunID || lease.WorkerID != "worker-a" || lease.LeaseToken == "" {
		t.Fatalf("lease = %+v, want worker-a token", lease)
	}

	competing := httptest.NewRequest(http.MethodPost, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(`{"worker_id":"worker-b","ttl_seconds":60}`))
	competing.Header.Set("Content-Type", "application/json")
	competing.Header.Set("X-Ultra-User-Id", "user-1")
	competingRec := httptest.NewRecorder()
	router.ServeHTTP(competingRec, competing)
	if competingRec.Code != http.StatusConflict {
		t.Fatalf("competing claim status = %d body=%s", competingRec.Code, competingRec.Body.String())
	}

	renewBody := `{"lease_token":"` + lease.LeaseToken + `","ttl_seconds":120}`
	renew := httptest.NewRequest(http.MethodPatch, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(renewBody))
	renew.Header.Set("Content-Type", "application/json")
	renew.Header.Set("X-Ultra-User-Id", "user-1")
	renewRec := httptest.NewRecorder()
	router.ServeHTTP(renewRec, renew)
	if renewRec.Code != http.StatusOK {
		t.Fatalf("renew status = %d body=%s", renewRec.Code, renewRec.Body.String())
	}

	releaseBody := `{"lease_token":"` + lease.LeaseToken + `"}`
	release := httptest.NewRequest(http.MethodDelete, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(releaseBody))
	release.Header.Set("Content-Type", "application/json")
	release.Header.Set("X-Ultra-User-Id", "user-1")
	releaseRec := httptest.NewRecorder()
	router.ServeHTTP(releaseRec, release)
	if releaseRec.Code != http.StatusOK {
		t.Fatalf("release status = %d body=%s", releaseRec.Code, releaseRec.Body.String())
	}

	reclaim := httptest.NewRequest(http.MethodPost, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(`{"worker_id":"worker-b","ttl_seconds":60}`))
	reclaim.Header.Set("Content-Type", "application/json")
	reclaim.Header.Set("X-Ultra-User-Id", "user-1")
	reclaimRec := httptest.NewRecorder()
	router.ServeHTTP(reclaimRec, reclaim)
	if reclaimRec.Code != http.StatusOK {
		t.Fatalf("reclaim status = %d body=%s", reclaimRec.Code, reclaimRec.Body.String())
	}
}

func TestV2WorkerTokenAuthorizesRunStatusAndLeaseLifecycle(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:     "test-version",
		Runs:        service,
		Store:       mem,
		WorkerToken: "trace-worker-secret",
	})

	thread, err := service.CreateThread(context.Background(), runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "worker token",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(context.Background(), runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "worker token run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "worker token run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	// Without any identity the run must stay hidden (anti-enumeration).
	anonymous := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID, nil)
	anonymousRec := httptest.NewRecorder()
	router.ServeHTTP(anonymousRec, anonymous)
	if anonymousRec.Code != http.StatusNotFound {
		t.Fatalf("anonymous run status = %d, want 404", anonymousRec.Code)
	}

	// A wrong worker token must be rejected, not fall through to user scoping.
	wrongToken := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID, nil)
	wrongToken.Header.Set("Authorization", "Bearer wrong-secret")
	wrongTokenRec := httptest.NewRecorder()
	router.ServeHTTP(wrongTokenRec, wrongToken)
	if wrongTokenRec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong token run status = %d, want 401", wrongTokenRec.Code)
	}

	// The worker token grants run visibility without user scoping.
	status := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID, nil)
	status.Header.Set("Authorization", "Bearer trace-worker-secret")
	statusRec := httptest.NewRecorder()
	router.ServeHTTP(statusRec, status)
	if statusRec.Code != http.StatusOK {
		t.Fatalf("worker token run status = %d body=%s", statusRec.Code, statusRec.Body.String())
	}
	var fetched domain.RunRecord
	if err := json.Unmarshal(statusRec.Body.Bytes(), &fetched); err != nil {
		t.Fatalf("decode run: %v", err)
	}
	if fetched.RunID != run.RunID || fetched.Status != domain.RunStatusQueued {
		t.Fatalf("fetched run = %+v, want queued %s", fetched, run.RunID)
	}

	claim := httptest.NewRequest(http.MethodPost, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(`{"worker_id":"worker-a","ttl_seconds":60}`))
	claim.Header.Set("Content-Type", "application/json")
	claim.Header.Set("X-Ultra-Worker-Token", "trace-worker-secret")
	claimRec := httptest.NewRecorder()
	router.ServeHTTP(claimRec, claim)
	if claimRec.Code != http.StatusOK {
		t.Fatalf("worker token claim status = %d body=%s", claimRec.Code, claimRec.Body.String())
	}
	var lease domain.RunLeaseRecord
	if err := json.Unmarshal(claimRec.Body.Bytes(), &lease); err != nil {
		t.Fatalf("decode lease: %v", err)
	}

	renewBody := `{"lease_token":"` + lease.LeaseToken + `","ttl_seconds":120}`
	renew := httptest.NewRequest(http.MethodPatch, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(renewBody))
	renew.Header.Set("Content-Type", "application/json")
	renew.Header.Set("Authorization", "Bearer trace-worker-secret")
	renewRec := httptest.NewRecorder()
	router.ServeHTTP(renewRec, renew)
	if renewRec.Code != http.StatusOK {
		t.Fatalf("worker token renew status = %d body=%s", renewRec.Code, renewRec.Body.String())
	}

	releaseBody := `{"lease_token":"` + lease.LeaseToken + `"}`
	release := httptest.NewRequest(http.MethodDelete, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(releaseBody))
	release.Header.Set("Content-Type", "application/json")
	release.Header.Set("Authorization", "Bearer trace-worker-secret")
	releaseRec := httptest.NewRecorder()
	router.ServeHTTP(releaseRec, release)
	if releaseRec.Code != http.StatusOK {
		t.Fatalf("worker token release status = %d body=%s", releaseRec.Code, releaseRec.Body.String())
	}
}

func TestV2WorkerTokenBypassesWorkOSGateForWorkerEndpoints(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:     "test-version",
		Runs:        service,
		Store:       mem,
		WorkerToken: "trace-worker-secret",
		WorkOS:      testWorkOSAuth(t, WorkOSAuthConfig{}),
	})

	thread, err := service.CreateThread(context.Background(), runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "workos worker token",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(context.Background(), runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "workos worker token run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "workos worker token run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	// Without a worker token the WorkOS gate rejects the unauthenticated call.
	anonymous := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID, nil)
	anonymousRec := httptest.NewRecorder()
	router.ServeHTTP(anonymousRec, anonymous)
	if anonymousRec.Code != http.StatusUnauthorized {
		t.Fatalf("anonymous status under workos = %d, want 401", anonymousRec.Code)
	}

	status := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID, nil)
	status.Header.Set("Authorization", "Bearer trace-worker-secret")
	statusRec := httptest.NewRecorder()
	router.ServeHTTP(statusRec, status)
	if statusRec.Code != http.StatusOK {
		t.Fatalf("worker token status under workos = %d body=%s", statusRec.Code, statusRec.Body.String())
	}

	claim := httptest.NewRequest(http.MethodPost, "/v2/runs/"+run.RunID+"/lease", strings.NewReader(`{"worker_id":"worker-a","ttl_seconds":60}`))
	claim.Header.Set("Content-Type", "application/json")
	claim.Header.Set("Authorization", "Bearer trace-worker-secret")
	claimRec := httptest.NewRecorder()
	router.ServeHTTP(claimRec, claim)
	if claimRec.Code != http.StatusOK {
		t.Fatalf("worker token claim under workos = %d body=%s", claimRec.Code, claimRec.Body.String())
	}

	heartbeat := httptest.NewRequest(http.MethodPost, "/v2/workers/heartbeat", strings.NewReader(`{"worker_id":"worker-a","status":"busy"}`))
	heartbeat.Header.Set("Content-Type", "application/json")
	heartbeat.Header.Set("Authorization", "Bearer trace-worker-secret")
	heartbeatRec := httptest.NewRecorder()
	router.ServeHTTP(heartbeatRec, heartbeat)
	if heartbeatRec.Code != http.StatusOK {
		t.Fatalf("worker token heartbeat under workos = %d body=%s", heartbeatRec.Code, heartbeatRec.Body.String())
	}
}

func TestV2WorkerTokenReadsRunOwnerProfile(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	service := runcontrol.NewService(mem, eventbus.NewMemoryBus())
	router := NewRouter(ServerDeps{
		Version:     "test-version",
		Runs:        service,
		Store:       mem,
		WorkerToken: "trace-worker-secret",
		WorkOS:      testWorkOSAuth(t, WorkOSAuthConfig{}),
	})

	if _, err := mem.CreateUser(ctx, domain.CreateUserInput{UserID: "user-1", Role: "researcher", Status: "active"}); err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
	if _, err := mem.UpdateUserProfile(ctx, domain.UpdateUserProfileInput{
		UserID: "user-1",
		Profile: domain.UserProfile{
			DisplayName:       "Ada Lovelace",
			ResearchInterests: "symbolic computation",
		},
	}); err != nil {
		t.Fatalf("UpdateUserProfile: %v", err)
	}
	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "user-1", Title: "worker profile"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "profile run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "profile run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	// Anonymous (no worker token) is rejected by the WorkOS gate.
	anon := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/user-profile", nil)
	anonRec := httptest.NewRecorder()
	router.ServeHTTP(anonRec, anon)
	if anonRec.Code != http.StatusUnauthorized {
		t.Fatalf("anonymous profile status = %d, want 401", anonRec.Code)
	}

	// The worker token resolves the run owner's profile without a WorkOS session.
	req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/user-profile", nil)
	req.Header.Set("X-Ultra-Worker-Token", "trace-worker-secret")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("worker profile status = %d body=%s", rec.Code, rec.Body.String())
	}
	var resp struct {
		User struct {
			UserID string `json:"user_id"`
		} `json:"user"`
		Profile domain.UserProfile `json:"profile"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode profile: %v body=%s", err, rec.Body.String())
	}
	if resp.User.UserID != "user-1" {
		t.Fatalf("profile user = %q, want user-1", resp.User.UserID)
	}
	if resp.Profile.DisplayName != "Ada Lovelace" || resp.Profile.ResearchInterests != "symbolic computation" {
		t.Fatalf("worker-read profile = %+v, want Ada / symbolic computation", resp.Profile)
	}
}

func TestV2WorkerTokenBypassesWorkOSGateForRunEvents(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:     "test-version",
		Runs:        service,
		Store:       mem,
		WorkerToken: "trace-worker-secret",
		WorkOS:      testWorkOSAuth(t, WorkOSAuthConfig{}),
	})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "workos worker token run events",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "workos worker token run events",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "workos worker token run events"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	for idx := 0; idx < 3; idx++ {
		if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "message.delta",
			Message:   "chunk",
			Payload:   domain.JSONMap{"idx": idx},
		}); err != nil {
			t.Fatalf("AppendRunEvent %d: %v", idx, err)
		}
	}

	anonymous := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?limit=10&after_sequence=0", nil)
	anonymousRec := httptest.NewRecorder()
	router.ServeHTTP(anonymousRec, anonymous)
	if anonymousRec.Code != http.StatusUnauthorized {
		t.Fatalf("anonymous events under workos = %d, want 401", anonymousRec.Code)
	}

	wrongToken := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?limit=10&after_sequence=0", nil)
	wrongToken.Header.Set("Authorization", "Bearer wrong-secret")
	wrongTokenRec := httptest.NewRecorder()
	router.ServeHTTP(wrongTokenRec, wrongToken)
	if wrongTokenRec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong worker token events under workos = %d, want 401", wrongTokenRec.Code)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?limit=10&after_sequence=0", nil)
	req.Header.Set("X-Ultra-Worker-Token", "trace-worker-secret")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("worker token events under workos = %d body=%s", rec.Code, rec.Body.String())
	}

	var response runEventsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode events: %v", err)
	}
	if response.RunID != run.RunID || response.Count != 4 {
		t.Fatalf("response run/count = %s/%d, want %s/4", response.RunID, response.Count, run.RunID)
	}
	for idx, event := range response.Events {
		if event.Sequence != int64(idx+1) {
			t.Fatalf("event %d sequence = %d, want %d", idx, event.Sequence, idx+1)
		}
	}
}

func TestV2AdminOverviewIncludesRuntimeTransportSummary(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:      "test-version",
		Runs:         service,
		Store:        mem,
		ArtifactRoot: "/tmp/ultra-artifacts",
		UploadRoot:   "/tmp/ultra-uploads",
		Runtime: RuntimeSummary{
			AppVersion:              "test-version",
			StoreBackend:            "memory",
			DispatchMode:            "nats_jetstream",
			JobTransport:            "nats_jetstream",
			EventTransport:          "nats_jetstream_to_local_fanout",
			StubWorkerEnabled:       false,
			NATSConfigured:          true,
			NATSStream:              "ULTRA_RUNS",
			NATSJobsSubject:         "ultra.runs.jobs",
			NATSRareSpotJobsSubject: "ultra.runs.rarespot.jobs",
			NATSEventsSubject:       "ultra.runs.events",
			NATSCancelSubject:       "ultra.runs.cancel",
			NATSEventConsumer:       "ultra-control-event-ingest",
			ArtifactRoot:            "/tmp/ultra-artifacts",
			UploadRoot:              "/tmp/ultra-uploads",
		},
		QueueDiagnostics: fakeQueueDiagnosticsProvider{
			diagnostics: eventbus.QueueDiagnostics{
				Available:      true,
				Mode:           "nats_jetstream",
				Stream:         "ULTRA_RUNS",
				StreamSubjects: []string{"ultra.runs.jobs", "ultra.runs.events", "ultra.runs.cancel"},
				StreamMessages: 42,
				StreamBytes:    4096,
				FirstSequence:  10,
				LastSequence:   52,
				ConsumerCount:  2,
				Consumers: []eventbus.QueueConsumerDiagnostics{{
					Name:                "ultra-deepagents-worker",
					Role:                "deepagents",
					Subject:             "ultra.runs.jobs",
					Active:              true,
					AckWaitSeconds:      600,
					MaxDeliver:          5,
					PendingMessages:     3,
					InFlightMessages:    1,
					RedeliveredMessages: 2,
					WaitingPullRequests: 1,
				}},
			},
		},
	})

	req := httptest.NewRequest(http.MethodGet, "/v2/admin/overview", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin overview status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload struct {
		Runtime RuntimeSummary        `json:"runtime"`
		Queue   adminQueueDiagnostics `json:"queue"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode admin overview: %v", err)
	}
	runtime := payload.Runtime
	if runtime.DispatchMode != "nats_jetstream" || runtime.JobTransport != "nats_jetstream" {
		t.Fatalf("runtime transport = %+v, want nats dispatch/job transport", runtime)
	}
	if !runtime.NATSConfigured || runtime.NATSStream != "ULTRA_RUNS" || runtime.NATSJobsSubject != "ultra.runs.jobs" {
		t.Fatalf("nats runtime fields = %+v, want configured subjects", runtime)
	}
	if runtime.StubWorkerEnabled {
		t.Fatalf("stub worker enabled = true, want false for NATS runtime: %+v", runtime)
	}
	if runtime.ArtifactRoot == "" || runtime.UploadRoot == "" {
		t.Fatalf("runtime roots = %+v, want artifact/upload roots for operator diagnostics", runtime)
	}
	if !payload.Queue.Available || payload.Queue.Stream != "ULTRA_RUNS" || payload.Queue.StreamMessages != 42 {
		t.Fatalf("queue diagnostics = %+v, want stream health", payload.Queue)
	}
	if len(payload.Queue.Consumers) != 1 {
		t.Fatalf("queue consumers = %+v, want one worker consumer", payload.Queue.Consumers)
	}
	consumer := payload.Queue.Consumers[0]
	if consumer.Name != "ultra-deepagents-worker" || consumer.PendingMessages != 3 || consumer.InFlightMessages != 1 || consumer.RedeliveredMessages != 2 {
		t.Fatalf("consumer diagnostics = %+v, want pending/in-flight/redelivery counts", consumer)
	}
}

func TestV2AdminOverviewIncludesUploadSessionMetrics(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	router := NewRouter(ServerDeps{
		Version: "test-version",
		Runs:    runcontrol.NewService(mem, bus),
		Store:   mem,
		Bus:     bus,
	})

	for _, input := range []domain.CreateUploadSessionInput{
		{
			SessionID:      "upload_session_active",
			OwnerUserID:    "lab-user",
			OwnerOrgID:     "org-lab",
			Status:         "active",
			TotalBytes:     1000,
			BytesReceived:  512,
			BytesVerified:  256,
			BytesCommitted: 0,
		},
		{
			SessionID:      "upload_session_paused",
			OwnerUserID:    "lab-user",
			OwnerOrgID:     "org-lab",
			Status:         "paused",
			TotalBytes:     2000,
			BytesReceived:  1000,
			BytesVerified:  1000,
			BytesCommitted: 0,
		},
		{
			SessionID:     "upload_session_canceled",
			OwnerUserID:   "lab-user",
			OwnerOrgID:    "org-lab",
			Status:        "canceled",
			TotalBytes:    300,
			BytesReceived: 100,
		},
		{
			SessionID:      "upload_session_completed",
			OwnerUserID:    "lab-user",
			OwnerOrgID:     "org-lab",
			Status:         "completed",
			TotalBytes:     700,
			BytesReceived:  700,
			BytesVerified:  700,
			BytesCommitted: 700,
		},
		{
			SessionID:     "upload_session_failed",
			OwnerUserID:   "lab-user",
			OwnerOrgID:    "org-lab",
			Status:        "failed",
			TotalBytes:    400,
			BytesReceived: 128,
		},
	} {
		if _, err := mem.CreateUploadSession(ctx, input); err != nil {
			t.Fatalf("CreateUploadSession(%s): %v", input.SessionID, err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/admin/overview", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin overview status = %d body=%s", rec.Code, rec.Body.String())
	}

	var payload struct {
		UploadSessions struct {
			Total          int   `json:"total"`
			Active         int   `json:"active"`
			Paused         int   `json:"paused"`
			Completed      int   `json:"completed"`
			Failed         int   `json:"failed"`
			Canceled       int   `json:"canceled"`
			Other          int   `json:"other"`
			BytesTotal     int64 `json:"bytes_total"`
			BytesReceived  int64 `json:"bytes_received"`
			BytesVerified  int64 `json:"bytes_verified"`
			BytesCommitted int64 `json:"bytes_committed"`
		} `json:"upload_sessions"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode admin overview: %v", err)
	}
	metrics := payload.UploadSessions
	if metrics.Total != 5 || metrics.Active != 1 || metrics.Paused != 1 || metrics.Completed != 1 || metrics.Failed != 1 || metrics.Canceled != 1 || metrics.Other != 0 {
		t.Fatalf("upload session counts = %+v, want one session in each lifecycle state", metrics)
	}
	if metrics.BytesTotal != 4400 || metrics.BytesReceived != 2440 || metrics.BytesVerified != 1956 || metrics.BytesCommitted != 700 {
		t.Fatalf("upload session bytes = %+v, want summed total/received/verified/committed bytes", metrics)
	}
}

func TestV2AdminOverviewIncludesPeriodizedActivityMetrics(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	now := domain.Now()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	router := NewRouter(ServerDeps{
		Version: "test-version",
		Runs:    runcontrol.NewService(mem, bus),
		Store:   mem,
		Bus:     bus,
	})

	threadOne, err := mem.CreateThread(ctx, domain.CreateThreadInput{UserID: "user-lab-1", Title: "first lab thread"})
	if err != nil {
		t.Fatalf("CreateThread one: %v", err)
	}
	threadTwo, err := mem.CreateThread(ctx, domain.CreateThreadInput{UserID: "user-lab-2", Title: "second lab thread"})
	if err != nil {
		t.Fatalf("CreateThread two: %v", err)
	}
	for _, message := range []domain.ThreadMessage{
		{ThreadID: threadOne.ThreadID, Role: "user", Content: "daily prompt", CreatedAt: now.Add(-2 * time.Hour)},
		{ThreadID: threadOne.ThreadID, Role: "assistant", Content: "daily answer", CreatedAt: now.Add(-90 * time.Minute)},
		{ThreadID: threadTwo.ThreadID, Role: "user", Content: "daily second user", CreatedAt: now.Add(-30 * time.Minute)},
		{ThreadID: threadOne.ThreadID, Role: "user", Content: "weekly prompt", CreatedAt: now.Add(-3 * 24 * time.Hour)},
		{ThreadID: threadOne.ThreadID, Role: "user", Content: "monthly prompt", CreatedAt: now.Add(-20 * 24 * time.Hour)},
		{ThreadID: threadOne.ThreadID, Role: "user", Content: "older prompt", CreatedAt: now.Add(-45 * 24 * time.Hour)},
	} {
		if _, err := mem.AppendThreadMessage(ctx, message); err != nil {
			t.Fatalf("AppendThreadMessage: %v", err)
		}
	}

	run, err := mem.CreateRun(ctx, domain.CreateRunInput{
		ThreadID: threadOne.ThreadID,
		UserID:   "user-lab-1",
		Goal:     "tool metric run",
		Metadata: domain.JSONMap{"selected_tool_names": []string{"execute"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	for _, event := range []domain.AppendRunEventInput{
		{RunID: run.RunID, ThreadID: threadOne.ThreadID, EventKind: "tool_call.started", TS: now.Add(-2 * time.Hour), Payload: domain.JSONMap{"tool_name": "execute"}},
		{RunID: run.RunID, ThreadID: threadOne.ThreadID, EventKind: "tool_call.completed", TS: now.Add(-119 * time.Minute), Payload: domain.JSONMap{"tool_name": "execute"}},
		{RunID: run.RunID, ThreadID: threadOne.ThreadID, EventKind: "tool_call.started", TS: now.Add(-3 * 24 * time.Hour), Payload: domain.JSONMap{"tool_name": "python"}},
		{RunID: run.RunID, ThreadID: threadOne.ThreadID, EventKind: "tool_call.started", TS: now.Add(-20 * 24 * time.Hour), Payload: domain.JSONMap{"tool_name": "rarespot"}},
		{RunID: run.RunID, ThreadID: threadOne.ThreadID, EventKind: "tool_call.started", TS: now.Add(-45 * 24 * time.Hour), Payload: domain.JSONMap{"tool_name": "archive"}},
		{RunID: run.RunID, ThreadID: threadOne.ThreadID, EventKind: "artifact.created", TS: now.Add(-2 * time.Hour), Payload: domain.JSONMap{"artifact_id": "artifact-daily"}},
		{RunID: run.RunID, ThreadID: threadOne.ThreadID, EventKind: "artifact.created", TS: now.Add(-20 * 24 * time.Hour), Payload: domain.JSONMap{"artifact_id": "artifact-monthly"}},
	} {
		if _, err := mem.AppendRunEvent(ctx, event); err != nil {
			t.Fatalf("AppendRunEvent: %v", err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/admin/overview", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin overview status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload struct {
		Activity []struct {
			Label             string `json:"label"`
			Window            string `json:"window"`
			Messages          int    `json:"messages"`
			UserMessages      int    `json:"user_messages"`
			AssistantMessages int    `json:"assistant_messages"`
			ToolCalls         int    `json:"tool_calls"`
			ActiveUsers       int    `json:"active_users"`
			Runs              int    `json:"runs"`
			Artifacts         int    `json:"artifacts"`
		} `json:"activity"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode admin overview: %v", err)
	}
	activity := map[string]struct {
		Label             string `json:"label"`
		Window            string `json:"window"`
		Messages          int    `json:"messages"`
		UserMessages      int    `json:"user_messages"`
		AssistantMessages int    `json:"assistant_messages"`
		ToolCalls         int    `json:"tool_calls"`
		ActiveUsers       int    `json:"active_users"`
		Runs              int    `json:"runs"`
		Artifacts         int    `json:"artifacts"`
	}{}
	for _, period := range payload.Activity {
		activity[period.Label] = period
	}
	for _, label := range []string{"Daily", "Weekly", "Monthly", "Total"} {
		if _, ok := activity[label]; !ok {
			t.Fatalf("activity labels = %+v, want %s period", payload.Activity, label)
		}
	}
	if daily := activity["Daily"]; daily.Window != "24h" || daily.Messages != 3 || daily.UserMessages != 2 || daily.AssistantMessages != 1 || daily.ToolCalls != 1 || daily.ActiveUsers != 2 || daily.Artifacts != 1 {
		t.Fatalf("daily activity = %+v, want 3 messages, 1 tool call, 2 active users, 1 artifact", daily)
	}
	if weekly := activity["Weekly"]; weekly.Window != "7d" || weekly.Messages != 4 || weekly.ToolCalls != 2 || weekly.ActiveUsers != 2 || weekly.Artifacts != 1 {
		t.Fatalf("weekly activity = %+v, want cumulative 7d metrics", weekly)
	}
	if monthly := activity["Monthly"]; monthly.Window != "30d" || monthly.Messages != 5 || monthly.ToolCalls != 3 || monthly.ActiveUsers != 2 || monthly.Artifacts != 2 {
		t.Fatalf("monthly activity = %+v, want cumulative 30d metrics", monthly)
	}
	if total := activity["Total"]; total.Window != "all" || total.Messages != 6 || total.ToolCalls != 4 || total.ActiveUsers != 2 || total.Artifacts != 2 {
		t.Fatalf("total activity = %+v, want all-time metrics", total)
	}
}

func TestV2WorkerHeartbeatFeedsAdminOverview(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version: "test-version",
		Runs:    service,
		Store:   mem,
		Bus:     bus,
	})

	body := `{
		"worker_id":"deepagents-worker-a",
		"worker_kind":"deepagents",
		"status":"busy",
		"current_run_id":"run_123",
		"hostname":"host-a",
		"version":"worker-test-version",
		"metadata":{"active_tasks":1}
	}`
	req := httptest.NewRequest(http.MethodPost, "/v2/workers/heartbeat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("worker heartbeat status = %d body=%s", rec.Code, rec.Body.String())
	}
	var heartbeat domain.WorkerHeartbeatRecord
	if err := json.Unmarshal(rec.Body.Bytes(), &heartbeat); err != nil {
		t.Fatalf("decode heartbeat: %v", err)
	}
	if heartbeat.WorkerID != "deepagents-worker-a" || heartbeat.Status != "busy" || heartbeat.CurrentRunID != "run_123" {
		t.Fatalf("heartbeat response = %+v, want busy worker", heartbeat)
	}

	overviewReq := httptest.NewRequest(http.MethodGet, "/v2/admin/overview", nil)
	overviewRec := httptest.NewRecorder()
	router.ServeHTTP(overviewRec, overviewReq)
	if overviewRec.Code != http.StatusOK {
		t.Fatalf("admin overview status = %d body=%s", overviewRec.Code, overviewRec.Body.String())
	}
	var overview struct {
		Workers []adminWorkerRecord `json:"workers"`
	}
	if err := json.Unmarshal(overviewRec.Body.Bytes(), &overview); err != nil {
		t.Fatalf("decode admin overview: %v", err)
	}
	if len(overview.Workers) != 1 {
		t.Fatalf("workers = %+v, want one worker", overview.Workers)
	}
	worker := overview.Workers[0]
	if worker.WorkerID != "deepagents-worker-a" || worker.WorkerKind != "deepagents" || !worker.Active || worker.Stale {
		t.Fatalf("admin worker = %+v, want active deepagents worker", worker)
	}
	if worker.CurrentRunID == nil || *worker.CurrentRunID != "run_123" {
		t.Fatalf("admin worker current_run_id = %v, want run_123", worker.CurrentRunID)
	}
	if worker.HeartbeatAgeSeconds == nil || *worker.HeartbeatAgeSeconds > 5 {
		t.Fatalf("heartbeat age = %v, want fresh worker heartbeat", worker.HeartbeatAgeSeconds)
	}
}

func TestV2AdminSurfacesStaleRunningRunSignals(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "stale admin",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long autonomous run",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := mem.UpdateRunStatus(ctx, run.RunID, domain.RunStatusRunning, "", ""); err != nil {
		t.Fatalf("UpdateRunStatus: %v", err)
	}
	if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_stale_tool",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "tool_call.started",
		TS:        domain.Now().Add(-24 * time.Minute),
		Payload:   domain.JSONMap{"tool_name": "execute"},
	}); err != nil {
		t.Fatalf("AppendRunEvent tool: %v", err)
	}
	if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_stale_delta",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "message.delta",
		TS:        domain.Now().Add(-23 * time.Minute),
		Payload:   domain.JSONMap{"delta": "working"},
	}); err != nil {
		t.Fatalf("AppendRunEvent delta: %v", err)
	}
	if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_stale_artifact",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "artifact.created",
		TS:        domain.Now().Add(-22 * time.Minute),
		Payload:   domain.JSONMap{"artifact_id": "artifact-1"},
	}); err != nil {
		t.Fatalf("AppendRunEvent artifact: %v", err)
	}
	if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_stale_heartbeat",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.heartbeat",
		TS:        domain.Now().Add(-20 * time.Minute),
		Payload:   domain.JSONMap{"stage": "silent_compute"},
	}); err != nil {
		t.Fatalf("AppendRunEvent: %v", err)
	}

	runsReq := httptest.NewRequest(http.MethodGet, "/v2/admin/runs?status=running", nil)
	runsRec := httptest.NewRecorder()
	router.ServeHTTP(runsRec, runsReq)
	if runsRec.Code != http.StatusOK {
		t.Fatalf("admin runs status = %d body=%s", runsRec.Code, runsRec.Body.String())
	}
	var runsPayload struct {
		Runs []struct {
			RunID                string   `json:"run_id"`
			Status               string   `json:"status"`
			Stale                bool     `json:"stale"`
			StaleReason          *string  `json:"stale_reason"`
			LastEventKind        *string  `json:"last_event_kind"`
			LastEventAt          *string  `json:"last_event_at"`
			LastEventSequence    *int64   `json:"last_event_sequence"`
			LastActivitySeconds  *float64 `json:"last_activity_age_seconds"`
			EventCount           int      `json:"event_count"`
			MessageDeltaCount    int      `json:"message_delta_count"`
			ToolCallCount        int      `json:"tool_call_count"`
			ArtifactCount        int      `json:"artifact_count"`
			HeartbeatCount       int      `json:"heartbeat_count"`
			LastToolName         *string  `json:"last_tool_name"`
			LastToolAt           *string  `json:"last_tool_at"`
			FirstDeltaSeconds    *float64 `json:"first_delta_latency_seconds"`
			FirstToolSeconds     *float64 `json:"first_tool_latency_seconds"`
			FirstArtifactSeconds *float64 `json:"first_artifact_latency_seconds"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(runsRec.Body.Bytes(), &runsPayload); err != nil {
		t.Fatalf("decode admin runs: %v", err)
	}
	if len(runsPayload.Runs) != 1 {
		t.Fatalf("runs = %+v, want one stale running run", runsPayload.Runs)
	}
	record := runsPayload.Runs[0]
	if record.RunID != run.RunID || record.Status != "running" {
		t.Fatalf("run record = %+v, want running %s", record, run.RunID)
	}
	if !record.Stale || record.StaleReason == nil || !strings.Contains(*record.StaleReason, "No worker event") {
		t.Fatalf("stale fields = %+v, want stale worker-event reason", record)
	}
	if record.LastEventKind == nil || *record.LastEventKind != "run.heartbeat" {
		t.Fatalf("last_event_kind = %v, want run.heartbeat", record.LastEventKind)
	}
	if record.LastEventAt == nil || record.LastEventSequence == nil || *record.LastEventSequence < 1 || record.EventCount < 1 {
		t.Fatalf("event metadata = %+v, want latest event details", record)
	}
	if record.LastActivitySeconds == nil || *record.LastActivitySeconds < 600 {
		t.Fatalf("last_activity_age_seconds = %v, want stale age", record.LastActivitySeconds)
	}
	if record.MessageDeltaCount != 1 || record.ToolCallCount != 1 || record.ArtifactCount != 1 || record.HeartbeatCount != 1 {
		t.Fatalf("event counts = %+v, want one delta/tool/artifact/heartbeat", record)
	}
	if record.LastToolName == nil || *record.LastToolName != "execute" || record.LastToolAt == nil {
		t.Fatalf("last tool metadata = %+v, want execute", record)
	}
	if record.FirstDeltaSeconds == nil || record.FirstToolSeconds == nil || record.FirstArtifactSeconds == nil {
		t.Fatalf("first event latencies = %+v, want latency diagnostics", record)
	}

	issuesReq := httptest.NewRequest(http.MethodGet, "/v2/admin/issues", nil)
	issuesRec := httptest.NewRecorder()
	router.ServeHTTP(issuesRec, issuesReq)
	if issuesRec.Code != http.StatusOK {
		t.Fatalf("admin issues status = %d body=%s", issuesRec.Code, issuesRec.Body.String())
	}
	var issuesPayload struct {
		Issues []adminIssueRecord `json:"issues"`
	}
	if err := json.Unmarshal(issuesRec.Body.Bytes(), &issuesPayload); err != nil {
		t.Fatalf("decode admin issues: %v", err)
	}
	if len(issuesPayload.Issues) != 1 {
		t.Fatalf("issues = %+v, want one stalled_run issue", issuesPayload.Issues)
	}
	issue := issuesPayload.Issues[0]
	if issue.IssueType != "stalled_run" || issue.RunID != run.RunID || issue.Severity != "high" {
		t.Fatalf("issue = %+v, want high stalled_run for %s", issue, run.RunID)
	}
	if issue.Metadata["last_event_kind"] != "run.heartbeat" {
		t.Fatalf("issue metadata = %+v, want last_event_kind", issue.Metadata)
	}
}

func TestV2AdminRunsIncludeActiveRunLeaseOwnership(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-lease",
		Title:  "lease admin",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-lease",
		Goal:     "long autonomous run with a lease",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	lease, err := mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "deepagents-worker-a",
		TTL:      10 * time.Minute,
		Now:      domain.Now(),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/admin/runs?status=running", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin runs status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload struct {
		Runs []struct {
			RunID                      string   `json:"run_id"`
			LeaseWorkerID              *string  `json:"lease_worker_id"`
			LeaseExpiresAt             *string  `json:"lease_expires_at"`
			LeaseActive                bool     `json:"lease_active"`
			LeaseExpired               bool     `json:"lease_expired"`
			LeaseSecondsRemaining      *float64 `json:"lease_seconds_remaining"`
			LeaseLastRenewedAt         *string  `json:"lease_last_renewed_at"`
			LeaseLastRenewedAgeSeconds *float64 `json:"lease_last_renewed_age_seconds"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode admin runs: %v", err)
	}
	if len(payload.Runs) != 1 {
		t.Fatalf("runs = %+v, want one running run", payload.Runs)
	}
	record := payload.Runs[0]
	if record.RunID != run.RunID {
		t.Fatalf("run_id = %q, want %q", record.RunID, run.RunID)
	}
	if record.LeaseWorkerID == nil || *record.LeaseWorkerID != "deepagents-worker-a" {
		t.Fatalf("lease_worker_id = %v, want worker owner", record.LeaseWorkerID)
	}
	if record.LeaseExpiresAt == nil || *record.LeaseExpiresAt != lease.LeaseExpiresAt.UTC().Format(time.RFC3339Nano) {
		t.Fatalf("lease_expires_at = %v, want %s", record.LeaseExpiresAt, lease.LeaseExpiresAt.UTC().Format(time.RFC3339Nano))
	}
	if !record.LeaseActive || record.LeaseExpired {
		t.Fatalf("lease active/expired = %t/%t, want active non-expired", record.LeaseActive, record.LeaseExpired)
	}
	if record.LeaseSecondsRemaining == nil || *record.LeaseSecondsRemaining <= 0 {
		t.Fatalf("lease_seconds_remaining = %v, want positive", record.LeaseSecondsRemaining)
	}
	if record.LeaseLastRenewedAt == nil || record.LeaseLastRenewedAgeSeconds == nil {
		t.Fatalf("lease renewal fields missing: %+v", record)
	}
}

func TestV2AdminRunsFlagExpiredRunLeaseAsStale(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-lease",
		Title:  "expired lease admin",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-lease",
		Goal:     "long autonomous run with an expired lease",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	_, err = mem.AcquireRunLease(ctx, domain.AcquireRunLeaseInput{
		RunID:    run.RunID,
		WorkerID: "deepagents-worker-expired",
		TTL:      time.Minute,
		Now:      domain.Now().Add(-10 * time.Minute),
	})
	if err != nil {
		t.Fatalf("AcquireRunLease: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/admin/runs?status=running", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin runs status = %d body=%s", rec.Code, rec.Body.String())
	}
	var payload struct {
		Runs []struct {
			RunID                 string   `json:"run_id"`
			Stale                 bool     `json:"stale"`
			StaleReason           *string  `json:"stale_reason"`
			LeaseWorkerID         *string  `json:"lease_worker_id"`
			LeaseActive           bool     `json:"lease_active"`
			LeaseExpired          bool     `json:"lease_expired"`
			LeaseSecondsRemaining *float64 `json:"lease_seconds_remaining"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode admin runs: %v", err)
	}
	if len(payload.Runs) != 1 {
		t.Fatalf("runs = %+v, want one running run", payload.Runs)
	}
	record := payload.Runs[0]
	if record.RunID != run.RunID {
		t.Fatalf("run_id = %q, want %q", record.RunID, run.RunID)
	}
	if record.LeaseWorkerID == nil || *record.LeaseWorkerID != "deepagents-worker-expired" {
		t.Fatalf("lease_worker_id = %v, want expired worker owner", record.LeaseWorkerID)
	}
	if record.LeaseActive || !record.LeaseExpired {
		t.Fatalf("lease active/expired = %t/%t, want expired inactive", record.LeaseActive, record.LeaseExpired)
	}
	if record.LeaseSecondsRemaining == nil || *record.LeaseSecondsRemaining != 0 {
		t.Fatalf("lease_seconds_remaining = %v, want zero for expired lease", record.LeaseSecondsRemaining)
	}
	if !record.Stale || record.StaleReason == nil || !strings.Contains(*record.StaleReason, "lease expired") {
		t.Fatalf("stale fields = %+v, want expired lease reason", record)
	}
}

func TestV2UploadAndResourceHandlers(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       service,
		Store:      mem,
		UploadRoot: uploadRoot,
	})

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "prairie.jpg")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	pngBytes := testPNGBytes(t, 3, 2)
	if _, err := part.Write(pngBytes); err != nil {
		t.Fatalf("write multipart file: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}

	uploadReq := httptest.NewRequest(http.MethodPost, "/v2/uploads", &body)
	uploadReq.Header.Set("Content-Type", writer.FormDataContentType())
	uploadReq.Header.Set("X-Ultra-User-Id", "field-researcher")
	uploadReq.Header.Set("X-Ultra-Org-Id", "smithsonian")
	uploadReq.Header.Set("X-Ultra-Role", "admin")
	uploadRec := httptest.NewRecorder()
	router.ServeHTTP(uploadRec, uploadReq)
	if uploadRec.Code != http.StatusOK {
		t.Fatalf("upload status = %d body=%s", uploadRec.Code, uploadRec.Body.String())
	}
	var uploadResponse struct {
		FileCount int `json:"file_count"`
		Uploaded  []struct {
			FileID       string `json:"file_id"`
			OriginalName string `json:"original_name"`
			ContentType  string `json:"content_type"`
			SizeBytes    int64  `json:"size_bytes"`
			SHA256       string `json:"sha256"`
			PreviewURL   string `json:"preview_url"`
			Principal    struct {
				UserID string `json:"user_id"`
				OrgID  string `json:"org_id"`
				Role   string `json:"role"`
			} `json:"principal"`
		} `json:"uploaded"`
	}
	if err := json.Unmarshal(uploadRec.Body.Bytes(), &uploadResponse); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if uploadResponse.FileCount != 1 || len(uploadResponse.Uploaded) != 1 {
		t.Fatalf("upload response = %+v, want one uploaded file", uploadResponse)
	}
	uploaded := uploadResponse.Uploaded[0]
	if uploaded.FileID == "" || uploaded.OriginalName != "prairie.jpg" || uploaded.SHA256 == "" {
		t.Fatalf("uploaded metadata = %+v, want id/name/checksum", uploaded)
	}
	if uploaded.Principal.UserID != "field-researcher" || uploaded.Principal.OrgID != "smithsonian" || uploaded.Principal.Role != "admin" {
		t.Fatalf("uploaded principal = %+v, want request principal", uploaded.Principal)
	}

	matches, err := filepath.Glob(filepath.Join(uploadRoot, uploaded.FileID+"__*"))
	if err != nil {
		t.Fatalf("glob uploaded file: %v", err)
	}
	if len(matches) != 1 {
		t.Fatalf("uploaded files under root = %v, want one match for file id", matches)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20&kind=image&source=upload", nil)
	listReq.Header.Set("X-Ultra-User-Id", "field-researcher")
	listReq.Header.Set("X-Ultra-Org-Id", "smithsonian")
	listReq.Header.Set("X-Ultra-Role", "admin")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list resources status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var listResponse struct {
		Count     int `json:"count"`
		Resources []struct {
			FileID       string `json:"file_id"`
			OriginalName string `json:"original_name"`
			SourceType   string `json:"source_type"`
			ResourceKind string `json:"resource_kind"`
			PreviewURL   string `json:"preview_url"`
			Principal    struct {
				UserID string `json:"user_id"`
				OrgID  string `json:"org_id"`
				Role   string `json:"role"`
			} `json:"principal"`
		} `json:"resources"`
	}
	if err := json.Unmarshal(listRec.Body.Bytes(), &listResponse); err != nil {
		t.Fatalf("decode resources response: %v", err)
	}
	if listResponse.Count != 1 || listResponse.Resources[0].FileID != uploaded.FileID {
		t.Fatalf("resources = %+v, want uploaded resource", listResponse)
	}
	if listResponse.Resources[0].SourceType != "upload" || listResponse.Resources[0].ResourceKind != "image" {
		t.Fatalf("resource classification = %+v, want uploaded image", listResponse.Resources[0])
	}
	if listResponse.Resources[0].Principal.UserID != "field-researcher" || listResponse.Resources[0].Principal.OrgID != "smithsonian" || listResponse.Resources[0].Principal.Role != "admin" {
		t.Fatalf("resource principal = %+v, want upload principal", listResponse.Resources[0].Principal)
	}

	otherUserReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20&kind=image", nil)
	otherUserReq.Header.Set("X-Ultra-User-Id", "other-researcher")
	otherUserReq.Header.Set("X-Ultra-Org-Id", "smithsonian")
	otherUserRec := httptest.NewRecorder()
	router.ServeHTTP(otherUserRec, otherUserReq)
	if otherUserRec.Code != http.StatusOK {
		t.Fatalf("other-user list resources status = %d body=%s", otherUserRec.Code, otherUserRec.Body.String())
	}
	var otherUserResponse struct {
		Count     int              `json:"count"`
		Resources []resourceRecord `json:"resources"`
	}
	if err := json.Unmarshal(otherUserRec.Body.Bytes(), &otherUserResponse); err != nil {
		t.Fatalf("decode other-user resources response: %v", err)
	}
	if otherUserResponse.Count != 0 || len(otherUserResponse.Resources) != 0 {
		t.Fatalf("other-user resources = %+v, want no field-researcher uploads", otherUserResponse)
	}

	otherUserGetReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+uploaded.FileID, nil)
	otherUserGetReq.Header.Set("X-Ultra-User-Id", "other-researcher")
	otherUserGetReq.Header.Set("X-Ultra-Org-Id", "smithsonian")
	otherUserGetRec := httptest.NewRecorder()
	router.ServeHTTP(otherUserGetRec, otherUserGetReq)
	if otherUserGetRec.Code != http.StatusNotFound {
		t.Fatalf("other-user get status = %d body=%s, want 404", otherUserGetRec.Code, otherUserGetRec.Body.String())
	}

	otherUserDisplayReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+uploaded.FileID+"/display", nil)
	otherUserDisplayReq.Header.Set("X-Ultra-User-Id", "other-researcher")
	otherUserDisplayReq.Header.Set("X-Ultra-Org-Id", "smithsonian")
	otherUserDisplayRec := httptest.NewRecorder()
	router.ServeHTTP(otherUserDisplayRec, otherUserDisplayReq)
	if otherUserDisplayRec.Code != http.StatusNotFound {
		t.Fatalf("other-user display status = %d body=%s, want 404", otherUserDisplayRec.Code, otherUserDisplayRec.Body.String())
	}

	bisqueFilterReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20&source=bisque_import", nil)
	bisqueFilterReq.Header.Set("X-Ultra-User-Id", "field-researcher")
	bisqueFilterReq.Header.Set("X-Ultra-Org-Id", "smithsonian")
	bisqueFilterRec := httptest.NewRecorder()
	router.ServeHTTP(bisqueFilterRec, bisqueFilterReq)
	if bisqueFilterRec.Code != http.StatusOK {
		t.Fatalf("bisque-filter list resources status = %d body=%s", bisqueFilterRec.Code, bisqueFilterRec.Body.String())
	}
	var bisqueFilterResponse struct {
		Count     int              `json:"count"`
		Resources []resourceRecord `json:"resources"`
	}
	if err := json.Unmarshal(bisqueFilterRec.Body.Bytes(), &bisqueFilterResponse); err != nil {
		t.Fatalf("decode bisque-filter resources response: %v", err)
	}
	if bisqueFilterResponse.Count != 0 || len(bisqueFilterResponse.Resources) != 0 {
		t.Fatalf("bisque-filter resources = %+v, want no local uploads", bisqueFilterResponse)
	}

	displayReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+uploaded.FileID+"/display", nil)
	displayReq.Header.Set("X-Ultra-User-Id", "field-researcher")
	displayReq.Header.Set("X-Ultra-Org-Id", "smithsonian")
	displayRec := httptest.NewRecorder()
	router.ServeHTTP(displayRec, displayReq)
	if displayRec.Code != http.StatusOK {
		t.Fatalf("display status = %d body=%s", displayRec.Code, displayRec.Body.String())
	}
	if !bytes.Equal(displayRec.Body.Bytes(), pngBytes) {
		t.Fatalf("display body = %q, want uploaded PNG bytes", displayRec.Body.String())
	}

	sliceReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+uploaded.FileID+"/slice?axis=z&z=0", nil)
	sliceReq.Header.Set("X-Ultra-User-Id", "field-researcher")
	sliceReq.Header.Set("X-Ultra-Org-Id", "smithsonian")
	sliceRec := httptest.NewRecorder()
	router.ServeHTTP(sliceRec, sliceReq)
	if sliceRec.Code != http.StatusOK {
		t.Fatalf("slice status = %d body=%s", sliceRec.Code, sliceRec.Body.String())
	}
	if !bytes.Equal(sliceRec.Body.Bytes(), pngBytes) {
		t.Fatalf("slice body = %q, want uploaded PNG bytes", sliceRec.Body.String())
	}

	viewerReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+uploaded.FileID+"/viewer", nil)
	viewerReq.Header.Set("X-Ultra-User-Id", "field-researcher")
	viewerReq.Header.Set("X-Ultra-Org-Id", "smithsonian")
	viewerRec := httptest.NewRecorder()
	router.ServeHTTP(viewerRec, viewerReq)
	if viewerRec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", viewerRec.Code, viewerRec.Body.String())
	}
	var viewerResponse struct {
		Kind         string `json:"kind"`
		FileID       string `json:"file_id"`
		OriginalName string `json:"original_name"`
		AxisSizes    struct {
			X int `json:"X"`
			Y int `json:"Y"`
			Z int `json:"Z"`
			C int `json:"C"`
			T int `json:"T"`
		} `json:"axis_sizes"`
		ServiceURLs struct {
			Display   string `json:"display"`
			Preview   string `json:"preview"`
			Histogram string `json:"histogram"`
		} `json:"service_urls"`
	}
	if err := json.Unmarshal(viewerRec.Body.Bytes(), &viewerResponse); err != nil {
		t.Fatalf("decode viewer response: %v", err)
	}
	if viewerResponse.Kind != "image" || viewerResponse.FileID != uploaded.FileID || viewerResponse.OriginalName != "prairie.jpg" {
		t.Fatalf("viewer identity = %+v, want uploaded image metadata", viewerResponse)
	}
	if viewerResponse.AxisSizes.X != 3 || viewerResponse.AxisSizes.Y != 2 || viewerResponse.AxisSizes.Z != 1 || viewerResponse.AxisSizes.T != 1 {
		t.Fatalf("viewer axis sizes = %+v, want 3x2 image", viewerResponse.AxisSizes)
	}
	if viewerResponse.ServiceURLs.Display != "/v2/uploads/"+uploaded.FileID+"/display" {
		t.Fatalf("viewer display URL = %q, want V2 display URL", viewerResponse.ServiceURLs.Display)
	}
	if viewerResponse.ServiceURLs.Histogram != "/v2/uploads/"+uploaded.FileID+"/histogram" {
		t.Fatalf("viewer histogram URL = %q, want V2 histogram URL", viewerResponse.ServiceURLs.Histogram)
	}

	captionReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+uploaded.FileID+"/caption", nil)
	captionReq.Header.Set("X-Ultra-User-Id", "field-researcher")
	captionReq.Header.Set("X-Ultra-Org-Id", "smithsonian")
	captionRec := httptest.NewRecorder()
	router.ServeHTTP(captionRec, captionReq)
	if captionRec.Code != http.StatusOK {
		t.Fatalf("caption status = %d body=%s", captionRec.Code, captionRec.Body.String())
	}
	var captionResponse struct {
		FileID  string `json:"file_id"`
		Caption string `json:"caption"`
		Source  string `json:"source"`
	}
	if err := json.Unmarshal(captionRec.Body.Bytes(), &captionResponse); err != nil {
		t.Fatalf("decode caption response: %v", err)
	}
	if captionResponse.FileID != uploaded.FileID || !strings.Contains(captionResponse.Caption, "prairie.jpg") || captionResponse.Source != "fallback" {
		t.Fatalf("caption response = %+v, want fallback caption for uploaded image", captionResponse)
	}
}

func TestV2ResourceDownloadServesOriginalBlobWithResourceScope(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	uploadedBytes := []byte("scientific field note bytes\n")
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "field-note.txt")
	if err != nil {
		t.Fatalf("create multipart file: %v", err)
	}
	if _, err := part.Write(uploadedBytes); err != nil {
		t.Fatalf("write multipart file: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	uploadReq := httptest.NewRequest(http.MethodPost, "/v2/uploads", &body)
	uploadReq.Header.Set("Content-Type", writer.FormDataContentType())
	uploadReq.Header.Set("X-Ultra-User-Id", "alice")
	uploadReq.Header.Set("X-Ultra-Org-Id", "org-a")
	uploadRec := httptest.NewRecorder()
	router.ServeHTTP(uploadRec, uploadReq)
	if uploadRec.Code != http.StatusOK {
		t.Fatalf("upload status = %d body=%s", uploadRec.Code, uploadRec.Body.String())
	}
	var uploadResponse uploadFilesResponse
	if err := json.Unmarshal(uploadRec.Body.Bytes(), &uploadResponse); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if len(uploadResponse.Uploaded) != 1 {
		t.Fatalf("uploaded = %+v, want one file", uploadResponse.Uploaded)
	}
	fileID := uploadResponse.Uploaded[0].FileID

	aliceDownloadReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/download", nil)
	aliceDownloadReq.Header.Set("X-Ultra-User-Id", "alice")
	aliceDownloadReq.Header.Set("X-Ultra-Org-Id", "org-a")
	aliceDownloadRec := httptest.NewRecorder()
	router.ServeHTTP(aliceDownloadRec, aliceDownloadReq)
	if aliceDownloadRec.Code != http.StatusOK {
		t.Fatalf("alice download status = %d body=%s", aliceDownloadRec.Code, aliceDownloadRec.Body.String())
	}
	if !bytes.Equal(aliceDownloadRec.Body.Bytes(), uploadedBytes) {
		t.Fatalf("alice download body = %q, want original bytes", aliceDownloadRec.Body.String())
	}
	disposition := aliceDownloadRec.Header().Get("Content-Disposition")
	if !strings.Contains(disposition, "attachment") || !strings.Contains(disposition, "field-note.txt") {
		t.Fatalf("content disposition = %q, want attachment filename", disposition)
	}

	otherDownloadReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/download", nil)
	otherDownloadReq.Header.Set("X-Ultra-User-Id", "charlie")
	otherDownloadReq.Header.Set("X-Ultra-Org-Id", "org-c")
	otherDownloadRec := httptest.NewRecorder()
	router.ServeHTTP(otherDownloadRec, otherDownloadReq)
	if otherDownloadRec.Code != http.StatusNotFound {
		t.Fatalf("other download status = %d body=%s, want 404", otherDownloadRec.Code, otherDownloadRec.Body.String())
	}

	shareBody := strings.NewReader(`{"grantee_user_id":"bob","grantee_org_id":"org-b","role":"read"}`)
	shareReq := httptest.NewRequest(http.MethodPost, "/v2/resources/"+fileID+"/shares", shareBody)
	shareReq.Header.Set("Content-Type", "application/json")
	shareReq.Header.Set("X-Ultra-User-Id", "alice")
	shareReq.Header.Set("X-Ultra-Org-Id", "org-a")
	shareRec := httptest.NewRecorder()
	router.ServeHTTP(shareRec, shareReq)
	if shareRec.Code != http.StatusCreated {
		t.Fatalf("share status = %d body=%s, want 201", shareRec.Code, shareRec.Body.String())
	}
	bobDownloadReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/download", nil)
	bobDownloadReq.Header.Set("X-Ultra-User-Id", "bob")
	bobDownloadReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobDownloadRec := httptest.NewRecorder()
	router.ServeHTTP(bobDownloadRec, bobDownloadReq)
	if bobDownloadRec.Code != http.StatusOK {
		t.Fatalf("bob download status = %d body=%s", bobDownloadRec.Code, bobDownloadRec.Body.String())
	}
	if !bytes.Equal(bobDownloadRec.Body.Bytes(), uploadedBytes) {
		t.Fatalf("bob download body = %q, want original bytes", bobDownloadRec.Body.String())
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, "/v2/resources/"+fileID, nil)
	deleteReq.Header.Set("X-Ultra-User-Id", "alice")
	deleteReq.Header.Set("X-Ultra-Org-Id", "org-a")
	deleteRec := httptest.NewRecorder()
	router.ServeHTTP(deleteRec, deleteReq)
	if deleteRec.Code != http.StatusOK {
		t.Fatalf("delete status = %d body=%s", deleteRec.Code, deleteRec.Body.String())
	}
	deletedDownloadReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/download", nil)
	deletedDownloadReq.Header.Set("X-Ultra-User-Id", "alice")
	deletedDownloadReq.Header.Set("X-Ultra-Org-Id", "org-a")
	deletedDownloadRec := httptest.NewRecorder()
	router.ServeHTTP(deletedDownloadRec, deletedDownloadReq)
	if deletedDownloadRec.Code != http.StatusNotFound {
		t.Fatalf("deleted download status = %d body=%s, want 404", deletedDownloadRec.Code, deletedDownloadRec.Body.String())
	}
}

func TestV2ResourceDownloadRejectsCatalogPathOutsideUploadRoot(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	outsideRoot := t.TempDir()
	outsidePath := filepath.Join(outsideRoot, "private-note.txt")
	outsideBytes := []byte("must not escape upload root")
	if err := os.WriteFile(outsidePath, outsideBytes, 0o644); err != nil {
		t.Fatalf("write outside fixture: %v", err)
	}
	sum := sha256.Sum256(outsideBytes)
	mem := store.NewMemoryStore()
	if _, err := mem.UpsertResource(context.Background(), domain.UpsertResourceInput{
		ResourceID:   "file_outside_root",
		OriginalName: "private-note.txt",
		ContentType:  "text/plain",
		SizeBytes:    int64(len(outsideBytes)),
		SHA256:       hex.EncodeToString(sum[:]),
		StorageURI:   fileStorageURI(outsidePath),
		StoragePath:  outsidePath,
		SourceType:   "upload",
		ResourceKind: "table",
		OwnerUserID:  "alice",
		OwnerOrgID:   "org-a",
		OwnerRole:    "researcher",
		Status:       "active",
		CreatedAt:    domain.Now(),
		UpdatedAt:    domain.Now(),
	}); err != nil {
		t.Fatalf("catalog outside resource: %v", err)
	}
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	downloadReq := httptest.NewRequest(http.MethodGet, "/v2/resources/file_outside_root/download", nil)
	downloadReq.Header.Set("X-Ultra-User-Id", "alice")
	downloadReq.Header.Set("X-Ultra-Org-Id", "org-a")
	downloadRec := httptest.NewRecorder()
	router.ServeHTTP(downloadRec, downloadReq)
	if downloadRec.Code != http.StatusNotFound {
		t.Fatalf("outside-root download status = %d body=%s, want 404", downloadRec.Code, downloadRec.Body.String())
	}
}

func TestResourceRecordFromCatalogExposesDataAgentMetadata(t *testing.T) {
	t.Parallel()

	createdAt := time.Date(2026, 6, 8, 14, 30, 0, 0, time.UTC)
	record := (ServerDeps{}).resourceRecordFromCatalog(t.TempDir(), domain.ResourceRecord{
		ResourceID:   "file_agent_captioned",
		OriginalName: "prairie-cell-image.png",
		ContentType:  "image/png",
		SizeBytes:    42,
		SHA256:       "abc123",
		SourceType:   "upload",
		ResourceKind: "image",
		OwnerUserID:  "field-scientist",
		OwnerOrgID:   "lab-alpha",
		CreatedAt:    createdAt,
		Metadata: domain.JSONMap{
			"label": "NPH",
			"data_agent": domain.JSONMap{
				"caption_resources": domain.JSONMap{
					"status":         "succeeded",
					"job_id":         "data_agent_job_caption",
					"summary_kind":   "caption_generation",
					"caption":        "Prairie microscopy image with deterministic metadata caption.",
					"caption_source": "deterministic_metadata",
					"completed_at":   createdAt.Format(time.RFC3339Nano),
				},
			},
		},
	})

	payload, err := json.Marshal(resourcesResponse{Count: 1, Resources: []resourceRecord{record}})
	if err != nil {
		t.Fatalf("marshal resources response: %v", err)
	}
	var decoded resourcesResponse
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("decode resources response: %v", err)
	}
	if decoded.Count != 1 || len(decoded.Resources) != 1 {
		t.Fatalf("decoded resources = %+v, want one resource", decoded)
	}
	metadata := decoded.Resources[0].Metadata
	if metadata["label"] != "NPH" {
		t.Fatalf("metadata label = %#v, want preserved NPH label", metadata["label"])
	}
	agent, ok := metadata["data_agent"].(map[string]any)
	if !ok {
		t.Fatalf("metadata data_agent = %#v, want object", metadata["data_agent"])
	}
	caption, ok := agent["caption_resources"].(map[string]any)
	if !ok {
		t.Fatalf("caption_resources = %#v, want object", agent["caption_resources"])
	}
	if caption["status"] != "succeeded" || caption["caption"] == "" || caption["job_id"] != "data_agent_job_caption" {
		t.Fatalf("caption_resources metadata = %#v, want persisted caption job state", caption)
	}
}

func TestV2ResourcesSearchMatchesDataAgentMetadata(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	createdAt := time.Date(2026, 6, 8, 15, 30, 0, 0, time.UTC)
	if _, err := mem.UpsertResource(context.Background(), domain.UpsertResourceInput{
		ResourceID:   "file_metadata_search",
		OwnerUserID:  "metadata-user",
		OwnerOrgID:   "metadata-org",
		OriginalName: "image-001.png",
		ContentType:  "image/png",
		SizeBytes:    128,
		SHA256:       "sha-metadata-search",
		SourceType:   "upload",
		ResourceKind: "image",
		Status:       "active",
		CreatedAt:    createdAt,
		UpdatedAt:    createdAt,
		Metadata: domain.JSONMap{
			"label": "NPH",
			"data_agent": domain.JSONMap{
				"caption_resources": domain.JSONMap{
					"status":  "succeeded",
					"caption": "Prairie microscopy image with deterministic metadata caption.",
				},
			},
		},
	}); err != nil {
		t.Fatalf("UpsertResource metadata search: %v", err)
	}
	if _, err := mem.UpsertResource(context.Background(), domain.UpsertResourceInput{
		ResourceID:   "file_metadata_other",
		OwnerUserID:  "metadata-user",
		OwnerOrgID:   "metadata-org",
		OriginalName: "control-image.png",
		ContentType:  "image/png",
		SizeBytes:    64,
		SHA256:       "sha-metadata-other",
		SourceType:   "upload",
		ResourceKind: "image",
		Status:       "active",
		CreatedAt:    createdAt.Add(time.Second),
		UpdatedAt:    createdAt.Add(time.Second),
		Metadata:     domain.JSONMap{"label": "Control"},
	}); err != nil {
		t.Fatalf("UpsertResource other: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/resources?q=deterministic%20metadata%20caption&limit=20", nil)
	req.Header.Set("X-Ultra-User-Id", "metadata-user")
	req.Header.Set("X-Ultra-Org-Id", "metadata-org")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("metadata search status = %d body=%s", rec.Code, rec.Body.String())
	}
	var response resourcesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode metadata search response: %v", err)
	}
	if response.Count != 1 || len(response.Resources) != 1 || response.Resources[0].FileID != "file_metadata_search" {
		t.Fatalf("metadata search response = %+v, want captioned resource only", response)
	}
	if response.Resources[0].Metadata["label"] != "NPH" {
		t.Fatalf("metadata search result metadata = %#v, want exposed NPH metadata", response.Resources[0].Metadata)
	}
}

func TestV2ResourcesFilterScientificMetadata(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	createdAt := time.Date(2026, 6, 9, 14, 0, 0, 0, time.UTC)
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_filter_nph_under_70",
			OwnerUserID:  "metadata-filter-user",
			OwnerOrgID:   "metadata-filter-org",
			OriginalName: "nph-under-70.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    256,
			SHA256:       "sha-filter-nph-under",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    createdAt,
			UpdatedAt:    createdAt,
			Metadata: domain.JSONMap{
				"label":       "NPH",
				"format":      "nifti",
				"subject_age": float64(68),
			},
		},
		{
			ResourceID:   "file_filter_nph_over_70",
			OwnerUserID:  "metadata-filter-user",
			OwnerOrgID:   "metadata-filter-org",
			OriginalName: "nph-over-70.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    256,
			SHA256:       "sha-filter-nph-over",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    createdAt.Add(time.Second),
			UpdatedAt:    createdAt.Add(time.Second),
			Metadata: domain.JSONMap{
				"label":       "NPH",
				"format":      "nifti",
				"subject_age": float64(73),
			},
		},
		{
			ResourceID:   "file_filter_control_under_70",
			OwnerUserID:  "metadata-filter-user",
			OwnerOrgID:   "metadata-filter-org",
			OriginalName: "control-under-70.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    256,
			SHA256:       "sha-filter-control",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    createdAt.Add(2 * time.Second),
			UpdatedAt:    createdAt.Add(2 * time.Second),
			Metadata: domain.JSONMap{
				"label":       "control",
				"format":      "nifti",
				"subject_age": float64(64),
			},
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/resources?metadata_filter=label:eq:NPH&metadata_filter=format:eq:nifti&metadata_filter=subject_age:lt:70&limit=20", nil)
	req.Header.Set("X-Ultra-User-Id", "metadata-filter-user")
	req.Header.Set("X-Ultra-Org-Id", "metadata-filter-org")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("metadata filter status = %d body=%s", rec.Code, rec.Body.String())
	}
	var response resourcesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode metadata filter response: %v", err)
	}
	if response.Count != 1 || len(response.Resources) != 1 || response.Resources[0].FileID != "file_filter_nph_under_70" {
		t.Fatalf("metadata filter response = %+v, want only NPH under-70 NIfTI", response)
	}
}

func TestV2ResourcesFilterScientificDescriptors(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	createdAt := time.Date(2026, 6, 9, 14, 15, 0, 0, time.UTC)
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_descriptor_nph",
			OwnerUserID:  "descriptor-filter-user",
			OwnerOrgID:   "descriptor-filter-org",
			OriginalName: "nph-ventriculomegaly.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    256,
			SHA256:       "sha-descriptor-nph",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    createdAt,
			UpdatedAt:    createdAt,
			Metadata: domain.JSONMap{
				"label":                  "NPH",
				"scientific_descriptors": []any{"ventriculomegaly", "MRI cohort"},
				"data_agent": domain.JSONMap{
					"extract_metadata": domain.JSONMap{
						"status":      "succeeded",
						"descriptors": []any{"Evans index high"},
					},
				},
			},
		},
		{
			ResourceID:   "file_descriptor_control",
			OwnerUserID:  "descriptor-filter-user",
			OwnerOrgID:   "descriptor-filter-org",
			OriginalName: "control-normal.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    256,
			SHA256:       "sha-descriptor-control",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    createdAt.Add(time.Second),
			UpdatedAt:    createdAt.Add(time.Second),
			Metadata: domain.JSONMap{
				"label":                  "control",
				"scientific_descriptors": []any{"normal ventricles", "MRI cohort"},
				"data_agent": domain.JSONMap{
					"extract_metadata": domain.JSONMap{
						"status":      "succeeded",
						"descriptors": []any{"Evans index normal"},
					},
				},
			},
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/resources?descriptor=ventriculomegaly&descriptors=Evans%20index%20high&limit=20", nil)
	req.Header.Set("X-Ultra-User-Id", "descriptor-filter-user")
	req.Header.Set("X-Ultra-Org-Id", "descriptor-filter-org")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("descriptor filter status = %d body=%s", rec.Code, rec.Body.String())
	}
	var response resourcesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode descriptor filter response: %v", err)
	}
	if response.Count != 1 || len(response.Resources) != 1 || response.Resources[0].FileID != "file_descriptor_nph" {
		t.Fatalf("descriptor filter response = %+v, want only NPH descriptor resource", response)
	}
}

func TestV2ResourcesFilterProcessingStatus(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	createdAt := time.Date(2026, 6, 9, 14, 30, 0, 0, time.UTC)
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_http_caption_ready",
			OwnerUserID:  "processing-filter-user",
			OwnerOrgID:   "processing-filter-org",
			OriginalName: "caption-ready.nii.gz",
			SizeBytes:    256,
			SHA256:       "sha-processing-caption",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    createdAt,
			UpdatedAt:    createdAt,
			Metadata: domain.JSONMap{
				"data_agent": domain.JSONMap{
					"caption_resources": domain.JSONMap{"status": "succeeded"},
				},
			},
		},
		{
			ResourceID:   "file_http_metadata_ready",
			OwnerUserID:  "processing-filter-user",
			OwnerOrgID:   "processing-filter-org",
			OriginalName: "metadata-ready.nii.gz",
			SizeBytes:    256,
			SHA256:       "sha-processing-metadata",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    createdAt.Add(time.Second),
			UpdatedAt:    createdAt.Add(time.Second),
			Metadata: domain.JSONMap{
				"data_agent": domain.JSONMap{
					"extract_metadata": domain.JSONMap{"status": "succeeded"},
				},
			},
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/resources?processing_status=caption_ready&limit=20", nil)
	req.Header.Set("X-Ultra-User-Id", "processing-filter-user")
	req.Header.Set("X-Ultra-Org-Id", "processing-filter-org")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("processing filter status = %d body=%s", rec.Code, rec.Body.String())
	}
	var response resourcesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode processing filter response: %v", err)
	}
	if response.Count != 1 || len(response.Resources) != 1 || response.Resources[0].FileID != "file_http_caption_ready" {
		t.Fatalf("processing filter response = %+v, want only caption-ready resource", response)
	}
}

func TestV2ResourcesFilterCreatedDate(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	createdAt := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_before_date_window",
			OwnerUserID:  "date-filter-user",
			OwnerOrgID:   "date-filter-org",
			OriginalName: "before-window.nii.gz",
			SizeBytes:    256,
			SHA256:       "sha-date-before",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    createdAt,
			UpdatedAt:    createdAt,
		},
		{
			ResourceID:   "file_inside_date_window",
			OwnerUserID:  "date-filter-user",
			OwnerOrgID:   "date-filter-org",
			OriginalName: "inside-window.nii.gz",
			SizeBytes:    256,
			SHA256:       "sha-date-inside",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    createdAt.Add(48 * time.Hour),
			UpdatedAt:    createdAt.Add(48 * time.Hour),
		},
		{
			ResourceID:   "file_after_date_window",
			OwnerUserID:  "date-filter-user",
			OwnerOrgID:   "date-filter-org",
			OriginalName: "after-window.nii.gz",
			SizeBytes:    256,
			SHA256:       "sha-date-after",
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			CreatedAt:    createdAt.Add(96 * time.Hour),
			UpdatedAt:    createdAt.Add(96 * time.Hour),
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/resources?created_after=2026-06-02T00:00:00Z&created_before=2026-06-04T23:59:59Z&limit=20", nil)
	req.Header.Set("X-Ultra-User-Id", "date-filter-user")
	req.Header.Set("X-Ultra-Org-Id", "date-filter-org")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("date filter status = %d body=%s", rec.Code, rec.Body.String())
	}
	var response resourcesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode date filter response: %v", err)
	}
	if response.Count != 1 || len(response.Resources) != 1 || response.Resources[0].FileID != "file_inside_date_window" {
		t.Fatalf("date filter response = %+v, want only inside-window resource", response)
	}
}

func TestV2ResourceShareGrantAllowsReadAccessForCollaborator(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "shared-nph-study.png")
	if err != nil {
		t.Fatalf("create multipart file: %v", err)
	}
	uploadedBytes := testPNGBytes(t, 3, 2)
	if _, err := part.Write(uploadedBytes); err != nil {
		t.Fatalf("write multipart file: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	uploadReq := httptest.NewRequest(http.MethodPost, "/v2/uploads", &body)
	uploadReq.Header.Set("Content-Type", writer.FormDataContentType())
	uploadReq.Header.Set("X-Ultra-User-Id", "alice")
	uploadReq.Header.Set("X-Ultra-Org-Id", "org-a")
	uploadReq.Header.Set("X-Ultra-Role", "researcher")
	uploadRec := httptest.NewRecorder()
	router.ServeHTTP(uploadRec, uploadReq)
	if uploadRec.Code != http.StatusOK {
		t.Fatalf("upload status = %d body=%s", uploadRec.Code, uploadRec.Body.String())
	}
	var uploadResponse uploadFilesResponse
	if err := json.Unmarshal(uploadRec.Body.Bytes(), &uploadResponse); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if len(uploadResponse.Uploaded) != 1 {
		t.Fatalf("uploaded = %+v, want one file", uploadResponse.Uploaded)
	}
	fileID := uploadResponse.Uploaded[0].FileID

	bobBeforeReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20", nil)
	bobBeforeReq.Header.Set("X-Ultra-User-Id", "bob")
	bobBeforeReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobBeforeRec := httptest.NewRecorder()
	router.ServeHTTP(bobBeforeRec, bobBeforeReq)
	if bobBeforeRec.Code != http.StatusOK {
		t.Fatalf("bob before list status = %d body=%s", bobBeforeRec.Code, bobBeforeRec.Body.String())
	}
	var bobBefore resourcesResponse
	if err := json.Unmarshal(bobBeforeRec.Body.Bytes(), &bobBefore); err != nil {
		t.Fatalf("decode bob before resources: %v", err)
	}
	if bobBefore.Count != 0 || len(bobBefore.Resources) != 0 {
		t.Fatalf("bob resources before share = %+v, want none", bobBefore)
	}

	shareBody := strings.NewReader(`{"grantee_user_id":"bob","grantee_org_id":"org-b","role":"read","metadata":{"reason":"collaborative review"}}`)
	shareReq := httptest.NewRequest(http.MethodPost, "/v2/resources/"+fileID+"/shares", shareBody)
	shareReq.Header.Set("Content-Type", "application/json")
	shareReq.Header.Set("X-Ultra-User-Id", "alice")
	shareReq.Header.Set("X-Ultra-Org-Id", "org-a")
	shareRec := httptest.NewRecorder()
	router.ServeHTTP(shareRec, shareReq)
	if shareRec.Code != http.StatusCreated {
		t.Fatalf("share status = %d body=%s, want 201", shareRec.Code, shareRec.Body.String())
	}
	var shareResponse struct {
		Grant domain.ResourceShareGrantRecord `json:"grant"`
	}
	if err := json.Unmarshal(shareRec.Body.Bytes(), &shareResponse); err != nil {
		t.Fatalf("decode share response: %v", err)
	}
	if shareResponse.Grant.ResourceID != fileID || shareResponse.Grant.GranteeUserID != "bob" || shareResponse.Grant.Role != "read" || shareResponse.Grant.Status != "active" {
		t.Fatalf("share response = %+v, want active read grant for Bob", shareResponse.Grant)
	}

	bobListReq := httptest.NewRequest(http.MethodGet, "/v2/resources?q=shared-nph&limit=20", nil)
	bobListReq.Header.Set("X-Ultra-User-Id", "bob")
	bobListReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobListRec := httptest.NewRecorder()
	router.ServeHTTP(bobListRec, bobListReq)
	if bobListRec.Code != http.StatusOK {
		t.Fatalf("bob list status = %d body=%s", bobListRec.Code, bobListRec.Body.String())
	}
	var bobList resourcesResponse
	if err := json.Unmarshal(bobListRec.Body.Bytes(), &bobList); err != nil {
		t.Fatalf("decode bob resources: %v", err)
	}
	if bobList.Count != 1 || len(bobList.Resources) != 1 || bobList.Resources[0].FileID != fileID {
		t.Fatalf("bob resources after share = %+v, want shared resource", bobList)
	}
	if bobList.Resources[0].Principal.UserID != "alice" {
		t.Fatalf("shared resource principal = %+v, want Alice owner preserved", bobList.Resources[0].Principal)
	}

	bobGetReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID, nil)
	bobGetReq.Header.Set("X-Ultra-User-Id", "bob")
	bobGetReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobGetRec := httptest.NewRecorder()
	router.ServeHTTP(bobGetRec, bobGetReq)
	if bobGetRec.Code != http.StatusOK {
		t.Fatalf("bob get status = %d body=%s", bobGetRec.Code, bobGetRec.Body.String())
	}

	bobDisplayReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/display", nil)
	bobDisplayReq.Header.Set("X-Ultra-User-Id", "bob")
	bobDisplayReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobDisplayRec := httptest.NewRecorder()
	router.ServeHTTP(bobDisplayRec, bobDisplayReq)
	if bobDisplayRec.Code != http.StatusOK {
		t.Fatalf("bob display status = %d body=%s", bobDisplayRec.Code, bobDisplayRec.Body.String())
	}
	if !bytes.Equal(bobDisplayRec.Body.Bytes(), uploadedBytes) {
		t.Fatalf("bob display body = %q, want uploaded bytes", bobDisplayRec.Body.String())
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/events?limit=10", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "alice")
	eventsReq.Header.Set("X-Ultra-Org-Id", "org-a")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("events status = %d body=%s", eventsRec.Code, eventsRec.Body.String())
	}
	var events resourceEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode events: %v", err)
	}
	if !resourceEventsContain(events.Events, "resource.shared") {
		t.Fatalf("events = %+v, want resource.shared audit event", events.Events)
	}
}

func TestV2ResourceShareGrantCanBeListedAndRevokedByOwner(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "revoked-nph-study.png")
	if err != nil {
		t.Fatalf("create multipart file: %v", err)
	}
	uploadedBytes := testPNGBytes(t, 3, 2)
	if _, err := part.Write(uploadedBytes); err != nil {
		t.Fatalf("write multipart file: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	uploadReq := httptest.NewRequest(http.MethodPost, "/v2/uploads", &body)
	uploadReq.Header.Set("Content-Type", writer.FormDataContentType())
	uploadReq.Header.Set("X-Ultra-User-Id", "alice")
	uploadReq.Header.Set("X-Ultra-Org-Id", "org-a")
	uploadRec := httptest.NewRecorder()
	router.ServeHTTP(uploadRec, uploadReq)
	if uploadRec.Code != http.StatusOK {
		t.Fatalf("upload status = %d body=%s", uploadRec.Code, uploadRec.Body.String())
	}
	var uploadResponse uploadFilesResponse
	if err := json.Unmarshal(uploadRec.Body.Bytes(), &uploadResponse); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if len(uploadResponse.Uploaded) != 1 {
		t.Fatalf("uploaded = %+v, want one file", uploadResponse.Uploaded)
	}
	fileID := uploadResponse.Uploaded[0].FileID

	shareBody := strings.NewReader(`{"grantee_user_id":"bob","grantee_org_id":"org-b","role":"read","metadata":{"reason":"temporary review"}}`)
	shareReq := httptest.NewRequest(http.MethodPost, "/v2/resources/"+fileID+"/shares", shareBody)
	shareReq.Header.Set("Content-Type", "application/json")
	shareReq.Header.Set("X-Ultra-User-Id", "alice")
	shareReq.Header.Set("X-Ultra-Org-Id", "org-a")
	shareRec := httptest.NewRecorder()
	router.ServeHTTP(shareRec, shareReq)
	if shareRec.Code != http.StatusCreated {
		t.Fatalf("share status = %d body=%s, want 201", shareRec.Code, shareRec.Body.String())
	}
	var shareResponse struct {
		Grant domain.ResourceShareGrantRecord `json:"grant"`
	}
	if err := json.Unmarshal(shareRec.Body.Bytes(), &shareResponse); err != nil {
		t.Fatalf("decode share response: %v", err)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/shares", nil)
	listReq.Header.Set("X-Ultra-User-Id", "alice")
	listReq.Header.Set("X-Ultra-Org-Id", "org-a")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list shares status = %d body=%s, want 200", listRec.Code, listRec.Body.String())
	}
	var listResponse struct {
		ResourceID string                            `json:"resource_id"`
		Count      int                               `json:"count"`
		Grants     []domain.ResourceShareGrantRecord `json:"grants"`
	}
	if err := json.Unmarshal(listRec.Body.Bytes(), &listResponse); err != nil {
		t.Fatalf("decode list shares response: %v", err)
	}
	if listResponse.ResourceID != fileID || listResponse.Count != 1 || len(listResponse.Grants) != 1 || listResponse.Grants[0].Status != "active" {
		t.Fatalf("list shares response = %+v, want one active grant", listResponse)
	}

	bobDisplayReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/display", nil)
	bobDisplayReq.Header.Set("X-Ultra-User-Id", "bob")
	bobDisplayReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobDisplayRec := httptest.NewRecorder()
	router.ServeHTTP(bobDisplayRec, bobDisplayReq)
	if bobDisplayRec.Code != http.StatusOK {
		t.Fatalf("bob display before revoke status = %d body=%s", bobDisplayRec.Code, bobDisplayRec.Body.String())
	}
	if !bytes.Equal(bobDisplayRec.Body.Bytes(), uploadedBytes) {
		t.Fatalf("bob display body before revoke = %q, want uploaded bytes", bobDisplayRec.Body.String())
	}

	revokeReq := httptest.NewRequest(http.MethodDelete, "/v2/resources/"+fileID+"/shares/"+shareResponse.Grant.GrantID, nil)
	revokeReq.Header.Set("X-Ultra-User-Id", "alice")
	revokeReq.Header.Set("X-Ultra-Org-Id", "org-a")
	revokeRec := httptest.NewRecorder()
	router.ServeHTTP(revokeRec, revokeReq)
	if revokeRec.Code != http.StatusOK {
		t.Fatalf("revoke share status = %d body=%s, want 200", revokeRec.Code, revokeRec.Body.String())
	}
	var revokeResponse struct {
		Grant domain.ResourceShareGrantRecord `json:"grant"`
	}
	if err := json.Unmarshal(revokeRec.Body.Bytes(), &revokeResponse); err != nil {
		t.Fatalf("decode revoke response: %v", err)
	}
	if revokeResponse.Grant.GrantID != shareResponse.Grant.GrantID || revokeResponse.Grant.Status != "revoked" || revokeResponse.Grant.RevokedAt.IsZero() {
		t.Fatalf("revoke response = %+v, want revoked grant with timestamp", revokeResponse.Grant)
	}

	bobGetReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID, nil)
	bobGetReq.Header.Set("X-Ultra-User-Id", "bob")
	bobGetReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobGetRec := httptest.NewRecorder()
	router.ServeHTTP(bobGetRec, bobGetReq)
	if bobGetRec.Code != http.StatusNotFound {
		t.Fatalf("bob get after revoke status = %d body=%s, want 404", bobGetRec.Code, bobGetRec.Body.String())
	}

	afterListReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/shares", nil)
	afterListReq.Header.Set("X-Ultra-User-Id", "alice")
	afterListReq.Header.Set("X-Ultra-Org-Id", "org-a")
	afterListRec := httptest.NewRecorder()
	router.ServeHTTP(afterListRec, afterListReq)
	if afterListRec.Code != http.StatusOK {
		t.Fatalf("list shares after revoke status = %d body=%s", afterListRec.Code, afterListRec.Body.String())
	}
	var afterListResponse struct {
		Grants []domain.ResourceShareGrantRecord `json:"grants"`
	}
	if err := json.Unmarshal(afterListRec.Body.Bytes(), &afterListResponse); err != nil {
		t.Fatalf("decode shares after revoke: %v", err)
	}
	if len(afterListResponse.Grants) != 1 || afterListResponse.Grants[0].Status != "revoked" {
		t.Fatalf("shares after revoke = %+v, want revoked grant retained", afterListResponse.Grants)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/events?limit=10", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "alice")
	eventsReq.Header.Set("X-Ultra-Org-Id", "org-a")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("events status = %d body=%s", eventsRec.Code, eventsRec.Body.String())
	}
	var events resourceEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode events: %v", err)
	}
	if !resourceEventsContain(events.Events, "resource.shared") || !resourceEventsContain(events.Events, "resource.share_revoked") {
		t.Fatalf("events = %+v, want share create and revoke audit events", events.Events)
	}
}

func TestV2ResourceEventsListIsScopedAndFilterable(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	base := time.Date(2026, 6, 8, 13, 0, 0, 0, time.UTC)
	inputs := []domain.UpsertResourceInput{
		{
			ResourceID:   "file_audit_active",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			OriginalName: "audit-active.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
			CreatedAt:    base,
		},
		{
			ResourceID:   "file_audit_deleted",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			OriginalName: "audit-deleted.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "deleted",
			CreatedAt:    base.Add(time.Minute),
			DeletedAt:    base.Add(4 * time.Minute),
		},
		{
			ResourceID:   "file_audit_bob_private",
			OwnerUserID:  "bob",
			OwnerOrgID:   "org-b",
			OriginalName: "bob-private.nii.gz",
			SourceType:   "upload",
			ResourceKind: "image",
			Status:       "active",
			CreatedAt:    base.Add(2 * time.Minute),
		},
	}
	for _, input := range inputs {
		if _, err := mem.UpsertResource(ctx, input); err != nil {
			t.Fatalf("UpsertResource(%s): %v", input.ResourceID, err)
		}
	}
	if _, err := mem.CreateResourceShareGrant(ctx, domain.CreateResourceShareGrantInput{
		ResourceID:      "file_audit_active",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		GranteeUserID:   "bob",
		GranteeOrgID:    "org-b",
		Role:            "read",
		Status:          "active",
		CreatedByUserID: "alice",
		CreatedAt:       base.Add(3 * time.Minute),
	}); err != nil {
		t.Fatalf("CreateResourceShareGrant: %v", err)
	}
	for _, event := range []domain.AppendResourceEventInput{
		{ResourceID: "file_audit_active", ActorUserID: "alice", ActorOrgID: "org-a", EventType: "resource.tagged", TS: base.Add(4 * time.Minute), Metadata: domain.JSONMap{"tag": "NPH"}},
		{ResourceID: "file_audit_deleted", ActorUserID: "alice", ActorOrgID: "org-a", EventType: "resource.deleted", TS: base.Add(5 * time.Minute)},
		{ResourceID: "file_audit_bob_private", ActorUserID: "bob", ActorOrgID: "org-b", EventType: "resource.tagged", TS: base.Add(6 * time.Minute), Metadata: domain.JSONMap{"tag": "private"}},
	} {
		if _, err := mem.CreateResourceEvent(ctx, event); err != nil {
			t.Fatalf("CreateResourceEvent(%s): %v", event.ResourceID, err)
		}
	}

	aliceReq := httptest.NewRequest(http.MethodGet, "/v2/resource-events?limit=10", nil)
	aliceReq.Header.Set("X-Ultra-User-Id", "alice")
	aliceReq.Header.Set("X-Ultra-Org-Id", "org-a")
	aliceRec := httptest.NewRecorder()
	router.ServeHTTP(aliceRec, aliceReq)
	if aliceRec.Code != http.StatusOK {
		t.Fatalf("alice events status = %d body=%s", aliceRec.Code, aliceRec.Body.String())
	}
	var aliceEvents struct {
		Count      int                          `json:"count"`
		TotalCount int                          `json:"total_count"`
		Limit      int                          `json:"limit"`
		Offset     int                          `json:"offset"`
		Events     []domain.ResourceEventRecord `json:"events"`
	}
	if err := json.Unmarshal(aliceRec.Body.Bytes(), &aliceEvents); err != nil {
		t.Fatalf("decode alice events: %v", err)
	}
	if aliceEvents.Count != 2 || aliceEvents.TotalCount != 2 || len(aliceEvents.Events) != 2 {
		t.Fatalf("alice events = %+v, want two owned events", aliceEvents)
	}
	if aliceEvents.Events[0].ResourceID != "file_audit_deleted" || aliceEvents.Events[1].ResourceID != "file_audit_active" {
		t.Fatalf("alice event resource order = %+v, want deleted then active", aliceEvents.Events)
	}

	deletedReq := httptest.NewRequest(http.MethodGet, "/v2/resource-events?event_type=resource.deleted&limit=10", nil)
	deletedReq.Header.Set("X-Ultra-User-Id", "alice")
	deletedReq.Header.Set("X-Ultra-Org-Id", "org-a")
	deletedRec := httptest.NewRecorder()
	router.ServeHTTP(deletedRec, deletedReq)
	if deletedRec.Code != http.StatusOK {
		t.Fatalf("deleted events status = %d body=%s", deletedRec.Code, deletedRec.Body.String())
	}
	var deletedEvents struct {
		Events []domain.ResourceEventRecord `json:"events"`
	}
	if err := json.Unmarshal(deletedRec.Body.Bytes(), &deletedEvents); err != nil {
		t.Fatalf("decode deleted events: %v", err)
	}
	if len(deletedEvents.Events) != 1 || deletedEvents.Events[0].ResourceID != "file_audit_deleted" {
		t.Fatalf("deleted events = %+v, want only deleted resource event", deletedEvents.Events)
	}

	bobReq := httptest.NewRequest(http.MethodGet, "/v2/resource-events?resource_id=file_audit_active&limit=10", nil)
	bobReq.Header.Set("X-Ultra-User-Id", "bob")
	bobReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobRec := httptest.NewRecorder()
	router.ServeHTTP(bobRec, bobReq)
	if bobRec.Code != http.StatusOK {
		t.Fatalf("bob events status = %d body=%s", bobRec.Code, bobRec.Body.String())
	}
	var bobEvents struct {
		Events []domain.ResourceEventRecord `json:"events"`
	}
	if err := json.Unmarshal(bobRec.Body.Bytes(), &bobEvents); err != nil {
		t.Fatalf("decode bob events: %v", err)
	}
	if len(bobEvents.Events) != 1 || bobEvents.Events[0].ResourceID != "file_audit_active" {
		t.Fatalf("bob events = %+v, want only shared resource event", bobEvents.Events)
	}

	foreignReq := httptest.NewRequest(http.MethodGet, "/v2/resource-events?limit=10", nil)
	foreignReq.Header.Set("X-Ultra-User-Id", "charlie")
	foreignReq.Header.Set("X-Ultra-Org-Id", "org-c")
	foreignRec := httptest.NewRecorder()
	router.ServeHTTP(foreignRec, foreignReq)
	if foreignRec.Code != http.StatusOK {
		t.Fatalf("foreign events status = %d body=%s", foreignRec.Code, foreignRec.Body.String())
	}
	var foreignEvents struct {
		Count  int                          `json:"count"`
		Events []domain.ResourceEventRecord `json:"events"`
	}
	if err := json.Unmarshal(foreignRec.Body.Bytes(), &foreignEvents); err != nil {
		t.Fatalf("decode foreign events: %v", err)
	}
	if foreignEvents.Count != 0 || len(foreignEvents.Events) != 0 {
		t.Fatalf("foreign events = %+v, want no leaked events", foreignEvents)
	}
}

func TestV2ResourceMetadataPatchMergesAndAuditsOwnerEdit(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_metadata_patch",
		OwnerUserID:  "alice",
		OwnerOrgID:   "org-a",
		OriginalName: "metadata-patch.nii.gz",
		ContentType:  "application/gzip",
		SourceType:   "upload",
		ResourceKind: "image",
		Status:       "active",
		CreatedAt:    time.Date(2026, 6, 8, 14, 0, 0, 0, time.UTC),
		Metadata: domain.JSONMap{
			"source_label": "raw",
			"review": domain.JSONMap{
				"reader": "lab-a",
			},
		},
	}); err != nil {
		t.Fatalf("UpsertResource: %v", err)
	}
	if _, err := mem.CreateResourceShareGrant(ctx, domain.CreateResourceShareGrantInput{
		ResourceID:      "file_metadata_patch",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		GranteeUserID:   "bob",
		GranteeOrgID:    "org-b",
		Role:            "read",
		Status:          "active",
		CreatedByUserID: "alice",
	}); err != nil {
		t.Fatalf("CreateResourceShareGrant: %v", err)
	}

	patchReq := httptest.NewRequest(http.MethodPatch, "/v2/resources/file_metadata_patch", strings.NewReader(`{
		"metadata": {
			"cohort": "NPH",
			"review": {"status": "checked"}
		}
	}`))
	patchReq.Header.Set("Content-Type", "application/json")
	patchReq.Header.Set("X-Ultra-User-Id", "alice")
	patchReq.Header.Set("X-Ultra-Org-Id", "org-a")
	patchRec := httptest.NewRecorder()
	router.ServeHTTP(patchRec, patchReq)
	if patchRec.Code != http.StatusOK {
		t.Fatalf("metadata patch status = %d body=%s", patchRec.Code, patchRec.Body.String())
	}
	var patched resourceResponse
	if err := json.Unmarshal(patchRec.Body.Bytes(), &patched); err != nil {
		t.Fatalf("decode patched resource: %v", err)
	}
	if patched.Resource.Metadata["cohort"] != "NPH" || patched.Resource.Metadata["source_label"] != "raw" {
		t.Fatalf("patched metadata = %+v, want merged source_label and cohort", patched.Resource.Metadata)
	}
	review, ok := patched.Resource.Metadata["review"].(map[string]any)
	if !ok {
		if typed, typedOK := patched.Resource.Metadata["review"].(domain.JSONMap); typedOK {
			review = map[string]any(typed)
			ok = true
		}
	}
	if !ok || review["reader"] != "lab-a" || review["status"] != "checked" {
		t.Fatalf("patched review metadata = %#v, want merged reader/status", patched.Resource.Metadata["review"])
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/file_metadata_patch/events?limit=10", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "alice")
	eventsReq.Header.Set("X-Ultra-Org-Id", "org-a")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("events status = %d body=%s", eventsRec.Code, eventsRec.Body.String())
	}
	var events resourceEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode events: %v", err)
	}
	metadataEvent, ok := resourceEventByType(events.Events, "resource.metadata_updated")
	if !ok {
		t.Fatalf("events = %+v, want resource.metadata_updated", events.Events)
	}
	if metadataEvent.ActorUserID != "alice" || metadataEvent.ActorOrgID != "org-a" {
		t.Fatalf("metadata event actor = %+v, want alice/org-a", metadataEvent)
	}
	keys, _ := metadataEvent.Metadata["metadata_keys"].([]any)
	if !stringSliceContainsAny(keys, "cohort") || !stringSliceContainsAny(keys, "review") {
		t.Fatalf("metadata event keys = %#v, want cohort and review; metadata=%+v", keys, metadataEvent.Metadata)
	}

	eventLogReq := httptest.NewRequest(http.MethodGet, "/v2/resource-events?event_type=resource.metadata_updated&limit=10", nil)
	eventLogReq.Header.Set("X-Ultra-User-Id", "alice")
	eventLogReq.Header.Set("X-Ultra-Org-Id", "org-a")
	eventLogRec := httptest.NewRecorder()
	router.ServeHTTP(eventLogRec, eventLogReq)
	if eventLogRec.Code != http.StatusOK {
		t.Fatalf("event log status = %d body=%s", eventLogRec.Code, eventLogRec.Body.String())
	}
	var eventLog resourceEventListResponse
	if err := json.Unmarshal(eventLogRec.Body.Bytes(), &eventLog); err != nil {
		t.Fatalf("decode event log: %v", err)
	}
	if eventLog.Count != 1 || eventLog.TotalCount != 1 || eventLog.Events[0].ResourceID != "file_metadata_patch" {
		t.Fatalf("event log = %+v, want one metadata update event", eventLog)
	}

	bobPatchReq := httptest.NewRequest(http.MethodPatch, "/v2/resources/file_metadata_patch", strings.NewReader(`{"metadata":{"cohort":"tampered"}}`))
	bobPatchReq.Header.Set("Content-Type", "application/json")
	bobPatchReq.Header.Set("X-Ultra-User-Id", "bob")
	bobPatchReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobPatchRec := httptest.NewRecorder()
	router.ServeHTTP(bobPatchRec, bobPatchReq)
	if bobPatchRec.Code != http.StatusNotFound {
		t.Fatalf("bob metadata patch status = %d body=%s, want 404", bobPatchRec.Code, bobPatchRec.Body.String())
	}
}

func TestV2ResourceBulkShareGrantCreatesReadGrantsForOwnedResources(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_bulk_share_a",
			OriginalName: "bulk-share-a.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    128,
			SHA256:       "sha-bulk-share-a",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "bulk-study",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_bulk_share_b",
			OriginalName: "bulk-share-b.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    256,
			SHA256:       "sha-bulk-share-b",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "bulk-study",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	shareReq := httptest.NewRequest(http.MethodPost, "/v2/resources/shares/bulk", strings.NewReader(`{
		"resource_ids":[" file_bulk_share_a ","file_bulk_share_b","file_bulk_share_a"],
		"grantee_user_id":"bob",
		"grantee_org_id":"org-b",
		"role":"read",
		"metadata":{"reason":"bulk review"}
	}`))
	shareReq.Header.Set("Content-Type", "application/json")
	shareReq.Header.Set("X-Ultra-User-Id", "alice")
	shareReq.Header.Set("X-Ultra-Org-Id", "org-a")
	shareRec := httptest.NewRecorder()
	router.ServeHTTP(shareRec, shareReq)
	if shareRec.Code != http.StatusCreated {
		t.Fatalf("bulk share status = %d body=%s, want 201", shareRec.Code, shareRec.Body.String())
	}
	var shareResponse struct {
		Count  int                               `json:"count"`
		Grants []domain.ResourceShareGrantRecord `json:"grants"`
	}
	if err := json.Unmarshal(shareRec.Body.Bytes(), &shareResponse); err != nil {
		t.Fatalf("decode bulk share response: %v", err)
	}
	if shareResponse.Count != 2 || len(shareResponse.Grants) != 2 {
		t.Fatalf("bulk share response = %+v, want two grants", shareResponse)
	}
	for index, wantResourceID := range []string{"file_bulk_share_a", "file_bulk_share_b"} {
		grant := shareResponse.Grants[index]
		if grant.ResourceID != wantResourceID || grant.GranteeUserID != "bob" || grant.GranteeOrgID != "org-b" || grant.Role != "read" || grant.Status != "active" {
			t.Fatalf("grant[%d] = %+v, want active read grant for Bob on %s", index, grant, wantResourceID)
		}
	}

	bobListReq := httptest.NewRequest(http.MethodGet, "/v2/resources?q=bulk-share&limit=20", nil)
	bobListReq.Header.Set("X-Ultra-User-Id", "bob")
	bobListReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobListRec := httptest.NewRecorder()
	router.ServeHTTP(bobListRec, bobListReq)
	if bobListRec.Code != http.StatusOK {
		t.Fatalf("bob list status = %d body=%s", bobListRec.Code, bobListRec.Body.String())
	}
	var bobList resourcesResponse
	if err := json.Unmarshal(bobListRec.Body.Bytes(), &bobList); err != nil {
		t.Fatalf("decode bob resources: %v", err)
	}
	if bobList.Count != 2 || len(bobList.Resources) != 2 {
		t.Fatalf("bob resources after bulk share = %+v, want two shared resources", bobList)
	}

	for _, resourceID := range []string{"file_bulk_share_a", "file_bulk_share_b"} {
		eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+resourceID+"/events?limit=10", nil)
		eventsReq.Header.Set("X-Ultra-User-Id", "alice")
		eventsReq.Header.Set("X-Ultra-Org-Id", "org-a")
		eventsRec := httptest.NewRecorder()
		router.ServeHTTP(eventsRec, eventsReq)
		if eventsRec.Code != http.StatusOK {
			t.Fatalf("events status for %s = %d body=%s", resourceID, eventsRec.Code, eventsRec.Body.String())
		}
		var events resourceEventsResponse
		if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
			t.Fatalf("decode events for %s: %v", resourceID, err)
		}
		if !resourceEventsContain(events.Events, "resource.shared") {
			t.Fatalf("events for %s = %+v, want resource.shared audit event", resourceID, events.Events)
		}
	}
}

func TestV2ResourceBulkTagAddsTagsFiltersAndAuditsResources(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_tag_http_a",
			OwnerUserID:  "tag-user",
			OwnerOrgID:   "tag-org",
			OriginalName: "tag-http-a.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
			Tags:         []string{"raw"},
		},
		{
			ResourceID:   "file_tag_http_b",
			OwnerUserID:  "tag-user",
			OwnerOrgID:   "tag-org",
			OriginalName: "tag-http-b.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    256,
			SourceType:   "upload",
			ResourceKind: "file",
			Status:       "active",
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	tagReq := httptest.NewRequest(http.MethodPost, "/v2/resources/tags/bulk", strings.NewReader(`{
		"resource_ids":["file_tag_http_a","file_tag_http_b"],
		"tags":["NPH","Under 70","nph"],
		"metadata":{"source":"resources_bulk_tag_panel"}
	}`))
	tagReq.Header.Set("Content-Type", "application/json")
	tagReq.Header.Set("X-Ultra-User-Id", "tag-user")
	tagReq.Header.Set("X-Ultra-Org-Id", "tag-org")
	tagRec := httptest.NewRecorder()
	router.ServeHTTP(tagRec, tagReq)
	if tagRec.Code != http.StatusOK {
		t.Fatalf("bulk tag status = %d body=%s, want 200", tagRec.Code, tagRec.Body.String())
	}
	var tagResponse struct {
		Count     int                          `json:"count"`
		Resources []resourceRecord             `json:"resources"`
		Events    []domain.ResourceEventRecord `json:"events"`
	}
	if err := json.Unmarshal(tagRec.Body.Bytes(), &tagResponse); err != nil {
		t.Fatalf("decode bulk tag response: %v", err)
	}
	if tagResponse.Count != 2 || len(tagResponse.Resources) != 2 || len(tagResponse.Events) != 2 {
		t.Fatalf("bulk tag response = %+v, want two resources and two events", tagResponse)
	}
	if !reflect.DeepEqual(tagResponse.Resources[0].Tags, []string{"raw", "NPH", "Under 70"}) {
		t.Fatalf("first tagged resource tags = %#v, want raw plus NPH tags", tagResponse.Resources[0].Tags)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources?tag=nph&tag=under+70&limit=20", nil)
	listReq.Header.Set("X-Ultra-User-Id", "tag-user")
	listReq.Header.Set("X-Ultra-Org-Id", "tag-org")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("tag-filtered resources status = %d body=%s, want 200", listRec.Code, listRec.Body.String())
	}
	var listed resourcesResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &listed); err != nil {
		t.Fatalf("decode tag-filtered resources: %v", err)
	}
	if listed.Count != 2 || len(listed.Resources) != 2 {
		t.Fatalf("tag-filtered resources = %+v, want both tagged resources", listed)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/file_tag_http_a/events", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "tag-user")
	eventsReq.Header.Set("X-Ultra-Org-Id", "tag-org")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("resource events status = %d body=%s, want 200", eventsRec.Code, eventsRec.Body.String())
	}
	var events resourceEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode resource events: %v", err)
	}
	if !resourceEventsContain(events.Events, "resource.tagged") {
		t.Fatalf("resource events = %+v, want resource.tagged audit event", events.Events)
	}
}

func TestV2ResourceCollectionShareCreatesReadGrantsForFolderMembers(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	ctx := context.Background()
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_folder_share_a",
			OriginalName: "folder-share-a.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    128,
			SHA256:       "sha-folder-share-a",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "folder-study",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_folder_share_b",
			OriginalName: "folder-share-b.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    256,
			SHA256:       "sha-folder-share-b",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "folder-study",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
		{
			ResourceID:   "file_folder_share_outside",
			OriginalName: "folder-share-outside.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    512,
			SHA256:       "sha-folder-share-outside",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "folder-study",
			Status:       "active",
			CreatedAt:    now.Add(2 * time.Second),
			UpdatedAt:    now.Add(2 * time.Second),
		},
	} {
		if _, err := mem.UpsertResource(ctx, resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}
	collection, err := mem.CreateResourceCollection(ctx, domain.CreateResourceCollectionInput{
		CollectionID:   "collection_folder_share",
		OwnerUserID:    "alice",
		OwnerOrgID:     "org-a",
		Name:           "NPH review folder",
		CollectionType: "folder",
		Status:         "active",
		CreatedAt:      now,
		UpdatedAt:      now,
	})
	if err != nil {
		t.Fatalf("CreateResourceCollection: %v", err)
	}
	if _, err := mem.AddResourcesToCollection(ctx, domain.AddResourcesToCollectionInput{
		CollectionID:  collection.CollectionID,
		OwnerUserID:   "alice",
		OwnerOrgID:    "org-a",
		ResourceIDs:   []string{"file_folder_share_a", "file_folder_share_b"},
		AddedByUserID: "alice",
		AddedAt:       now,
	}); err != nil {
		t.Fatalf("AddResourcesToCollection: %v", err)
	}

	shareReq := httptest.NewRequest(http.MethodPost, "/v2/resource-collections/collection_folder_share/shares", strings.NewReader(`{
		"grantee_user_id":"bob",
		"grantee_org_id":"org-b",
		"role":"read",
		"metadata":{"reason":"folder review"}
	}`))
	shareReq.Header.Set("Content-Type", "application/json")
	shareReq.Header.Set("X-Ultra-User-Id", "alice")
	shareReq.Header.Set("X-Ultra-Org-Id", "org-a")
	shareRec := httptest.NewRecorder()
	router.ServeHTTP(shareRec, shareReq)
	if shareRec.Code != http.StatusCreated {
		t.Fatalf("folder share status = %d body=%s, want 201", shareRec.Code, shareRec.Body.String())
	}
	var shareResponse struct {
		Count      int                               `json:"count"`
		Collection domain.ResourceCollectionRecord   `json:"collection"`
		Grants     []domain.ResourceShareGrantRecord `json:"grants"`
	}
	if err := json.Unmarshal(shareRec.Body.Bytes(), &shareResponse); err != nil {
		t.Fatalf("decode folder share response: %v", err)
	}
	if shareResponse.Collection.CollectionID != "collection_folder_share" {
		t.Fatalf("collection = %+v, want folder collection", shareResponse.Collection)
	}
	if shareResponse.Count != 2 || len(shareResponse.Grants) != 2 {
		t.Fatalf("folder share response = %+v, want two grants", shareResponse)
	}
	for index, wantResourceID := range []string{"file_folder_share_a", "file_folder_share_b"} {
		grant := shareResponse.Grants[index]
		if grant.ResourceID != wantResourceID || grant.GranteeUserID != "bob" || grant.GranteeOrgID != "org-b" || grant.Role != "read" || grant.Status != "active" {
			t.Fatalf("grant[%d] = %+v, want active read grant for Bob on %s", index, grant, wantResourceID)
		}
	}

	bobListReq := httptest.NewRequest(http.MethodGet, "/v2/resources?q=folder-share&limit=20", nil)
	bobListReq.Header.Set("X-Ultra-User-Id", "bob")
	bobListReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobListRec := httptest.NewRecorder()
	router.ServeHTTP(bobListRec, bobListReq)
	if bobListRec.Code != http.StatusOK {
		t.Fatalf("bob list status = %d body=%s", bobListRec.Code, bobListRec.Body.String())
	}
	var bobList resourcesResponse
	if err := json.Unmarshal(bobListRec.Body.Bytes(), &bobList); err != nil {
		t.Fatalf("decode bob resources: %v", err)
	}
	if bobList.Count != 2 || len(bobList.Resources) != 2 {
		t.Fatalf("bob resources after folder share = %+v, want two shared folder resources only", bobList)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/file_folder_share_a/events?limit=10", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "alice")
	eventsReq.Header.Set("X-Ultra-Org-Id", "org-a")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("events status = %d body=%s", eventsRec.Code, eventsRec.Body.String())
	}
	var events resourceEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode folder share events: %v", err)
	}
	sharedEvent, ok := resourceEventByType(events.Events, "resource.shared")
	if !ok {
		t.Fatalf("events = %+v, want resource.shared", events.Events)
	}
	if sharedEvent.Metadata["collection_id"] != "collection_folder_share" || sharedEvent.Metadata["source"] != "resource_collection_share" {
		t.Fatalf("resource.shared metadata = %+v, want collection share context", sharedEvent.Metadata)
	}

	addFutureReq := httptest.NewRequest(http.MethodPost, "/v2/resource-collections/collection_folder_share/resources", strings.NewReader(`{
		"resource_ids":["file_folder_share_outside"],
		"metadata":{"source":"late_folder_add"}
	}`))
	addFutureReq.Header.Set("Content-Type", "application/json")
	addFutureReq.Header.Set("X-Ultra-User-Id", "alice")
	addFutureReq.Header.Set("X-Ultra-Org-Id", "org-a")
	addFutureRec := httptest.NewRecorder()
	router.ServeHTTP(addFutureRec, addFutureReq)
	if addFutureRec.Code != http.StatusOK {
		t.Fatalf("future folder add status = %d body=%s, want 200", addFutureRec.Code, addFutureRec.Body.String())
	}

	bobFutureReq := httptest.NewRequest(http.MethodGet, "/v2/resources?q=folder-share&limit=20", nil)
	bobFutureReq.Header.Set("X-Ultra-User-Id", "bob")
	bobFutureReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobFutureRec := httptest.NewRecorder()
	router.ServeHTTP(bobFutureRec, bobFutureReq)
	if bobFutureRec.Code != http.StatusOK {
		t.Fatalf("bob future list status = %d body=%s", bobFutureRec.Code, bobFutureRec.Body.String())
	}
	var bobFutureList resourcesResponse
	if err := json.Unmarshal(bobFutureRec.Body.Bytes(), &bobFutureList); err != nil {
		t.Fatalf("decode bob future resources: %v", err)
	}
	if bobFutureList.Count != 3 || len(bobFutureList.Resources) != 3 {
		t.Fatalf("bob resources after future folder add = %+v, want three inherited shared resources", bobFutureList)
	}

	futureEventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/file_folder_share_outside/events?limit=10", nil)
	futureEventsReq.Header.Set("X-Ultra-User-Id", "alice")
	futureEventsReq.Header.Set("X-Ultra-Org-Id", "org-a")
	futureEventsRec := httptest.NewRecorder()
	router.ServeHTTP(futureEventsRec, futureEventsReq)
	if futureEventsRec.Code != http.StatusOK {
		t.Fatalf("future events status = %d body=%s", futureEventsRec.Code, futureEventsRec.Body.String())
	}
	var futureEvents resourceEventsResponse
	if err := json.Unmarshal(futureEventsRec.Body.Bytes(), &futureEvents); err != nil {
		t.Fatalf("decode future folder share events: %v", err)
	}
	futureSharedEvent, ok := resourceEventByType(futureEvents.Events, "resource.shared")
	if !ok {
		t.Fatalf("future events = %+v, want inherited resource.shared event", futureEvents.Events)
	}
	if futureSharedEvent.Metadata["collection_id"] != "collection_folder_share" || futureSharedEvent.Metadata["source"] != "resource_collection_share_inherited" {
		t.Fatalf("future resource.shared metadata = %+v, want inherited collection share context", futureSharedEvent.Metadata)
	}
}

func TestV2ResourcesListExposesAndFiltersShareSummary(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_private_alice",
			OriginalName: "private-alice.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    64,
			SHA256:       "sha-private-alice",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_shared_by_alice",
			OriginalName: "shared-by-alice.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    96,
			SHA256:       "sha-shared-by-alice",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
		{
			ResourceID:   "file_shared_with_bob",
			OriginalName: "shared-with-bob.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    128,
			SHA256:       "sha-shared-with-bob",
			OwnerUserID:  "carol",
			OwnerOrgID:   "org-c",
			Status:       "active",
			CreatedAt:    now.Add(2 * time.Second),
			UpdatedAt:    now.Add(2 * time.Second),
		},
		{
			ResourceID:   "file_public_alice",
			OriginalName: "public-alice.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    160,
			SHA256:       "sha-public-alice",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(3 * time.Second),
			UpdatedAt:    now.Add(3 * time.Second),
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}
	if _, err := mem.CreateResourceShareGrant(context.Background(), domain.CreateResourceShareGrantInput{
		ResourceID:      "file_shared_by_alice",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		GranteeUserID:   "bob",
		GranteeOrgID:    "org-b",
		Role:            "read",
		Status:          "active",
		CreatedByUserID: "alice",
	}); err != nil {
		t.Fatalf("CreateResourceShareGrant alice->bob: %v", err)
	}
	if _, err := mem.CreateResourceShareGrant(context.Background(), domain.CreateResourceShareGrantInput{
		ResourceID:      "file_shared_with_bob",
		OwnerUserID:     "carol",
		OwnerOrgID:      "org-c",
		GranteeUserID:   "bob",
		GranteeOrgID:    "org-b",
		Role:            "read",
		Status:          "active",
		CreatedByUserID: "carol",
	}); err != nil {
		t.Fatalf("CreateResourceShareGrant carol->bob: %v", err)
	}
	publicReq := httptest.NewRequest(http.MethodPost, "/v2/resources/file_public_alice/shares", strings.NewReader(`{
		"public":true,
		"role":"read",
		"metadata":{"reason":"publication supplement"}
	}`))
	publicReq.Header.Set("Content-Type", "application/json")
	publicReq.Header.Set("X-Ultra-User-Id", "alice")
	publicReq.Header.Set("X-Ultra-Org-Id", "org-a")
	publicRec := httptest.NewRecorder()
	router.ServeHTTP(publicRec, publicReq)
	if publicRec.Code != http.StatusCreated {
		t.Fatalf("public share status = %d body=%s, want 201", publicRec.Code, publicRec.Body.String())
	}

	type listedResource struct {
		FileID       string         `json:"file_id"`
		ShareSummary map[string]any `json:"share_summary"`
	}
	decodeResources := func(rec *httptest.ResponseRecorder) []listedResource {
		t.Helper()
		var response struct {
			Resources []listedResource `json:"resources"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
			t.Fatalf("decode resources response: %v", err)
		}
		return response.Resources
	}

	aliceReq := httptest.NewRequest(http.MethodGet, "/v2/resources?sharing=shared_by_me&limit=20", nil)
	aliceReq.Header.Set("X-Ultra-User-Id", "alice")
	aliceReq.Header.Set("X-Ultra-Org-Id", "org-a")
	aliceRec := httptest.NewRecorder()
	router.ServeHTTP(aliceRec, aliceReq)
	if aliceRec.Code != http.StatusOK {
		t.Fatalf("alice shared_by_me status = %d body=%s", aliceRec.Code, aliceRec.Body.String())
	}
	aliceResources := decodeResources(aliceRec)
	if len(aliceResources) != 1 || aliceResources[0].FileID != "file_shared_by_alice" {
		t.Fatalf("alice shared_by_me resources = %+v, want shared owned resource only", aliceResources)
	}
	if aliceResources[0].ShareSummary["share_status"] != "shared_by_me" || int(aliceResources[0].ShareSummary["active_grant_count"].(float64)) != 1 {
		t.Fatalf("alice share summary = %+v, want shared_by_me with one active grant", aliceResources[0].ShareSummary)
	}

	privateReq := httptest.NewRequest(http.MethodGet, "/v2/resources?sharing=private&limit=20", nil)
	privateReq.Header.Set("X-Ultra-User-Id", "alice")
	privateReq.Header.Set("X-Ultra-Org-Id", "org-a")
	privateRec := httptest.NewRecorder()
	router.ServeHTTP(privateRec, privateReq)
	if privateRec.Code != http.StatusOK {
		t.Fatalf("alice private status = %d body=%s", privateRec.Code, privateRec.Body.String())
	}
	privateResources := decodeResources(privateRec)
	if len(privateResources) != 1 || privateResources[0].FileID != "file_private_alice" || privateResources[0].ShareSummary["share_status"] != "private" {
		t.Fatalf("alice private resources = %+v, want private resource with private summary", privateResources)
	}

	alicePublicReq := httptest.NewRequest(http.MethodGet, "/v2/resources?sharing=public&limit=20", nil)
	alicePublicReq.Header.Set("X-Ultra-User-Id", "alice")
	alicePublicReq.Header.Set("X-Ultra-Org-Id", "org-a")
	alicePublicRec := httptest.NewRecorder()
	router.ServeHTTP(alicePublicRec, alicePublicReq)
	if alicePublicRec.Code != http.StatusOK {
		t.Fatalf("alice public status = %d body=%s", alicePublicRec.Code, alicePublicRec.Body.String())
	}
	alicePublicResources := decodeResources(alicePublicRec)
	if len(alicePublicResources) != 1 || alicePublicResources[0].FileID != "file_public_alice" || alicePublicResources[0].ShareSummary["share_status"] != "public" {
		t.Fatalf("alice public resources = %+v, want public resource with public summary", alicePublicResources)
	}

	bobReq := httptest.NewRequest(http.MethodGet, "/v2/resources?sharing=shared_with_me&limit=20", nil)
	bobReq.Header.Set("X-Ultra-User-Id", "bob")
	bobReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobRec := httptest.NewRecorder()
	router.ServeHTTP(bobRec, bobReq)
	if bobRec.Code != http.StatusOK {
		t.Fatalf("bob shared_with_me status = %d body=%s", bobRec.Code, bobRec.Body.String())
	}
	bobResources := decodeResources(bobRec)
	if len(bobResources) != 2 {
		t.Fatalf("bob shared_with_me resources = %+v, want two shared resources", bobResources)
	}
	for _, resource := range bobResources {
		if resource.ShareSummary["share_status"] != "shared_with_me" {
			t.Fatalf("bob resource %s share summary = %+v, want shared_with_me", resource.FileID, resource.ShareSummary)
		}
	}

	bobPublicReq := httptest.NewRequest(http.MethodGet, "/v2/resources?sharing=public&limit=20", nil)
	bobPublicReq.Header.Set("X-Ultra-User-Id", "bob")
	bobPublicReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobPublicRec := httptest.NewRecorder()
	router.ServeHTTP(bobPublicRec, bobPublicReq)
	if bobPublicRec.Code != http.StatusOK {
		t.Fatalf("bob public status = %d body=%s", bobPublicRec.Code, bobPublicRec.Body.String())
	}
	bobPublicResources := decodeResources(bobPublicRec)
	if len(bobPublicResources) != 1 || bobPublicResources[0].FileID != "file_public_alice" || bobPublicResources[0].ShareSummary["share_status"] != "public" {
		t.Fatalf("bob public resources = %+v, want public resource with public summary", bobPublicResources)
	}
	bobPrivateReq := httptest.NewRequest(http.MethodGet, "/v2/resources?sharing=private&limit=20", nil)
	bobPrivateReq.Header.Set("X-Ultra-User-Id", "bob")
	bobPrivateReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobPrivateRec := httptest.NewRecorder()
	router.ServeHTTP(bobPrivateRec, bobPrivateReq)
	if bobPrivateRec.Code != http.StatusOK {
		t.Fatalf("bob private status = %d body=%s", bobPrivateRec.Code, bobPrivateRec.Body.String())
	}
	if resources := decodeResources(bobPrivateRec); len(resources) != 0 {
		t.Fatalf("bob private resources = %+v, want no private resources leaked", resources)
	}
}

func resourceEventsContain(events []domain.ResourceEventRecord, eventType string) bool {
	for _, event := range events {
		if event.EventType == eventType {
			return true
		}
	}
	return false
}

func resourceEventByType(events []domain.ResourceEventRecord, eventType string) (domain.ResourceEventRecord, bool) {
	for _, event := range events {
		if event.EventType == eventType {
			return event, true
		}
	}
	return domain.ResourceEventRecord{}, false
}

func stringSliceContainsAny(values []any, want string) bool {
	for _, value := range values {
		if got, ok := value.(string); ok && got == want {
			return true
		}
	}
	return false
}

type uploadSessionEventPayload struct {
	EventType   string         `json:"event_type"`
	ActorUserID string         `json:"actor_user_id"`
	ActorOrgID  string         `json:"actor_org_id"`
	Metadata    map[string]any `json:"metadata"`
}

func uploadSessionEventsContain(events []uploadSessionEventPayload, eventType string) bool {
	for _, event := range events {
		if event.EventType == eventType {
			return true
		}
	}
	return false
}

func TestV2ResourcesListComesFromCatalogWhenNFSFileIsMissing(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "catalog-only.png")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	if _, err := part.Write(testPNGBytes(t, 4, 3)); err != nil {
		t.Fatalf("write multipart file: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	uploadReq := httptest.NewRequest(http.MethodPost, "/v2/uploads", &body)
	uploadReq.Header.Set("Content-Type", writer.FormDataContentType())
	uploadReq.Header.Set("X-Ultra-User-Id", "catalog-user")
	uploadReq.Header.Set("X-Ultra-Org-Id", "catalog-org")
	uploadRec := httptest.NewRecorder()
	router.ServeHTTP(uploadRec, uploadReq)
	if uploadRec.Code != http.StatusOK {
		t.Fatalf("upload status = %d body=%s", uploadRec.Code, uploadRec.Body.String())
	}
	var uploadResponse uploadFilesResponse
	if err := json.Unmarshal(uploadRec.Body.Bytes(), &uploadResponse); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if len(uploadResponse.Uploaded) != 1 {
		t.Fatalf("uploaded = %+v, want one file", uploadResponse.Uploaded)
	}
	uploaded := uploadResponse.Uploaded[0]
	matches, err := filepath.Glob(filepath.Join(uploadRoot, uploaded.FileID+"__*"))
	if err != nil || len(matches) != 1 {
		t.Fatalf("uploaded files = %v err=%v, want one file", matches, err)
	}
	if err := os.Remove(matches[0]); err != nil {
		t.Fatalf("remove uploaded blob to prove list is catalog-backed: %v", err)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20&kind=image&source=upload", nil)
	listReq.Header.Set("X-Ultra-User-Id", "catalog-user")
	listReq.Header.Set("X-Ultra-Org-Id", "catalog-org")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list resources status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var listResponse resourcesResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &listResponse); err != nil {
		t.Fatalf("decode resources response: %v", err)
	}
	if listResponse.Count != 1 || len(listResponse.Resources) != 1 {
		t.Fatalf("resources = %+v, want catalog row even when blob is missing", listResponse)
	}
	if got := listResponse.Resources[0].FileID; got != uploaded.FileID {
		t.Fatalf("resource file_id = %q, want %q", got, uploaded.FileID)
	}
	if listResponse.Resources[0].StagedLocally {
		t.Fatalf("staged_locally = true, want false after blob was removed")
	}
}

func TestV2ResourcesListMigratesExistingUploadsOnceThenUsesCatalog(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	migratedFileID := writeTestUploadFile(t, uploadRoot, "migrated-image.png", testPNGBytes(t, 3, 3))

	firstReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20", nil)
	firstReq.Header.Set("X-Ultra-User-Id", "test-user")
	firstReq.Header.Set("X-Ultra-Org-Id", "test-org")
	firstRec := httptest.NewRecorder()
	router.ServeHTTP(firstRec, firstReq)
	if firstRec.Code != http.StatusOK {
		t.Fatalf("first list status = %d body=%s", firstRec.Code, firstRec.Body.String())
	}
	var first resourcesResponse
	if err := json.Unmarshal(firstRec.Body.Bytes(), &first); err != nil {
		t.Fatalf("decode first resources: %v", err)
	}
	if first.Count != 1 || len(first.Resources) != 1 || first.Resources[0].FileID != migratedFileID {
		t.Fatalf("first resources = %+v, want migrated upload only", first)
	}

	orphanPath := filepath.Join(uploadRoot, "file_uncataloged__orphan.png")
	if err := os.WriteFile(orphanPath, testPNGBytes(t, 4, 4), 0o644); err != nil {
		t.Fatalf("write uncataloged upload after migration: %v", err)
	}
	if err := writeUploadMetadata(uploadRoot, "file_uncataloged", requestPrincipal{UserID: "test-user", OrgID: "test-org", Role: "researcher"}); err != nil {
		t.Fatalf("write uncataloged metadata: %v", err)
	}

	secondReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20", nil)
	secondReq.Header.Set("X-Ultra-User-Id", "test-user")
	secondReq.Header.Set("X-Ultra-Org-Id", "test-org")
	secondRec := httptest.NewRecorder()
	router.ServeHTTP(secondRec, secondReq)
	if secondRec.Code != http.StatusOK {
		t.Fatalf("second list status = %d body=%s", secondRec.Code, secondRec.Body.String())
	}
	var second resourcesResponse
	if err := json.Unmarshal(secondRec.Body.Bytes(), &second); err != nil {
		t.Fatalf("decode second resources: %v", err)
	}
	if second.Count != 1 || len(second.Resources) != 1 || second.Resources[0].FileID != migratedFileID {
		t.Fatalf("second resources = %+v, want unchanged catalog-backed list", second)
	}
}

func TestV2UploadRejectsResourceQuotaAndCleansBlob(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name      string
		metadata  domain.JSONMap
		projectID string
		existing  domain.UpsertResourceInput
	}{
		{
			name: "user quota",
			metadata: domain.JSONMap{
				"resource_quota_count": 1,
			},
			existing: domain.UpsertResourceInput{
				ResourceID:   "file_existing_user_quota",
				OriginalName: "existing.png",
				SizeBytes:    32,
				SourceType:   "upload",
				ResourceKind: "image",
				OwnerUserID:  "quota-user",
				Status:       "active",
			},
		},
		{
			name: "project quota",
			metadata: domain.JSONMap{
				"resource_project_quotas": map[string]any{
					"project-alpha": map[string]any{"max_resources": 1},
				},
			},
			projectID: "project-alpha",
			existing: domain.UpsertResourceInput{
				ResourceID:   "file_existing_project_quota",
				OriginalName: "existing-project.png",
				SizeBytes:    32,
				SourceType:   "upload",
				ResourceKind: "image",
				ProjectID:    "project-alpha",
				OwnerUserID:  "quota-user",
				Status:       "active",
			},
		},
	} {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			uploadRoot := t.TempDir()
			mem := store.NewMemoryStore()
			router := NewRouter(ServerDeps{
				Version:    "test-version",
				Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
				Store:      mem,
				UploadRoot: uploadRoot,
			})
			if _, err := mem.CreateUser(context.Background(), domain.CreateUserInput{
				UserID:   "quota-user",
				Email:    "quota@example.org",
				Metadata: tc.metadata,
			}); err != nil {
				t.Fatalf("CreateUser: %v", err)
			}
			if _, err := mem.UpsertResource(context.Background(), tc.existing); err != nil {
				t.Fatalf("UpsertResource existing: %v", err)
			}

			var body bytes.Buffer
			writer := multipart.NewWriter(&body)
			if tc.projectID != "" {
				if err := writer.WriteField("project_id", tc.projectID); err != nil {
					t.Fatalf("WriteField project_id: %v", err)
				}
			}
			part, err := writer.CreateFormFile("files", "blocked.png")
			if err != nil {
				t.Fatalf("CreateFormFile: %v", err)
			}
			if _, err := part.Write(testPNGBytes(t, 2, 2)); err != nil {
				t.Fatalf("write multipart file: %v", err)
			}
			if err := writer.Close(); err != nil {
				t.Fatalf("close multipart writer: %v", err)
			}

			req := httptest.NewRequest(http.MethodPost, "/v2/uploads", &body)
			req.Header.Set("Content-Type", writer.FormDataContentType())
			req.Header.Set("X-Ultra-User-Id", "quota-user")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)
			if rec.Code != http.StatusRequestEntityTooLarge {
				t.Fatalf("upload status = %d body=%s, want quota response", rec.Code, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), "resource_quota_exceeded") {
				t.Fatalf("quota body = %s, want resource_quota_exceeded", rec.Body.String())
			}
			page, err := mem.ListResourcesForUser(context.Background(), domain.ResourceListInput{UserID: "quota-user", Limit: 20})
			if err != nil {
				t.Fatalf("ListResourcesForUser: %v", err)
			}
			if page.TotalCount != 1 || len(page.Resources) != 1 || page.Resources[0].ResourceID != tc.existing.ResourceID {
				t.Fatalf("catalog resources = %+v, want only pre-existing resource", page)
			}
			files, err := listUploadResources(uploadRoot)
			if err != nil {
				t.Fatalf("list upload root after quota failure: %v", err)
			}
			if len(files) != 0 {
				t.Fatalf("upload root resources after quota failure = %+v, want cleaned root", files)
			}
		})
	}
}

func TestV2UploadRejectsDeclaredBodyAboveDirectUploadLimit(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "too-large.bin")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	if _, err := part.Write([]byte("small body")); err != nil {
		t.Fatalf("write multipart file: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v2/uploads", bytes.NewReader(body.Bytes()))
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("X-Ultra-User-Id", "direct-limit-user")
	req.ContentLength = 6 << 30
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("upload status = %d body=%s, want 413", rec.Code, rec.Body.String())
	}
}

func TestV2UploadSessionResumesChunkAndCommitsResource(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	payload := []byte("field science pdf bytes")
	payloadSHA := sha256.Sum256(payload)
	firstChunk := payload[:6]
	secondChunk := payload[6:]
	firstSHA := sha256.Sum256(firstChunk)
	secondSHA := sha256.Sum256(secondChunk)

	createBody := fmt.Sprintf(`{
		"idempotency_key":"field-session-1",
		"project_id":"field-project",
		"total_bytes":%d,
		"files":[{
			"file_token":"paper-1",
			"original_name":"field-paper.pdf",
			"relative_path":"papers/field-paper.pdf",
			"content_type":"application/pdf",
			"size_bytes":%d,
			"declared_sha256":"%s"
		}]
	}`, len(payload), len(payload), hex.EncodeToString(payloadSHA[:]))
	createReq := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions", strings.NewReader(createBody))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("X-Ultra-User-Id", "field-user")
	createReq.Header.Set("X-Ultra-Org-Id", "field-org")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusCreated {
		t.Fatalf("create upload session status = %d body=%s", createRec.Code, createRec.Body.String())
	}
	var created struct {
		Session domain.UploadSessionRecord       `json:"session"`
		Files   []domain.UploadSessionFileRecord `json:"files"`
		Chunks  []domain.UploadChunkRecord       `json:"chunks"`
	}
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created upload session: %v", err)
	}
	if created.Session.SessionID == "" || created.Session.Status != "active" {
		t.Fatalf("created session = %+v, want active session with id", created.Session)
	}
	if len(created.Files) != 1 || created.Files[0].Status != "pending" {
		t.Fatalf("created files = %+v, want one pending file", created.Files)
	}

	chunkURL := "/v2/upload-sessions/" + created.Session.SessionID + "/files/paper-1/chunks/"
	secondReq := httptest.NewRequest(http.MethodPut, chunkURL+"1", bytes.NewReader(secondChunk))
	secondReq.Header.Set("X-Ultra-User-Id", "field-user")
	secondReq.Header.Set("X-Ultra-Org-Id", "field-org")
	secondReq.Header.Set("X-Upload-Offset", strconv.Itoa(len(firstChunk)))
	secondReq.Header.Set("X-Upload-Chunk-Sha256", hex.EncodeToString(secondSHA[:]))
	secondRec := httptest.NewRecorder()
	router.ServeHTTP(secondRec, secondReq)
	if secondRec.Code != http.StatusOK {
		t.Fatalf("second chunk status = %d body=%s", secondRec.Code, secondRec.Body.String())
	}

	statusReq := httptest.NewRequest(http.MethodGet, "/v2/upload-sessions/"+created.Session.SessionID, nil)
	statusReq.Header.Set("X-Ultra-User-Id", "field-user")
	statusReq.Header.Set("X-Ultra-Org-Id", "field-org")
	statusRec := httptest.NewRecorder()
	router.ServeHTTP(statusRec, statusReq)
	if statusRec.Code != http.StatusOK {
		t.Fatalf("status after partial upload = %d body=%s", statusRec.Code, statusRec.Body.String())
	}
	var partial struct {
		Session domain.UploadSessionRecord `json:"session"`
		Chunks  []domain.UploadChunkRecord `json:"chunks"`
	}
	if err := json.Unmarshal(statusRec.Body.Bytes(), &partial); err != nil {
		t.Fatalf("decode partial upload session: %v", err)
	}
	if partial.Session.BytesVerified != int64(len(secondChunk)) || len(partial.Chunks) != 1 {
		t.Fatalf("partial session = %+v chunks=%+v, want one verified resumed chunk", partial.Session, partial.Chunks)
	}

	resumeReq := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions", strings.NewReader(createBody))
	resumeReq.Header.Set("Content-Type", "application/json")
	resumeReq.Header.Set("X-Ultra-User-Id", "field-user")
	resumeReq.Header.Set("X-Ultra-Org-Id", "field-org")
	resumeRec := httptest.NewRecorder()
	router.ServeHTTP(resumeRec, resumeReq)
	if resumeRec.Code != http.StatusOK {
		t.Fatalf("resume create status = %d body=%s, want existing session", resumeRec.Code, resumeRec.Body.String())
	}
	var resumed struct {
		Session domain.UploadSessionRecord `json:"session"`
		Chunks  []domain.UploadChunkRecord `json:"chunks"`
	}
	if err := json.Unmarshal(resumeRec.Body.Bytes(), &resumed); err != nil {
		t.Fatalf("decode resumed upload session: %v", err)
	}
	if resumed.Session.SessionID != created.Session.SessionID || resumed.Session.BytesVerified != int64(len(secondChunk)) || len(resumed.Chunks) != 1 {
		t.Fatalf("resumed session = %+v chunks=%+v, want same partial session and chunk", resumed.Session, resumed.Chunks)
	}

	firstReq := httptest.NewRequest(http.MethodPut, chunkURL+"0", bytes.NewReader(firstChunk))
	firstReq.Header.Set("X-Ultra-User-Id", "field-user")
	firstReq.Header.Set("X-Ultra-Org-Id", "field-org")
	firstReq.Header.Set("X-Upload-Offset", "0")
	firstReq.Header.Set("X-Upload-Chunk-Sha256", hex.EncodeToString(firstSHA[:]))
	firstRec := httptest.NewRecorder()
	router.ServeHTTP(firstRec, firstReq)
	if firstRec.Code != http.StatusOK {
		t.Fatalf("first chunk status = %d body=%s", firstRec.Code, firstRec.Body.String())
	}

	completeReq := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions/"+created.Session.SessionID+"/files/paper-1/complete", nil)
	completeReq.Header.Set("X-Ultra-User-Id", "field-user")
	completeReq.Header.Set("X-Ultra-Org-Id", "field-org")
	completeRec := httptest.NewRecorder()
	router.ServeHTTP(completeRec, completeReq)
	if completeRec.Code != http.StatusOK {
		t.Fatalf("complete session file status = %d body=%s", completeRec.Code, completeRec.Body.String())
	}
	var completed struct {
		Session  domain.UploadSessionRecord     `json:"session"`
		File     domain.UploadSessionFileRecord `json:"file"`
		Resource uploadedFileRecord             `json:"resource"`
	}
	if err := json.Unmarshal(completeRec.Body.Bytes(), &completed); err != nil {
		t.Fatalf("decode completed upload session file: %v", err)
	}
	if completed.Session.Status != "completed" || completed.Session.BytesCommitted != int64(len(payload)) {
		t.Fatalf("completed session = %+v, want committed bytes and completed status", completed.Session)
	}
	if completed.File.ComputedSHA256 != hex.EncodeToString(payloadSHA[:]) || completed.File.Status != "completed" {
		t.Fatalf("completed file = %+v, want computed sha and completed status", completed.File)
	}
	if completed.Resource.FileID == "" || completed.Resource.SHA256 != hex.EncodeToString(payloadSHA[:]) {
		t.Fatalf("completed resource = %+v, want cataloged upload resource", completed.Resource)
	}
	committedPath := filepath.Join(uploadRoot, completed.Resource.FileID+"__field-paper.pdf")
	committedBytes, err := os.ReadFile(committedPath)
	if err != nil {
		t.Fatalf("read committed upload: %v", err)
	}
	if !bytes.Equal(committedBytes, payload) {
		t.Fatalf("committed payload = %q, want %q", committedBytes, payload)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources?q=field-paper.pdf&limit=20", nil)
	listReq.Header.Set("X-Ultra-User-Id", "field-user")
	listReq.Header.Set("X-Ultra-Org-Id", "field-org")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list resources status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var listed resourcesResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &listed); err != nil {
		t.Fatalf("decode listed resources: %v", err)
	}
	if listed.Count != 1 || len(listed.Resources) != 1 || listed.Resources[0].FileID != completed.Resource.FileID {
		t.Fatalf("listed resources = %+v, want committed session resource", listed)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/upload-sessions/"+created.Session.SessionID, nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "field-user")
	eventsReq.Header.Set("X-Ultra-Org-Id", "field-org")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("status with upload session events = %d body=%s", eventsRec.Code, eventsRec.Body.String())
	}
	var audited struct {
		Events []uploadSessionEventPayload `json:"events"`
	}
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &audited); err != nil {
		t.Fatalf("decode upload session events: %v", err)
	}
	for _, want := range []string{"upload_session.created", "upload_session.file_completed", "upload_session.completed"} {
		if !uploadSessionEventsContain(audited.Events, want) {
			t.Fatalf("upload session events = %+v, want %s", audited.Events, want)
		}
	}
	foundFileCompleted := false
	for _, event := range audited.Events {
		if event.EventType == "upload_session.file_completed" &&
			event.ActorUserID == "field-user" &&
			event.ActorOrgID == "field-org" &&
			event.Metadata["file_token"] == "paper-1" &&
			event.Metadata["resource_id"] == completed.Resource.FileID {
			foundFileCompleted = true
			break
		}
	}
	if !foundFileCompleted {
		t.Fatalf("upload session events = %+v, want file_completed metadata with file token/resource/actor", audited.Events)
	}
}

func TestV2UploadSessionCommittedResourceSupportsCoreFileManagerActions(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	payload := []byte("nifti volume bytes for nph cohort")
	payloadSHA := sha256.Sum256(payload)
	chunkSHA := sha256.Sum256(payload)
	declaredSHA := hex.EncodeToString(payloadSHA[:])

	createBody := fmt.Sprintf(`{
		"idempotency_key":"core-file-manager-session",
		"project_id":"nph-study",
		"total_bytes":%d,
		"files":[{
			"file_token":"nph-volume-a",
			"original_name":"subject-a-nph.nii.gz",
			"relative_path":"nph/subject-a-nph.nii.gz",
			"content_type":"application/gzip",
			"size_bytes":%d,
			"declared_sha256":"%s"
		}]
	}`, len(payload), len(payload), declaredSHA)
	created := createUploadSessionForTest(t, router, createBody, "alice", "org-a", http.StatusCreated)
	uploadChunkForTest(t, router, created.Session.SessionID, "nph-volume-a", 0, 0, payload, hex.EncodeToString(chunkSHA[:]), "alice", "org-a")
	completed := completeUploadSessionFileForTest(t, router, created.Session.SessionID, "nph-volume-a", "alice", "org-a", http.StatusOK)
	resourceID := completed.Resource.FileID
	if resourceID == "" || completed.Resource.SHA256 != declaredSHA || completed.Resource.ProjectID != "nph-study" {
		t.Fatalf("completed resource = %+v, want cataloged nph-study upload with checksum", completed.Resource)
	}

	ownerDownloadReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+resourceID+"/download", nil)
	ownerDownloadReq.Header.Set("X-Ultra-User-Id", "alice")
	ownerDownloadReq.Header.Set("X-Ultra-Org-Id", "org-a")
	ownerDownloadRec := httptest.NewRecorder()
	router.ServeHTTP(ownerDownloadRec, ownerDownloadReq)
	if ownerDownloadRec.Code != http.StatusOK {
		t.Fatalf("owner download status = %d body=%s, want 200", ownerDownloadRec.Code, ownerDownloadRec.Body.String())
	}
	if !bytes.Equal(ownerDownloadRec.Body.Bytes(), payload) {
		t.Fatalf("owner download body = %q, want committed payload", ownerDownloadRec.Body.String())
	}

	createFolderReq := httptest.NewRequest(http.MethodPost, "/v2/resource-collections", strings.NewReader(`{
		"name":"NPH review folder",
		"description":"Shared cohort files for review",
		"collection_type":"folder",
		"project_id":"nph-study",
		"metadata":{"label":"NPH"}
	}`))
	createFolderReq.Header.Set("Content-Type", "application/json")
	createFolderReq.Header.Set("X-Ultra-User-Id", "alice")
	createFolderReq.Header.Set("X-Ultra-Org-Id", "org-a")
	createFolderRec := httptest.NewRecorder()
	router.ServeHTTP(createFolderRec, createFolderReq)
	if createFolderRec.Code != http.StatusCreated {
		t.Fatalf("create folder status = %d body=%s, want 201", createFolderRec.Code, createFolderRec.Body.String())
	}
	var createdFolder resourceCollectionResponse
	if err := json.Unmarshal(createFolderRec.Body.Bytes(), &createdFolder); err != nil {
		t.Fatalf("decode created folder: %v", err)
	}
	if createdFolder.Collection.CollectionID == "" || createdFolder.Collection.Name != "NPH review folder" {
		t.Fatalf("created folder = %+v, want named folder", createdFolder.Collection)
	}

	addReq := httptest.NewRequest(
		http.MethodPost,
		"/v2/resource-collections/"+createdFolder.Collection.CollectionID+"/resources",
		strings.NewReader(fmt.Sprintf(`{"resource_ids":[%q]}`, resourceID)),
	)
	addReq.Header.Set("Content-Type", "application/json")
	addReq.Header.Set("X-Ultra-User-Id", "alice")
	addReq.Header.Set("X-Ultra-Org-Id", "org-a")
	addRec := httptest.NewRecorder()
	router.ServeHTTP(addRec, addReq)
	if addRec.Code != http.StatusOK {
		t.Fatalf("add resource to folder status = %d body=%s, want 200", addRec.Code, addRec.Body.String())
	}

	ownerFolderReq := httptest.NewRequest(http.MethodGet, "/v2/resource-collections/"+createdFolder.Collection.CollectionID+"/resources?q=subject-a&limit=10", nil)
	ownerFolderReq.Header.Set("X-Ultra-User-Id", "alice")
	ownerFolderReq.Header.Set("X-Ultra-Org-Id", "org-a")
	ownerFolderRec := httptest.NewRecorder()
	router.ServeHTTP(ownerFolderRec, ownerFolderReq)
	if ownerFolderRec.Code != http.StatusOK {
		t.Fatalf("owner folder resources status = %d body=%s, want 200", ownerFolderRec.Code, ownerFolderRec.Body.String())
	}
	var ownerFolder resourcesResponse
	if err := json.Unmarshal(ownerFolderRec.Body.Bytes(), &ownerFolder); err != nil {
		t.Fatalf("decode owner folder resources: %v", err)
	}
	if ownerFolder.Count != 1 || len(ownerFolder.Resources) != 1 || ownerFolder.Resources[0].FileID != resourceID {
		t.Fatalf("owner folder resources = %+v, want committed upload resource", ownerFolder)
	}
	if ownerFolder.Resources[0].Principal.UserID != "alice" || ownerFolder.Resources[0].SHA256 != declaredSHA || ownerFolder.Resources[0].Status != "active" {
		t.Fatalf("owner folder resource = %+v, want provenance, checksum, active state", ownerFolder.Resources[0])
	}

	shareReq := httptest.NewRequest(http.MethodPost, "/v2/resource-collections/"+createdFolder.Collection.CollectionID+"/shares", strings.NewReader(`{
		"grantee_user_id":"bob",
		"grantee_org_id":"org-b",
		"role":"read",
		"metadata":{"reason":"collaborative NPH review"}
	}`))
	shareReq.Header.Set("Content-Type", "application/json")
	shareReq.Header.Set("X-Ultra-User-Id", "alice")
	shareReq.Header.Set("X-Ultra-Org-Id", "org-a")
	shareRec := httptest.NewRecorder()
	router.ServeHTTP(shareRec, shareReq)
	if shareRec.Code != http.StatusCreated {
		t.Fatalf("share folder status = %d body=%s, want 201", shareRec.Code, shareRec.Body.String())
	}
	var shareResponse resourceCollectionShareGrantsCreateResponse
	if err := json.Unmarshal(shareRec.Body.Bytes(), &shareResponse); err != nil {
		t.Fatalf("decode folder share response: %v", err)
	}
	if shareResponse.Count != 1 || len(shareResponse.Grants) != 1 || shareResponse.Grants[0].ResourceID != resourceID {
		t.Fatalf("folder share response = %+v, want one inherited resource grant", shareResponse)
	}

	bobCollectionsReq := httptest.NewRequest(http.MethodGet, "/v2/resource-collections?collection_type=folder&q=NPH%20review&limit=10", nil)
	bobCollectionsReq.Header.Set("X-Ultra-User-Id", "bob")
	bobCollectionsReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobCollectionsRec := httptest.NewRecorder()
	router.ServeHTTP(bobCollectionsRec, bobCollectionsReq)
	if bobCollectionsRec.Code != http.StatusOK {
		t.Fatalf("bob folder list status = %d body=%s, want 200", bobCollectionsRec.Code, bobCollectionsRec.Body.String())
	}
	var bobCollections resourceCollectionsResponse
	if err := json.Unmarshal(bobCollectionsRec.Body.Bytes(), &bobCollections); err != nil {
		t.Fatalf("decode bob folder list: %v", err)
	}
	if bobCollections.Count != 1 || len(bobCollections.Collections) != 1 || bobCollections.Collections[0].CollectionID != createdFolder.Collection.CollectionID {
		t.Fatalf("bob folder list = %+v, want shared NPH folder", bobCollections)
	}

	bobFolderReq := httptest.NewRequest(http.MethodGet, "/v2/resource-collections/"+createdFolder.Collection.CollectionID+"/resources?q=subject-a&limit=10", nil)
	bobFolderReq.Header.Set("X-Ultra-User-Id", "bob")
	bobFolderReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobFolderRec := httptest.NewRecorder()
	router.ServeHTTP(bobFolderRec, bobFolderReq)
	if bobFolderRec.Code != http.StatusOK {
		t.Fatalf("bob folder resources status = %d body=%s, want 200", bobFolderRec.Code, bobFolderRec.Body.String())
	}
	var bobFolder resourcesResponse
	if err := json.Unmarshal(bobFolderRec.Body.Bytes(), &bobFolder); err != nil {
		t.Fatalf("decode bob folder resources: %v", err)
	}
	if bobFolder.Count != 1 || len(bobFolder.Resources) != 1 || bobFolder.Resources[0].FileID != resourceID {
		t.Fatalf("bob folder resources = %+v, want shared committed upload resource", bobFolder)
	}
	if !bobFolder.Resources[0].ShareSummary.SharedWithMe {
		t.Fatalf("bob shared resource summary = %+v, want shared_with_me", bobFolder.Resources[0].ShareSummary)
	}

	bobDownloadReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+resourceID+"/download", nil)
	bobDownloadReq.Header.Set("X-Ultra-User-Id", "bob")
	bobDownloadReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobDownloadRec := httptest.NewRecorder()
	router.ServeHTTP(bobDownloadRec, bobDownloadReq)
	if bobDownloadRec.Code != http.StatusOK {
		t.Fatalf("bob download status = %d body=%s, want 200", bobDownloadRec.Code, bobDownloadRec.Body.String())
	}
	if !bytes.Equal(bobDownloadRec.Body.Bytes(), payload) {
		t.Fatalf("bob download body = %q, want original payload", bobDownloadRec.Body.String())
	}

	bobRenameReq := httptest.NewRequest(http.MethodPatch, "/v2/resource-collections/"+createdFolder.Collection.CollectionID, strings.NewReader(`{"name":"Bob rename attempt"}`))
	bobRenameReq.Header.Set("Content-Type", "application/json")
	bobRenameReq.Header.Set("X-Ultra-User-Id", "bob")
	bobRenameReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobRenameRec := httptest.NewRecorder()
	router.ServeHTTP(bobRenameRec, bobRenameReq)
	if bobRenameRec.Code != http.StatusNotFound {
		t.Fatalf("bob rename shared folder status = %d body=%s, want owner-only 404", bobRenameRec.Code, bobRenameRec.Body.String())
	}

	charlieFolderReq := httptest.NewRequest(http.MethodGet, "/v2/resource-collections/"+createdFolder.Collection.CollectionID+"/resources?limit=10", nil)
	charlieFolderReq.Header.Set("X-Ultra-User-Id", "charlie")
	charlieFolderReq.Header.Set("X-Ultra-Org-Id", "org-c")
	charlieFolderRec := httptest.NewRecorder()
	router.ServeHTTP(charlieFolderRec, charlieFolderReq)
	if charlieFolderRec.Code != http.StatusNotFound {
		t.Fatalf("charlie folder resources status = %d body=%s, want 404", charlieFolderRec.Code, charlieFolderRec.Body.String())
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+resourceID+"/events?limit=20", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "alice")
	eventsReq.Header.Set("X-Ultra-Org-Id", "org-a")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("resource events status = %d body=%s, want 200", eventsRec.Code, eventsRec.Body.String())
	}
	var events resourceEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode resource events: %v", err)
	}
	for _, want := range []string{"resource.uploaded", "resource.collection_added", "resource.shared"} {
		if !resourceEventsContain(events.Events, want) {
			t.Fatalf("resource events = %+v, want %s audit event", events.Events, want)
		}
	}
}

func TestV2UploadSessionPauseResumeBlocksWrites(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	payload := []byte("pause resume field upload")
	payloadSHA := sha256.Sum256(payload)
	chunkSHA := sha256.Sum256(payload)
	createBody := uploadSessionCreateBody("pause-session-1", "pause-file", "pause-paper.pdf", "application/pdf", payload, hex.EncodeToString(payloadSHA[:]))

	created := createUploadSessionForTest(t, router, createBody, "pause-user", "pause-org", http.StatusCreated)
	pauseReq := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions/"+created.Session.SessionID+"/pause", nil)
	pauseReq.Header.Set("X-Ultra-User-Id", "pause-user")
	pauseReq.Header.Set("X-Ultra-Org-Id", "pause-org")
	pauseRec := httptest.NewRecorder()
	router.ServeHTTP(pauseRec, pauseReq)
	if pauseRec.Code != http.StatusOK {
		t.Fatalf("pause upload session status = %d body=%s, want 200", pauseRec.Code, pauseRec.Body.String())
	}
	var paused uploadSessionResponse
	if err := json.Unmarshal(pauseRec.Body.Bytes(), &paused); err != nil {
		t.Fatalf("decode paused upload session: %v", err)
	}
	if paused.Session.SessionID != created.Session.SessionID || paused.Session.Status != "paused" {
		t.Fatalf("paused session = %+v, want same paused session", paused.Session)
	}

	chunkReq := httptest.NewRequest(http.MethodPut, "/v2/upload-sessions/"+created.Session.SessionID+"/files/pause-file/chunks/0", bytes.NewReader(payload))
	chunkReq.Header.Set("X-Ultra-User-Id", "pause-user")
	chunkReq.Header.Set("X-Ultra-Org-Id", "pause-org")
	chunkReq.Header.Set("X-Upload-Offset", "0")
	chunkReq.Header.Set("X-Upload-Chunk-Sha256", hex.EncodeToString(chunkSHA[:]))
	chunkRec := httptest.NewRecorder()
	router.ServeHTTP(chunkRec, chunkReq)
	if chunkRec.Code != http.StatusConflict {
		t.Fatalf("chunk while paused status = %d body=%s, want 409", chunkRec.Code, chunkRec.Body.String())
	}
	if !strings.Contains(chunkRec.Body.String(), "paused") {
		t.Fatalf("chunk while paused body = %s, want paused conflict", chunkRec.Body.String())
	}

	completeRec := completeUploadSessionFileRaw(t, router, created.Session.SessionID, "pause-file", "pause-user", "pause-org")
	if completeRec.Code != http.StatusConflict {
		t.Fatalf("complete while paused status = %d body=%s, want 409", completeRec.Code, completeRec.Body.String())
	}
	if !strings.Contains(completeRec.Body.String(), "paused") {
		t.Fatalf("complete while paused body = %s, want paused conflict", completeRec.Body.String())
	}

	resumeReq := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions/"+created.Session.SessionID+"/resume", nil)
	resumeReq.Header.Set("X-Ultra-User-Id", "pause-user")
	resumeReq.Header.Set("X-Ultra-Org-Id", "pause-org")
	resumeRec := httptest.NewRecorder()
	router.ServeHTTP(resumeRec, resumeReq)
	if resumeRec.Code != http.StatusOK {
		t.Fatalf("resume upload session status = %d body=%s, want 200", resumeRec.Code, resumeRec.Body.String())
	}
	var resumed uploadSessionResponse
	if err := json.Unmarshal(resumeRec.Body.Bytes(), &resumed); err != nil {
		t.Fatalf("decode resumed upload session: %v", err)
	}
	if resumed.Session.SessionID != created.Session.SessionID || resumed.Session.Status != "active" {
		t.Fatalf("resumed session = %+v, want same active session", resumed.Session)
	}

	uploadChunkForTest(t, router, created.Session.SessionID, "pause-file", 0, 0, payload, hex.EncodeToString(chunkSHA[:]), "pause-user", "pause-org")
	completed := completeUploadSessionFileForTest(t, router, created.Session.SessionID, "pause-file", "pause-user", "pause-org", http.StatusOK)
	if completed.Session.Status != "completed" || completed.Resource.SHA256 != hex.EncodeToString(payloadSHA[:]) {
		t.Fatalf("completed after resume = %+v resource=%+v, want committed resource", completed.Session, completed.Resource)
	}

	statusReq := httptest.NewRequest(http.MethodGet, "/v2/upload-sessions/"+created.Session.SessionID, nil)
	statusReq.Header.Set("X-Ultra-User-Id", "pause-user")
	statusReq.Header.Set("X-Ultra-Org-Id", "pause-org")
	statusRec := httptest.NewRecorder()
	router.ServeHTTP(statusRec, statusReq)
	if statusRec.Code != http.StatusOK {
		t.Fatalf("status after pause/resume/complete = %d body=%s", statusRec.Code, statusRec.Body.String())
	}
	var audited struct {
		Events []uploadSessionEventPayload `json:"events"`
	}
	if err := json.Unmarshal(statusRec.Body.Bytes(), &audited); err != nil {
		t.Fatalf("decode upload session event stream: %v", err)
	}
	for _, want := range []string{"upload_session.paused", "upload_session.resumed", "upload_session.completed"} {
		if !uploadSessionEventsContain(audited.Events, want) {
			t.Fatalf("upload session events = %+v, want %s", audited.Events, want)
		}
	}
}

func TestV2UploadSessionCancelIsAudited(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	payload := []byte("cancel session audit")
	payloadSHA := sha256.Sum256(payload)
	createBody := uploadSessionCreateBody("cancel-audit-session", "cancel-file", "cancel-paper.pdf", "application/pdf", payload, hex.EncodeToString(payloadSHA[:]))
	created := createUploadSessionForTest(t, router, createBody, "cancel-user", "cancel-org", http.StatusCreated)

	cancelReq := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions/"+created.Session.SessionID+"/cancel", nil)
	cancelReq.Header.Set("X-Ultra-User-Id", "cancel-user")
	cancelReq.Header.Set("X-Ultra-Org-Id", "cancel-org")
	cancelRec := httptest.NewRecorder()
	router.ServeHTTP(cancelRec, cancelReq)
	if cancelRec.Code != http.StatusOK {
		t.Fatalf("cancel upload session status = %d body=%s, want 200", cancelRec.Code, cancelRec.Body.String())
	}
	var canceled struct {
		Session domain.UploadSessionRecord  `json:"session"`
		Events  []uploadSessionEventPayload `json:"events"`
	}
	if err := json.Unmarshal(cancelRec.Body.Bytes(), &canceled); err != nil {
		t.Fatalf("decode canceled upload session: %v", err)
	}
	if canceled.Session.Status != "canceled" {
		t.Fatalf("canceled session = %+v, want canceled status", canceled.Session)
	}
	if !uploadSessionEventsContain(canceled.Events, "upload_session.created") || !uploadSessionEventsContain(canceled.Events, "upload_session.canceled") {
		t.Fatalf("upload session events = %+v, want created and canceled audit events", canceled.Events)
	}
	foundCancel := false
	for _, event := range canceled.Events {
		if event.EventType == "upload_session.canceled" &&
			event.ActorUserID == "cancel-user" &&
			event.ActorOrgID == "cancel-org" &&
			event.Metadata["status"] == "canceled" {
			foundCancel = true
			break
		}
	}
	if !foundCancel {
		t.Fatalf("upload session events = %+v, want canceled metadata with actor and status", canceled.Events)
	}
}

func TestV2UploadSessionCancelRejectsCompletedSession(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	payload := []byte("completed upload sessions are terminal")
	payloadSHA := sha256.Sum256(payload)
	chunkSHA := sha256.Sum256(payload)
	createBody := uploadSessionCreateBody("terminal-cancel-session", "terminal-cancel-file", "terminal-paper.pdf", "application/pdf", payload, hex.EncodeToString(payloadSHA[:]))
	created := createUploadSessionForTest(t, router, createBody, "terminal-user", "terminal-org", http.StatusCreated)

	uploadChunkForTest(t, router, created.Session.SessionID, "terminal-cancel-file", 0, 0, payload, hex.EncodeToString(chunkSHA[:]), "terminal-user", "terminal-org")
	completed := completeUploadSessionFileForTest(t, router, created.Session.SessionID, "terminal-cancel-file", "terminal-user", "terminal-org", http.StatusOK)
	if completed.Session.Status != "completed" {
		t.Fatalf("completed session = %+v, want completed before cancel", completed.Session)
	}

	cancelReq := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions/"+created.Session.SessionID+"/cancel", nil)
	cancelReq.Header.Set("X-Ultra-User-Id", "terminal-user")
	cancelReq.Header.Set("X-Ultra-Org-Id", "terminal-org")
	cancelRec := httptest.NewRecorder()
	router.ServeHTTP(cancelRec, cancelReq)
	if cancelRec.Code != http.StatusConflict {
		t.Fatalf("cancel completed upload session status = %d body=%s, want 409", cancelRec.Code, cancelRec.Body.String())
	}
	if !strings.Contains(cancelRec.Body.String(), "completed") {
		t.Fatalf("cancel completed upload session body = %s, want completed conflict", cancelRec.Body.String())
	}

	statusReq := httptest.NewRequest(http.MethodGet, "/v2/upload-sessions/"+created.Session.SessionID, nil)
	statusReq.Header.Set("X-Ultra-User-Id", "terminal-user")
	statusReq.Header.Set("X-Ultra-Org-Id", "terminal-org")
	statusRec := httptest.NewRecorder()
	router.ServeHTTP(statusRec, statusReq)
	if statusRec.Code != http.StatusOK {
		t.Fatalf("status after rejected terminal cancel = %d body=%s", statusRec.Code, statusRec.Body.String())
	}
	var current struct {
		Session domain.UploadSessionRecord  `json:"session"`
		Events  []uploadSessionEventPayload `json:"events"`
	}
	if err := json.Unmarshal(statusRec.Body.Bytes(), &current); err != nil {
		t.Fatalf("decode terminal upload session status: %v", err)
	}
	if current.Session.Status != "completed" || current.Session.BytesCommitted != int64(len(payload)) {
		t.Fatalf("session after rejected terminal cancel = %+v, want completed with committed bytes", current.Session)
	}
	if uploadSessionEventsContain(current.Events, "upload_session.canceled") {
		t.Fatalf("upload session events = %+v, want no canceled event after rejected terminal cancel", current.Events)
	}
	if !uploadSessionEventsContain(current.Events, "upload_session.completed") {
		t.Fatalf("upload session events = %+v, want completed event preserved", current.Events)
	}
}

func TestV2UploadSessionCancelRejectsCanceledSession(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	payload := []byte("already canceled upload sessions stay terminal")
	payloadSHA := sha256.Sum256(payload)
	createBody := uploadSessionCreateBody("repeated-cancel-session", "repeated-cancel-file", "repeated-cancel.pdf", "application/pdf", payload, hex.EncodeToString(payloadSHA[:]))
	created := createUploadSessionForTest(t, router, createBody, "repeated-cancel-user", "repeated-cancel-org", http.StatusCreated)

	firstCancelReq := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions/"+created.Session.SessionID+"/cancel", nil)
	firstCancelReq.Header.Set("X-Ultra-User-Id", "repeated-cancel-user")
	firstCancelReq.Header.Set("X-Ultra-Org-Id", "repeated-cancel-org")
	firstCancelRec := httptest.NewRecorder()
	router.ServeHTTP(firstCancelRec, firstCancelReq)
	if firstCancelRec.Code != http.StatusOK {
		t.Fatalf("initial cancel upload session status = %d body=%s, want 200", firstCancelRec.Code, firstCancelRec.Body.String())
	}

	secondCancelReq := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions/"+created.Session.SessionID+"/cancel", nil)
	secondCancelReq.Header.Set("X-Ultra-User-Id", "repeated-cancel-user")
	secondCancelReq.Header.Set("X-Ultra-Org-Id", "repeated-cancel-org")
	secondCancelRec := httptest.NewRecorder()
	router.ServeHTTP(secondCancelRec, secondCancelReq)
	if secondCancelRec.Code != http.StatusConflict {
		t.Fatalf("repeated cancel upload session status = %d body=%s, want 409", secondCancelRec.Code, secondCancelRec.Body.String())
	}
	if !strings.Contains(secondCancelRec.Body.String(), "canceled") {
		t.Fatalf("repeated cancel upload session body = %s, want canceled conflict", secondCancelRec.Body.String())
	}

	statusReq := httptest.NewRequest(http.MethodGet, "/v2/upload-sessions/"+created.Session.SessionID, nil)
	statusReq.Header.Set("X-Ultra-User-Id", "repeated-cancel-user")
	statusReq.Header.Set("X-Ultra-Org-Id", "repeated-cancel-org")
	statusRec := httptest.NewRecorder()
	router.ServeHTTP(statusRec, statusReq)
	if statusRec.Code != http.StatusOK {
		t.Fatalf("status after repeated cancel = %d body=%s", statusRec.Code, statusRec.Body.String())
	}
	var current struct {
		Session domain.UploadSessionRecord  `json:"session"`
		Events  []uploadSessionEventPayload `json:"events"`
	}
	if err := json.Unmarshal(statusRec.Body.Bytes(), &current); err != nil {
		t.Fatalf("decode repeated-cancel upload session status: %v", err)
	}
	if current.Session.Status != "canceled" {
		t.Fatalf("session after repeated cancel = %+v, want canceled terminal state", current.Session)
	}
	cancelEvents := 0
	for _, event := range current.Events {
		if event.EventType == "upload_session.canceled" {
			cancelEvents++
		}
	}
	if cancelEvents != 1 {
		t.Fatalf("upload session events = %+v, want exactly one canceled event", current.Events)
	}
}

func TestV2UploadSessionRejectsChecksumMismatch(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	payload := []byte("valid chunk bytes")
	payloadSHA := sha256.Sum256(payload)
	createBody := fmt.Sprintf(`{
		"total_bytes":%d,
		"files":[{
			"file_token":"nii-1",
			"original_name":"brain.nii",
			"content_type":"application/x-nifti",
			"size_bytes":%d,
			"declared_sha256":"%s"
		}]
	}`, len(payload), len(payload), hex.EncodeToString(payloadSHA[:]))
	createReq := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions", strings.NewReader(createBody))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("X-Ultra-User-Id", "field-user")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusCreated {
		t.Fatalf("create upload session status = %d body=%s", createRec.Code, createRec.Body.String())
	}
	var created struct {
		Session domain.UploadSessionRecord `json:"session"`
	}
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created upload session: %v", err)
	}

	chunkReq := httptest.NewRequest(http.MethodPut, "/v2/upload-sessions/"+created.Session.SessionID+"/files/nii-1/chunks/0", bytes.NewReader(payload))
	chunkReq.Header.Set("X-Ultra-User-Id", "field-user")
	chunkReq.Header.Set("X-Upload-Offset", "0")
	chunkReq.Header.Set("X-Upload-Chunk-Sha256", strings.Repeat("0", 64))
	chunkRec := httptest.NewRecorder()
	router.ServeHTTP(chunkRec, chunkReq)
	if chunkRec.Code != http.StatusBadRequest {
		t.Fatalf("mismatched chunk status = %d body=%s, want 400", chunkRec.Code, chunkRec.Body.String())
	}
	if !strings.Contains(chunkRec.Body.String(), "chunk checksum mismatch") {
		t.Fatalf("mismatch body = %s, want checksum mismatch", chunkRec.Body.String())
	}

	statusReq := httptest.NewRequest(http.MethodGet, "/v2/upload-sessions/"+created.Session.SessionID, nil)
	statusReq.Header.Set("X-Ultra-User-Id", "field-user")
	statusRec := httptest.NewRecorder()
	router.ServeHTTP(statusRec, statusReq)
	if statusRec.Code != http.StatusOK {
		t.Fatalf("status after mismatch = %d body=%s", statusRec.Code, statusRec.Body.String())
	}
	var status struct {
		Session domain.UploadSessionRecord `json:"session"`
		Chunks  []domain.UploadChunkRecord `json:"chunks"`
	}
	if err := json.Unmarshal(statusRec.Body.Bytes(), &status); err != nil {
		t.Fatalf("decode status after mismatch: %v", err)
	}
	if status.Session.BytesReceived != 0 || status.Session.BytesVerified != 0 {
		t.Fatalf("status after mismatch = %+v, want no durable received or verified bytes", status.Session)
	}
	if len(status.Chunks) != 1 {
		t.Fatalf("chunks after mismatch = %+v, want one durable failed chunk attempt", status.Chunks)
	}
	failedChunk := status.Chunks[0]
	if failedChunk.Status != "failed" || failedChunk.Error != "chunk checksum mismatch" || failedChunk.SizeBytes != int64(len(payload)) {
		t.Fatalf("failed chunk after mismatch = %+v, want failed checksum record with attempted size", failedChunk)
	}
}

func TestV2UploadSessionRetriesFailedChecksumChunkAndCommits(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	payload := []byte("retry after failed checksum chunk")
	payloadSHA := sha256.Sum256(payload)
	createBody := uploadSessionCreateBody("retry-failed-chunk-session", "retry-failed-chunk-file", "retry-failed.nii.gz", "application/x-nifti", payload, hex.EncodeToString(payloadSHA[:]))
	created := createUploadSessionForTest(t, router, createBody, "retry-chunk-user", "retry-chunk-org", http.StatusCreated)

	badReq := httptest.NewRequest(http.MethodPut, "/v2/upload-sessions/"+created.Session.SessionID+"/files/retry-failed-chunk-file/chunks/0", bytes.NewReader(payload))
	badReq.Header.Set("X-Ultra-User-Id", "retry-chunk-user")
	badReq.Header.Set("X-Ultra-Org-Id", "retry-chunk-org")
	badReq.Header.Set("X-Upload-Offset", "0")
	badReq.Header.Set("X-Upload-Chunk-Sha256", strings.Repeat("0", 64))
	badRec := httptest.NewRecorder()
	router.ServeHTTP(badRec, badReq)
	if badRec.Code != http.StatusBadRequest {
		t.Fatalf("failed chunk attempt status = %d body=%s, want 400", badRec.Code, badRec.Body.String())
	}

	statusAfterFailureReq := httptest.NewRequest(http.MethodGet, "/v2/upload-sessions/"+created.Session.SessionID, nil)
	statusAfterFailureReq.Header.Set("X-Ultra-User-Id", "retry-chunk-user")
	statusAfterFailureReq.Header.Set("X-Ultra-Org-Id", "retry-chunk-org")
	statusAfterFailureRec := httptest.NewRecorder()
	router.ServeHTTP(statusAfterFailureRec, statusAfterFailureReq)
	if statusAfterFailureRec.Code != http.StatusOK {
		t.Fatalf("status after failed chunk attempt = %d body=%s", statusAfterFailureRec.Code, statusAfterFailureRec.Body.String())
	}
	var failedStatus struct {
		Session domain.UploadSessionRecord `json:"session"`
		Chunks  []domain.UploadChunkRecord `json:"chunks"`
	}
	if err := json.Unmarshal(statusAfterFailureRec.Body.Bytes(), &failedStatus); err != nil {
		t.Fatalf("decode status after failed chunk attempt: %v", err)
	}
	if failedStatus.Session.BytesReceived != 0 || failedStatus.Session.BytesVerified != 0 || len(failedStatus.Chunks) != 1 || failedStatus.Chunks[0].Status != "failed" {
		t.Fatalf("status after failed chunk attempt = session %+v chunks %+v, want one uncounted failed chunk", failedStatus.Session, failedStatus.Chunks)
	}

	uploadChunkForTest(t, router, created.Session.SessionID, "retry-failed-chunk-file", 0, 0, payload, hex.EncodeToString(payloadSHA[:]), "retry-chunk-user", "retry-chunk-org")

	statusAfterRetryReq := httptest.NewRequest(http.MethodGet, "/v2/upload-sessions/"+created.Session.SessionID, nil)
	statusAfterRetryReq.Header.Set("X-Ultra-User-Id", "retry-chunk-user")
	statusAfterRetryReq.Header.Set("X-Ultra-Org-Id", "retry-chunk-org")
	statusAfterRetryRec := httptest.NewRecorder()
	router.ServeHTTP(statusAfterRetryRec, statusAfterRetryReq)
	if statusAfterRetryRec.Code != http.StatusOK {
		t.Fatalf("status after chunk retry = %d body=%s", statusAfterRetryRec.Code, statusAfterRetryRec.Body.String())
	}
	var retriedStatus struct {
		Session domain.UploadSessionRecord `json:"session"`
		Chunks  []domain.UploadChunkRecord `json:"chunks"`
	}
	if err := json.Unmarshal(statusAfterRetryRec.Body.Bytes(), &retriedStatus); err != nil {
		t.Fatalf("decode status after chunk retry: %v", err)
	}
	if retriedStatus.Session.BytesReceived != int64(len(payload)) || retriedStatus.Session.BytesVerified != int64(len(payload)) {
		t.Fatalf("status after chunk retry = %+v, want only retried bytes counted", retriedStatus.Session)
	}
	if len(retriedStatus.Chunks) != 1 || retriedStatus.Chunks[0].Status != "verified" || retriedStatus.Chunks[0].Error != "" {
		t.Fatalf("chunks after retry = %+v, want failed manifest replaced by verified chunk", retriedStatus.Chunks)
	}

	completed := completeUploadSessionFileForTest(t, router, created.Session.SessionID, "retry-failed-chunk-file", "retry-chunk-user", "retry-chunk-org", http.StatusOK)
	if completed.Session.Status != "completed" || completed.Session.BytesCommitted != int64(len(payload)) || completed.Resource.SHA256 != hex.EncodeToString(payloadSHA[:]) {
		t.Fatalf("completed after failed chunk retry = session %+v resource %+v, want committed retried payload", completed.Session, completed.Resource)
	}
}

func TestV2UploadSessionCompleteFileRetryIsIdempotent(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	files, body, totalBytes := largeFolderUploadFixture(t, 2, "idempotent-complete-session")
	created := createUploadSessionForTest(t, router, string(body), "complete-retry-user", "complete-retry-org", http.StatusCreated)

	firstFile := files[0]
	uploadChunkForTest(t, router, created.Session.SessionID, firstFile.token, 0, 0, firstFile.payload, firstFile.sha, "complete-retry-user", "complete-retry-org")
	firstComplete := completeUploadSessionFileForTest(t, router, created.Session.SessionID, firstFile.token, "complete-retry-user", "complete-retry-org", http.StatusOK)
	if firstComplete.Session.Status != "active" || firstComplete.File.Status != "completed" || firstComplete.Resource.FileID == "" {
		t.Fatalf("first complete = session %+v file %+v resource %+v, want active session with completed file", firstComplete.Session, firstComplete.File, firstComplete.Resource)
	}

	retriedComplete := completeUploadSessionFileForTest(t, router, created.Session.SessionID, firstFile.token, "complete-retry-user", "complete-retry-org", http.StatusOK)
	if retriedComplete.Session.Status != "active" || retriedComplete.File.Status != "completed" || retriedComplete.Resource.FileID != firstComplete.Resource.FileID {
		t.Fatalf("retried complete = session %+v file %+v resource %+v, want same completed file/resource on active session", retriedComplete.Session, retriedComplete.File, retriedComplete.Resource)
	}

	secondFile := files[1]
	uploadChunkForTest(t, router, created.Session.SessionID, secondFile.token, 0, 0, secondFile.payload, secondFile.sha, "complete-retry-user", "complete-retry-org")
	secondComplete := completeUploadSessionFileForTest(t, router, created.Session.SessionID, secondFile.token, "complete-retry-user", "complete-retry-org", http.StatusOK)
	if secondComplete.Session.Status != "completed" || secondComplete.Session.BytesCommitted != totalBytes {
		t.Fatalf("second complete = %+v, want completed session with %d committed bytes", secondComplete.Session, totalBytes)
	}

	statusReq := httptest.NewRequest(http.MethodGet, "/v2/upload-sessions/"+created.Session.SessionID, nil)
	statusReq.Header.Set("X-Ultra-User-Id", "complete-retry-user")
	statusReq.Header.Set("X-Ultra-Org-Id", "complete-retry-org")
	statusRec := httptest.NewRecorder()
	router.ServeHTTP(statusRec, statusReq)
	if statusRec.Code != http.StatusOK {
		t.Fatalf("status after idempotent complete retry = %d body=%s", statusRec.Code, statusRec.Body.String())
	}
	var current struct {
		Events []uploadSessionEventPayload `json:"events"`
	}
	if err := json.Unmarshal(statusRec.Body.Bytes(), &current); err != nil {
		t.Fatalf("decode upload session events after idempotent complete retry: %v", err)
	}
	firstFileCompletedEvents := 0
	for _, event := range current.Events {
		if event.EventType == "upload_session.file_completed" && event.Metadata["file_token"] == firstFile.token {
			firstFileCompletedEvents++
		}
	}
	if firstFileCompletedEvents != 1 {
		t.Fatalf("upload session events = %+v, want one file_completed event for retried file", current.Events)
	}
}

// TestV2UploadSessionCompleteFileConcurrentDoesNotDuplicate fires many simultaneous
// /complete calls for the SAME file (a client retry / double-submit / network replay).
// Without serialization each would mint a distinct resourceID and catalog a separate
// resource for one upload — duplicate entry, leaked bytes on disk, double-charged quota.
// The completion lock must make exactly one resource win and the rest observe it.
func TestV2UploadSessionCompleteFileConcurrentDoesNotDuplicate(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	const userID, orgID = "concurrent-complete-user", "concurrent-complete-org"
	files, body, _ := largeFolderUploadFixture(t, 1, "concurrent-complete-session")
	created := createUploadSessionForTest(t, router, string(body), userID, orgID, http.StatusCreated)
	f := files[0]
	uploadChunkForTest(t, router, created.Session.SessionID, f.token, 0, 0, f.payload, f.sha, userID, orgID)

	const racers = 8
	recs := make([]*httptest.ResponseRecorder, racers)
	start := make(chan struct{})
	var wg sync.WaitGroup
	url := fmt.Sprintf("/v2/upload-sessions/%s/files/%s/complete", created.Session.SessionID, f.token)
	for i := 0; i < racers; i++ {
		wg.Add(1)
		go func(i int) { // no t.* calls in goroutines: build + serve inline
			defer wg.Done()
			req := httptest.NewRequest(http.MethodPost, url, nil)
			req.Header.Set("X-Ultra-User-Id", userID)
			req.Header.Set("X-Ultra-Org-Id", orgID)
			rec := httptest.NewRecorder()
			<-start // line everyone up to maximize contention
			router.ServeHTTP(rec, req)
			recs[i] = rec
		}(i)
	}
	close(start)
	wg.Wait()

	resourceIDs := map[string]struct{}{}
	okCount := 0
	for i, rec := range recs {
		switch rec.Code {
		case http.StatusOK:
			okCount++
			var resp uploadSessionFileCompleteResponse
			if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
				t.Fatalf("racer %d decode: %v", i, err)
			}
			if resp.Resource.FileID == "" {
				t.Fatalf("racer %d completed with empty resource id: %s", i, rec.Body.String())
			}
			resourceIDs[resp.Resource.FileID] = struct{}{}
		case http.StatusConflict:
			// acceptable: a loser may legitimately 409 if it observes an in-progress commit
		default:
			t.Fatalf("racer %d status = %d body=%s, want 200 or 409", i, rec.Code, rec.Body.String())
		}
	}
	if okCount == 0 {
		t.Fatalf("no racer completed the file successfully")
	}
	if len(resourceIDs) != 1 {
		t.Fatalf("concurrent completion produced %d distinct resources %v, want exactly 1", len(resourceIDs), resourceIDs)
	}

	// The catalog must hold exactly one resource — no duplicate / orphaned entry.
	records, err := mem.ListResources(context.Background(), 100, 0)
	if err != nil {
		t.Fatalf("ListResources: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("catalog has %d resources after concurrent completion, want exactly 1", len(records))
	}

	// And exactly one file_completed event was recorded.
	statusReq := httptest.NewRequest(http.MethodGet, "/v2/upload-sessions/"+created.Session.SessionID, nil)
	statusReq.Header.Set("X-Ultra-User-Id", userID)
	statusReq.Header.Set("X-Ultra-Org-Id", orgID)
	statusRec := httptest.NewRecorder()
	router.ServeHTTP(statusRec, statusReq)
	var current struct {
		Events []uploadSessionEventPayload `json:"events"`
	}
	if err := json.Unmarshal(statusRec.Body.Bytes(), &current); err != nil {
		t.Fatalf("decode upload session events: %v", err)
	}
	completedEvents := 0
	for _, event := range current.Events {
		if event.EventType == "upload_session.file_completed" && event.Metadata["file_token"] == f.token {
			completedEvents++
		}
	}
	if completedEvents != 1 {
		t.Fatalf("file_completed events = %d, want exactly 1", completedEvents)
	}
}

func TestV2UploadSessionResumesAfterHandlerRestart(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	payload := []byte("restart durable upload bytes")
	firstChunk := payload[:8]
	secondChunk := payload[8:]
	payloadSHA := sha256.Sum256(payload)
	firstSHA := sha256.Sum256(firstChunk)
	secondSHA := sha256.Sum256(secondChunk)
	createBody := uploadSessionCreateBody("restart-session-1", "restart-file", "restart-paper.pdf", "application/pdf", payload, hex.EncodeToString(payloadSHA[:]))

	created := createUploadSessionForTest(t, router, createBody, "restart-user", "restart-org", http.StatusCreated)
	if created.Limits.MaxParallelChunks != 8 || created.Limits.MaxParallelFiles != 4 || created.Limits.MaxFilesPerSession < 1000 {
		t.Fatalf("upload session limits = %+v, want default backpressure hints", created.Limits)
	}
	uploadChunkForTest(t, router, created.Session.SessionID, "restart-file", 0, 0, firstChunk, hex.EncodeToString(firstSHA[:]), "restart-user", "restart-org")

	restartedRouter := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	resumed := createUploadSessionForTest(t, restartedRouter, createBody, "restart-user", "restart-org", http.StatusOK)
	if resumed.Session.SessionID != created.Session.SessionID || resumed.Session.BytesVerified != int64(len(firstChunk)) || len(resumed.Chunks) != 1 {
		t.Fatalf("resumed after restart = %+v chunks=%+v, want same verified partial session", resumed.Session, resumed.Chunks)
	}

	uploadChunkForTest(t, restartedRouter, resumed.Session.SessionID, "restart-file", 1, int64(len(firstChunk)), secondChunk, hex.EncodeToString(secondSHA[:]), "restart-user", "restart-org")
	completed := completeUploadSessionFileForTest(t, restartedRouter, resumed.Session.SessionID, "restart-file", "restart-user", "restart-org", http.StatusOK)
	if completed.Session.Status != "completed" || completed.Resource.SHA256 != hex.EncodeToString(payloadSHA[:]) {
		t.Fatalf("completed after restart = %+v resource=%+v, want completed expected sha", completed.Session, completed.Resource)
	}
}

func TestV2UploadSessionRejectsIdempotencyReplayWithDifferentManifest(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: t.TempDir(),
	})
	firstPayload := []byte("first folder manifest")
	firstSHA := sha256.Sum256(firstPayload)
	firstBody := uploadSessionCreateBody("field-folder-idem", "scan-a", "scan-a.nii.gz", "application/gzip", firstPayload, hex.EncodeToString(firstSHA[:]))

	created := createUploadSessionForTest(t, router, firstBody, "idem-user", "idem-org", http.StatusCreated)
	replayed := createUploadSessionForTest(t, router, firstBody, "idem-user", "idem-org", http.StatusOK)
	if replayed.Session.SessionID != created.Session.SessionID {
		t.Fatalf("exact idempotency replay returned session %q, want original %q", replayed.Session.SessionID, created.Session.SessionID)
	}

	secondPayload := []byte("different folder manifest")
	secondSHA := sha256.Sum256(secondPayload)
	secondBody := uploadSessionCreateBody("field-folder-idem", "scan-b", "scan-b.nii.gz", "application/gzip", secondPayload, hex.EncodeToString(secondSHA[:]))
	req := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions", strings.NewReader(secondBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-User-Id", "idem-user")
	req.Header.Set("X-Ultra-Org-Id", "idem-org")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusConflict {
		t.Fatalf("changed idempotency replay status = %d body=%s, want 409", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "idempotency") {
		t.Fatalf("changed idempotency replay body = %s, want idempotency conflict", rec.Body.String())
	}
}

func TestV2UploadSessionResumesAfterPostgresBackedHandlerRestart(t *testing.T) {
	dsn := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL"))
	if dsn == "" {
		t.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	firstPool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New first: %v", err)
	}
	if err := store.ApplyPostgresSchema(ctx, firstPool); err != nil {
		firstPool.Close()
		t.Fatalf("ApplyPostgresSchema: %v", err)
	}

	uploadRoot := t.TempDir()
	firstStore := store.NewPostgresStore(firstPool)
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(firstStore, eventbus.NewMemoryBus()),
		Store:      firstStore,
		UploadRoot: uploadRoot,
	})
	payload := []byte("postgres durable restart upload bytes")
	firstChunk := payload[:17]
	secondChunk := payload[17:]
	payloadSHA := sha256.Sum256(payload)
	firstSHA := sha256.Sum256(firstChunk)
	secondSHA := sha256.Sum256(secondChunk)
	suffix := strings.ReplaceAll(domain.NewID("pg_restart"), "-", "_")
	userID := "pg-restart-user-" + suffix
	orgID := "pg-restart-org-" + suffix
	fileToken := "pg-restart-file-" + suffix
	createBody := uploadSessionCreateBody("pg-restart-session-"+suffix, fileToken, "postgres-restart-paper.pdf", "application/pdf", payload, hex.EncodeToString(payloadSHA[:]))

	created := createUploadSessionForTest(t, router, createBody, userID, orgID, http.StatusCreated)
	uploadChunkForTest(t, router, created.Session.SessionID, fileToken, 0, 0, firstChunk, hex.EncodeToString(firstSHA[:]), userID, orgID)
	firstPool.Close()

	secondPool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool.New second: %v", err)
	}
	defer secondPool.Close()
	secondStore := store.NewPostgresStore(secondPool)
	restartedRouter := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(secondStore, eventbus.NewMemoryBus()),
		Store:      secondStore,
		UploadRoot: uploadRoot,
	})

	resumed := createUploadSessionForTest(t, restartedRouter, createBody, userID, orgID, http.StatusOK)
	if resumed.Session.SessionID != created.Session.SessionID || resumed.Session.BytesVerified != int64(len(firstChunk)) || len(resumed.Chunks) != 1 {
		t.Fatalf("postgres resumed session = %+v chunks=%+v, want same verified partial session", resumed.Session, resumed.Chunks)
	}
	statusReq := httptest.NewRequest(http.MethodGet, "/v2/upload-sessions/"+created.Session.SessionID, nil)
	statusReq.Header.Set("X-Ultra-User-Id", userID)
	statusReq.Header.Set("X-Ultra-Org-Id", orgID)
	statusRec := httptest.NewRecorder()
	restartedRouter.ServeHTTP(statusRec, statusReq)
	if statusRec.Code != http.StatusOK {
		t.Fatalf("postgres status after restart = %d body=%s", statusRec.Code, statusRec.Body.String())
	}
	var status uploadSessionResponse
	if err := json.Unmarshal(statusRec.Body.Bytes(), &status); err != nil {
		t.Fatalf("decode postgres status after restart: %v", err)
	}
	if status.Session.BytesVerified != int64(len(firstChunk)) || len(status.Chunks) != 1 || status.Chunks[0].Status != "verified" {
		t.Fatalf("postgres status after restart = %+v chunks=%+v, want persisted verified chunk", status.Session, status.Chunks)
	}

	uploadChunkForTest(t, restartedRouter, resumed.Session.SessionID, fileToken, 1, int64(len(firstChunk)), secondChunk, hex.EncodeToString(secondSHA[:]), userID, orgID)
	completed := completeUploadSessionFileForTest(t, restartedRouter, resumed.Session.SessionID, fileToken, userID, orgID, http.StatusOK)
	if completed.Session.Status != "completed" || completed.Session.BytesCommitted != int64(len(payload)) || completed.Resource.SHA256 != hex.EncodeToString(payloadSHA[:]) {
		t.Fatalf("postgres completed after restart = %+v resource=%+v, want committed expected sha", completed.Session, completed.Resource)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources?q=postgres-restart-paper.pdf&limit=20", nil)
	listReq.Header.Set("X-Ultra-User-Id", userID)
	listReq.Header.Set("X-Ultra-Org-Id", orgID)
	listRec := httptest.NewRecorder()
	restartedRouter.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("postgres list resources after restart = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var listed resourcesResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &listed); err != nil {
		t.Fatalf("decode postgres listed resources: %v", err)
	}
	if listed.Count != 1 || len(listed.Resources) != 1 || listed.Resources[0].FileID != completed.Resource.FileID {
		t.Fatalf("postgres listed resources = %+v, want committed upload resource", listed)
	}
}

func TestV2UploadSessionRejectsTooManyFilesForBackpressure(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: t.TempDir(),
	})
	createReq := createUploadSessionRequest{
		IdempotencyKey: "too-many-files-session",
		TotalBytes:     int64(uploadSessionMaxFilesPerBatch + 1),
		Files:          make([]createUploadSessionFileRequest, 0, uploadSessionMaxFilesPerBatch+1),
	}
	for i := 0; i <= uploadSessionMaxFilesPerBatch; i++ {
		createReq.Files = append(createReq.Files, createUploadSessionFileRequest{
			FileToken:    fmt.Sprintf("tile-%05d", i),
			OriginalName: fmt.Sprintf("tile_%05d.ome.tiff", i),
			ContentType:  "image/tiff",
			SizeBytes:    1,
		})
	}
	body, err := json.Marshal(createReq)
	if err != nil {
		t.Fatalf("marshal oversized upload-session manifest: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-User-Id", "too-many-user")
	req.Header.Set("X-Ultra-Org-Id", "too-many-org")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("oversized upload-session manifest status = %d body=%s, want 400", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "cannot include more than 10000 files") {
		t.Fatalf("oversized upload-session body = %s, want file-count limit", rec.Body.String())
	}
}

func TestV2UploadSessionDeduplicatesExistingResourceByChecksum(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	payload := []byte("duplicate scientific payload")
	payloadSHA := sha256.Sum256(payload)
	chunkSHA := sha256.Sum256(payload)

	firstBody := uploadSessionCreateBody("dedupe-session-1", "dedupe-a", "same-paper.pdf", "application/pdf", payload, hex.EncodeToString(payloadSHA[:]))
	first := createUploadSessionForTest(t, router, firstBody, "dedupe-user", "dedupe-org", http.StatusCreated)
	uploadChunkForTest(t, router, first.Session.SessionID, "dedupe-a", 0, 0, payload, hex.EncodeToString(chunkSHA[:]), "dedupe-user", "dedupe-org")
	firstComplete := completeUploadSessionFileForTest(t, router, first.Session.SessionID, "dedupe-a", "dedupe-user", "dedupe-org", http.StatusOK)

	secondBody := uploadSessionCreateBody("dedupe-session-2", "dedupe-b", "same-paper.pdf", "application/pdf", payload, hex.EncodeToString(payloadSHA[:]))
	second := createUploadSessionForTest(t, router, secondBody, "dedupe-user", "dedupe-org", http.StatusCreated)
	uploadChunkForTest(t, router, second.Session.SessionID, "dedupe-b", 0, 0, payload, hex.EncodeToString(chunkSHA[:]), "dedupe-user", "dedupe-org")
	secondComplete := completeUploadSessionFileForTest(t, router, second.Session.SessionID, "dedupe-b", "dedupe-user", "dedupe-org", http.StatusOK)

	if secondComplete.Resource.FileID != firstComplete.Resource.FileID {
		t.Fatalf("duplicate upload returned file_id %q, want existing %q", secondComplete.Resource.FileID, firstComplete.Resource.FileID)
	}
	resources, err := listUploadResources(uploadRoot)
	if err != nil {
		t.Fatalf("list upload resources: %v", err)
	}
	if len(resources) != 1 {
		t.Fatalf("upload root resources = %+v, want one committed blob after dedupe", resources)
	}
}

func TestV2UploadSessionChunkRejectsBodyBeyondDeclaredFileSize(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	declaredPayload := []byte("tiny")
	declaredSHA := sha256.Sum256(declaredPayload)
	created := createUploadSessionForTest(
		t,
		router,
		uploadSessionCreateBody("oversize-chunk-session", "oversize-file", "oversize.bin", "application/octet-stream", declaredPayload, hex.EncodeToString(declaredSHA[:])),
		"oversize-user",
		"oversize-org",
		http.StatusCreated,
	)

	oversizedPayload := []byte("this payload is larger than declared")
	chunkSHA := sha256.Sum256(oversizedPayload)
	req := httptest.NewRequest(
		http.MethodPut,
		"/v2/upload-sessions/"+created.Session.SessionID+"/files/oversize-file/chunks/0",
		bytes.NewReader(oversizedPayload),
	)
	req.Header.Set("X-Ultra-User-Id", "oversize-user")
	req.Header.Set("X-Ultra-Org-Id", "oversize-org")
	req.Header.Set("X-Upload-Offset", "0")
	req.Header.Set("X-Upload-Chunk-Sha256", hex.EncodeToString(chunkSHA[:]))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversized chunk status = %d body=%s, want 413", rec.Code, rec.Body.String())
	}
}

func TestV2UploadSessionCompletesZeroByteFileWithoutChunks(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	emptySHA := sha256.Sum256(nil)
	createBody := uploadSessionCreateBody("empty-file-session-1", "empty-file", "empty-marker.txt", "text/plain", nil, hex.EncodeToString(emptySHA[:]))
	created := createUploadSessionForTest(t, router, createBody, "empty-user", "empty-org", http.StatusCreated)
	if created.Session.TotalBytes != 0 || len(created.Chunks) != 0 {
		t.Fatalf("created empty file session = %+v chunks=%+v, want zero-byte session with no chunks", created.Session, created.Chunks)
	}

	completed := completeUploadSessionFileForTest(t, router, created.Session.SessionID, "empty-file", "empty-user", "empty-org", http.StatusOK)
	if completed.Session.Status != "completed" || completed.Session.BytesCommitted != 0 {
		t.Fatalf("completed empty file session = %+v, want completed with zero committed bytes", completed.Session)
	}
	if completed.File.Status != "completed" || completed.File.SizeBytes != 0 {
		t.Fatalf("completed empty file record = %+v, want completed zero-byte file", completed.File)
	}
	if completed.Resource.OriginalName != "empty-marker.txt" || completed.Resource.SizeBytes != 0 || completed.Resource.SHA256 != hex.EncodeToString(emptySHA[:]) {
		t.Fatalf("completed empty resource = %+v, want empty file catalog record", completed.Resource)
	}
	committedPath := filepath.Join(uploadRoot, completed.Resource.FileID+"__empty-marker.txt")
	committedBytes, err := os.ReadFile(committedPath)
	if err != nil {
		t.Fatalf("read committed empty upload: %v", err)
	}
	if len(committedBytes) != 0 {
		t.Fatalf("committed empty file length = %d, want 0", len(committedBytes))
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources?q=empty-marker.txt&limit=20", nil)
	listReq.Header.Set("X-Ultra-User-Id", "empty-user")
	listReq.Header.Set("X-Ultra-Org-Id", "empty-org")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list resources after empty upload = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var listed resourcesResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &listed); err != nil {
		t.Fatalf("decode empty upload resources: %v", err)
	}
	if listed.Count != 1 || len(listed.Resources) != 1 || listed.Resources[0].FileID != completed.Resource.FileID {
		t.Fatalf("listed empty upload resources = %+v, want committed empty resource", listed)
	}
}

func TestV2UploadSessionLargeFolderCatalogsManySmallFilesWithoutPerFileChunkScans(t *testing.T) {
	t.Parallel()

	const fileCount = 64
	uploadRoot := t.TempDir()
	counted := &countingUploadStore{
		MemoryStore:     store.NewMemoryStore(),
		chunksBySession: map[string][]domain.UploadChunkRecord{},
	}
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(counted, eventbus.NewMemoryBus()),
		Store:      counted,
		UploadRoot: uploadRoot,
	})

	files, body, totalBytes := largeFolderUploadFixture(t, fileCount, "large-folder-session-1")

	created := createUploadSessionForTest(t, router, string(body), "large-folder-user", "large-folder-org", http.StatusCreated)
	if len(created.Files) != fileCount || created.Session.TotalBytes != totalBytes {
		t.Fatalf("created large folder session files=%d total=%d, want files=%d total=%d", len(created.Files), created.Session.TotalBytes, fileCount, totalBytes)
	}
	if counted.listSessionFilesCalls != 0 {
		t.Fatalf("large folder create re-listed all session files %d times; want create response to reuse inserted file records", counted.listSessionFilesCalls)
	}
	if counted.listChunksBySessionCalls != 0 || counted.listChunkByFileCalls != 0 {
		t.Fatalf("large folder create listed chunks by session=%d by file=%d; want no chunk hydration before uploads start", counted.listChunksBySessionCalls, counted.listChunkByFileCalls)
	}
	if counted.upsertSessionFileCalls != 0 {
		t.Fatalf("large folder create upserted files individually %d times; want batched manifest insert", counted.upsertSessionFileCalls)
	}
	for _, file := range created.Files {
		if file.RelativePath == "" {
			t.Fatalf("created file %q lost relative path", file.FileToken)
		}
	}
	counted.listSessionFilesCalls = 0
	counted.updateSessionCalls = 0
	counted.listResourcesCalls = 0

	var lastComplete uploadSessionFileCompleteResponse
	for _, file := range files {
		uploadChunkForTest(t, router, created.Session.SessionID, file.token, 0, 0, file.payload, file.sha, "large-folder-user", "large-folder-org")
		lastComplete = completeUploadSessionFileForTest(t, router, created.Session.SessionID, file.token, "large-folder-user", "large-folder-org", http.StatusOK)
	}
	if lastComplete.Session.Status != "completed" || lastComplete.Session.BytesCommitted != totalBytes {
		t.Fatalf("completed large folder session = %+v, want completed with %d committed bytes", lastComplete.Session, totalBytes)
	}
	counted.upsertResourceCalls = 0
	counted.ownerLookupCalls = 0
	counted.ownerBatchLookupCalls = 0

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=100", nil)
	listReq.Header.Set("X-Ultra-User-Id", "large-folder-user")
	listReq.Header.Set("X-Ultra-Org-Id", "large-folder-org")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list large folder resources status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var listed resourcesResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &listed); err != nil {
		t.Fatalf("decode large folder resources: %v", err)
	}
	if listed.Count != fileCount || len(listed.Resources) != fileCount {
		t.Fatalf("listed large folder resources count=%d len=%d, want %d", listed.Count, len(listed.Resources), fileCount)
	}
	if counted.listChunkByFileCalls > fileCount+2 {
		t.Fatalf("large folder upload made %d per-file chunk scans for %d files; want aggregate session accounting", counted.listChunkByFileCalls, fileCount)
	}
	if counted.sessionTotalsCalls > 2 {
		t.Fatalf("large folder upload read aggregate session totals %d times for %d files; want stored counters on the hot path", counted.sessionTotalsCalls, fileCount)
	}
	if counted.listChunksBySessionCalls > 1 {
		t.Fatalf("large folder upload materialized session chunks %d times; want totals aggregation for byte accounting", counted.listChunksBySessionCalls)
	}
	if counted.listSessionFilesCalls > 2 {
		t.Fatalf("large folder upload listed all session files %d times after create for %d files; want direct file lookup during chunk and complete", counted.listSessionFilesCalls, fileCount)
	}
	if counted.updateSessionCalls > 2 {
		t.Fatalf("large folder upload updated session %d times for %d one-chunk files; want stored accounting to avoid per-file session writes", counted.updateSessionCalls, fileCount)
	}
	if counted.listResourcesCalls != 0 {
		t.Fatalf("large folder upload scanned the full resource catalog %d times with no configured quotas; want quota accounting to stay lazy", counted.listResourcesCalls)
	}
	if counted.upsertResourceCalls != 0 {
		t.Fatalf("large folder first list re-cataloged %d already committed resources; want migration to skip catalog rows created during upload completion", counted.upsertResourceCalls)
	}
	if counted.ownerLookupCalls != 0 {
		t.Fatalf("large folder first list checked %d already committed resources one by one; want batched owner/resource existence checks", counted.ownerLookupCalls)
	}
	if counted.ownerBatchLookupCalls > 1 {
		t.Fatalf("large folder first list made %d batched owner/resource checks for one owner; want at most one", counted.ownerBatchLookupCalls)
	}
}

func TestOrganizationByIDUsesExactLookupWhenAvailable(t *testing.T) {
	t.Parallel()

	org := domain.Organization{
		OrgID:     "bench-org",
		Name:      "Benchmark Organization",
		Status:    "active",
		CreatedAt: domain.Now(),
		UpdatedAt: domain.Now(),
		Metadata:  domain.JSONMap{"source": "test"},
	}
	mem := store.NewMemoryStore()
	store := &exactOrganizationLookupStore{MemoryStore: mem, org: org}
	found, ok, err := (ServerDeps{Store: store}).organizationByID(context.Background(), " bench-org ")
	if err != nil {
		t.Fatalf("organizationByID: %v", err)
	}
	if !ok || found.OrgID != org.OrgID {
		t.Fatalf("organizationByID = %+v found=%t, want exact org", found, ok)
	}
	if store.getCalls != 1 {
		t.Fatalf("GetOrganization calls = %d, want 1", store.getCalls)
	}
	if store.listCalls != 0 {
		t.Fatalf("ListOrganizations calls = %d, want exact lookup without full list scan", store.listCalls)
	}
}

func BenchmarkV2UploadSessionManySmallFiles(b *testing.B) {
	for _, fileCount := range []int{1000, 10000} {
		b.Run(fmt.Sprintf("%d_files", fileCount), func(b *testing.B) {
			files, body, totalBytes := largeFolderUploadFixture(b, fileCount, "bench-folder-session")
			b.ReportAllocs()
			b.SetBytes(totalBytes)
			for i := 0; i < b.N; i++ {
				uploadRoot := b.TempDir()
				mem := store.NewMemoryStore()
				router := NewRouter(ServerDeps{
					Version:    "test-version",
					Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
					Store:      mem,
					UploadRoot: uploadRoot,
				})
				created := createUploadSessionForBench(b, router, string(body), "bench-folder-user", "bench-folder-org", http.StatusCreated)
				for _, file := range files {
					uploadChunkForBench(b, router, created.Session.SessionID, file.token, 0, 0, file.payload, file.sha, "bench-folder-user", "bench-folder-org")
					completeUploadSessionFileForBench(b, router, created.Session.SessionID, file.token, "bench-folder-user", "bench-folder-org", http.StatusOK)
				}
				totalListed := 0
				for offset := 0; offset < fileCount; offset += 1000 {
					listReq := httptest.NewRequest(http.MethodGet, fmt.Sprintf("/v2/resources?limit=1000&offset=%d", offset), nil)
					listReq.Header.Set("X-Ultra-User-Id", "bench-folder-user")
					listReq.Header.Set("X-Ultra-Org-Id", "bench-folder-org")
					listRec := httptest.NewRecorder()
					router.ServeHTTP(listRec, listReq)
					if listRec.Code != http.StatusOK {
						b.Fatalf("list benchmark resources status = %d body=%s", listRec.Code, listRec.Body.String())
					}
					var listed resourcesResponse
					if err := json.Unmarshal(listRec.Body.Bytes(), &listed); err != nil {
						b.Fatalf("decode benchmark resources: %v", err)
					}
					totalListed += len(listed.Resources)
					if listed.Count != fileCount {
						b.Fatalf("benchmark listed resources count=%d, want %d", listed.Count, fileCount)
					}
				}
				if totalListed != fileCount {
					b.Fatalf("benchmark listed resources across pages=%d, want %d", totalListed, fileCount)
				}
			}
		})
	}
}

func BenchmarkV2UploadSessionManySmallFilesPostgres(b *testing.B) {
	dsn := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_TEST_DATABASE_URL"))
	if dsn == "" {
		b.Skip("ULTRA_CONTROL_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		b.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()
	if err := store.ApplyPostgresSchema(ctx, pool); err != nil {
		b.Fatalf("ApplyPostgresSchema: %v", err)
	}

	for _, fileCount := range []int{1000, 10000} {
		b.Run(fmt.Sprintf("%d_files", fileCount), func(b *testing.B) {
			files, body, totalBytes := largeFolderUploadFixture(b, fileCount, "bench-postgres-folder-session")
			b.ReportAllocs()
			b.SetBytes(totalBytes)
			for i := 0; i < b.N; i++ {
				uploadRoot := b.TempDir()
				pgStore := store.NewPostgresStore(pool)
				router := NewRouter(ServerDeps{
					Version:    "test-version",
					Runs:       runcontrol.NewService(pgStore, eventbus.NewMemoryBus()),
					Store:      pgStore,
					UploadRoot: uploadRoot,
				})
				suffix := strings.ReplaceAll(domain.NewID("pg_bench"), "-", "_")
				userID := "bench-postgres-folder-user-" + suffix
				orgID := "bench-postgres-folder-org-" + suffix
				created := createUploadSessionForBench(b, router, string(body), userID, orgID, http.StatusCreated)
				for _, file := range files {
					uploadChunkForBench(b, router, created.Session.SessionID, file.token, 0, 0, file.payload, file.sha, userID, orgID)
					completeUploadSessionFileForBench(b, router, created.Session.SessionID, file.token, userID, orgID, http.StatusOK)
				}
				totalListed := 0
				for offset := 0; offset < fileCount; offset += 1000 {
					listReq := httptest.NewRequest(http.MethodGet, fmt.Sprintf("/v2/resources?limit=1000&offset=%d", offset), nil)
					listReq.Header.Set("X-Ultra-User-Id", userID)
					listReq.Header.Set("X-Ultra-Org-Id", orgID)
					listRec := httptest.NewRecorder()
					router.ServeHTTP(listRec, listReq)
					if listRec.Code != http.StatusOK {
						b.Fatalf("list postgres benchmark resources status = %d body=%s", listRec.Code, listRec.Body.String())
					}
					var listed resourcesResponse
					if err := json.Unmarshal(listRec.Body.Bytes(), &listed); err != nil {
						b.Fatalf("decode postgres benchmark resources: %v", err)
					}
					totalListed += len(listed.Resources)
					if listed.Count != fileCount {
						b.Fatalf("postgres benchmark listed resources count=%d, want %d", listed.Count, fileCount)
					}
				}
				if totalListed != fileCount {
					b.Fatalf("postgres benchmark listed resources across pages=%d, want %d", totalListed, fileCount)
				}
			}
		})
	}
}

func TestCopyWithPooledBufferAvoidsWriterToFastPath(t *testing.T) {
	t.Parallel()

	source := &writerToProbeReader{data: []byte("microscopy tile bytes")}
	var destination bytes.Buffer

	copied, err := copyWithPooledBuffer(&destination, source)
	if err != nil {
		t.Fatalf("copyWithPooledBuffer: %v", err)
	}
	if copied != int64(len(source.data)) {
		t.Fatalf("copied = %d, want %d", copied, len(source.data))
	}
	if destination.String() != string(source.data) {
		t.Fatalf("destination = %q, want %q", destination.String(), string(source.data))
	}
	if source.writerToCalled {
		t.Fatalf("copyWithPooledBuffer used WriterTo fast path instead of the pooled buffer")
	}
}

type writerToProbeReader struct {
	data           []byte
	offset         int
	writerToCalled bool
}

func (r *writerToProbeReader) Read(p []byte) (int, error) {
	if r.offset >= len(r.data) {
		return 0, io.EOF
	}
	n := copy(p, r.data[r.offset:])
	r.offset += n
	return n, nil
}

func (r *writerToProbeReader) WriteTo(w io.Writer) (int64, error) {
	r.writerToCalled = true
	n, err := w.Write(r.data[r.offset:])
	r.offset += n
	return int64(n), err
}

func TestV2UploadSessionRejectsTamperedVerifiedChunkBeforeCommit(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	payload := []byte("science-data")
	chunkSHA := sha256.Sum256(payload)
	createBody := uploadSessionCreateBody("tamper-session-1", "tamper-file", "tamper.bin", "application/octet-stream", payload, "")

	created := createUploadSessionForTest(t, router, createBody, "tamper-user", "tamper-org", http.StatusCreated)
	uploadChunkForTest(t, router, created.Session.SessionID, "tamper-file", 0, 0, payload, hex.EncodeToString(chunkSHA[:]), "tamper-user", "tamper-org")
	chunkPath := uploadSessionChunkPath(uploadRoot, created.Session.SessionID, "tamper-file", 0)
	if err := os.WriteFile(chunkPath, []byte("changed-data"), 0o644); err != nil {
		t.Fatalf("tamper staged chunk: %v", err)
	}

	rec := completeUploadSessionFileRaw(t, router, created.Session.SessionID, "tamper-file", "tamper-user", "tamper-org")
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("tampered complete status = %d body=%s, want 400", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "chunk checksum mismatch") {
		t.Fatalf("tampered complete body = %s, want chunk checksum mismatch", rec.Body.String())
	}
	resources, err := listUploadResources(uploadRoot)
	if err != nil {
		t.Fatalf("list upload resources: %v", err)
	}
	if len(resources) != 0 {
		t.Fatalf("resources after tampered commit = %+v, want none", resources)
	}
}

func TestV2UploadSessionRejectsConflictingVerifiedChunkReplay(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	original := []byte("chunk-v1")
	conflicting := []byte("chunk-v2")
	originalSHA := sha256.Sum256(original)
	conflictingSHA := sha256.Sum256(conflicting)
	createBody := uploadSessionCreateBody("verified-conflict-session-1", "verified-conflict-file", "field-scan.bin", "application/octet-stream", original, "")

	created := createUploadSessionForTest(t, router, createBody, "verified-conflict-user", "verified-conflict-org", http.StatusCreated)
	uploadChunkForTest(t, router, created.Session.SessionID, "verified-conflict-file", 0, 0, original, hex.EncodeToString(originalSHA[:]), "verified-conflict-user", "verified-conflict-org")

	conflictReq := httptest.NewRequest(
		http.MethodPut,
		"/v2/upload-sessions/"+created.Session.SessionID+"/files/verified-conflict-file/chunks/0",
		bytes.NewReader(conflicting),
	)
	conflictReq.Header.Set("X-Ultra-User-Id", "verified-conflict-user")
	conflictReq.Header.Set("X-Ultra-Org-Id", "verified-conflict-org")
	conflictReq.Header.Set("X-Upload-Offset", "0")
	conflictReq.Header.Set("X-Upload-Chunk-Sha256", hex.EncodeToString(conflictingSHA[:]))
	conflictRec := httptest.NewRecorder()
	router.ServeHTTP(conflictRec, conflictReq)
	if conflictRec.Code != http.StatusConflict {
		t.Fatalf("conflicting verified chunk replay status = %d body=%s, want 409", conflictRec.Code, conflictRec.Body.String())
	}

	completed := completeUploadSessionFileForTest(t, router, created.Session.SessionID, "verified-conflict-file", "verified-conflict-user", "verified-conflict-org", http.StatusOK)
	committedPath := filepath.Join(uploadRoot, completed.Resource.FileID+"__field-scan.bin")
	committedBytes, err := os.ReadFile(committedPath)
	if err != nil {
		t.Fatalf("read committed upload: %v", err)
	}
	if !bytes.Equal(committedBytes, original) {
		t.Fatalf("committed bytes = %q, want original verified chunk %q", committedBytes, original)
	}
	if completed.Resource.SHA256 != hex.EncodeToString(originalSHA[:]) {
		t.Fatalf("completed resource sha = %q, want original verified digest", completed.Resource.SHA256)
	}
}

type countingUploadStore struct {
	*store.MemoryStore
	listSessionFilesCalls    int
	listChunkByFileCalls     int
	listChunksBySessionCalls int
	sessionTotalsCalls       int
	updateSessionCalls       int
	listResourcesCalls       int
	upsertSessionFileCalls   int
	upsertResourceCalls      int
	ownerLookupCalls         int
	ownerBatchLookupCalls    int
	chunksBySession          map[string][]domain.UploadChunkRecord
}

type exactOrganizationLookupStore struct {
	*store.MemoryStore
	org       domain.Organization
	getCalls  int
	listCalls int
}

func (s *exactOrganizationLookupStore) ListOrganizations(context.Context, int, string) ([]domain.Organization, error) {
	s.listCalls++
	return []domain.Organization{s.org}, nil
}

func (s *exactOrganizationLookupStore) GetOrganization(_ context.Context, orgID string) (domain.Organization, bool, error) {
	s.getCalls++
	if strings.TrimSpace(orgID) != s.org.OrgID {
		return domain.Organization{}, false, nil
	}
	return s.org, true, nil
}

type uploadFileFixture struct {
	token   string
	name    string
	path    string
	payload []byte
	sha     string
}

type testHelper interface {
	Helper()
	Fatalf(string, ...any)
}

func largeFolderUploadFixture(tb testHelper, fileCount int, idempotencyKey string) ([]uploadFileFixture, []byte, int64) {
	tb.Helper()
	files := make([]uploadFileFixture, 0, fileCount)
	createReq := createUploadSessionRequest{
		IdempotencyKey:     idempotencyKey,
		BrowserFingerprint: "field-folder-2026-06",
		ProjectID:          "frontier-field-project",
		Files:              make([]createUploadSessionFileRequest, 0, fileCount),
	}
	var totalBytes int64
	for i := 0; i < fileCount; i++ {
		payload := []byte(fmt.Sprintf("small microscopy tile %03d", i))
		sum := sha256.Sum256(payload)
		name := fmt.Sprintf("tile_%03d.ome.tiff", i)
		token := fmt.Sprintf("tile-%03d", i)
		relativePath := fmt.Sprintf("field-run-a/plate-%02d/%s", i/16, name)
		files = append(files, uploadFileFixture{
			token:   token,
			name:    name,
			path:    relativePath,
			payload: payload,
			sha:     hex.EncodeToString(sum[:]),
		})
		totalBytes += int64(len(payload))
		createReq.Files = append(createReq.Files, createUploadSessionFileRequest{
			FileToken:      token,
			OriginalName:   name,
			RelativePath:   relativePath,
			ContentType:    "image/tiff",
			SizeBytes:      int64(len(payload)),
			DeclaredSHA256: hex.EncodeToString(sum[:]),
		})
	}
	createReq.TotalBytes = totalBytes
	body, err := json.Marshal(createReq)
	if err != nil {
		tb.Fatalf("marshal large folder create request: %v", err)
	}
	return files, body, totalBytes
}

func (s *countingUploadStore) UpsertUploadChunk(ctx context.Context, input domain.UpsertUploadChunkInput) (domain.UploadChunkRecord, error) {
	chunk, err := s.MemoryStore.UpsertUploadChunk(ctx, input)
	if err != nil {
		return domain.UploadChunkRecord{}, err
	}
	existing := s.chunksBySession[chunk.SessionID]
	replaced := false
	for i := range existing {
		if existing[i].FileToken == chunk.FileToken && existing[i].ChunkIndex == chunk.ChunkIndex {
			existing[i] = chunk
			replaced = true
			break
		}
	}
	if !replaced {
		existing = append(existing, chunk)
	}
	s.chunksBySession[chunk.SessionID] = existing
	return chunk, nil
}

func (s *countingUploadStore) UpdateUploadSession(ctx context.Context, input domain.UpdateUploadSessionInput) (domain.UploadSessionRecord, error) {
	s.updateSessionCalls++
	return s.MemoryStore.UpdateUploadSession(ctx, input)
}

func (s *countingUploadStore) UpsertResource(ctx context.Context, input domain.UpsertResourceInput) (domain.ResourceRecord, error) {
	s.upsertResourceCalls++
	return s.MemoryStore.UpsertResource(ctx, input)
}

func (s *countingUploadStore) GetResourceForOwner(ctx context.Context, resourceID string, userID string, orgID string) (domain.ResourceRecord, error) {
	s.ownerLookupCalls++
	return s.MemoryStore.GetResourceForOwner(ctx, resourceID, userID, orgID)
}

func (s *countingUploadStore) ListResourceIDsForOwner(ctx context.Context, userID string, orgID string, resourceIDs []string) (map[string]bool, error) {
	s.ownerBatchLookupCalls++
	return s.MemoryStore.ListResourceIDsForOwner(ctx, userID, orgID, resourceIDs)
}

func (s *countingUploadStore) UpsertUploadSessionFile(ctx context.Context, input domain.UpsertUploadSessionFileInput) (domain.UploadSessionFileRecord, error) {
	s.upsertSessionFileCalls++
	return s.MemoryStore.UpsertUploadSessionFile(ctx, input)
}

func (s *countingUploadStore) ListResources(ctx context.Context, limit int, offset int) ([]domain.ResourceRecord, error) {
	s.listResourcesCalls++
	return s.MemoryStore.ListResources(ctx, limit, offset)
}

func (s *countingUploadStore) ListUploadSessionFiles(ctx context.Context, sessionID string) ([]domain.UploadSessionFileRecord, error) {
	s.listSessionFilesCalls++
	return s.MemoryStore.ListUploadSessionFiles(ctx, sessionID)
}

func (s *countingUploadStore) ListUploadChunks(ctx context.Context, sessionID string, fileToken string) ([]domain.UploadChunkRecord, error) {
	s.listChunkByFileCalls++
	return s.MemoryStore.ListUploadChunks(ctx, sessionID, fileToken)
}

func (s *countingUploadStore) ListUploadSessionChunks(ctx context.Context, sessionID string) ([]domain.UploadChunkRecord, error) {
	_ = ctx
	s.listChunksBySessionCalls++
	chunks := append([]domain.UploadChunkRecord(nil), s.chunksBySession[strings.TrimSpace(sessionID)]...)
	sort.Slice(chunks, func(i, j int) bool {
		if chunks[i].FileToken == chunks[j].FileToken {
			return chunks[i].ChunkIndex < chunks[j].ChunkIndex
		}
		return chunks[i].FileToken < chunks[j].FileToken
	})
	return chunks, nil
}

func (s *countingUploadStore) GetUploadSessionTotals(ctx context.Context, sessionID string) (domain.UploadSessionTotals, error) {
	s.sessionTotalsCalls++
	return s.MemoryStore.GetUploadSessionTotals(ctx, sessionID)
}

type overLimitDataAgentQueryStore struct {
	*store.MemoryStore
	totalCount    int
	lastListInput domain.ResourceListInput
}

func (s *overLimitDataAgentQueryStore) ListResourcesForUser(ctx context.Context, input domain.ResourceListInput) (domain.ResourceListPage, error) {
	s.lastListInput = input
	if s.totalCount <= 0 {
		return s.MemoryStore.ListResourcesForUser(ctx, input)
	}
	return domain.ResourceListPage{
		TotalCount: s.totalCount,
		Limit:      input.Limit,
		Offset:     input.Offset,
	}, nil
}

func uploadSessionCreateBody(idempotencyKey string, fileToken string, originalName string, contentType string, payload []byte, declaredSHA string) string {
	declaredField := ""
	if declaredSHA != "" {
		declaredField = fmt.Sprintf(`,"declared_sha256":%q`, declaredSHA)
	}
	return fmt.Sprintf(`{
		"idempotency_key":%q,
		"total_bytes":%d,
		"files":[{
			"file_token":%q,
			"original_name":%q,
			"content_type":%q,
			"size_bytes":%d%s
		}]
	}`, idempotencyKey, len(payload), fileToken, originalName, contentType, len(payload), declaredField)
}

func createUploadSessionForTest(t *testing.T, router http.Handler, body string, userID string, orgID string, wantStatus int) uploadSessionResponse {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-User-Id", userID)
	req.Header.Set("X-Ultra-Org-Id", orgID)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != wantStatus {
		t.Fatalf("create upload session status = %d body=%s, want %d", rec.Code, rec.Body.String(), wantStatus)
	}
	var response uploadSessionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode upload session response: %v", err)
	}
	return response
}

func createUploadSessionForBench(b *testing.B, router http.Handler, body string, userID string, orgID string, wantStatus int) uploadSessionResponse {
	b.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v2/upload-sessions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-User-Id", userID)
	req.Header.Set("X-Ultra-Org-Id", orgID)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != wantStatus {
		b.Fatalf("create upload session status = %d body=%s, want %d", rec.Code, rec.Body.String(), wantStatus)
	}
	var response uploadSessionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		b.Fatalf("decode upload session response: %v", err)
	}
	return response
}

func uploadChunkForTest(t *testing.T, router http.Handler, sessionID string, fileToken string, chunkIndex int, offset int64, payload []byte, chunkSHA string, userID string, orgID string) uploadChunkResponse {
	t.Helper()
	req := httptest.NewRequest(
		http.MethodPut,
		fmt.Sprintf("/v2/upload-sessions/%s/files/%s/chunks/%d", sessionID, fileToken, chunkIndex),
		bytes.NewReader(payload),
	)
	req.Header.Set("X-Ultra-User-Id", userID)
	req.Header.Set("X-Ultra-Org-Id", orgID)
	req.Header.Set("X-Upload-Offset", strconv.FormatInt(offset, 10))
	req.Header.Set("X-Upload-Chunk-Sha256", chunkSHA)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("upload chunk status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var response uploadChunkResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode upload chunk response: %v", err)
	}
	return response
}

func uploadChunkForBench(b *testing.B, router http.Handler, sessionID string, fileToken string, chunkIndex int, offset int64, payload []byte, chunkSHA string, userID string, orgID string) uploadChunkResponse {
	b.Helper()
	req := httptest.NewRequest(
		http.MethodPut,
		fmt.Sprintf("/v2/upload-sessions/%s/files/%s/chunks/%d", sessionID, fileToken, chunkIndex),
		bytes.NewReader(payload),
	)
	req.Header.Set("X-Ultra-User-Id", userID)
	req.Header.Set("X-Ultra-Org-Id", orgID)
	req.Header.Set("X-Upload-Offset", strconv.FormatInt(offset, 10))
	req.Header.Set("X-Upload-Chunk-Sha256", chunkSHA)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		b.Fatalf("upload chunk status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var response uploadChunkResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		b.Fatalf("decode upload chunk response: %v", err)
	}
	return response
}

func completeUploadSessionFileForTest(t *testing.T, router http.Handler, sessionID string, fileToken string, userID string, orgID string, wantStatus int) uploadSessionFileCompleteResponse {
	t.Helper()
	rec := completeUploadSessionFileRaw(t, router, sessionID, fileToken, userID, orgID)
	if rec.Code != wantStatus {
		t.Fatalf("complete upload session file status = %d body=%s, want %d", rec.Code, rec.Body.String(), wantStatus)
	}
	var response uploadSessionFileCompleteResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode upload session complete response: %v", err)
	}
	return response
}

func completeUploadSessionFileForBench(b *testing.B, router http.Handler, sessionID string, fileToken string, userID string, orgID string, wantStatus int) uploadSessionFileCompleteResponse {
	b.Helper()
	req := httptest.NewRequest(http.MethodPost, fmt.Sprintf("/v2/upload-sessions/%s/files/%s/complete", sessionID, fileToken), nil)
	req.Header.Set("X-Ultra-User-Id", userID)
	req.Header.Set("X-Ultra-Org-Id", orgID)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != wantStatus {
		b.Fatalf("complete upload session file status = %d body=%s, want %d", rec.Code, rec.Body.String(), wantStatus)
	}
	var response uploadSessionFileCompleteResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		b.Fatalf("decode upload session complete response: %v", err)
	}
	return response
}

func completeUploadSessionFileRaw(t *testing.T, router http.Handler, sessionID string, fileToken string, userID string, orgID string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, fmt.Sprintf("/v2/upload-sessions/%s/files/%s/complete", sessionID, fileToken), nil)
	req.Header.Set("X-Ultra-User-Id", userID)
	req.Header.Set("X-Ultra-Org-Id", orgID)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

func TestUploadCatalogMigrationSmokeExistingRoot(t *testing.T) {
	root := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_UPLOAD_ROOT_SMOKE"))
	if root == "" {
		t.Skip("set ULTRA_CONTROL_UPLOAD_ROOT_SMOKE to run migration smoke against an existing upload root")
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		t.Fatalf("resolve smoke upload root: %v", err)
	}
	before, err := listUploadResources(absRoot)
	if err != nil {
		t.Fatalf("list existing upload root before migration: %v", err)
	}
	mem := store.NewMemoryStore()
	deps := ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: absRoot,
	}
	if err := deps.ensureUploadCatalogMigrated(context.Background(), absRoot); err != nil {
		t.Fatalf("migrate existing upload root: %v", err)
	}
	records, err := mem.ListResources(context.Background(), len(before)+10, 0)
	if err != nil {
		t.Fatalf("list migrated catalog resources: %v", err)
	}
	if len(records) != len(before) {
		t.Fatalf("migrated catalog rows = %d, want %d existing upload resources from %s", len(records), len(before), absRoot)
	}
}

func TestV2ResourceCollectionsCreateAndBulkAddResources(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_nph_a",
			OriginalName: "nph-a.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    128,
			OwnerUserID:  "nph-user",
			OwnerOrgID:   "nph-org",
			ProjectID:    "nph-study",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_nph_b",
			OriginalName: "nph-b.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    256,
			OwnerUserID:  "nph-user",
			OwnerOrgID:   "nph-org",
			ProjectID:    "nph-study",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	createReq := httptest.NewRequest(http.MethodPost, "/v2/resource-collections", strings.NewReader(`{
		"name":"NPH NIfTI cohort",
		"description":"NIfTI files labeled NPH for header inspection",
		"collection_type":"folder",
		"project_id":"nph-study",
		"metadata":{"label":"NPH"}
	}`))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("X-Ultra-User-Id", "nph-user")
	createReq.Header.Set("X-Ultra-Org-Id", "nph-org")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusCreated {
		t.Fatalf("create resource collection status = %d body=%s, want 201", createRec.Code, createRec.Body.String())
	}
	var created struct {
		Collection domain.ResourceCollectionRecord `json:"collection"`
	}
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created collection: %v", err)
	}
	if created.Collection.CollectionID == "" || created.Collection.Name != "NPH NIfTI cohort" || created.Collection.CollectionType != "folder" {
		t.Fatalf("created collection = %+v, want folder with id", created.Collection)
	}

	addReq := httptest.NewRequest(
		http.MethodPost,
		"/v2/resource-collections/"+created.Collection.CollectionID+"/resources",
		strings.NewReader(`{"resource_ids":["file_nph_a","file_nph_b"]}`),
	)
	addReq.Header.Set("Content-Type", "application/json")
	addReq.Header.Set("X-Ultra-User-Id", "nph-user")
	addReq.Header.Set("X-Ultra-Org-Id", "nph-org")
	addRec := httptest.NewRecorder()
	router.ServeHTTP(addRec, addReq)
	if addRec.Code != http.StatusOK {
		t.Fatalf("add collection resources status = %d body=%s, want 200", addRec.Code, addRec.Body.String())
	}
	var added struct {
		Collection domain.ResourceCollectionRecord `json:"collection"`
		AddedCount int                             `json:"added_count"`
	}
	if err := json.Unmarshal(addRec.Body.Bytes(), &added); err != nil {
		t.Fatalf("decode add resources response: %v", err)
	}
	if added.AddedCount != 2 || added.Collection.ResourceCount != 2 {
		t.Fatalf("added response = %+v, want two members and updated count", added)
	}

	membersReq := httptest.NewRequest(http.MethodGet, "/v2/resource-collections/"+created.Collection.CollectionID+"/resources?limit=10", nil)
	membersReq.Header.Set("X-Ultra-User-Id", "nph-user")
	membersReq.Header.Set("X-Ultra-Org-Id", "nph-org")
	membersRec := httptest.NewRecorder()
	router.ServeHTTP(membersRec, membersReq)
	if membersRec.Code != http.StatusOK {
		t.Fatalf("list collection resources status = %d body=%s, want 200", membersRec.Code, membersRec.Body.String())
	}
	var members resourcesResponse
	if err := json.Unmarshal(membersRec.Body.Bytes(), &members); err != nil {
		t.Fatalf("decode collection resources: %v", err)
	}
	if members.Count != 2 || len(members.Resources) != 2 || members.Resources[0].FileID != "file_nph_a" || members.Resources[1].FileID != "file_nph_b" {
		t.Fatalf("collection resources = %+v, want two NPH files in insertion order", members)
	}

	filteredMembersReq := httptest.NewRequest(http.MethodGet, "/v2/resource-collections/"+created.Collection.CollectionID+"/resources?q=nph-a&kind=image&source=upload&limit=10", nil)
	filteredMembersReq.Header.Set("X-Ultra-User-Id", "nph-user")
	filteredMembersReq.Header.Set("X-Ultra-Org-Id", "nph-org")
	filteredMembersRec := httptest.NewRecorder()
	router.ServeHTTP(filteredMembersRec, filteredMembersReq)
	if filteredMembersRec.Code != http.StatusOK {
		t.Fatalf("filtered collection resources status = %d body=%s, want 200", filteredMembersRec.Code, filteredMembersRec.Body.String())
	}
	var filteredMembers resourcesResponse
	if err := json.Unmarshal(filteredMembersRec.Body.Bytes(), &filteredMembers); err != nil {
		t.Fatalf("decode filtered collection resources: %v", err)
	}
	if filteredMembers.Count != 1 || len(filteredMembers.Resources) != 1 || filteredMembers.Resources[0].FileID != "file_nph_a" {
		t.Fatalf("filtered collection resources = %+v, want only file_nph_a", filteredMembers)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resource-collections?collection_type=folder&limit=10", nil)
	listReq.Header.Set("X-Ultra-User-Id", "nph-user")
	listReq.Header.Set("X-Ultra-Org-Id", "nph-org")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list resource collections status = %d body=%s, want 200", listRec.Code, listRec.Body.String())
	}
	var listed struct {
		Count       int                               `json:"count"`
		Collections []domain.ResourceCollectionRecord `json:"collections"`
	}
	if err := json.Unmarshal(listRec.Body.Bytes(), &listed); err != nil {
		t.Fatalf("decode collection list: %v", err)
	}
	if listed.Count != 1 || len(listed.Collections) != 1 || listed.Collections[0].ResourceCount != 2 {
		t.Fatalf("collection list = %+v, want one folder with resource count", listed)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/file_nph_a/events", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "nph-user")
	eventsReq.Header.Set("X-Ultra-Org-Id", "nph-org")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("resource events status = %d body=%s, want 200", eventsRec.Code, eventsRec.Body.String())
	}
	var events resourceEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode resource events: %v", err)
	}
	foundCollectionEvent := false
	for _, event := range events.Events {
		if event.EventType == "resource.collection_added" && event.Metadata["collection_id"] == created.Collection.CollectionID {
			foundCollectionEvent = true
			break
		}
	}
	if !foundCollectionEvent {
		t.Fatalf("resource events = %+v, want collection membership audit event", events.Events)
	}
}

func TestV2ResourcesFileManagerRenameAndRemoveFromFolder(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_manager_a",
			OriginalName: "nph-a.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    128,
			OwnerUserID:  "file-manager-user",
			OwnerOrgID:   "file-manager-org",
			ProjectID:    "nph-study",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_manager_b",
			OriginalName: "nph-b.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "image",
			SourceType:   "upload",
			SizeBytes:    256,
			OwnerUserID:  "file-manager-user",
			OwnerOrgID:   "file-manager-org",
			ProjectID:    "nph-study",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	renameFileReq := httptest.NewRequest(http.MethodPatch, "/v2/resources/file_manager_a", strings.NewReader(`{"original_name":"nph-a-reviewed.nii.gz"}`))
	renameFileReq.Header.Set("Content-Type", "application/json")
	renameFileReq.Header.Set("X-Ultra-User-Id", "file-manager-user")
	renameFileReq.Header.Set("X-Ultra-Org-Id", "file-manager-org")
	renameFileRec := httptest.NewRecorder()
	router.ServeHTTP(renameFileRec, renameFileReq)
	if renameFileRec.Code != http.StatusOK {
		t.Fatalf("rename resource status = %d body=%s, want 200", renameFileRec.Code, renameFileRec.Body.String())
	}
	var renamedFile resourceResponse
	if err := json.Unmarshal(renameFileRec.Body.Bytes(), &renamedFile); err != nil {
		t.Fatalf("decode renamed resource: %v", err)
	}
	if renamedFile.Resource.OriginalName != "nph-a-reviewed.nii.gz" {
		t.Fatalf("renamed resource = %+v, want updated original_name", renamedFile.Resource)
	}

	collection, err := mem.CreateResourceCollection(context.Background(), domain.CreateResourceCollectionInput{
		CollectionID:   "collection_file_manager",
		OwnerUserID:    "file-manager-user",
		OwnerOrgID:     "file-manager-org",
		ProjectID:      "nph-study",
		Name:           "NPH review",
		CollectionType: "folder",
		Status:         "active",
		CreatedAt:      now,
		UpdatedAt:      now,
	})
	if err != nil {
		t.Fatalf("CreateResourceCollection: %v", err)
	}
	if _, err := mem.AddResourcesToCollection(context.Background(), domain.AddResourcesToCollectionInput{
		CollectionID:  collection.CollectionID,
		OwnerUserID:   "file-manager-user",
		OwnerOrgID:    "file-manager-org",
		ResourceIDs:   []string{"file_manager_a", "file_manager_b"},
		AddedByUserID: "file-manager-user",
		AddedAt:       now.Add(2 * time.Second),
	}); err != nil {
		t.Fatalf("AddResourcesToCollection: %v", err)
	}

	renameFolderReq := httptest.NewRequest(http.MethodPatch, "/v2/resource-collections/"+collection.CollectionID, strings.NewReader(`{"name":"NPH review renamed"}`))
	renameFolderReq.Header.Set("Content-Type", "application/json")
	renameFolderReq.Header.Set("X-Ultra-User-Id", "file-manager-user")
	renameFolderReq.Header.Set("X-Ultra-Org-Id", "file-manager-org")
	renameFolderRec := httptest.NewRecorder()
	router.ServeHTTP(renameFolderRec, renameFolderReq)
	if renameFolderRec.Code != http.StatusOK {
		t.Fatalf("rename folder status = %d body=%s, want 200", renameFolderRec.Code, renameFolderRec.Body.String())
	}
	var renamedFolder resourceCollectionResponse
	if err := json.Unmarshal(renameFolderRec.Body.Bytes(), &renamedFolder); err != nil {
		t.Fatalf("decode renamed folder: %v", err)
	}
	if renamedFolder.Collection.Name != "NPH review renamed" || renamedFolder.Collection.ResourceCount != 2 {
		t.Fatalf("renamed folder = %+v, want renamed folder preserving resource count", renamedFolder.Collection)
	}

	removeReq := httptest.NewRequest(http.MethodDelete, "/v2/resource-collections/"+collection.CollectionID+"/resources/file_manager_a", nil)
	removeReq.Header.Set("X-Ultra-User-Id", "file-manager-user")
	removeReq.Header.Set("X-Ultra-Org-Id", "file-manager-org")
	removeRec := httptest.NewRecorder()
	router.ServeHTTP(removeRec, removeReq)
	if removeRec.Code != http.StatusOK {
		t.Fatalf("remove resource from folder status = %d body=%s, want 200", removeRec.Code, removeRec.Body.String())
	}
	var removed removeResourcesFromCollectionResponse
	if err := json.Unmarshal(removeRec.Body.Bytes(), &removed); err != nil {
		t.Fatalf("decode removed membership: %v", err)
	}
	if removed.RemovedCount != 1 || removed.Collection.ResourceCount != 1 {
		t.Fatalf("removed membership = %+v, want one removed and one remaining", removed)
	}

	membersReq := httptest.NewRequest(http.MethodGet, "/v2/resource-collections/"+collection.CollectionID+"/resources?limit=10", nil)
	membersReq.Header.Set("X-Ultra-User-Id", "file-manager-user")
	membersReq.Header.Set("X-Ultra-Org-Id", "file-manager-org")
	membersRec := httptest.NewRecorder()
	router.ServeHTTP(membersRec, membersReq)
	if membersRec.Code != http.StatusOK {
		t.Fatalf("list folder resources status = %d body=%s, want 200", membersRec.Code, membersRec.Body.String())
	}
	var members resourcesResponse
	if err := json.Unmarshal(membersRec.Body.Bytes(), &members); err != nil {
		t.Fatalf("decode folder resources: %v", err)
	}
	if members.Count != 1 || len(members.Resources) != 1 || members.Resources[0].FileID != "file_manager_b" {
		t.Fatalf("folder resources = %+v, want only file_manager_b after removing file_manager_a", members)
	}

	libraryReq := httptest.NewRequest(http.MethodGet, "/v2/resources?q=nph-a-reviewed&limit=10", nil)
	libraryReq.Header.Set("X-Ultra-User-Id", "file-manager-user")
	libraryReq.Header.Set("X-Ultra-Org-Id", "file-manager-org")
	libraryRec := httptest.NewRecorder()
	router.ServeHTTP(libraryRec, libraryReq)
	if libraryRec.Code != http.StatusOK {
		t.Fatalf("library resources after folder removal status = %d body=%s, want 200", libraryRec.Code, libraryRec.Body.String())
	}
	var library resourcesResponse
	if err := json.Unmarshal(libraryRec.Body.Bytes(), &library); err != nil {
		t.Fatalf("decode library resources after folder removal: %v", err)
	}
	if library.Count != 1 || len(library.Resources) != 1 || library.Resources[0].FileID != "file_manager_a" {
		t.Fatalf("library resources after folder removal = %+v, want removed resource still present in all resources", library)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/file_manager_a/events", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "file-manager-user")
	eventsReq.Header.Set("X-Ultra-Org-Id", "file-manager-org")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("resource events status = %d body=%s, want 200", eventsRec.Code, eventsRec.Body.String())
	}
	var events resourceEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode resource events: %v", err)
	}
	seenRename := false
	seenFolderRename := false
	seenRemoved := false
	for _, event := range events.Events {
		switch event.EventType {
		case "resource.renamed":
			seenRename = event.Metadata["previous_name"] == "nph-a.nii.gz" && event.Metadata["name"] == "nph-a-reviewed.nii.gz"
		case "resource.collection_renamed":
			seenFolderRename = event.Metadata["collection_id"] == collection.CollectionID && event.Metadata["collection_name"] == "NPH review renamed"
		case "resource.collection_removed":
			seenRemoved = event.Metadata["collection_id"] == collection.CollectionID
		}
	}
	if !seenRename || !seenFolderRename || !seenRemoved {
		t.Fatalf("resource events = %+v, want resource rename, folder rename, and folder removal events", events.Events)
	}
}

func TestV2ResourceCollectionDeleteAndRestoreLifecycle(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_folder_lifecycle_a",
			OriginalName: "folder-lifecycle-a.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    128,
			OwnerUserID:  "folder-user",
			OwnerOrgID:   "folder-org",
			ProjectID:    "folder-study",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_folder_lifecycle_b",
			OriginalName: "folder-lifecycle-b.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    256,
			OwnerUserID:  "folder-user",
			OwnerOrgID:   "folder-org",
			ProjectID:    "folder-study",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}
	collection, err := mem.CreateResourceCollection(context.Background(), domain.CreateResourceCollectionInput{
		CollectionID:   "collection_folder_lifecycle",
		OwnerUserID:    "folder-user",
		OwnerOrgID:     "folder-org",
		ProjectID:      "folder-study",
		Name:           "Recoverable NPH folder",
		CollectionType: "folder",
		Status:         "active",
		CreatedAt:      now,
		UpdatedAt:      now,
	})
	if err != nil {
		t.Fatalf("CreateResourceCollection: %v", err)
	}
	if _, err := mem.AddResourcesToCollection(context.Background(), domain.AddResourcesToCollectionInput{
		CollectionID:  collection.CollectionID,
		OwnerUserID:   "folder-user",
		OwnerOrgID:    "folder-org",
		ResourceIDs:   []string{"file_folder_lifecycle_a", "file_folder_lifecycle_b"},
		AddedByUserID: "folder-user",
		AddedAt:       now.Add(2 * time.Second),
	}); err != nil {
		t.Fatalf("AddResourcesToCollection: %v", err)
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, "/v2/resource-collections/"+collection.CollectionID, nil)
	deleteReq.Header.Set("X-Ultra-User-Id", "folder-user")
	deleteReq.Header.Set("X-Ultra-Org-Id", "folder-org")
	deleteRec := httptest.NewRecorder()
	router.ServeHTTP(deleteRec, deleteReq)
	if deleteRec.Code != http.StatusOK {
		t.Fatalf("delete resource collection status = %d body=%s, want 200", deleteRec.Code, deleteRec.Body.String())
	}
	var deleted resourceCollectionResponse
	if err := json.Unmarshal(deleteRec.Body.Bytes(), &deleted); err != nil {
		t.Fatalf("decode deleted collection: %v", err)
	}
	if deleted.Collection.Status != "deleted" || deleted.Collection.ResourceCount != 2 {
		t.Fatalf("deleted collection = %+v, want deleted folder preserving member count", deleted.Collection)
	}

	activeListReq := httptest.NewRequest(http.MethodGet, "/v2/resource-collections?collection_type=folder&limit=10", nil)
	activeListReq.Header.Set("X-Ultra-User-Id", "folder-user")
	activeListReq.Header.Set("X-Ultra-Org-Id", "folder-org")
	activeListRec := httptest.NewRecorder()
	router.ServeHTTP(activeListRec, activeListReq)
	if activeListRec.Code != http.StatusOK {
		t.Fatalf("active collection list status = %d body=%s, want 200", activeListRec.Code, activeListRec.Body.String())
	}
	var activeListed resourceCollectionsResponse
	if err := json.Unmarshal(activeListRec.Body.Bytes(), &activeListed); err != nil {
		t.Fatalf("decode active collection list: %v", err)
	}
	if activeListed.Count != 0 {
		t.Fatalf("active collection list = %+v, want deleted folder hidden", activeListed.Collections)
	}

	deletedListReq := httptest.NewRequest(http.MethodGet, "/v2/resource-collections?collection_type=folder&status=deleted&limit=10", nil)
	deletedListReq.Header.Set("X-Ultra-User-Id", "folder-user")
	deletedListReq.Header.Set("X-Ultra-Org-Id", "folder-org")
	deletedListRec := httptest.NewRecorder()
	router.ServeHTTP(deletedListRec, deletedListReq)
	if deletedListRec.Code != http.StatusOK {
		t.Fatalf("deleted collection list status = %d body=%s, want 200", deletedListRec.Code, deletedListRec.Body.String())
	}
	var deletedListed resourceCollectionsResponse
	if err := json.Unmarshal(deletedListRec.Body.Bytes(), &deletedListed); err != nil {
		t.Fatalf("decode deleted collection list: %v", err)
	}
	if deletedListed.Count != 1 || deletedListed.Collections[0].Status != "deleted" || deletedListed.Collections[0].ResourceCount != 2 {
		t.Fatalf("deleted collection list = %+v, want recoverable deleted folder", deletedListed.Collections)
	}

	membersWhileDeletedReq := httptest.NewRequest(http.MethodGet, "/v2/resource-collections/"+collection.CollectionID+"/resources?limit=10", nil)
	membersWhileDeletedReq.Header.Set("X-Ultra-User-Id", "folder-user")
	membersWhileDeletedReq.Header.Set("X-Ultra-Org-Id", "folder-org")
	membersWhileDeletedRec := httptest.NewRecorder()
	router.ServeHTTP(membersWhileDeletedRec, membersWhileDeletedReq)
	if membersWhileDeletedRec.Code != http.StatusNotFound {
		t.Fatalf("deleted collection members status = %d body=%s, want 404", membersWhileDeletedRec.Code, membersWhileDeletedRec.Body.String())
	}

	restoreReq := httptest.NewRequest(http.MethodPost, "/v2/resource-collections/"+collection.CollectionID+"/restore", nil)
	restoreReq.Header.Set("X-Ultra-User-Id", "folder-user")
	restoreReq.Header.Set("X-Ultra-Org-Id", "folder-org")
	restoreRec := httptest.NewRecorder()
	router.ServeHTTP(restoreRec, restoreReq)
	if restoreRec.Code != http.StatusOK {
		t.Fatalf("restore resource collection status = %d body=%s, want 200", restoreRec.Code, restoreRec.Body.String())
	}
	var restored resourceCollectionResponse
	if err := json.Unmarshal(restoreRec.Body.Bytes(), &restored); err != nil {
		t.Fatalf("decode restored collection: %v", err)
	}
	if restored.Collection.Status != "active" || restored.Collection.ResourceCount != 2 {
		t.Fatalf("restored collection = %+v, want active folder preserving member count", restored.Collection)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/file_folder_lifecycle_a/events", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "folder-user")
	eventsReq.Header.Set("X-Ultra-Org-Id", "folder-org")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("resource events status = %d body=%s, want 200", eventsRec.Code, eventsRec.Body.String())
	}
	var events resourceEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode resource events: %v", err)
	}
	seenDelete := false
	seenRestore := false
	for _, event := range events.Events {
		if event.Metadata["collection_id"] != collection.CollectionID {
			continue
		}
		switch event.EventType {
		case "resource.collection_deleted":
			seenDelete = true
		case "resource.collection_restored":
			seenRestore = true
		}
	}
	if !seenDelete || !seenRestore {
		t.Fatalf("resource events = %+v, want folder delete and restore audit events", events.Events)
	}
}

func TestV2DatasetSnapshotsCreateFromFolder(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_dataset_a",
			OriginalName: "dataset-a.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    128,
			SHA256:       "sha-dataset-a",
			OwnerUserID:  "dataset-user",
			OwnerOrgID:   "dataset-org",
			ProjectID:    "dataset-study",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_dataset_b",
			OriginalName: "dataset-b.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    256,
			SHA256:       "sha-dataset-b",
			OwnerUserID:  "dataset-user",
			OwnerOrgID:   "dataset-org",
			ProjectID:    "dataset-study",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}
	collection, err := mem.CreateResourceCollection(context.Background(), domain.CreateResourceCollectionInput{
		OwnerUserID:    "dataset-user",
		OwnerOrgID:     "dataset-org",
		ProjectID:      "dataset-study",
		Name:           "NPH dataset source folder",
		CollectionType: "folder",
		Status:         "active",
		CreatedAt:      now,
		UpdatedAt:      now,
	})
	if err != nil {
		t.Fatalf("CreateResourceCollection: %v", err)
	}
	if _, err := mem.AddResourcesToCollection(context.Background(), domain.AddResourcesToCollectionInput{
		CollectionID:  collection.CollectionID,
		OwnerUserID:   "dataset-user",
		OwnerOrgID:    "dataset-org",
		ResourceIDs:   []string{"file_dataset_a", "file_dataset_b"},
		AddedByUserID: "dataset-user",
		AddedAt:       now.Add(2 * time.Second),
	}); err != nil {
		t.Fatalf("AddResourcesToCollection: %v", err)
	}

	createReq := httptest.NewRequest(http.MethodPost, "/v2/dataset-snapshots", strings.NewReader(`{
		"name":"NPH training cohort v1",
		"description":"Frozen folder manifest for training",
		"source_collection_id":"`+collection.CollectionID+`",
		"project_id":"dataset-study",
		"metadata":{"label":"NPH"}
	}`))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("X-Ultra-User-Id", "dataset-user")
	createReq.Header.Set("X-Ultra-Org-Id", "dataset-org")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusCreated {
		t.Fatalf("create dataset snapshot status = %d body=%s, want 201", createRec.Code, createRec.Body.String())
	}
	var created datasetSnapshotResponse
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode dataset snapshot: %v", err)
	}
	if created.Snapshot.Name != "NPH training cohort v1" || created.Snapshot.ResourceCount != 2 || created.Snapshot.TotalBytes != 384 {
		t.Fatalf("created snapshot = %+v, want two-resource frozen dataset", created.Snapshot)
	}
	if len(created.Resources) != 2 || created.Resources[0].ResourceID != "file_dataset_a" || created.Resources[0].SHA256 != "sha-dataset-a" {
		t.Fatalf("created snapshot resources = %+v, want frozen ordered manifest", created.Resources)
	}

	getReq := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots/"+created.Snapshot.SnapshotID, nil)
	getReq.Header.Set("X-Ultra-User-Id", "dataset-user")
	getReq.Header.Set("X-Ultra-Org-Id", "dataset-org")
	getRec := httptest.NewRecorder()
	router.ServeHTTP(getRec, getReq)
	if getRec.Code != http.StatusOK {
		t.Fatalf("get dataset snapshot status = %d body=%s, want 200", getRec.Code, getRec.Body.String())
	}
	var loaded datasetSnapshotResponse
	if err := json.Unmarshal(getRec.Body.Bytes(), &loaded); err != nil {
		t.Fatalf("decode loaded dataset snapshot: %v", err)
	}
	if loaded.Snapshot.SnapshotID != created.Snapshot.SnapshotID || len(loaded.Resources) != 2 || loaded.Resources[1].ResourceID != "file_dataset_b" {
		t.Fatalf("loaded snapshot = %+v resources=%+v, want created manifest", loaded.Snapshot, loaded.Resources)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots?project_id=dataset-study", nil)
	listReq.Header.Set("X-Ultra-User-Id", "dataset-user")
	listReq.Header.Set("X-Ultra-Org-Id", "dataset-org")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list dataset snapshots status = %d body=%s, want 200", listRec.Code, listRec.Body.String())
	}
	var listed datasetSnapshotsResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &listed); err != nil {
		t.Fatalf("decode listed dataset snapshots: %v", err)
	}
	if listed.Count != 1 || len(listed.Snapshots) != 1 || listed.Snapshots[0].SnapshotID != created.Snapshot.SnapshotID {
		t.Fatalf("listed snapshots = %+v, want created project-scoped dataset snapshot", listed)
	}
}

func TestV2DatasetSnapshotDeleteAndRestoreLifecycle(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_dataset_lifecycle_a",
			OriginalName: "lifecycle-a.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    128,
			SHA256:       "sha-lifecycle-a",
			OwnerUserID:  "dataset-user",
			OwnerOrgID:   "dataset-org",
			ProjectID:    "dataset-study",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_dataset_lifecycle_b",
			OriginalName: "lifecycle-b.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    256,
			SHA256:       "sha-lifecycle-b",
			OwnerUserID:  "dataset-user",
			OwnerOrgID:   "dataset-org",
			ProjectID:    "dataset-study",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}
	snapshot, _, err := mem.CreateDatasetSnapshot(context.Background(), domain.CreateDatasetSnapshotInput{
		SnapshotID:      "dataset_snapshot_lifecycle",
		OwnerUserID:     "dataset-user",
		OwnerOrgID:      "dataset-org",
		ProjectID:       "dataset-study",
		Name:            "Lifecycle dataset snapshot",
		ResourceIDs:     []string{"file_dataset_lifecycle_a", "file_dataset_lifecycle_b"},
		CreatedByUserID: "dataset-user",
		CreatedAt:       now.Add(2 * time.Second),
		Metadata:        domain.JSONMap{"source": "test"},
	})
	if err != nil {
		t.Fatalf("CreateDatasetSnapshot: %v", err)
	}
	if _, err := mem.CreateDatasetSnapshotShareGrant(context.Background(), domain.CreateDatasetSnapshotShareGrantInput{
		SnapshotID:      snapshot.SnapshotID,
		OwnerUserID:     "dataset-user",
		OwnerOrgID:      "dataset-org",
		GranteeUserID:   "bob",
		GranteeOrgID:    "dataset-org",
		Role:            "read",
		CreatedByUserID: "dataset-user",
		CreatedAt:       now.Add(3 * time.Second),
		UpdatedAt:       now.Add(3 * time.Second),
		Metadata:        domain.JSONMap{"reason": "collaboration"},
	}); err != nil {
		t.Fatalf("CreateDatasetSnapshotShareGrant: %v", err)
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, "/v2/dataset-snapshots/"+snapshot.SnapshotID, nil)
	deleteReq.Header.Set("X-Ultra-User-Id", "dataset-user")
	deleteReq.Header.Set("X-Ultra-Org-Id", "dataset-org")
	deleteRec := httptest.NewRecorder()
	router.ServeHTTP(deleteRec, deleteReq)
	if deleteRec.Code != http.StatusOK {
		t.Fatalf("delete dataset snapshot status = %d body=%s, want 200", deleteRec.Code, deleteRec.Body.String())
	}
	var deleted datasetSnapshotResponse
	if err := json.Unmarshal(deleteRec.Body.Bytes(), &deleted); err != nil {
		t.Fatalf("decode deleted dataset snapshot: %v", err)
	}
	if deleted.Snapshot.Status != "deleted" || deleted.Snapshot.ResourceCount != 2 || len(deleted.Resources) != 2 {
		t.Fatalf("deleted snapshot = %+v resources=%+v, want deleted manifest response", deleted.Snapshot, deleted.Resources)
	}

	activeListReq := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots?project_id=dataset-study", nil)
	activeListReq.Header.Set("X-Ultra-User-Id", "dataset-user")
	activeListReq.Header.Set("X-Ultra-Org-Id", "dataset-org")
	activeListRec := httptest.NewRecorder()
	router.ServeHTTP(activeListRec, activeListReq)
	if activeListRec.Code != http.StatusOK {
		t.Fatalf("active list dataset snapshots status = %d body=%s, want 200", activeListRec.Code, activeListRec.Body.String())
	}
	var activeList datasetSnapshotsResponse
	if err := json.Unmarshal(activeListRec.Body.Bytes(), &activeList); err != nil {
		t.Fatalf("decode active list: %v", err)
	}
	if activeList.Count != 0 || len(activeList.Snapshots) != 0 {
		t.Fatalf("active snapshots = %+v, want deleted snapshot hidden", activeList)
	}

	deletedListReq := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots?project_id=dataset-study&status=deleted", nil)
	deletedListReq.Header.Set("X-Ultra-User-Id", "dataset-user")
	deletedListReq.Header.Set("X-Ultra-Org-Id", "dataset-org")
	deletedListRec := httptest.NewRecorder()
	router.ServeHTTP(deletedListRec, deletedListReq)
	if deletedListRec.Code != http.StatusOK {
		t.Fatalf("deleted list dataset snapshots status = %d body=%s, want 200", deletedListRec.Code, deletedListRec.Body.String())
	}
	var deletedList datasetSnapshotsResponse
	if err := json.Unmarshal(deletedListRec.Body.Bytes(), &deletedList); err != nil {
		t.Fatalf("decode deleted list: %v", err)
	}
	if deletedList.Count != 1 || len(deletedList.Snapshots) != 1 || deletedList.Snapshots[0].SnapshotID != snapshot.SnapshotID {
		t.Fatalf("deleted snapshots = %+v, want owner-visible deleted snapshot", deletedList)
	}

	bobDeletedListReq := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots?status=deleted", nil)
	bobDeletedListReq.Header.Set("X-Ultra-User-Id", "bob")
	bobDeletedListReq.Header.Set("X-Ultra-Org-Id", "dataset-org")
	bobDeletedListRec := httptest.NewRecorder()
	router.ServeHTTP(bobDeletedListRec, bobDeletedListReq)
	if bobDeletedListRec.Code != http.StatusOK {
		t.Fatalf("bob deleted list status = %d body=%s, want 200", bobDeletedListRec.Code, bobDeletedListRec.Body.String())
	}
	var bobDeletedList datasetSnapshotsResponse
	if err := json.Unmarshal(bobDeletedListRec.Body.Bytes(), &bobDeletedList); err != nil {
		t.Fatalf("decode Bob deleted list: %v", err)
	}
	if bobDeletedList.Count != 0 || len(bobDeletedList.Snapshots) != 0 {
		t.Fatalf("bob deleted snapshots = %+v, want deleted snapshot hidden from collaborators", bobDeletedList)
	}

	getDeletedReq := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots/"+snapshot.SnapshotID, nil)
	getDeletedReq.Header.Set("X-Ultra-User-Id", "dataset-user")
	getDeletedReq.Header.Set("X-Ultra-Org-Id", "dataset-org")
	getDeletedRec := httptest.NewRecorder()
	router.ServeHTTP(getDeletedRec, getDeletedReq)
	if getDeletedRec.Code != http.StatusNotFound {
		t.Fatalf("get deleted dataset snapshot status = %d body=%s, want 404", getDeletedRec.Code, getDeletedRec.Body.String())
	}

	eventsWhileDeletedReq := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots/"+snapshot.SnapshotID+"/events?limit=10", nil)
	eventsWhileDeletedReq.Header.Set("X-Ultra-User-Id", "dataset-user")
	eventsWhileDeletedReq.Header.Set("X-Ultra-Org-Id", "dataset-org")
	eventsWhileDeletedRec := httptest.NewRecorder()
	router.ServeHTTP(eventsWhileDeletedRec, eventsWhileDeletedReq)
	if eventsWhileDeletedRec.Code != http.StatusOK {
		t.Fatalf("events while deleted status = %d body=%s, want 200", eventsWhileDeletedRec.Code, eventsWhileDeletedRec.Body.String())
	}
	var eventsWhileDeleted datasetSnapshotEventsResponse
	if err := json.Unmarshal(eventsWhileDeletedRec.Body.Bytes(), &eventsWhileDeleted); err != nil {
		t.Fatalf("decode events while deleted: %v", err)
	}
	if len(eventsWhileDeleted.Events) == 0 || eventsWhileDeleted.Events[0].EventType != "dataset_snapshot.deleted" {
		t.Fatalf("events while deleted = %+v, want latest deleted audit event", eventsWhileDeleted)
	}

	restoreReq := httptest.NewRequest(http.MethodPost, "/v2/dataset-snapshots/"+snapshot.SnapshotID+"/restore", nil)
	restoreReq.Header.Set("X-Ultra-User-Id", "dataset-user")
	restoreReq.Header.Set("X-Ultra-Org-Id", "dataset-org")
	restoreRec := httptest.NewRecorder()
	router.ServeHTTP(restoreRec, restoreReq)
	if restoreRec.Code != http.StatusOK {
		t.Fatalf("restore dataset snapshot status = %d body=%s, want 200", restoreRec.Code, restoreRec.Body.String())
	}
	var restored datasetSnapshotResponse
	if err := json.Unmarshal(restoreRec.Body.Bytes(), &restored); err != nil {
		t.Fatalf("decode restored dataset snapshot: %v", err)
	}
	if restored.Snapshot.Status != "active" || len(restored.Resources) != 2 || restored.Resources[1].SHA256 != "sha-lifecycle-b" {
		t.Fatalf("restored snapshot = %+v resources=%+v, want active frozen manifest", restored.Snapshot, restored.Resources)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots/"+snapshot.SnapshotID+"/events?limit=10", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "dataset-user")
	eventsReq.Header.Set("X-Ultra-Org-Id", "dataset-org")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("events after restore status = %d body=%s, want 200", eventsRec.Code, eventsRec.Body.String())
	}
	var events datasetSnapshotEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode restored events: %v", err)
	}
	seenDeleted := false
	seenRestored := false
	for _, event := range events.Events {
		switch event.EventType {
		case "dataset_snapshot.deleted":
			seenDeleted = true
			if event.Metadata["snapshot_id"] != snapshot.SnapshotID {
				t.Fatalf("deleted event metadata = %+v, want snapshot_id", event.Metadata)
			}
		case "dataset_snapshot.restored":
			seenRestored = true
			if event.Metadata["resource_count"] != float64(2) {
				t.Fatalf("restored event metadata = %+v, want resource_count 2", event.Metadata)
			}
		}
	}
	if !seenDeleted || !seenRestored {
		t.Fatalf("events = %+v, want deleted and restored lifecycle events", events.Events)
	}
}

func TestV2DatasetSnapshotsCreateFromResourceQuery(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_query_dataset_a",
			OriginalName: "NPH_shunt_001_69yo.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    128,
			SHA256:       "sha-query-dataset-a",
			OwnerUserID:  "dataset-user",
			OwnerOrgID:   "dataset-org",
			ProjectID:    "dataset-study",
			Status:       "active",
			Tags:         []string{"NPH", "Under 70"},
			CreatedAt:    now,
			UpdatedAt:    now,
			Metadata:     domain.JSONMap{"age": 69, "diagnosis": "NPH"},
		},
		{
			ResourceID:   "file_query_dataset_b",
			OriginalName: "NPH_shunt_002_62yo.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    256,
			SHA256:       "sha-query-dataset-b",
			OwnerUserID:  "dataset-user",
			OwnerOrgID:   "dataset-org",
			ProjectID:    "dataset-study",
			Status:       "active",
			Tags:         []string{"NPH", "Under 70"},
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
			Metadata:     domain.JSONMap{"age": 62, "diagnosis": "NPH"},
		},
		{
			ResourceID:   "file_query_dataset_over70",
			OriginalName: "NPH_shunt_003_74yo.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    512,
			SHA256:       "sha-query-dataset-over70",
			OwnerUserID:  "dataset-user",
			OwnerOrgID:   "dataset-org",
			ProjectID:    "dataset-study",
			Status:       "active",
			Tags:         []string{"NPH", "Over 70"},
			CreatedAt:    now.Add(2 * time.Second),
			UpdatedAt:    now.Add(2 * time.Second),
			Metadata:     domain.JSONMap{"age": 74, "diagnosis": "NPH"},
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	createReq := httptest.NewRequest(http.MethodPost, "/v2/dataset-snapshots", strings.NewReader(`{
		"name":"NPH under 70 query cohort",
		"description":"Frozen Resources query result for training",
		"metadata":{"source":"resources_query_toolbar"},
		"resource_query":{
			"q":"NPH",
			"kind":"file",
			"source":"upload",
			"project_id":"dataset-study",
			"tags":["Under 70"]
		}
	}`))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("X-Ultra-User-Id", "dataset-user")
	createReq.Header.Set("X-Ultra-Org-Id", "dataset-org")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusCreated {
		t.Fatalf("create query dataset snapshot status = %d body=%s, want 201", createRec.Code, createRec.Body.String())
	}
	var created datasetSnapshotResponse
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode query dataset snapshot: %v", err)
	}
	if created.Snapshot.Name != "NPH under 70 query cohort" || created.Snapshot.ResourceCount != 2 || created.Snapshot.TotalBytes != 384 {
		t.Fatalf("created query snapshot = %+v, want two under-70 matching resources", created.Snapshot)
	}
	got := []string{created.Resources[0].ResourceID, created.Resources[1].ResourceID}
	if got[0] != "file_query_dataset_b" || got[1] != "file_query_dataset_a" {
		t.Fatalf("created query resources = %v, want newest matching resources first", got)
	}
}

func TestV2DatasetSnapshotShareGrantAllowsCollaboratorInspect(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_shared_dataset_a",
			OriginalName: "shared-dataset-a.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    128,
			SHA256:       "sha-shared-dataset-a",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "dataset-study",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "file_shared_dataset_b",
			OriginalName: "shared-dataset-b.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    256,
			SHA256:       "sha-shared-dataset-b",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			ProjectID:    "dataset-study",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}
	snapshot, _, err := mem.CreateDatasetSnapshot(context.Background(), domain.CreateDatasetSnapshotInput{
		SnapshotID:      "dataset_snapshot_shared_http",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		ProjectID:       "dataset-study",
		Name:            "Shared dataset snapshot",
		ResourceIDs:     []string{"file_shared_dataset_a", "file_shared_dataset_b"},
		CreatedByUserID: "alice",
		CreatedAt:       now.Add(2 * time.Second),
		Metadata:        domain.JSONMap{"label": "NPH"},
	})
	if err != nil {
		t.Fatalf("CreateDatasetSnapshot: %v", err)
	}

	bobGetBefore := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots/"+snapshot.SnapshotID, nil)
	bobGetBefore.Header.Set("X-Ultra-User-Id", "bob")
	bobGetBefore.Header.Set("X-Ultra-Org-Id", "org-b")
	bobGetBeforeRec := httptest.NewRecorder()
	router.ServeHTTP(bobGetBeforeRec, bobGetBefore)
	if bobGetBeforeRec.Code != http.StatusNotFound {
		t.Fatalf("bob pre-share get status = %d body=%s, want 404", bobGetBeforeRec.Code, bobGetBeforeRec.Body.String())
	}

	shareReq := httptest.NewRequest(http.MethodPost, "/v2/dataset-snapshots/"+snapshot.SnapshotID+"/shares", strings.NewReader(`{
		"grantee_user_id":"bob",
		"grantee_org_id":"org-b",
		"role":"read",
		"metadata":{"reason":"collaborator review"}
	}`))
	shareReq.Header.Set("Content-Type", "application/json")
	shareReq.Header.Set("X-Ultra-User-Id", "alice")
	shareReq.Header.Set("X-Ultra-Org-Id", "org-a")
	shareRec := httptest.NewRecorder()
	router.ServeHTTP(shareRec, shareReq)
	if shareRec.Code != http.StatusCreated {
		t.Fatalf("create dataset snapshot share status = %d body=%s, want 201", shareRec.Code, shareRec.Body.String())
	}
	var shared datasetSnapshotShareGrantResponse
	if err := json.Unmarshal(shareRec.Body.Bytes(), &shared); err != nil {
		t.Fatalf("decode created dataset snapshot share: %v", err)
	}
	if shared.Grant.SnapshotID != snapshot.SnapshotID || shared.Grant.GranteeUserID != "bob" || shared.Grant.Status != "active" {
		t.Fatalf("created share = %+v, want active Bob grant", shared.Grant)
	}

	bobEventsReq := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots/"+snapshot.SnapshotID+"/events?limit=10", nil)
	bobEventsReq.Header.Set("X-Ultra-User-Id", "bob")
	bobEventsReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobEventsRec := httptest.NewRecorder()
	router.ServeHTTP(bobEventsRec, bobEventsReq)
	if bobEventsRec.Code != http.StatusOK {
		t.Fatalf("bob dataset snapshot events status = %d body=%s, want 200", bobEventsRec.Code, bobEventsRec.Body.String())
	}
	var bobEvents datasetSnapshotEventsResponse
	if err := json.Unmarshal(bobEventsRec.Body.Bytes(), &bobEvents); err != nil {
		t.Fatalf("decode Bob dataset snapshot events: %v", err)
	}
	if bobEvents.Count != 2 || len(bobEvents.Events) != 2 || bobEvents.Events[0].EventType != "dataset_snapshot.shared" {
		t.Fatalf("bob dataset snapshot events = %+v, want shared plus created audit events", bobEvents)
	}

	bobGet := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots/"+snapshot.SnapshotID, nil)
	bobGet.Header.Set("X-Ultra-User-Id", "bob")
	bobGet.Header.Set("X-Ultra-Org-Id", "org-b")
	bobGetRec := httptest.NewRecorder()
	router.ServeHTTP(bobGetRec, bobGet)
	if bobGetRec.Code != http.StatusOK {
		t.Fatalf("bob shared get status = %d body=%s, want 200", bobGetRec.Code, bobGetRec.Body.String())
	}
	var loaded datasetSnapshotResponse
	if err := json.Unmarshal(bobGetRec.Body.Bytes(), &loaded); err != nil {
		t.Fatalf("decode Bob dataset snapshot: %v", err)
	}
	if loaded.Snapshot.OwnerUserID != "alice" || len(loaded.Resources) != 2 || loaded.Resources[0].SHA256 != "sha-shared-dataset-a" {
		t.Fatalf("bob loaded snapshot = %+v resources=%+v, want frozen shared manifest", loaded.Snapshot, loaded.Resources)
	}

	bobList := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots?limit=20", nil)
	bobList.Header.Set("X-Ultra-User-Id", "bob")
	bobList.Header.Set("X-Ultra-Org-Id", "org-b")
	bobListRec := httptest.NewRecorder()
	router.ServeHTTP(bobListRec, bobList)
	if bobListRec.Code != http.StatusOK {
		t.Fatalf("bob shared list status = %d body=%s, want 200", bobListRec.Code, bobListRec.Body.String())
	}
	var listed datasetSnapshotsResponse
	if err := json.Unmarshal(bobListRec.Body.Bytes(), &listed); err != nil {
		t.Fatalf("decode Bob shared dataset snapshots: %v", err)
	}
	if listed.Count != 1 || len(listed.Snapshots) != 1 || listed.Snapshots[0].SnapshotID != snapshot.SnapshotID {
		t.Fatalf("bob listed snapshots = %+v, want shared snapshot", listed)
	}

	listSharesReq := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots/"+snapshot.SnapshotID+"/shares?status=active", nil)
	listSharesReq.Header.Set("X-Ultra-User-Id", "alice")
	listSharesReq.Header.Set("X-Ultra-Org-Id", "org-a")
	listSharesRec := httptest.NewRecorder()
	router.ServeHTTP(listSharesRec, listSharesReq)
	if listSharesRec.Code != http.StatusOK {
		t.Fatalf("list dataset snapshot shares status = %d body=%s, want 200", listSharesRec.Code, listSharesRec.Body.String())
	}
	var listedShares datasetSnapshotShareGrantsResponse
	if err := json.Unmarshal(listSharesRec.Body.Bytes(), &listedShares); err != nil {
		t.Fatalf("decode dataset snapshot shares: %v", err)
	}
	if listedShares.Count != 1 || listedShares.Grants[0].GrantID != shared.Grant.GrantID {
		t.Fatalf("listed shares = %+v, want created grant", listedShares)
	}

	revokeReq := httptest.NewRequest(http.MethodDelete, "/v2/dataset-snapshots/"+snapshot.SnapshotID+"/shares/"+shared.Grant.GrantID, nil)
	revokeReq.Header.Set("X-Ultra-User-Id", "alice")
	revokeReq.Header.Set("X-Ultra-Org-Id", "org-a")
	revokeRec := httptest.NewRecorder()
	router.ServeHTTP(revokeRec, revokeReq)
	if revokeRec.Code != http.StatusOK {
		t.Fatalf("revoke dataset snapshot share status = %d body=%s, want 200", revokeRec.Code, revokeRec.Body.String())
	}
	var revoked datasetSnapshotShareGrantResponse
	if err := json.Unmarshal(revokeRec.Body.Bytes(), &revoked); err != nil {
		t.Fatalf("decode revoked dataset snapshot share: %v", err)
	}
	if revoked.Grant.Status != "revoked" || revoked.Grant.RevokedAt.IsZero() {
		t.Fatalf("revoked share = %+v, want revoked lifecycle", revoked.Grant)
	}

	ownerEventsReq := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots/"+snapshot.SnapshotID+"/events?limit=10", nil)
	ownerEventsReq.Header.Set("X-Ultra-User-Id", "alice")
	ownerEventsReq.Header.Set("X-Ultra-Org-Id", "org-a")
	ownerEventsRec := httptest.NewRecorder()
	router.ServeHTTP(ownerEventsRec, ownerEventsReq)
	if ownerEventsRec.Code != http.StatusOK {
		t.Fatalf("owner dataset snapshot events status = %d body=%s, want 200", ownerEventsRec.Code, ownerEventsRec.Body.String())
	}
	var ownerEvents datasetSnapshotEventsResponse
	if err := json.Unmarshal(ownerEventsRec.Body.Bytes(), &ownerEvents); err != nil {
		t.Fatalf("decode owner dataset snapshot events: %v", err)
	}
	gotEventTypes := []string{}
	for _, event := range ownerEvents.Events {
		gotEventTypes = append(gotEventTypes, event.EventType)
	}
	if !reflect.DeepEqual(gotEventTypes, []string{"dataset_snapshot.share_revoked", "dataset_snapshot.shared", "dataset_snapshot.created"}) {
		t.Fatalf("owner dataset snapshot event types = %v, want revoke/share/create", gotEventTypes)
	}

	bobGetAfter := httptest.NewRequest(http.MethodGet, "/v2/dataset-snapshots/"+snapshot.SnapshotID, nil)
	bobGetAfter.Header.Set("X-Ultra-User-Id", "bob")
	bobGetAfter.Header.Set("X-Ultra-Org-Id", "org-b")
	bobGetAfterRec := httptest.NewRecorder()
	router.ServeHTTP(bobGetAfterRec, bobGetAfter)
	if bobGetAfterRec.Code != http.StatusNotFound {
		t.Fatalf("bob post-revoke get status = %d body=%s, want 404", bobGetAfterRec.Code, bobGetAfterRec.Body.String())
	}
}

func TestV2DataAgentJobsCreateListInspectAndAudit(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_agent_http_a",
			OriginalName: "agent-a.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    128,
			SHA256:       "sha-agent-a",
			OwnerUserID:  "agent-user",
			OwnerOrgID:   "agent-org",
			ProjectID:    "agent-study",
			Status:       "active",
			Tags:         []string{"NPH", "Under 70"},
			CreatedAt:    now,
			UpdatedAt:    now,
			Metadata:     domain.JSONMap{"diagnosis": "NPH", "age": 69},
		},
		{
			ResourceID:   "file_agent_http_b",
			OriginalName: "agent-b.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    256,
			SHA256:       "sha-agent-b",
			OwnerUserID:  "agent-user",
			OwnerOrgID:   "agent-org",
			ProjectID:    "agent-study",
			Status:       "active",
			Tags:         []string{"NPH", "Under 70"},
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
			Metadata:     domain.JSONMap{"diagnosis": "NPH", "age": 62},
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	createReq := httptest.NewRequest(http.MethodPost, "/v2/data-agent/jobs", strings.NewReader(`{
		"job_type":"caption_resources",
		"resource_ids":["file_agent_http_a","file_agent_http_b"],
		"project_id":"agent-study",
		"input_selector":{"mode":"short_caption","label":"NPH"},
		"metadata":{"requested_from":"resources_page"}
	}`))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("X-Ultra-User-Id", "agent-user")
	createReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusAccepted {
		t.Fatalf("create data-agent job status = %d body=%s, want 202", createRec.Code, createRec.Body.String())
	}
	var created dataAgentJobResponse
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created data-agent job: %v", err)
	}
	if created.Job.JobType != "caption_resources" || created.Job.Status != "queued" || created.Job.ResourceCount != 2 {
		t.Fatalf("created job = %+v, want queued caption_resources over two resources", created.Job)
	}
	if len(created.Events) != 1 || created.Events[0].EventType != "data_agent.job.created" {
		t.Fatalf("created job events = %+v, want created event", created.Events)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/data-agent/jobs?status=queued&job_type=caption_resources&limit=10", nil)
	listReq.Header.Set("X-Ultra-User-Id", "agent-user")
	listReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list data-agent jobs status = %d body=%s, want 200", listRec.Code, listRec.Body.String())
	}
	var listed dataAgentJobsResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &listed); err != nil {
		t.Fatalf("decode listed data-agent jobs: %v", err)
	}
	if listed.Count != 1 || len(listed.Jobs) != 1 || listed.Jobs[0].JobID != created.Job.JobID {
		t.Fatalf("listed jobs = %+v, want created job", listed)
	}

	getReq := httptest.NewRequest(http.MethodGet, "/v2/data-agent/jobs/"+created.Job.JobID, nil)
	getReq.Header.Set("X-Ultra-User-Id", "agent-user")
	getReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	getRec := httptest.NewRecorder()
	router.ServeHTTP(getRec, getReq)
	if getRec.Code != http.StatusOK {
		t.Fatalf("get data-agent job status = %d body=%s, want 200", getRec.Code, getRec.Body.String())
	}
	var loaded dataAgentJobResponse
	if err := json.Unmarshal(getRec.Body.Bytes(), &loaded); err != nil {
		t.Fatalf("decode loaded data-agent job: %v", err)
	}
	if loaded.Job.JobID != created.Job.JobID || len(loaded.Events) != 1 || loaded.Events[0].Sequence != 1 {
		t.Fatalf("loaded job = %+v events=%+v, want created job with audit trail", loaded.Job, loaded.Events)
	}

	progressReq := httptest.NewRequest(
		http.MethodPatch,
		"/v2/data-agent/jobs/"+created.Job.JobID+"/status",
		strings.NewReader(`{"status":"running","progress_completed":1,"progress_total":2,"message":"Captioned first resource","event_metadata":{"resource_id":"file_agent_http_a"}}`),
	)
	progressReq.Header.Set("Content-Type", "application/json")
	progressReq.Header.Set("X-Ultra-User-Id", "agent-user")
	progressReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	progressRec := httptest.NewRecorder()
	router.ServeHTTP(progressRec, progressReq)
	if progressRec.Code != http.StatusOK {
		t.Fatalf("progress data-agent job status = %d body=%s, want 200", progressRec.Code, progressRec.Body.String())
	}
	var progressed dataAgentJobResponse
	if err := json.Unmarshal(progressRec.Body.Bytes(), &progressed); err != nil {
		t.Fatalf("decode progressed data-agent job: %v", err)
	}
	if progressed.Job.Status != "running" || progressed.Job.ProgressCompleted != 1 || len(progressed.Events) != 2 || progressed.Events[1].EventType != "data_agent.job.progressed" {
		t.Fatalf("progressed job = %+v events=%+v, want running job with progress event", progressed.Job, progressed.Events)
	}

	cancelReq := httptest.NewRequest(
		http.MethodPost,
		"/v2/data-agent/jobs/"+created.Job.JobID+"/control",
		strings.NewReader(`{"action":"cancel","reason":"User paused the field upload."}`),
	)
	cancelReq.Header.Set("Content-Type", "application/json")
	cancelReq.Header.Set("X-Ultra-User-Id", "agent-user")
	cancelReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	cancelRec := httptest.NewRecorder()
	router.ServeHTTP(cancelRec, cancelReq)
	if cancelRec.Code != http.StatusOK {
		t.Fatalf("cancel data-agent job status = %d body=%s, want 200", cancelRec.Code, cancelRec.Body.String())
	}
	var canceled dataAgentJobResponse
	if err := json.Unmarshal(cancelRec.Body.Bytes(), &canceled); err != nil {
		t.Fatalf("decode canceled data-agent job: %v", err)
	}
	if canceled.Job.Status != "canceled" || canceled.Job.Error == "" || len(canceled.Events) != 3 || canceled.Events[2].EventType != "data_agent.job.canceled" {
		t.Fatalf("canceled job = %+v events=%+v, want canceled job with audit event", canceled.Job, canceled.Events)
	}

	retryReq := httptest.NewRequest(
		http.MethodPost,
		"/v2/data-agent/jobs/"+created.Job.JobID+"/control",
		strings.NewReader(`{"action":"retry","reason":"Connectivity recovered."}`),
	)
	retryReq.Header.Set("Content-Type", "application/json")
	retryReq.Header.Set("X-Ultra-User-Id", "agent-user")
	retryReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	retryRec := httptest.NewRecorder()
	router.ServeHTTP(retryRec, retryReq)
	if retryRec.Code != http.StatusOK {
		t.Fatalf("retry data-agent job status = %d body=%s, want 200", retryRec.Code, retryRec.Body.String())
	}
	var retried dataAgentJobResponse
	if err := json.Unmarshal(retryRec.Body.Bytes(), &retried); err != nil {
		t.Fatalf("decode retried data-agent job: %v", err)
	}
	if retried.Job.Status != "queued" || retried.Job.ProgressCompleted != 0 || retried.Job.Error != "" || len(retried.Events) != 4 || retried.Events[3].EventType != "data_agent.job.retried" {
		t.Fatalf("retried job = %+v events=%+v, want reset queued job with retry event", retried.Job, retried.Events)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/file_agent_http_a/events", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "agent-user")
	eventsReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("resource events status = %d body=%s, want 200", eventsRec.Code, eventsRec.Body.String())
	}
	var resourceEvents resourceEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &resourceEvents); err != nil {
		t.Fatalf("decode resource events: %v", err)
	}
	foundJobEvent := false
	for _, event := range resourceEvents.Events {
		if event.EventType == "resource.data_agent_job_queued" && event.Metadata["job_id"] == created.Job.JobID {
			foundJobEvent = true
			break
		}
	}
	if !foundJobEvent {
		t.Fatalf("resource events = %+v, want data-agent queued audit event", resourceEvents.Events)
	}
}

func seedDataAgentHTTPResources(t *testing.T, mem *store.MemoryStore) {
	t.Helper()
	now := domain.Now()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "file_agent_http_a",
			OriginalName: "agent-a.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    128,
			SHA256:       "sha-agent-a",
			OwnerUserID:  "agent-user",
			OwnerOrgID:   "agent-org",
			ProjectID:    "agent-study",
			Status:       "active",
			Tags:         []string{"NPH", "Under 70"},
			CreatedAt:    now,
			UpdatedAt:    now,
			Metadata:     domain.JSONMap{"diagnosis": "NPH", "age": 69},
		},
		{
			ResourceID:   "file_agent_http_b",
			OriginalName: "agent-b.nii.gz",
			ContentType:  "application/gzip",
			ResourceKind: "file",
			SourceType:   "upload",
			SizeBytes:    256,
			SHA256:       "sha-agent-b",
			OwnerUserID:  "agent-user",
			OwnerOrgID:   "agent-org",
			ProjectID:    "agent-study",
			Status:       "active",
			Tags:         []string{"NPH", "Under 70"},
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
			Metadata:     domain.JSONMap{"diagnosis": "NPH", "age": 62},
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}
}

func TestV2DataAgentJobCreateDispatchesQueueEnvelopeAndAuditEvent(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	dispatcher := &recordingDataAgentJobPublisher{}
	router := NewRouter(ServerDeps{
		Version:       "test-version",
		Runs:          runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:         mem,
		UploadRoot:    uploadRoot,
		DataAgentJobs: dispatcher,
	})
	seedDataAgentHTTPResources(t, mem)

	createReq := httptest.NewRequest(http.MethodPost, "/v2/data-agent/jobs", strings.NewReader(`{
		"job_type":"caption_resources",
		"resource_ids":["file_agent_http_a","file_agent_http_b"],
		"project_id":"agent-study",
		"input_selector":{"mode":"short_caption","label":"NPH"},
		"metadata":{"requested_from":"resources_page"}
	}`))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("X-Ultra-User-Id", "agent-user")
	createReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusAccepted {
		t.Fatalf("create data-agent job status = %d body=%s, want 202", createRec.Code, createRec.Body.String())
	}
	var created dataAgentJobResponse
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created data-agent job: %v", err)
	}
	if len(dispatcher.jobs) != 1 {
		t.Fatalf("published data-agent jobs = %+v, want exactly one dispatch", dispatcher.jobs)
	}
	dispatched := dispatcher.jobs[0]
	if dispatched.JobID != created.Job.JobID || dispatched.JobType != "caption_resources" {
		t.Fatalf("dispatched job = %+v, want created caption job", dispatched)
	}
	if dispatched.DispatchID == "" {
		t.Fatalf("dispatched job missing dispatch_id")
	}
	if dispatched.OwnerUserID != "agent-user" || dispatched.OwnerOrgID != "agent-org" || dispatched.ProjectID != "agent-study" {
		t.Fatalf("dispatched owner/project = %+v, want request principal/project", dispatched)
	}
	if !reflect.DeepEqual(dispatched.ResourceIDs, []string{"file_agent_http_a", "file_agent_http_b"}) {
		t.Fatalf("dispatched resource ids = %#v, want selected resources", dispatched.ResourceIDs)
	}
	if dispatched.InputSelector["label"] != "NPH" || dispatched.InputSelector["mode"] != "short_caption" {
		t.Fatalf("dispatched input selector = %#v, want original selector", dispatched.InputSelector)
	}
	if dispatched.Metadata["requested_from"] != "resources_page" {
		t.Fatalf("dispatched metadata = %#v, want original metadata", dispatched.Metadata)
	}
	if len(created.Events) != 2 || created.Events[1].EventType != "data_agent.job.dispatched" {
		t.Fatalf("created events = %+v, want created plus dispatched audit event", created.Events)
	}
	if created.Events[1].Metadata["dispatch_id"] != dispatched.DispatchID {
		t.Fatalf("dispatch event metadata = %#v, want dispatch_id %q", created.Events[1].Metadata, dispatched.DispatchID)
	}
}

func TestV2DataAgentJobCreateDispatchesResourceQuerySelector(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	dispatcher := &recordingDataAgentJobPublisher{}
	router := NewRouter(ServerDeps{
		Version:       "test-version",
		Runs:          runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:         mem,
		UploadRoot:    uploadRoot,
		DataAgentJobs: dispatcher,
	})
	seedDataAgentHTTPResources(t, mem)

	createReq := httptest.NewRequest(http.MethodPost, "/v2/data-agent/jobs", strings.NewReader(`{
		"job_type":"create_dataset_snapshot",
		"project_id":"agent-study",
		"input_selector":{"snapshot_name":"NPH under 70 query cohort"},
		"resource_query":{
			"q":"NPH",
			"kind":"file",
			"source":"upload",
			"project_id":"agent-study",
			"tags":["Under 70"]
		},
		"metadata":{"requested_from":"resources_query_toolbar"}
	}`))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("X-Ultra-User-Id", "agent-user")
	createReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusAccepted {
		t.Fatalf("create query data-agent job status = %d body=%s, want 202", createRec.Code, createRec.Body.String())
	}
	var created dataAgentJobResponse
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created query data-agent job: %v", err)
	}
	if len(dispatcher.jobs) != 1 {
		t.Fatalf("published data-agent jobs = %+v, want one query dispatch", dispatcher.jobs)
	}
	dispatched := dispatcher.jobs[0]
	if dispatched.JobType != "create_dataset_snapshot" || len(dispatched.ResourceIDs) != 0 || dispatched.ResourceCount != 2 {
		t.Fatalf("dispatched job = %+v, want query selector with counted resources and no preselected IDs", dispatched)
	}
	query, ok := dispatched.InputSelector["resource_query"].(domain.JSONMap)
	if !ok {
		t.Fatalf("dispatched input selector = %#v, want resource_query object", dispatched.InputSelector)
	}
	if query["q"] != "NPH" || query["kind"] != "file" || query["source"] != "upload" || query["project_id"] != "agent-study" {
		t.Fatalf("dispatched query = %#v, want normalized Resources query", query)
	}
	tags, ok := query["tags"].([]string)
	if !ok || !reflect.DeepEqual(tags, []string{"Under 70"}) {
		t.Fatalf("dispatched query tags = %#v, want normalized tag filter", query["tags"])
	}
	if dispatched.InputSelector["snapshot_name"] != "NPH under 70 query cohort" {
		t.Fatalf("dispatched input selector = %#v, want snapshot name preserved", dispatched.InputSelector)
	}
	if created.Job.ResourceCount != 2 || created.Job.ProgressTotal != 2 || created.Job.InputSelector["resource_ids"] != nil {
		t.Fatalf("created job = %+v, want counted selector job without expanded resource ids", created.Job)
	}
	if len(created.Events) != 2 || created.Events[1].EventType != "data_agent.job.dispatched" {
		t.Fatalf("created events = %+v, want created plus dispatched audit event", created.Events)
	}
}

func TestV2DataAgentJobCreateRejectsResourceQueryAboveWorkerLimit(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	limited := &overLimitDataAgentQueryStore{
		MemoryStore: mem,
		totalCount:  domain.DataAgentQueryResourceHardLimit + 1,
	}
	dispatcher := &recordingDataAgentJobPublisher{}
	router := NewRouter(ServerDeps{
		Version:       "test-version",
		Runs:          runcontrol.NewService(limited, eventbus.NewMemoryBus()),
		Store:         limited,
		UploadRoot:    uploadRoot,
		DataAgentJobs: dispatcher,
	})

	createReq := httptest.NewRequest(http.MethodPost, "/v2/data-agent/jobs", strings.NewReader(`{
		"job_type":"extract_metadata",
		"project_id":"agent-study",
		"resource_query":{
			"q":"NPH",
			"kind":"file",
			"source":"upload",
			"project_id":"agent-study"
		}
	}`))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("X-Ultra-User-Id", "agent-user")
	createReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusBadRequest {
		t.Fatalf("create over-limit query data-agent job status = %d body=%s, want 400", createRec.Code, createRec.Body.String())
	}
	if len(dispatcher.jobs) != 0 {
		t.Fatalf("published data-agent jobs = %+v, want no dispatch for over-limit query", dispatcher.jobs)
	}
	if !strings.Contains(createRec.Body.String(), strconv.Itoa(domain.DataAgentQueryResourceHardLimit+1)+" resources") {
		t.Fatalf("create over-limit query body = %s, want resource count in error", createRec.Body.String())
	}
	if limited.lastListInput.Query != "NPH" || limited.lastListInput.Limit != 1 || limited.lastListInput.UserID != "agent-user" {
		t.Fatalf("count query input = %+v, want owner-scoped one-row count query", limited.lastListInput)
	}
}

func TestV2DataAgentJobCreateAcceptsBatchTagResources(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	dispatcher := &recordingDataAgentJobPublisher{}
	router := NewRouter(ServerDeps{
		Version:       "test-version",
		Runs:          runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:         mem,
		UploadRoot:    uploadRoot,
		DataAgentJobs: dispatcher,
	})
	seedDataAgentHTTPResources(t, mem)

	createReq := httptest.NewRequest(http.MethodPost, "/v2/data-agent/jobs", strings.NewReader(`{
		"job_type":"batch_tag_resources",
		"resource_ids":["file_agent_http_a","file_agent_http_b"],
		"input_selector":{"tags":["NPH","Under 70","NPH"]},
		"metadata":{"requested_from":"resources_data_agent_launcher"}
	}`))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("X-Ultra-User-Id", "agent-user")
	createReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusAccepted {
		t.Fatalf("create batch-tag data-agent job status = %d body=%s, want 202", createRec.Code, createRec.Body.String())
	}
	var created dataAgentJobResponse
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created data-agent job: %v", err)
	}
	if created.Job.JobType != "batch_tag_resources" {
		t.Fatalf("created job type = %q, want batch_tag_resources", created.Job.JobType)
	}
	if len(dispatcher.jobs) != 1 {
		t.Fatalf("published data-agent jobs = %+v, want exactly one dispatch", dispatcher.jobs)
	}
	dispatched := dispatcher.jobs[0]
	if dispatched.JobType != "batch_tag_resources" || !reflect.DeepEqual(dispatched.ResourceIDs, []string{"file_agent_http_a", "file_agent_http_b"}) {
		t.Fatalf("dispatched job = %+v, want batch tag over selected resources", dispatched)
	}
	if !reflect.DeepEqual(metadataStringSlice(dispatched.InputSelector["tags"]), []string{"NPH", "Under 70", "NPH"}) {
		t.Fatalf("dispatched input selector = %#v, want tag request preserved", dispatched.InputSelector)
	}
}

func TestV2DataAgentJobRetryDispatchesFreshQueueEnvelope(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	dispatcher := &recordingDataAgentJobPublisher{}
	router := NewRouter(ServerDeps{
		Version:       "test-version",
		Runs:          runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:         mem,
		UploadRoot:    uploadRoot,
		DataAgentJobs: dispatcher,
	})
	seedDataAgentHTTPResources(t, mem)

	createReq := httptest.NewRequest(http.MethodPost, "/v2/data-agent/jobs", strings.NewReader(`{
		"job_type":"extract_metadata",
		"resource_ids":["file_agent_http_a"],
		"project_id":"agent-study",
		"input_selector":{"format":"nifti"}
	}`))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("X-Ultra-User-Id", "agent-user")
	createReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusAccepted {
		t.Fatalf("create data-agent job status = %d body=%s, want 202", createRec.Code, createRec.Body.String())
	}
	var created dataAgentJobResponse
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created data-agent job: %v", err)
	}
	if len(dispatcher.jobs) != 1 || dispatcher.jobs[0].DispatchID == "" {
		t.Fatalf("initial dispatched jobs = %+v, want one dispatch with id", dispatcher.jobs)
	}
	initialDispatchID := dispatcher.jobs[0].DispatchID

	cancelReq := httptest.NewRequest(
		http.MethodPost,
		"/v2/data-agent/jobs/"+created.Job.JobID+"/control",
		strings.NewReader(`{"action":"cancel","reason":"field connection dropped"}`),
	)
	cancelReq.Header.Set("Content-Type", "application/json")
	cancelReq.Header.Set("X-Ultra-User-Id", "agent-user")
	cancelReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	cancelRec := httptest.NewRecorder()
	router.ServeHTTP(cancelRec, cancelReq)
	if cancelRec.Code != http.StatusOK {
		t.Fatalf("cancel data-agent job status = %d body=%s, want 200", cancelRec.Code, cancelRec.Body.String())
	}
	if len(dispatcher.jobs) != 1 {
		t.Fatalf("cancel published data-agent jobs = %+v, want no new dispatch", dispatcher.jobs)
	}

	retryReq := httptest.NewRequest(
		http.MethodPost,
		"/v2/data-agent/jobs/"+created.Job.JobID+"/control",
		strings.NewReader(`{"action":"retry","reason":"connection recovered"}`),
	)
	retryReq.Header.Set("Content-Type", "application/json")
	retryReq.Header.Set("X-Ultra-User-Id", "agent-user")
	retryReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	retryRec := httptest.NewRecorder()
	router.ServeHTTP(retryRec, retryReq)
	if retryRec.Code != http.StatusOK {
		t.Fatalf("retry data-agent job status = %d body=%s, want 200", retryRec.Code, retryRec.Body.String())
	}
	var retried dataAgentJobResponse
	if err := json.Unmarshal(retryRec.Body.Bytes(), &retried); err != nil {
		t.Fatalf("decode retried data-agent job: %v", err)
	}
	if len(dispatcher.jobs) != 2 {
		t.Fatalf("published data-agent jobs = %+v, want initial and retry dispatches", dispatcher.jobs)
	}
	retryDispatch := dispatcher.jobs[1]
	if retryDispatch.JobID != created.Job.JobID || retryDispatch.DispatchID == "" || retryDispatch.DispatchID == initialDispatchID {
		t.Fatalf("retry dispatch = %+v, want same job with fresh dispatch id different from %q", retryDispatch, initialDispatchID)
	}
	if retryDispatch.JobType != "extract_metadata" || !reflect.DeepEqual(retryDispatch.ResourceIDs, []string{"file_agent_http_a"}) {
		t.Fatalf("retry dispatch = %+v, want original metadata extraction selection", retryDispatch)
	}
	if len(retried.Events) < 2 || retried.Events[len(retried.Events)-2].EventType != "data_agent.job.retried" || retried.Events[len(retried.Events)-1].EventType != "data_agent.job.dispatched" {
		t.Fatalf("retry events = %+v, want retried then dispatched audit events", retried.Events)
	}
	if retried.Events[len(retried.Events)-1].Metadata["dispatch_id"] != retryDispatch.DispatchID {
		t.Fatalf("retry dispatch event metadata = %#v, want dispatch_id %q", retried.Events[len(retried.Events)-1].Metadata, retryDispatch.DispatchID)
	}
}

func TestV2DataAgentJobLeaseClaimRenewAndRelease(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	seedDataAgentHTTPResources(t, mem)

	createReq := httptest.NewRequest(http.MethodPost, "/v2/data-agent/jobs", strings.NewReader(`{
		"job_type":"extract_metadata",
		"resource_ids":["file_agent_http_a"],
		"project_id":"agent-study",
		"input_selector":{"format":"nifti"}
	}`))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("X-Ultra-User-Id", "agent-user")
	createReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusAccepted {
		t.Fatalf("create data-agent job status = %d body=%s, want 202", createRec.Code, createRec.Body.String())
	}
	var created dataAgentJobResponse
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created data-agent job: %v", err)
	}

	claim := httptest.NewRequest(http.MethodPost, "/v2/data-agent/jobs/"+created.Job.JobID+"/lease", strings.NewReader(`{"worker_id":"data-agent-worker-a","ttl_seconds":60}`))
	claim.Header.Set("Content-Type", "application/json")
	claim.Header.Set("X-Ultra-User-Id", "agent-user")
	claim.Header.Set("X-Ultra-Org-Id", "agent-org")
	claimRec := httptest.NewRecorder()
	router.ServeHTTP(claimRec, claim)
	if claimRec.Code != http.StatusOK {
		t.Fatalf("claim data-agent lease status = %d body=%s, want 200", claimRec.Code, claimRec.Body.String())
	}
	var lease domain.DataAgentJobLeaseRecord
	if err := json.Unmarshal(claimRec.Body.Bytes(), &lease); err != nil {
		t.Fatalf("decode data-agent lease: %v body=%s", err, claimRec.Body.String())
	}
	if lease.JobID != created.Job.JobID || lease.WorkerID != "data-agent-worker-a" || lease.LeaseToken == "" {
		t.Fatalf("lease = %+v, want worker-a token for created job", lease)
	}

	competing := httptest.NewRequest(http.MethodPost, "/v2/data-agent/jobs/"+created.Job.JobID+"/lease", strings.NewReader(`{"worker_id":"data-agent-worker-b","ttl_seconds":60}`))
	competing.Header.Set("Content-Type", "application/json")
	competing.Header.Set("X-Ultra-User-Id", "agent-user")
	competing.Header.Set("X-Ultra-Org-Id", "agent-org")
	competingRec := httptest.NewRecorder()
	router.ServeHTTP(competingRec, competing)
	if competingRec.Code != http.StatusConflict {
		t.Fatalf("competing data-agent lease status = %d body=%s, want 409", competingRec.Code, competingRec.Body.String())
	}

	getReq := httptest.NewRequest(http.MethodGet, "/v2/data-agent/jobs/"+created.Job.JobID, nil)
	getReq.Header.Set("X-Ultra-User-Id", "agent-user")
	getReq.Header.Set("X-Ultra-Org-Id", "agent-org")
	getRec := httptest.NewRecorder()
	router.ServeHTTP(getRec, getReq)
	if getRec.Code != http.StatusOK {
		t.Fatalf("get leased data-agent job status = %d body=%s, want 200", getRec.Code, getRec.Body.String())
	}
	var loaded dataAgentJobResponse
	if err := json.Unmarshal(getRec.Body.Bytes(), &loaded); err != nil {
		t.Fatalf("decode loaded data-agent job: %v", err)
	}
	if loaded.Job.Status != "running" || len(loaded.Events) != 2 || loaded.Events[1].EventType != "data_agent.job.leased" {
		t.Fatalf("loaded leased job = %+v events=%+v, want running job with lease audit event", loaded.Job, loaded.Events)
	}

	renewBody := `{"lease_token":"` + lease.LeaseToken + `","ttl_seconds":120}`
	renew := httptest.NewRequest(http.MethodPatch, "/v2/data-agent/jobs/"+created.Job.JobID+"/lease", strings.NewReader(renewBody))
	renew.Header.Set("Content-Type", "application/json")
	renew.Header.Set("X-Ultra-User-Id", "agent-user")
	renew.Header.Set("X-Ultra-Org-Id", "agent-org")
	renewRec := httptest.NewRecorder()
	router.ServeHTTP(renewRec, renew)
	if renewRec.Code != http.StatusOK {
		t.Fatalf("renew data-agent lease status = %d body=%s, want 200", renewRec.Code, renewRec.Body.String())
	}

	releaseBody := `{"lease_token":"` + lease.LeaseToken + `"}`
	release := httptest.NewRequest(http.MethodDelete, "/v2/data-agent/jobs/"+created.Job.JobID+"/lease", strings.NewReader(releaseBody))
	release.Header.Set("Content-Type", "application/json")
	release.Header.Set("X-Ultra-User-Id", "agent-user")
	release.Header.Set("X-Ultra-Org-Id", "agent-org")
	releaseRec := httptest.NewRecorder()
	router.ServeHTTP(releaseRec, release)
	if releaseRec.Code != http.StatusOK {
		t.Fatalf("release data-agent lease status = %d body=%s, want 200", releaseRec.Code, releaseRec.Body.String())
	}

	reclaim := httptest.NewRequest(http.MethodPost, "/v2/data-agent/jobs/"+created.Job.JobID+"/lease", strings.NewReader(`{"worker_id":"data-agent-worker-b","ttl_seconds":60}`))
	reclaim.Header.Set("Content-Type", "application/json")
	reclaim.Header.Set("X-Ultra-User-Id", "agent-user")
	reclaim.Header.Set("X-Ultra-Org-Id", "agent-org")
	reclaimRec := httptest.NewRecorder()
	router.ServeHTTP(reclaimRec, reclaim)
	if reclaimRec.Code != http.StatusOK {
		t.Fatalf("reclaim data-agent lease status = %d body=%s, want 200", reclaimRec.Code, reclaimRec.Body.String())
	}
}

func TestV2ResourceDeleteIsSoftAndRestoreReactivatesCatalogRow(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	fileID := writeTestUploadFile(t, uploadRoot, "restore-me.png", testPNGBytes(t, 2, 2))

	migrateReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20", nil)
	migrateReq.Header.Set("X-Ultra-User-Id", "test-user")
	migrateReq.Header.Set("X-Ultra-Org-Id", "test-org")
	migrateRec := httptest.NewRecorder()
	router.ServeHTTP(migrateRec, migrateReq)
	if migrateRec.Code != http.StatusOK {
		t.Fatalf("initial list status = %d body=%s", migrateRec.Code, migrateRec.Body.String())
	}

	matches, err := filepath.Glob(filepath.Join(uploadRoot, fileID+"__*"))
	if err != nil || len(matches) != 1 {
		t.Fatalf("uploaded fixture files = %v err=%v, want one file", matches, err)
	}
	deleteReq := httptest.NewRequest(http.MethodDelete, "/v2/resources/"+fileID, nil)
	deleteReq.Header.Set("X-Ultra-User-Id", "test-user")
	deleteReq.Header.Set("X-Ultra-Org-Id", "test-org")
	deleteRec := httptest.NewRecorder()
	router.ServeHTTP(deleteRec, deleteReq)
	if deleteRec.Code != http.StatusOK {
		t.Fatalf("delete status = %d body=%s", deleteRec.Code, deleteRec.Body.String())
	}
	if _, err := os.Stat(matches[0]); err != nil {
		t.Fatalf("soft delete removed the physical blob: %v", err)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20", nil)
	listReq.Header.Set("X-Ultra-User-Id", "test-user")
	listReq.Header.Set("X-Ultra-Org-Id", "test-org")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list after delete status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var listResponse resourcesResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &listResponse); err != nil {
		t.Fatalf("decode list after delete: %v", err)
	}
	if listResponse.Count != 0 || len(listResponse.Resources) != 0 {
		t.Fatalf("resources after delete = %+v, want hidden soft-deleted resource", listResponse)
	}

	restoreReq := httptest.NewRequest(http.MethodPost, "/v2/resources/"+fileID+"/restore", nil)
	restoreReq.Header.Set("X-Ultra-User-Id", "test-user")
	restoreReq.Header.Set("X-Ultra-Org-Id", "test-org")
	restoreRec := httptest.NewRecorder()
	router.ServeHTTP(restoreRec, restoreReq)
	if restoreRec.Code != http.StatusOK {
		t.Fatalf("restore status = %d body=%s", restoreRec.Code, restoreRec.Body.String())
	}
	listAgainRec := httptest.NewRecorder()
	router.ServeHTTP(listAgainRec, listReq)
	if listAgainRec.Code != http.StatusOK {
		t.Fatalf("list after restore status = %d body=%s", listAgainRec.Code, listAgainRec.Body.String())
	}
	var restoredResponse resourcesResponse
	if err := json.Unmarshal(listAgainRec.Body.Bytes(), &restoredResponse); err != nil {
		t.Fatalf("decode list after restore: %v", err)
	}
	if restoredResponse.Count != 1 || restoredResponse.Resources[0].FileID != fileID {
		t.Fatalf("resources after restore = %+v, want restored resource", restoredResponse)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/events?limit=10", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "test-user")
	eventsReq.Header.Set("X-Ultra-Org-Id", "test-org")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("events status = %d body=%s", eventsRec.Code, eventsRec.Body.String())
	}
	var eventsResponse resourceEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &eventsResponse); err != nil {
		t.Fatalf("decode resource events: %v", err)
	}
	if eventsResponse.ResourceID != fileID || eventsResponse.Count < 3 {
		t.Fatalf("events response = %+v, want migrated/deleted/restored audit events", eventsResponse)
	}
	seenEvents := map[string]bool{}
	for _, event := range eventsResponse.Events {
		seenEvents[event.EventType] = true
	}
	for _, want := range []string{"resource.migrated", "resource.deleted", "resource.restored"} {
		if !seenEvents[want] {
			t.Fatalf("resource events = %+v, missing %s", eventsResponse.Events, want)
		}
	}

	foreignEventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+fileID+"/events", nil)
	foreignEventsReq.Header.Set("X-Ultra-User-Id", "other-user")
	foreignEventsReq.Header.Set("X-Ultra-Org-Id", "test-org")
	foreignEventsRec := httptest.NewRecorder()
	router.ServeHTTP(foreignEventsRec, foreignEventsReq)
	if foreignEventsRec.Code != http.StatusNotFound {
		t.Fatalf("foreign events status = %d body=%s, want 404", foreignEventsRec.Code, foreignEventsRec.Body.String())
	}
}

func TestV2ResourceBulkDeleteIsAtomicAndAudited(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	now := time.Now().UTC()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "bulk_delete_a",
			OriginalName: "bulk-delete-a.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    64,
			SHA256:       "sha-bulk-delete-a",
			SourceType:   "upload",
			ResourceKind: "file",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "bulk_delete_b",
			OriginalName: "bulk-delete-b.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    96,
			SHA256:       "sha-bulk-delete-b",
			SourceType:   "upload",
			ResourceKind: "file",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
		{
			ResourceID:   "bulk_delete_foreign",
			OriginalName: "foreign.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SHA256:       "sha-bulk-delete-foreign",
			SourceType:   "upload",
			ResourceKind: "file",
			OwnerUserID:  "carol",
			OwnerOrgID:   "org-c",
			Status:       "active",
			CreatedAt:    now.Add(2 * time.Second),
			UpdatedAt:    now.Add(2 * time.Second),
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}

	atomicReq := httptest.NewRequest(http.MethodPost, "/v2/resources/delete/bulk", strings.NewReader(`{
		"resource_ids":["bulk_delete_a","bulk_delete_foreign"]
	}`))
	atomicReq.Header.Set("Content-Type", "application/json")
	atomicReq.Header.Set("X-Ultra-User-Id", "alice")
	atomicReq.Header.Set("X-Ultra-Org-Id", "org-a")
	atomicRec := httptest.NewRecorder()
	router.ServeHTTP(atomicRec, atomicReq)
	if atomicRec.Code != http.StatusNotFound {
		t.Fatalf("atomic delete status = %d body=%s, want 404", atomicRec.Code, atomicRec.Body.String())
	}
	if _, err := mem.GetResourceForUser(context.Background(), "bulk_delete_a", "alice", "org-a"); err != nil {
		t.Fatalf("owned resource was mutated after failed atomic bulk delete: %v", err)
	}

	deleteReq := httptest.NewRequest(http.MethodPost, "/v2/resources/delete/bulk", strings.NewReader(`{
		"resource_ids":["bulk_delete_a","bulk_delete_b","bulk_delete_a"]
	}`))
	deleteReq.Header.Set("Content-Type", "application/json")
	deleteReq.Header.Set("X-Ultra-User-Id", "alice")
	deleteReq.Header.Set("X-Ultra-Org-Id", "org-a")
	deleteRec := httptest.NewRecorder()
	router.ServeHTTP(deleteRec, deleteReq)
	if deleteRec.Code != http.StatusOK {
		t.Fatalf("bulk delete status = %d body=%s, want 200", deleteRec.Code, deleteRec.Body.String())
	}
	var deleted bulkLifecycleResourcesResponse
	if err := json.Unmarshal(deleteRec.Body.Bytes(), &deleted); err != nil {
		t.Fatalf("decode bulk delete response: %v", err)
	}
	if deleted.Count != 2 || len(deleted.Resources) != 2 || len(deleted.Events) != 2 {
		t.Fatalf("bulk delete response = %+v, want two resources and events", deleted)
	}
	for _, resource := range deleted.Resources {
		if resource.Status != "deleted" {
			t.Fatalf("deleted resource %s status = %q, want deleted", resource.FileID, resource.Status)
		}
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20", nil)
	listReq.Header.Set("X-Ultra-User-Id", "alice")
	listReq.Header.Set("X-Ultra-Org-Id", "org-a")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list after bulk delete status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var listed resourcesResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &listed); err != nil {
		t.Fatalf("decode list after bulk delete: %v", err)
	}
	if listed.Count != 0 || len(listed.Resources) != 0 {
		t.Fatalf("active resources after bulk delete = %+v, want none", listed)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resource-events?event_type=resource.deleted&limit=20", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "alice")
	eventsReq.Header.Set("X-Ultra-Org-Id", "org-a")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("resource event list status = %d body=%s", eventsRec.Code, eventsRec.Body.String())
	}
	var events resourceEventListResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode resource events: %v", err)
	}
	if events.Count != 2 || events.TotalCount != 2 {
		t.Fatalf("bulk delete events = %+v, want two deleted events", events)
	}
}

func TestV2ResourceBulkRestoreIsAtomicAndAudited(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	now := time.Now().UTC()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "bulk_restore_a",
			OriginalName: "bulk-restore-a.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    64,
			SHA256:       "sha-bulk-restore-a",
			SourceType:   "upload",
			ResourceKind: "file",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "bulk_restore_b",
			OriginalName: "bulk-restore-b.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    96,
			SHA256:       "sha-bulk-restore-b",
			SourceType:   "upload",
			ResourceKind: "file",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
		{
			ResourceID:   "bulk_restore_foreign",
			OriginalName: "foreign-restore.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SHA256:       "sha-bulk-restore-foreign",
			SourceType:   "upload",
			ResourceKind: "file",
			OwnerUserID:  "carol",
			OwnerOrgID:   "org-c",
			Status:       "active",
			CreatedAt:    now.Add(2 * time.Second),
			UpdatedAt:    now.Add(2 * time.Second),
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}
	for _, resourceID := range []string{"bulk_restore_a", "bulk_restore_b"} {
		if _, err := mem.SoftDeleteResourceForUser(context.Background(), resourceID, "alice", "org-a", now.Add(time.Minute)); err != nil {
			t.Fatalf("SoftDeleteResourceForUser(%s): %v", resourceID, err)
		}
	}
	if _, err := mem.SoftDeleteResourceForUser(context.Background(), "bulk_restore_foreign", "carol", "org-c", now.Add(time.Minute)); err != nil {
		t.Fatalf("SoftDeleteResourceForUser(foreign): %v", err)
	}

	atomicReq := httptest.NewRequest(http.MethodPost, "/v2/resources/restore/bulk", strings.NewReader(`{
		"resource_ids":["bulk_restore_a","bulk_restore_foreign"]
	}`))
	atomicReq.Header.Set("Content-Type", "application/json")
	atomicReq.Header.Set("X-Ultra-User-Id", "alice")
	atomicReq.Header.Set("X-Ultra-Org-Id", "org-a")
	atomicRec := httptest.NewRecorder()
	router.ServeHTTP(atomicRec, atomicReq)
	if atomicRec.Code != http.StatusNotFound {
		t.Fatalf("atomic restore status = %d body=%s, want 404", atomicRec.Code, atomicRec.Body.String())
	}
	deletedAfterFailedRestore, err := mem.ListResourcesForUser(context.Background(), domain.ResourceListInput{
		UserID: "alice",
		OrgID:  "org-a",
		Status: "deleted",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("ListResourcesForUser deleted after failed restore: %v", err)
	}
	if deletedAfterFailedRestore.TotalCount != 2 {
		t.Fatalf("deleted resources after failed restore = %+v, want both owned resources untouched", deletedAfterFailedRestore)
	}

	restoreReq := httptest.NewRequest(http.MethodPost, "/v2/resources/restore/bulk", strings.NewReader(`{
		"resource_ids":["bulk_restore_a","bulk_restore_b","bulk_restore_a"]
	}`))
	restoreReq.Header.Set("Content-Type", "application/json")
	restoreReq.Header.Set("X-Ultra-User-Id", "alice")
	restoreReq.Header.Set("X-Ultra-Org-Id", "org-a")
	restoreRec := httptest.NewRecorder()
	router.ServeHTTP(restoreRec, restoreReq)
	if restoreRec.Code != http.StatusOK {
		t.Fatalf("bulk restore status = %d body=%s, want 200", restoreRec.Code, restoreRec.Body.String())
	}
	var restored bulkLifecycleResourcesResponse
	if err := json.Unmarshal(restoreRec.Body.Bytes(), &restored); err != nil {
		t.Fatalf("decode bulk restore response: %v", err)
	}
	if restored.Count != 2 || len(restored.Resources) != 2 || len(restored.Events) != 2 {
		t.Fatalf("bulk restore response = %+v, want two resources and events", restored)
	}
	for _, resource := range restored.Resources {
		if resource.Status != "active" {
			t.Fatalf("restored resource %s status = %q, want active", resource.FileID, resource.Status)
		}
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resource-events?event_type=resource.restored&limit=20", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "alice")
	eventsReq.Header.Set("X-Ultra-Org-Id", "org-a")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("resource restored event list status = %d body=%s", eventsRec.Code, eventsRec.Body.String())
	}
	var events resourceEventListResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode resource restored events: %v", err)
	}
	if events.Count != 2 || events.TotalCount != 2 {
		t.Fatalf("bulk restore events = %+v, want two restored events", events)
	}
	for _, event := range events.Events {
		if event.EventType != "resource.restored" {
			t.Fatalf("bulk restore event = %+v, want resource.restored", event)
		}
	}
}

func TestV2ResourcesListDeletedStatusIsOwnerOnlyAndRestorable(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:      mem,
		UploadRoot: uploadRoot,
	})
	now := time.Now().UTC()
	for _, resource := range []domain.UpsertResourceInput{
		{
			ResourceID:   "deleted_alice_resource",
			OriginalName: "deleted-alice.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    128,
			SHA256:       "sha-deleted-alice",
			SourceType:   "upload",
			ResourceKind: "file",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now,
			UpdatedAt:    now,
		},
		{
			ResourceID:   "active_alice_resource",
			OriginalName: "active-alice.nii.gz",
			ContentType:  "application/gzip",
			SizeBytes:    256,
			SHA256:       "sha-active-alice",
			SourceType:   "upload",
			ResourceKind: "file",
			OwnerUserID:  "alice",
			OwnerOrgID:   "org-a",
			Status:       "active",
			CreatedAt:    now.Add(time.Second),
			UpdatedAt:    now.Add(time.Second),
		},
	} {
		if _, err := mem.UpsertResource(context.Background(), resource); err != nil {
			t.Fatalf("UpsertResource(%s): %v", resource.ResourceID, err)
		}
	}
	if _, err := mem.CreateResourceShareGrant(context.Background(), domain.CreateResourceShareGrantInput{
		ResourceID:      "deleted_alice_resource",
		OwnerUserID:     "alice",
		OwnerOrgID:      "org-a",
		GranteeUserID:   "bob",
		GranteeOrgID:    "org-b",
		Role:            "read",
		Status:          "active",
		CreatedByUserID: "alice",
		CreatedAt:       now,
	}); err != nil {
		t.Fatalf("CreateResourceShareGrant: %v", err)
	}
	if _, err := mem.SoftDeleteResourceForUser(context.Background(), "deleted_alice_resource", "alice", "org-a", now.Add(time.Minute)); err != nil {
		t.Fatalf("SoftDeleteResourceForUser: %v", err)
	}

	activeReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20", nil)
	activeReq.Header.Set("X-Ultra-User-Id", "alice")
	activeReq.Header.Set("X-Ultra-Org-Id", "org-a")
	activeRec := httptest.NewRecorder()
	router.ServeHTTP(activeRec, activeReq)
	if activeRec.Code != http.StatusOK {
		t.Fatalf("active list status = %d body=%s", activeRec.Code, activeRec.Body.String())
	}
	var active resourcesResponse
	if err := json.Unmarshal(activeRec.Body.Bytes(), &active); err != nil {
		t.Fatalf("decode active resources: %v", err)
	}
	if active.Count != 1 || active.Resources[0].FileID != "active_alice_resource" {
		t.Fatalf("active resources = %+v, want only active resource", active)
	}

	deletedReq := httptest.NewRequest(http.MethodGet, "/v2/resources?status=deleted&limit=20", nil)
	deletedReq.Header.Set("X-Ultra-User-Id", "alice")
	deletedReq.Header.Set("X-Ultra-Org-Id", "org-a")
	deletedRec := httptest.NewRecorder()
	router.ServeHTTP(deletedRec, deletedReq)
	if deletedRec.Code != http.StatusOK {
		t.Fatalf("deleted list status = %d body=%s", deletedRec.Code, deletedRec.Body.String())
	}
	var deleted resourcesResponse
	if err := json.Unmarshal(deletedRec.Body.Bytes(), &deleted); err != nil {
		t.Fatalf("decode deleted resources: %v", err)
	}
	if deleted.Count != 1 || deleted.Resources[0].FileID != "deleted_alice_resource" || deleted.Resources[0].Status != "deleted" {
		t.Fatalf("deleted resources = %+v, want owner-visible deleted resource", deleted)
	}

	bobDeletedReq := httptest.NewRequest(http.MethodGet, "/v2/resources?status=deleted&limit=20", nil)
	bobDeletedReq.Header.Set("X-Ultra-User-Id", "bob")
	bobDeletedReq.Header.Set("X-Ultra-Org-Id", "org-b")
	bobDeletedRec := httptest.NewRecorder()
	router.ServeHTTP(bobDeletedRec, bobDeletedReq)
	if bobDeletedRec.Code != http.StatusOK {
		t.Fatalf("bob deleted list status = %d body=%s", bobDeletedRec.Code, bobDeletedRec.Body.String())
	}
	var bobDeleted resourcesResponse
	if err := json.Unmarshal(bobDeletedRec.Body.Bytes(), &bobDeleted); err != nil {
		t.Fatalf("decode bob deleted resources: %v", err)
	}
	if bobDeleted.Count != 0 || len(bobDeleted.Resources) != 0 {
		t.Fatalf("bob deleted resources = %+v, want no deleted resource leak through stale share", bobDeleted)
	}

	restoreReq := httptest.NewRequest(http.MethodPost, "/v2/resources/deleted_alice_resource/restore", nil)
	restoreReq.Header.Set("X-Ultra-User-Id", "alice")
	restoreReq.Header.Set("X-Ultra-Org-Id", "org-a")
	restoreRec := httptest.NewRecorder()
	router.ServeHTTP(restoreRec, restoreReq)
	if restoreRec.Code != http.StatusOK {
		t.Fatalf("restore status = %d body=%s", restoreRec.Code, restoreRec.Body.String())
	}
	var restored resourceResponse
	if err := json.Unmarshal(restoreRec.Body.Bytes(), &restored); err != nil {
		t.Fatalf("decode restored resource: %v", err)
	}
	if restored.Resource.FileID != "deleted_alice_resource" || restored.Resource.Status != "active" {
		t.Fatalf("restored resource = %+v, want active restored resource", restored.Resource)
	}
}

func TestOmeTiffUploadViewerKeepsScientificMetadata(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
	})
	fileID := writeTestUploadFile(
		t,
		uploadRoot,
		"large-specimen.ome.tiff",
		testOmeTIFFStackBytes(t, 2, 1, 2, 3, []string{"DAPI", "EGFP", "Brightfield"}, []uint16{
			10, 20, // z0 c0
			30, 40, // z0 c1
			50, 60, // z0 c2
			70, 80, // z1 c0
			90, 100, // z1 c1
			600, 900, // z1 c2
		}),
	)
	matches, err := filepath.Glob(filepath.Join(uploadRoot, fileID+"__*"))
	if err != nil || len(matches) != 1 {
		t.Fatalf("uploaded OME-TIFF matches = %v err=%v, want one file", matches, err)
	}
	record, err := uploadResourceFromPath(uploadRoot, matches[0])
	if err != nil {
		t.Fatalf("parse uploaded OME-TIFF resource: %v", err)
	}
	omeDescription, err := tiffImageDescription(matches[0])
	prefixLength := len(omeDescription)
	if prefixLength > 120 {
		prefixLength = 120
	}
	if err != nil || !strings.Contains(omeDescription, `<Pixels`) {
		t.Fatalf("OME-TIFF fixture description err=%v prefix=%q, want OME Pixels metadata", err, omeDescription[:prefixLength])
	}
	descriptor := uploadImageDescriptorForPath(matches[0], record.ContentType)
	if descriptor.OME == nil {
		t.Fatalf("OME-TIFF descriptor = %+v, want parsed OME metadata", descriptor)
	}
	if record.ContentType != "image/tiff" || record.ResourceKind != "image" {
		t.Fatalf("parsed OME-TIFF record = %+v, want image/tiff image", record)
	}
	resources, err := listUploadResources(uploadRoot)
	if err != nil || len(resources) != 1 {
		t.Fatalf("listUploadResources = %v err=%v, want one resource", resources, err)
	}
	principal := requestPrincipal{UserID: "test-user", OrgID: "test-org", Role: "researcher"}
	if !resourceVisibleToPrincipal(resources[0], principal) {
		t.Fatalf("resource %+v should be visible to fixture principal %+v", resources[0], principal)
	}
	if !resourceMatchesQuery(resources[0], "ome") {
		t.Fatalf("resource %+v should match q=ome", resources[0])
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20&kind=image&q=ome", nil)
	listReq.Header.Set("X-Ultra-User-Id", "test-user")
	listReq.Header.Set("X-Ultra-Org-Id", "test-org")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list resources status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var listResponse struct {
		Count     int              `json:"count"`
		Resources []resourceRecord `json:"resources"`
	}
	if err := json.Unmarshal(listRec.Body.Bytes(), &listResponse); err != nil {
		t.Fatalf("decode resources response: %v", err)
	}
	if listResponse.Count != 1 || len(listResponse.Resources) != 1 {
		t.Fatalf("resources = %+v, want one OME-TIFF image", listResponse)
	}
	if listResponse.Resources[0].ResourceKind != "image" || listResponse.Resources[0].ContentType != "image/tiff" {
		t.Fatalf("OME-TIFF resource classification = %+v, want image/tiff image", listResponse.Resources[0])
	}

	viewerReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
	viewerReq.Header.Set("X-Ultra-User-Id", "test-user")
	viewerReq.Header.Set("X-Ultra-Org-Id", "test-org")
	viewerRec := httptest.NewRecorder()
	router.ServeHTTP(viewerRec, viewerReq)
	if viewerRec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", viewerRec.Code, viewerRec.Body.String())
	}
	var viewerResponse struct {
		Kind           string `json:"kind"`
		Modality       string `json:"modality"`
		BackendMode    string `json:"backend_mode"`
		DimsOrder      string `json:"dims_order"`
		IsMultichannel bool   `json:"is_multichannel"`
		AxisSizes      struct {
			X int `json:"X"`
			Y int `json:"Y"`
			Z int `json:"Z"`
			C int `json:"C"`
			T int `json:"T"`
		} `json:"axis_sizes"`
		Metadata struct {
			Reader      string   `json:"reader"`
			DimsOrder   string   `json:"dims_order"`
			ArrayShape  []int    `json:"array_shape"`
			ArrayDType  string   `json:"array_dtype"`
			Warnings    []string `json:"warnings"`
			ContentType string   `json:"content_type"`
		} `json:"metadata"`
		Viewer struct {
			Status           string   `json:"status"`
			Available        []string `json:"available_surfaces"`
			AssetPreparation struct {
				Status          string `json:"status"`
				NativeSupported bool   `json:"native_supported"`
				TilePyramid     string `json:"tile_pyramid"`
			} `json:"asset_preparation"`
		} `json:"viewer"`
	}
	if err := json.Unmarshal(viewerRec.Body.Bytes(), &viewerResponse); err != nil {
		t.Fatalf("decode viewer response: %v", err)
	}
	if viewerResponse.Kind != "image" || viewerResponse.Modality != "microscopy" {
		t.Fatalf("viewer identity = %+v, want microscopy image", viewerResponse)
	}
	if viewerResponse.AxisSizes.X != 2 || viewerResponse.AxisSizes.Y != 1 {
		t.Fatalf("viewer axis sizes = %+v, want real TIFF dimensions", viewerResponse.AxisSizes)
	}
	if viewerResponse.AxisSizes.C != 3 || viewerResponse.AxisSizes.Z != 2 || viewerResponse.AxisSizes.T != 1 {
		t.Fatalf("viewer axis sizes = %+v, want parsed OME stack dimensions", viewerResponse.AxisSizes)
	}
	if !viewerResponse.IsMultichannel {
		t.Fatalf("is_multichannel = false, want true for OME channel stack")
	}
	if viewerResponse.DimsOrder != "ZCYX" || viewerResponse.Metadata.DimsOrder != "ZCYX" {
		t.Fatalf("viewer dims order = %q metadata=%q, want ZCYX", viewerResponse.DimsOrder, viewerResponse.Metadata.DimsOrder)
	}
	if viewerResponse.Metadata.ArrayDType != "uint16" {
		t.Fatalf("array_dtype = %q, want uint16", viewerResponse.Metadata.ArrayDType)
	}
	if !slicesEqualInts(viewerResponse.Metadata.ArrayShape, []int{2, 3, 1, 2}) {
		t.Fatalf("array_shape = %v, want [2 3 1 2]", viewerResponse.Metadata.ArrayShape)
	}
	if viewerResponse.Metadata.Reader != "ome-tiff+xml+go-image" {
		t.Fatalf("viewer metadata = %+v, want TIFF reader metadata", viewerResponse.Metadata)
	}
	if viewerResponse.Viewer.Status != "ready" || viewerResponse.Viewer.AssetPreparation.TilePyramid != "deferred" {
		t.Fatalf("viewer prep = %+v, want OME stack slice delivery", viewerResponse.Viewer)
	}
	if viewerResponse.Viewer.AssetPreparation.NativeSupported {
		t.Fatalf("native_supported = true, want false until full multiscale OME pyramid serving is prepared")
	}
	if len(viewerResponse.Viewer.Available) != 2 || viewerResponse.Viewer.Available[0] != "2d" || viewerResponse.Viewer.Available[1] != "metadata" {
		t.Fatalf("available surfaces = %v, want 2d + metadata for OME-TIFF stack", viewerResponse.Viewer.Available)
	}

	displayReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/slice?axis=z&z=1&channels=2&window_min=600&window_max=900", nil)
	displayReq.Header.Set("X-Ultra-User-Id", "test-user")
	displayReq.Header.Set("X-Ultra-Org-Id", "test-org")
	displayRec := httptest.NewRecorder()
	router.ServeHTTP(displayRec, displayReq)
	if displayRec.Code != http.StatusOK {
		t.Fatalf("display status = %d body=%s", displayRec.Code, displayRec.Body.String())
	}
	if got := displayRec.Header().Get("Content-Type"); got != "image/png" {
		t.Fatalf("display content type = %q, want image/png", got)
	}
	displayImage, format, err := image.Decode(bytes.NewReader(displayRec.Body.Bytes()))
	if err != nil {
		t.Fatalf("decode display PNG: %v", err)
	}
	if format != "png" || displayImage.Bounds().Dx() != 2 || displayImage.Bounds().Dy() != 1 {
		t.Fatalf("display config = %dx%d %s, want 2x1 png", displayImage.Bounds().Dx(), displayImage.Bounds().Dy(), format)
	}
	first := color.GrayModel.Convert(displayImage.At(0, 0)).(color.Gray).Y
	second := color.GrayModel.Convert(displayImage.At(1, 0)).(color.Gray).Y
	if first > 5 || second < 250 {
		t.Fatalf("selected OME plane pixels = %d,%d, want windowed z1 c2 low/high contrast", first, second)
	}
}

func TestOmeTiffUploadViewerReadsLargeImageDescription(t *testing.T) {
	uploadRoot := t.TempDir()
	padding := strings.Repeat("x", 17*1024*1024)
	fileID := writeTestUploadFile(
		t,
		uploadRoot,
		"large-description.ome.tiff",
		testOmeTIFFStackBytesWithDescriptionPadding(t, 1, 1, 1, 2, []string{"DAPI", "Brightfield"}, []uint16{
			10, 900,
		}, padding),
	)
	matches, err := filepath.Glob(filepath.Join(uploadRoot, fileID+"__*"))
	if err != nil || len(matches) != 1 {
		t.Fatalf("uploaded large OME-TIFF matches = %v err=%v, want one file", matches, err)
	}

	descriptor := uploadImageDescriptorForPath(matches[0], "image/tiff")
	if descriptor.OME == nil {
		t.Fatalf("OME-TIFF descriptor = %+v, want parsed metadata from large ImageDescription", descriptor)
	}
	if descriptor.ChannelCount != 2 || descriptor.Depth != 1 || descriptor.Width != 1 || descriptor.Height != 1 {
		t.Fatalf("OME-TIFF descriptor dimensions = %+v, want 1x1 z1 c2", descriptor)
	}
	if got := descriptor.OME.Channels[1].Name; got != "Brightfield" {
		t.Fatalf("OME channel[1] = %q, want Brightfield", got)
	}
}

func TestNiftiUploadViewerServesScalarVolume(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
	})
	values := []uint16{
		10, 20, 30, 40,
		50, 60, 70, 80,
		90, 100, 110, 120,
	}
	fileID := writeTestUploadFile(t, uploadRoot, "brain-volume.nii", testNifti1Uint16Bytes(t, 2, 2, 3, values))

	viewerReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
	viewerReq.Header.Set("X-Ultra-User-Id", "test-user")
	viewerReq.Header.Set("X-Ultra-Org-Id", "test-org")
	viewerRec := httptest.NewRecorder()
	router.ServeHTTP(viewerRec, viewerReq)
	if viewerRec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", viewerRec.Code, viewerRec.Body.String())
	}
	var viewerResponse struct {
		Kind        string `json:"kind"`
		Modality    string `json:"modality"`
		BackendMode string `json:"backend_mode"`
		DimsOrder   string `json:"dims_order"`
		IsVolume    bool   `json:"is_volume"`
		AxisSizes   struct {
			X int `json:"X"`
			Y int `json:"Y"`
			Z int `json:"Z"`
			C int `json:"C"`
			T int `json:"T"`
		} `json:"axis_sizes"`
		ServiceURLs struct {
			ScalarVolume string `json:"scalar_volume"`
			Slice        string `json:"slice"`
			Preview      string `json:"preview"`
		} `json:"service_urls"`
		DisplayDefaults struct {
			FusionMethod string `json:"fusion_method"`
		} `json:"display_defaults"`
		Metadata struct {
			Reader     string  `json:"reader"`
			ArrayShape []int   `json:"array_shape"`
			ArrayDType string  `json:"array_dtype"`
			ArrayMin   float64 `json:"array_min"`
			ArrayMax   float64 `json:"array_max"`
		} `json:"metadata"`
		Viewer struct {
			DefaultSurface   string   `json:"default_surface"`
			Available        []string `json:"available_surfaces"`
			VolumeMode       string   `json:"volume_mode"`
			RenderPolicy     string   `json:"render_policy"`
			DeliveryMode     string   `json:"delivery_mode"`
			FirstPaintMode   string   `json:"first_paint_mode"`
			TexturePolicy    string   `json:"texture_policy"`
			AssetPreparation struct {
				NativeSupported      bool   `json:"native_supported"`
				VolumeRepresentation string `json:"volume_representation"`
			} `json:"asset_preparation"`
		} `json:"viewer"`
	}
	if err := json.Unmarshal(viewerRec.Body.Bytes(), &viewerResponse); err != nil {
		t.Fatalf("decode viewer response: %v", err)
	}
	if viewerResponse.Kind != "image" || viewerResponse.Modality != "medical" || viewerResponse.BackendMode != "scalar" {
		t.Fatalf("viewer identity = %+v, want scalar medical image", viewerResponse)
	}
	if !viewerResponse.IsVolume || viewerResponse.DimsOrder != "ZYX" {
		t.Fatalf("volume flags = is_volume:%v dims:%q, want true ZYX", viewerResponse.IsVolume, viewerResponse.DimsOrder)
	}
	if viewerResponse.AxisSizes.X != 2 || viewerResponse.AxisSizes.Y != 2 || viewerResponse.AxisSizes.Z != 3 || viewerResponse.AxisSizes.C != 1 || viewerResponse.AxisSizes.T != 1 {
		t.Fatalf("axis sizes = %+v, want X=2 Y=2 Z=3 C=1 T=1", viewerResponse.AxisSizes)
	}
	if viewerResponse.ServiceURLs.ScalarVolume != "/v2/uploads/"+fileID+"/scalar-volume" {
		t.Fatalf("scalar volume URL = %q, want V2 scalar-volume URL", viewerResponse.ServiceURLs.ScalarVolume)
	}
	if viewerResponse.DisplayDefaults.FusionMethod != "a" {
		t.Fatalf("fusion default = %q, want composite volume projection", viewerResponse.DisplayDefaults.FusionMethod)
	}
	if viewerResponse.Metadata.Reader != "nifti-1" || viewerResponse.Metadata.ArrayDType != "uint16" {
		t.Fatalf("metadata = %+v, want NIfTI uint16 metadata", viewerResponse.Metadata)
	}
	if len(viewerResponse.Metadata.ArrayShape) != 3 || viewerResponse.Metadata.ArrayShape[0] != 3 || viewerResponse.Metadata.ArrayShape[1] != 2 || viewerResponse.Metadata.ArrayShape[2] != 2 {
		t.Fatalf("array_shape = %v, want [3 2 2]", viewerResponse.Metadata.ArrayShape)
	}
	if viewerResponse.Metadata.ArrayMin != 10 || viewerResponse.Metadata.ArrayMax != 120 {
		t.Fatalf("array range = %f..%f, want 10..120", viewerResponse.Metadata.ArrayMin, viewerResponse.Metadata.ArrayMax)
	}
	if viewerResponse.Viewer.DefaultSurface != "volume" || viewerResponse.Viewer.VolumeMode != "scalar" || viewerResponse.Viewer.DeliveryMode != "scalar" {
		t.Fatalf("viewer mode = %+v, want scalar volume default", viewerResponse.Viewer)
	}
	if len(viewerResponse.Viewer.Available) != 4 || viewerResponse.Viewer.Available[0] != "2d" || viewerResponse.Viewer.Available[1] != "mpr" || viewerResponse.Viewer.Available[2] != "volume" || viewerResponse.Viewer.Available[3] != "metadata" {
		t.Fatalf("available surfaces = %v, want 2d/mpr/volume/metadata", viewerResponse.Viewer.Available)
	}
	if !viewerResponse.Viewer.AssetPreparation.NativeSupported || viewerResponse.Viewer.AssetPreparation.VolumeRepresentation != "scalar" {
		t.Fatalf("asset preparation = %+v, want native scalar volume", viewerResponse.Viewer.AssetPreparation)
	}

	volumeReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/scalar-volume", nil)
	volumeReq.Header.Set("X-Ultra-User-Id", "test-user")
	volumeReq.Header.Set("X-Ultra-Org-Id", "test-org")
	volumeRec := httptest.NewRecorder()
	router.ServeHTTP(volumeRec, volumeReq)
	if volumeRec.Code != http.StatusOK {
		t.Fatalf("scalar-volume status = %d body=%s", volumeRec.Code, volumeRec.Body.String())
	}
	if got := volumeRec.Header().Get("Content-Type"); got != "application/octet-stream" {
		t.Fatalf("scalar-volume content type = %q, want application/octet-stream", got)
	}
	header := volumeRec.Header()
	if header.Get("x-volume-width") != "2" || header.Get("x-volume-height") != "2" || header.Get("x-volume-depth") != "3" {
		t.Fatalf("volume dimension headers = width:%q height:%q depth:%q", header.Get("x-volume-width"), header.Get("x-volume-height"), header.Get("x-volume-depth"))
	}
	if header.Get("x-volume-dtype") != "uint16" || header.Get("x-volume-bytes-per-voxel") != "2" {
		t.Fatalf("volume dtype headers = dtype:%q bytes:%q", header.Get("x-volume-dtype"), header.Get("x-volume-bytes-per-voxel"))
	}
	if header.Get("x-volume-raw-min") != "10" || header.Get("x-volume-raw-max") != "120" {
		t.Fatalf("volume range headers = min:%q max:%q", header.Get("x-volume-raw-min"), header.Get("x-volume-raw-max"))
	}
	if len(volumeRec.Body.Bytes()) != len(values)*2 {
		t.Fatalf("scalar-volume byte length = %d, want %d", len(volumeRec.Body.Bytes()), len(values)*2)
	}
	for index, want := range values {
		got := binary.LittleEndian.Uint16(volumeRec.Body.Bytes()[index*2 : index*2+2])
		if got != want {
			t.Fatalf("voxel[%d] = %d, want %d", index, got, want)
		}
	}
}

func TestNiftiUploadViewerServesFloat32ScalarVolume(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
	})
	values := []float32{-1.5, 0.5, 2.5, 4.5}
	fileID := writeTestUploadFile(t, uploadRoot, "float-brain.nii", testNifti1Float32Bytes(t, 2, 2, 1, values))

	viewerReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
	viewerReq.Header.Set("X-Ultra-User-Id", "test-user")
	viewerReq.Header.Set("X-Ultra-Org-Id", "test-org")
	viewerRec := httptest.NewRecorder()
	router.ServeHTTP(viewerRec, viewerReq)
	if viewerRec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", viewerRec.Code, viewerRec.Body.String())
	}
	var viewerResponse struct {
		Metadata struct {
			ArrayDType string  `json:"array_dtype"`
			ArrayMin   float64 `json:"array_min"`
			ArrayMax   float64 `json:"array_max"`
		} `json:"metadata"`
	}
	if err := json.Unmarshal(viewerRec.Body.Bytes(), &viewerResponse); err != nil {
		t.Fatalf("decode viewer response: %v", err)
	}
	if viewerResponse.Metadata.ArrayDType != "float32" {
		t.Fatalf("array_dtype = %q, want float32", viewerResponse.Metadata.ArrayDType)
	}
	if viewerResponse.Metadata.ArrayMin != -1.5 || viewerResponse.Metadata.ArrayMax != 4.5 {
		t.Fatalf("array range = %f..%f, want -1.5..4.5", viewerResponse.Metadata.ArrayMin, viewerResponse.Metadata.ArrayMax)
	}

	volumeReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/scalar-volume", nil)
	volumeReq.Header.Set("X-Ultra-User-Id", "test-user")
	volumeReq.Header.Set("X-Ultra-Org-Id", "test-org")
	volumeRec := httptest.NewRecorder()
	router.ServeHTTP(volumeRec, volumeReq)
	if volumeRec.Code != http.StatusOK {
		t.Fatalf("scalar-volume status = %d body=%s", volumeRec.Code, volumeRec.Body.String())
	}
	header := volumeRec.Header()
	if header.Get("x-volume-dtype") != "float32" || header.Get("x-volume-bytes-per-voxel") != "4" {
		t.Fatalf("volume dtype headers = dtype:%q bytes:%q", header.Get("x-volume-dtype"), header.Get("x-volume-bytes-per-voxel"))
	}
	if header.Get("x-volume-raw-min") != "-1.5" || header.Get("x-volume-raw-max") != "4.5" {
		t.Fatalf("volume range headers = min:%q max:%q", header.Get("x-volume-raw-min"), header.Get("x-volume-raw-max"))
	}
	if len(volumeRec.Body.Bytes()) != len(values)*4 {
		t.Fatalf("scalar-volume byte length = %d, want %d", len(volumeRec.Body.Bytes()), len(values)*4)
	}
	for index, want := range values {
		got := math.Float32frombits(binary.LittleEndian.Uint32(volumeRec.Body.Bytes()[index*4 : index*4+4]))
		if got != want {
			t.Fatalf("voxel[%d] = %f, want %f", index, got, want)
		}
	}

	sliceReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/slice?axis=z&z=0&window_min=-1.5&window_max=4.5", nil)
	sliceReq.Header.Set("X-Ultra-User-Id", "test-user")
	sliceReq.Header.Set("X-Ultra-Org-Id", "test-org")
	sliceRec := httptest.NewRecorder()
	router.ServeHTTP(sliceRec, sliceReq)
	if sliceRec.Code != http.StatusOK {
		t.Fatalf("slice status = %d body=%s", sliceRec.Code, sliceRec.Body.String())
	}
	img, err := png.Decode(bytes.NewReader(sliceRec.Body.Bytes()))
	if err != nil {
		t.Fatalf("decode slice png: %v", err)
	}
	gotFirst := color.GrayModel.Convert(img.At(0, 0)).(color.Gray).Y
	gotLast := color.GrayModel.Convert(img.At(1, 1)).(color.Gray).Y
	if gotFirst != 0 || gotLast != 255 {
		t.Fatalf("windowed float32 slice endpoints = %d..%d, want 0..255", gotFirst, gotLast)
	}
}

func TestNiftiUploadViewerUsesCTLikeVolumeDefaults(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
	})
	values := []float32{-1024, -700, 40, 1200}
	fileID := writeTestUploadFile(t, uploadRoot, "ct-head.nii", testNifti1Float32Bytes(t, 2, 2, 1, values))

	viewerReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
	viewerReq.Header.Set("X-Ultra-User-Id", "test-user")
	viewerReq.Header.Set("X-Ultra-Org-Id", "test-org")
	viewerRec := httptest.NewRecorder()
	router.ServeHTTP(viewerRec, viewerReq)
	if viewerRec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", viewerRec.Code, viewerRec.Body.String())
	}
	var viewerResponse struct {
		DisplayDefaults struct {
			Enhancement            string  `json:"enhancement"`
			FusionMethod           string  `json:"fusion_method"`
			ScalarColormap         string  `json:"scalar_colormap"`
			VolumeSignalFloor      float64 `json:"volume_signal_floor"`
			VolumeDensity          float64 `json:"volume_density"`
			VolumeLighting         bool    `json:"volume_lighting"`
			VolumeLightingStrength float64 `json:"volume_lighting_strength"`
			VolumeViewPreset       string  `json:"volume_view_preset"`
			VolumeCameraMode       string  `json:"volume_camera_mode"`
		} `json:"display_defaults"`
	}
	if err := json.Unmarshal(viewerRec.Body.Bytes(), &viewerResponse); err != nil {
		t.Fatalf("decode viewer response: %v", err)
	}
	defaults := viewerResponse.DisplayDefaults
	if defaults.Enhancement != "hounsfield:40.000:80.000" {
		t.Fatalf("enhancement = %q, want CT brain window 40/80", defaults.Enhancement)
	}
	if defaults.FusionMethod != "a" || defaults.ScalarColormap != "grayscale" {
		t.Fatalf("projection/color defaults = %+v, want composite grayscale", defaults)
	}
	if defaults.VolumeSignalFloor != 0.12 || defaults.VolumeDensity != 1.75 {
		t.Fatalf("transfer defaults = floor:%f density:%f, want 0.12/1.75", defaults.VolumeSignalFloor, defaults.VolumeDensity)
	}
	if !defaults.VolumeLighting || defaults.VolumeLightingStrength != 0.72 {
		t.Fatalf("lighting defaults = enabled:%v strength:%f, want enabled 0.72", defaults.VolumeLighting, defaults.VolumeLightingStrength)
	}
	if defaults.VolumeViewPreset != "iso" || defaults.VolumeCameraMode != "orthographic" {
		t.Fatalf("camera defaults = preset:%q mode:%q, want iso orthographic", defaults.VolumeViewPreset, defaults.VolumeCameraMode)
	}
}

func TestNiftiUploadViewerServesSignedInt16ScalarVolume(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
	})
	values := []int16{-1024, -256, 0, 512}
	fileID := writeTestUploadFile(t, uploadRoot, "signed-brain.nii", testNifti1Int16Bytes(t, 2, 2, 1, values))

	viewerReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
	viewerReq.Header.Set("X-Ultra-User-Id", "test-user")
	viewerReq.Header.Set("X-Ultra-Org-Id", "test-org")
	viewerRec := httptest.NewRecorder()
	router.ServeHTTP(viewerRec, viewerReq)
	if viewerRec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", viewerRec.Code, viewerRec.Body.String())
	}
	var viewerResponse struct {
		Metadata struct {
			ArrayDType string  `json:"array_dtype"`
			ArrayMin   float64 `json:"array_min"`
			ArrayMax   float64 `json:"array_max"`
		} `json:"metadata"`
	}
	if err := json.Unmarshal(viewerRec.Body.Bytes(), &viewerResponse); err != nil {
		t.Fatalf("decode viewer response: %v", err)
	}
	if viewerResponse.Metadata.ArrayDType != "int16" {
		t.Fatalf("array_dtype = %q, want int16", viewerResponse.Metadata.ArrayDType)
	}
	if viewerResponse.Metadata.ArrayMin != -1024 || viewerResponse.Metadata.ArrayMax != 512 {
		t.Fatalf("array range = %f..%f, want -1024..512", viewerResponse.Metadata.ArrayMin, viewerResponse.Metadata.ArrayMax)
	}

	volumeReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/scalar-volume", nil)
	volumeReq.Header.Set("X-Ultra-User-Id", "test-user")
	volumeReq.Header.Set("X-Ultra-Org-Id", "test-org")
	volumeRec := httptest.NewRecorder()
	router.ServeHTTP(volumeRec, volumeReq)
	if volumeRec.Code != http.StatusOK {
		t.Fatalf("scalar-volume status = %d body=%s", volumeRec.Code, volumeRec.Body.String())
	}
	header := volumeRec.Header()
	if header.Get("x-volume-dtype") != "int16" || header.Get("x-volume-bytes-per-voxel") != "2" {
		t.Fatalf("volume dtype headers = dtype:%q bytes:%q", header.Get("x-volume-dtype"), header.Get("x-volume-bytes-per-voxel"))
	}
	if header.Get("x-volume-raw-min") != "-1024" || header.Get("x-volume-raw-max") != "512" {
		t.Fatalf("volume range headers = min:%q max:%q", header.Get("x-volume-raw-min"), header.Get("x-volume-raw-max"))
	}
	if len(volumeRec.Body.Bytes()) != len(values)*2 {
		t.Fatalf("scalar-volume byte length = %d, want %d", len(volumeRec.Body.Bytes()), len(values)*2)
	}
	for index, want := range values {
		got := int16(binary.LittleEndian.Uint16(volumeRec.Body.Bytes()[index*2 : index*2+2]))
		if got != want {
			t.Fatalf("voxel[%d] = %d, want %d", index, got, want)
		}
	}

	sliceReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/slice?axis=z&z=0&window_min=-1024&window_max=512", nil)
	sliceReq.Header.Set("X-Ultra-User-Id", "test-user")
	sliceReq.Header.Set("X-Ultra-Org-Id", "test-org")
	sliceRec := httptest.NewRecorder()
	router.ServeHTTP(sliceRec, sliceReq)
	if sliceRec.Code != http.StatusOK {
		t.Fatalf("slice status = %d body=%s", sliceRec.Code, sliceRec.Body.String())
	}
	img, err := png.Decode(bytes.NewReader(sliceRec.Body.Bytes()))
	if err != nil {
		t.Fatalf("decode slice png: %v", err)
	}
	gotFirst := color.GrayModel.Convert(img.At(0, 0)).(color.Gray).Y
	gotLast := color.GrayModel.Convert(img.At(1, 1)).(color.Gray).Y
	if gotFirst != 0 || gotLast != 255 {
		t.Fatalf("windowed int16 slice endpoints = %d..%d, want 0..255", gotFirst, gotLast)
	}
}

func TestNiftiUploadHistogramServesSignedInt16ScalarVolume(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
	})
	values := []int16{-1024, -256, 0, 512}
	fileID := writeTestUploadFile(t, uploadRoot, "signed-histogram.nii", testNifti1Int16Bytes(t, 2, 2, 1, values))

	histogramReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/histogram?bins=8", nil)
	histogramReq.Header.Set("X-Ultra-User-Id", "test-user")
	histogramReq.Header.Set("X-Ultra-Org-Id", "test-org")
	histogramRec := httptest.NewRecorder()
	router.ServeHTTP(histogramRec, histogramReq)
	if histogramRec.Code != http.StatusOK {
		t.Fatalf("histogram status = %d body=%s", histogramRec.Code, histogramRec.Body.String())
	}
	var response struct {
		FileID    string `json:"file_id"`
		Bins      int    `json:"bins"`
		DType     string `json:"dtype"`
		Source    string `json:"source"`
		Histogram struct {
			Bins           []int     `json:"bins"`
			Edges          []float64 `json:"edges"`
			Min            float64   `json:"min"`
			Max            float64   `json:"max"`
			ChannelIndices []int     `json:"channel_indices"`
			TimeIndex      int       `json:"time_index"`
		} `json:"histogram"`
		SampleCount int `json:"sample_count"`
	}
	if err := json.Unmarshal(histogramRec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode histogram response: %v", err)
	}
	if response.FileID != fileID || response.Bins != 8 || response.Source != "scalar-volume" {
		t.Fatalf("histogram identity = %+v, want scalar 8-bin histogram for upload", response)
	}
	if response.DType != "int16" {
		t.Fatalf("dtype = %q, want int16", response.DType)
	}
	if response.Histogram.Min != -1024 || response.Histogram.Max != 512 {
		t.Fatalf("histogram range = %f..%f, want signed source range", response.Histogram.Min, response.Histogram.Max)
	}
	if response.SampleCount != 4 {
		t.Fatalf("sample_count = %d, want four voxels", response.SampleCount)
	}
	if len(response.Histogram.Bins) != 8 || len(response.Histogram.Edges) != 9 {
		t.Fatalf("histogram sizes = bins %d edges %d, want 8 and 9", len(response.Histogram.Bins), len(response.Histogram.Edges))
	}
	total := 0
	nonzero := 0
	for _, count := range response.Histogram.Bins {
		total += count
		if count > 0 {
			nonzero++
		}
	}
	if total != 4 || nonzero != 4 {
		t.Fatalf("histogram counts = %v, want four signed samples in four bins", response.Histogram.Bins)
	}
	if len(response.Histogram.ChannelIndices) != 1 || response.Histogram.ChannelIndices[0] != 0 || response.Histogram.TimeIndex != 0 {
		t.Fatalf("histogram axes = channels %v t=%d, want channel 0 time 0", response.Histogram.ChannelIndices, response.Histogram.TimeIndex)
	}
}

func TestNiftiUploadViewerServesSelectedScalarTimepoint(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
	})
	// A 2-volume series: the NIfTI 4th dimension is time, so this is two
	// timepoints (not two channels). Each timepoint is one contiguous slab.
	timepointValues := []uint16{
		10, 20, 30, 40,
		100, 200, 300, 400,
	}
	fileID := writeTestUploadFile(t, uploadRoot, "two-timepoint-volume.nii", testNifti1Uint16TimeBytes(t, 2, 1, 2, 2, timepointValues))

	viewerReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
	viewerReq.Header.Set("X-Ultra-User-Id", "test-user")
	viewerReq.Header.Set("X-Ultra-Org-Id", "test-org")
	viewerRec := httptest.NewRecorder()
	router.ServeHTTP(viewerRec, viewerReq)
	if viewerRec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", viewerRec.Code, viewerRec.Body.String())
	}
	var viewerResponse struct {
		DimsOrder      string `json:"dims_order"`
		IsTimeseries   bool   `json:"is_timeseries"`
		IsMultichannel bool   `json:"is_multichannel"`
		AxisSizes      struct {
			X int `json:"X"`
			Y int `json:"Y"`
			Z int `json:"Z"`
			C int `json:"C"`
			T int `json:"T"`
		} `json:"axis_sizes"`
		SelectedIndices struct {
			T int `json:"T"`
		} `json:"selected_indices"`
		Metadata struct {
			ArrayShape []int    `json:"array_shape"`
			Warnings   []string `json:"warnings"`
		} `json:"metadata"`
		Viewer struct {
			DisplayCapabilities []string `json:"display_capabilities"`
		} `json:"viewer"`
	}
	if err := json.Unmarshal(viewerRec.Body.Bytes(), &viewerResponse); err != nil {
		t.Fatalf("decode viewer response: %v", err)
	}
	if viewerResponse.DimsOrder != "TZYX" || !viewerResponse.IsTimeseries || viewerResponse.IsMultichannel {
		t.Fatalf("viewer dims/timeseries/multichannel = %q/%v/%v, want TZYX true false", viewerResponse.DimsOrder, viewerResponse.IsTimeseries, viewerResponse.IsMultichannel)
	}
	if viewerResponse.AxisSizes.X != 2 || viewerResponse.AxisSizes.Y != 1 || viewerResponse.AxisSizes.Z != 2 || viewerResponse.AxisSizes.T != 2 || viewerResponse.AxisSizes.C != 1 {
		t.Fatalf("axis sizes = %+v, want X=2 Y=1 Z=2 T=2 C=1", viewerResponse.AxisSizes)
	}
	if !sliceContains(viewerResponse.Viewer.DisplayCapabilities, "time_navigation") {
		t.Fatalf("display capabilities = %v, want time_navigation", viewerResponse.Viewer.DisplayCapabilities)
	}
	if len(viewerResponse.Metadata.ArrayShape) != 4 || viewerResponse.Metadata.ArrayShape[0] != 2 || viewerResponse.Metadata.ArrayShape[1] != 2 || viewerResponse.Metadata.ArrayShape[2] != 1 || viewerResponse.Metadata.ArrayShape[3] != 2 {
		t.Fatalf("array_shape = %v, want [2 2 1 2] (T,Z,Y,X)", viewerResponse.Metadata.ArrayShape)
	}

	volumeReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/scalar-volume?t=1", nil)
	volumeReq.Header.Set("X-Ultra-User-Id", "test-user")
	volumeReq.Header.Set("X-Ultra-Org-Id", "test-org")
	volumeRec := httptest.NewRecorder()
	router.ServeHTTP(volumeRec, volumeReq)
	if volumeRec.Code != http.StatusOK {
		t.Fatalf("scalar-volume status = %d body=%s", volumeRec.Code, volumeRec.Body.String())
	}
	header := volumeRec.Header()
	if header.Get("x-volume-time") != "1" || header.Get("x-volume-time-count") != "2" {
		t.Fatalf("time headers = idx:%q count:%q, want 1/2", header.Get("x-volume-time"), header.Get("x-volume-time-count"))
	}
	if header.Get("x-volume-raw-min") != "100" || header.Get("x-volume-raw-max") != "400" {
		t.Fatalf("timepoint range headers = min:%q max:%q, want 100..400", header.Get("x-volume-raw-min"), header.Get("x-volume-raw-max"))
	}
	selectedValues := []uint16{100, 200, 300, 400}
	if len(volumeRec.Body.Bytes()) != len(selectedValues)*2 {
		t.Fatalf("scalar-volume byte length = %d, want %d", len(volumeRec.Body.Bytes()), len(selectedValues)*2)
	}
	for index, want := range selectedValues {
		got := binary.LittleEndian.Uint16(volumeRec.Body.Bytes()[index*2 : index*2+2])
		if got != want {
			t.Fatalf("selected timepoint voxel[%d] = %d, want %d", index, got, want)
		}
	}

	sliceReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/slice?axis=z&z=1&t=1&window_min=100&window_max=400", nil)
	sliceReq.Header.Set("X-Ultra-User-Id", "test-user")
	sliceReq.Header.Set("X-Ultra-Org-Id", "test-org")
	sliceRec := httptest.NewRecorder()
	router.ServeHTTP(sliceRec, sliceReq)
	if sliceRec.Code != http.StatusOK {
		t.Fatalf("slice status = %d body=%s", sliceRec.Code, sliceRec.Body.String())
	}
	img, err := png.Decode(bytes.NewReader(sliceRec.Body.Bytes()))
	if err != nil {
		t.Fatalf("decode slice png: %v", err)
	}
	gotFirst := color.GrayModel.Convert(img.At(0, 0)).(color.Gray).Y
	gotLast := color.GrayModel.Convert(img.At(1, 0)).(color.Gray).Y
	if gotFirst != 170 || gotLast != 255 {
		t.Fatalf("selected timepoint slice pixels = %d,%d, want 170,255", gotFirst, gotLast)
	}
}

func TestNiftiUploadViewerServesSelectedScalarChannel(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
	})
	// A genuine multi-component volume uses the NIfTI 5th dimension for channels
	// (dim[5]); the 4th stays singleton (T=1).
	channelValues := []uint16{
		10, 20, 30, 40,
		100, 200, 300, 400,
	}
	fileID := writeTestUploadFile(t, uploadRoot, "two-channel-volume.nii", testNifti1Uint16ChannelBytes(t, 2, 1, 2, 2, channelValues))

	viewerReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/viewer", nil)
	viewerReq.Header.Set("X-Ultra-User-Id", "test-user")
	viewerReq.Header.Set("X-Ultra-Org-Id", "test-org")
	viewerRec := httptest.NewRecorder()
	router.ServeHTTP(viewerRec, viewerReq)
	if viewerRec.Code != http.StatusOK {
		t.Fatalf("viewer status = %d body=%s", viewerRec.Code, viewerRec.Body.String())
	}
	var viewerResponse struct {
		DimsOrder      string `json:"dims_order"`
		IsMultichannel bool   `json:"is_multichannel"`
		IsTimeseries   bool   `json:"is_timeseries"`
		AxisSizes      struct {
			X int `json:"X"`
			Y int `json:"Y"`
			Z int `json:"Z"`
			C int `json:"C"`
			T int `json:"T"`
		} `json:"axis_sizes"`
		DisplayDefaults struct {
			Channels      []int    `json:"channels"`
			ChannelColors []string `json:"channel_colors"`
			VolumeChannel int      `json:"volume_channel"`
		} `json:"display_defaults"`
		Metadata struct {
			ArrayShape []int `json:"array_shape"`
		} `json:"metadata"`
		Viewer struct {
			DisplayCapabilities []string `json:"display_capabilities"`
		} `json:"viewer"`
	}
	if err := json.Unmarshal(viewerRec.Body.Bytes(), &viewerResponse); err != nil {
		t.Fatalf("decode viewer response: %v", err)
	}
	if viewerResponse.DimsOrder != "CZYX" || !viewerResponse.IsMultichannel || viewerResponse.IsTimeseries {
		t.Fatalf("viewer dims/multichannel/timeseries = %q/%v/%v, want CZYX true false", viewerResponse.DimsOrder, viewerResponse.IsMultichannel, viewerResponse.IsTimeseries)
	}
	if viewerResponse.AxisSizes.X != 2 || viewerResponse.AxisSizes.Y != 1 || viewerResponse.AxisSizes.Z != 2 || viewerResponse.AxisSizes.C != 2 || viewerResponse.AxisSizes.T != 1 {
		t.Fatalf("axis sizes = %+v, want X=2 Y=1 Z=2 C=2 T=1", viewerResponse.AxisSizes)
	}
	if !slicesEqualInts(viewerResponse.DisplayDefaults.Channels, []int{0}) || viewerResponse.DisplayDefaults.VolumeChannel != 0 || len(viewerResponse.DisplayDefaults.ChannelColors) != 2 {
		t.Fatalf("display defaults = %+v, want single visible channel 0 with two colors", viewerResponse.DisplayDefaults)
	}
	if !sliceContains(viewerResponse.Viewer.DisplayCapabilities, "channel_visibility") {
		t.Fatalf("display capabilities = %v, want channel_visibility", viewerResponse.Viewer.DisplayCapabilities)
	}
	if len(viewerResponse.Metadata.ArrayShape) != 4 || viewerResponse.Metadata.ArrayShape[0] != 2 || viewerResponse.Metadata.ArrayShape[1] != 2 || viewerResponse.Metadata.ArrayShape[2] != 1 || viewerResponse.Metadata.ArrayShape[3] != 2 {
		t.Fatalf("array_shape = %v, want [2 2 1 2] (C,Z,Y,X)", viewerResponse.Metadata.ArrayShape)
	}

	volumeReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/scalar-volume?channel=1", nil)
	volumeReq.Header.Set("X-Ultra-User-Id", "test-user")
	volumeReq.Header.Set("X-Ultra-Org-Id", "test-org")
	volumeRec := httptest.NewRecorder()
	router.ServeHTTP(volumeRec, volumeReq)
	if volumeRec.Code != http.StatusOK {
		t.Fatalf("scalar-volume status = %d body=%s", volumeRec.Code, volumeRec.Body.String())
	}
	header := volumeRec.Header()
	if header.Get("x-volume-channel") != "1" {
		t.Fatalf("x-volume-channel = %q, want 1", header.Get("x-volume-channel"))
	}
	if header.Get("x-volume-raw-min") != "100" || header.Get("x-volume-raw-max") != "400" {
		t.Fatalf("channel range headers = min:%q max:%q, want 100..400", header.Get("x-volume-raw-min"), header.Get("x-volume-raw-max"))
	}
	selectedValues := []uint16{100, 200, 300, 400}
	if len(volumeRec.Body.Bytes()) != len(selectedValues)*2 {
		t.Fatalf("scalar-volume byte length = %d, want %d", len(volumeRec.Body.Bytes()), len(selectedValues)*2)
	}
	for index, want := range selectedValues {
		got := binary.LittleEndian.Uint16(volumeRec.Body.Bytes()[index*2 : index*2+2])
		if got != want {
			t.Fatalf("selected channel voxel[%d] = %d, want %d", index, got, want)
		}
	}
}

func TestOmeTiffDisplayAutoWindowsSixteenBitPreview(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
	})
	fileID := writeTestUploadFile(
		t,
		uploadRoot,
		"dim-specimen.ome.tiff",
		testGray16TIFFBytes(t, 4, 1, []uint16{1000, 1001, 1002, 1003}),
	)

	displayReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/display", nil)
	displayReq.Header.Set("X-Ultra-User-Id", "test-user")
	displayReq.Header.Set("X-Ultra-Org-Id", "test-org")
	displayRec := httptest.NewRecorder()
	router.ServeHTTP(displayRec, displayReq)
	if displayRec.Code != http.StatusOK {
		t.Fatalf("display status = %d body=%s", displayRec.Code, displayRec.Body.String())
	}
	displayImage, format, err := image.Decode(bytes.NewReader(displayRec.Body.Bytes()))
	if err != nil {
		t.Fatalf("decode display PNG: %v", err)
	}
	if format != "png" {
		t.Fatalf("display format = %q, want png", format)
	}
	dark := color.GrayModel.Convert(displayImage.At(0, 0)).(color.Gray).Y
	bright := color.GrayModel.Convert(displayImage.At(3, 0)).(color.Gray).Y
	if dark > 5 || bright < 250 {
		t.Fatalf("display gray range = dark %d bright %d, want auto-windowed 8-bit contrast", dark, bright)
	}
}

func TestOmeTiffDisplayHonorsRequestedWindow(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
	})
	fileID := writeTestUploadFile(
		t,
		uploadRoot,
		"windowed-specimen.ome.tiff",
		testGray16TIFFBytes(t, 4, 1, []uint16{1000, 1001, 1002, 1003}),
	)

	displayReq := httptest.NewRequest(
		http.MethodGet,
		"/v2/uploads/"+fileID+"/display?enhancement=hounsfield:1001.5:1",
		nil,
	)
	displayReq.Header.Set("X-Ultra-User-Id", "test-user")
	displayReq.Header.Set("X-Ultra-Org-Id", "test-org")
	displayRec := httptest.NewRecorder()
	router.ServeHTTP(displayRec, displayReq)
	if displayRec.Code != http.StatusOK {
		t.Fatalf("display status = %d body=%s", displayRec.Code, displayRec.Body.String())
	}
	displayImage, format, err := image.Decode(bytes.NewReader(displayRec.Body.Bytes()))
	if err != nil {
		t.Fatalf("decode display PNG: %v", err)
	}
	if format != "png" {
		t.Fatalf("display format = %q, want png", format)
	}
	belowWindow := color.GrayModel.Convert(displayImage.At(1, 0)).(color.Gray).Y
	aboveWindow := color.GrayModel.Convert(displayImage.At(2, 0)).(color.Gray).Y
	if belowWindow > 5 || aboveWindow < 250 {
		t.Fatalf("requested window mapped pixels to %d/%d, want low/high contrast split", belowWindow, aboveWindow)
	}
}

func TestUploadedImageDisplayCanSelectRGBChannel(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
	})
	fileID := writeTestUploadFile(
		t,
		uploadRoot,
		"three-channel.png",
		testRGBPNGBytes(t, 2, 1, []color.RGBA{
			{R: 200, G: 10, B: 0, A: 255},
			{R: 0, G: 180, B: 20, A: 255},
		}),
	)

	displayReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/display?channels=1", nil)
	displayReq.Header.Set("X-Ultra-User-Id", "test-user")
	displayReq.Header.Set("X-Ultra-Org-Id", "test-org")
	displayRec := httptest.NewRecorder()
	router.ServeHTTP(displayRec, displayReq)
	if displayRec.Code != http.StatusOK {
		t.Fatalf("display status = %d body=%s", displayRec.Code, displayRec.Body.String())
	}
	displayImage, format, err := image.Decode(bytes.NewReader(displayRec.Body.Bytes()))
	if err != nil {
		t.Fatalf("decode display PNG: %v", err)
	}
	if format != "png" {
		t.Fatalf("display format = %q, want png", format)
	}
	first := color.RGBAModel.Convert(displayImage.At(0, 0)).(color.RGBA)
	second := color.RGBAModel.Convert(displayImage.At(1, 0)).(color.RGBA)
	if first.R != 10 || first.G != 10 || first.B != 10 || second.R != 180 || second.G != 180 || second.B != 180 {
		t.Fatalf("selected green channel pixels = %#v %#v, want grayscale 10 and 180", first, second)
	}
}

func TestOmeTiffUploadHistogramPreservesSixteenBitValues(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
	})
	fileID := writeTestUploadFile(
		t,
		uploadRoot,
		"histology-intensity.ome.tiff",
		testGray16TIFFBytes(t, 4, 1, []uint16{1000, 1001, 1002, 1003}),
	)

	histogramReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/histogram?bins=8", nil)
	histogramReq.Header.Set("X-Ultra-User-Id", "test-user")
	histogramReq.Header.Set("X-Ultra-Org-Id", "test-org")
	histogramRec := httptest.NewRecorder()
	router.ServeHTTP(histogramRec, histogramReq)
	if histogramRec.Code != http.StatusOK {
		t.Fatalf("histogram status = %d body=%s", histogramRec.Code, histogramRec.Body.String())
	}
	var response struct {
		FileID    string `json:"file_id"`
		Bins      int    `json:"bins"`
		DType     string `json:"dtype"`
		Source    string `json:"source"`
		Histogram struct {
			Bins           []int     `json:"bins"`
			Edges          []float64 `json:"edges"`
			Min            float64   `json:"min"`
			Max            float64   `json:"max"`
			ChannelIndices []int     `json:"channel_indices"`
			TimeIndex      int       `json:"time_index"`
		} `json:"histogram"`
		SampleCount int `json:"sample_count"`
	}
	if err := json.Unmarshal(histogramRec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode histogram response: %v", err)
	}
	if response.FileID != fileID || response.Bins != 8 || response.Source != "decoded-image" {
		t.Fatalf("histogram identity = %+v, want decoded 8-bin histogram for upload", response)
	}
	if response.DType != "uint16" {
		t.Fatalf("dtype = %q, want uint16", response.DType)
	}
	if response.Histogram.Min != 1000 || response.Histogram.Max != 1003 {
		t.Fatalf("histogram range = %f..%f, want original uint16 source range", response.Histogram.Min, response.Histogram.Max)
	}
	if response.SampleCount != 4 {
		t.Fatalf("sample_count = %d, want four pixels", response.SampleCount)
	}
	if len(response.Histogram.Bins) != 8 || len(response.Histogram.Edges) != 9 {
		t.Fatalf("histogram sizes = bins %d edges %d, want 8 and 9", len(response.Histogram.Bins), len(response.Histogram.Edges))
	}
	total := 0
	nonzero := 0
	for _, count := range response.Histogram.Bins {
		total += count
		if count > 0 {
			nonzero++
		}
	}
	if total != 4 || nonzero != 4 {
		t.Fatalf("histogram counts = %v, want four source samples in four bins", response.Histogram.Bins)
	}
	if len(response.Histogram.ChannelIndices) != 1 || response.Histogram.ChannelIndices[0] != 0 || response.Histogram.TimeIndex != 0 {
		t.Fatalf("histogram axes = channels %v t=%d, want channel 0 time 0", response.Histogram.ChannelIndices, response.Histogram.TimeIndex)
	}
}

func TestV2ResourcesUseBisqueSessionCookiePrincipal(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	credentialStore := NewBisqueCredentialStore()
	router := NewRouter(ServerDeps{
		Version:           "test-version",
		UploadRoot:        uploadRoot,
		BisqueCredentials: credentialStore,
	})

	loginReq := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"amil"}`))
	loginReq.Header.Set("Content-Type", "application/json")
	loginRec := httptest.NewRecorder()
	router.ServeHTTP(loginRec, loginReq)
	if loginRec.Code != http.StatusOK {
		t.Fatalf("login status = %d body=%s", loginRec.Code, loginRec.Body.String())
	}
	sessionCookie := loginRec.Result().Cookies()[0]

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "field-image.png")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	pngBytes := testPNGBytes(t, 4, 3)
	if _, err := part.Write(pngBytes); err != nil {
		t.Fatalf("write multipart file: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}

	uploadReq := httptest.NewRequest(http.MethodPost, "/v2/uploads", &body)
	uploadReq.Header.Set("Content-Type", writer.FormDataContentType())
	uploadReq.AddCookie(sessionCookie)
	uploadRec := httptest.NewRecorder()
	router.ServeHTTP(uploadRec, uploadReq)
	if uploadRec.Code != http.StatusOK {
		t.Fatalf("upload status = %d body=%s", uploadRec.Code, uploadRec.Body.String())
	}
	var uploadResponse uploadFilesResponse
	if err := json.Unmarshal(uploadRec.Body.Bytes(), &uploadResponse); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if len(uploadResponse.Uploaded) != 1 {
		t.Fatalf("uploaded = %+v, want one file", uploadResponse.Uploaded)
	}
	uploaded := uploadResponse.Uploaded[0]
	if uploaded.Principal.UserID != "bisque:amil" {
		t.Fatalf("uploaded principal user = %q, want bisque:amil", uploaded.Principal.UserID)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20", nil)
	listReq.AddCookie(sessionCookie)
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var listResponse resourcesResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &listResponse); err != nil {
		t.Fatalf("decode list response: %v", err)
	}
	if listResponse.Count != 1 || len(listResponse.Resources) != 1 || listResponse.Resources[0].FileID != uploaded.FileID {
		t.Fatalf("resources = %+v, want only BisQue user's upload", listResponse)
	}

	otherReq := httptest.NewRequest(http.MethodGet, "/v2/resources?limit=20", nil)
	otherReq.AddCookie(&http.Cookie{Name: "ultra_dev_auth", Value: "bisque:other"})
	otherRec := httptest.NewRecorder()
	router.ServeHTTP(otherRec, otherReq)
	if otherRec.Code != http.StatusOK {
		t.Fatalf("other list status = %d body=%s", otherRec.Code, otherRec.Body.String())
	}
	var otherResponse resourcesResponse
	if err := json.Unmarshal(otherRec.Body.Bytes(), &otherResponse); err != nil {
		t.Fatalf("decode other list response: %v", err)
	}
	if otherResponse.Count != 0 || len(otherResponse.Resources) != 0 {
		t.Fatalf("other resources = %+v, want no BisQue user's upload", otherResponse)
	}

	displayReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+uploaded.FileID+"/display", nil)
	displayReq.AddCookie(sessionCookie)
	displayRec := httptest.NewRecorder()
	router.ServeHTTP(displayRec, displayReq)
	if displayRec.Code != http.StatusOK || !bytes.Equal(displayRec.Body.Bytes(), pngBytes) {
		t.Fatalf("display status = %d bytes=%d, want same-user image bytes", displayRec.Code, displayRec.Body.Len())
	}

	otherDisplayReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+uploaded.FileID+"/display", nil)
	otherDisplayReq.AddCookie(&http.Cookie{Name: "ultra_dev_auth", Value: "bisque:other"})
	otherDisplayRec := httptest.NewRecorder()
	router.ServeHTTP(otherDisplayRec, otherDisplayReq)
	if otherDisplayRec.Code != http.StatusNotFound {
		t.Fatalf("other display status = %d body=%s, want 404", otherDisplayRec.Code, otherDisplayRec.Body.String())
	}
}

func TestResourceWithoutOwnerIsOnlyVisibleToLocalDevPrincipal(t *testing.T) {
	t.Parallel()

	resource := resourceRecord{FileID: "legacy_file", OriginalName: "legacy.png"}
	if !resourceVisibleToPrincipal(resource, requestPrincipal{UserID: "local-user", OrgID: "local-org"}) {
		t.Fatalf("legacy resource should remain visible to local dev principal")
	}
	if resourceVisibleToPrincipal(resource, requestPrincipal{UserID: "bisque:amil", OrgID: "local-org"}) {
		t.Fatalf("legacy resource without owner sidecar should not be visible to linked BisQue users")
	}
}

func TestV2Sam3InteractiveSegmentationIsExplicitlyNotConfigured(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Runs:    runcontrol.NewService(store.NewMemoryStore(), eventbus.NewMemoryBus()),
		Store:   store.NewMemoryStore(),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/segment/sam3/interactive", strings.NewReader(`{"file_ids":["file_1"],"annotations":[]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("SAM3 status = %d body=%s, want 501 not configured", rec.Code, rec.Body.String())
	}
	var response map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode SAM3 response: %v", err)
	}
	if response["status"] != "not_configured" || response["service"] != "ultra-control-v2" {
		t.Fatalf("SAM3 response = %#v, want explicit V2 not-configured payload", response)
	}
}

func TestV2BisqueImportIsExplicitlyNotConfigured(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Runs:    runcontrol.NewService(store.NewMemoryStore(), eventbus.NewMemoryBus()),
		Store:   store.NewMemoryStore(),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/uploads/from-bisque", strings.NewReader(`{"resources":["https://bisque.example.org/data_service/image/1"]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("BisQue import status = %d body=%s, want 501 not configured", rec.Code, rec.Body.String())
	}
	var response map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode BisQue import response: %v", err)
	}
	if response["status"] != "not_configured" || response["service"] != "bisque" {
		t.Fatalf("BisQue import response = %#v, want explicit BisQue not-configured payload", response)
	}
}

func TestV2BisqueSearchUsesLinkedAccountCredentials(t *testing.T) {
	t.Parallel()

	var gotAuth string
	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		if r.URL.Path != "/data_service/image/" {
			t.Fatalf("BisQue path = %q, want /data_service/image/", r.URL.Path)
		}
		if r.URL.Query().Get("tag_query") != "species:prairie_dog" {
			t.Fatalf("tag_query = %q", r.URL.Query().Get("tag_query"))
		}
		w.Header().Set("Content-Type", "application/xml")
		_, _ = w.Write([]byte(`<response><image uri="` + bisque.URL + `/data_service/image/abc" name="prairie.jpg" resource_uniq="abc"><tag name="species" value="prairie_dog"/></image></response>`))
	}))
	defer bisque.Close()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			DevUsername:   "ada",
			DevPassword:   "secret",
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(`{"resource_type":"image","tag_query":"species:prairie_dog","limit":5}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("search status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	wantAuth := "Basic " + base64.StdEncoding.EncodeToString([]byte("ada:secret"))
	if gotAuth != wantAuth {
		t.Fatalf("Authorization = %q, want linked account basic auth", gotAuth)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode search response: %v", err)
	}
	if body["count"].(float64) != 1 {
		t.Fatalf("search count = %#v, want 1", body["count"])
	}
	results := body["results"].([]any)
	first := results[0].(map[string]any)
	if first["resource_uri"] != bisque.URL+"/data_service/image/abc" || first["name"] != "prairie.jpg" {
		t.Fatalf("first result = %#v", first)
	}
}

func TestV2BisqueSearchUsesBisqueResponseCountWhenAvailable(t *testing.T) {
	t.Parallel()

	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/data_service/image/" {
			t.Fatalf("BisQue path = %q, want /data_service/image/", r.URL.Path)
		}
		if r.URL.Query().Get("limit") != "1" {
			t.Fatalf("limit = %q, want 1", r.URL.Query().Get("limit"))
		}
		w.Header().Set("Content-Type", "application/xml")
		_, _ = w.Write([]byte(`<response count="42"><image uri="` + bisque.URL + `/data_service/image/abc" name="prairie.jpg" resource_uniq="abc"/></response>`))
	}))
	defer bisque.Close()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(`{"resource_type":"image","limit":1}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("search status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode search response: %v", err)
	}
	if body["count"].(float64) != 42 {
		t.Fatalf("search count = %#v, want BisQue response total 42", body["count"])
	}
	results := body["results"].([]any)
	if len(results) != 1 {
		t.Fatalf("results length = %d, want only returned page length 1", len(results))
	}
}

func TestV2BisqueSearchSkipsDataServiceListWrapper(t *testing.T) {
	t.Parallel()

	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		_, _ = w.Write([]byte(`<resource uri="` + bisque.URL + `/data_service/image/?limit=1&amp;view=short"><image uri="` + bisque.URL + `/data_service/image/abc" name="prairie.jpg" resource_uniq="abc"/></resource>`))
	}))
	defer bisque.Close()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(`{"resource_type":"image","limit":1}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("search status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode search response: %v", err)
	}
	if body["count"].(float64) != 1 {
		t.Fatalf("search count = %#v, want only child image counted", body["count"])
	}
	results := body["results"].([]any)
	first := results[0].(map[string]any)
	if first["resource_type"] != "image" || first["resource_uniq"] != "abc" {
		t.Fatalf("first result = %#v, want concrete image child", first)
	}
}

func TestV2BisqueSearchCanCountAllPagesWhenBisqueOmitsTotal(t *testing.T) {
	t.Parallel()

	var requestedOffsets []string
	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/data_service/image/" {
			t.Fatalf("BisQue path = %q, want /data_service/image/", r.URL.Path)
		}
		if r.URL.Query().Get("tag_query") != "owner:amil" {
			t.Fatalf("tag_query = %q, want owner:amil", r.URL.Query().Get("tag_query"))
		}
		if r.URL.Query().Get("wpublic") != "owner,shared" {
			t.Fatalf("wpublic = %q, want owner,shared", r.URL.Query().Get("wpublic"))
		}
		requestedOffsets = append(requestedOffsets, r.URL.Query().Get("offset"))
		w.Header().Set("Content-Type", "application/xml")
		switch r.URL.Query().Get("offset") {
		case "":
			if r.URL.Query().Get("limit") != "1" {
				t.Fatalf("preview limit = %q, want 1", r.URL.Query().Get("limit"))
			}
			_, _ = w.Write([]byte(`<resource><image uri="` + bisque.URL + `/data_service/image/first" name="first.jpg" resource_uniq="first"/></resource>`))
		case "0":
			if r.URL.Query().Get("limit") != "700" {
				t.Fatalf("count page limit = %q, want 700", r.URL.Query().Get("limit"))
			}
			_, _ = w.Write([]byte(`<resource><image uri="` + bisque.URL + `/data_service/image/1" name="one.jpg" resource_uniq="one"/><image uri="` + bisque.URL + `/data_service/image/2" name="two.jpg" resource_uniq="two"/></resource>`))
		default:
			t.Fatalf("unexpected count offset %q", r.URL.Query().Get("offset"))
		}
	}))
	defer bisque.Close()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(`{"resource_type":"image","tag_query":"owner:amil","limit":1,"count_all":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("search status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode search response: %v", err)
	}
	if body["count"].(float64) != 2 {
		t.Fatalf("search count = %#v, want counted total 2", body["count"])
	}
	results := body["results"].([]any)
	if len(results) != 1 {
		t.Fatalf("results length = %d, want preview page length 1", len(results))
	}
	if strings.Join(requestedOffsets, ",") != ",0" {
		t.Fatalf("requested offsets = %#v, want preview then count page", requestedOffsets)
	}
}

func TestV2BisqueSearchSupportsOwnerScopeRecencyAndNameExtensionFilters(t *testing.T) {
	t.Parallel()

	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/data_service/image/" {
			t.Fatalf("BisQue path = %q, want /data_service/image/", r.URL.Path)
		}
		if r.URL.Query().Get("wpublic") != "owner" {
			t.Fatalf("wpublic = %q, want owner", r.URL.Query().Get("wpublic"))
		}
		if r.URL.Query().Get("tag_order") != "@ts:desc" {
			t.Fatalf("tag_order = %q, want @ts:desc", r.URL.Query().Get("tag_order"))
		}
		if r.URL.Query().Get("query") != "EnrNE_" {
			t.Fatalf("query = %q, want EnrNE_", r.URL.Query().Get("query"))
		}
		w.Header().Set("Content-Type", "application/xml")
		_, _ = w.Write([]byte(`<resource total="3">
			<image uri="` + bisque.URL + `/data_service/image/png" name="EnrNE_recent.PNG" resource_uniq="png"/>
			<image uri="` + bisque.URL + `/data_service/image/jpg" name="EnrNE_old.jpg" resource_uniq="jpg"/>
			<image uri="` + bisque.URL + `/data_service/image/other" name="Other_recent.PNG" resource_uniq="other"/>
		</resource>`))
	}))
	defer bisque.Close()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(`{"resource_type":"image","scope":"owner","sort":"recent","name_contains":"EnrNE_","extensions":["png"],"limit":10}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("search status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode search response: %v", err)
	}
	if body["count"].(float64) != 1 {
		t.Fatalf("search count = %#v, want filtered total 1", body["count"])
	}
	results := body["results"].([]any)
	if len(results) != 1 {
		t.Fatalf("results length = %d, want 1 filtered PNG", len(results))
	}
	first := results[0].(map[string]any)
	if first["name"] != "EnrNE_recent.PNG" {
		t.Fatalf("first result = %#v, want only matching PNG", first)
	}
}

func TestV2BisqueSearchFiltersNiftiFileExtensions(t *testing.T) {
	t.Parallel()

	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/data_service/file/" {
			t.Fatalf("BisQue path = %q, want /data_service/file/", r.URL.Path)
		}
		if r.URL.Query().Get("wpublic") != "owner" {
			t.Fatalf("wpublic = %q, want owner", r.URL.Query().Get("wpublic"))
		}
		w.Header().Set("Content-Type", "application/xml")
		_, _ = w.Write([]byte(`<resource>
			<file uri="` + bisque.URL + `/data_service/file/nii" name="brain.NII" resource_uniq="nii"/>
			<file uri="` + bisque.URL + `/data_service/file/niigz" name="mask.nii.gz" resource_uniq="niigz"/>
			<file uri="` + bisque.URL + `/data_service/file/nifti" name="atlas.nifti" resource_uniq="nifti"/>
			<file uri="` + bisque.URL + `/data_service/file/txt" name="notes.txt" resource_uniq="txt"/>
		</resource>`))
	}))
	defer bisque.Close()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(`{"resource_type":"file","scope":"owner","extensions":["nii","nii.gz","nifti"],"limit":10}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("search status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode search response: %v", err)
	}
	if body["count"].(float64) != 3 {
		t.Fatalf("search count = %#v, want 3 filtered NIfTI files", body["count"])
	}
	results := body["results"].([]any)
	names := make([]string, 0, len(results))
	for _, result := range results {
		names = append(names, result.(map[string]any)["name"].(string))
	}
	if strings.Join(names, ",") != "brain.NII,mask.nii.gz,atlas.nifti" {
		t.Fatalf("result names = %v, want only NIfTI files", names)
	}
}

// ctScanFixtureXML mirrors the live BisQue view=full payload: the modality and
// numeric age tags are serialized as child <tag> elements. Ages 7 and 100 are
// the lexical-comparison traps ("7" > "50", "100" < "50").
func ctScanFixtureXML(root string) string {
	rows := []struct {
		uniq, name, modality, age string
	}{
		{"chest44", "ct_chest_age44.nii.gz", "CT", "44"},
		{"head52", "ct_head_age52.nii.gz", "CT", "52"},
		{"abd61", "ct_abdomen_age61.nii.gz", "CT", "61"},
		{"pelvis7", "ct_pelvis_age7.nii.gz", "CT", "7"},
		{"spine100", "ct_spine_age100.nii.gz", "CT", "100"},
		{"knee50", "ct_knee_age50.nii.gz", "CT", "50"},
	}
	var b strings.Builder
	b.WriteString("<resource>")
	for _, row := range rows {
		b.WriteString(`<image uri="` + root + `/data_service/` + row.uniq + `" name="` + row.name + `" resource_uniq="` + row.uniq + `">`)
		b.WriteString(`<tag name="modality" value="` + row.modality + `"/>`)
		b.WriteString(`<tag name="age" type="number" value="` + row.age + `"/>`)
		b.WriteString(`</image>`)
	}
	b.WriteString("</resource>")
	return b.String()
}

func TestV2BisqueSearchNumericMetadataFilterBeatsLexicalComparison(t *testing.T) {
	t.Parallel()

	var gotView string
	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotView = r.URL.Query().Get("view")
		if r.URL.Query().Get("tag_query") != "modality:CT" {
			t.Fatalf("tag_query = %q, want modality:CT", r.URL.Query().Get("tag_query"))
		}
		w.Header().Set("Content-Type", "application/xml")
		_, _ = w.Write([]byte(ctScanFixtureXML(bisque.URL)))
	}))
	defer bisque.Close()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	searchNames := func(payload string) []string {
		req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(payload))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("search status = %d body=%s, want 200", rec.Code, rec.Body.String())
		}
		var body map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("decode search response: %v", err)
		}
		results := body["results"].([]any)
		names := make([]string, 0, len(results))
		for _, result := range results {
			names = append(names, result.(map[string]any)["name"].(string))
		}
		return names
	}

	gt50 := searchNames(`{"resource_type":"image","tag_query":"modality:CT","metadata_filters":[{"tag":"age","op":"gt","value":"50"}],"count_all":true,"limit":25}`)
	if gotView != "full" {
		t.Fatalf("view = %q, want full so tags are serialized for client-side filtering", gotView)
	}
	// Numerically correct: 52, 61, 100. NOT 7 (lexically "7">"50") and includes
	// 100 (lexically "100"<"50"). 44 and 50 excluded.
	if strings.Join(gt50, ",") != "ct_head_age52.nii.gz,ct_abdomen_age61.nii.gz,ct_spine_age100.nii.gz" {
		t.Fatalf("age>50 results = %v, want {52,61,100}", gt50)
	}

	gte50 := searchNames(`{"resource_type":"image","tag_query":"modality:CT","metadata_filters":[{"tag":"age","op":"gte","value":"50"}],"count_all":true,"limit":25}`)
	// Server-order preserved: 52, 61, 100, 50 (knee=50 now included vs the gt case).
	if strings.Join(gte50, ",") != "ct_head_age52.nii.gz,ct_abdomen_age61.nii.gz,ct_spine_age100.nii.gz,ct_knee_age50.nii.gz" {
		t.Fatalf("age>=50 results = %v, want {52,61,100,50}", gte50)
	}

	pediatric := searchNames(`{"resource_type":"image","tag_query":"modality:CT","metadata_filters":[{"tag":"age","op":"lt","value":"18"}],"count_all":true,"limit":25}`)
	if strings.Join(pediatric, ",") != "ct_pelvis_age7.nii.gz" {
		t.Fatalf("age<18 results = %v, want only age 7", pediatric)
	}

	band := searchNames(`{"resource_type":"image","tag_query":"modality:CT","metadata_filters":[{"tag":"age","op":"gte","value":"44"},{"tag":"age","op":"lte","value":"52"}],"count_all":true,"limit":25}`)
	if strings.Join(band, ",") != "ct_chest_age44.nii.gz,ct_head_age52.nii.gz,ct_knee_age50.nii.gz" {
		t.Fatalf("44<=age<=52 results = %v, want {44,52,50}", band)
	}
}

func TestBisqueClientViewURLIsCanonicalUnescaped(t *testing.T) {
	t.Parallel()

	service := NewBisqueService(BisqueServiceConfig{
		RootURL:      "https://bisque2.ece.ucsb.edu",
		AllowedRoots: []string{"https://bisque2.ece.ucsb.edu"},
		HTTPClient:   http.DefaultClient,
	})

	got := service.clientViewURL("https://bisque2.ece.ucsb.edu/data_service/00-LLkbXPVgwiddnNQcTSPEKk")
	want := "https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-LLkbXPVgwiddnNQcTSPEKk"
	if got != want {
		t.Fatalf("clientViewURL = %q, want canonical unescaped %q", got, want)
	}
	if strings.Contains(got, "%3A") || strings.Contains(got, "%2F") {
		t.Fatalf("client view URL must not percent-encode the resource reference: %q", got)
	}

	// A relative resource URI is resolved against the configured root so the
	// viewer always receives an absolute resource reference.
	rel := service.clientViewURL("/data_service/00-REL")
	if rel != "https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-REL" {
		t.Fatalf("relative resource URI not resolved to absolute view URL: %q", rel)
	}
}

func TestV2BisqueSearchResultsCarryCanonicalViewURL(t *testing.T) {
	t.Parallel()

	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		_, _ = w.Write([]byte(`<resource><image uri="` + bisque.URL + `/data_service/00-abc" name="scan.png" resource_uniq="00-abc"/></resource>`))
	}))
	defer bisque.Close()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(`{"resource_type":"image","limit":1}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("search status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var body bisqueSearchResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode search response: %v", err)
	}
	if len(body.Results) != 1 {
		t.Fatalf("results = %+v, want one", body.Results)
	}
	want := bisque.URL + "/client_service/view?resource=" + bisque.URL + "/data_service/00-abc"
	if body.Results[0].ClientViewURL != want {
		t.Fatalf("client_view_url = %q, want canonical unescaped %q", body.Results[0].ClientViewURL, want)
	}
}

func TestV2BisqueSearchRejectsNonNumericRelationalFilter(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       "https://bisque.example.test",
			HTTPClient:    http.DefaultClient,
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(`{"resource_type":"image","metadata_filters":[{"tag":"age","op":"gt","value":"old"}]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s, want 400 for non-numeric relational filter", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "numeric value") {
		t.Fatalf("error body = %s, want numeric-value guidance", rec.Body.String())
	}
}

func TestV2BisqueSearchPrefersLinkedSessionCredentialsOverFallback(t *testing.T) {
	t.Parallel()

	var gotAuth string
	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/xml")
		if r.URL.Path == "/auth_service/session" {
			_, _ = w.Write([]byte(`<response><tag name="user" value="linked-user"/></response>`))
			return
		}
		_, _ = w.Write([]byte(`<response><image uri="` + bisque.URL + `/data_service/image/session" name="session-owned.jpg" resource_uniq="session"/></response>`))
	}))
	defer bisque.Close()

	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			DevUsername:   "fallback-user",
			DevPassword:   "fallback-secret",
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	login := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"linked-user","password":"linked-secret"}`))
	login.Header.Set("Content-Type", "application/json")
	loginRec := httptest.NewRecorder()
	router.ServeHTTP(loginRec, login)
	if loginRec.Code != http.StatusOK {
		t.Fatalf("login status = %d body=%s", loginRec.Code, loginRec.Body.String())
	}
	if len(loginRec.Result().Cookies()) == 0 {
		t.Fatalf("login did not set linked-account cookie")
	}

	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(`{"resource_type":"image","limit":1}`))
	req.Header.Set("Content-Type", "application/json")
	req.AddCookie(loginRec.Result().Cookies()[0])
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("search status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	wantAuth := "Basic " + base64.StdEncoding.EncodeToString([]byte("linked-user:linked-secret"))
	if gotAuth != wantAuth {
		t.Fatalf("Authorization = %q, want linked session credentials", gotAuth)
	}
	if strings.Contains(gotAuth, "fallback") {
		t.Fatalf("Authorization unexpectedly used fallback credentials: %q", gotAuth)
	}
}

type fakeBisqueUploadServer struct {
	mu          sync.Mutex
	uploads     []string
	datasetXML  []string
	uploadCount int
}

func newFakeBisqueUploadServer(t *testing.T) (*httptest.Server, *fakeBisqueUploadServer) {
	t.Helper()
	state := &fakeBisqueUploadServer{}
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/import/transfer":
			if err := r.ParseMultipartForm(32 << 20); err != nil {
				t.Errorf("parse multipart upload: %v", err)
			}
			file, header, err := r.FormFile("file")
			if err != nil {
				t.Errorf("missing multipart file field: %v", err)
				return
			}
			defer func() {
				_ = file.Close()
			}()
			state.mu.Lock()
			state.uploadCount++
			uniq := fmt.Sprintf("00-PUSH%d", state.uploadCount)
			state.uploads = append(state.uploads, header.Filename)
			state.mu.Unlock()
			_, _ = w.Write([]byte(`<resource type="uploaded"><image uri="` + server.URL + `/data_service/` + uniq + `" name="` + header.Filename + `" resource_uniq="` + uniq + `"/></resource>`))
		case r.Method == http.MethodPost && r.URL.Path == "/data_service/dataset":
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Errorf("read dataset body: %v", err)
			}
			state.mu.Lock()
			state.datasetXML = append(state.datasetXML, string(body))
			state.mu.Unlock()
			var dataset bisqueDatasetXML
			if err := xml.Unmarshal(body, &dataset); err != nil {
				t.Errorf("decode dataset XML: %v", err)
			}
			_, _ = w.Write([]byte(`<dataset uri="` + server.URL + `/data_service/00-DATASET" name="` + dataset.Name + `" resource_uniq="00-DATASET"/>`))
		case r.URL.Path == "/auth_service/session":
			_, _ = w.Write([]byte(`<session><tag name="user" value="amil"/></session>`))
		default:
			_, _ = w.Write([]byte(`<resource/>`))
		}
	}))
	t.Cleanup(server.Close)
	return server, state
}

func TestV2BisqueCreateDatasetPostsMemberXML(t *testing.T) {
	t.Parallel()

	bisque, state := newFakeBisqueUploadServer(t)
	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			DevUsername:   "ada",
			DevPassword:   "secret",
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	payload := `{"name":"analysis outputs","resource_uris":["` + bisque.URL + `/data_service/00-A","` + bisque.URL + `/data_service/00-B","` + bisque.URL + `/data_service/00-A"]}`
	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/datasets", strings.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("create dataset status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	if len(state.datasetXML) != 1 {
		t.Fatalf("dataset posts = %d, want 1", len(state.datasetXML))
	}
	var dataset bisqueDatasetXML
	if err := xml.Unmarshal([]byte(state.datasetXML[0]), &dataset); err != nil {
		t.Fatalf("decode dataset XML: %v", err)
	}
	if dataset.Name != "analysis outputs" {
		t.Fatalf("dataset name = %q", dataset.Name)
	}
	if len(dataset.Members) != 2 {
		t.Fatalf("dataset members = %+v, want duplicate URI removed", dataset.Members)
	}
	if dataset.Members[0].URI != bisque.URL+"/data_service/00-A" || dataset.Members[0].Type != "object" || dataset.Members[0].Index != 0 {
		t.Fatalf("first member = %+v", dataset.Members[0])
	}
	if dataset.Members[1].URI != bisque.URL+"/data_service/00-B" || dataset.Members[1].Index != 1 {
		t.Fatalf("second member = %+v", dataset.Members[1])
	}
	var response BisqueDatasetRecord
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.ResourceUniq != "00-DATASET" || response.MemberCount != 2 || response.Name != "analysis outputs" {
		t.Fatalf("dataset response = %+v", response)
	}
	if response.ClientViewURL == "" {
		t.Fatalf("dataset response missing client view URL: %+v", response)
	}
}

func TestV2BisquePushRejectsDatasetMembersOutsideAllowedRoots(t *testing.T) {
	t.Parallel()

	bisque, _ := newFakeBisqueUploadServer(t)
	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			DevUsername:   "ada",
			DevPassword:   "secret",
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	payload := `{"name":"bad","resource_uris":["https://evil.example.com/data_service/00-A"]}`
	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/datasets", strings.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("dataset status = %d body=%s, want 400 for disallowed member root", rec.Code, rec.Body.String())
	}
}

func TestV2BisquePushFolderCreatesDataset(t *testing.T) {
	t.Parallel()

	bisque, state := newFakeBisqueUploadServer(t)
	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
		Store:      store.NewMemoryStore(),
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			DevUsername:   "ada",
			DevPassword:   "secret",
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    uploadRoot,
			MaxImportSize: 8 << 20,
		}),
	})

	login := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"amil"}`))
	login.Header.Set("Content-Type", "application/json")
	loginRec := httptest.NewRecorder()
	router.ServeHTTP(loginRec, login)
	if loginRec.Code != http.StatusOK {
		t.Fatalf("login status = %d body=%s", loginRec.Code, loginRec.Body.String())
	}
	sessionCookie := loginRec.Result().Cookies()[0]

	fileIDs := make([]string, 0, 2)
	for _, name := range []string{"scan-a.png", "scan-b.png"} {
		var body bytes.Buffer
		writer := multipart.NewWriter(&body)
		part, err := writer.CreateFormFile("files", name)
		if err != nil {
			t.Fatalf("CreateFormFile: %v", err)
		}
		if _, err := part.Write(testPNGBytes(t, 4, 3)); err != nil {
			t.Fatalf("write multipart file: %v", err)
		}
		if err := writer.Close(); err != nil {
			t.Fatalf("close multipart writer: %v", err)
		}
		uploadReq := httptest.NewRequest(http.MethodPost, "/v2/uploads", &body)
		uploadReq.Header.Set("Content-Type", writer.FormDataContentType())
		uploadReq.AddCookie(sessionCookie)
		uploadRec := httptest.NewRecorder()
		router.ServeHTTP(uploadRec, uploadReq)
		if uploadRec.Code != http.StatusOK {
			t.Fatalf("upload status = %d body=%s", uploadRec.Code, uploadRec.Body.String())
		}
		var uploadResponse uploadFilesResponse
		if err := json.Unmarshal(uploadRec.Body.Bytes(), &uploadResponse); err != nil {
			t.Fatalf("decode upload response: %v", err)
		}
		fileIDs = append(fileIDs, uploadResponse.Uploaded[0].FileID)
	}

	collectionReq := httptest.NewRequest(http.MethodPost, "/v2/resource-collections", strings.NewReader(`{"name":"NIfTI Batch","collection_type":"folder"}`))
	collectionReq.Header.Set("Content-Type", "application/json")
	collectionReq.AddCookie(sessionCookie)
	collectionRec := httptest.NewRecorder()
	router.ServeHTTP(collectionRec, collectionReq)
	if collectionRec.Code != http.StatusOK && collectionRec.Code != http.StatusCreated {
		t.Fatalf("create collection status = %d body=%s", collectionRec.Code, collectionRec.Body.String())
	}
	var collectionResponse resourceCollectionResponse
	if err := json.Unmarshal(collectionRec.Body.Bytes(), &collectionResponse); err != nil {
		t.Fatalf("decode collection response: %v", err)
	}
	collectionID := collectionResponse.Collection.CollectionID

	addPayload := `{"resource_ids":["` + strings.Join(fileIDs, `","`) + `"]}`
	addReq := httptest.NewRequest(http.MethodPost, "/v2/resource-collections/"+collectionID+"/resources", strings.NewReader(addPayload))
	addReq.Header.Set("Content-Type", "application/json")
	addReq.AddCookie(sessionCookie)
	addRec := httptest.NewRecorder()
	router.ServeHTTP(addRec, addReq)
	if addRec.Code != http.StatusOK {
		t.Fatalf("add members status = %d body=%s", addRec.Code, addRec.Body.String())
	}

	pushReq := httptest.NewRequest(http.MethodPost, "/v2/bisque/push", strings.NewReader(`{"collection_ids":["`+collectionID+`"]}`))
	pushReq.Header.Set("Content-Type", "application/json")
	pushReq.AddCookie(sessionCookie)
	pushRec := httptest.NewRecorder()
	router.ServeHTTP(pushRec, pushReq)
	if pushRec.Code != http.StatusOK {
		t.Fatalf("push status = %d body=%s, want 200", pushRec.Code, pushRec.Body.String())
	}
	var pushResponse bisquePushResponse
	if err := json.Unmarshal(pushRec.Body.Bytes(), &pushResponse); err != nil {
		t.Fatalf("decode push response: %v", err)
	}
	if pushResponse.Count != 2 || len(pushResponse.Uploads) != 2 {
		t.Fatalf("push uploads = %+v, want both folder members uploaded", pushResponse)
	}
	if len(pushResponse.Datasets) != 1 {
		t.Fatalf("push datasets = %+v, want one dataset", pushResponse.Datasets)
	}
	dataset := pushResponse.Datasets[0]
	if dataset.Name != "NIfTI Batch" || dataset.MemberCount != 2 || dataset.CollectionID != collectionID || dataset.ResourceUniq != "00-DATASET" {
		t.Fatalf("dataset = %+v", dataset)
	}
	if len(state.uploads) != 2 {
		t.Fatalf("BisQue uploads = %v, want two transfers", state.uploads)
	}
	if len(state.datasetXML) != 1 {
		t.Fatalf("dataset posts = %d, want 1", len(state.datasetXML))
	}
	var datasetDoc bisqueDatasetXML
	if err := xml.Unmarshal([]byte(state.datasetXML[0]), &datasetDoc); err != nil {
		t.Fatalf("decode dataset XML: %v", err)
	}
	if datasetDoc.Name != "NIfTI Batch" || len(datasetDoc.Members) != 2 {
		t.Fatalf("dataset XML = %+v", datasetDoc)
	}
	for i, member := range datasetDoc.Members {
		want := fmt.Sprintf("%s/data_service/00-PUSH%d", bisque.URL, i+1)
		if member.URI != want || member.Type != "object" {
			t.Fatalf("dataset member %d = %+v, want %s", i, member, want)
		}
	}
}

func TestV2BisquePushUploadsLooseFilesWithOptionalDataset(t *testing.T) {
	t.Parallel()

	bisque, state := newFakeBisqueUploadServer(t)
	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
		Store:      store.NewMemoryStore(),
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			DevUsername:   "ada",
			DevPassword:   "secret",
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    uploadRoot,
			MaxImportSize: 8 << 20,
		}),
	})

	login := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"amil"}`))
	login.Header.Set("Content-Type", "application/json")
	loginRec := httptest.NewRecorder()
	router.ServeHTTP(loginRec, login)
	sessionCookie := loginRec.Result().Cookies()[0]

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "result.png")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	if _, err := part.Write(testPNGBytes(t, 4, 3)); err != nil {
		t.Fatalf("write multipart file: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	uploadReq := httptest.NewRequest(http.MethodPost, "/v2/uploads", &body)
	uploadReq.Header.Set("Content-Type", writer.FormDataContentType())
	uploadReq.AddCookie(sessionCookie)
	uploadRec := httptest.NewRecorder()
	router.ServeHTTP(uploadRec, uploadReq)
	if uploadRec.Code != http.StatusOK {
		t.Fatalf("upload status = %d body=%s", uploadRec.Code, uploadRec.Body.String())
	}
	var uploadResponse uploadFilesResponse
	if err := json.Unmarshal(uploadRec.Body.Bytes(), &uploadResponse); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	fileID := uploadResponse.Uploaded[0].FileID

	pushReq := httptest.NewRequest(http.MethodPost, "/v2/bisque/push", strings.NewReader(`{"file_ids":["`+fileID+`"],"dataset_name":"Loose Results"}`))
	pushReq.Header.Set("Content-Type", "application/json")
	pushReq.AddCookie(sessionCookie)
	pushRec := httptest.NewRecorder()
	router.ServeHTTP(pushRec, pushReq)
	if pushRec.Code != http.StatusOK {
		t.Fatalf("push status = %d body=%s, want 200", pushRec.Code, pushRec.Body.String())
	}
	var pushResponse bisquePushResponse
	if err := json.Unmarshal(pushRec.Body.Bytes(), &pushResponse); err != nil {
		t.Fatalf("decode push response: %v", err)
	}
	if pushResponse.Count != 1 || len(pushResponse.Uploads) != 1 || pushResponse.Uploads[0].FileID != fileID {
		t.Fatalf("push response uploads = %+v", pushResponse)
	}
	if len(pushResponse.Datasets) != 1 || pushResponse.Datasets[0].Name != "Loose Results" || pushResponse.Datasets[0].MemberCount != 1 {
		t.Fatalf("push response datasets = %+v", pushResponse.Datasets)
	}
	if len(state.uploads) != 1 || len(state.datasetXML) != 1 {
		t.Fatalf("BisQue calls = uploads %v datasets %d", state.uploads, len(state.datasetXML))
	}
}

func TestV2BisquePushRequiresLinkedAccount(t *testing.T) {
	t.Parallel()

	bisque, _ := newFakeBisqueUploadServer(t)
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: t.TempDir(),
		Store:      store.NewMemoryStore(),
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	pushReq := httptest.NewRequest(http.MethodPost, "/v2/bisque/push", strings.NewReader(`{"file_ids":["file_missing"]}`))
	pushReq.Header.Set("Content-Type", "application/json")
	pushRec := httptest.NewRecorder()
	router.ServeHTTP(pushRec, pushReq)
	if pushRec.Code != http.StatusBadRequest {
		t.Fatalf("push status = %d body=%s, want 400 when no BisQue account is linked", pushRec.Code, pushRec.Body.String())
	}
	if !strings.Contains(pushRec.Body.String(), "link your BisQue account") {
		t.Fatalf("push error body = %s, want linking guidance", pushRec.Body.String())
	}
}

func TestWorkerBisqueSearchUsesRunScopedSessionInWorkOSMode(t *testing.T) {
	t.Parallel()

	var mu sync.Mutex
	gotAuths := []string{}
	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		gotAuths = append(gotAuths, r.Header.Get("Authorization"))
		mu.Unlock()
		w.Header().Set("Content-Type", "application/xml")
		_, _ = w.Write([]byte(`<resource><image uri="` + bisque.URL + `/data_service/00-abc" name="scan.png" resource_uniq="00-abc"/></resource>`))
	}))
	defer bisque.Close()

	memory := store.NewMemoryStore()
	thread, err := memory.CreateThread(context.Background(), domain.CreateThreadInput{Title: "worker"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	credentialStore := NewBisqueCredentialStore()
	sessionID := credentialStore.Put(BisqueCredentials{Username: "amil", Password: "bean123"})
	run, err := memory.CreateRun(context.Background(), domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "workos:user_e2e",
		Goal:     "bisque worker test",
		Metadata: domain.JSONMap{"bisque_session_id": sessionID},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	router := NewRouter(ServerDeps{
		Version:           "test-version",
		Store:             memory,
		WorkerToken:       "worker-secret",
		WorkOS:            testWorkOSAuth(t, WorkOSAuthConfig{}),
		BisqueCredentials: credentialStore,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	makeSearch := func(workerToken string, runID string, bisqueSession string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(`{"resource_type":"image","limit":1}`))
		req.Header.Set("Content-Type", "application/json")
		if workerToken != "" {
			req.Header.Set("X-Ultra-Worker-Token", workerToken)
		}
		if runID != "" {
			req.Header.Set("X-Ultra-Run-Id", runID)
		}
		if bisqueSession != "" {
			req.Header.Set("X-Ultra-Bisque-Session-Id", bisqueSession)
		}
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}

	// Without any credentials the protected group rejects the request.
	if rec := makeSearch("", "", ""); rec.Code != http.StatusUnauthorized {
		t.Fatalf("anonymous search status = %d, want 401", rec.Code)
	}
	// A valid worker token alone (no run) is not enough for bisque endpoints.
	if rec := makeSearch("worker-secret", "", sessionID); rec.Code != http.StatusUnauthorized {
		t.Fatalf("worker search without run status = %d, want 401", rec.Code)
	}
	// Worker token + run whose metadata matches the session uses linked credentials.
	if rec := makeSearch("worker-secret", run.RunID, sessionID); rec.Code != http.StatusOK {
		t.Fatalf("worker search status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	wantAuth := "Basic " + base64.StdEncoding.EncodeToString([]byte("amil:bean123"))
	mu.Lock()
	lastAuth := gotAuths[len(gotAuths)-1]
	mu.Unlock()
	if lastAuth != wantAuth {
		t.Fatalf("Authorization = %q, want linked run-scoped credentials", lastAuth)
	}
	// A session id that does not match the run metadata is ignored.
	if rec := makeSearch("worker-secret", run.RunID, "bisque_session_stolen"); rec.Code != http.StatusOK {
		t.Fatalf("mismatched session search status = %d, want 200 with anonymous upstream", rec.Code)
	}
	mu.Lock()
	lastAuth = gotAuths[len(gotAuths)-1]
	mu.Unlock()
	if lastAuth == wantAuth {
		t.Fatalf("mismatched session id reused linked credentials")
	}
	// An invalid worker token is rejected outright.
	if rec := makeSearch("wrong-secret", run.RunID, sessionID); rec.Code != http.StatusUnauthorized {
		t.Fatalf("invalid worker token status = %d, want 401", rec.Code)
	}
}

func TestWorkerBisqueSessionRejectedWhenRunOwnerDiffers(t *testing.T) {
	t.Parallel()

	var mu sync.Mutex
	gotAuths := []string{}
	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		gotAuths = append(gotAuths, r.Header.Get("Authorization"))
		mu.Unlock()
		w.Header().Set("Content-Type", "application/xml")
		_, _ = w.Write([]byte(`<resource><image uri="` + bisque.URL + `/data_service/00-abc" name="scan.png" resource_uniq="00-abc"/></resource>`))
	}))
	defer bisque.Close()

	memory := store.NewMemoryStore()
	cipher, err := NewBisqueCredentialCipher(bytes.Repeat([]byte{9}, 32), "test-key")
	if err != nil {
		t.Fatalf("NewBisqueCredentialCipher: %v", err)
	}
	credentialStore := NewPersistentBisqueCredentialStore(memory, cipher, bisque.URL)
	victimSession, err := credentialStore.PutLinked(context.Background(), BisqueCredentialLinkInput{
		Credentials: BisqueCredentials{Username: "victim", Password: "victim-secret"},
		UserID:      "workos:user_victim",
		OrgID:       "workos-org",
		RootURL:     bisque.URL,
	})
	if err != nil {
		t.Fatalf("PutLinked: %v", err)
	}
	thread, err := memory.CreateThread(context.Background(), domain.CreateThreadInput{Title: "attacker"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	attackerRun, err := memory.CreateRun(context.Background(), domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "workos:user_attacker",
		Goal:     "cross-user session use",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	victimThread, err := memory.CreateThread(context.Background(), domain.CreateThreadInput{Title: "victim"})
	if err != nil {
		t.Fatalf("CreateThread victim: %v", err)
	}
	victimRun, err := memory.CreateRun(context.Background(), domain.CreateRunInput{
		ThreadID: victimThread.ThreadID,
		UserID:   "workos:user_victim",
		Goal:     "legitimate run",
	})
	if err != nil {
		t.Fatalf("CreateRun victim: %v", err)
	}

	router := NewRouter(ServerDeps{
		Version:           "test-version",
		Store:             memory,
		WorkerToken:       "worker-secret",
		WorkOS:            testWorkOSAuth(t, WorkOSAuthConfig{}),
		BisqueCredentials: credentialStore,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	search := func(runID string) {
		req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(`{"resource_type":"image","limit":1}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Ultra-Worker-Token", "worker-secret")
		req.Header.Set("X-Ultra-Run-Id", runID)
		req.Header.Set("X-Ultra-Bisque-Session-Id", victimSession)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("search status = %d body=%s, want 200", rec.Code, rec.Body.String())
		}
	}

	victimAuth := "Basic " + base64.StdEncoding.EncodeToString([]byte("victim:victim-secret"))
	search(attackerRun.RunID)
	mu.Lock()
	attackerAuth := gotAuths[len(gotAuths)-1]
	mu.Unlock()
	if attackerAuth == victimAuth {
		t.Fatalf("another user's run reused the victim's linked BisQue credentials")
	}
	search(victimRun.RunID)
	mu.Lock()
	ownerAuth := gotAuths[len(gotAuths)-1]
	mu.Unlock()
	if ownerAuth != victimAuth {
		t.Fatalf("run owner did not get their own linked credentials, Authorization = %q", ownerAuth)
	}
}

func TestWorkerUploadAttributesFilesToRunOwner(t *testing.T) {
	t.Parallel()

	memory := store.NewMemoryStore()
	thread, err := memory.CreateThread(context.Background(), domain.CreateThreadInput{Title: "worker uploads"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := memory.CreateRun(context.Background(), domain.CreateRunInput{
		ThreadID: thread.ThreadID,
		UserID:   "workos:user_e2e",
		Goal:     "stage outputs",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	router := NewRouter(ServerDeps{
		Version:     "test-version",
		Store:       memory,
		UploadRoot:  t.TempDir(),
		WorkerToken: "worker-secret",
		WorkOS:      testWorkOSAuth(t, WorkOSAuthConfig{}),
	})

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "overlay.png")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	if _, err := part.Write(testPNGBytes(t, 4, 3)); err != nil {
		t.Fatalf("write multipart file: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v2/uploads", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("X-Ultra-Worker-Token", "worker-secret")
	req.Header.Set("X-Ultra-Run-Id", run.RunID)
	req.Header.Set("X-Ultra-User-Id", "spoofed-user")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("worker upload status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var uploadResponse uploadFilesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &uploadResponse); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if len(uploadResponse.Uploaded) != 1 {
		t.Fatalf("uploaded = %+v, want one file", uploadResponse.Uploaded)
	}
	if uploadResponse.Uploaded[0].Principal.UserID != "workos:user_e2e" {
		t.Fatalf("uploaded principal = %q, want run owner workos:user_e2e", uploadResponse.Uploaded[0].Principal.UserID)
	}
}

func TestPersistentBisqueCredentialsSurviveCredentialStoreRestart(t *testing.T) {
	t.Parallel()

	var gotAuth string
	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/xml")
		if r.URL.Path == "/auth_service/session" {
			_, _ = w.Write([]byte(`<response><tag name="user" value="linked-user"/></response>`))
			return
		}
		_, _ = w.Write([]byte(`<response><image uri="` + bisque.URL + `/data_service/image/persistent" name="persistent.jpg" resource_uniq="persistent"/></response>`))
	}))
	defer bisque.Close()

	persistent := store.NewMemoryStore()
	cipher, err := NewBisqueCredentialCipher(bytes.Repeat([]byte{7}, 32), "test-key")
	if err != nil {
		t.Fatalf("NewBisqueCredentialCipher: %v", err)
	}
	firstCredentialStore := NewPersistentBisqueCredentialStore(persistent, cipher, bisque.URL)
	router := NewRouter(ServerDeps{
		Version:           "test-version",
		BisqueCredentials: firstCredentialStore,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			DevUsername:   "fallback-user",
			DevPassword:   "fallback-secret",
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	login := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"linked-user","password":"linked-secret"}`))
	login.Header.Set("Content-Type", "application/json")
	loginRec := httptest.NewRecorder()
	router.ServeHTTP(loginRec, login)
	if loginRec.Code != http.StatusOK {
		t.Fatalf("login status = %d body=%s", loginRec.Code, loginRec.Body.String())
	}
	linkedCookie := loginRec.Result().Cookies()[0]

	restartedCredentialStore := NewPersistentBisqueCredentialStore(persistent, cipher, bisque.URL)
	restartedRouter := NewRouter(ServerDeps{
		Version:           "test-version",
		BisqueCredentials: restartedCredentialStore,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			DevUsername:   "fallback-user",
			DevPassword:   "fallback-secret",
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})
	gotAuth = ""
	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(`{"resource_type":"image","limit":1}`))
	req.Header.Set("Content-Type", "application/json")
	req.AddCookie(linkedCookie)
	rec := httptest.NewRecorder()
	restartedRouter.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("search status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	wantAuth := "Basic " + base64.StdEncoding.EncodeToString([]byte("linked-user:linked-secret"))
	if gotAuth != wantAuth {
		t.Fatalf("Authorization = %q, want persistent linked credentials after credential-store restart", gotAuth)
	}
}

func TestPersistentBisqueCredentialsStoreEncryptedPasswordOnly(t *testing.T) {
	t.Parallel()

	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		if r.URL.Path == "/auth_service/session" {
			_, _ = w.Write([]byte(`<response><tag name="user" value="linked-user"/></response>`))
			return
		}
		_, _ = w.Write([]byte(`<response><image uri="` + bisque.URL + `/data_service/image/linked" name="linked.jpg" resource_uniq="linked"/></response>`))
	}))
	defer bisque.Close()

	persistent := store.NewMemoryStore()
	cipher, err := NewBisqueCredentialCipher(bytes.Repeat([]byte{9}, 32), "test-key")
	if err != nil {
		t.Fatalf("NewBisqueCredentialCipher: %v", err)
	}
	credentialStore := NewPersistentBisqueCredentialStore(persistent, cipher, bisque.URL)
	router := NewRouter(ServerDeps{
		Version:           "test-version",
		BisqueCredentials: credentialStore,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	login := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"linked-user","password":"linked-secret"}`))
	login.Header.Set("Content-Type", "application/json")
	loginRec := httptest.NewRecorder()
	router.ServeHTTP(loginRec, login)
	if loginRec.Code != http.StatusOK {
		t.Fatalf("login status = %d body=%s", loginRec.Code, loginRec.Body.String())
	}
	sessionID := linkedBisqueSessionIDFromCookie(t, loginRec.Result().Cookies())

	record, found, err := persistent.GetBisqueCredentialBySessionID(context.Background(), sessionID)
	if err != nil {
		t.Fatalf("GetBisqueCredentialBySessionID: %v", err)
	}
	if !found {
		t.Fatalf("missing persisted BisQue credential record")
	}
	if record.PasswordCiphertext == "" || record.PasswordNonce == "" {
		t.Fatalf("persisted record must include ciphertext and nonce: %#v", record)
	}
	if strings.Contains(record.PasswordCiphertext, "linked-secret") || strings.Contains(prettyJSON(record.Metadata), "linked-secret") {
		t.Fatalf("persisted BisQue credential leaked plaintext secret: %#v", record)
	}

	recovered, ok, err := credentialStore.GetWithContext(context.Background(), sessionID)
	if err != nil {
		t.Fatalf("GetWithContext: %v", err)
	}
	if !ok || recovered.Username != "linked-user" || recovered.Password != "linked-secret" {
		t.Fatalf("recovered credentials = %#v/%v, want decrypted linked credentials", recovered, ok)
	}
}

func TestBisqueUnlinkDeletesPersistentCredentials(t *testing.T) {
	t.Parallel()

	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		if r.URL.Path == "/auth_service/session" {
			_, _ = w.Write([]byte(`<response><tag name="user" value="linked-user"/></response>`))
			return
		}
		_, _ = w.Write([]byte(`<response><image uri="` + bisque.URL + `/data_service/image/linked" name="linked.jpg" resource_uniq="linked"/></response>`))
	}))
	defer bisque.Close()

	persistent := store.NewMemoryStore()
	cipher, err := NewBisqueCredentialCipher(bytes.Repeat([]byte{8}, 32), "test-key")
	if err != nil {
		t.Fatalf("NewBisqueCredentialCipher: %v", err)
	}
	credentialStore := NewPersistentBisqueCredentialStore(persistent, cipher, bisque.URL)
	router := NewRouter(ServerDeps{
		Version:           "test-version",
		BisqueCredentials: credentialStore,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	login := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"linked-user","password":"linked-secret"}`))
	login.Header.Set("Content-Type", "application/json")
	loginRec := httptest.NewRecorder()
	router.ServeHTTP(loginRec, login)
	if loginRec.Code != http.StatusOK {
		t.Fatalf("login status = %d body=%s", loginRec.Code, loginRec.Body.String())
	}
	linkedCookie := loginRec.Result().Cookies()[0]

	unlink := httptest.NewRequest(http.MethodPost, "/v2/bisque/unlink", nil)
	unlink.AddCookie(linkedCookie)
	unlinkRec := httptest.NewRecorder()
	router.ServeHTTP(unlinkRec, unlink)
	if unlinkRec.Code != http.StatusOK {
		t.Fatalf("unlink status = %d body=%s, want 200", unlinkRec.Code, unlinkRec.Body.String())
	}

	restartedCredentialStore := NewPersistentBisqueCredentialStore(persistent, cipher, bisque.URL)
	sessionReq := httptest.NewRequest(http.MethodGet, "/v2/auth/session", nil)
	sessionReq.AddCookie(linkedCookie)
	sessionRec := httptest.NewRecorder()
	NewRouter(ServerDeps{
		Version:           "test-version",
		BisqueCredentials: restartedCredentialStore,
	}).ServeHTTP(sessionRec, sessionReq)
	if sessionRec.Code != http.StatusOK {
		t.Fatalf("session status = %d body=%s", sessionRec.Code, sessionRec.Body.String())
	}
	var session map[string]any
	if err := json.Unmarshal(sessionRec.Body.Bytes(), &session); err != nil {
		t.Fatalf("decode session: %v", err)
	}
	if session["authenticated"] != false || session["bisque_linked"] != false {
		t.Fatalf("session after unlink = %#v, want unauthenticated and not linked", session)
	}
}

func TestV2BisqueSearchUsesRunScopedLinkedSessionHeader(t *testing.T) {
	t.Parallel()

	var gotAuth string
	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/xml")
		if r.URL.Path == "/auth_service/session" {
			_, _ = w.Write([]byte(`<response><tag name="user" value="linked-user"/></response>`))
			return
		}
		_, _ = w.Write([]byte(`<response><image uri="` + bisque.URL + `/data_service/image/linked" name="linked.jpg" resource_uniq="linked"/></response>`))
	}))
	defer bisque.Close()

	credentialStore := NewBisqueCredentialStore()
	router := NewRouter(ServerDeps{
		Version:           "test-version",
		BisqueCredentials: credentialStore,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			DevUsername:   "fallback-user",
			DevPassword:   "fallback-secret",
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})

	login := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"linked-user","password":"linked-secret"}`))
	login.Header.Set("Content-Type", "application/json")
	loginRec := httptest.NewRecorder()
	router.ServeHTTP(loginRec, login)
	if loginRec.Code != http.StatusOK {
		t.Fatalf("login status = %d body=%s", loginRec.Code, loginRec.Body.String())
	}
	sessionID := linkedBisqueSessionIDFromCookie(t, loginRec.Result().Cookies())
	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/search", strings.NewReader(`{"resource_type":"image","limit":1}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-Run-Id", "run-linked")
	req.Header.Set("X-Ultra-Bisque-Session-Id", sessionID)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("search status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	wantAuth := "Basic " + base64.StdEncoding.EncodeToString([]byte("linked-user:linked-secret"))
	if gotAuth != wantAuth {
		t.Fatalf("Authorization = %q, want run-scoped linked session credentials", gotAuth)
	}
	if strings.Contains(gotAuth, "fallback") {
		t.Fatalf("Authorization unexpectedly used fallback credentials: %q", gotAuth)
	}
}

func TestCreateRunPassesBisqueSessionOnlyInTransientJobMetadata(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	credentialStore := NewBisqueCredentialStore()
	router := NewRouter(ServerDeps{
		Version:           "test-version",
		Runs:              runcontrol.NewService(mem, bus),
		Store:             mem,
		BisqueCredentials: credentialStore,
	})

	sessionID, err := credentialStore.PutLinked(context.Background(), BisqueCredentialLinkInput{
		Credentials: BisqueCredentials{Username: "linked-user", Password: "linked-secret"},
		UserID:      "bisque:linked-user",
		OrgID:       "local-org",
		RootURL:     "https://bisque.example.org",
		Metadata:    domain.JSONMap{"source": "test"},
	})
	if err != nil {
		t.Fatalf("PutLinked: %v", err)
	}
	sessionCookie := &http.Cookie{Name: "ultra_dev_auth", Value: "bisque_session:" + sessionID}

	threadReq := httptest.NewRequest(http.MethodPost, "/v2/threads", strings.NewReader(`{"title":"BisQue linked run"}`))
	threadReq.Header.Set("Content-Type", "application/json")
	threadReq.AddCookie(sessionCookie)
	threadRec := httptest.NewRecorder()
	router.ServeHTTP(threadRec, threadReq)
	if threadRec.Code != http.StatusOK {
		t.Fatalf("thread status = %d body=%s", threadRec.Code, threadRec.Body.String())
	}
	var thread domain.ThreadRecord
	if err := json.Unmarshal(threadRec.Body.Bytes(), &thread); err != nil {
		t.Fatalf("decode thread: %v", err)
	}

	runReq := httptest.NewRequest(http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", strings.NewReader(`{"goal":"Use bqapi to inspect selected images","selected_tool_names":["bisque"]}`))
	runReq.Header.Set("Content-Type", "application/json")
	runReq.AddCookie(sessionCookie)
	runRec := httptest.NewRecorder()
	router.ServeHTTP(runRec, runReq)
	if runRec.Code != http.StatusOK {
		t.Fatalf("run status = %d body=%s", runRec.Code, runRec.Body.String())
	}
	var run domain.RunRecord
	if err := json.Unmarshal(runRec.Body.Bytes(), &run); err != nil {
		t.Fatalf("decode run: %v", err)
	}
	if _, exposed := run.Metadata["bisque_session_id"]; exposed {
		t.Fatalf("run metadata exposed BisQue session reference: %#v", run.Metadata)
	}
	storedRun, err := mem.GetRun(context.Background(), run.RunID)
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if _, exposed := storedRun.Metadata["bisque_session_id"]; exposed {
		t.Fatalf("stored run metadata exposed BisQue session reference: %#v", storedRun.Metadata)
	}

	select {
	case job := <-bus.Jobs():
		if job.Metadata["bisque_session_id"] != sessionID {
			t.Fatalf("job bisque_session_id = %#v, want transient session id", job.Metadata["bisque_session_id"])
		}
		if strings.Contains(prettyJSON(job.Metadata), "linked-secret") {
			t.Fatalf("job metadata leaked BisQue password: %s", prettyJSON(job.Metadata))
		}
	case <-time.After(time.Second):
		t.Fatalf("timed out waiting for run job")
	}
}

func TestV2BisqueImportDownloadsResourceIntoUploadStore(t *testing.T) {
	t.Parallel()

	pngBytes := testPNGBytes(t, 3, 2)
	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/data_service/image/abc":
			w.Header().Set("Content-Type", "application/xml")
			_, _ = w.Write([]byte(`<image uri="` + bisque.URL + `/data_service/image/abc" name="prairie.png" resource_uniq="abc" />`))
		case "/blob_service/abc":
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(pngBytes)
		default:
			t.Fatalf("unexpected BisQue path %s", r.URL.Path)
		}
	}))
	defer bisque.Close()

	uploadRoot := t.TempDir()
	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
		Store:      mem,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			DevUsername:   "ada",
			DevPassword:   "secret",
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    uploadRoot,
			MaxImportSize: 8 << 20,
		}),
	})

	req := httptest.NewRequest(http.MethodPost, "/v2/uploads/from-bisque", strings.NewReader(`{"resources":["`+bisque.URL+`/data_service/image/abc"]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("import status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var body struct {
		FileCount int `json:"file_count"`
		Uploaded  []struct {
			FileID       string `json:"file_id"`
			OriginalName string `json:"original_name"`
			SourceURI    string `json:"source_uri"`
		} `json:"uploaded"`
		Imports []struct {
			Status      string `json:"status"`
			ResourceURI string `json:"resource_uri"`
			Uploaded    struct {
				FileID string `json:"file_id"`
			} `json:"uploaded"`
			ClientViewURL   string `json:"client_view_url"`
			ImageServiceURL string `json:"image_service_url"`
		} `json:"imports"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode import response: %v", err)
	}
	if body.FileCount != 1 || len(body.Uploaded) != 1 || len(body.Imports) != 1 {
		t.Fatalf("import body = %+v, want one imported upload", body)
	}
	if body.Uploaded[0].OriginalName != "prairie.png" {
		t.Fatalf("original name = %q, want prairie.png", body.Uploaded[0].OriginalName)
	}
	if body.Uploaded[0].SourceURI != bisque.URL+"/data_service/image/abc" {
		t.Fatalf("source_uri = %q", body.Uploaded[0].SourceURI)
	}
	if body.Imports[0].Status != "imported" || body.Imports[0].Uploaded.FileID == "" {
		t.Fatalf("import row = %+v", body.Imports[0])
	}
	if body.Imports[0].ClientViewURL == "" || body.Imports[0].ImageServiceURL == "" {
		t.Fatalf("import links missing: %+v", body.Imports[0])
	}
	_, path, err := findUploadResource(uploadRoot, body.Uploaded[0].FileID)
	if err != nil {
		t.Fatalf("find imported upload: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read imported upload: %v", err)
	}
	if !bytes.Equal(data, pngBytes) {
		t.Fatalf("imported bytes changed: got %d want %d", len(data), len(pngBytes))
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+body.Uploaded[0].FileID+"/events", nil)
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("resource events status = %d body=%s, want 200", eventsRec.Code, eventsRec.Body.String())
	}
	var events resourceEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode import resource events: %v", err)
	}
	foundImportEvent := false
	for _, event := range events.Events {
		if event.EventType != "resource.imported" {
			continue
		}
		if event.Metadata["source_type"] == "bisque_import" &&
			event.Metadata["source_uri"] == bisque.URL+"/data_service/image/abc" &&
			event.Metadata["bisque_resource_uri"] == body.Imports[0].ResourceURI &&
			event.Metadata["bisque_resource_uniq"] == "abc" &&
			event.Metadata["import_status"] == "imported" {
			foundImportEvent = true
			break
		}
	}
	if !foundImportEvent {
		t.Fatalf("resource events = %+v, want resource.imported audit metadata with BisQue source provenance", events.Events)
	}
}

func TestV2BisqueUploadPostsLocalUploadToBisque(t *testing.T) {
	t.Parallel()

	var postedBytes []byte
	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/import/transfer" {
			t.Fatalf("upload path = %q, want /import/transfer", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got == "" {
			t.Fatalf("upload request missing Authorization")
		}
		reader, err := r.MultipartReader()
		if err != nil {
			t.Fatalf("multipart reader: %v", err)
		}
		for {
			part, err := reader.NextPart()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatalf("multipart next: %v", err)
			}
			if part.FormName() == "file" {
				postedBytes, _ = io.ReadAll(part)
			}
		}
		w.Header().Set("Content-Type", "application/xml")
		_, _ = w.Write([]byte(`<response><image uri="` + bisque.URL + `/data_service/image/uploaded" name="analysis.png" resource_uniq="uploaded"/></response>`))
	}))
	defer bisque.Close()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			DevUsername:   "ada",
			DevPassword:   "secret",
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    uploadRoot,
			MaxImportSize: 8 << 20,
		}),
	})
	fileID := writeTestUploadFile(t, uploadRoot, "analysis.png", []byte("plot-bytes"))

	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/upload", strings.NewReader(`{"file_ids":["`+fileID+`"]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-User-Id", "test-user")
	req.Header.Set("X-Ultra-Org-Id", "test-org")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("upload status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	if string(postedBytes) != "plot-bytes" {
		t.Fatalf("posted bytes = %q, want plot-bytes", string(postedBytes))
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if body["count"].(float64) != 1 {
		t.Fatalf("upload count = %#v, want 1", body["count"])
	}
	results := body["uploads"].([]any)
	first := results[0].(map[string]any)
	if first["resource_uri"] != bisque.URL+"/data_service/image/uploaded" {
		t.Fatalf("uploaded resource uri = %#v", first)
	}
}

func TestV2BisqueUploadParsesBQAPIUploadedWrapper(t *testing.T) {
	t.Parallel()

	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/import/transfer" {
			t.Fatalf("upload path = %q, want /import/transfer", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/xml")
		_, _ = w.Write([]byte(`<resource type="uploaded"><image uri="` + bisque.URL + `/data_service/image/wrapped" name="wrapped.png" resource_uniq="wrapped"/></resource>`))
	}))
	defer bisque.Close()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			DevUsername:   "ada",
			DevPassword:   "secret",
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    uploadRoot,
			MaxImportSize: 8 << 20,
		}),
	})
	fileID := writeTestUploadFile(t, uploadRoot, "wrapped.png", []byte("plot-bytes"))

	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/upload", strings.NewReader(`{"file_ids":["`+fileID+`"]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-User-Id", "test-user")
	req.Header.Set("X-Ultra-Org-Id", "test-org")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("upload status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var body struct {
		Uploads []BisqueUploadRecord `json:"uploads"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if len(body.Uploads) != 1 {
		t.Fatalf("upload count = %d, want 1", len(body.Uploads))
	}
	if body.Uploads[0].ResourceURI != bisque.URL+"/data_service/image/wrapped" || body.Uploads[0].ResourceUniq != "wrapped" {
		t.Fatalf("wrapped upload parsed as %+v", body.Uploads[0])
	}
}

func TestV2BisqueUploadPostsArtifactToBisque(t *testing.T) {
	t.Parallel()

	var postedBytes []byte
	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/import/transfer" {
			t.Fatalf("upload path = %q, want /import/transfer", r.URL.Path)
		}
		reader, err := r.MultipartReader()
		if err != nil {
			t.Fatalf("multipart reader: %v", err)
		}
		for {
			part, err := reader.NextPart()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatalf("multipart next: %v", err)
			}
			if part.FormName() == "file" {
				postedBytes, _ = io.ReadAll(part)
			}
		}
		w.Header().Set("Content-Type", "application/xml")
		_, _ = w.Write([]byte(`<response><file uri="` + bisque.URL + `/data_service/file/report" name="report.md" resource_uniq="report"/></response>`))
	}))
	defer bisque.Close()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	artifactRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:      "test-version",
		Runs:         service,
		Store:        mem,
		ArtifactRoot: artifactRoot,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			DevUsername:   "ada",
			DevPassword:   "secret",
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			MaxImportSize: 8 << 20,
		}),
	})
	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "user-1", Title: "artifact upload"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{ThreadID: thread.ThreadID, UserID: "user-1", Goal: "artifact upload"})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	reportPath := filepath.Join(artifactRoot, run.RunID, "report.md")
	if err := os.MkdirAll(filepath.Dir(reportPath), 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	if err := os.WriteFile(reportPath, []byte("report-bytes"), 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	artifact, err := mem.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:    run.RunID,
		ThreadID: thread.ThreadID,
		Kind:     "report",
		Path:     "report.md",
		MimeType: "text/markdown",
		Title:    "Report",
	})
	if err != nil {
		t.Fatalf("CreateArtifact: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v2/bisque/upload", strings.NewReader(`{"artifact_ids":["`+artifact.ArtifactID+`"]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-User-Id", "user-1")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("upload status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	if string(postedBytes) != "report-bytes" {
		t.Fatalf("posted bytes = %q, want report-bytes", string(postedBytes))
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if body["count"].(float64) != 1 {
		t.Fatalf("upload count = %#v, want 1", body["count"])
	}
	first := body["uploads"].([]any)[0].(map[string]any)
	if first["artifact_id"] != artifact.ArtifactID || first["resource_uri"] != bisque.URL+"/data_service/file/report" {
		t.Fatalf("upload result = %#v", first)
	}
}

func testPNGBytes(t *testing.T, width int, height int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, color.RGBA{R: uint8(40 * x), G: uint8(60 * y), B: 120, A: 255})
		}
	}
	var buffer bytes.Buffer
	if err := png.Encode(&buffer, img); err != nil {
		t.Fatalf("encode test PNG: %v", err)
	}
	return buffer.Bytes()
}

func testRGBPNGBytes(t *testing.T, width int, height int, pixels []color.RGBA) []byte {
	t.Helper()
	if width <= 0 || height <= 0 {
		t.Fatalf("invalid test PNG dimensions %dx%d", width, height)
	}
	if len(pixels) != width*height {
		t.Fatalf("invalid test PNG pixel count %d, want %d", len(pixels), width*height)
	}
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.SetRGBA(x, y, pixels[y*width+x])
		}
	}
	var buffer bytes.Buffer
	if err := png.Encode(&buffer, img); err != nil {
		t.Fatalf("encode test RGB PNG: %v", err)
	}
	return buffer.Bytes()
}

func testMinimalTIFFBytes(t *testing.T, width int, height int) []byte {
	return testGray16TIFFBytes(t, width, height, nil)
}

func testOmeTIFFStackBytes(t *testing.T, width int, height int, depth int, channels int, channelNames []string, values []uint16) []byte {
	t.Helper()
	return testOmeTIFFStackBytesWithDescriptionPadding(t, width, height, depth, channels, channelNames, values, "")
}

func testOmeTIFFStackBytesWithDescriptionPadding(t *testing.T, width int, height int, depth int, channels int, channelNames []string, values []uint16, descriptionPadding string) []byte {
	t.Helper()
	if width <= 0 || height <= 0 || depth <= 0 || channels <= 0 {
		t.Fatalf("invalid OME-TIFF dimensions %dx%dx%d c=%d", width, height, depth, channels)
	}
	planePixels := width * height
	planeCount := depth * channels
	if len(values) != planeCount*planePixels {
		t.Fatalf("invalid OME-TIFF pixel count %d, want %d", len(values), planeCount*planePixels)
	}
	if len(channelNames) != channels {
		t.Fatalf("channelNames length = %d, want %d", len(channelNames), channels)
	}

	var ome strings.Builder
	ome.WriteString(`<?xml version="1.0" encoding="UTF-8"?>`)
	if descriptionPadding != "" {
		ome.WriteString(`<!--`)
		ome.WriteString(descriptionPadding)
		ome.WriteString(`-->`)
	}
	ome.WriteString(`<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image ID="Image:0" Name="fixture"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" SizeT="1"`)
	fmt.Fprintf(&ome, ` SizeC="%d" SizeZ="%d" SizeY="%d" SizeX="%d" PhysicalSizeX="0.5" PhysicalSizeXUnit="um" PhysicalSizeY="0.5" PhysicalSizeYUnit="um" PhysicalSizeZ="1.25" PhysicalSizeZUnit="um">`, channels, depth, height, width)
	for channelIndex, name := range channelNames {
		fmt.Fprintf(&ome, `<Channel ID="Channel:0:%d" Name="%s" Color="%d" SamplesPerPixel="1"/>`, channelIndex, name, 0x00ffffff)
	}
	for z := 0; z < depth; z++ {
		for c := 0; c < channels; c++ {
			ifd := z*channels + c
			fmt.Fprintf(&ome, `<TiffData IFD="%d" FirstT="0" FirstZ="%d" FirstC="%d" PlaneCount="1"/>`, ifd, z, c)
		}
	}
	ome.WriteString(`</Pixels></Image></OME>`)
	description := []byte(ome.String() + "\x00")

	const headerSize = 8
	const firstIFDEntries = 10
	const otherIFDEntries = 9
	firstIFDSize := 2 + firstIFDEntries*12 + 4
	otherIFDSize := 2 + otherIFDEntries*12 + 4
	ifdOffsets := make([]int, planeCount)
	nextIFDOffset := headerSize
	for index := range ifdOffsets {
		ifdOffsets[index] = nextIFDOffset
		if index == 0 {
			nextIFDOffset += firstIFDSize
		} else {
			nextIFDOffset += otherIFDSize
		}
	}
	descriptionOffset := nextIFDOffset
	dataOffset := descriptionOffset + len(description)
	pixelBytes := planePixels * 2
	output := make([]byte, dataOffset+planeCount*pixelBytes)
	output[0] = 'I'
	output[1] = 'I'
	put16 := func(offset int, value int) {
		output[offset] = byte(value)
		output[offset+1] = byte(value >> 8)
	}
	put32 := func(offset int, value int) {
		output[offset] = byte(value)
		output[offset+1] = byte(value >> 8)
		output[offset+2] = byte(value >> 16)
		output[offset+3] = byte(value >> 24)
	}
	put16(2, 42)
	put32(4, ifdOffsets[0])
	copy(output[descriptionOffset:], description)

	writeIFD := func(pageIndex int, includeDescription bool) {
		entryCount := otherIFDEntries
		if includeDescription {
			entryCount = firstIFDEntries
		}
		base := ifdOffsets[pageIndex]
		put16(base, entryCount)
		entryOffset := base + 2
		addEntry := func(tag int, dataType int, count int, value int) {
			put16(entryOffset, tag)
			put16(entryOffset+2, dataType)
			put32(entryOffset+4, count)
			if dataType == 3 && count == 1 {
				put16(entryOffset+8, value)
				put16(entryOffset+10, 0)
			} else {
				put32(entryOffset+8, value)
			}
			entryOffset += 12
		}
		addEntry(256, 4, 1, width)
		addEntry(257, 4, 1, height)
		addEntry(258, 3, 1, 16)
		addEntry(259, 3, 1, 1)
		addEntry(262, 3, 1, 1)
		if includeDescription {
			addEntry(270, 2, len(description), descriptionOffset)
		}
		addEntry(273, 4, 1, dataOffset+pageIndex*pixelBytes)
		addEntry(277, 3, 1, 1)
		addEntry(278, 4, 1, height)
		addEntry(279, 4, 1, pixelBytes)
		nextOffset := 0
		if pageIndex+1 < len(ifdOffsets) {
			nextOffset = ifdOffsets[pageIndex+1]
		}
		put32(entryOffset, nextOffset)
	}
	for pageIndex := range ifdOffsets {
		writeIFD(pageIndex, pageIndex == 0)
	}
	for index, value := range values {
		put16(dataOffset+index*2, int(value))
	}
	return output
}

func testGray16TIFFBytes(t *testing.T, width int, height int, values []uint16) []byte {
	t.Helper()
	if width <= 0 || height <= 0 {
		t.Fatalf("invalid test TIFF dimensions %dx%d", width, height)
	}
	if values != nil && len(values) != width*height {
		t.Fatalf("invalid test TIFF pixel count %d, want %d", len(values), width*height)
	}
	const entryCount = 9
	const headerSize = 8
	const ifdSize = 2 + entryCount*12 + 4
	dataOffset := headerSize + ifdSize
	pixelBytes := width * height * 2
	output := make([]byte, dataOffset+pixelBytes)
	output[0] = 'I'
	output[1] = 'I'
	put16 := func(offset int, value int) {
		output[offset] = byte(value)
		output[offset+1] = byte(value >> 8)
	}
	put32 := func(offset int, value int) {
		output[offset] = byte(value)
		output[offset+1] = byte(value >> 8)
		output[offset+2] = byte(value >> 16)
		output[offset+3] = byte(value >> 24)
	}
	put16(2, 42)
	put32(4, 8)
	put16(8, entryCount)
	entryOffset := 10
	addEntry := func(tag int, dataType int, count int, value int) {
		put16(entryOffset, tag)
		put16(entryOffset+2, dataType)
		put32(entryOffset+4, count)
		if dataType == 3 && count == 1 {
			put16(entryOffset+8, value)
			put16(entryOffset+10, 0)
		} else {
			put32(entryOffset+8, value)
		}
		entryOffset += 12
	}
	addEntry(256, 4, 1, width)      // ImageWidth
	addEntry(257, 4, 1, height)     // ImageLength
	addEntry(258, 3, 1, 16)         // BitsPerSample
	addEntry(259, 3, 1, 1)          // Compression: none
	addEntry(262, 3, 1, 1)          // PhotometricInterpretation: BlackIsZero
	addEntry(273, 4, 1, dataOffset) // StripOffsets
	addEntry(277, 3, 1, 1)          // SamplesPerPixel
	addEntry(278, 4, 1, height)     // RowsPerStrip
	addEntry(279, 4, 1, pixelBytes) // StripByteCounts
	put32(entryOffset, 0)
	for index, value := range values {
		put16(dataOffset+index*2, int(value))
	}
	return output
}

func testNifti1Uint16Bytes(t *testing.T, width int, height int, depth int, values []uint16) []byte {
	t.Helper()
	if width <= 0 || height <= 0 || depth <= 0 {
		t.Fatalf("invalid NIfTI dimensions %dx%dx%d", width, height, depth)
	}
	voxelCount := width * height * depth
	if len(values) != voxelCount {
		t.Fatalf("NIfTI fixture has %d values, want %d", len(values), voxelCount)
	}
	const headerSize = 348
	const voxOffset = 352
	output := make([]byte, voxOffset+voxelCount*2)
	binary.LittleEndian.PutUint32(output[0:4], uint32(headerSize))
	binary.LittleEndian.PutUint16(output[40:42], 3)
	binary.LittleEndian.PutUint16(output[42:44], uint16(width))
	binary.LittleEndian.PutUint16(output[44:46], uint16(height))
	binary.LittleEndian.PutUint16(output[46:48], uint16(depth))
	binary.LittleEndian.PutUint16(output[70:72], 512)
	binary.LittleEndian.PutUint16(output[72:74], 16)
	for axis := 1; axis <= 3; axis++ {
		binary.LittleEndian.PutUint32(output[76+axis*4:80+axis*4], math.Float32bits(1))
	}
	binary.LittleEndian.PutUint32(output[108:112], math.Float32bits(float32(voxOffset)))
	copy(output[344:348], []byte{'n', '+', '1', 0})
	for index, value := range values {
		binary.LittleEndian.PutUint16(output[voxOffset+index*2:voxOffset+index*2+2], value)
	}
	return output
}

// testNifti1Uint16TimeBytes builds a 4D NIfTI (dim[0]=4) whose 4th dimension is
// time — the fMRI layout. Timepoint volumes are stored as consecutive slabs.
func testNifti1Uint16TimeBytes(t *testing.T, width int, height int, depth int, timepoints int, values []uint16) []byte {
	t.Helper()
	if width <= 0 || height <= 0 || depth <= 0 || timepoints <= 0 {
		t.Fatalf("invalid NIfTI dimensions %dx%dx%dx%d", width, height, depth, timepoints)
	}
	voxelCount := width * height * depth * timepoints
	if len(values) != voxelCount {
		t.Fatalf("NIfTI fixture has %d values, want %d", len(values), voxelCount)
	}
	const headerSize = 348
	const voxOffset = 352
	output := make([]byte, voxOffset+voxelCount*2)
	binary.LittleEndian.PutUint32(output[0:4], uint32(headerSize))
	binary.LittleEndian.PutUint16(output[40:42], 4)
	binary.LittleEndian.PutUint16(output[42:44], uint16(width))
	binary.LittleEndian.PutUint16(output[44:46], uint16(height))
	binary.LittleEndian.PutUint16(output[46:48], uint16(depth))
	binary.LittleEndian.PutUint16(output[48:50], uint16(timepoints))
	binary.LittleEndian.PutUint16(output[70:72], 512)
	binary.LittleEndian.PutUint16(output[72:74], 16)
	for axis := 1; axis <= 3; axis++ {
		binary.LittleEndian.PutUint32(output[76+axis*4:80+axis*4], math.Float32bits(1))
	}
	binary.LittleEndian.PutUint32(output[108:112], math.Float32bits(float32(voxOffset)))
	copy(output[344:348], []byte{'n', '+', '1', 0})
	for index, value := range values {
		binary.LittleEndian.PutUint16(output[voxOffset+index*2:voxOffset+index*2+2], value)
	}
	return output
}

// testNifti1Uint16ChannelBytes builds a genuine multi-component NIfTI: the 4th
// dimension is singleton time and the 5th (dim[5]) carries channels.
func testNifti1Uint16ChannelBytes(t *testing.T, width int, height int, depth int, channels int, values []uint16) []byte {
	t.Helper()
	if width <= 0 || height <= 0 || depth <= 0 || channels <= 0 {
		t.Fatalf("invalid NIfTI dimensions %dx%dx%dx%d", width, height, depth, channels)
	}
	voxelCount := width * height * depth * channels
	if len(values) != voxelCount {
		t.Fatalf("NIfTI fixture has %d values, want %d", len(values), voxelCount)
	}
	const headerSize = 348
	const voxOffset = 352
	output := make([]byte, voxOffset+voxelCount*2)
	binary.LittleEndian.PutUint32(output[0:4], uint32(headerSize))
	binary.LittleEndian.PutUint16(output[40:42], 5)
	binary.LittleEndian.PutUint16(output[42:44], uint16(width))
	binary.LittleEndian.PutUint16(output[44:46], uint16(height))
	binary.LittleEndian.PutUint16(output[46:48], uint16(depth))
	binary.LittleEndian.PutUint16(output[48:50], 1)
	binary.LittleEndian.PutUint16(output[50:52], uint16(channels))
	binary.LittleEndian.PutUint16(output[70:72], 512)
	binary.LittleEndian.PutUint16(output[72:74], 16)
	for axis := 1; axis <= 3; axis++ {
		binary.LittleEndian.PutUint32(output[76+axis*4:80+axis*4], math.Float32bits(1))
	}
	binary.LittleEndian.PutUint32(output[108:112], math.Float32bits(float32(voxOffset)))
	copy(output[344:348], []byte{'n', '+', '1', 0})
	for index, value := range values {
		binary.LittleEndian.PutUint16(output[voxOffset+index*2:voxOffset+index*2+2], value)
	}
	return output
}

func testNifti1Int16Bytes(t *testing.T, width int, height int, depth int, values []int16) []byte {
	t.Helper()
	if width <= 0 || height <= 0 || depth <= 0 {
		t.Fatalf("invalid NIfTI dimensions %dx%dx%d", width, height, depth)
	}
	voxelCount := width * height * depth
	if len(values) != voxelCount {
		t.Fatalf("NIfTI fixture has %d values, want %d", len(values), voxelCount)
	}
	const headerSize = 348
	const voxOffset = 352
	output := make([]byte, voxOffset+voxelCount*2)
	binary.LittleEndian.PutUint32(output[0:4], uint32(headerSize))
	binary.LittleEndian.PutUint16(output[40:42], 3)
	binary.LittleEndian.PutUint16(output[42:44], uint16(width))
	binary.LittleEndian.PutUint16(output[44:46], uint16(height))
	binary.LittleEndian.PutUint16(output[46:48], uint16(depth))
	binary.LittleEndian.PutUint16(output[70:72], 4)
	binary.LittleEndian.PutUint16(output[72:74], 16)
	for axis := 1; axis <= 3; axis++ {
		binary.LittleEndian.PutUint32(output[76+axis*4:80+axis*4], math.Float32bits(1))
	}
	binary.LittleEndian.PutUint32(output[108:112], math.Float32bits(float32(voxOffset)))
	copy(output[344:348], []byte{'n', '+', '1', 0})
	for index, value := range values {
		binary.LittleEndian.PutUint16(output[voxOffset+index*2:voxOffset+index*2+2], uint16(value))
	}
	return output
}

func testNifti1Float32Bytes(t *testing.T, width int, height int, depth int, values []float32) []byte {
	t.Helper()
	if width <= 0 || height <= 0 || depth <= 0 {
		t.Fatalf("invalid NIfTI dimensions %dx%dx%d", width, height, depth)
	}
	voxelCount := width * height * depth
	if len(values) != voxelCount {
		t.Fatalf("NIfTI fixture has %d values, want %d", len(values), voxelCount)
	}
	const headerSize = 348
	const voxOffset = 352
	output := make([]byte, voxOffset+voxelCount*4)
	binary.LittleEndian.PutUint32(output[0:4], uint32(headerSize))
	binary.LittleEndian.PutUint16(output[40:42], 3)
	binary.LittleEndian.PutUint16(output[42:44], uint16(width))
	binary.LittleEndian.PutUint16(output[44:46], uint16(height))
	binary.LittleEndian.PutUint16(output[46:48], uint16(depth))
	binary.LittleEndian.PutUint16(output[70:72], 16)
	binary.LittleEndian.PutUint16(output[72:74], 32)
	for axis := 1; axis <= 3; axis++ {
		binary.LittleEndian.PutUint32(output[76+axis*4:80+axis*4], math.Float32bits(1))
	}
	binary.LittleEndian.PutUint32(output[108:112], math.Float32bits(float32(voxOffset)))
	copy(output[344:348], []byte{'n', '+', '1', 0})
	for index, value := range values {
		binary.LittleEndian.PutUint32(output[voxOffset+index*4:voxOffset+index*4+4], math.Float32bits(value))
	}
	return output
}

func slicesEqualInts(left []int, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func sliceContains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func writeTestUploadFile(t *testing.T, uploadRoot string, originalName string, data []byte) string {
	t.Helper()
	if err := os.MkdirAll(uploadRoot, 0o755); err != nil {
		t.Fatalf("create upload root: %v", err)
	}
	fileID := domain.NewID("file")
	safeName := safeOriginalFilename(originalName)
	path := filepath.Join(uploadRoot, fileID+"__"+safeName)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write upload fixture: %v", err)
	}
	if err := writeUploadMetadata(uploadRoot, fileID, requestPrincipal{UserID: "test-user", OrgID: "test-org", Role: "researcher"}); err != nil {
		t.Fatalf("write upload fixture metadata: %v", err)
	}
	return fileID
}

func TestCreateRunReusesIdempotencyKeyFromHeader(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(context.Background(), runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "idempotency",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}

	body := `{"user_id":"user-1","goal":"hello","messages":[{"role":"user","content":"hello"}]}`
	firstReq := httptest.NewRequest(http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", strings.NewReader(body))
	firstReq.Header.Set("Content-Type", "application/json")
	firstReq.Header.Set("Idempotency-Key", "prompt-key-http")
	firstRec := httptest.NewRecorder()
	router.ServeHTTP(firstRec, firstReq)
	if firstRec.Code != http.StatusOK {
		t.Fatalf("first create run status = %d body=%s", firstRec.Code, firstRec.Body.String())
	}

	secondReq := httptest.NewRequest(http.MethodPost, "/v2/threads/"+thread.ThreadID+"/runs", strings.NewReader(body))
	secondReq.Header.Set("Content-Type", "application/json")
	secondReq.Header.Set("Idempotency-Key", "prompt-key-http")
	secondRec := httptest.NewRecorder()
	router.ServeHTTP(secondRec, secondReq)
	if secondRec.Code != http.StatusOK {
		t.Fatalf("second create run status = %d body=%s", secondRec.Code, secondRec.Body.String())
	}

	var firstRun domain.RunRecord
	if err := json.Unmarshal(firstRec.Body.Bytes(), &firstRun); err != nil {
		t.Fatalf("decode first run: %v", err)
	}
	var secondRun domain.RunRecord
	if err := json.Unmarshal(secondRec.Body.Bytes(), &secondRun); err != nil {
		t.Fatalf("decode second run: %v", err)
	}
	if secondRun.RunID != firstRun.RunID {
		t.Fatalf("second run id = %q, want original %q", secondRun.RunID, firstRun.RunID)
	}

	events, err := mem.ListRunEvents(context.Background(), firstRun.RunID, 10)
	if err != nil {
		t.Fatalf("ListRunEvents: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("events = %d, want exactly one accepted event", len(events))
	}
	select {
	case <-bus.Jobs():
	case <-time.After(time.Second):
		t.Fatalf("expected first job")
	}
	select {
	case job := <-bus.Jobs():
		t.Fatalf("unexpected duplicate job: %+v", job)
	default:
	}
}

func TestListRunEventsSupportsAfterSequenceCursor(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "long trace",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "run a long trace",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "run a long trace"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	for idx := 0; idx < 5; idx++ {
		if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
			RunID:     run.RunID,
			ThreadID:  thread.ThreadID,
			EventKind: "message.delta",
			Message:   "chunk",
			Payload:   domain.JSONMap{"idx": idx},
		}); err != nil {
			t.Fatalf("AppendRunEvent %d: %v", idx, err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?limit=2&after_sequence=3", nil)
	req.Header.Set("X-Ultra-User-Id", "user-1")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("events status = %d body=%s", rec.Code, rec.Body.String())
	}

	var response runEventsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode events: %v", err)
	}
	if response.Count != 2 {
		t.Fatalf("count = %d, want 2", response.Count)
	}
	got := []int64{response.Events[0].Sequence, response.Events[1].Sequence}
	want := []int64{4, 5}
	if got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("sequences = %v, want %v", got, want)
	}
}

func TestCancelRunPublishesCanceledEventAndCancelSignal(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(context.Background(), runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "cancel",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(context.Background(), runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainRunEvents(bus)

	req := httptest.NewRequest(http.MethodPost, "/v2/runs/"+run.RunID+"/cancel", strings.NewReader(`{"reason":"user requested"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Ultra-User-Id", "user-1")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("cancel status = %d body=%s", rec.Code, rec.Body.String())
	}

	select {
	case event := <-bus.Events():
		if event.EventKind != "run.canceled" || event.RunID != run.RunID {
			t.Fatalf("event = %+v, want run.canceled for run", event)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected canceled event fanout")
	}
	select {
	case cancel := <-bus.Cancellations():
		if cancel.RunID != run.RunID || cancel.Reason != "user requested" {
			t.Fatalf("cancel signal = %+v, want run/reason", cancel)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected cancel signal")
	}
}

func TestAdminRequeueRunPublishesRecoveryJob(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{
		UserID: "user-1",
		Title:  "requeue",
	})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "long run",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "long run"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	drainJobs(bus)
	drainRunEvents(bus)
	if _, err := service.IngestRunEvent(ctx, domain.AppendRunEventInput{
		EventID:   "evt_" + run.RunID + "_started",
		RunID:     run.RunID,
		ThreadID:  thread.ThreadID,
		EventKind: "run.started",
	}); err != nil {
		t.Fatalf("IngestRunEvent started: %v", err)
	}
	drainRunEvents(bus)

	req := httptest.NewRequest(http.MethodPost, "/v2/admin/runs/"+run.RunID+"/requeue", strings.NewReader(`{"reason":"expired lease"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin requeue status = %d body=%s", rec.Code, rec.Body.String())
	}
	var response adminRunActionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode requeue response: %v", err)
	}
	if response.RunID != run.RunID || response.Status != string(domain.RunStatusRunning) || !response.Updated {
		t.Fatalf("requeue response = %+v, want running updated response", response)
	}
	select {
	case job := <-bus.Jobs():
		if job.RunID != run.RunID || job.DispatchID == "" {
			t.Fatalf("job = %+v, want original run with fresh dispatch id", job)
		}
		if got := job.Metadata["requeue_reason"]; got != "expired lease" {
			t.Fatalf("job requeue reason = %#v, want expired lease", got)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected requeued job")
	}
	select {
	case event := <-bus.Events():
		if event.EventKind != "run.requeued" || event.RunID != run.RunID {
			t.Fatalf("event = %+v, want run.requeued", event)
		}
	case <-time.After(time.Second):
		t.Fatalf("expected run.requeued fanout")
	}
}

func TestArtifactDownloadServesFilesUnderArtifactRootAndRejectsTraversal(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	artifactRoot := t.TempDir()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:      "test-version",
		Runs:         service,
		Store:        mem,
		ArtifactRoot: artifactRoot,
	})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "user-1", Title: "artifacts"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "artifact",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}

	reportPath := filepath.Join(artifactRoot, run.RunID, "report.md")
	if err := os.MkdirAll(filepath.Dir(reportPath), 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	if err := os.WriteFile(reportPath, []byte("# RareSpot report\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	artifact, err := mem.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:    run.RunID,
		ThreadID: thread.ThreadID,
		Kind:     "report",
		Path:     "report.md",
		MimeType: "text/markdown",
		Title:    "RareSpot report",
	})
	if err != nil {
		t.Fatalf("CreateArtifact report: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v2/artifacts/"+artifact.ArtifactID+"/download", nil)
	req.Header.Set("X-Ultra-User-Id", "user-1")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("download status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "RareSpot report") {
		t.Fatalf("download body = %q, want report content", rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); !strings.Contains(got, "text/markdown") {
		t.Fatalf("content type = %q, want markdown", got)
	}

	pathReq := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/artifacts/download?path=report.md", nil)
	pathReq.Header.Set("X-Ultra-User-Id", "user-1")
	pathRec := httptest.NewRecorder()
	router.ServeHTTP(pathRec, pathReq)
	if pathRec.Code != http.StatusOK {
		t.Fatalf("path download status = %d body=%s", pathRec.Code, pathRec.Body.String())
	}
	if !strings.Contains(pathRec.Body.String(), "RareSpot report") {
		t.Fatalf("path download body = %q, want report content", pathRec.Body.String())
	}

	traversal, err := mem.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:    run.RunID,
		ThreadID: thread.ThreadID,
		Kind:     "report",
		Path:     "../secret.md",
		MimeType: "text/markdown",
	})
	if err != nil {
		t.Fatalf("CreateArtifact traversal: %v", err)
	}
	traversalReq := httptest.NewRequest(http.MethodGet, "/v2/artifacts/"+traversal.ArtifactID+"/download", nil)
	traversalReq.Header.Set("X-Ultra-User-Id", "user-1")
	traversalRec := httptest.NewRecorder()
	router.ServeHTTP(traversalRec, traversalReq)
	if traversalRec.Code != http.StatusBadRequest {
		t.Fatalf("traversal status = %d body=%s, want 400", traversalRec.Code, traversalRec.Body.String())
	}
}

func TestArtifactPromotionCopiesArtifactIntoResourceCatalog(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	artifactRoot := t.TempDir()
	uploadRoot := t.TempDir()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:      "test-version",
		Runs:         service,
		Store:        mem,
		Bus:          bus,
		ArtifactRoot: artifactRoot,
		UploadRoot:   uploadRoot,
	})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "user-1", Title: "artifact promote"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "user-1",
		Goal:     "artifact promote",
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	artifactPath := filepath.Join(artifactRoot, run.RunID, "outputs", "detections.csv")
	if err := os.MkdirAll(filepath.Dir(artifactPath), 0o755); err != nil {
		t.Fatalf("MkdirAll artifact: %v", err)
	}
	if err := os.WriteFile(artifactPath, []byte("x,y\n1,2\n"), 0o644); err != nil {
		t.Fatalf("WriteFile artifact: %v", err)
	}
	artifact, err := mem.CreateArtifact(ctx, domain.CreateArtifactInput{
		RunID:    run.RunID,
		ThreadID: thread.ThreadID,
		Kind:     "table",
		Path:     "outputs/detections.csv",
		MimeType: "text/csv",
		Title:    "Detections",
	})
	if err != nil {
		t.Fatalf("CreateArtifact: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v2/artifacts/"+artifact.ArtifactID+"/promote-resource", strings.NewReader(`{"original_name":"saved-detections.csv"}`))
	req.Header.Set("X-Ultra-User-Id", "user-1")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("promote status = %d body=%s", rec.Code, rec.Body.String())
	}
	var promoted resourceResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &promoted); err != nil {
		t.Fatalf("decode promoted resource: %v", err)
	}
	if promoted.Resource.FileID == "" || promoted.Resource.OriginalName != "saved-detections.csv" {
		t.Fatalf("promoted resource = %+v, want saved filename", promoted.Resource)
	}
	if promoted.Resource.SourceType != "artifact" || promoted.Resource.ResourceKind != "table" {
		t.Fatalf("promoted source/kind = %+v, want artifact table", promoted.Resource)
	}
	if !strings.Contains(promoted.Resource.SourceURI, artifact.ArtifactID) {
		t.Fatalf("promoted source_uri = %q, want artifact id", promoted.Resource.SourceURI)
	}
	copiedPath := filepath.Join(uploadRoot, promoted.Resource.FileID+"__saved-detections.csv")
	copied, err := os.ReadFile(copiedPath)
	if err != nil {
		t.Fatalf("read promoted upload copy: %v", err)
	}
	if string(copied) != "x,y\n1,2\n" {
		t.Fatalf("copied artifact content = %q", string(copied))
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v2/resources?source=artifact", nil)
	listReq.Header.Set("X-Ultra-User-Id", "user-1")
	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list promoted status = %d body=%s", listRec.Code, listRec.Body.String())
	}
	var list resourcesResponse
	if err := json.Unmarshal(listRec.Body.Bytes(), &list); err != nil {
		t.Fatalf("decode resources: %v", err)
	}
	if list.Count != 1 || len(list.Resources) != 1 || list.Resources[0].FileID != promoted.Resource.FileID {
		t.Fatalf("resources = %+v, want promoted resource only", list)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v2/resources/"+promoted.Resource.FileID+"/events", nil)
	eventsReq.Header.Set("X-Ultra-User-Id", "user-1")
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("events status = %d body=%s", eventsRec.Code, eventsRec.Body.String())
	}
	var events resourceEventsResponse
	if err := json.Unmarshal(eventsRec.Body.Bytes(), &events); err != nil {
		t.Fatalf("decode promoted resource events: %v", err)
	}
	promotionEvent, ok := resourceEventByType(events.Events, "resource.artifact_promoted")
	if !ok {
		t.Fatalf("promoted resource events = %+v, want resource.artifact_promoted", events.Events)
	}
	if !resourceEventsContain(events.Events, "resource.promoted") {
		t.Fatalf("promoted resource events = %+v, want resource.promoted catalog event", events.Events)
	}
	wantMetadata := map[string]string{
		"artifact_id":            artifact.ArtifactID,
		"run_id":                 run.RunID,
		"artifact_kind":          artifact.Kind,
		"artifact_path":          artifact.Path,
		"artifact_title":         artifact.Title,
		"artifact_mime_type":     artifact.MimeType,
		"promoted_original_name": "saved-detections.csv",
		"source_uri":             promoted.Resource.SourceURI,
		"resource_kind":          promoted.Resource.ResourceKind,
	}
	for key, want := range wantMetadata {
		if got, _ := promotionEvent.Metadata[key].(string); got != want {
			t.Fatalf("resource.artifact_promoted metadata[%q] = %#v, want %q; metadata=%+v", key, promotionEvent.Metadata[key], want, promotionEvent.Metadata)
		}
	}

	bobReq := httptest.NewRequest(http.MethodPost, "/v2/artifacts/"+artifact.ArtifactID+"/promote-resource", nil)
	bobReq.Header.Set("X-Ultra-User-Id", "bob")
	bobRec := httptest.NewRecorder()
	router.ServeHTTP(bobRec, bobReq)
	if bobRec.Code != http.StatusNotFound {
		t.Fatalf("bob promote status = %d body=%s, want 404", bobRec.Code, bobRec.Body.String())
	}
}

func TestAdminResourceReconcilerDetectsUploadCatalogDrift(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	uploadRoot := t.TempDir()

	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		Runs:       service,
		Store:      mem,
		Bus:        bus,
		UploadRoot: uploadRoot,
	})

	orphanPath := filepath.Join(uploadRoot, "file_orphan__orphan.txt")
	if err := os.WriteFile(orphanPath, []byte("orphan"), 0o644); err != nil {
		t.Fatalf("WriteFile orphan: %v", err)
	}

	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_missing",
		OriginalName: "missing.txt",
		ContentType:  "text/plain",
		StoragePath:  "file_missing__missing.txt",
		SourceType:   "upload",
		ResourceKind: "file",
		OwnerUserID:  "local-user",
		Status:       "active",
	}); err != nil {
		t.Fatalf("UpsertResource missing: %v", err)
	}

	driftPath := filepath.Join(uploadRoot, "file_drift__drift.txt")
	if err := os.WriteFile(driftPath, []byte("new-bytes"), 0o644); err != nil {
		t.Fatalf("WriteFile drift: %v", err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_drift",
		OriginalName: "drift.txt",
		ContentType:  "text/plain",
		StoragePath:  filepath.Base(driftPath),
		SourceType:   "upload",
		ResourceKind: "file",
		OwnerUserID:  "local-user",
		Status:       "active",
		SHA256:       strings.Repeat("0", 64),
	}); err != nil {
		t.Fatalf("UpsertResource drift: %v", err)
	}

	previewPath := filepath.Join(uploadRoot, "file_preview__preview.png")
	previewBytes := []byte("not-a-png")
	if err := os.WriteFile(previewPath, previewBytes, 0o644); err != nil {
		t.Fatalf("WriteFile preview: %v", err)
	}
	previewSHA, err := sha256File(previewPath)
	if err != nil {
		t.Fatalf("sha preview: %v", err)
	}
	if _, err := mem.UpsertResource(ctx, domain.UpsertResourceInput{
		ResourceID:   "file_preview",
		OriginalName: "preview.png",
		ContentType:  "image/png",
		StoragePath:  filepath.Base(previewPath),
		SourceType:   "upload",
		ResourceKind: "image",
		OwnerUserID:  "local-user",
		Status:       "active",
		SizeBytes:    int64(len(previewBytes)),
		SHA256:       previewSHA,
	}); err != nil {
		t.Fatalf("UpsertResource preview: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v2/admin/resources/reconcile", nil)
	req.Header.Set("X-Ultra-User-Id", "local-user")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("reconcile status = %d body=%s", rec.Code, rec.Body.String())
	}
	var response resourceReconcileResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode reconcile response: %v", err)
	}
	issueTypes := map[string]bool{}
	for _, issue := range response.Issues {
		issueTypes[issue.IssueType] = true
	}
	for _, want := range []string{"missing_catalog_row", "missing_sidecar", "missing_blob", "checksum_drift", "failed_preview"} {
		if !issueTypes[want] {
			t.Fatalf("issue types = %#v, missing %s; response=%+v", issueTypes, want, response)
		}
	}
	if response.IssueCount != len(response.Issues) || response.Summary["checksum_drift"] != 1 {
		t.Fatalf("reconcile response = %+v, want issue count and drift summary", response)
	}
}

func jsonArrayEquals(value any, want []string) bool {
	values, ok := value.([]any)
	if !ok || len(values) != len(want) {
		return false
	}
	for index, item := range values {
		if item != want[index] {
			return false
		}
	}
	return true
}

func drainRunEvents(bus *eventbus.MemoryBus) {
	for {
		select {
		case <-bus.Events():
		default:
			return
		}
	}
}

func drainJobs(bus *eventbus.MemoryBus) {
	for {
		select {
		case <-bus.Jobs():
		default:
			return
		}
	}
}

func linkedBisqueSessionIDFromCookie(t *testing.T, cookies []*http.Cookie) string {
	t.Helper()
	for _, cookie := range cookies {
		if cookie.Name != "ultra_dev_auth" {
			continue
		}
		value := cookie.Value
		if decoded, err := url.QueryUnescape(value); err == nil {
			value = decoded
		}
		sessionID, ok := strings.CutPrefix(value, "bisque_session:")
		if !ok {
			t.Fatalf("auth cookie = %q, want bisque_session cookie", value)
		}
		if strings.TrimSpace(sessionID) == "" {
			t.Fatalf("auth cookie has empty BisQue session id")
		}
		return strings.TrimSpace(sessionID)
	}
	t.Fatalf("missing ultra_dev_auth cookie")
	return ""
}
