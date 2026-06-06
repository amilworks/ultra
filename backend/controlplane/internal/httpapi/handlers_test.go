package httpapi

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
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
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

type fakeQueueDiagnosticsProvider struct {
	diagnostics eventbus.QueueDiagnostics
	err         error
}

func (p fakeQueueDiagnosticsProvider) QueueDiagnostics(context.Context) (eventbus.QueueDiagnostics, error) {
	return p.diagnostics, p.err
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

func TestNiftiUploadViewerServesSelectedScalarChannel(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:    "test-version",
		UploadRoot: uploadRoot,
	})
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
	if viewerResponse.DimsOrder != "CZYX" || !viewerResponse.IsMultichannel {
		t.Fatalf("viewer dims/multichannel = %q/%v, want CZYX true", viewerResponse.DimsOrder, viewerResponse.IsMultichannel)
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
		t.Fatalf("array_shape = %v, want [2 2 1 2]", viewerResponse.Metadata.ArrayShape)
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

	sliceReq := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/slice?axis=z&z=1&channels=1&window_min=100&window_max=400", nil)
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
		t.Fatalf("selected channel slice pixels = %d,%d, want 170,255", gotFirst, gotLast)
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
	binary.LittleEndian.PutUint16(output[40:42], 4)
	binary.LittleEndian.PutUint16(output[42:44], uint16(width))
	binary.LittleEndian.PutUint16(output[44:46], uint16(height))
	binary.LittleEndian.PutUint16(output[46:48], uint16(depth))
	binary.LittleEndian.PutUint16(output[48:50], uint16(channels))
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
