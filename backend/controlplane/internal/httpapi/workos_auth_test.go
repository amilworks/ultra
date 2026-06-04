package httpapi

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

const testWorkOSCookiePassword = "test-workos-cookie-password-for-ultra-control-plane"

func TestWorkOSLoginReturnsAuthKitAuthorizationURLAndStateCookie(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{
		WorkOS: testWorkOSAuth(t, WorkOSAuthConfig{
			BaseURL:     "https://workos.example.test",
			RedirectURI: "https://ultra.example.test/v2/auth/workos/callback",
		}),
	})

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v2/auth/login", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("login status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}

	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode login response: %v", err)
	}
	if body["provider"] != "workos" || body["mode"] != "workos" {
		t.Fatalf("login response = %#v, want WorkOS provider/mode", body)
	}
	authorizationURL := strings.TrimSpace(fmt.Sprint(body["authorization_url"]))
	if authorizationURL == "" {
		t.Fatalf("login response = %#v, want authorization_url", body)
	}
	parsed, err := url.Parse(authorizationURL)
	if err != nil {
		t.Fatalf("parse authorization_url: %v", err)
	}
	if parsed.Host != "workos.example.test" || parsed.Path != "/user_management/authorize" {
		t.Fatalf("authorization_url = %s, want WorkOS authorize endpoint", authorizationURL)
	}
	query := parsed.Query()
	for key, want := range map[string]string{
		"client_id":             "client_test",
		"redirect_uri":          "https://ultra.example.test/v2/auth/workos/callback",
		"response_type":         "code",
		"code_challenge_method": "S256",
		"provider":              "authkit",
	} {
		if got := query.Get(key); got != want {
			t.Fatalf("authorization_url query %s = %q, want %q", key, got, want)
		}
	}
	if query.Get("state") == "" || query.Get("code_challenge") == "" {
		t.Fatalf("authorization_url query = %s, want state and code_challenge", parsed.RawQuery)
	}
	if findCookie(rec.Result().Cookies(), workOSStateCookieName) == nil {
		t.Fatalf("login did not set %s cookie", workOSStateCookieName)
	}
}

func TestWorkOSCallbackMintsSessionAndSessionEndpointReturnsPrincipal(t *testing.T) {
	t.Parallel()

	var authenticateBody map[string]any
	workosServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/user_management/authenticate" {
			t.Fatalf("unexpected WorkOS request %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer sk_test" {
			t.Fatalf("WorkOS Authorization = %q, want bearer API key", got)
		}
		data, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read WorkOS request body: %v", err)
		}
		if err := json.Unmarshal(data, &authenticateBody); err != nil {
			t.Fatalf("decode WorkOS request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{
			"access_token": %q,
			"refresh_token": "refresh_test",
			"user": {
				"id": "user_123",
				"email": "scientist@example.org",
				"first_name": "Ada",
				"last_name": "Lovelace",
				"email_verified": true
			},
			"organization_id": "org_456"
		}`, testWorkOSJWT("sess_789", "org_456", "admin", time.Now().Add(time.Hour)))
	}))
	defer workosServer.Close()

	mem := store.NewMemoryStore()
	if _, err := mem.CreateUser(context.Background(), domain.CreateUserInput{
		UserID: "workos:user_123",
		Email:  "scientist@example.org",
		Role:   "admin",
		Status: "active",
		OrgID:  "org_456",
	}); err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
	router := NewRouter(ServerDeps{
		Store: mem,
		WorkOS: testWorkOSAuth(t, WorkOSAuthConfig{
			BaseURL:              workosServer.URL,
			RedirectURI:          "https://ultra.example.test/v2/auth/workos/callback",
			PostLoginRedirectURI: "https://ultra.example.test/",
		}),
	})

	loginRec := httptest.NewRecorder()
	router.ServeHTTP(loginRec, httptest.NewRequest(http.MethodPost, "/v2/auth/login", nil))
	if loginRec.Code != http.StatusOK {
		t.Fatalf("login status = %d body=%s", loginRec.Code, loginRec.Body.String())
	}
	var loginBody map[string]any
	if err := json.Unmarshal(loginRec.Body.Bytes(), &loginBody); err != nil {
		t.Fatalf("decode login response: %v", err)
	}
	authorizationURL, err := url.Parse(strings.TrimSpace(fmt.Sprint(loginBody["authorization_url"])))
	if err != nil {
		t.Fatalf("parse authorization_url: %v", err)
	}
	state := authorizationURL.Query().Get("state")
	if state == "" {
		t.Fatalf("missing state in authorization_url")
	}

	callbackReq := httptest.NewRequest(http.MethodGet, "/v2/auth/workos/callback?code=code_abc&state="+url.QueryEscape(state), nil)
	for _, cookie := range loginRec.Result().Cookies() {
		callbackReq.AddCookie(cookie)
	}
	callbackRec := httptest.NewRecorder()
	router.ServeHTTP(callbackRec, callbackReq)
	if callbackRec.Code != http.StatusFound {
		t.Fatalf("callback status = %d body=%s, want 302", callbackRec.Code, callbackRec.Body.String())
	}
	if got := callbackRec.Result().Header.Get("Location"); got != "https://ultra.example.test/" {
		t.Fatalf("callback Location = %q, want app redirect", got)
	}
	if authenticateBody["grant_type"] != "authorization_code" || authenticateBody["code"] != "code_abc" {
		t.Fatalf("WorkOS authenticate body = %#v, want authorization code exchange", authenticateBody)
	}
	if authenticateBody["client_id"] != "client_test" || authenticateBody["client_secret"] != "sk_test" {
		t.Fatalf("WorkOS authenticate body = %#v, want client credentials", authenticateBody)
	}
	if strings.TrimSpace(fmt.Sprint(authenticateBody["code_verifier"])) == "" {
		t.Fatalf("WorkOS authenticate body = %#v, want PKCE code_verifier", authenticateBody)
	}
	sessionCookie := findCookie(callbackRec.Result().Cookies(), workOSSessionCookieName)
	if sessionCookie == nil || strings.TrimSpace(sessionCookie.Value) == "" {
		t.Fatalf("callback did not set %s cookie", workOSSessionCookieName)
	}
	if stateCookie := findCookie(callbackRec.Result().Cookies(), workOSStateCookieName); stateCookie == nil || stateCookie.MaxAge >= 0 {
		t.Fatalf("callback did not clear %s cookie", workOSStateCookieName)
	}

	sessionReq := httptest.NewRequest(http.MethodGet, "/v2/auth/session", nil)
	sessionReq.AddCookie(sessionCookie)
	sessionRec := httptest.NewRecorder()
	router.ServeHTTP(sessionRec, sessionReq)
	if sessionRec.Code != http.StatusOK {
		t.Fatalf("session status = %d body=%s", sessionRec.Code, sessionRec.Body.String())
	}
	var session map[string]any
	if err := json.Unmarshal(sessionRec.Body.Bytes(), &session); err != nil {
		t.Fatalf("decode session response: %v", err)
	}
	if session["authenticated"] != true || session["mode"] != "workos" {
		t.Fatalf("session = %#v, want authenticated WorkOS session", session)
	}
	if session["username"] != "scientist@example.org" {
		t.Fatalf("session username = %#v, want WorkOS email", session["username"])
	}
	user, ok := session["user"].(map[string]any)
	if !ok {
		t.Fatalf("session user = %#v, want object", session["user"])
	}
	for key, want := range map[string]string{
		"id":       "workos:user_123",
		"email":    "scientist@example.org",
		"org_id":   "org_456",
		"role":     "admin",
		"username": "scientist@example.org",
	} {
		if got := fmt.Sprint(user[key]); got != want {
			t.Fatalf("session user[%s] = %q, want %q", key, got, want)
		}
	}
}

func TestWorkOSSessionDrivesRequestPrincipalAndResourceOwnership(t *testing.T) {
	t.Parallel()

	uploadRoot := t.TempDir()
	bus := eventbus.NewMemoryBus()
	memoryStore := store.NewMemoryStore()
	if _, err := memoryStore.CreateUser(context.Background(), domain.CreateUserInput{
		UserID: "workos:user_123",
		Email:  "scientist@example.org",
		Role:   "admin",
		Status: "active",
		OrgID:  "org_456",
	}); err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
	router := NewRouter(ServerDeps{
		Runs:       runcontrol.NewService(memoryStore, bus),
		Store:      memoryStore,
		Bus:        bus,
		UploadRoot: uploadRoot,
		WorkOS: testWorkOSAuth(t, WorkOSAuthConfig{
			BaseURL: "https://workos.example.test",
		}),
	})
	sessionCookie := testWorkOSSessionCookie(t, "user_123", "scientist@example.org", "org_456", "admin")

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("files", "specimen.png")
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
	if len(uploadResponse.Uploaded) != 1 {
		t.Fatalf("uploaded = %+v, want one file", uploadResponse.Uploaded)
	}
	if got := uploadResponse.Uploaded[0].Principal.UserID; got != "workos:user_123" {
		t.Fatalf("uploaded principal user = %q, want WorkOS user", got)
	}
	if got := uploadResponse.Uploaded[0].Principal.OrgID; got != "org_456" {
		t.Fatalf("uploaded principal org = %q, want WorkOS org", got)
	}
}

func TestWorkOSSessionCreatesActiveUltraAccountAndAllowsAppAccess(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	router := NewRouter(ServerDeps{
		Runs:  runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store: mem,
		WorkOS: testWorkOSAuth(t, WorkOSAuthConfig{
			BaseURL: "https://workos.example.test",
		}),
	})
	sessionCookie := testWorkOSSessionCookie(t, "user_pending", "pending@example.org", "org_456", "researcher")

	sessionReq := httptest.NewRequest(http.MethodGet, "/v2/auth/session", nil)
	sessionReq.AddCookie(sessionCookie)
	sessionRec := httptest.NewRecorder()
	router.ServeHTTP(sessionRec, sessionReq)
	if sessionRec.Code != http.StatusOK {
		t.Fatalf("session status = %d body=%s", sessionRec.Code, sessionRec.Body.String())
	}
	var session map[string]any
	if err := json.Unmarshal(sessionRec.Body.Bytes(), &session); err != nil {
		t.Fatalf("decode session: %v", err)
	}
	if session["authenticated"] != true || session["account_status"] != "active" {
		t.Fatalf("session = %#v, want authenticated active account", session)
	}

	users, err := mem.ListUsers(context.Background(), 10, "pending@example.org")
	if err != nil {
		t.Fatalf("ListUsers: %v", err)
	}
	if len(users) != 1 {
		t.Fatalf("users = %+v, want one auto-created active WorkOS account", users)
	}
	if users[0].UserID != "workos:user_pending" || users[0].Status != "active" {
		t.Fatalf("WorkOS account = %+v, want workos:user_pending active", users[0])
	}

	threadsReq := httptest.NewRequest(http.MethodGet, "/v2/threads", nil)
	threadsReq.AddCookie(sessionCookie)
	threadsRec := httptest.NewRecorder()
	router.ServeHTTP(threadsRec, threadsReq)
	if threadsRec.Code != http.StatusOK {
		t.Fatalf("threads status = %d body=%s, want authenticated access", threadsRec.Code, threadsRec.Body.String())
	}
}

func TestWorkOSSessionUsesActiveUltraAccountForRoleAndOrg(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	if _, err := mem.CreateUser(context.Background(), domain.CreateUserInput{
		UserID:      "ultra-ada",
		Email:       "scientist@example.org",
		DisplayName: "Ada Lovelace",
		Role:        "admin",
		Status:      "active",
		OrgID:       "approved-org",
	}); err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
	router := NewRouter(ServerDeps{
		Runs:  runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store: mem,
		WorkOS: testWorkOSAuth(t, WorkOSAuthConfig{
			BaseURL: "https://workos.example.test",
		}),
	})
	sessionCookie := testWorkOSSessionCookie(t, "user_123", "scientist@example.org", "workos-org", "researcher")

	sessionReq := httptest.NewRequest(http.MethodGet, "/v2/auth/session", nil)
	sessionReq.AddCookie(sessionCookie)
	sessionRec := httptest.NewRecorder()
	router.ServeHTTP(sessionRec, sessionReq)
	if sessionRec.Code != http.StatusOK {
		t.Fatalf("session status = %d body=%s", sessionRec.Code, sessionRec.Body.String())
	}
	var session map[string]any
	if err := json.Unmarshal(sessionRec.Body.Bytes(), &session); err != nil {
		t.Fatalf("decode session: %v", err)
	}
	if session["authenticated"] != true || session["account_status"] != "active" || session["is_admin"] != true {
		t.Fatalf("session = %#v, want authenticated active admin account", session)
	}
	user, ok := session["user"].(map[string]any)
	if !ok {
		t.Fatalf("session user = %#v, want object", session["user"])
	}
	if user["id"] != "ultra-ada" || user["role"] != "admin" || user["org_id"] != "approved-org" {
		t.Fatalf("session user = %#v, want Ultra account principal", user)
	}

	adminReq := httptest.NewRequest(http.MethodGet, "/v2/admin/users", nil)
	adminReq.AddCookie(sessionCookie)
	adminRec := httptest.NewRecorder()
	router.ServeHTTP(adminRec, adminReq)
	if adminRec.Code != http.StatusOK {
		t.Fatalf("admin users status = %d body=%s, want active admin access", adminRec.Code, adminRec.Body.String())
	}
}

func TestWorkOSActiveResearcherCannotAccessAdminRoutes(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	if _, err := mem.CreateUser(context.Background(), domain.CreateUserInput{
		UserID: "workos:user_researcher",
		Email:  "researcher@example.org",
		Role:   "researcher",
		Status: "active",
		OrgID:  "approved-org",
	}); err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
	router := NewRouter(ServerDeps{
		Runs:  runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store: mem,
		WorkOS: testWorkOSAuth(t, WorkOSAuthConfig{
			BaseURL: "https://workos.example.test",
		}),
	})
	sessionCookie := testWorkOSSessionCookie(t, "user_researcher", "researcher@example.org", "approved-org", "admin")

	adminReq := httptest.NewRequest(http.MethodGet, "/v2/admin/overview", nil)
	adminReq.AddCookie(sessionCookie)
	adminRec := httptest.NewRecorder()
	router.ServeHTTP(adminRec, adminReq)
	if adminRec.Code != http.StatusForbidden {
		t.Fatalf("admin overview status = %d body=%s, want 403 for non-admin Ultra account", adminRec.Code, adminRec.Body.String())
	}
}

func TestWorkOSDisabledUltraAccountDoesNotBlockAuthKitAccess(t *testing.T) {
	t.Parallel()

	mem := store.NewMemoryStore()
	if _, err := mem.CreateUser(context.Background(), domain.CreateUserInput{
		UserID: "workos:user_disabled",
		Email:  "disabled@example.org",
		Role:   "admin",
		Status: "disabled",
		OrgID:  "approved-org",
	}); err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
	router := NewRouter(ServerDeps{
		Runs:  runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store: mem,
		WorkOS: testWorkOSAuth(t, WorkOSAuthConfig{
			BaseURL: "https://workos.example.test",
		}),
	})
	sessionCookie := testWorkOSSessionCookie(t, "user_disabled", "disabled@example.org", "approved-org", "admin")

	sessionReq := httptest.NewRequest(http.MethodGet, "/v2/auth/session", nil)
	sessionReq.AddCookie(sessionCookie)
	sessionRec := httptest.NewRecorder()
	router.ServeHTTP(sessionRec, sessionReq)
	if sessionRec.Code != http.StatusOK {
		t.Fatalf("session status = %d body=%s", sessionRec.Code, sessionRec.Body.String())
	}
	var session map[string]any
	if err := json.Unmarshal(sessionRec.Body.Bytes(), &session); err != nil {
		t.Fatalf("decode session: %v", err)
	}
	if session["authenticated"] != true || session["account_status"] != "disabled" {
		t.Fatalf("session = %#v, want authenticated WorkOS session with stored disabled status", session)
	}

	resourcesReq := httptest.NewRequest(http.MethodGet, "/v2/resources", nil)
	resourcesReq.AddCookie(sessionCookie)
	resourcesRec := httptest.NewRecorder()
	router.ServeHTTP(resourcesRec, resourcesReq)
	if resourcesRec.Code != http.StatusOK {
		t.Fatalf("resources status = %d body=%s, want AuthKit-controlled access", resourcesRec.Code, resourcesRec.Body.String())
	}
}

func TestWorkOSProtectedRoutesRejectUnauthenticatedRequests(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{
		WorkOS: testWorkOSAuth(t, WorkOSAuthConfig{
			BaseURL: "https://workos.example.test",
		}),
	})

	for _, tc := range []struct {
		method string
		path   string
	}{
		{method: http.MethodGet, path: "/v2/resources?limit=20"},
		{method: http.MethodPost, path: "/v2/uploads"},
		{method: http.MethodGet, path: "/v2/threads"},
	} {
		req := httptest.NewRequest(tc.method, tc.path, nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("%s %s status = %d body=%s, want 401", tc.method, tc.path, rec.Code, rec.Body.String())
		}
	}
}

func TestWorkOSLogoutClearsSessionCookieAndReturnsLogoutURL(t *testing.T) {
	t.Parallel()

	router := NewRouter(ServerDeps{
		WorkOS: testWorkOSAuth(t, WorkOSAuthConfig{
			BaseURL:              "https://workos.example.test",
			LogoutRedirectURI:    "https://ultra.example.test/auth/signed-out",
			PostLoginRedirectURI: "https://ultra.example.test/",
		}),
	})
	sessionCookie := testWorkOSSessionCookie(t, "user_123", "scientist@example.org", "org_456", "admin")

	req := httptest.NewRequest(http.MethodPost, "/v2/auth/logout", nil)
	req.AddCookie(sessionCookie)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("logout status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode logout response: %v", err)
	}
	logoutURL := strings.TrimSpace(fmt.Sprint(body["logout_url"]))
	if !strings.HasPrefix(logoutURL, "https://workos.example.test/user_management/sessions/logout?") {
		t.Fatalf("logout_url = %q, want WorkOS logout URL", logoutURL)
	}
	parsed, err := url.Parse(logoutURL)
	if err != nil {
		t.Fatalf("parse logout_url: %v", err)
	}
	if got := parsed.Query().Get("session_id"); got != "sess_user_123" {
		t.Fatalf("logout session_id = %q, want sess_user_123", got)
	}
	if got := parsed.Query().Get("return_to"); got != "https://ultra.example.test/auth/signed-out" {
		t.Fatalf("logout return_to = %q, want configured redirect", got)
	}
	if cleared := findCookie(rec.Result().Cookies(), workOSSessionCookieName); cleared == nil || cleared.MaxAge >= 0 {
		t.Fatalf("logout did not clear %s cookie", workOSSessionCookieName)
	}
}

func TestWorkOSAuthenticatedUserCanLinkBisqueCredentials(t *testing.T) {
	t.Parallel()

	var bisque *httptest.Server
	bisque = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		if r.URL.Path == "/auth_service/session" {
			_, _ = w.Write([]byte(`<response><tag name="user" value="bisque-user"/></response>`))
			return
		}
		_, _ = w.Write([]byte(`<response><image uri="` + bisque.URL + `/data_service/image/linked" name="linked.jpg" resource_uniq="linked"/></response>`))
	}))
	defer bisque.Close()

	credentialStore := NewBisqueCredentialStore()
	mem := store.NewMemoryStore()
	if _, err := mem.CreateUser(context.Background(), domain.CreateUserInput{
		UserID: "workos:user_123",
		Email:  "scientist@example.org",
		Role:   "researcher",
		Status: "active",
		OrgID:  "org_456",
	}); err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
	router := NewRouter(ServerDeps{
		BisqueCredentials: credentialStore,
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       bisque.URL,
			AllowedRoots:  []string{bisque.URL},
			HTTPClient:    bisque.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
		Store: mem,
		WorkOS: testWorkOSAuth(t, WorkOSAuthConfig{
			BaseURL: "https://workos.example.test",
		}),
	})
	sessionCookie := testWorkOSSessionCookie(t, "user_123", "scientist@example.org", "org_456", "admin")

	req := httptest.NewRequest(http.MethodPost, "/v2/auth/login", strings.NewReader(`{"username":"bisque-user","password":"bisque-secret"}`))
	req.Header.Set("Content-Type", "application/json")
	req.AddCookie(sessionCookie)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("link status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	if findCookie(rec.Result().Cookies(), bisqueSessionCookieName) == nil {
		t.Fatalf("link did not set %s cookie", bisqueSessionCookieName)
	}
	var session map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &session); err != nil {
		t.Fatalf("decode link response: %v", err)
	}
	if session["authenticated"] != true || session["mode"] != "workos" || session["bisque_linked"] != true {
		t.Fatalf("link response = %#v, want authenticated WorkOS session with linked BisQue", session)
	}
	user, ok := session["user"].(map[string]any)
	if !ok || user["id"] != "workos:user_123" || user["org_id"] != "org_456" {
		t.Fatalf("link response user = %#v, want WorkOS principal", session["user"])
	}

	credentials, found, err := credentialStore.GetWithContext(context.Background(), bisqueSessionIDFromRequest(reqWithCookies(rec.Result().Cookies())))
	if err != nil {
		t.Fatalf("GetWithContext: %v", err)
	}
	if !found || credentials.Username != "bisque-user" || credentials.Password != "bisque-secret" {
		t.Fatalf("stored BisQue credentials = %#v/%v, want linked credentials", credentials, found)
	}
}

func testWorkOSAuth(t *testing.T, overrides WorkOSAuthConfig) *WorkOSAuth {
	t.Helper()
	cfg := WorkOSAuthConfig{
		Enabled:        true,
		ClientID:       "client_test",
		APIKey:         "sk_test",
		RedirectURI:    "https://ultra.example.test/v2/auth/workos/callback",
		CookiePassword: testWorkOSCookiePassword,
		CookieSecure:   true,
		BaseURL:        "https://workos.example.test",
	}
	if overrides.ClientID != "" {
		cfg.ClientID = overrides.ClientID
	}
	if overrides.APIKey != "" {
		cfg.APIKey = overrides.APIKey
	}
	if overrides.RedirectURI != "" {
		cfg.RedirectURI = overrides.RedirectURI
	}
	if overrides.CookiePassword != "" {
		cfg.CookiePassword = overrides.CookiePassword
	}
	if overrides.BaseURL != "" {
		cfg.BaseURL = overrides.BaseURL
	}
	if overrides.LogoutRedirectURI != "" {
		cfg.LogoutRedirectURI = overrides.LogoutRedirectURI
	}
	if overrides.PostLoginRedirectURI != "" {
		cfg.PostLoginRedirectURI = overrides.PostLoginRedirectURI
	}
	cfg.CookieSecure = overrides.CookieSecure || cfg.CookieSecure
	auth, err := NewWorkOSAuth(cfg)
	if err != nil {
		t.Fatalf("NewWorkOSAuth: %v", err)
	}
	return auth
}

func reqWithCookies(cookies []*http.Cookie) *http.Request {
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	for _, cookie := range cookies {
		req.AddCookie(cookie)
	}
	return req
}

func testWorkOSSessionCookie(t *testing.T, userID string, email string, orgID string, role string) *http.Cookie {
	t.Helper()
	sealed, err := sealWorkOSSessionForTest(userID, email, orgID, role)
	if err != nil {
		t.Fatalf("seal WorkOS session: %v", err)
	}
	return &http.Cookie{Name: workOSSessionCookieName, Value: sealed, Path: "/"}
}

func sealWorkOSSessionForTest(userID string, email string, orgID string, role string) (string, error) {
	return sealWorkOSSessionValue(
		testWorkOSJWT("sess_"+userID, orgID, role, time.Now().Add(time.Hour)),
		"refresh_"+userID,
		workOSUserSnapshot{
			ID:    userID,
			Email: email,
		},
		nil,
		testWorkOSCookiePassword,
	)
}

func testWorkOSJWT(sessionID string, orgID string, role string, expiresAt time.Time) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"HS256"}`))
	payload := base64.RawURLEncoding.EncodeToString(fmt.Appendf(nil, `{"sid":%q,"org_id":%q,"role":%q,"exp":%d}`, sessionID, orgID, role, expiresAt.Unix()))
	return header + "." + payload + ".fakesig"
}

func findCookie(cookies []*http.Cookie, name string) *http.Cookie {
	for _, cookie := range cookies {
		if cookie.Name == name {
			return cookie
		}
	}
	return nil
}
