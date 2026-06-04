package httpapi

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/workos/workos-go/v8"
)

const (
	workOSSessionCookieName   = "ultra_workos_session"
	workOSStateCookieName     = "ultra_workos_oauth_state"
	workOSStateCookieMaxAge   = 10 * time.Minute
	workOSSessionCookieMaxAge = 7 * 24 * time.Hour
)

type WorkOSAuthConfig struct {
	Enabled              bool
	ClientID             string
	APIKey               string
	RedirectURI          string
	PostLoginRedirectURI string
	LogoutRedirectURI    string
	CookiePassword       string
	CookieSecure         bool
	BaseURL              string
}

type WorkOSAuth struct {
	client               *workos.Client
	enabled              bool
	redirectURI          string
	postLoginRedirectURI string
	logoutRedirectURI    string
	cookiePassword       string
	cookieSecure         bool
	baseURL              string
}

type workOSOAuthState struct {
	State        string `json:"state"`
	CodeVerifier string `json:"code_verifier"`
	CreatedAt    int64  `json:"created_at"`
}

type workOSUserSnapshot struct {
	ID        string
	Email     string
	FirstName *string
	LastName  *string
}

type workOSSessionSnapshot struct {
	Principal     requestPrincipal
	UserID        string
	Email         string
	FirstName     string
	LastName      string
	SessionID     string
	AccountStatus string
	Permissions   []string
	Entitlements  []string
}

type workOSPrincipalContextKey struct{}

func NewWorkOSAuth(cfg WorkOSAuthConfig) (*WorkOSAuth, error) {
	if !cfg.Enabled {
		return nil, nil
	}
	clientID := strings.TrimSpace(cfg.ClientID)
	apiKey := strings.TrimSpace(cfg.APIKey)
	redirectURI := strings.TrimSpace(cfg.RedirectURI)
	cookiePassword := strings.TrimSpace(cfg.CookiePassword)
	if clientID == "" {
		return nil, errors.New("WorkOS client id is required")
	}
	if apiKey == "" {
		return nil, errors.New("WorkOS API key is required")
	}
	if redirectURI == "" {
		return nil, errors.New("WorkOS redirect URI is required")
	}
	if cookiePassword == "" {
		return nil, errors.New("WorkOS cookie password is required")
	}
	options := []workos.ClientOption{
		workos.WithClientID(clientID),
		workos.WithAppInfo("bisque-ultra-control-plane", "dev", "https://github.com/amilworks/bisque-ultra"),
	}
	baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if baseURL != "" {
		options = append(options, workos.WithBaseURL(baseURL))
	}
	return &WorkOSAuth{
		client:               workos.NewClient(apiKey, options...),
		enabled:              true,
		redirectURI:          redirectURI,
		postLoginRedirectURI: firstNonEmpty(strings.TrimSpace(cfg.PostLoginRedirectURI), "/"),
		logoutRedirectURI:    strings.TrimSpace(cfg.LogoutRedirectURI),
		cookiePassword:       cookiePassword,
		cookieSecure:         cfg.CookieSecure,
		baseURL:              baseURL,
	}, nil
}

func (auth *WorkOSAuth) Enabled() bool {
	return auth != nil && auth.enabled
}

func (auth *WorkOSAuth) handleLogin(w http.ResponseWriter, r *http.Request) {
	authKitProvider := string(workos.UserManagementAuthenticationProviderAuthkit)
	result, err := auth.client.GetAuthKitPKCEAuthorizationURL(workos.AuthKitAuthorizationURLParams{
		RedirectURI: auth.redirectURI,
		Provider:    &authKitProvider,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	sealedState, err := workos.Seal(workOSOAuthState{
		State:        result.State,
		CodeVerifier: result.CodeVerifier,
		CreatedAt:    time.Now().Unix(),
	}, auth.cookiePassword)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	auth.setCookie(w, workOSStateCookieName, sealedState, int(workOSStateCookieMaxAge.Seconds()))
	writeJSON(w, http.StatusOK, map[string]any{
		"authenticated":     false,
		"provider":          "workos",
		"mode":              "workos",
		"authorization_url": result.URL,
	})
}

func (auth *WorkOSAuth) handleCallback(w http.ResponseWriter, r *http.Request) {
	code := strings.TrimSpace(r.URL.Query().Get("code"))
	state := strings.TrimSpace(r.URL.Query().Get("state"))
	if code == "" || state == "" {
		writeError(w, http.StatusBadRequest, errors.New("missing WorkOS authorization code or state"))
		return
	}
	stateCookie, err := r.Cookie(workOSStateCookieName)
	if err != nil || strings.TrimSpace(stateCookie.Value) == "" {
		writeError(w, http.StatusBadRequest, errors.New("missing WorkOS state cookie"))
		return
	}
	expected, err := workos.Unseal[workOSOAuthState](stateCookie.Value, auth.cookiePassword)
	if err != nil || expected.State == "" || expected.CodeVerifier == "" {
		writeError(w, http.StatusBadRequest, errors.New("invalid WorkOS state cookie"))
		return
	}
	if time.Since(time.Unix(expected.CreatedAt, 0)) > workOSStateCookieMaxAge {
		writeError(w, http.StatusBadRequest, errors.New("expired WorkOS state cookie"))
		return
	}
	if state != expected.State {
		writeError(w, http.StatusBadRequest, errors.New("invalid WorkOS OAuth state"))
		return
	}
	authResponse, err := auth.client.AuthKitPKCECodeExchange(r.Context(), workos.AuthKitPKCECodeExchangeParams{
		Code:         code,
		CodeVerifier: expected.CodeVerifier,
	})
	if err != nil {
		writeError(w, http.StatusBadGateway, fmt.Errorf("WorkOS code exchange failed: %w", err))
		return
	}
	user := workOSUserSnapshot{}
	if authResponse.User != nil {
		user.ID = authResponse.User.ID
		user.Email = authResponse.User.Email
		user.FirstName = authResponse.User.FirstName
		user.LastName = authResponse.User.LastName
	}
	sealedSession, err := sealWorkOSSessionValue(
		authResponse.AccessToken,
		authResponse.RefreshToken,
		user,
		authResponse.Impersonator,
		auth.cookiePassword,
	)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	auth.setCookie(w, workOSSessionCookieName, sealedSession, int(workOSSessionCookieMaxAge.Seconds()))
	auth.clearCookie(w, workOSStateCookieName)
	http.Redirect(w, r, auth.postLoginRedirectURI, http.StatusFound)
}

func (auth *WorkOSAuth) handleSession(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, auth.sessionResponseForRequest(w, r))
}

func (auth *WorkOSAuth) sessionResponseForRequest(w http.ResponseWriter, r *http.Request) map[string]any {
	snapshot, authenticated := auth.authenticateRequest(w, r)
	if !authenticated {
		return map[string]any{
			"authenticated": false,
			"user":          nil,
			"mode":          "workos",
			"provider":      "workos",
			"bisque_linked": false,
		}
	}
	return snapshot.sessionResponse()
}

func (auth *WorkOSAuth) handleLogout(w http.ResponseWriter, r *http.Request) {
	var logoutURL string
	if cookie, err := r.Cookie(workOSSessionCookieName); err == nil && strings.TrimSpace(cookie.Value) != "" {
		session := workos.NewSession(auth.client, cookie.Value, auth.cookiePassword)
		if url, err := session.GetLogoutURL(r.Context(), auth.logoutRedirectURI); err == nil {
			logoutURL = url
		}
	}
	auth.clearCookie(w, workOSSessionCookieName)
	auth.clearCookie(w, workOSStateCookieName)
	setDevAuthCookie(w, "signed_out")
	writeJSON(w, http.StatusOK, map[string]any{
		"authenticated": false,
		"user":          nil,
		"mode":          "workos",
		"provider":      "workos",
		"bisque_linked": false,
		"logout_url":    logoutURL,
	})
}

func (auth *WorkOSAuth) authenticateRequest(w http.ResponseWriter, r *http.Request) (workOSSessionSnapshot, bool) {
	if snapshot, ok := workOSSnapshotFromContext(r.Context()); ok {
		return snapshot, true
	}
	cookie, err := r.Cookie(workOSSessionCookieName)
	if err != nil || strings.TrimSpace(cookie.Value) == "" {
		return workOSSessionSnapshot{}, false
	}
	session := workos.NewSession(auth.client, cookie.Value, auth.cookiePassword)
	result, err := session.Authenticate()
	if err != nil {
		return workOSSessionSnapshot{}, false
	}
	if result.NeedsRefresh {
		refreshed, refreshErr := session.Refresh(r.Context())
		if refreshErr != nil || !refreshed.Authenticated || strings.TrimSpace(refreshed.SealedSession) == "" {
			return workOSSessionSnapshot{}, false
		}
		auth.setCookie(w, workOSSessionCookieName, refreshed.SealedSession, int(workOSSessionCookieMaxAge.Seconds()))
		session = workos.NewSession(auth.client, refreshed.SealedSession, auth.cookiePassword)
		result, err = session.Authenticate()
		if err != nil {
			return workOSSessionSnapshot{}, false
		}
	}
	if !result.Authenticated || result.User == nil {
		return workOSSessionSnapshot{}, false
	}
	return snapshotFromWorkOSResult(result), true
}

func (auth *WorkOSAuth) principalFromRequest(r *http.Request) (requestPrincipal, bool) {
	if snapshot, ok := workOSSnapshotFromContext(r.Context()); ok {
		return snapshot.Principal, true
	}
	if !auth.Enabled() {
		return requestPrincipal{}, false
	}
	cookie, err := r.Cookie(workOSSessionCookieName)
	if err != nil || strings.TrimSpace(cookie.Value) == "" {
		return requestPrincipal{}, false
	}
	result, err := workos.AuthenticateSession(cookie.Value, auth.cookiePassword)
	if err != nil || !result.Authenticated || result.User == nil {
		return requestPrincipal{}, false
	}
	return snapshotFromWorkOSResult(result).Principal, true
}

func (auth *WorkOSAuth) requireAuth(next http.Handler) http.Handler {
	if !auth.Enabled() {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		snapshot, authenticated := auth.authenticateRequest(w, r)
		if !authenticated {
			writeError(w, http.StatusUnauthorized, errors.New("authentication required"))
			return
		}
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), workOSPrincipalContextKey{}, snapshot)))
	})
}

func (snapshot workOSSessionSnapshot) sessionResponse() map[string]any {
	username := firstNonEmpty(snapshot.Email, snapshot.UserID)
	firstName := strings.TrimSpace(snapshot.FirstName)
	lastName := strings.TrimSpace(snapshot.LastName)
	displayName := strings.TrimSpace(strings.Join([]string{firstName, lastName}, " "))
	session := map[string]any{
		"authenticated": true,
		"provider":      "workos",
		"mode":          "workos",
		"username":      username,
		"is_admin":      strings.EqualFold(snapshot.Principal.Role, "admin"),
		"bisque_linked": false,
		"user": map[string]any{
			"id":           snapshot.Principal.UserID,
			"workos_id":    snapshot.UserID,
			"username":     username,
			"email":        snapshot.Email,
			"display_name": displayName,
			"org_id":       snapshot.Principal.OrgID,
			"role":         snapshot.Principal.Role,
			"permissions":  snapshot.Permissions,
			"entitlements": snapshot.Entitlements,
		},
	}
	if status := strings.TrimSpace(snapshot.AccountStatus); status != "" {
		session["account_status"] = status
	}
	return session
}

func snapshotFromWorkOSResult(result *workos.AuthenticateSessionResult) workOSSessionSnapshot {
	userID := ""
	email := ""
	firstName := ""
	lastName := ""
	if result.User != nil {
		userID = strings.TrimSpace(result.User.ID)
		email = strings.TrimSpace(result.User.Email)
		if result.User.FirstName != nil {
			firstName = strings.TrimSpace(*result.User.FirstName)
		}
		if result.User.LastName != nil {
			lastName = strings.TrimSpace(*result.User.LastName)
		}
	}
	principalUserID := "workos:" + devPrincipalIDSegment(firstNonEmpty(userID, email), "user")
	role := firstNonEmpty(strings.TrimSpace(result.Role), "researcher")
	orgID := firstNonEmpty(strings.TrimSpace(result.OrganizationID), "workos-org")
	return workOSSessionSnapshot{
		Principal: requestPrincipal{
			UserID: principalUserID,
			OrgID:  orgID,
			Role:   role,
		},
		UserID:       userID,
		Email:        email,
		FirstName:    firstName,
		LastName:     lastName,
		SessionID:    strings.TrimSpace(result.SessionID),
		Permissions:  result.Permissions,
		Entitlements: result.Entitlements,
	}
}

func workOSSnapshotFromContext(ctx context.Context) (workOSSessionSnapshot, bool) {
	snapshot, ok := ctx.Value(workOSPrincipalContextKey{}).(workOSSessionSnapshot)
	return snapshot, ok
}

func sealWorkOSSessionValue(accessToken string, refreshToken string, user workOSUserSnapshot, impersonator *workos.AuthenticateResponseImpersonator, cookiePassword string) (string, error) {
	workosUser := &workos.User{
		ID:        strings.TrimSpace(user.ID),
		Email:     strings.TrimSpace(user.Email),
		FirstName: user.FirstName,
		LastName:  user.LastName,
	}
	return workos.SealSessionFromAuthResponse(accessToken, refreshToken, workosUser, impersonator, cookiePassword)
}

func (auth *WorkOSAuth) setCookie(w http.ResponseWriter, name string, value string, maxAge int) {
	http.SetCookie(w, &http.Cookie{
		Name:     name,
		Value:    value,
		Path:     "/",
		HttpOnly: true,
		Secure:   auth.cookieSecure,
		SameSite: http.SameSiteLaxMode,
		MaxAge:   maxAge,
	})
}

func (auth *WorkOSAuth) clearCookie(w http.ResponseWriter, name string) {
	http.SetCookie(w, &http.Cookie{
		Name:     name,
		Value:    "",
		Path:     "/",
		HttpOnly: true,
		Secure:   auth.cookieSecure,
		SameSite: http.SameSiteLaxMode,
		MaxAge:   -1,
	})
}

func safeRedirectPath(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "/"
	}
	if parsed, err := url.Parse(value); err == nil && parsed.IsAbs() {
		return value
	}
	if strings.HasPrefix(value, "/") {
		return value
	}
	return "/"
}
