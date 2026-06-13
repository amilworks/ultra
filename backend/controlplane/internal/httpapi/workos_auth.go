package httpapi

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/workos/workos-go/v8"
	"golang.org/x/sync/singleflight"
)

const (
	workOSSessionCookieName   = "ultra_workos_session"
	workOSStateCookieName     = "ultra_workos_oauth_state"
	workOSStateCookieMaxAge   = 10 * time.Minute
	workOSSessionCookieMaxAge = 7 * 24 * time.Hour
	// WorkOS requires session cookie passwords to be at least 32 characters.
	workOSCookiePasswordMinLength = 32
	// How long a refreshed session stays addressable by the stale cookie that
	// triggered the refresh, so requests already in flight with the old cookie
	// do not retry the single-use refresh token.
	workOSRefreshGraceWindow = time.Minute
	workOSRefreshTimeout     = 30 * time.Second
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

	// WorkOS rotates refresh tokens on every use, so concurrent requests
	// holding the same stale session cookie must share one refresh call
	// instead of racing the single-use token.
	refreshGroup    singleflight.Group
	refreshMu       sync.Mutex
	recentRefreshes map[string]workOSRecentRefresh
}

type workOSRecentRefresh struct {
	sealedSession string
	expiresAt     time.Time
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
	if len(cookiePassword) < workOSCookiePasswordMinLength {
		return nil, fmt.Errorf("WorkOS cookie password must be at least %d characters; generate one with `openssl rand -hex 32`", workOSCookiePasswordMinLength)
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
		recentRefreshes:      make(map[string]workOSRecentRefresh),
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

// handleCallback finishes the AuthKit redirect. It runs as a top-level
// browser navigation, so every failure redirects back into the app with an
// auth_error message instead of rendering a JSON dead end.
func (auth *WorkOSAuth) handleCallback(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	if errParam := strings.TrimSpace(query.Get("error")); errParam != "" {
		slog.Warn("workos callback returned error", "error", errParam, "description", query.Get("error_description"))
		message := "WorkOS could not complete sign-in. Please try again."
		if errParam == "access_denied" {
			message = "Sign-in was cancelled. Please try again."
		}
		auth.redirectCallbackError(w, r, message)
		return
	}
	code := strings.TrimSpace(query.Get("code"))
	state := strings.TrimSpace(query.Get("state"))
	if code == "" || state == "" {
		auth.redirectCallbackError(w, r, "The sign-in response was incomplete. Please try again.")
		return
	}
	stateCookie, err := r.Cookie(workOSStateCookieName)
	if err != nil || strings.TrimSpace(stateCookie.Value) == "" {
		auth.redirectCallbackError(w, r, "Your sign-in attempt expired. Please try again.")
		return
	}
	expected, err := workos.Unseal[workOSOAuthState](stateCookie.Value, auth.cookiePassword)
	if err != nil || expected.State == "" || expected.CodeVerifier == "" {
		auth.redirectCallbackError(w, r, "Your sign-in attempt could not be verified. Please try again.")
		return
	}
	if time.Since(time.Unix(expected.CreatedAt, 0)) > workOSStateCookieMaxAge {
		auth.redirectCallbackError(w, r, "Your sign-in attempt expired. Please try again.")
		return
	}
	if state != expected.State {
		auth.redirectCallbackError(w, r, "Your sign-in attempt could not be verified. Please try again.")
		return
	}
	authResponse, err := auth.client.AuthKitPKCECodeExchange(r.Context(), workos.AuthKitPKCECodeExchangeParams{
		Code:         code,
		CodeVerifier: expected.CodeVerifier,
	})
	if err != nil {
		slog.Error("workos code exchange failed", "error", err)
		auth.redirectCallbackError(w, r, "WorkOS could not complete sign-in. Please try again.")
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
		slog.Error("workos session seal failed", "error", err)
		auth.redirectCallbackError(w, r, "Sign-in could not be completed. Please try again.")
		return
	}
	auth.setCookie(w, workOSSessionCookieName, sealedSession, int(workOSSessionCookieMaxAge.Seconds()))
	auth.clearCookie(w, workOSStateCookieName)
	http.Redirect(w, r, auth.postLoginRedirectURI, http.StatusFound)
}

func (auth *WorkOSAuth) redirectCallbackError(w http.ResponseWriter, r *http.Request, message string) {
	auth.clearCookie(w, workOSStateCookieName)
	http.Redirect(w, r, appendAuthErrorParam(auth.postLoginRedirectURI, message), http.StatusFound)
}

func appendAuthErrorParam(target string, message string) string {
	parsed, err := url.Parse(strings.TrimSpace(target))
	if err != nil || parsed.String() == "" {
		parsed = &url.URL{Path: "/"}
	}
	query := parsed.Query()
	query.Set("auth_error", message)
	parsed.RawQuery = query.Encode()
	return parsed.String()
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
		sealedSession, refreshErr := auth.refreshSealedSession(cookie.Value)
		if refreshErr != nil {
			if errors.Is(refreshErr, errWorkOSRefreshRevoked) {
				// The session is permanently dead; drop the cookie so later
				// requests stop retrying the revoked token against WorkOS.
				auth.clearCookie(w, workOSSessionCookieName)
			}
			return workOSSessionSnapshot{}, false
		}
		auth.setCookie(w, workOSSessionCookieName, sealedSession, int(workOSSessionCookieMaxAge.Seconds()))
		session = workos.NewSession(auth.client, sealedSession, auth.cookiePassword)
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

// errWorkOSRefreshRevoked marks a refresh rejected with invalid_grant: the
// token can never succeed again, unlike transient WorkOS or network failures.
var errWorkOSRefreshRevoked = errors.New("WorkOS refresh token revoked")

// refreshSealedSession exchanges the refresh token inside a stale session
// cookie for a freshly sealed session. All concurrent requests carrying the
// same stale cookie share a single WorkOS call, and the result is kept for a
// short grace window so requests that were already in flight with the old
// cookie value reuse it instead of burning the rotated refresh token.
func (auth *WorkOSAuth) refreshSealedSession(staleCookie string) (string, error) {
	digest := sha256.Sum256([]byte(staleCookie))
	key := hex.EncodeToString(digest[:])

	auth.refreshMu.Lock()
	if recent, ok := auth.recentRefreshes[key]; ok && time.Now().Before(recent.expiresAt) {
		auth.refreshMu.Unlock()
		return recent.sealedSession, nil
	}
	auth.refreshMu.Unlock()

	sealed, err, _ := auth.refreshGroup.Do(key, func() (any, error) {
		// Deliberately detached from the request context: the refresh outcome
		// is shared by every request waiting on this key, so the first
		// caller's cancellation must not fail the others.
		ctx, cancel := context.WithTimeout(context.Background(), workOSRefreshTimeout)
		defer cancel()
		refreshed, err := workos.NewSession(auth.client, staleCookie, auth.cookiePassword).Refresh(ctx)
		if err != nil {
			return "", err
		}
		if !refreshed.Authenticated || strings.TrimSpace(refreshed.SealedSession) == "" {
			if refreshed.Reason == "refresh_token_revoked" {
				return "", errWorkOSRefreshRevoked
			}
			return "", fmt.Errorf("WorkOS session refresh rejected: %s", firstNonEmpty(refreshed.Reason, "unknown"))
		}
		auth.refreshMu.Lock()
		now := time.Now()
		for cachedKey, cached := range auth.recentRefreshes {
			if now.After(cached.expiresAt) {
				delete(auth.recentRefreshes, cachedKey)
			}
		}
		auth.recentRefreshes[key] = workOSRecentRefresh{
			sealedSession: refreshed.SealedSession,
			expiresAt:     now.Add(workOSRefreshGraceWindow),
		}
		auth.refreshMu.Unlock()
		return refreshed.SealedSession, nil
	})
	if err != nil {
		slog.Warn("workos session refresh failed", "error", err)
		return "", err
	}
	return sealed.(string), nil
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
