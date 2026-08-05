package httpapi

// Google Drive integration, picker model: the user connects their Google
// account once (OAuth code flow, drive.file scope — Ultra can only ever see
// files the user explicitly picks), the browser runs Google's Picker with a
// short-lived access token, and the CONTROL PLANE pulls each picked file
// server-side into the normal uploads pipeline, so everything downstream
// (Lens, staging, sharing, thumbnails) works unchanged.
//
// Reliability contract, deliberately boring:
//   - The refresh token is the only durable secret. AES-256-GCM at rest
//     (the same cipher discipline as BisQue credentials); it never leaves
//     this process — the browser only ever holds sub-hour access tokens.
//   - One import request = one file = one atomic outcome. The client
//     orchestrates the batch, so a crashed control plane fails only the
//     in-flight file and every failure is individually retryable.
//   - Downloads stream (never buffered whole), verify Drive's md5Checksum
//     when present, retry transient failures with backoff, and honor
//     Retry-After on 429/503.
//   - A revoked grant (invalid_grant) flips the stored credential to
//     status "broken" and surfaces as 409 reconnect_required — never a
//     silent retry loop.

import (
	"context"
	"crypto/hmac"
	"crypto/md5" // #nosec G501 — integrity check against Drive's own md5Checksum, not a security boundary.
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

const (
	googleAuthURL  = "https://accounts.google.com/o/oauth2/v2/auth"
	googleTokenURL = "https://oauth2.googleapis.com/token"
	googleAPIBase  = "https://www.googleapis.com"
	// drive.file: per-picked-file access only. openid email: lets the UI
	// show which account is connected. Nothing broader, ever.
	googleDriveScopes = "https://www.googleapis.com/auth/drive.file openid email"

	googleStateTTL         = 10 * time.Minute
	googleDownloadAttempts = 3
	googleRequestTimeout   = 30 * time.Second
)

var errGoogleReconnectRequired = errors.New("google drive access was revoked; reconnect required")

type GoogleDriveConfig struct {
	ClientID       string
	ClientSecret   string
	RedirectURL    string
	PickerAPIKey   string
	MaxImportBytes int64
	// Test seams; production leaves these empty and gets the real Google.
	AuthBase  string
	TokenURL  string
	APIBase   string
	HTTPDoer  *http.Client
	StateKey  []byte
	NowFn     func() time.Time
}

type GooglePersistentCredentialStore interface {
	UpsertGoogleCredential(context.Context, domain.GoogleCredentialRecord) error
	GetGoogleCredentialForUser(ctx context.Context, userID string) (domain.GoogleCredentialRecord, bool, error)
	DeleteGoogleCredentialForUser(ctx context.Context, userID string) error
}

type GoogleDriveService struct {
	cfg        GoogleDriveConfig
	store      GooglePersistentCredentialStore
	cipher     *BisqueCredentialCipher
	httpClient *http.Client
	now        func() time.Time

	mu           sync.Mutex
	accessTokens map[string]googleAccessToken
	refreshing   map[string]chan struct{}
}

type googleAccessToken struct {
	token   string
	expires time.Time
}

func NewGoogleDriveService(cfg GoogleDriveConfig, store GooglePersistentCredentialStore, cipher *BisqueCredentialCipher) *GoogleDriveService {
	client := cfg.HTTPDoer
	if client == nil {
		client = &http.Client{Timeout: 0} // per-request contexts carry the deadlines
	}
	now := cfg.NowFn
	if now == nil {
		now = time.Now
	}
	if cfg.AuthBase == "" {
		cfg.AuthBase = googleAuthURL
	}
	if cfg.TokenURL == "" {
		cfg.TokenURL = googleTokenURL
	}
	if cfg.APIBase == "" {
		cfg.APIBase = googleAPIBase
	}
	return &GoogleDriveService{
		cfg:          cfg,
		store:        store,
		cipher:       cipher,
		httpClient:   client,
		now:          now,
		accessTokens: map[string]googleAccessToken{},
		refreshing:   map[string]chan struct{}{},
	}
}

func (service *GoogleDriveService) Enabled() bool {
	return service != nil &&
		strings.TrimSpace(service.cfg.ClientID) != "" &&
		strings.TrimSpace(service.cfg.ClientSecret) != "" &&
		strings.TrimSpace(service.cfg.RedirectURL) != "" &&
		service.store != nil && service.cipher != nil
}

// --- OAuth state: stateless HMAC token so a control-plane restart between
// authorize and callback cannot strand the user mid-consent.

func (service *GoogleDriveService) stateKey() []byte {
	if len(service.cfg.StateKey) > 0 {
		return service.cfg.StateKey
	}
	return []byte(service.cfg.ClientSecret)
}

func (service *GoogleDriveService) mintState(userID string) string {
	expires := service.now().Add(googleStateTTL).Unix()
	payload := fmt.Sprintf("%s|%d", userID, expires)
	mac := hmac.New(sha256.New, service.stateKey())
	mac.Write([]byte(payload))
	return base64.RawURLEncoding.EncodeToString([]byte(payload)) + "." +
		base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func (service *GoogleDriveService) verifyState(state string) (string, error) {
	parts := strings.Split(strings.TrimSpace(state), ".")
	if len(parts) != 2 {
		return "", errors.New("malformed state")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return "", errors.New("malformed state payload")
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", errors.New("malformed state signature")
	}
	mac := hmac.New(sha256.New, service.stateKey())
	mac.Write(payload)
	if !hmac.Equal(signature, mac.Sum(nil)) {
		return "", errors.New("state signature mismatch")
	}
	fields := strings.Split(string(payload), "|")
	if len(fields) != 2 {
		return "", errors.New("malformed state fields")
	}
	expires, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil || service.now().Unix() > expires {
		return "", errors.New("state expired; restart the connect flow")
	}
	return fields[0], nil
}

func (service *GoogleDriveService) AuthorizeURL(userID string) string {
	query := url.Values{}
	query.Set("client_id", service.cfg.ClientID)
	query.Set("redirect_uri", service.cfg.RedirectURL)
	query.Set("response_type", "code")
	query.Set("scope", googleDriveScopes)
	// offline + consent: Google only issues a refresh token on a consenting
	// grant, and a reconnect after revocation must never silently produce a
	// session with no refresh token.
	query.Set("access_type", "offline")
	query.Set("prompt", "consent")
	query.Set("include_granted_scopes", "true")
	query.Set("state", service.mintState(userID))
	return service.cfg.AuthBase + "?" + query.Encode()
}

type googleTokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int64  `json:"expires_in"`
	IDToken      string `json:"id_token"`
	Error        string `json:"error"`
	ErrorDesc    string `json:"error_description"`
}

func (service *GoogleDriveService) exchange(ctx context.Context, form url.Values) (googleTokenResponse, error) {
	form.Set("client_id", service.cfg.ClientID)
	form.Set("client_secret", service.cfg.ClientSecret)
	ctx, cancel := context.WithTimeout(ctx, googleRequestTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, service.cfg.TokenURL,
		strings.NewReader(form.Encode()))
	if err != nil {
		return googleTokenResponse{}, err
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	response, err := service.httpClient.Do(request)
	if err != nil {
		return googleTokenResponse{}, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return googleTokenResponse{}, err
	}
	var token googleTokenResponse
	if err := json.Unmarshal(body, &token); err != nil {
		return googleTokenResponse{}, fmt.Errorf("google token endpoint returned unparseable response (%d)", response.StatusCode)
	}
	if token.Error != "" {
		if token.Error == "invalid_grant" {
			return token, errGoogleReconnectRequired
		}
		return token, fmt.Errorf("google token endpoint: %s (%s)", token.Error, token.ErrorDesc)
	}
	if response.StatusCode != http.StatusOK {
		return token, fmt.Errorf("google token endpoint returned %d", response.StatusCode)
	}
	return token, nil
}

// emailFromIDToken decodes the (unverified) id_token payload. The token came
// straight from Google's token endpoint over TLS on a confidential-client
// exchange, and it is used for DISPLAY only — never authorization.
func emailFromIDToken(idToken string) string {
	parts := strings.Split(idToken, ".")
	if len(parts) < 2 {
		return ""
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return ""
	}
	var claims struct {
		Email string `json:"email"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return ""
	}
	return strings.TrimSpace(claims.Email)
}

func (service *GoogleDriveService) CompleteConnect(ctx context.Context, principal requestPrincipal, code string) (domain.GoogleCredentialRecord, error) {
	form := url.Values{}
	form.Set("grant_type", "authorization_code")
	form.Set("code", code)
	form.Set("redirect_uri", service.cfg.RedirectURL)
	token, err := service.exchange(ctx, form)
	if err != nil {
		return domain.GoogleCredentialRecord{}, err
	}
	if strings.TrimSpace(token.RefreshToken) == "" {
		return domain.GoogleCredentialRecord{}, errors.New("google did not return a refresh token; remove Ultra at myaccount.google.com/permissions and reconnect")
	}
	ciphertext, nonce, err := service.cipher.Encrypt(token.RefreshToken)
	if err != nil {
		return domain.GoogleCredentialRecord{}, err
	}
	now := service.now().UTC()
	record := domain.GoogleCredentialRecord{
		UserID:                 principal.UserID,
		OrgID:                  principal.OrgID,
		AccountEmail:           emailFromIDToken(token.IDToken),
		RefreshTokenCiphertext: ciphertext,
		RefreshTokenNonce:      nonce,
		RefreshTokenKeyID:      service.cipher.KeyID(),
		RefreshTokenAlgorithm:  service.cipher.Algorithm(),
		Scopes:                 googleDriveScopes,
		Status:                 domain.GoogleCredentialStatusActive,
		CreatedAt:              now,
		UpdatedAt:              now,
		Metadata:               domain.JSONMap{},
	}
	if err := service.store.UpsertGoogleCredential(ctx, record); err != nil {
		return domain.GoogleCredentialRecord{}, err
	}
	service.dropCachedToken(principal.UserID)
	if strings.TrimSpace(token.AccessToken) != "" {
		service.cacheToken(principal.UserID, token.AccessToken, token.ExpiresIn)
	}
	return record, nil
}

func (service *GoogleDriveService) Disconnect(ctx context.Context, userID string) error {
	record, found, err := service.store.GetGoogleCredentialForUser(ctx, userID)
	if err != nil {
		return err
	}
	if found {
		// Best-effort revocation at Google; local deletion is authoritative.
		if refreshToken, decryptErr := service.cipher.Decrypt(
			record.RefreshTokenCiphertext, record.RefreshTokenNonce,
		); decryptErr == nil {
			revokeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
			form := url.Values{"token": {refreshToken}}
			request, requestErr := http.NewRequestWithContext(revokeCtx, http.MethodPost,
				"https://oauth2.googleapis.com/revoke", strings.NewReader(form.Encode()))
			if requestErr == nil {
				request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
				if response, doErr := service.httpClient.Do(request); doErr == nil {
					_ = response.Body.Close()
				}
			}
			cancel()
		}
	}
	service.dropCachedToken(userID)
	return service.store.DeleteGoogleCredentialForUser(ctx, userID)
}

func (service *GoogleDriveService) cacheToken(userID string, token string, expiresIn int64) {
	service.mu.Lock()
	defer service.mu.Unlock()
	service.accessTokens[userID] = googleAccessToken{
		token: token,
		// A minute of slack so a token never expires mid-download setup.
		expires: service.now().Add(time.Duration(expiresIn)*time.Second - time.Minute),
	}
}

func (service *GoogleDriveService) dropCachedToken(userID string) {
	service.mu.Lock()
	defer service.mu.Unlock()
	delete(service.accessTokens, userID)
}

// AccessToken returns a live access token for the user, refreshing at most
// once concurrently per user (a picker-token call and two imports arriving
// together must produce one refresh, not three).
func (service *GoogleDriveService) AccessToken(ctx context.Context, userID string) (string, time.Time, error) {
	for {
		service.mu.Lock()
		if cached, ok := service.accessTokens[userID]; ok && service.now().Before(cached.expires) {
			service.mu.Unlock()
			return cached.token, cached.expires, nil
		}
		waiter, inflight := service.refreshing[userID]
		if !inflight {
			done := make(chan struct{})
			service.refreshing[userID] = done
			service.mu.Unlock()

			token, expires, err := service.refresh(ctx, userID)

			service.mu.Lock()
			delete(service.refreshing, userID)
			close(done)
			service.mu.Unlock()
			return token, expires, err
		}
		service.mu.Unlock()
		select {
		case <-waiter:
		case <-ctx.Done():
			return "", time.Time{}, ctx.Err()
		}
	}
}

func (service *GoogleDriveService) refresh(ctx context.Context, userID string) (string, time.Time, error) {
	record, found, err := service.store.GetGoogleCredentialForUser(ctx, userID)
	if err != nil {
		return "", time.Time{}, err
	}
	if !found {
		return "", time.Time{}, errGoogleReconnectRequired
	}
	refreshToken, err := service.cipher.Decrypt(record.RefreshTokenCiphertext, record.RefreshTokenNonce)
	if err != nil {
		return "", time.Time{}, fmt.Errorf("stored google credential cannot be decrypted: %w", err)
	}
	form := url.Values{}
	form.Set("grant_type", "refresh_token")
	form.Set("refresh_token", refreshToken)
	token, err := service.exchange(ctx, form)
	if err != nil {
		if errors.Is(err, errGoogleReconnectRequired) {
			record.Status = domain.GoogleCredentialStatusBroken
			record.UpdatedAt = service.now().UTC()
			_ = service.store.UpsertGoogleCredential(ctx, record)
		}
		return "", time.Time{}, err
	}
	service.cacheToken(userID, token.AccessToken, token.ExpiresIn)
	if record.Status != domain.GoogleCredentialStatusActive {
		record.Status = domain.GoogleCredentialStatusActive
		record.UpdatedAt = service.now().UTC()
		_ = service.store.UpsertGoogleCredential(ctx, record)
	}
	expires := service.now().Add(time.Duration(token.ExpiresIn) * time.Second)
	return token.AccessToken, expires, nil
}

// --- Drive file import ---

type googleDriveFileMetadata struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	MimeType    string `json:"mimeType"`
	Size        string `json:"size"`
	MD5Checksum string `json:"md5Checksum"`
}

func (service *GoogleDriveService) fileMetadata(ctx context.Context, token string, fileID string) (googleDriveFileMetadata, error) {
	endpoint := fmt.Sprintf(
		"%s/drive/v3/files/%s?fields=id,name,mimeType,size,md5Checksum&supportsAllDrives=true",
		service.cfg.APIBase, url.PathEscape(fileID),
	)
	var metadata googleDriveFileMetadata
	err := service.driveGetJSON(ctx, token, endpoint, &metadata)
	return metadata, err
}

func (service *GoogleDriveService) driveGetJSON(ctx context.Context, token string, endpoint string, out any) error {
	var lastErr error
	for attempt := 1; attempt <= googleDownloadAttempts; attempt++ {
		requestCtx, cancel := context.WithTimeout(ctx, googleRequestTimeout)
		request, err := http.NewRequestWithContext(requestCtx, http.MethodGet, endpoint, nil)
		if err != nil {
			cancel()
			return err
		}
		request.Header.Set("Authorization", "Bearer "+token)
		response, err := service.httpClient.Do(request)
		if err != nil {
			cancel()
			lastErr = err
		} else {
			body, readErr := io.ReadAll(io.LimitReader(response.Body, 1<<20))
			_ = response.Body.Close()
			cancel()
			switch {
			case readErr != nil:
				lastErr = readErr
			case response.StatusCode == http.StatusOK:
				return json.Unmarshal(body, out)
			case response.StatusCode == http.StatusUnauthorized:
				return errGoogleReconnectRequired
			case response.StatusCode == http.StatusNotFound:
				return fmt.Errorf("google drive file not found (or Ultra was not granted access to it)")
			case response.StatusCode == http.StatusTooManyRequests || response.StatusCode >= 500:
				lastErr = fmt.Errorf("google drive returned %d", response.StatusCode)
				waitForRetryAfter(ctx, response.Header.Get("Retry-After"), attempt)
				continue
			default:
				return fmt.Errorf("google drive returned %d: %s", response.StatusCode, truncateForError(body))
			}
		}
		backoff(ctx, attempt)
	}
	return lastErr
}

type GoogleImportInput struct {
	FileID string
}

type GoogleImportResult struct {
	Uploaded uploadedFileRecord
	Metadata googleDriveFileMetadata
}

// ImportFile pulls ONE picked Drive file into the uploads store. Atomic from
// the caller's view: either the file is fully downloaded, verified, and
// recorded, or an error names exactly what failed.
func (service *GoogleDriveService) ImportFile(ctx context.Context, uploadRoot string, principal requestPrincipal, input GoogleImportInput) (GoogleImportResult, error) {
	fileID := strings.TrimSpace(input.FileID)
	if fileID == "" {
		return GoogleImportResult{}, errors.New("a google drive file id is required")
	}
	token, _, err := service.AccessToken(ctx, principal.UserID)
	if err != nil {
		return GoogleImportResult{}, err
	}
	metadata, err := service.fileMetadata(ctx, token, fileID)
	if err != nil {
		return GoogleImportResult{}, err
	}
	if strings.HasPrefix(metadata.MimeType, "application/vnd.google-apps.") {
		if metadata.MimeType == "application/vnd.google-apps.folder" {
			return GoogleImportResult{}, errors.New("folders cannot be imported yet — pick the files inside it")
		}
		return GoogleImportResult{}, fmt.Errorf(
			"%q is a native Google document; download it as a regular file in Drive first", metadata.Name)
	}
	size := int64(0)
	if metadata.Size != "" {
		size, _ = strconv.ParseInt(metadata.Size, 10, 64)
	}
	if service.cfg.MaxImportBytes > 0 && size > service.cfg.MaxImportBytes {
		return GoogleImportResult{}, fmt.Errorf(
			"%q is %d bytes, above the %d byte import limit", metadata.Name, size, service.cfg.MaxImportBytes)
	}

	var lastErr error
	for attempt := 1; attempt <= googleDownloadAttempts; attempt++ {
		record, err := service.downloadOnce(ctx, uploadRoot, principal, token, metadata)
		if err == nil {
			return GoogleImportResult{Uploaded: record, Metadata: metadata}, nil
		}
		if errors.Is(err, errGoogleReconnectRequired) {
			// One forced refresh: the cached token may simply have aged out.
			service.dropCachedToken(principal.UserID)
			token, _, err = service.AccessToken(ctx, principal.UserID)
			if err != nil {
				return GoogleImportResult{}, err
			}
			record, err = service.downloadOnce(ctx, uploadRoot, principal, token, metadata)
			if err == nil {
				return GoogleImportResult{Uploaded: record, Metadata: metadata}, nil
			}
		}
		if ctx.Err() != nil {
			return GoogleImportResult{}, ctx.Err()
		}
		lastErr = err
		backoff(ctx, attempt)
	}
	return GoogleImportResult{}, lastErr
}

func (service *GoogleDriveService) downloadOnce(ctx context.Context, uploadRoot string, principal requestPrincipal, token string, metadata googleDriveFileMetadata) (uploadedFileRecord, error) {
	endpoint := fmt.Sprintf("%s/drive/v3/files/%s?alt=media&supportsAllDrives=true",
		service.cfg.APIBase, url.PathEscape(metadata.ID))
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return uploadedFileRecord{}, err
	}
	request.Header.Set("Authorization", "Bearer "+token)
	response, err := service.httpClient.Do(request)
	if err != nil {
		return uploadedFileRecord{}, err
	}
	defer response.Body.Close()
	switch {
	case response.StatusCode == http.StatusUnauthorized:
		return uploadedFileRecord{}, errGoogleReconnectRequired
	case response.StatusCode == http.StatusTooManyRequests || response.StatusCode >= 500:
		waitForRetryAfter(ctx, response.Header.Get("Retry-After"), 1)
		return uploadedFileRecord{}, fmt.Errorf("google drive download returned %d", response.StatusCode)
	case response.StatusCode != http.StatusOK:
		body, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return uploadedFileRecord{}, fmt.Errorf("google drive download returned %d: %s", response.StatusCode, truncateForError(body))
	}

	body := io.Reader(response.Body)
	if service.cfg.MaxImportBytes > 0 {
		// +1 so an over-limit stream is detected as such, not truncated.
		body = io.LimitReader(response.Body, service.cfg.MaxImportBytes+1)
	}
	hasher := md5.New() // #nosec G401 — matches Drive's integrity checksum.
	record, err := saveImportedUploadFile(
		uploadRoot,
		metadata.Name,
		response.Header.Get("Content-Type"),
		"gdrive://"+metadata.ID,
		io.TeeReader(body, hasher),
		principal,
	)
	if err != nil {
		return uploadedFileRecord{}, err
	}
	if service.cfg.MaxImportBytes > 0 && record.SizeBytes > service.cfg.MaxImportBytes {
		return uploadedFileRecord{}, fmt.Errorf(
			"%q exceeded the %d byte import limit mid-download", metadata.Name, service.cfg.MaxImportBytes)
	}
	if metadata.MD5Checksum != "" {
		downloaded := hex.EncodeToString(hasher.Sum(nil))
		if !strings.EqualFold(downloaded, metadata.MD5Checksum) {
			return uploadedFileRecord{}, fmt.Errorf(
				"%q failed integrity verification (md5 %s != drive %s) — the download was corrupted, retry",
				metadata.Name, downloaded, metadata.MD5Checksum)
		}
	}
	return record, nil
}

func waitForRetryAfter(ctx context.Context, header string, attempt int) {
	delay := time.Duration(attempt) * time.Second
	if header != "" {
		if seconds, err := strconv.Atoi(strings.TrimSpace(header)); err == nil && seconds > 0 && seconds <= 60 {
			delay = time.Duration(seconds) * time.Second
		}
	}
	select {
	case <-time.After(delay):
	case <-ctx.Done():
	}
}

func backoff(ctx context.Context, attempt int) {
	select {
	case <-time.After(time.Duration(attempt*attempt) * time.Second):
	case <-ctx.Done():
	}
}

func truncateForError(body []byte) string {
	text := strings.TrimSpace(string(body))
	if len(text) > 200 {
		return text[:200] + "…"
	}
	return text
}

// --- HTTP handlers ---

func writeGoogleDriveNotConfigured(w http.ResponseWriter) {
	writeJSON(w, http.StatusServiceUnavailable, map[string]string{
		"error": "google drive integration is not configured on this deployment",
	})
}

func (deps ServerDeps) googleDrivePrincipal(w http.ResponseWriter, r *http.Request) (requestPrincipal, bool) {
	principal := principalFromRequest(r, "")
	if strings.TrimSpace(principal.UserID) == "" {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "sign in to connect google drive"})
		return requestPrincipal{}, false
	}
	return principal, true
}

func (deps ServerDeps) handleGoogleDriveStatus(w http.ResponseWriter, r *http.Request) {
	if !deps.GoogleDrive.Enabled() {
		writeJSON(w, http.StatusOK, map[string]any{"enabled": false, "connected": false})
		return
	}
	principal, ok := deps.googleDrivePrincipal(w, r)
	if !ok {
		return
	}
	record, found, err := deps.GoogleDrive.store.GetGoogleCredentialForUser(r.Context(), principal.UserID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	response := map[string]any{
		"enabled":   true,
		"connected": found,
	}
	if found {
		response["account_email"] = record.AccountEmail
		response["status"] = record.Status
		response["connected_at"] = record.CreatedAt.Format(time.RFC3339)
	}
	writeJSON(w, http.StatusOK, response)
}

func (deps ServerDeps) handleGoogleDriveAuthorize(w http.ResponseWriter, r *http.Request) {
	if !deps.GoogleDrive.Enabled() {
		writeGoogleDriveNotConfigured(w)
		return
	}
	principal, ok := deps.googleDrivePrincipal(w, r)
	if !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{
		"authorize_url": deps.GoogleDrive.AuthorizeURL(principal.UserID),
	})
}

// The callback lands in a popup. It reports the outcome to the opener via
// postMessage and closes; without an opener (popup blocked → same-tab flow)
// it falls back to a redirect with a query flag the app reads on load.
const googleCallbackPage = `<!doctype html>
<meta charset="utf-8"><title>Google Drive</title>
<body style="font: 15px system-ui; color:#444; display:grid; place-items:center; height:96vh; margin:0">
<p>%s You can close this window.</p>
<script>
(function () {
  var payload = { type: "ultra-google-drive", status: %q };
  try {
    if (window.opener && !window.opener.closed) {
      window.opener.postMessage(payload, %q);
      window.close();
      return;
    }
  } catch (e) {}
  window.location.replace(%q);
})();
</script>`

func (deps ServerDeps) googleCallbackOrigin() string {
	parsed, err := url.Parse(deps.GoogleDrive.cfg.RedirectURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "*"
	}
	return parsed.Scheme + "://" + parsed.Host
}

func (deps ServerDeps) writeGoogleCallbackPage(w http.ResponseWriter, status string, message string) {
	origin := deps.googleCallbackOrigin()
	fallback := origin + "/?google_drive=" + url.QueryEscape(status)
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	fmt.Fprintf(w, googleCallbackPage, html.EscapeString(message), status, origin, fallback)
}

func (deps ServerDeps) handleGoogleDriveCallback(w http.ResponseWriter, r *http.Request) {
	if !deps.GoogleDrive.Enabled() {
		writeGoogleDriveNotConfigured(w)
		return
	}
	query := r.URL.Query()
	if oauthError := strings.TrimSpace(query.Get("error")); oauthError != "" {
		deps.writeGoogleCallbackPage(w, "denied", "Google Drive was not connected.")
		return
	}
	stateUserID, err := deps.GoogleDrive.verifyState(query.Get("state"))
	if err != nil {
		deps.writeGoogleCallbackPage(w, "error", "The connect link expired — try again from Ultra.")
		return
	}
	// The browser completing the callback must be the signed-in user who
	// started the flow; the HMAC state alone is not a session.
	principal := principalFromRequest(r, "")
	if principal.UserID == "" || principal.UserID != stateUserID {
		deps.writeGoogleCallbackPage(w, "error", "This connect link belongs to a different Ultra session — try again.")
		return
	}
	if _, err := deps.GoogleDrive.CompleteConnect(r.Context(), principal, query.Get("code")); err != nil {
		deps.writeGoogleCallbackPage(w, "error", "Google Drive could not be connected: "+err.Error())
		return
	}
	deps.writeGoogleCallbackPage(w, "connected", "Google Drive is connected.")
}

func (deps ServerDeps) handleGoogleDrivePickerToken(w http.ResponseWriter, r *http.Request) {
	if !deps.GoogleDrive.Enabled() {
		writeGoogleDriveNotConfigured(w)
		return
	}
	principal, ok := deps.googleDrivePrincipal(w, r)
	if !ok {
		return
	}
	token, expires, err := deps.GoogleDrive.AccessToken(r.Context(), principal.UserID)
	if err != nil {
		writeGoogleDriveError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"access_token":   token,
		"expires_at":     expires.UTC().Format(time.RFC3339),
		"picker_api_key": deps.GoogleDrive.cfg.PickerAPIKey,
		"app_id":         appIDFromClientID(deps.GoogleDrive.cfg.ClientID),
	})
}

// appIDFromClientID extracts the numeric project number Google's Picker wants
// as setAppId (the prefix of a standard OAuth client id).
func appIDFromClientID(clientID string) string {
	head, _, found := strings.Cut(clientID, "-")
	if !found {
		return ""
	}
	for _, r := range head {
		if r < '0' || r > '9' {
			return ""
		}
	}
	return head
}

type googleDriveImportRequest struct {
	FileID string `json:"file_id"`
}

func (deps ServerDeps) handleGoogleDriveImport(w http.ResponseWriter, r *http.Request) {
	if !deps.GoogleDrive.Enabled() {
		writeGoogleDriveNotConfigured(w)
		return
	}
	principal, ok := deps.googleDrivePrincipal(w, r)
	if !ok {
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var request googleDriveImportRequest
	if !decodeJSON(w, r, &request) {
		return
	}
	result, err := deps.GoogleDrive.ImportFile(r.Context(), root, principal, GoogleImportInput{FileID: request.FileID})
	if err != nil {
		writeGoogleDriveError(w, err)
		return
	}
	if err := deps.catalogUploadedFileWithEventMetadata(r.Context(), root, result.Uploaded, "resource.imported", domain.JSONMap{
		"source":               "google_drive",
		"google_drive_file_id": result.Metadata.ID,
		"google_drive_mime":    result.Metadata.MimeType,
		"google_drive_md5":     result.Metadata.MD5Checksum,
	}); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"uploaded": result.Uploaded})
}

func (deps ServerDeps) handleGoogleDriveDisconnect(w http.ResponseWriter, r *http.Request) {
	if !deps.GoogleDrive.Enabled() {
		writeGoogleDriveNotConfigured(w)
		return
	}
	principal, ok := deps.googleDrivePrincipal(w, r)
	if !ok {
		return
	}
	if err := deps.GoogleDrive.Disconnect(r.Context(), principal.UserID); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "disconnected"})
}

func writeGoogleDriveError(w http.ResponseWriter, err error) {
	if errors.Is(err, errGoogleReconnectRequired) {
		writeJSON(w, http.StatusConflict, map[string]string{
			"error": "reconnect_required",
			"detail": "Google Drive access was revoked or expired — reconnect your account.",
		})
		return
	}
	writeError(w, http.StatusBadGateway, err)
}
