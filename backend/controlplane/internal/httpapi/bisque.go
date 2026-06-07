package httpapi

import (
	"bytes"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

var bisqueResourceTypePattern = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)
var bisqueScopePattern = regexp.MustCompile(`^[A-Za-z0-9_, -]+$`)

type BisqueServiceConfig struct {
	RootURL       string
	DevUsername   string
	DevPassword   string
	AllowedRoots  []string
	HTTPClient    *http.Client
	UploadRoot    string
	MaxImportSize int64
}

type BisqueService struct {
	rootURL       string
	root          *url.URL
	allowedRoots  []*url.URL
	username      string
	password      string
	client        *http.Client
	uploadRoot    string
	maxImportSize int64
}

type BisqueCredentials struct {
	Username string
	Password string
}

type BisquePersistentCredentialStore interface {
	UpsertBisqueCredential(context.Context, domain.UpsertBisqueCredentialInput) (domain.BisqueCredentialRecord, error)
	GetBisqueCredentialBySessionID(context.Context, string) (domain.BisqueCredentialRecord, bool, error)
	DeleteBisqueCredentialBySessionID(context.Context, string) error
}

type BisqueCredentialCipher struct {
	aead  cipher.AEAD
	keyID string
}

type BisqueCredentialLinkInput struct {
	Credentials    BisqueCredentials
	UserID         string
	OrgID          string
	RootURL        string
	LastVerifiedAt time.Time
	Metadata       domain.JSONMap
}

type BisqueCredentialStore struct {
	mu          sync.RWMutex
	credentials map[string]BisqueCredentials
	persistent  BisquePersistentCredentialStore
	cipher      *BisqueCredentialCipher
	rootURL     string
}

func NewBisqueCredentialStore() *BisqueCredentialStore {
	return &BisqueCredentialStore{credentials: map[string]BisqueCredentials{}}
}

func NewPersistentBisqueCredentialStore(persistent BisquePersistentCredentialStore, cipher *BisqueCredentialCipher, rootURL string) *BisqueCredentialStore {
	return &BisqueCredentialStore{
		credentials: map[string]BisqueCredentials{},
		persistent:  persistent,
		cipher:      cipher,
		rootURL:     strings.TrimRight(strings.TrimSpace(rootURL), "/"),
	}
}

func NewBisqueCredentialCipher(key []byte, keyID string) (*BisqueCredentialCipher, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("BisQue credential encryption key must be 32 bytes for AES-256-GCM, got %d", len(key))
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	keyID = strings.TrimSpace(keyID)
	if keyID == "" {
		keyID = "default"
	}
	return &BisqueCredentialCipher{aead: aead, keyID: keyID}, nil
}

func NewBisqueCredentialCipherFromString(rawKey string, keyID string) (*BisqueCredentialCipher, error) {
	key, err := decodeBisqueCredentialKey(rawKey)
	if err != nil {
		return nil, err
	}
	return NewBisqueCredentialCipher(key, keyID)
}

func decodeBisqueCredentialKey(rawKey string) ([]byte, error) {
	rawKey = strings.TrimSpace(rawKey)
	if rawKey == "" {
		return nil, errors.New("BisQue credential encryption key is required")
	}
	if prefixed, ok := strings.CutPrefix(rawKey, "base64:"); ok {
		decoded, err := base64.StdEncoding.DecodeString(strings.TrimSpace(prefixed))
		if err != nil {
			return nil, err
		}
		return decoded, nil
	}
	if prefixed, ok := strings.CutPrefix(rawKey, "hex:"); ok {
		decoded, err := hex.DecodeString(strings.TrimSpace(prefixed))
		if err != nil {
			return nil, err
		}
		return decoded, nil
	}
	if decoded, err := base64.StdEncoding.DecodeString(rawKey); err == nil && len(decoded) == 32 {
		return decoded, nil
	}
	if decoded, err := hex.DecodeString(rawKey); err == nil && len(decoded) == 32 {
		return decoded, nil
	}
	if len([]byte(rawKey)) == 32 {
		return []byte(rawKey), nil
	}
	return nil, fmt.Errorf("BisQue credential encryption key must decode to 32 bytes")
}

func (cipher *BisqueCredentialCipher) KeyID() string {
	if cipher == nil {
		return ""
	}
	return cipher.keyID
}

func (cipher *BisqueCredentialCipher) Algorithm() string {
	if cipher == nil {
		return ""
	}
	return "AES-256-GCM"
}

func (cipher *BisqueCredentialCipher) Encrypt(plaintext string) (string, string, error) {
	if cipher == nil || cipher.aead == nil {
		return "", "", errors.New("BisQue credential cipher is not configured")
	}
	nonce := make([]byte, cipher.aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return "", "", err
	}
	ciphertext := cipher.aead.Seal(nil, nonce, []byte(plaintext), nil)
	return base64.StdEncoding.EncodeToString(ciphertext), base64.StdEncoding.EncodeToString(nonce), nil
}

func (cipher *BisqueCredentialCipher) Decrypt(ciphertextText string, nonceText string) (string, error) {
	if cipher == nil || cipher.aead == nil {
		return "", errors.New("BisQue credential cipher is not configured")
	}
	ciphertext, err := base64.StdEncoding.DecodeString(strings.TrimSpace(ciphertextText))
	if err != nil {
		return "", err
	}
	nonce, err := base64.StdEncoding.DecodeString(strings.TrimSpace(nonceText))
	if err != nil {
		return "", err
	}
	plaintext, err := cipher.aead.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return "", err
	}
	return string(plaintext), nil
}

func (store *BisqueCredentialStore) Put(credentials BisqueCredentials) string {
	if store == nil {
		return ""
	}
	sessionID := domain.NewID("bisque_session")
	store.mu.Lock()
	defer store.mu.Unlock()
	store.credentials[sessionID] = BisqueCredentials{
		Username: strings.TrimSpace(credentials.Username),
		Password: credentials.Password,
	}
	return sessionID
}

func (store *BisqueCredentialStore) PutLinked(ctx context.Context, input BisqueCredentialLinkInput) (string, error) {
	if store == nil {
		return "", errors.New("BisQue credential store is not configured")
	}
	if store.persistent == nil || store.cipher == nil {
		return store.Put(input.Credentials), nil
	}
	rootURL := strings.TrimRight(strings.TrimSpace(input.RootURL), "/")
	if rootURL == "" {
		rootURL = store.rootURL
	}
	if rootURL == "" {
		return "", errors.New("BisQue root URL is required for persistent credentials")
	}
	username := strings.TrimSpace(input.Credentials.Username)
	if username == "" {
		return "", errors.New("BisQue username is required")
	}
	ciphertext, nonce, err := store.cipher.Encrypt(input.Credentials.Password)
	if err != nil {
		return "", err
	}
	record, err := store.persistent.UpsertBisqueCredential(contextOrBackground(ctx), domain.UpsertBisqueCredentialInput{
		UserID:             input.UserID,
		OrgID:              input.OrgID,
		RootURL:            rootURL,
		Username:           username,
		PasswordCiphertext: ciphertext,
		PasswordNonce:      nonce,
		PasswordKeyID:      store.cipher.KeyID(),
		PasswordAlgorithm:  store.cipher.Algorithm(),
		Status:             "active",
		LastVerifiedAt:     input.LastVerifiedAt,
		Metadata:           input.Metadata,
	})
	if err != nil {
		return "", err
	}
	store.cache(record.SessionID, BisqueCredentials{Username: username, Password: input.Credentials.Password})
	return record.SessionID, nil
}

func (store *BisqueCredentialStore) Get(sessionID string) (BisqueCredentials, bool) {
	credentials, ok, _ := store.GetWithContext(context.Background(), sessionID)
	return credentials, ok
}

func (store *BisqueCredentialStore) GetWithContext(ctx context.Context, sessionID string) (BisqueCredentials, bool, error) {
	if store == nil {
		return BisqueCredentials{}, false, nil
	}
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return BisqueCredentials{}, false, nil
	}
	store.mu.RLock()
	credentials, ok := store.credentials[sessionID]
	store.mu.RUnlock()
	if ok {
		return credentials, true, nil
	}
	if store.persistent == nil || store.cipher == nil {
		return BisqueCredentials{}, false, nil
	}
	record, found, err := store.persistent.GetBisqueCredentialBySessionID(contextOrBackground(ctx), sessionID)
	if err != nil || !found {
		return BisqueCredentials{}, false, err
	}
	if strings.TrimSpace(record.Status) != "" && strings.TrimSpace(record.Status) != "active" {
		return BisqueCredentials{}, false, nil
	}
	password, err := store.cipher.Decrypt(record.PasswordCiphertext, record.PasswordNonce)
	if err != nil {
		return BisqueCredentials{}, false, err
	}
	credentials = BisqueCredentials{
		Username: strings.TrimSpace(record.Username),
		Password: password,
	}
	if credentials.Username == "" {
		return BisqueCredentials{}, false, nil
	}
	store.cache(sessionID, credentials)
	return credentials, true, nil
}

func (store *BisqueCredentialStore) Delete(sessionID string) {
	if store == nil {
		return
	}
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	delete(store.credentials, sessionID)
}

func (store *BisqueCredentialStore) Unlink(ctx context.Context, sessionID string) error {
	if store == nil {
		return nil
	}
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil
	}
	store.Delete(sessionID)
	if store.persistent == nil {
		return nil
	}
	return store.persistent.DeleteBisqueCredentialBySessionID(contextOrBackground(ctx), sessionID)
}

func (store *BisqueCredentialStore) cache(sessionID string, credentials BisqueCredentials) {
	if store == nil {
		return
	}
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	store.credentials[sessionID] = BisqueCredentials{
		Username: strings.TrimSpace(credentials.Username),
		Password: credentials.Password,
	}
}

func contextOrBackground(ctx context.Context) context.Context {
	if ctx == nil {
		return context.Background()
	}
	return ctx
}

func NewBisqueService(config BisqueServiceConfig) *BisqueService {
	rootText := strings.TrimSpace(config.RootURL)
	if rootText == "" {
		return nil
	}
	root, err := url.Parse(rootText)
	if err != nil || root.Scheme == "" || root.Host == "" {
		return nil
	}
	allowed := make([]*url.URL, 0, len(config.AllowedRoots)+1)
	allowed = append(allowed, root)
	for _, raw := range config.AllowedRoots {
		parsed, err := url.Parse(strings.TrimSpace(raw))
		if err == nil && parsed.Scheme != "" && parsed.Host != "" {
			allowed = append(allowed, parsed)
		}
	}
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	maxImportSize := config.MaxImportSize
	if maxImportSize <= 0 {
		maxImportSize = 512 << 20
	}
	return &BisqueService{
		rootURL:       strings.TrimRight(root.String(), "/"),
		root:          root,
		allowedRoots:  allowed,
		username:      strings.TrimSpace(config.DevUsername),
		password:      config.DevPassword,
		client:        client,
		uploadRoot:    strings.TrimSpace(config.UploadRoot),
		maxImportSize: maxImportSize,
	}
}

type bisqueSearchRequest struct {
	ResourceType   string   `json:"resource_type"`
	TagQuery       string   `json:"tag_query"`
	TagOrder       string   `json:"tag_order"`
	Query          string   `json:"query"`
	NameContains   string   `json:"name_contains"`
	Extensions     []string `json:"extensions"`
	FileExtensions []string `json:"file_extensions"`
	Scope          string   `json:"scope"`
	Sort           string   `json:"sort"`
	Limit          int      `json:"limit"`
	Offset         int      `json:"offset"`
	CountAll       bool     `json:"count_all"`
}

type bisqueSearchResponse struct {
	Count   int              `json:"count"`
	Results []BisqueResource `json:"results"`
}

type bisqueImportRequest struct {
	Resources    []string `json:"resources"`
	ResourceURIs []string `json:"resource_uris"`
}

type bisqueImportResponse struct {
	FileCount int                  `json:"file_count"`
	Uploaded  []uploadedFileRecord `json:"uploaded"`
	Imports   []BisqueImportRecord `json:"imports"`
}

type bisqueUploadRequest struct {
	FileIDs     []string `json:"file_ids"`
	ArtifactIDs []string `json:"artifact_ids"`
}

type bisqueUploadResponse struct {
	Count   int                  `json:"count"`
	Uploads []BisqueUploadRecord `json:"uploads"`
}

type BisqueResource struct {
	ResourceURI     string            `json:"resource_uri"`
	Name            string            `json:"name,omitempty"`
	ResourceUniq    string            `json:"resource_uniq,omitempty"`
	ResourceType    string            `json:"resource_type,omitempty"`
	Tags            map[string]string `json:"tags,omitempty"`
	ClientViewURL   string            `json:"client_view_url,omitempty"`
	ImageServiceURL string            `json:"image_service_url,omitempty"`
}

type BisqueImportRecord struct {
	InputURL        string             `json:"input_url"`
	Status          string             `json:"status"`
	ResourceURI     string             `json:"resource_uri"`
	ResourceUniq    string             `json:"resource_uniq,omitempty"`
	Uploaded        uploadedFileRecord `json:"uploaded"`
	ClientViewURL   string             `json:"client_view_url,omitempty"`
	ImageServiceURL string             `json:"image_service_url,omitempty"`
}

type BisqueUploadRecord struct {
	FileID       string `json:"file_id,omitempty"`
	ArtifactID   string `json:"artifact_id,omitempty"`
	ResourceURI  string `json:"resource_uri"`
	Name         string `json:"name,omitempty"`
	ResourceUniq string `json:"resource_uniq,omitempty"`
}

func (deps ServerDeps) handleBisqueSearch(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	var req bisqueSearchRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	response, err := deps.Bisque.Search(r.Context(), req, deps.bisqueCredentialsFromRequest(r.Context(), r))
	if err != nil {
		writeBisqueError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func (deps ServerDeps) handleImportBisqueResources(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var req bisqueImportRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	resources := append([]string{}, req.Resources...)
	resources = append(resources, req.ResourceURIs...)
	if len(resources) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("at least one BisQue resource URI is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	uploaded := make([]uploadedFileRecord, 0, len(resources))
	imports := make([]BisqueImportRecord, 0, len(resources))
	for _, resourceURI := range resources {
		record, imported, err := deps.Bisque.ImportResource(r.Context(), root, resourceURI, principal, deps.bisqueCredentialsFromRequest(r.Context(), r))
		if err != nil {
			writeBisqueError(w, err)
			return
		}
		if err := deps.catalogUploadedFile(r.Context(), root, record, "resource.imported"); err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		uploaded = append(uploaded, record)
		imports = append(imports, imported)
	}
	writeJSON(w, http.StatusOK, bisqueImportResponse{FileCount: len(uploaded), Uploaded: uploaded, Imports: imports})
}

func (deps ServerDeps) handleBisqueUpload(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var req bisqueUploadRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if len(req.FileIDs) == 0 && len(req.ArtifactIDs) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("at least one file_id or artifact_id is required"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	uploads := make([]BisqueUploadRecord, 0, len(req.FileIDs)+len(req.ArtifactIDs))
	for _, fileID := range req.FileIDs {
		record, path, err := deps.findUploadResourceForRequest(r.Context(), root, principal, fileID)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		uploaded, err := deps.Bisque.UploadFile(r.Context(), record, path, deps.bisqueCredentialsFromRequest(r.Context(), r))
		if err != nil {
			writeBisqueError(w, err)
			return
		}
		uploads = append(uploads, uploaded)
	}
	for _, artifactID := range req.ArtifactIDs {
		if deps.Store == nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "artifact store is not configured"})
			return
		}
		if strings.TrimSpace(deps.ArtifactRoot) == "" {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "artifact root is not configured"})
			return
		}
		artifact, err := deps.Store.GetArtifactForUser(r.Context(), artifactID, principal.UserID)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		path, err := resolveArtifactDownloadPath(deps.ArtifactRoot, artifact)
		if err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		uploaded, err := deps.Bisque.UploadNamedFile(r.Context(), artifactName(artifact, path), path, deps.bisqueCredentialsFromRequest(r.Context(), r))
		if err != nil {
			writeBisqueError(w, err)
			return
		}
		uploaded.ArtifactID = artifact.ArtifactID
		uploads = append(uploads, uploaded)
	}
	writeJSON(w, http.StatusOK, bisqueUploadResponse{Count: len(uploads), Uploads: uploads})
}

func (deps ServerDeps) bisqueJobMetadataFromRequest(r *http.Request) domain.JSONMap {
	metadata := domain.JSONMap{}
	if deps.BisqueCredentials == nil {
		return metadata
	}
	sessionID := bisqueSessionIDFromRequest(r)
	if sessionID == "" {
		return metadata
	}
	if _, ok, err := deps.BisqueCredentials.GetWithContext(r.Context(), sessionID); !ok || err != nil {
		return metadata
	}
	metadata["bisque_session_id"] = sessionID
	return metadata
}

func (deps ServerDeps) bisqueCredentialsFromRequest(ctx context.Context, r *http.Request) BisqueCredentials {
	if deps.BisqueCredentials == nil {
		return BisqueCredentials{}
	}
	if credentials, ok, _ := deps.BisqueCredentials.GetWithContext(ctx, bisqueSessionIDFromRequest(r)); ok {
		return credentials
	}
	if workerSessionID := bisqueSessionIDFromWorkerHeader(r); workerSessionID != "" {
		if credentials, ok, _ := deps.BisqueCredentials.GetWithContext(ctx, workerSessionID); ok {
			return credentials
		}
	}
	return BisqueCredentials{}
}

func (deps ServerDeps) bisqueRootURL() string {
	if deps.Bisque == nil {
		return ""
	}
	return deps.Bisque.rootURL
}

func bisqueSessionIDFromWorkerHeader(r *http.Request) string {
	runID := strings.TrimSpace(r.Header.Get("X-Ultra-Run-Id"))
	if runID == "" {
		return ""
	}
	sessionID := strings.TrimSpace(r.Header.Get("X-Ultra-Bisque-Session-Id"))
	if sessionID == "" {
		return ""
	}
	return sessionID
}

func (service *BisqueService) VerifyCredentials(ctx context.Context, credentials BisqueCredentials) error {
	if service == nil {
		return bisqueClientError("BisQue credential verification is not configured")
	}
	if strings.TrimSpace(credentials.Username) == "" || credentials.Password == "" {
		return bisqueClientError("BisQue username and password are required")
	}
	verifyURL := service.endpoint("/auth_service/session")
	data, _, err := service.fetchNoRedirect(ctx, http.MethodGet, verifyURL, nil, "", credentials)
	if err != nil {
		return err
	}
	if !bisqueSessionHasAuthenticatedUser(data) {
		return bisqueClientError("BisQue credentials could not be verified")
	}
	return nil
}

func (service *BisqueService) fetchNoRedirect(ctx context.Context, method string, rawURL string, body io.Reader, contentType string, credentials BisqueCredentials) ([]byte, string, error) {
	if service == nil {
		return nil, "", bisqueClientError("BisQue integration is not configured")
	}
	client := *service.client
	client.CheckRedirect = func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}
	return service.fetchWithClient(ctx, &client, method, rawURL, body, contentType, credentials)
}

func (service *BisqueService) Search(ctx context.Context, req bisqueSearchRequest, credentials BisqueCredentials) (bisqueSearchResponse, error) {
	resourceType := strings.TrimSpace(req.ResourceType)
	if resourceType == "" {
		resourceType = "image"
	}
	if !bisqueResourceTypePattern.MatchString(resourceType) {
		return bisqueSearchResponse{}, bisqueClientError("unsafe BisQue resource type")
	}
	limit := req.Limit
	if limit <= 0 {
		limit = 25
	}
	if limit > 100 {
		limit = 100
	}
	filters := bisqueSearchFiltersFromRequest(req)
	if filters.enabled() {
		results, total, err := service.filteredSearchResults(ctx, resourceType, req, filters, limit, credentials)
		if err != nil {
			return bisqueSearchResponse{}, err
		}
		return bisqueSearchResponse{Count: total, Results: service.withLinks(results)}, nil
	}
	searchURL, err := service.searchURL(resourceType, req, limit, req.Offset, false)
	if err != nil {
		return bisqueSearchResponse{}, err
	}
	data, _, err := service.fetch(ctx, http.MethodGet, searchURL, nil, "", credentials)
	if err != nil {
		return bisqueSearchResponse{}, err
	}
	results, total := parseBisqueSearchResponse(data)
	if req.CountAll {
		total, err = service.countAllSearchResults(ctx, resourceType, req, credentials)
		if err != nil {
			return bisqueSearchResponse{}, err
		}
	}
	results = service.withLinks(results)
	return bisqueSearchResponse{Count: total, Results: results}, nil
}

func (service *BisqueService) searchURL(resourceType string, req bisqueSearchRequest, limit int, offset int, includeZeroOffset bool) (string, error) {
	endpoint := service.endpoint("/data_service/" + resourceType + "/")
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return "", err
	}
	query := parsed.Query()
	if tagQuery := strings.TrimSpace(req.TagQuery); tagQuery != "" {
		query.Set("tag_query", tagQuery)
	}
	if tagOrder := bisqueTagOrder(req); tagOrder != "" {
		query.Set("tag_order", tagOrder)
	}
	filters := bisqueSearchFiltersFromRequest(req)
	if textQuery := bisqueServerTextQuery(req, filters); textQuery != "" {
		query.Set("query", textQuery)
	}
	query.Set("wpublic", bisqueWPublicScope(req.Scope))
	query.Set("limit", strconv.Itoa(limit))
	if offset > 0 || includeZeroOffset {
		query.Set("offset", strconv.Itoa(offset))
	}
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}

func (service *BisqueService) filteredSearchResults(ctx context.Context, resourceType string, req bisqueSearchRequest, filters bisqueSearchFilters, limit int, credentials BisqueCredentials) ([]BisqueResource, int, error) {
	pageLimit := limit
	if req.CountAll || pageLimit < 100 {
		pageLimit = 100
	}
	if pageLimit > 700 {
		pageLimit = 700
	}
	preview := []BisqueResource{}
	total := 0
	for offset := max(0, req.Offset); ; offset += pageLimit {
		searchURL, err := service.searchURL(resourceType, req, pageLimit, offset, offset == 0 && req.Offset > 0)
		if err != nil {
			return nil, 0, err
		}
		data, _, err := service.fetch(ctx, http.MethodGet, searchURL, nil, "", credentials)
		if err != nil {
			return nil, 0, err
		}
		resources, _ := parseBisqueSearchResponse(data)
		for _, resource := range resources {
			if !filters.matches(resource) {
				continue
			}
			if len(preview) < limit {
				preview = append(preview, resource)
			}
			total++
		}
		if !req.CountAll && len(preview) >= limit {
			return preview, len(preview), nil
		}
		if len(resources) < pageLimit {
			if req.CountAll {
				return preview, total, nil
			}
			return preview, len(preview), nil
		}
	}
}

func (service *BisqueService) countAllSearchResults(ctx context.Context, resourceType string, req bisqueSearchRequest, credentials BisqueCredentials) (int, error) {
	const pageLimit = 700
	total := 0
	for offset := 0; ; offset += pageLimit {
		searchURL, err := service.searchURL(resourceType, req, pageLimit, offset, true)
		if err != nil {
			return 0, err
		}
		data, _, err := service.fetch(ctx, http.MethodGet, searchURL, nil, "", credentials)
		if err != nil {
			return 0, err
		}
		resources, _ := parseBisqueSearchResponse(data)
		total += len(resources)
		if len(resources) < pageLimit {
			return total, nil
		}
	}
}

type bisqueSearchFilters struct {
	nameContains string
	extensions   []string
}

func bisqueSearchFiltersFromRequest(req bisqueSearchRequest) bisqueSearchFilters {
	return bisqueSearchFilters{
		nameContains: strings.ToLower(strings.TrimSpace(req.NameContains)),
		extensions:   bisqueNormalizedExtensions(append(req.Extensions, req.FileExtensions...)),
	}
}

func (filters bisqueSearchFilters) enabled() bool {
	return filters.nameContains != "" || len(filters.extensions) > 0
}

func (filters bisqueSearchFilters) matches(resource BisqueResource) bool {
	haystack := strings.ToLower(strings.Join([]string{
		resource.Name,
		resource.ResourceURI,
		resource.ResourceUniq,
		resource.ResourceType,
	}, " "))
	if filters.nameContains != "" && !strings.Contains(haystack, filters.nameContains) {
		return false
	}
	if len(filters.extensions) == 0 {
		return true
	}
	name := strings.ToLower(strings.TrimSpace(resource.Name))
	if name == "" {
		name = strings.ToLower(pathBaseFromURL(resource.ResourceURI))
	}
	for _, extension := range filters.extensions {
		if strings.HasSuffix(name, extension) {
			return true
		}
	}
	return false
}

func bisqueNormalizedExtensions(rawExtensions []string) []string {
	seen := map[string]bool{}
	normalized := []string{}
	for _, raw := range rawExtensions {
		cleaned := strings.TrimSpace(strings.ToLower(raw))
		if cleaned == "" {
			continue
		}
		cleaned = strings.TrimPrefix(cleaned, ".")
		candidates := []string{cleaned}
		if cleaned == "nifti" {
			candidates = []string{"nii", "nii.gz", "nifti"}
		}
		for _, candidate := range candidates {
			candidate = "." + strings.TrimPrefix(strings.TrimSpace(candidate), ".")
			if candidate == "." || seen[candidate] {
				continue
			}
			seen[candidate] = true
			normalized = append(normalized, candidate)
		}
	}
	return normalized
}

func bisqueServerTextQuery(req bisqueSearchRequest, filters bisqueSearchFilters) string {
	if textQuery := strings.TrimSpace(req.Query); textQuery != "" {
		return textQuery
	}
	if nameContains := strings.TrimSpace(req.NameContains); nameContains != "" {
		return nameContains
	}
	if len(filters.extensions) == 1 {
		return filters.extensions[0]
	}
	return ""
}

func bisqueTagOrder(req bisqueSearchRequest) string {
	if tagOrder := strings.TrimSpace(req.TagOrder); tagOrder != "" {
		return tagOrder
	}
	switch strings.ToLower(strings.TrimSpace(req.Sort)) {
	case "recent", "newest", "latest", "modified_desc", "created_desc":
		return "@ts:desc"
	case "oldest", "modified_asc", "created_asc":
		return "@ts:asc"
	case "name", "name_asc":
		return "@name"
	case "name_desc":
		return "@name:desc"
	default:
		return ""
	}
}

func bisqueWPublicScope(scope string) string {
	cleaned := strings.ToLower(strings.TrimSpace(scope))
	switch cleaned {
	case "", "default", "owner,shared", "shared,owner":
		return "owner,shared"
	case "owner", "mine", "my":
		return "owner"
	case "shared":
		return "shared"
	case "public":
		return "1"
	case "all":
		return "owner,shared,1"
	default:
		if bisqueScopePattern.MatchString(cleaned) {
			return strings.ReplaceAll(cleaned, " ", "")
		}
		return "owner,shared"
	}
}

func (service *BisqueService) ImportResource(ctx context.Context, uploadRoot string, rawURI string, principal requestPrincipal, credentials BisqueCredentials) (uploadedFileRecord, BisqueImportRecord, error) {
	resourceURI, err := service.normalizeAllowedURL(rawURI)
	if err != nil {
		return uploadedFileRecord{}, BisqueImportRecord{}, err
	}
	resource, err := service.fetchResource(ctx, resourceURI, credentials)
	if err != nil {
		return uploadedFileRecord{}, BisqueImportRecord{}, err
	}
	if resource.ResourceURI == "" {
		resource.ResourceURI = resourceURI
	}
	blobURL, err := service.blobURL(resource)
	if err != nil {
		return uploadedFileRecord{}, BisqueImportRecord{}, err
	}
	data, contentType, err := service.fetch(ctx, http.MethodGet, blobURL, nil, "", credentials)
	if err != nil {
		return uploadedFileRecord{}, BisqueImportRecord{}, err
	}
	name := strings.TrimSpace(resource.Name)
	if name == "" {
		name = pathBaseFromURL(resourceURI)
	}
	if name == "" {
		name = "bisque-resource"
	}
	record, err := saveImportedUploadFile(uploadRoot, name, contentType, resource.ResourceURI, bytes.NewReader(data), principal)
	if err != nil {
		return uploadedFileRecord{}, BisqueImportRecord{}, err
	}
	resource = service.withLinks([]BisqueResource{resource})[0]
	imported := BisqueImportRecord{
		InputURL:        resourceURI,
		Status:          "imported",
		ResourceURI:     resource.ResourceURI,
		ResourceUniq:    resource.ResourceUniq,
		Uploaded:        record,
		ClientViewURL:   resource.ClientViewURL,
		ImageServiceURL: resource.ImageServiceURL,
	}
	return record, imported, nil
}

func (service *BisqueService) UploadFile(ctx context.Context, record resourceRecord, path string, credentials BisqueCredentials) (BisqueUploadRecord, error) {
	uploaded, err := service.UploadNamedFile(ctx, record.OriginalName, path, credentials)
	if err != nil {
		return BisqueUploadRecord{}, err
	}
	uploaded.FileID = record.FileID
	return uploaded, nil
}

func (service *BisqueService) UploadNamedFile(ctx context.Context, originalName string, path string, credentials BisqueCredentials) (BisqueUploadRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return BisqueUploadRecord{}, err
	}
	defer func() {
		_ = file.Close()
	}()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", safeOriginalFilename(originalName))
	if err != nil {
		return BisqueUploadRecord{}, err
	}
	if _, err := io.Copy(part, file); err != nil {
		return BisqueUploadRecord{}, err
	}
	if err := writer.Close(); err != nil {
		return BisqueUploadRecord{}, err
	}
	data, _, err := service.fetch(ctx, http.MethodPost, service.endpoint("/import/transfer"), &body, writer.FormDataContentType(), credentials)
	if err != nil {
		return BisqueUploadRecord{}, err
	}
	resources := service.withLinks(parseBisqueResources(data))
	if len(resources) == 0 {
		return BisqueUploadRecord{}, bisqueServerError("BisQue upload response did not include a resource")
	}
	uploaded := resources[0]
	return BisqueUploadRecord{
		ResourceURI:  uploaded.ResourceURI,
		Name:         uploaded.Name,
		ResourceUniq: uploaded.ResourceUniq,
	}, nil
}

func (service *BisqueService) fetchResource(ctx context.Context, resourceURI string, credentials BisqueCredentials) (BisqueResource, error) {
	data, _, err := service.fetch(ctx, http.MethodGet, resourceURI, nil, "", credentials)
	if err != nil {
		return BisqueResource{}, err
	}
	resources := service.withLinks(parseBisqueResources(data))
	if len(resources) == 0 {
		return BisqueResource{}, bisqueServerError("BisQue metadata response did not include a resource")
	}
	return resources[0], nil
}

func (service *BisqueService) fetch(ctx context.Context, method string, rawURL string, body io.Reader, contentType string, credentials BisqueCredentials) ([]byte, string, error) {
	return service.fetchWithClient(ctx, service.client, method, rawURL, body, contentType, credentials)
}

func (service *BisqueService) fetchWithClient(ctx context.Context, client *http.Client, method string, rawURL string, body io.Reader, contentType string, credentials BisqueCredentials) ([]byte, string, error) {
	normalized, err := service.normalizeAllowedURL(rawURL)
	if err != nil {
		return nil, "", err
	}
	request, err := http.NewRequestWithContext(ctx, method, normalized, body)
	if err != nil {
		return nil, "", err
	}
	if contentType != "" {
		request.Header.Set("Content-Type", contentType)
	}
	username, password := service.credentialsForRequest(credentials)
	if username != "" || password != "" {
		request.SetBasicAuth(username, password)
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, "", err
	}
	defer func() {
		_ = response.Body.Close()
	}()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, "", bisqueUpstreamError{statusCode: response.StatusCode, message: "BisQue upstream returned " + response.Status}
	}
	limited := io.LimitReader(response.Body, service.maxImportSize+1)
	data, err := io.ReadAll(limited)
	if err != nil {
		return nil, "", err
	}
	if int64(len(data)) > service.maxImportSize {
		return nil, "", bisqueClientError("BisQue response exceeded import size limit")
	}
	return data, response.Header.Get("Content-Type"), nil
}

func (service *BisqueService) credentialsForRequest(credentials BisqueCredentials) (string, string) {
	username := strings.TrimSpace(credentials.Username)
	if username != "" || credentials.Password != "" {
		return username, credentials.Password
	}
	return service.username, service.password
}

func (service *BisqueService) endpoint(path string) string {
	return service.rootURL + "/" + strings.TrimLeft(path, "/")
}

func (service *BisqueService) normalizeAllowedURL(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", bisqueClientError("BisQue URL is required")
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return "", bisqueClientError("invalid BisQue URL")
	}
	if !parsed.IsAbs() {
		parsed = service.root.ResolveReference(parsed)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", bisqueClientError("BisQue URL must use http or https")
	}
	for _, allowed := range service.allowedRoots {
		if sameURLAuthority(parsed, allowed) && pathWithinURLRoot(parsed.EscapedPath(), allowed.EscapedPath()) {
			return parsed.String(), nil
		}
	}
	return "", bisqueClientError("BisQue URL is outside the configured allowed roots")
}

func (service *BisqueService) blobURL(resource BisqueResource) (string, error) {
	resourceUniq := strings.TrimSpace(resource.ResourceUniq)
	if resourceUniq == "" {
		resourceUniq = pathBaseFromURL(resource.ResourceURI)
	}
	if resourceUniq == "" {
		return "", bisqueClientError("BisQue resource has no resource_uniq for blob download")
	}
	return service.endpoint("/blob_service/" + url.PathEscape(resourceUniq)), nil
}

func (service *BisqueService) withLinks(resources []BisqueResource) []BisqueResource {
	for i := range resources {
		if resources[i].ResourceURI != "" {
			resources[i].ClientViewURL = service.endpoint("/client_service/view") + "?resource=" + url.QueryEscape(resources[i].ResourceURI)
		}
		if resources[i].ResourceUniq != "" {
			resources[i].ImageServiceURL = service.endpoint("/image_service/" + url.PathEscape(resources[i].ResourceUniq))
		}
	}
	return resources
}

type bisqueXMLResource struct {
	XMLName      xml.Name
	Count        string              `xml:"count,attr"`
	Total        string              `xml:"total,attr"`
	Type         string              `xml:"type,attr"`
	URI          string              `xml:"uri,attr"`
	Name         string              `xml:"name,attr"`
	ResourceUniq string              `xml:"resource_uniq,attr"`
	Tags         []bisqueXMLTag      `xml:"tag"`
	Children     []bisqueXMLResource `xml:",any"`
}

type bisqueXMLTag struct {
	Name  string `xml:"name,attr"`
	Value string `xml:"value,attr"`
}

func bisqueSessionHasAuthenticatedUser(data []byte) bool {
	var root bisqueXMLResource
	if err := xml.Unmarshal(data, &root); err != nil {
		return false
	}
	return bisqueXMLHasTag(root, "user")
}

func bisqueXMLHasTag(resource bisqueXMLResource, name string) bool {
	for _, tag := range resource.Tags {
		if strings.EqualFold(strings.TrimSpace(tag.Name), name) {
			return true
		}
	}
	for _, child := range resource.Children {
		if bisqueXMLHasTag(child, name) {
			return true
		}
	}
	return false
}

func parseBisqueResources(data []byte) []BisqueResource {
	resources, _ := parseBisqueSearchResponse(data)
	return resources
}

func parseBisqueSearchResponse(data []byte) ([]BisqueResource, int) {
	resources := []BisqueResource{}
	var root bisqueXMLResource
	if err := xml.Unmarshal(data, &root); err != nil {
		return resources, 0
	}
	collectBisqueResources(root, &resources)
	return resources, bisqueSearchCount(root, len(resources))
}

func bisqueSearchCount(root bisqueXMLResource, fallback int) int {
	for _, candidate := range []string{root.Total, root.Count} {
		total, err := strconv.Atoi(strings.TrimSpace(candidate))
		if err == nil && total >= 0 {
			return total
		}
	}
	return fallback
}

func collectBisqueResources(parsed bisqueXMLResource, resources *[]BisqueResource) {
	if isBisqueResourceElement(parsed.XMLName.Local) && !isBisqueWrapperResource(parsed) {
		tags := map[string]string{}
		for _, tag := range parsed.Tags {
			if tag.Name != "" {
				tags[tag.Name] = tag.Value
			}
		}
		*resources = append(*resources, BisqueResource{
			ResourceURI:  strings.TrimSpace(parsed.URI),
			Name:         strings.TrimSpace(parsed.Name),
			ResourceUniq: strings.TrimSpace(parsed.ResourceUniq),
			ResourceType: parsed.XMLName.Local,
			Tags:         tags,
		})
	}
	for _, child := range parsed.Children {
		collectBisqueResources(child, resources)
	}
}

func isBisqueWrapperResource(parsed bisqueXMLResource) bool {
	if !strings.EqualFold(parsed.XMLName.Local, "resource") ||
		strings.TrimSpace(parsed.ResourceUniq) != "" {
		return false
	}
	if strings.EqualFold(strings.TrimSpace(parsed.Type), "uploaded") &&
		strings.TrimSpace(parsed.URI) == "" {
		return true
	}
	return strings.TrimSpace(parsed.Type) == "" &&
		strings.TrimSpace(parsed.Name) == "" &&
		len(parsed.Children) > 0
}

func isBisqueResourceElement(name string) bool {
	switch strings.ToLower(name) {
	case "image", "dataset", "resource", "file", "mex", "table", "gobject":
		return true
	default:
		return false
	}
}

func saveImportedUploadFile(root string, originalName string, contentType string, sourceURI string, source io.Reader, principal requestPrincipal) (uploadedFileRecord, error) {
	if err := os.MkdirAll(root, 0o755); err != nil {
		return uploadedFileRecord{}, err
	}
	fileID := domain.NewID("file")
	safeName := safeOriginalFilename(originalName)
	target := filepath.Join(root, fileID+"__"+safeName)
	if !pathIsUnderRoot(root, target) {
		return uploadedFileRecord{}, errUnsafeArtifactPath
	}
	destination, err := os.Create(target)
	if err != nil {
		return uploadedFileRecord{}, err
	}
	hasher := sha256.New()
	size, copyErr := io.Copy(io.MultiWriter(destination, hasher), source)
	closeErr := destination.Close()
	if copyErr != nil {
		_ = os.Remove(target)
		return uploadedFileRecord{}, copyErr
	}
	if closeErr != nil {
		_ = os.Remove(target)
		return uploadedFileRecord{}, closeErr
	}
	info, err := os.Stat(target)
	if err != nil {
		_ = os.Remove(target)
		return uploadedFileRecord{}, err
	}
	if info.Size() > 0 {
		size = info.Size()
	}
	if err := writeUploadMetadataRecord(root, fileID, uploadMetadataRecord{
		Principal:  principal.record(),
		SourceURI:  strings.TrimSpace(sourceURI),
		SourceType: "bisque_import",
	}); err != nil {
		_ = os.Remove(target)
		return uploadedFileRecord{}, err
	}
	return uploadedFileRecord{
		FileID:       fileID,
		OriginalName: safeName,
		ContentType:  contentTypeForUpload(safeName, contentType),
		SizeBytes:    size,
		SHA256:       hex.EncodeToString(hasher.Sum(nil)),
		CreatedAt:    info.ModTime().UTC().Format(time.RFC3339Nano),
		SourceURI:    strings.TrimSpace(sourceURI),
		PreviewURL:   "/v2/uploads/" + url.PathEscape(fileID) + "/preview",
		Principal:    principal.record(),
	}, nil
}

func pathBaseFromURL(raw string) string {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return ""
	}
	return strings.Trim(strings.TrimSpace(filepath.Base(parsed.Path)), "/")
}

func artifactName(artifact domain.ArtifactRecord, resolvedPath string) string {
	for _, candidate := range []string{
		artifact.Path,
		artifact.SourcePath,
		artifact.PreviewPath,
		fileStoragePath(artifact.StorageURI),
		resolvedPath,
		artifact.Title,
		artifact.ArtifactID,
	} {
		name := strings.TrimSpace(filepath.Base(candidate))
		if name != "" && name != "." && name != string(filepath.Separator) {
			return name
		}
	}
	return "artifact"
}

func sameURLAuthority(a *url.URL, b *url.URL) bool {
	return strings.EqualFold(a.Scheme, b.Scheme) && strings.EqualFold(a.Host, b.Host)
}

func pathWithinURLRoot(candidate string, root string) bool {
	candidate = "/" + strings.TrimLeft(candidate, "/")
	root = "/" + strings.TrimLeft(root, "/")
	if root == "/" {
		return true
	}
	return candidate == root || strings.HasPrefix(candidate, strings.TrimRight(root, "/")+"/")
}

type bisqueUpstreamError struct {
	statusCode int
	message    string
}

func (err bisqueUpstreamError) Error() string {
	return err.message
}

type bisqueClientError string

func (err bisqueClientError) Error() string {
	return string(err)
}

type bisqueServerError string

func (err bisqueServerError) Error() string {
	return string(err)
}

func writeBisqueNotConfigured(w http.ResponseWriter) {
	writeJSON(w, http.StatusNotImplemented, map[string]any{
		"status":  "not_configured",
		"service": "bisque",
		"detail":  "BisQue integration is not configured for this control plane",
	})
}

func writeBisqueError(w http.ResponseWriter, err error) {
	var upstream bisqueUpstreamError
	switch {
	case errors.As(err, &upstream):
		writeJSON(w, http.StatusBadGateway, map[string]any{"error": upstream.Error(), "upstream_status": upstream.statusCode})
	case errors.As(err, new(bisqueClientError)):
		writeError(w, http.StatusBadRequest, err)
	default:
		writeError(w, http.StatusInternalServerError, err)
	}
}

func prettyJSON(value any) string {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Sprintf("%v", value)
	}
	return string(data)
}
