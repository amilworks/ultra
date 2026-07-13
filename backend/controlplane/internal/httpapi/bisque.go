package httpapi

import (
	"bytes"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
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
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

var bisqueResourceTypePattern = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)
var bisqueScopePattern = regexp.MustCompile(`^[A-Za-z0-9_, -]+$`)
var bisqueSHA256Pattern = regexp.MustCompile(`^[a-f0-9]{64}$`)

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
	GetActiveBisqueCredentialForUser(context.Context, string, string) (domain.BisqueCredentialRecord, bool, error)
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

// OwnerPrincipal reports the exact tenant a durable linked session belongs to.
// Sessions held only in the in-memory cache have no trustworthy owner binding
// and therefore cannot authorize worker-side external mutations.
func (store *BisqueCredentialStore) OwnerPrincipal(ctx context.Context, sessionID string) (string, string, bool, error) {
	if store == nil || store.persistent == nil {
		return "", "", false, nil
	}
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return "", "", false, nil
	}
	record, found, err := store.persistent.GetBisqueCredentialBySessionID(contextOrBackground(ctx), sessionID)
	if err != nil {
		return "", "", false, err
	}
	if !found {
		return "", "", false, nil
	}
	return strings.TrimSpace(record.UserID), strings.TrimSpace(record.OrgID), true, nil
}

func (store *BisqueCredentialStore) OwnerUserID(ctx context.Context, sessionID string) (string, bool, error) {
	userID, _, known, err := store.OwnerPrincipal(ctx, sessionID)
	return userID, known, err
}

// ResolveLinkedSessionForUser returns the session id and credentials of a user's
// active linked BisQue account, looked up by account identity rather than the
// request's session cookie. This is what makes linked-account detection (and
// therefore agent tool registration) survive cookie expiry and cross-browser use.
func (store *BisqueCredentialStore) ResolveLinkedSessionForUser(ctx context.Context, userID string, orgID string) (string, BisqueCredentials, bool) {
	if store == nil || store.persistent == nil || store.cipher == nil {
		return "", BisqueCredentials{}, false
	}
	userID = strings.TrimSpace(userID)
	if userID == "" {
		return "", BisqueCredentials{}, false
	}
	record, found, err := store.persistent.GetActiveBisqueCredentialForUser(contextOrBackground(ctx), userID, orgID)
	if err != nil || !found {
		return "", BisqueCredentials{}, false
	}
	sessionID := strings.TrimSpace(record.SessionID)
	if sessionID == "" {
		return "", BisqueCredentials{}, false
	}
	password, err := store.cipher.Decrypt(record.PasswordCiphertext, record.PasswordNonce)
	if err != nil {
		return "", BisqueCredentials{}, false
	}
	credentials := BisqueCredentials{Username: strings.TrimSpace(record.Username), Password: password}
	if credentials.Username == "" {
		return "", BisqueCredentials{}, false
	}
	store.cache(sessionID, credentials)
	return sessionID, credentials, true
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
	ResourceType    string                 `json:"resource_type"`
	TagQuery        string                 `json:"tag_query"`
	TagOrder        string                 `json:"tag_order"`
	Query           string                 `json:"query"`
	NameContains    string                 `json:"name_contains"`
	Extensions      []string               `json:"extensions"`
	FileExtensions  []string               `json:"file_extensions"`
	MetadataFilters []bisqueMetadataFilter `json:"metadata_filters"`
	Scope           string                 `json:"scope"`
	Sort            string                 `json:"sort"`
	Limit           int                    `json:"limit"`
	Offset          int                    `json:"offset"`
	CountAll        bool                   `json:"count_all"`
}

// bisqueMetadataFilter compares a resource tag against a value. Numeric
// comparisons (gt/gte/lt/lte) are evaluated client-side because BisQue's
// tag_query compares numeric tags lexically ("7" > "50", "100" < "50"), which
// silently returns wrong results for relational queries such as "age above 50".
type bisqueMetadataFilter struct {
	Tag   string `json:"tag"`
	Op    string `json:"op"`
	Value string `json:"value"`
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

type preparedBisqueFileUpload struct {
	record  resourceRecord
	path    string
	content []byte
}

type preparedBisqueArtifactUpload struct {
	artifact domain.ArtifactRecord
	path     string
	content  []byte
}

type bisqueCreateDatasetRequest struct {
	Name         string   `json:"name"`
	ResourceURIs []string `json:"resource_uris"`
}

type bisquePushRequest struct {
	FileIDs       []string `json:"file_ids"`
	CollectionIDs []string `json:"collection_ids"`
	DatasetName   string   `json:"dataset_name"`
}

type bisquePushResponse struct {
	Count    int                   `json:"count"`
	Uploads  []BisqueUploadRecord  `json:"uploads"`
	Datasets []BisqueDatasetRecord `json:"datasets"`
}

type BisqueResource struct {
	ResourceURI     string            `json:"resource_uri"`
	Name            string            `json:"name,omitempty"`
	ResourceUniq    string            `json:"resource_uniq,omitempty"`
	ResourceType    string            `json:"resource_type,omitempty"`
	Tags            map[string]string `json:"tags,omitempty"`
	ClientViewURL   string            `json:"client_view_url,omitempty"`
	ImageServiceURL string            `json:"image_service_url,omitempty"`
	// MEX (module-execution) fields, populated only for resource_type == "mex".
	Status     string           `json:"status,omitempty"`
	ModuleName string           `json:"module_name,omitempty"`
	ModuleURI  string           `json:"module_uri,omitempty"`
	CreatedAt  string           `json:"created_at,omitempty"`
	StartedAt  string           `json:"started_at,omitempty"`
	Inputs     []BisqueMexParam `json:"inputs,omitempty"`
	Outputs    []BisqueMexParam `json:"outputs,omitempty"`
}

// BisqueMexParam is one input or output of a module run. Resource-typed params
// (image/file/table/dataset/resource/mex) carry the referenced resource URI and
// its viewable link; scalar params (number/string/boolean) carry Value only.
type BisqueMexParam struct {
	Name          string `json:"name"`
	Type          string `json:"type,omitempty"`
	Value         string `json:"value,omitempty"`
	ResourceURI   string `json:"resource_uri,omitempty"`
	ResourceUniq  string `json:"resource_uniq,omitempty"`
	ClientViewURL string `json:"client_view_url,omitempty"`
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
	FileID        string `json:"file_id,omitempty"`
	ArtifactID    string `json:"artifact_id,omitempty"`
	ResourceURI   string `json:"resource_uri"`
	Name          string `json:"name,omitempty"`
	ResourceUniq  string `json:"resource_uniq,omitempty"`
	ClientViewURL string `json:"client_view_url,omitempty"`
}

type BisqueDatasetRecord struct {
	CollectionID  string `json:"collection_id,omitempty"`
	Name          string `json:"name"`
	ResourceURI   string `json:"resource_uri"`
	ResourceUniq  string `json:"resource_uniq,omitempty"`
	MemberCount   int    `json:"member_count"`
	ClientViewURL string `json:"client_view_url,omitempty"`
}

type bisqueModuleRunRequest struct {
	MexURI  string `json:"mex_uri"`
	MexUniq string `json:"mex_uniq"`
	Mex     string `json:"mex"`
}

func (deps ServerDeps) handleBisqueSearch(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	authority, authorized := deps.authorizeBisqueRequest(w, r)
	if !authorized {
		return
	}
	var req bisqueSearchRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	response, err := deps.Bisque.Search(r.Context(), req, authority.Credentials)
	if err != nil {
		writeBisqueError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func (deps ServerDeps) handleBisqueModuleRun(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	authority, authorized := deps.authorizeBisqueRequest(w, r)
	if !authorized {
		return
	}
	var req bisqueModuleRunRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	mexRef := firstNonEmpty(strings.TrimSpace(req.MexURI), strings.TrimSpace(req.Mex), strings.TrimSpace(req.MexUniq))
	if mexRef == "" {
		writeError(w, http.StatusBadRequest, errors.New("mex_uri or mex_uniq is required"))
		return
	}
	run, err := deps.Bisque.GetModuleRun(r.Context(), mexRef, authority.Credentials)
	if err != nil {
		writeBisqueError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, run)
}

func (deps ServerDeps) handleImportBisqueResources(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	authority, authorized := deps.authorizeBisqueRequest(w, r)
	if !authorized {
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
	principal := authority.Principal
	uploaded := make([]uploadedFileRecord, 0, len(resources))
	imports := make([]BisqueImportRecord, 0, len(resources))
	for _, resourceURI := range resources {
		record, imported, err := deps.Bisque.ImportResource(r.Context(), root, resourceURI, principal, authority.Credentials)
		if err != nil {
			writeBisqueError(w, err)
			return
		}
		if err := deps.catalogUploadedFileWithEventMetadata(r.Context(), root, record, "resource.imported", domain.JSONMap{
			"input_url":                imported.InputURL,
			"import_status":            imported.Status,
			"bisque_resource_uri":      imported.ResourceURI,
			"bisque_resource_uniq":     imported.ResourceUniq,
			"bisque_client_view_url":   imported.ClientViewURL,
			"bisque_image_service_url": imported.ImageServiceURL,
		}); err != nil {
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
	authority, authorized := deps.authorizeBisqueRequest(w, r, domain.RemoteMutationIntentBisqueUpload)
	if !authorized {
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
	req.FileIDs = uniqueTrimmedStringValues(req.FileIDs)
	req.ArtifactIDs = uniqueTrimmedStringValues(req.ArtifactIDs)
	principal := authority.Principal
	preparedFiles := make([]preparedBisqueFileUpload, 0, len(req.FileIDs))
	for _, fileID := range req.FileIDs {
		record, path, err := deps.findUploadResourceForRequest(r.Context(), root, principal, fileID)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		if authority.Worker && !workerRunAllowsBisqueResourceRecord(authority.Run, record) {
			writeStoreError(w, store.ErrNotFound)
			return
		}
		preparedFiles = append(preparedFiles, preparedBisqueFileUpload{record: record, path: path})
	}
	preparedArtifacts := make([]preparedBisqueArtifactUpload, 0, len(req.ArtifactIDs))
	if len(req.ArtifactIDs) > 0 && deps.Store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "artifact store is not configured"})
		return
	}
	if len(req.ArtifactIDs) > 0 && strings.TrimSpace(deps.ArtifactRoot) == "" {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "artifact root is not configured"})
		return
	}
	for _, artifactID := range req.ArtifactIDs {
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
		if authority.Worker && !workerRunAllowsBisqueArtifact(authority.Run, artifact) {
			writeStoreError(w, store.ErrNotFound)
			return
		}
		preparedArtifacts = append(preparedArtifacts, preparedBisqueArtifactUpload{artifact: artifact, path: path})
	}
	mutationTargets := make([]bisqueMutationUploadTarget, 0, len(preparedFiles)+len(preparedArtifacts))
	for index, prepared := range preparedFiles {
		target, content, err := verifiedBisqueMutationTarget(
			"resource", prepared.record.OriginalName, prepared.path, prepared.record.SHA256, prepared.record.SizeBytes,
		)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		preparedFiles[index].content = content
		mutationTargets = append(mutationTargets, target)
	}
	for index, prepared := range preparedArtifacts {
		target, content, err := verifiedBisqueMutationTarget(
			"artifact", artifactName(prepared.artifact, prepared.path), prepared.path,
			prepared.artifact.SHA256, prepared.artifact.SizeBytes,
		)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		preparedArtifacts[index].content = content
		mutationTargets = append(mutationTargets, target)
	}
	receipt, cached, replayed, err := deps.beginWorkerBisqueMutation(
		r.Context(), authority, "bisque.upload",
		bisqueUploadMutationRequest{Targets: canonicalBisqueUploadTargets(mutationTargets)},
	)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if replayed {
		writeJSON(w, http.StatusOK, cached)
		return
	}

	// All local targets are authorized and resolved before the first external
	// write, so one invalid item cannot leave a partially uploaded batch.
	uploads := make([]BisqueUploadRecord, 0, len(preparedFiles)+len(preparedArtifacts))
	for _, prepared := range preparedFiles {
		uploaded, err := deps.Bisque.UploadFileBytes(r.Context(), prepared.record, prepared.content, authority.Credentials)
		if err != nil {
			deps.markWorkerBisqueMutationAmbiguous(r.Context(), receipt, err)
			writeBisqueError(w, err)
			return
		}
		uploads = append(uploads, uploaded)
	}
	for _, prepared := range preparedArtifacts {
		uploaded, err := deps.Bisque.UploadNamedBytes(
			r.Context(), artifactName(prepared.artifact, prepared.path), prepared.content, authority.Credentials,
		)
		if err != nil {
			deps.markWorkerBisqueMutationAmbiguous(r.Context(), receipt, err)
			writeBisqueError(w, err)
			return
		}
		uploaded.ArtifactID = prepared.artifact.ArtifactID
		uploads = append(uploads, uploaded)
	}
	response := bisqueUploadResponse{Count: len(uploads), Uploads: uploads}
	if err := deps.completeWorkerBisqueMutation(r.Context(), receipt, response); err != nil {
		deps.markWorkerBisqueMutationAmbiguous(r.Context(), receipt, err)
		writeError(w, http.StatusServiceUnavailable, errors.New("BisQue mutation completed upstream but its durable receipt could not be sealed"))
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func (deps ServerDeps) handleBisqueCreateDataset(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	authority, authorized := deps.authorizeBisqueRequest(w, r, domain.RemoteMutationIntentBisqueCreateDataset)
	if !authorized {
		return
	}
	var req bisqueCreateDatasetRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	req.Name = strings.TrimSpace(req.Name)
	req.ResourceURIs = canonicalBisqueDatasetMembers(req.ResourceURIs)
	if req.Name == "" || len(req.ResourceURIs) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("dataset name and at least one resource_uri are required"))
		return
	}
	for index, rawURI := range req.ResourceURIs {
		normalized, err := deps.Bisque.normalizeAllowedURL(rawURI)
		if err != nil {
			writeBisqueError(w, err)
			return
		}
		req.ResourceURIs[index] = normalized
	}
	req.ResourceURIs = canonicalBisqueDatasetMembers(req.ResourceURIs)
	if authority.Worker {
		uploadedURIs, err := deps.workerBisqueUploadedResourceURIs(r.Context(), authority.Run.RunID)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		for _, resourceURI := range req.ResourceURIs {
			if !workerRunAllowsBisqueDatasetURI(authority.Run, resourceURI, uploadedURIs) {
				writeStoreError(w, store.ErrNotFound)
				return
			}
		}
	}
	receipt, cached, replayed, err := deps.beginWorkerBisqueMutation(
		r.Context(), authority, "bisque.create_dataset",
		bisqueDatasetMutationRequest{Name: req.Name, ResourceURIs: req.ResourceURIs},
	)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	if replayed {
		writeJSON(w, http.StatusOK, cached)
		return
	}
	dataset, err := deps.Bisque.CreateDataset(r.Context(), req.Name, req.ResourceURIs, authority.Credentials)
	if err != nil {
		deps.markWorkerBisqueMutationAmbiguous(r.Context(), receipt, err)
		writeBisqueError(w, err)
		return
	}
	if err := deps.completeWorkerBisqueMutation(r.Context(), receipt, dataset); err != nil {
		deps.markWorkerBisqueMutationAmbiguous(r.Context(), receipt, err)
		writeError(w, http.StatusServiceUnavailable, errors.New("BisQue mutation completed upstream but its durable receipt could not be sealed"))
		return
	}
	writeJSON(w, http.StatusOK, dataset)
}

func (deps ServerDeps) handleBisquePush(w http.ResponseWriter, r *http.Request) {
	if deps.Bisque == nil {
		writeBisqueNotConfigured(w)
		return
	}
	authority, authorized := deps.authorizeBisqueRequest(w, r)
	if !authorized {
		return
	}
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	var req bisquePushRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	fileIDs := uniqueTrimmedStringValues(req.FileIDs)
	collectionIDs := uniqueTrimmedStringValues(req.CollectionIDs)
	if len(fileIDs) == 0 && len(collectionIDs) == 0 {
		writeError(w, http.StatusBadRequest, errors.New("at least one file_id or collection_id is required"))
		return
	}
	credentials := authority.Credentials
	if strings.TrimSpace(credentials.Username) == "" && strings.TrimSpace(deps.Bisque.username) == "" {
		writeError(w, http.StatusBadRequest, errors.New("link your BisQue account in Settings before pushing resources"))
		return
	}
	principal := authority.Principal
	uploads := []BisqueUploadRecord{}
	datasets := []BisqueDatasetRecord{}
	datasetName := strings.TrimSpace(req.DatasetName)

	looseUploads, err := deps.pushBisqueFiles(r.Context(), root, principal, fileIDs, credentials)
	if err != nil {
		writeBisquePushError(w, err)
		return
	}
	uploads = append(uploads, looseUploads...)
	if datasetName != "" && len(collectionIDs) == 0 {
		dataset, err := deps.Bisque.CreateDataset(r.Context(), datasetName, bisqueUploadResourceURIs(looseUploads), credentials)
		if err != nil {
			writeBisqueError(w, err)
			return
		}
		datasets = append(datasets, dataset)
	}

	for _, collectionID := range collectionIDs {
		collections, ok := deps.resourceCollectionStore()
		if !ok {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "resource collections are not configured"})
			return
		}
		collection, err := collections.GetResourceCollectionForUser(r.Context(), collectionID, principal.UserID, principal.OrgID)
		if err != nil {
			writeStoreError(w, err)
			return
		}
		page, err := collections.ListResourcesForCollectionForUser(r.Context(), domain.ResourceCollectionResourceListInput{
			CollectionID: collectionID,
			UserID:       principal.UserID,
			OrgID:        principal.OrgID,
			Limit:        uploadSessionMaxFilesPerBatch,
		})
		if err != nil {
			writeStoreError(w, err)
			return
		}
		if page.TotalCount > len(page.Resources) {
			writeError(w, http.StatusBadRequest, fmt.Errorf("folder %q contains too many resources to push in one request", collection.Name))
			return
		}
		if len(page.Resources) == 0 {
			writeError(w, http.StatusBadRequest, fmt.Errorf("folder %q has no resources to push", collection.Name))
			return
		}
		memberIDs := make([]string, 0, len(page.Resources))
		for _, resource := range page.Resources {
			memberIDs = append(memberIDs, resource.ResourceID)
		}
		memberUploads, err := deps.pushBisqueFiles(r.Context(), root, principal, memberIDs, credentials)
		if err != nil {
			writeBisquePushError(w, err)
			return
		}
		uploads = append(uploads, memberUploads...)
		name := strings.TrimSpace(collection.Name)
		if datasetName != "" && len(collectionIDs) == 1 {
			name = datasetName
		}
		if name == "" {
			name = "ultra-folder"
		}
		dataset, err := deps.Bisque.CreateDataset(r.Context(), name, bisqueUploadResourceURIs(memberUploads), credentials)
		if err != nil {
			writeBisqueError(w, err)
			return
		}
		dataset.CollectionID = collection.CollectionID
		datasets = append(datasets, dataset)
	}
	writeJSON(w, http.StatusOK, bisquePushResponse{Count: len(uploads), Uploads: uploads, Datasets: datasets})
}

func (deps ServerDeps) pushBisqueFiles(ctx context.Context, root string, principal requestPrincipal, fileIDs []string, credentials BisqueCredentials) ([]BisqueUploadRecord, error) {
	uploads := make([]BisqueUploadRecord, 0, len(fileIDs))
	for _, fileID := range fileIDs {
		record, path, err := deps.findUploadResourceForRequest(ctx, root, principal, fileID)
		if err != nil {
			return nil, bisquePushFileError{fileID: fileID, name: record.OriginalName, err: err}
		}
		uploaded, err := deps.Bisque.UploadFile(ctx, record, path, credentials)
		if err != nil {
			return nil, err
		}
		uploads = append(uploads, uploaded)
	}
	return uploads, nil
}

func bisqueUploadResourceURIs(uploads []BisqueUploadRecord) []string {
	uris := make([]string, 0, len(uploads))
	for _, upload := range uploads {
		if strings.TrimSpace(upload.ResourceURI) != "" {
			uris = append(uris, upload.ResourceURI)
		}
	}
	return uris
}

type bisquePushFileError struct {
	fileID string
	name   string
	err    error
}

func (err bisquePushFileError) Error() string {
	label := strings.TrimSpace(err.name)
	if label == "" {
		label = err.fileID
	}
	return fmt.Sprintf("resource %q is not available for push: %v", label, err.err)
}

func (err bisquePushFileError) Unwrap() error {
	return err.err
}

func writeBisquePushError(w http.ResponseWriter, err error) {
	var pushErr bisquePushFileError
	if errors.As(err, &pushErr) {
		status := http.StatusInternalServerError
		if errors.Is(pushErr.err, store.ErrNotFound) {
			status = http.StatusNotFound
		} else if errors.Is(pushErr.err, store.ErrConflict) {
			status = http.StatusConflict
		}
		writeError(w, status, err)
		return
	}
	writeBisqueError(w, err)
}

func (deps ServerDeps) bisqueJobMetadataFromRequest(r *http.Request) domain.JSONMap {
	metadata := domain.JSONMap{}
	if deps.BisqueCredentials == nil {
		return metadata
	}
	if sessionID := deps.resolveBisqueSessionForRequest(r); sessionID != "" {
		metadata["bisque_session_id"] = sessionID
	}
	return metadata
}

// bisqueRunBindingFromRequest snapshots a non-secret digest of the exact
// durable linked account selected when the authenticated user creates a run.
// The opaque session id remains transient job metadata; its digest prevents an
// old run from inheriting a newly linked account after unlink/relink.
func (deps ServerDeps) bisqueRunBindingFromRequest(r *http.Request, principal requestPrincipal) (string, domain.JSONMap, bool) {
	if deps.BisqueCredentials == nil {
		return "", nil, false
	}
	sessionID := deps.resolveBisqueSessionForRequest(r)
	if sessionID == "" {
		return "", nil, false
	}
	ownerUserID, ownerOrgID, known, err := deps.BisqueCredentials.OwnerPrincipal(r.Context(), sessionID)
	if err != nil || !known || ownerUserID == "" || ownerOrgID == "" ||
		ownerUserID != strings.TrimSpace(principal.UserID) || ownerOrgID != strings.TrimSpace(principal.OrgID) {
		return "", nil, false
	}
	return sessionID, domain.JSONMap{
		"schema_version": "ultra.bisque_account_binding.v1",
		"authority":      "control_plane",
		"session_sha256": bisqueSessionDigest(sessionID),
		"root_url":       deps.bisqueRootURL(),
		"owner_user_id":  ownerUserID,
		"owner_org_id":   ownerOrgID,
	}, true
}

func bisqueSessionDigest(sessionID string) string {
	digest := sha256.Sum256([]byte(strings.TrimSpace(sessionID)))
	return hex.EncodeToString(digest[:])
}

func bisqueAccountBindingForRun(run domain.RunRecord) (domain.JSONMap, bool) {
	binding, ok := jsonMapValue(run.Metadata[domain.BisqueAccountBindingMetadataKey])
	if !ok {
		return nil, false
	}
	schema, _ := safeMetadataString(binding["schema_version"], 128)
	authority, _ := safeMetadataString(binding["authority"], 128)
	sessionSHA, _ := safeMetadataString(binding["session_sha256"], 128)
	ownerUserID, _ := safeMetadataString(binding["owner_user_id"], 512)
	ownerOrgID, _ := safeMetadataString(binding["owner_org_id"], 512)
	rootURL, _ := safeMetadataString(binding["root_url"], 4096)
	if schema != "ultra.bisque_account_binding.v1" || authority != "control_plane" ||
		!bisqueSHA256Pattern.MatchString(sessionSHA) || ownerUserID == "" ||
		ownerOrgID == "" || rootURL == "" || ownerUserID != strings.TrimSpace(run.UserID) ||
		ownerOrgID != trustedRunOrgID(run) {
		return nil, false
	}
	return domain.JSONMap{
		"schema_version": schema,
		"authority":      authority,
		"session_sha256": sessionSHA,
		"root_url":       rootURL,
		"owner_user_id":  ownerUserID,
		"owner_org_id":   ownerOrgID,
	}, true
}

func bisqueSessionMatchesRunBinding(run domain.RunRecord, sessionID string, expectedRoot string) bool {
	binding, ok := bisqueAccountBindingForRun(run)
	if !ok || strings.TrimSpace(sessionID) == "" {
		return false
	}
	sessionSHA, _ := safeMetadataString(binding["session_sha256"], 128)
	rootURL, _ := safeMetadataString(binding["root_url"], 4096)
	return subtle.ConstantTimeCompare([]byte(sessionSHA), []byte(bisqueSessionDigest(sessionID))) == 1 &&
		strings.TrimRight(rootURL, "/") == strings.TrimRight(strings.TrimSpace(expectedRoot), "/")
}

// resolveBisqueSessionForRequest returns the BisQue session id to attach to a
// run for this user: the cookie session when present, otherwise the user's
// durably-linked account session looked up by identity. The durable fallback is
// what lets a linked user run BisQue tools after their session cookie has
// expired or from a different browser — the reported "tools not registered" bug.
func (deps ServerDeps) resolveBisqueSessionForRequest(r *http.Request) string {
	if deps.BisqueCredentials == nil {
		return ""
	}
	if sessionID := bisqueSessionIDFromRequest(r); sessionID != "" {
		if _, ok, err := deps.BisqueCredentials.GetWithContext(r.Context(), sessionID); ok && err == nil {
			return sessionID
		}
	}
	principal := deps.principalFromRequest(r, "")
	if strings.TrimSpace(principal.UserID) == "" {
		return ""
	}
	if sessionID, _, ok := deps.BisqueCredentials.ResolveLinkedSessionForUser(r.Context(), principal.UserID, principal.OrgID); ok {
		return sessionID
	}
	return ""
}

func (deps ServerDeps) bisqueCredentialsFromRequest(ctx context.Context, r *http.Request) BisqueCredentials {
	// A presented worker credential never consumes browser cookies or global
	// service credentials. Worker handlers obtain credentials only from
	// authorizeBisqueRequest after run/binding/lease validation.
	if workerTokenFromRequest(r) != "" {
		return BisqueCredentials{}
	}
	return deps.directBisqueCredentialsFromRequest(ctx, r, deps.principalFromRequest(r, ""))
}

func (deps ServerDeps) directBisqueCredentialsFromRequest(ctx context.Context, r *http.Request, principal requestPrincipal) BisqueCredentials {
	if deps.BisqueCredentials == nil {
		return BisqueCredentials{}
	}
	if credentials, ok, _ := deps.BisqueCredentials.GetWithContext(ctx, bisqueSessionIDFromRequest(r)); ok {
		return credentials
	}
	// Durable direct-user fallback survives an expired browser cookie. Worker
	// fallback is intentionally implemented only in authorizeBisqueRequest,
	// where it must match the immutable run binding digest.
	if strings.TrimSpace(principal.UserID) != "" {
		if _, credentials, ok := deps.BisqueCredentials.ResolveLinkedSessionForUser(ctx, principal.UserID, principal.OrgID); ok {
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
	if _, err := resolveBisqueMetadataFilters(req.MetadataFilters); err != nil {
		return bisqueSearchResponse{}, err
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
		// Prefer BisQue's authoritative view=count total (one request); fall back
		// to paging only if the server doesn't answer with a count tag. The old
		// paging-and-sum path is O(N) requests and over-counted by one at exact
		// page-size multiples.
		if counted, ok, cerr := service.countViaViewCount(ctx, resourceType, req, credentials); cerr == nil && ok {
			total = counted
		} else {
			total, err = service.countAllSearchResults(ctx, resourceType, req, credentials)
			if err != nil {
				return bisqueSearchResponse{}, err
			}
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
	if len(filters.metadata) > 0 {
		// Tag values are only serialized at view=full; client-side metadata
		// filtering needs them present in each resource element.
		query.Set("view", "full")
	}
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
	metadata     []bisqueResolvedMetadataFilter
}

func bisqueSearchFiltersFromRequest(req bisqueSearchRequest) bisqueSearchFilters {
	metadata, _ := resolveBisqueMetadataFilters(req.MetadataFilters)
	return bisqueSearchFilters{
		nameContains: strings.ToLower(strings.TrimSpace(req.NameContains)),
		extensions:   bisqueNormalizedExtensions(append(req.Extensions, req.FileExtensions...)),
		metadata:     metadata,
	}
}

func (filters bisqueSearchFilters) enabled() bool {
	return filters.nameContains != "" || len(filters.extensions) > 0 || len(filters.metadata) > 0
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
	for _, filter := range filters.metadata {
		if !filter.matches(resource) {
			return false
		}
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

var bisqueMetadataFilterOps = map[string]bool{
	"eq": true, "ne": true, "gt": true, "gte": true, "lt": true, "lte": true, "contains": true,
}

type bisqueResolvedMetadataFilter struct {
	tag      string
	op       string
	value    string
	valueLow string
	numeric  bool
	numValue float64
}

// resolveBisqueMetadataFilters validates and normalizes the requested metadata
// filters. Relational ops require a numeric value because they are evaluated
// numerically client-side (BisQue's lexical tag comparison is the bug this
// path exists to avoid).
func resolveBisqueMetadataFilters(raw []bisqueMetadataFilter) ([]bisqueResolvedMetadataFilter, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	resolved := make([]bisqueResolvedMetadataFilter, 0, len(raw))
	for _, filter := range raw {
		tag := strings.TrimSpace(filter.Tag)
		if tag == "" {
			return nil, bisqueClientError("BisQue metadata filter requires a tag name")
		}
		op := strings.ToLower(strings.TrimSpace(filter.Op))
		if op == "" {
			op = "eq"
		}
		if !bisqueMetadataFilterOps[op] {
			return nil, bisqueClientError("unsupported BisQue metadata filter op: " + op)
		}
		value := strings.TrimSpace(filter.Value)
		resolvedFilter := bisqueResolvedMetadataFilter{
			tag:      tag,
			op:       op,
			value:    value,
			valueLow: strings.ToLower(value),
		}
		if num, err := strconv.ParseFloat(value, 64); err == nil {
			resolvedFilter.numeric = true
			resolvedFilter.numValue = num
		}
		switch op {
		case "gt", "gte", "lt", "lte":
			if !resolvedFilter.numeric {
				return nil, bisqueClientError("BisQue metadata filter op " + op + " requires a numeric value")
			}
		}
		resolved = append(resolved, resolvedFilter)
	}
	return resolved, nil
}

func (filter bisqueResolvedMetadataFilter) matches(resource BisqueResource) bool {
	raw, ok := bisqueTagValue(resource, filter.tag)
	if !ok {
		return false
	}
	trimmed := strings.TrimSpace(raw)
	switch filter.op {
	case "contains":
		return strings.Contains(strings.ToLower(trimmed), filter.valueLow)
	case "eq":
		if filter.numeric {
			if num, err := strconv.ParseFloat(trimmed, 64); err == nil {
				return num == filter.numValue
			}
		}
		return strings.EqualFold(trimmed, filter.value)
	case "ne":
		if filter.numeric {
			if num, err := strconv.ParseFloat(trimmed, 64); err == nil {
				return num != filter.numValue
			}
		}
		return !strings.EqualFold(trimmed, filter.value)
	case "gt", "gte", "lt", "lte":
		num, err := strconv.ParseFloat(trimmed, 64)
		if err != nil {
			return false
		}
		switch filter.op {
		case "gt":
			return num > filter.numValue
		case "gte":
			return num >= filter.numValue
		case "lt":
			return num < filter.numValue
		case "lte":
			return num <= filter.numValue
		}
	}
	return false
}

func bisqueTagValue(resource BisqueResource, tag string) (string, bool) {
	if len(resource.Tags) == 0 {
		return "", false
	}
	if value, ok := resource.Tags[tag]; ok {
		return value, true
	}
	for name, value := range resource.Tags {
		if strings.EqualFold(name, tag) {
			return value, true
		}
	}
	return "", false
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

func (service *BisqueService) UploadFileBytes(ctx context.Context, record resourceRecord, content []byte, credentials BisqueCredentials) (BisqueUploadRecord, error) {
	uploaded, err := service.UploadNamedBytes(ctx, record.OriginalName, content, credentials)
	if err != nil {
		return BisqueUploadRecord{}, err
	}
	uploaded.FileID = record.FileID
	return uploaded, nil
}

func (service *BisqueService) UploadNamedFile(ctx context.Context, originalName string, path string, credentials BisqueCredentials) (BisqueUploadRecord, error) {
	// A directory bundle (e.g. an OME-Zarr: a .ome.zarr tree of chunk files) is not a single
	// uploadable file. Without this guard os.Open succeeds on the directory and io.Copy below
	// fails with EISDIR, which writeBisqueError maps to an opaque 500 that propagates and fails
	// the whole run. Return a typed client error so the caller gets a clean, actionable 4xx.
	if info, statErr := os.Stat(path); statErr == nil && info.IsDir() {
		return BisqueUploadRecord{}, bisqueClientError(fmt.Sprintf(
			"resource %q is a directory bundle (e.g. an OME-Zarr), not a single file; bundle upload to BisQue is not supported",
			originalName,
		))
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return BisqueUploadRecord{}, err
	}
	return service.UploadNamedBytes(ctx, originalName, content, credentials)
}

func (service *BisqueService) UploadNamedBytes(ctx context.Context, originalName string, content []byte, credentials BisqueCredentials) (BisqueUploadRecord, error) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", safeOriginalFilename(originalName))
	if err != nil {
		return BisqueUploadRecord{}, err
	}
	if _, err := part.Write(content); err != nil {
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
		ResourceURI:   uploaded.ResourceURI,
		Name:          uploaded.Name,
		ResourceUniq:  uploaded.ResourceUniq,
		ClientViewURL: uploaded.ClientViewURL,
	}, nil
}

type bisqueDatasetMemberXML struct {
	XMLName xml.Name `xml:"value"`
	Index   int      `xml:"index,attr"`
	Type    string   `xml:"type,attr"`
	URI     string   `xml:",chardata"`
}

type bisqueDatasetXML struct {
	XMLName xml.Name                 `xml:"dataset"`
	Name    string                   `xml:"name,attr"`
	Members []bisqueDatasetMemberXML `xml:"value"`
}

func (service *BisqueService) CreateDataset(ctx context.Context, name string, memberURIs []string, credentials BisqueCredentials) (BisqueDatasetRecord, error) {
	if service == nil {
		return BisqueDatasetRecord{}, bisqueClientError("BisQue integration is not configured")
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return BisqueDatasetRecord{}, bisqueClientError("BisQue dataset name is required")
	}
	members := make([]bisqueDatasetMemberXML, 0, len(memberURIs))
	seen := map[string]bool{}
	for _, rawURI := range memberURIs {
		memberURI, err := service.normalizeAllowedURL(rawURI)
		if err != nil {
			return BisqueDatasetRecord{}, err
		}
		if seen[memberURI] {
			continue
		}
		seen[memberURI] = true
		members = append(members, bisqueDatasetMemberXML{
			Index: len(members),
			Type:  "object",
			URI:   memberURI,
		})
	}
	if len(members) == 0 {
		return BisqueDatasetRecord{}, bisqueClientError("BisQue dataset requires at least one member resource")
	}
	payload, err := xml.Marshal(bisqueDatasetXML{Name: name, Members: members})
	if err != nil {
		return BisqueDatasetRecord{}, err
	}
	data, _, err := service.fetch(ctx, http.MethodPost, service.endpoint("/data_service/dataset"), bytes.NewReader(payload), "text/xml", credentials)
	if err != nil {
		return BisqueDatasetRecord{}, err
	}
	resources := service.withLinks(parseBisqueResources(data))
	if len(resources) == 0 {
		return BisqueDatasetRecord{}, bisqueServerError("BisQue dataset response did not include a resource")
	}
	created := resources[0]
	datasetName := created.Name
	if strings.TrimSpace(datasetName) == "" {
		datasetName = name
	}
	return BisqueDatasetRecord{
		Name:          datasetName,
		ResourceURI:   created.ResourceURI,
		ResourceUniq:  created.ResourceUniq,
		MemberCount:   len(members),
		ClientViewURL: created.ClientViewURL,
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

// GetModuleRun fetches a single module-execution (MEX) at view=deep so its
// nested inputs and outputs — the resulting segmentation masks, tables, files —
// are resolved. mexRef may be a full resource URI or a bare resource_uniq.
func (service *BisqueService) GetModuleRun(ctx context.Context, mexRef string, credentials BisqueCredentials) (BisqueResource, error) {
	if service == nil {
		return BisqueResource{}, bisqueClientError("BisQue integration is not configured")
	}
	mexRef = strings.TrimSpace(mexRef)
	if mexRef == "" {
		return BisqueResource{}, bisqueClientError("a BisQue module run URI or resource_uniq is required")
	}
	if !strings.Contains(mexRef, "/") {
		// Bare resource_uniq such as 00-ZDLCqkxPUFZssqEtLovR9C.
		mexRef = service.endpoint("/data_service/" + url.PathEscape(mexRef))
	}
	deepURL, err := service.deepViewURL(mexRef)
	if err != nil {
		return BisqueResource{}, err
	}
	data, _, err := service.fetch(ctx, http.MethodGet, deepURL, nil, "", credentials)
	if err != nil {
		return BisqueResource{}, err
	}
	resources := service.withLinks(parseBisqueResources(data))
	mex := bisqueFirstMexResource(resources)
	if mex == nil {
		return BisqueResource{}, bisqueClientError("the requested BisQue resource is not a module run (mex)")
	}
	return *mex, nil
}

func (service *BisqueService) deepViewURL(resourceURI string) (string, error) {
	normalized, err := service.normalizeAllowedURL(resourceURI)
	if err != nil {
		return "", err
	}
	parsed, err := url.Parse(normalized)
	if err != nil {
		return "", err
	}
	query := parsed.Query()
	query.Set("view", "deep")
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}

func bisqueFirstMexResource(resources []BisqueResource) *BisqueResource {
	for i := range resources {
		if strings.EqualFold(resources[i].ResourceType, "mex") {
			return &resources[i]
		}
	}
	return nil
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
			resources[i].ClientViewURL = service.clientViewURL(resources[i].ResourceURI)
		}
		// image_service_url only makes sense for image resources; a mex or
		// dataset has no image-service rendering, so don't fabricate one.
		if resources[i].ResourceUniq != "" && strings.EqualFold(resources[i].ResourceType, "image") {
			resources[i].ImageServiceURL = service.endpoint("/image_service/" + url.PathEscape(resources[i].ResourceUniq))
		}
		service.withParamLinks(resources[i].Inputs)
		service.withParamLinks(resources[i].Outputs)
		if moduleURI := strings.TrimSpace(resources[i].ModuleURI); moduleURI != "" {
			resources[i].ModuleURI = moduleURI
		}
	}
	return resources
}

func (service *BisqueService) withParamLinks(params []BisqueMexParam) {
	for j := range params {
		if strings.TrimSpace(params[j].ResourceURI) != "" {
			params[j].ClientViewURL = service.clientViewURL(params[j].ResourceURI)
		}
	}
}

// clientViewURL builds the canonical BisQue viewer link for a resource. The
// BisQue web UI (bq_ui_upload.js, uploaded.html, every ResourceBrowser code
// path) appends the full resource URI to /client_service/view?resource= without
// percent-encoding it — ':' and '/' are legal in a URL query component, and
// BisQue resource URIs are clean base58 paths, so the raw form is both valid and
// the link users see in BisQue itself. This is uniform across images, tables,
// and datasets. Relative resource URIs are resolved against the configured root
// so the viewer always receives an absolute resource reference.
func (service *BisqueService) clientViewURL(resourceURI string) string {
	resourceURI = strings.TrimSpace(resourceURI)
	if resourceURI == "" {
		return ""
	}
	if !strings.HasPrefix(resourceURI, "http://") && !strings.HasPrefix(resourceURI, "https://") {
		resourceURI = service.rootURL + "/" + strings.TrimLeft(resourceURI, "/")
	}
	return service.endpoint("/client_service/view") + "?resource=" + resourceURI
}

type bisqueXMLResource struct {
	XMLName      xml.Name
	Count        string              `xml:"count,attr"`
	Total        string              `xml:"total,attr"`
	Type         string              `xml:"type,attr"`
	URI          string              `xml:"uri,attr"`
	Name         string              `xml:"name,attr"`
	Value        string              `xml:"value,attr"`
	Created      string              `xml:"created,attr"`
	TS           string              `xml:"ts,attr"`
	ResourceUniq string              `xml:"resource_uniq,attr"`
	Tags         []bisqueXMLTag      `xml:"tag"`
	Children     []bisqueXMLResource `xml:",any"`
}

// bisqueXMLTag is recursive so nested MEX containers — <tag name="outputs">
// holding <tag name="Segmentation" type="image" value="...">  — are preserved
// rather than flattened away.
type bisqueXMLTag struct {
	Name  string         `xml:"name,attr"`
	Type  string         `xml:"type,attr"`
	Value string         `xml:"value,attr"`
	Tags  []bisqueXMLTag `xml:"tag"`
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
		resource := BisqueResource{
			ResourceURI:  strings.TrimSpace(parsed.URI),
			Name:         strings.TrimSpace(parsed.Name),
			ResourceUniq: strings.TrimSpace(parsed.ResourceUniq),
			ResourceType: parsed.XMLName.Local,
			Tags:         tags,
		}
		if strings.EqualFold(parsed.XMLName.Local, "mex") {
			enrichBisqueMexResource(&resource, parsed)
		}
		*resources = append(*resources, resource)
	}
	for _, child := range parsed.Children {
		collectBisqueResources(child, resources)
	}
}

// enrichBisqueMexResource lifts a module run's status, module reference, and
// nested input/output parameters out of the raw XML. Status is the mex element's
// own value attribute and the module is its type attribute (a module URI) — both
// of which the generic resource parser drops. Inputs/outputs are only present at
// view=deep; in list (short) views they are absent and these slices stay empty.
func enrichBisqueMexResource(resource *BisqueResource, parsed bisqueXMLResource) {
	resource.Status = strings.TrimSpace(parsed.Value)
	resource.ModuleName = strings.TrimSpace(parsed.Name)
	if moduleURI := strings.TrimSpace(parsed.Type); strings.Contains(moduleURI, "/data_service/") {
		resource.ModuleURI = moduleURI
	}
	resource.CreatedAt = firstNonEmpty(strings.TrimSpace(parsed.Created), strings.TrimSpace(parsed.TS))
	for _, tag := range parsed.Tags {
		switch strings.ToLower(strings.TrimSpace(tag.Name)) {
		case "inputs":
			resource.Inputs = bisqueMexParamsFromContainer(tag)
		case "outputs":
			resource.Outputs = bisqueMexParamsFromContainer(tag)
		case "start-time":
			resource.StartedAt = strings.TrimSpace(tag.Value)
		}
	}
}

func bisqueMexParamsFromContainer(container bisqueXMLTag) []BisqueMexParam {
	params := make([]BisqueMexParam, 0, len(container.Tags))
	for _, tag := range container.Tags {
		name := strings.TrimSpace(tag.Name)
		if name == "" {
			continue
		}
		paramType := strings.TrimSpace(tag.Type)
		value := strings.TrimSpace(tag.Value)
		param := BisqueMexParam{Name: name, Type: paramType, Value: value}
		if bisqueParamIsResourceReference(paramType, value) {
			param.ResourceURI = value
			param.ResourceUniq = pathBaseFromURL(value)
		}
		params = append(params, param)
	}
	return params
}

// bisqueParamIsResourceReference reports whether a MEX param value points at a
// BisQue resource (so it gets a viewable link and can be downloaded) rather than
// being a scalar like a number or threshold.
func bisqueParamIsResourceReference(paramType string, value string) bool {
	if !strings.Contains(value, "/data_service/") {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(paramType)) {
	case "image", "file", "table", "dataset", "resource", "mex":
		return true
	default:
		// Some inputs (e.g. resource_url) are typed by the referenced resource
		// kind; trust the data_service URI shape when the value clearly is one.
		return true
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
	// A bare <resource> envelope with no uniq/type/name is the data_service
	// list wrapper, whether or not it carries children. The previous guard
	// required len(Children) > 0, so an EMPTY result body (`<resource uri=…/>`,
	// zero children) fell through and the wrapper itself was counted as one
	// resource — making every zero-result search report count=1 (and emit a
	// phantom empty resource). Dropping the children requirement makes an empty
	// search correctly report 0 while still skipping a populated wrapper (whose
	// real children are collected by the recursion).
	return strings.TrimSpace(parsed.Type) == "" &&
		strings.TrimSpace(parsed.Name) == ""
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
