package domain

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"strings"
	"time"
)

const (
	NoteAccessSelectionKey                = "note_access"
	NotePrivacyLineageMetadataKey         = "note_privacy_lineage"
	ModelNotesProposalsEnabledMetadataKey = "model_notes_proposals_enabled"
	MaxRunNoteReferences                  = 20
)

type NoteAccessMode string

const (
	NoteAccessModeSelected NoteAccessMode = "selected"
	NoteAccessModeSearch   NoteAccessMode = "search"
)

type NoteReference struct {
	NoteID   string `json:"note_id"`
	Revision int64  `json:"revision"`
}

type NoteAccessScope struct {
	Mode                NoteAccessMode  `json:"mode"`
	Notes               []NoteReference `json:"notes"`
	AllowAppendProposal bool            `json:"allow_append_proposal"`
}

type NoteSearchInput struct {
	UserID string
	Query  string
	Limit  int
}

type NoteSearchHit struct {
	NoteID    string    `json:"note_id"`
	Title     string    `json:"title"`
	Snippet   string    `json:"snippet"`
	Pinned    bool      `json:"pinned"`
	Revision  int64     `json:"revision"`
	UpdatedAt time.Time `json:"updated_at"`
}

type NoteReadGrantRecord struct {
	TokenHash string
	RunID     string
	UserID    string
	NoteID    string
	Revision  int64
	ExpiresAt time.Time
	CreatedAt time.Time
}

const (
	NoteAppendProposalStatusPending   = "pending"
	NoteAppendProposalStatusCommitted = "committed"
	NoteAppendProposalStatusExpired   = "expired"
)

type NoteAppendProposalRecord struct {
	ProposalID          string    `json:"proposal_id"`
	RunID               string    `json:"run_id,omitempty"`
	NoteID              string    `json:"note_id"`
	NoteTitle           string    `json:"note_title"`
	UserID              string    `json:"-"`
	BaseRevision        int64     `json:"base_revision"`
	BodyMarkdown        string    `json:"body_markdown,omitempty"`
	BodySHA256          string    `json:"-"`
	CommittedBodySHA256 string    `json:"-"`
	IdempotencyKey      string    `json:"-"`
	RequestDigest       string    `json:"-"`
	Status              string    `json:"status"`
	OperationID         string    `json:"operation_id,omitempty"`
	ExpiresAt           time.Time `json:"expires_at"`
	CreatedAt           time.Time `json:"created_at"`
	UpdatedAt           time.Time `json:"updated_at"`
}

type CreateNoteAppendProposalInput struct {
	ProposalID       string
	RunID            string
	UserID           string
	NoteID           string
	ExpectedRevision int64
	BodyMarkdown     string
	ReadTokenHash    string
	IdempotencyKey   string
	RequestDigest    string
	Now              time.Time
	ExpiresAt        time.Time
}

type CommitNoteAppendProposalInput struct {
	ProposalID   string
	OperationID  string
	UserID       string
	BodyMarkdown *string
	Now          time.Time
}

type UndoNoteAppendOperationInput struct {
	OperationID string
	UserID      string
	Now         time.Time
}

type NoteAppendOperationRecord struct {
	OperationID         string     `json:"operation_id"`
	ProposalID          string     `json:"proposal_id"`
	RunID               string     `json:"run_id,omitempty"`
	NoteID              string     `json:"note_id"`
	NoteTitle           string     `json:"note_title"`
	UserID              string     `json:"-"`
	BeforeRevision      int64      `json:"before_revision"`
	AfterRevision       int64      `json:"after_revision"`
	UndoRevision        int64      `json:"undo_revision,omitempty"`
	AppendedBytes       int        `json:"appended_bytes"`
	BeforeContentDigest string     `json:"before_content_digest"`
	AfterContentDigest  string     `json:"after_content_digest"`
	CreatedAt           time.Time  `json:"created_at"`
	UndoneAt            *time.Time `json:"undone_at,omitempty"`
}

// ComputeNoteContentDigest hashes an unambiguous length-prefixed title/body
// encoding. PostgreSQL extensions are deliberately not required for this
// provenance value; the revision remains the sole CAS authority.
func ComputeNoteContentDigest(title string, body string) string {
	hash := sha256.New()
	var size [8]byte
	binary.BigEndian.PutUint64(size[:], uint64(len(title)))
	_, _ = hash.Write(size[:])
	_, _ = hash.Write([]byte(title))
	binary.BigEndian.PutUint64(size[:], uint64(len(body)))
	_, _ = hash.Write(size[:])
	_, _ = hash.Write([]byte(body))
	return hex.EncodeToString(hash.Sum(nil))
}

func NoteBodySHA256(body string) string {
	sum := sha256.Sum256([]byte(body))
	return hex.EncodeToString(sum[:])
}

// ParseNoteAccessScope reads only the reserved Notes sub-object. It returns
// present=false for an absent scope and valid=false for every malformed or
// unsupported shape, so callers never silently broaden model access.
func ParseNoteAccessScope(selection JSONMap) (scope NoteAccessScope, present bool, valid bool) {
	value, present := selection[NoteAccessSelectionKey]
	if !present || value == nil {
		return NoteAccessScope{}, false, true
	}
	mapped, ok := noteStringMap(value)
	if !ok {
		return NoteAccessScope{}, true, false
	}
	mode, ok := mapped["mode"].(string)
	if !ok {
		return NoteAccessScope{}, true, false
	}
	scope.Mode = NoteAccessMode(strings.TrimSpace(mode))
	if scope.Mode != NoteAccessModeSelected && scope.Mode != NoteAccessModeSearch {
		return NoteAccessScope{}, true, false
	}
	if raw, exists := mapped["allow_append_proposal"]; exists {
		allowed, ok := raw.(bool)
		if !ok {
			return NoteAccessScope{}, true, false
		}
		scope.AllowAppendProposal = allowed
	}
	values, ok := noteAnySlice(mapped["notes"])
	if mapped["notes"] != nil && !ok {
		return NoteAccessScope{}, true, false
	}
	if len(values) > MaxRunNoteReferences {
		return NoteAccessScope{}, true, false
	}
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		item, ok := noteStringMap(value)
		if !ok {
			return NoteAccessScope{}, true, false
		}
		noteID, ok := item["note_id"].(string)
		noteID = strings.TrimSpace(noteID)
		if !ok || noteID == "" || len(noteID) > 512 {
			return NoteAccessScope{}, true, false
		}
		if _, duplicate := seen[noteID]; duplicate {
			return NoteAccessScope{}, true, false
		}
		seen[noteID] = struct{}{}
		revision := int64(0)
		if raw, exists := item["revision"]; exists && raw != nil {
			parsed, ok := notePositiveInt64(raw)
			if !ok {
				return NoteAccessScope{}, true, false
			}
			revision = parsed
		}
		scope.Notes = append(scope.Notes, NoteReference{NoteID: noteID, Revision: revision})
	}
	if scope.Mode == NoteAccessModeSelected && len(scope.Notes) == 0 {
		return NoteAccessScope{}, true, false
	}
	return scope, true, true
}

func NoteAccessScopeFromRun(run RunRecord) (NoteAccessScope, bool) {
	selection, ok := noteStringMap(run.Metadata["selection_context"])
	if !ok {
		return NoteAccessScope{}, false
	}
	scope, present, valid := ParseNoteAccessScope(JSONMap(selection))
	return scope, present && valid
}

// RunHasNoteAccessSelection detects the legacy direct-access marker. Presence,
// rather than successful scope parsing, is intentional: canonical runs contain
// a valid object, while malformed historical metadata must still fail closed.
func RunHasNoteAccessSelection(run RunRecord) bool {
	selection, ok := noteStringMap(run.Metadata["selection_context"])
	if !ok {
		return false
	}
	_, present := selection[NoteAccessSelectionKey]
	return present
}

// RunHasNotePrivacyLineage identifies every run whose response may contain
// private Note-derived text. New runs carry a server-authored top-level marker;
// the legacy selection-context check keeps already-stored Note-enabled runs
// protected during rolling deploys and after upgrades. Key presence is
// deliberately fail-closed because malformed privacy metadata must never make
// a run eligible for cross-conversation recall.
func RunHasNotePrivacyLineage(run RunRecord) bool {
	if _, present := run.Metadata[NotePrivacyLineageMetadataKey]; present {
		return true
	}
	return RunHasNoteAccessSelection(run)
}

func CanonicalNoteAccessSelection(selection JSONMap, scope NoteAccessScope) JSONMap {
	result := JSONMap{}
	for key, value := range selection {
		if key != NoteAccessSelectionKey {
			result[key] = value
		}
	}
	notes := make([]JSONMap, 0, len(scope.Notes))
	for _, note := range scope.Notes {
		notes = append(notes, JSONMap{"note_id": note.NoteID, "revision": note.Revision})
	}
	result[NoteAccessSelectionKey] = JSONMap{
		"mode":                  string(scope.Mode),
		"notes":                 notes,
		"allow_append_proposal": scope.AllowAppendProposal,
	}
	return result
}

func (scope NoteAccessScope) Contains(noteID string) bool {
	_, ok := scope.Reference(noteID)
	return ok
}

func (scope NoteAccessScope) Reference(noteID string) (NoteReference, bool) {
	for _, note := range scope.Notes {
		if note.NoteID == noteID {
			return note, true
		}
	}
	return NoteReference{}, false
}

func NoteAccessScopesEqual(a NoteAccessScope, b NoteAccessScope) bool {
	if a.Mode != b.Mode || a.AllowAppendProposal != b.AllowAppendProposal || len(a.Notes) != len(b.Notes) {
		return false
	}
	for index := range a.Notes {
		if a.Notes[index] != b.Notes[index] {
			return false
		}
	}
	return true
}

// NoteAccessRequestMatchesStoredScope compares a raw create-run capability
// with the canonical scope sealed into an existing run. An omitted revision is
// intentionally a wildcard for idempotent replay: the first request resolved
// it to an immutable stored revision, and a retry must not consult mutable Note
// state. An explicitly supplied revision remains part of the request identity.
func NoteAccessRequestMatchesStoredScope(requested NoteAccessScope, stored NoteAccessScope) bool {
	if requested.Mode != stored.Mode ||
		requested.AllowAppendProposal != stored.AllowAppendProposal ||
		len(requested.Notes) != len(stored.Notes) {
		return false
	}
	for index := range requested.Notes {
		if requested.Notes[index].NoteID != stored.Notes[index].NoteID {
			return false
		}
		if requested.Notes[index].Revision > 0 &&
			requested.Notes[index].Revision != stored.Notes[index].Revision {
			return false
		}
	}
	return true
}

func noteStringMap(value any) (map[string]any, bool) {
	switch typed := value.(type) {
	case JSONMap:
		return map[string]any(typed), true
	case map[string]any:
		return typed, true
	default:
		return nil, false
	}
}

func noteAnySlice(value any) ([]any, bool) {
	if value == nil {
		return nil, true
	}
	switch typed := value.(type) {
	case []any:
		return typed, true
	case []JSONMap:
		values := make([]any, len(typed))
		for index := range typed {
			values[index] = typed[index]
		}
		return values, true
	default:
		return nil, false
	}
}

func notePositiveInt64(value any) (int64, bool) {
	switch typed := value.(type) {
	case int:
		if typed > 0 {
			return int64(typed), true
		}
	case int64:
		if typed > 0 {
			return typed, true
		}
	case float64:
		if typed > 0 && typed <= math.MaxInt64 && typed == math.Trunc(typed) {
			return int64(typed), true
		}
	}
	return 0, false
}
