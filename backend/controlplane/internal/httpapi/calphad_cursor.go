package httpapi

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"time"
)

const (
	calphadLedgerCursorVersion  = "ultra.calphad.ledger-cursor.v1"
	maxCalphadLedgerCursorBytes = 4096
)

type calphadLedgerCursor struct {
	ResourceID   string
	OwnerUserID  string
	OwnerOrgID   string
	RevisionID   string
	CreatedAt    time.Time
	ValidationID string
}

// ServerDeps has no stable, deployment-wide cursor-signing secret. This
// canonical token therefore carries no authorization: every use is rechecked
// against the authenticated user/org, resource, immutable revision, and an
// exact existing (created_at, validation_id) anchor. Forging it can at most
// select another real page in the caller's own ledger, never disclose another
// tenant or invent an anchor.
type calphadLedgerCursorPayload struct {
	Version      string `json:"version"`
	ResourceID   string `json:"resource_id"`
	OwnerUserID  string `json:"owner_user_id"`
	OwnerOrgID   string `json:"owner_org_id"`
	RevisionID   string `json:"revision_id"`
	CreatedAt    string `json:"created_at"`
	ValidationID string `json:"validation_id"`
}

func validCalphadCursorText(value string, maximum int, allowEmpty bool) bool {
	if value != strings.TrimSpace(value) || len(value) > maximum || (!allowEmpty && value == "") {
		return false
	}
	for _, character := range value {
		if character < 32 || character == 127 {
			return false
		}
	}
	return true
}

func encodeCalphadLedgerCursor(cursor calphadLedgerCursor) (string, error) {
	if !validCalphadCursorText(cursor.ResourceID, 512, false) ||
		!validCalphadCursorText(cursor.OwnerUserID, 512, false) ||
		!validCalphadCursorText(cursor.OwnerOrgID, 512, true) ||
		!validCalphadCursorText(cursor.RevisionID, 512, false) ||
		!validCalphadCursorText(cursor.ValidationID, 512, false) || cursor.CreatedAt.IsZero() {
		return "", errors.New("invalid CALPHAD ledger cursor fields")
	}
	payload := calphadLedgerCursorPayload{
		Version: calphadLedgerCursorVersion, ResourceID: cursor.ResourceID,
		OwnerUserID: cursor.OwnerUserID, OwnerOrgID: cursor.OwnerOrgID,
		RevisionID: cursor.RevisionID, CreatedAt: cursor.CreatedAt.UTC().Format(time.RFC3339Nano),
		ValidationID: cursor.ValidationID,
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(encoded), nil
}

func decodeCalphadLedgerCursor(token string) (calphadLedgerCursor, error) {
	if token == "" || token != strings.TrimSpace(token) || len(token) > maxCalphadLedgerCursorBytes {
		return calphadLedgerCursor{}, errors.New("invalid CALPHAD ledger cursor")
	}
	raw, err := base64.RawURLEncoding.DecodeString(token)
	if err != nil || len(raw) == 0 || base64.RawURLEncoding.EncodeToString(raw) != token {
		return calphadLedgerCursor{}, errors.New("invalid CALPHAD ledger cursor encoding")
	}
	var payload calphadLedgerCursorPayload
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&payload); err != nil {
		return calphadLedgerCursor{}, errors.New("invalid CALPHAD ledger cursor payload")
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return calphadLedgerCursor{}, errors.New("invalid CALPHAD ledger cursor payload")
	}
	canonical, err := json.Marshal(payload)
	if err != nil || !bytes.Equal(raw, canonical) || payload.Version != calphadLedgerCursorVersion {
		return calphadLedgerCursor{}, errors.New("non-canonical CALPHAD ledger cursor")
	}
	createdAt, err := time.Parse(time.RFC3339Nano, payload.CreatedAt)
	if err != nil || createdAt.IsZero() || createdAt.Location() != time.UTC ||
		createdAt.Format(time.RFC3339Nano) != payload.CreatedAt ||
		!validCalphadCursorText(payload.ResourceID, 512, false) ||
		!validCalphadCursorText(payload.OwnerUserID, 512, false) ||
		!validCalphadCursorText(payload.OwnerOrgID, 512, true) ||
		!validCalphadCursorText(payload.RevisionID, 512, false) ||
		!validCalphadCursorText(payload.ValidationID, 512, false) {
		return calphadLedgerCursor{}, errors.New("invalid CALPHAD ledger cursor fields")
	}
	return calphadLedgerCursor{
		ResourceID: payload.ResourceID, OwnerUserID: payload.OwnerUserID,
		OwnerOrgID: payload.OwnerOrgID, RevisionID: payload.RevisionID,
		CreatedAt: createdAt, ValidationID: payload.ValidationID,
	}, nil
}
