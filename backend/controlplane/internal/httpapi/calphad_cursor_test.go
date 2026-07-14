package httpapi

import (
	"encoding/base64"
	"strings"
	"testing"
	"time"
)

func TestCalphadLedgerCursorIsStrictCanonicalAndRoundTripsExactBinding(t *testing.T) {
	t.Parallel()
	want := calphadLedgerCursor{
		ResourceID: "resource-1", OwnerUserID: "owner-1", OwnerOrgID: "org-1",
		RevisionID: "revision-1", CreatedAt: time.Date(2026, 7, 11, 8, 9, 10, 123456000, time.UTC),
		ValidationID: "validation-1",
	}
	token, err := encodeCalphadLedgerCursor(want)
	if err != nil {
		t.Fatalf("encode cursor: %v", err)
	}
	got, err := decodeCalphadLedgerCursor(token)
	if err != nil || got != want {
		t.Fatalf("decoded cursor=%+v err=%v, want %+v", got, err, want)
	}
	raw, err := base64.RawURLEncoding.DecodeString(token)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	for _, test := range []struct {
		name  string
		token string
	}{
		{name: "empty", token: ""},
		{name: "surrounding whitespace", token: " " + token},
		{name: "padded base64", token: token + "="},
		{name: "noncanonical JSON whitespace", token: base64.RawURLEncoding.EncodeToString(append([]byte(" "), raw...))},
		{name: "unknown field", token: base64.RawURLEncoding.EncodeToString([]byte(`{"version":"ultra.calphad.ledger-cursor.v1","resource_id":"resource-1","owner_user_id":"owner-1","owner_org_id":"org-1","revision_id":"revision-1","created_at":"2026-07-11T08:09:10.123456Z","validation_id":"validation-1","extra":true}`))},
		{name: "duplicate field", token: base64.RawURLEncoding.EncodeToString([]byte(`{"version":"ultra.calphad.ledger-cursor.v1","resource_id":"resource-1","resource_id":"resource-1","owner_user_id":"owner-1","owner_org_id":"org-1","revision_id":"revision-1","created_at":"2026-07-11T08:09:10.123456Z","validation_id":"validation-1"}`))},
		{name: "non UTC timestamp", token: base64.RawURLEncoding.EncodeToString([]byte(`{"version":"ultra.calphad.ledger-cursor.v1","resource_id":"resource-1","owner_user_id":"owner-1","owner_org_id":"org-1","revision_id":"revision-1","created_at":"2026-07-11T08:09:10.123456+00:00","validation_id":"validation-1"}`))},
		{name: "oversized", token: strings.Repeat("a", maxCalphadLedgerCursorBytes+1)},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if _, err := decodeCalphadLedgerCursor(test.token); err == nil {
				t.Fatal("malformed cursor was accepted")
			}
		})
	}
}
