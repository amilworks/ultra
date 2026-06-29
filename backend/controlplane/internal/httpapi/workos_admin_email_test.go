package httpapi

import "testing"

func TestNormalizeAdminEmails(t *testing.T) {
	set := normalizeAdminEmails([]string{" Amil@UCSB.edu ", "", "  ", "second@example.org"})
	if len(set) != 2 {
		t.Fatalf("expected 2 normalized emails, got %d (%v)", len(set), set)
	}
	if !set["amil@ucsb.edu"] {
		t.Fatalf("expected lower-cased trimmed key amil@ucsb.edu")
	}
}

func TestIsAdminEmail(t *testing.T) {
	auth := &WorkOSAuth{adminEmails: normalizeAdminEmails([]string{"amil@ucsb.edu"})}
	cases := map[string]bool{
		"amil@ucsb.edu":   true,
		"AMIL@UCSB.EDU":   true,
		" amil@ucsb.edu ": true,
		"someone@else.io": false,
		"":                false,
	}
	for email, want := range cases {
		if got := auth.isAdminEmail(email); got != want {
			t.Fatalf("isAdminEmail(%q) = %v, want %v", email, got, want)
		}
	}

	// No allowlist configured → never an admin via this path.
	empty := &WorkOSAuth{}
	if empty.isAdminEmail("amil@ucsb.edu") {
		t.Fatalf("empty allowlist must not grant admin")
	}
	if (*WorkOSAuth)(nil).isAdminEmail("amil@ucsb.edu") {
		t.Fatalf("nil auth must not grant admin")
	}
}

func TestApplyAdminEmailOverrideWinsOverStoredRole(t *testing.T) {
	deps := ServerDeps{WorkOS: &WorkOSAuth{enabled: true, adminEmails: normalizeAdminEmails([]string{"amil@ucsb.edu"})}}

	// A stored "researcher" role must be overridden to admin for an allowlisted
	// email — this is the path that was previously losing to the DB role.
	resolved := deps.applyAdminEmailOverride(workOSSessionSnapshot{
		Principal: requestPrincipal{Role: "researcher"},
		Email:     "amil@ucsb.edu",
	})
	if resolved.Principal.Role != "admin" {
		t.Fatalf("expected admin, got %q", resolved.Principal.Role)
	}

	// A non-allowlisted email keeps its stored role.
	other := deps.applyAdminEmailOverride(workOSSessionSnapshot{
		Principal: requestPrincipal{Role: "researcher"},
		Email:     "someone@else.io",
	})
	if other.Principal.Role != "researcher" {
		t.Fatalf("expected researcher unchanged, got %q", other.Principal.Role)
	}
}
