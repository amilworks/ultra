package app

import (
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
)

// The SPA's deep-link parser only treats pathname "/" as a Lens link, so the
// operator-asserted public URL must reach the link builder as a bare
// scheme://host origin — and an unparseable value must yield no origin at all
// (the links then stay relative, which works on every host).
func TestExplicitPublicURLNormalizesToBareOrigin(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		value string
		want  string
	}{
		{name: "bare origin", value: "https://ultra.example.edu", want: "https://ultra.example.edu"},
		{name: "trailing slash", value: "https://ultra.example.edu/", want: "https://ultra.example.edu"},
		{name: "origin with port", value: "http://localhost:8000", want: "http://localhost:8000"},
		{name: "path-bearing", value: "https://ultra.example.edu/app/", want: "https://ultra.example.edu"},
		{name: "query-bearing", value: "https://ultra.example.edu/?view=lens", want: "https://ultra.example.edu"},
		{name: "fragment-bearing", value: "https://ultra.example.edu/#top", want: "https://ultra.example.edu"},
		{name: "surrounding whitespace", value: "  https://ultra.example.edu/  ", want: "https://ultra.example.edu"},
		{name: "schemeless host", value: "ultra.example.edu", want: ""},
		{name: "garbage", value: "not a url", want: ""},
		{name: "empty", value: "", want: ""},
		{name: "whitespace only", value: "   ", want: ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := explicitPublicURL(config.Config{PublicURL: tc.value}); got != tc.want {
				t.Fatalf("explicitPublicURL(%q) = %q, want %q", tc.value, got, tc.want)
			}
		})
	}
}
