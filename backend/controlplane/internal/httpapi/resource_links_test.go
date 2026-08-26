package httpapi

import "testing"

func TestLensURLForResource(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		publicURL  string
		resourceID string
		want       string
	}{
		{name: "relative when no public url", publicURL: "", resourceID: "file_abc", want: "/?view=lens&resource=file_abc"},
		{name: "absolute with public url", publicURL: "https://ultra.example.com", resourceID: "file_abc", want: "https://ultra.example.com/?view=lens&resource=file_abc"},
		{name: "trailing slashes trimmed", publicURL: "https://ultra.example.com///", resourceID: "file_abc", want: "https://ultra.example.com/?view=lens&resource=file_abc"},
		{name: "surrounding whitespace trimmed", publicURL: "  https://ultra.example.com/ ", resourceID: "file_abc", want: "https://ultra.example.com/?view=lens&resource=file_abc"},
		{name: "unsafe id characters escaped", publicURL: "", resourceID: "a b&c=d/e?f#g%h+i", want: "/?view=lens&resource=a+b%26c%3Dd%2Fe%3Ff%23g%25h%2Bi"},
		{name: "empty id omitted", publicURL: "https://ultra.example.com", resourceID: "", want: ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := lensURLForResource(tc.publicURL, tc.resourceID); got != tc.want {
				t.Fatalf("lensURLForResource(%q, %q) = %q, want %q", tc.publicURL, tc.resourceID, got, tc.want)
			}
		})
	}
}
