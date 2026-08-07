package domain

import "testing"

func TestIsCanonicalResourceID(t *testing.T) {
	t.Parallel()

	for _, valid := range []string{"file_123", "file-123", "bisque:123", "image.ome"} {
		if !IsCanonicalResourceID(valid) {
			t.Fatalf("IsCanonicalResourceID(%q) = false, want true", valid)
		}
	}
	for _, invalid := range []string{"", ".", "..", " file", "file ", "a/b", `a\\b`, "a\x00b"} {
		if IsCanonicalResourceID(invalid) {
			t.Fatalf("IsCanonicalResourceID(%q) = true, want false", invalid)
		}
	}
}
