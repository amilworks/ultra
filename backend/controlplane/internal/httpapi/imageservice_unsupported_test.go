package httpapi

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// imageServiceUndecodable must distinguish a permanent format-level failure (the
// engine recognized the file but cannot decode it -> 415/422) from a transport
// error / sidecar outage. Only the former should surface an "unsupported" viewer;
// the latter must still fall back to the legacy native path.
func TestImageServiceUndecodable(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"415 unsupported media", &imageServiceStatusError{status: http.StatusUnsupportedMediaType, msg: "x"}, true},
		{"422 unprocessable", &imageServiceStatusError{status: http.StatusUnprocessableEntity, msg: "x"}, true},
		{"502 bad gateway", &imageServiceStatusError{status: http.StatusBadGateway, msg: "x"}, false},
		{"500 internal", &imageServiceStatusError{status: http.StatusInternalServerError, msg: "x"}, false},
		{"plain transport error", errors.New("dial tcp 127.0.0.1:8099: connect: connection refused"), false},
		{"nil", nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := imageServiceUndecodable(tc.err); got != tc.want {
				t.Fatalf("imageServiceUndecodable(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

// writeUnsupportedFormatViewer must emit a structured descriptor (HTTP 200) the
// frontend can render as a calm "preview unavailable, download instead" card, not a
// broken 1x1 canvas: kind:"unsupported", decodable:false, the format, a download URL,
// and a human message naming the format.
func TestWriteUnsupportedFormatViewer(t *testing.T) {
	rec := httptest.NewRecorder()
	(ServerDeps{}).writeUnsupportedFormatViewer(rec, resourceRecord{
		FileID:       "abc123",
		OriginalName: "Training_20240812-czQC.lif",
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (structured, not a transport error)", rec.Code)
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode body: %v (%s)", err, rec.Body.String())
	}
	if out["kind"] != "unsupported" {
		t.Fatalf("kind = %v, want unsupported", out["kind"])
	}
	if out["decodable"] != false {
		t.Fatalf("decodable = %v, want false", out["decodable"])
	}
	if out["format"] != "lif" {
		t.Fatalf("format = %v, want lif", out["format"])
	}
	su, ok := out["service_urls"].(map[string]any)
	if !ok {
		t.Fatalf("service_urls missing/wrong type: %v", out["service_urls"])
	}
	dl, _ := su["download"].(string)
	if !strings.Contains(dl, "abc123") || !strings.Contains(dl, "/download") {
		t.Fatalf("download url = %q, want it to reference the resource download endpoint", dl)
	}
	msg, _ := out["message"].(string)
	if !strings.Contains(strings.ToUpper(msg), "LIF") {
		t.Fatalf("message = %q, want it to name the LIF format", msg)
	}
	// axis_sizes must still be present (zeroed) so unconditional consumers don't choke.
	if _, ok := out["axis_sizes"].(map[string]any); !ok {
		t.Fatalf("axis_sizes missing: %v", out["axis_sizes"])
	}
}
