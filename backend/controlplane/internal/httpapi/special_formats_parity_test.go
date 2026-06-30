package httpapi

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The Go special-format registry must match backend/shared/special_formats.json (the
// canonical spec the Python registry also mirrors). Adding a format = edit the JSON +
// both registries; this test fails on drift.
func TestSpecialFormatsParityWithSharedJSON(t *testing.T) {
	path := filepath.Join("..", "..", "..", "shared", "special_formats.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read shared spec: %v", err)
	}
	var spec struct {
		Formats []struct {
			ID           string   `json:"id"`
			Extensions   []string `json:"extensions"`
			Shape        string   `json:"shape"`
			DirMarkers   []string `json:"dir_markers"`
			Serve        string   `json:"serve"`
			ResourceKind string   `json:"resource_kind"`
		} `json:"formats"`
	}
	if err := json.Unmarshal(raw, &spec); err != nil {
		t.Fatalf("parse shared spec: %v", err)
	}
	if len(spec.Formats) != len(specialFormats) {
		t.Fatalf("format count drift: json=%d go=%d", len(spec.Formats), len(specialFormats))
	}
	byID := map[string]specialFormat{}
	for _, sf := range specialFormats {
		byID[sf.ID] = sf
	}
	for _, f := range spec.Formats {
		sf, ok := byID[f.ID]
		if !ok {
			t.Fatalf("json format %q missing from Go registry", f.ID)
		}
		if got, want := join(sf.Extensions), join(f.Extensions); got != want {
			t.Errorf("%s extensions: go=%q json=%q", f.ID, got, want)
		}
		if sf.Shape != f.Shape || sf.Serve != f.Serve || sf.ResourceKind != f.ResourceKind {
			t.Errorf("%s shape/serve/kind drift: go=%v json=%v", f.ID, sf, f)
		}
		if join(sf.DirMarkers) != join(f.DirMarkers) {
			t.Errorf("%s dir_markers drift", f.ID)
		}
	}
}

func TestDetectSpecialFormatByName(t *testing.T) {
	for _, name := range []string{"scan.ome.zarr", "scan.zarr", "scan.zarr_3"} {
		if sf := detectSpecialFormatByName(name); sf == nil || sf.ID != "ome-zarr" {
			t.Errorf("%q: expected ome-zarr, got %v", name, sf)
		}
	}
	if sf := detectSpecialFormatByName("scan.tif"); sf != nil {
		t.Errorf("scan.tif: expected nil, got %v", sf)
	}
}

func TestNgffServiceUnavailableGuard(t *testing.T) {
	zarr := resourceRecord{OriginalName: "scan.ome.zarr"}
	plain := resourceRecord{OriginalName: "scan.tif"}

	// Configured: OME-Zarr serves via the ngff-service, and the unavailable guard is OFF.
	configured := ServerDeps{NgffServiceURL: "http://ngff.test"}
	if !configured.servesViaNgff(zarr, "scan.ome.zarr") {
		t.Fatal("ome-zarr should serve via ngff when configured")
	}
	if configured.ngffServiceUnavailable(zarr, "scan.ome.zarr") {
		t.Fatal("guard must be off when ngff is configured")
	}

	// Unconfigured: it can NOT serve via ngff; the guard fires for OME-Zarr so it is never handed
	// to libbioimage (which only returns a confusing empty (0,0,0) region for a directory store),
	// but stays off for a plain image libbioimage can decode.
	unconfigured := ServerDeps{NgffServiceURL: ""}
	if unconfigured.servesViaNgff(zarr, "scan.ome.zarr") {
		t.Fatal("ome-zarr cannot serve via ngff when unconfigured")
	}
	if !unconfigured.ngffServiceUnavailable(zarr, "scan.ome.zarr") {
		t.Fatal("guard must fire for an ome-zarr when ngff is unconfigured")
	}
	if unconfigured.ngffServiceUnavailable(plain, "scan.tif") {
		t.Fatal("guard must NOT fire for a plain libbioimage-decodable image")
	}
}

func join(xs []string) string {
	out := ""
	for i, x := range xs {
		if i > 0 {
			out += ","
		}
		out += x
	}
	return out
}
