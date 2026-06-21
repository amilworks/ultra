package httpapi

import (
	"os"
	"path/filepath"
	"strings"
)

// Special-format registry (Go side). Mirrors backend/shared/special_formats.json and
// the Python registry (imaging/special_formats.py); special_formats_parity_test.go
// asserts this matches the shared JSON. These are formats that are directory/archive
// shaped or need a non-default serving backend.
//
// TEMPLATE — to add a special format: append a row here, mirror it in the shared JSON +
// the Python registry, and wire its serve backend. Detection is by extension (incl.
// series-suffixed names like "scan.zarr_3") OR, for directory formats, by marker files.
type specialFormat struct {
	ID           string
	Extensions   []string // lowercase, with leading dot
	Shape        string   // "file" | "directory" | "archive"
	DirMarkers   []string // for directory formats: files that must be present at the root
	Serve        string   // "ngff" | "libbioimage" | "transcode"
	ResourceKind string   // "image" | ...
}

var specialFormats = []specialFormat{
	{
		ID:           "ome-zarr",
		Extensions:   []string{".ome.zarr", ".zarr"},
		Shape:        "directory",
		DirMarkers:   []string{".zattrs", ".zgroup"},
		Serve:        "ngff",
		ResourceKind: "image",
	},
}

// detectSpecialFormatByName matches a resource's name against the registry's extensions,
// tolerating a series/page suffix (e.g. "img.zarr_3").
func detectSpecialFormatByName(name string) *specialFormat {
	lower := strings.ToLower(strings.TrimSpace(name))
	for i := range specialFormats {
		for _, ext := range specialFormats[i].Extensions {
			if strings.HasSuffix(lower, ext) || strings.Contains(lower, ext+"_") {
				return &specialFormats[i]
			}
		}
	}
	return nil
}

// detectSpecialFormatByDir matches a directory against directory-format marker files
// (e.g. an OME-Zarr group has .zattrs + .zgroup at its root).
func detectSpecialFormatByDir(path string) *specialFormat {
	info, err := os.Stat(path)
	if err != nil || !info.IsDir() {
		return nil
	}
	for i := range specialFormats {
		sf := &specialFormats[i]
		if sf.Shape != "directory" || len(sf.DirMarkers) == 0 {
			continue
		}
		all := true
		for _, marker := range sf.DirMarkers {
			if _, err := os.Stat(filepath.Join(path, marker)); err != nil {
				all = false
				break
			}
		}
		if all {
			return sf
		}
	}
	return nil
}

// specialFormatForResource resolves the special format of a resource by name first, then
// (for unresolved names) by probing the on-disk directory markers.
func specialFormatForResource(record resourceRecord, path string) *specialFormat {
	if sf := detectSpecialFormatByName(record.OriginalName); sf != nil {
		return sf
	}
	return detectSpecialFormatByDir(path)
}

// servesViaNgff reports whether a resource should be served by the ngff-service.
func (deps ServerDeps) servesViaNgff(record resourceRecord, path string) bool {
	if strings.TrimSpace(deps.NgffServiceURL) == "" {
		return false
	}
	sf := specialFormatForResource(record, path)
	return sf != nil && sf.Serve == "ngff"
}

// ngffDeps returns a copy of deps with ImageServiceURL pointed at the ngff-service, so
// the existing proxy helpers (cache, backpressure, fallback) route to it unchanged.
func (deps ServerDeps) ngffDeps() ServerDeps {
	d := deps
	d.ImageServiceURL = deps.NgffServiceURL
	return d
}
