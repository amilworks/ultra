package httpapi

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newBisqueCountRouter(t *testing.T, handler http.HandlerFunc) (*httptest.Server, http.Handler) {
	t.Helper()
	server := httptest.NewServer(handler)
	router := NewRouter(ServerDeps{
		Version: "test-version",
		Bisque: NewBisqueService(BisqueServiceConfig{
			RootURL:       server.URL,
			AllowedRoots:  []string{server.URL},
			HTTPClient:    server.Client(),
			UploadRoot:    t.TempDir(),
			MaxImportSize: 8 << 20,
		}),
	})
	return server, router
}

func postJSON(t *testing.T, router http.Handler, path, body string) map[string]any {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("POST %s status = %d body=%s", path, rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode %s response: %v (%s)", path, err, rec.Body.String())
	}
	return out
}

// A zero-result data_service body is a bare <resource uri=…/> envelope with no
// children. It must count as 0, not 1 — the bug that made resource_type=annotation
// report a phantom "1" in production.
func TestV2BisqueSearchEmptyResultReportsZero(t *testing.T) {
	t.Parallel()
	server, router := newBisqueCountRouter(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		_, _ = w.Write([]byte(`<resource uri="` + r.URL.String() + `"/>`))
	})
	defer server.Close()

	body := postJSON(t, router, "/v2/bisque/search", `{"resource_type":"annotation","limit":5}`)
	if body["count"].(float64) != 0 {
		t.Fatalf("count = %#v, want 0 (empty result must not count the list wrapper)", body["count"])
	}
	if results, ok := body["results"].([]any); ok && len(results) != 0 {
		t.Fatalf("results = %d, want 0 (no phantom resource)", len(results))
	}
}

// count_all must prefer BisQue's authoritative view=count total and NOT page.
func TestV2BisqueSearchCountAllUsesViewCount(t *testing.T) {
	t.Parallel()
	var sawViewCount bool
	var pagedOffsets []string
	var server *httptest.Server
	server, router := newBisqueCountRouter(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		if r.URL.Query().Get("view") == "count" {
			sawViewCount = true
			_, _ = w.Write([]byte(`<resource><tag name="count" value="1234" type="number"/></resource>`))
			return
		}
		if off := r.URL.Query().Get("offset"); off != "" {
			pagedOffsets = append(pagedOffsets, off)
		}
		_, _ = w.Write([]byte(`<resource><image uri="` + server.URL + `/data_service/image/a" resource_uniq="a" name="a.jpg"/></resource>`))
	})
	defer server.Close()

	body := postJSON(t, router, "/v2/bisque/search", `{"resource_type":"image","limit":1,"count_all":true}`)
	if !sawViewCount {
		t.Fatalf("count_all did not use view=count")
	}
	if body["count"].(float64) != 1234 {
		t.Fatalf("count = %#v, want 1234 from view=count", body["count"])
	}
	if len(pagedOffsets) != 0 {
		t.Fatalf("count_all paged offsets %v, want none (view=count answered)", pagedOffsets)
	}
}

// DatasetMembers reads the /value sub-collection: view=count for the count,
// view=short (paged) for the resolved members with names.
func TestV2BisqueDatasetMembers(t *testing.T) {
	t.Parallel()
	var server *httptest.Server
	server, router := newBisqueCountRouter(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/data_service/dataset/00-ds/value" {
			t.Fatalf("path = %q, want /data_service/dataset/00-ds/value", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/xml")
		if r.URL.Query().Get("view") == "count" {
			_, _ = w.Write([]byte(`<resource><tag name="count" value="5" type="number"/></resource>`))
			return
		}
		if r.URL.Query().Get("view") != "short" {
			t.Fatalf("member view = %q, want short", r.URL.Query().Get("view"))
		}
		// Simulated page for limit=2 offset=1.
		_, _ = w.Write([]byte(`<resource>` +
			`<image uri="` + server.URL + `/data_service/00-img1" resource_uniq="00-img1" name="b.jpg"/>` +
			`<image uri="` + server.URL + `/data_service/00-img2" resource_uniq="00-img2" name="c.jpg"/>` +
			`</resource>`))
	})
	defer server.Close()

	body := postJSON(t, router, "/v2/bisque/dataset-members", `{"dataset_uniq":"00-ds","limit":2,"offset":1}`)
	if body["member_count"].(float64) != 5 {
		t.Fatalf("member_count = %#v, want 5 (from view=count)", body["member_count"])
	}
	members := body["members"].([]any)
	if len(members) != 2 {
		t.Fatalf("members len = %d, want 2", len(members))
	}
	first := members[0].(map[string]any)
	if first["resource_uniq"] != "00-img1" || first["name"] != "b.jpg" {
		t.Fatalf("first member = %#v, want 00-img1/b.jpg", first)
	}
}

// Annotation counts must walk the FULL gobject tree from /gobject?view=deep and
// count the primitive shapes (rectangles), grouped by class label — not the
// group containers, not the image's truncated view=deep, not /gobject?view=count.
func TestV2BisqueImageAnnotationCountWalksGobjectTree(t *testing.T) {
	t.Parallel()
	var sawImageViewDeep, sawGobjectViewCount bool
	server, router := newBisqueCountRouter(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		switch {
		case strings.HasSuffix(r.URL.Path, "/gobject"):
			if r.URL.Query().Get("view") == "count" {
				sawGobjectViewCount = true
				_, _ = w.Write([]byte(`<resource><tag name="count" value="0" type="number"/></resource>`))
				return
			}
			if r.URL.Query().Get("view") != "deep" {
				t.Fatalf("gobject view = %q, want deep", r.URL.Query().Get("view"))
			}
			// gt2 -> {burrow: 2 rectangles, prairie_dog: 1 rectangle}
			_, _ = w.Write([]byte(`<resource uri="/data_service/00-ax/gobject?view=deep">` +
				`<gobject name="gt2">` +
				`<gobject name="burrow">` +
				`<rectangle><vertex/><vertex/></rectangle>` +
				`<rectangle><vertex/><vertex/></rectangle>` +
				`</gobject>` +
				`<gobject name="prairie_dog">` +
				`<rectangle><vertex/><vertex/></rectangle>` +
				`</gobject>` +
				`</gobject>` +
				`</resource>`))
		case r.URL.Path == "/data_service/image/00-ax":
			if r.URL.Query().Get("view") == "deep" {
				sawImageViewDeep = true
			}
			_, _ = w.Write([]byte(`<image uri="x" name="AX.JPG" resource_uniq="00-ax"/>`))
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	})
	defer server.Close()

	body := postJSON(t, router, "/v2/bisque/image-annotations", `{"image_uniq":"00-ax"}`)
	if sawImageViewDeep {
		t.Fatalf("used the image's truncated view=deep; must use the /gobject sub-collection")
	}
	if sawGobjectViewCount {
		t.Fatalf("used /gobject?view=count (ACL-undercounts); must use view=deep")
	}
	if body["annotation_count"].(float64) != 3 {
		t.Fatalf("annotation_count = %#v, want 3 rectangles (not the gt2/burrow/prairie_dog groups)", body["annotation_count"])
	}
	if body["group_count"].(float64) != 3 {
		t.Fatalf("group_count = %#v, want 3 (gt2, burrow, prairie_dog)", body["group_count"])
	}
	labels := body["label_counts"].(map[string]any)
	if labels["burrow"].(float64) != 2 || labels["prairie_dog"].(float64) != 1 {
		t.Fatalf("label_counts = %#v, want burrow:2 prairie_dog:1", labels)
	}
	if body["name"] != "AX.JPG" {
		t.Fatalf("name = %#v, want AX.JPG", body["name"])
	}
}

// End-to-end: how many images in a dataset have annotations, with per-class totals.
func TestV2BisqueDatasetAnnotationsCountsShapes(t *testing.T) {
	t.Parallel()
	var server *httptest.Server
	server, router := newBisqueCountRouter(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/xml")
		switch {
		case strings.HasPrefix(r.URL.Path, "/data_service/dataset/"):
			if r.URL.Query().Get("view") == "count" {
				_, _ = w.Write([]byte(`<resource><tag name="count" value="4" type="number"/></resource>`))
				return
			}
			var b strings.Builder
			b.WriteString(`<resource>`)
			for i := 0; i < 4; i++ {
				b.WriteString(fmt.Sprintf(`<image uri="%s/data_service/00-img%d" resource_uniq="00-img%d" name="img%d.jpg"/>`, server.URL, i, i, i))
			}
			b.WriteString(`</resource>`)
			_, _ = w.Write([]byte(b.String()))
		case strings.HasSuffix(r.URL.Path, "/gobject"):
			base := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/data_service/"), "/gobject")
			switch base {
			case "00-img0": // burrow: 2
				_, _ = w.Write([]byte(`<resource><gobject name="gt2"><gobject name="burrow">` +
					`<rectangle><vertex/></rectangle><rectangle><vertex/></rectangle>` +
					`</gobject></gobject></resource>`))
			case "00-img2": // burrow: 1, prairie_dog: 1
				_, _ = w.Write([]byte(`<resource><gobject name="gt2">` +
					`<gobject name="burrow"><rectangle><vertex/></rectangle></gobject>` +
					`<gobject name="prairie_dog"><rectangle><vertex/></rectangle></gobject>` +
					`</gobject></resource>`))
			default: // no annotations
				_, _ = w.Write([]byte(`<resource uri="` + r.URL.String() + `"/>`))
			}
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	})
	defer server.Close()

	body := postJSON(t, router, "/v2/bisque/dataset-annotations", `{"dataset_uniq":"00-ds"}`)
	if body["member_count"].(float64) != 4 {
		t.Fatalf("member_count = %#v, want 4", body["member_count"])
	}
	if body["images_checked"].(float64) != 4 {
		t.Fatalf("images_checked = %#v, want 4", body["images_checked"])
	}
	if body["images_with_annotations"].(float64) != 2 {
		t.Fatalf("images_with_annotations = %#v, want 2", body["images_with_annotations"])
	}
	if body["total_annotations"].(float64) != 4 {
		t.Fatalf("total_annotations = %#v, want 4 (2 + 1 + 1 rectangles)", body["total_annotations"])
	}
	totals := body["label_totals"].(map[string]any)
	if totals["burrow"].(float64) != 3 || totals["prairie_dog"].(float64) != 1 {
		t.Fatalf("label_totals = %#v, want burrow:3 prairie_dog:1", totals)
	}
	if len(body["annotated_images"].([]any)) != 2 {
		t.Fatalf("annotated_images = %d, want 2", len(body["annotated_images"].([]any)))
	}
}
