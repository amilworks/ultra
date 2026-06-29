package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

const textViewerUser, textViewerOrg = "u-text", "o-text"

func seedTextResource(t *testing.T, mem *store.MemoryStore, root, id, name, contentType, kind string, data []byte) {
	t.Helper()
	// Stage the bytes in a subdirectory and record StoragePath: the catalog
	// migration only re-catalogs top-level upload-root files, so a subdir keeps it
	// from clobbering our seeded ownership.
	storageRel := filepath.Join("staged", id+"__"+safeOriginalFilename(name))
	abs := filepath.Join(root, storageRel)
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(abs, data, 0o644); err != nil {
		t.Fatalf("write resource file: %v", err)
	}
	if _, err := mem.UpsertResource(context.Background(), domain.UpsertResourceInput{
		ResourceID:   id,
		OriginalName: name,
		ContentType:  contentType,
		SizeBytes:    int64(len(data)),
		StoragePath:  storageRel,
		SourceType:   "upload",
		ResourceKind: kind,
		OwnerUserID:  textViewerUser,
		OwnerOrgID:   textViewerOrg,
		Status:       "active",
	}); err != nil {
		t.Fatalf("seed resource %s: %v", id, err)
	}
}

func TestResourceTextHeadBoundedWindow(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	root := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: root})

	var sb strings.Builder
	for i := 0; i < 400; i++ {
		sb.WriteString(fmt.Sprintf("line-%04d the quick brown fox jumps over the lazy dog\n", i))
	}
	body := []byte(sb.String())
	seedTextResource(t, mem, root, "txt-1", "notes.txt", "text/plain", "document", body)

	rec := analysisAuthedGet(t, router, "/v2/resources/txt-1/text-head?max_bytes=1024", textViewerUser, textViewerOrg)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", rec.Code, rec.Body.String())
	}
	var resp resourceTextHeadResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !resp.Truncated {
		t.Fatalf("expected truncated=true for a 1024-byte window over %d bytes", len(body))
	}
	if resp.TotalSizeBytes != int64(len(body)) {
		t.Fatalf("total = %d, want %d", resp.TotalSizeBytes, len(body))
	}
	if resp.Format != "text" {
		t.Fatalf("format = %q, want text", resp.Format)
	}
	// The window must end on a newline (no partial last line).
	if !strings.HasSuffix(resp.Text, "\n") {
		t.Fatalf("text should end on a line boundary, got tail %q", tail(resp.Text, 24))
	}
	if int64(len(resp.Text)) != resp.ReturnedBytes {
		t.Fatalf("returned_bytes = %d, len(text) = %d", resp.ReturnedBytes, len(resp.Text))
	}
	if resp.NextOffset != resp.ReturnedBytes {
		t.Fatalf("next_offset = %d, want returned_bytes %d (offset 0)", resp.NextOffset, resp.ReturnedBytes)
	}
	if resp.ApproxTotalLines < int64(resp.LineCount) {
		t.Fatalf("approx_total_lines %d should be >= window line_count %d", resp.ApproxTotalLines, resp.LineCount)
	}
	// Loading from next_offset should advance.
	rec2 := analysisAuthedGet(t, router, fmt.Sprintf("/v2/resources/txt-1/text-head?max_bytes=1024&offset=%d", resp.NextOffset), textViewerUser, textViewerOrg)
	var resp2 resourceTextHeadResponse
	if err := json.Unmarshal(rec2.Body.Bytes(), &resp2); err != nil {
		t.Fatalf("decode page2: %v", err)
	}
	if resp2.Offset != resp.NextOffset {
		t.Fatalf("page2 offset = %d, want %d", resp2.Offset, resp.NextOffset)
	}
	if strings.HasPrefix(resp2.Text, "line-0000") {
		t.Fatalf("page2 should not restart at the beginning")
	}
}

func TestResourceTextHeadFullSmallFile(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	root := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: root})

	body := []byte("{\n  \"a\": 1,\n  \"b\": true\n}\n")
	seedTextResource(t, mem, root, "json-1", "config.json", "application/json", "document", body)

	rec := analysisAuthedGet(t, router, "/v2/resources/json-1/text-head", textViewerUser, textViewerOrg)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", rec.Code, rec.Body.String())
	}
	var resp resourceTextHeadResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Truncated {
		t.Fatalf("small file should not be truncated")
	}
	if resp.Text != string(body) {
		t.Fatalf("text = %q, want full body", resp.Text)
	}
	if resp.Format != "json" {
		t.Fatalf("format = %q, want json", resp.Format)
	}
	if resp.LineCount != 4 {
		t.Fatalf("line_count = %d, want 4", resp.LineCount)
	}
}

func TestResourceCsvRowsCursorPagination(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	root := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: root})

	var sb strings.Builder
	sb.WriteString("id,name,score\n")
	for i := 0; i < 10; i++ {
		sb.WriteString(fmt.Sprintf("%d,row-%d,%d\n", i, i, i*7))
	}
	seedTextResource(t, mem, root, "csv-1", "data.csv", "text/csv", "table", []byte(sb.String()))

	rec := analysisAuthedGet(t, router, "/v2/resources/csv-1/csv/rows?limit=4", textViewerUser, textViewerOrg)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", rec.Code, rec.Body.String())
	}
	var page1 resourceCsvRowsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &page1); err != nil {
		t.Fatalf("decode page1: %v", err)
	}
	if len(page1.Columns) != 3 || page1.Columns[0] != "id" {
		t.Fatalf("columns = %v, want [id name score]", page1.Columns)
	}
	if page1.ReturnedRows != 4 {
		t.Fatalf("page1 rows = %d, want 4", page1.ReturnedRows)
	}
	if !page1.HasMore {
		t.Fatalf("page1 should have more")
	}
	if page1.Delimiter != "," {
		t.Fatalf("delimiter = %q, want ,", page1.Delimiter)
	}
	if page1.Rows[0][1] != "row-0" {
		t.Fatalf("first data row = %v", page1.Rows[0])
	}

	// Cursor to the next page.
	rec2 := analysisAuthedGet(t, router, fmt.Sprintf("/v2/resources/csv-1/csv/rows?limit=4&offset_bytes=%d", page1.NextOffsetBytes), textViewerUser, textViewerOrg)
	var page2 resourceCsvRowsResponse
	if err := json.Unmarshal(rec2.Body.Bytes(), &page2); err != nil {
		t.Fatalf("decode page2: %v", err)
	}
	if len(page2.Columns) != 0 {
		t.Fatalf("page2 must not repeat the header, got %v", page2.Columns)
	}
	if page2.Rows[0][1] != "row-4" {
		t.Fatalf("page2 first row = %v, want row-4", page2.Rows[0])
	}
}

func TestResourceCsvRowsQuoteAwareNewlines(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	root := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: root})

	body := "id,note\n1,\"line one\nline two\"\n2,plain\n"
	seedTextResource(t, mem, root, "csv-q", "notes.csv", "text/csv", "table", []byte(body))

	rec := analysisAuthedGet(t, router, "/v2/resources/csv-q/csv/rows?limit=10", textViewerUser, textViewerOrg)
	var page resourceCsvRowsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &page); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if page.ReturnedRows != 2 {
		t.Fatalf("rows = %d, want 2 (the embedded newline must not split a record)", page.ReturnedRows)
	}
	if page.Rows[0][1] != "line one\nline two" {
		t.Fatalf("quoted cell = %q, want embedded newline preserved", page.Rows[0][1])
	}
}

func TestResourceCsvRowsSemicolonDelimiterAcrossPages(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	root := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: root})

	var sb strings.Builder
	sb.WriteString("id;name;score\n")
	for i := 0; i < 8; i++ {
		sb.WriteString(fmt.Sprintf("%d;row-%d;%d\n", i, i, i))
	}
	seedTextResource(t, mem, root, "semi-1", "data.csv", "text/csv", "table", []byte(sb.String()))

	// Page 1 sniffs ';'. Page 2 omits the delimiter param entirely — the server must
	// still sniff ';' from the continuation window, not fall back to comma.
	rec1 := analysisAuthedGet(t, router, "/v2/resources/semi-1/csv/rows?limit=3", textViewerUser, textViewerOrg)
	var page1 resourceCsvRowsResponse
	if err := json.Unmarshal(rec1.Body.Bytes(), &page1); err != nil {
		t.Fatalf("decode page1: %v", err)
	}
	if page1.Delimiter != ";" {
		t.Fatalf("page1 delimiter = %q, want ;", page1.Delimiter)
	}
	rec2 := analysisAuthedGet(t, router, fmt.Sprintf("/v2/resources/semi-1/csv/rows?limit=3&offset_bytes=%d", page1.NextOffsetBytes), textViewerUser, textViewerOrg)
	var page2 resourceCsvRowsResponse
	if err := json.Unmarshal(rec2.Body.Bytes(), &page2); err != nil {
		t.Fatalf("decode page2: %v", err)
	}
	if page2.Delimiter != ";" {
		t.Fatalf("page2 delimiter = %q, want ; (continuation must re-sniff)", page2.Delimiter)
	}
	if len(page2.Rows) == 0 || len(page2.Rows[0]) != 3 {
		t.Fatalf("page2 rows must split into 3 columns, got %v", page2.Rows)
	}
}

func TestResourceCsvRowsLastRecordReachableWithoutTrailingNewline(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	root := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: root})

	// No trailing newline: the final record is complete and must never be dropped
	// or truncated across cursor pages.
	body := "id,name\n1,alpha\n2,beta\n3,gamma"
	seedTextResource(t, mem, root, "tail-1", "data.csv", "text/csv", "table", []byte(body))

	collected := [][]string{}
	offset := int64(0)
	for guard := 0; guard < 10; guard++ {
		rec := analysisAuthedGet(t, router, fmt.Sprintf("/v2/resources/tail-1/csv/rows?limit=2&offset_bytes=%d", offset), textViewerUser, textViewerOrg)
		var page resourceCsvRowsResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &page); err != nil {
			t.Fatalf("decode: %v", err)
		}
		collected = append(collected, page.Rows...)
		if !page.HasMore {
			break
		}
		if page.NextOffsetBytes <= offset {
			t.Fatalf("cursor did not advance (offset %d -> %d)", offset, page.NextOffsetBytes)
		}
		offset = page.NextOffsetBytes
	}
	want := [][]string{{"1", "alpha"}, {"2", "beta"}, {"3", "gamma"}}
	if fmt.Sprint(collected) != fmt.Sprint(want) {
		t.Fatalf("paginated rows = %v, want %v (last record must be reachable + intact)", collected, want)
	}
}

func TestResourceCsvRowsTabDelimiterFromExtension(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	root := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: root})

	body := "id\tname\n1\talpha\n2\tbeta\n"
	seedTextResource(t, mem, root, "tsv-1", "data.tsv", "text/tab-separated-values", "table", []byte(body))

	rec := analysisAuthedGet(t, router, "/v2/resources/tsv-1/csv/rows", textViewerUser, textViewerOrg)
	var page resourceCsvRowsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &page); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if page.Delimiter != "\t" {
		t.Fatalf("delimiter = %q, want tab", page.Delimiter)
	}
	if len(page.Columns) != 2 || page.Columns[1] != "name" {
		t.Fatalf("columns = %v", page.Columns)
	}
}

func TestDownloadResourceServesByteRange(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	root := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: root})

	seedTextResource(t, mem, root, "rng-1", "data.csv", "text/csv", "table", []byte("0123456789ABCDEF"))

	req := httptest.NewRequest(http.MethodGet, "/v2/resources/rng-1/download", nil)
	req.Header.Set("X-Ultra-User-Id", textViewerUser)
	req.Header.Set("X-Ultra-Org-Id", textViewerOrg)
	req.Header.Set("Range", "bytes=0-3")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusPartialContent {
		t.Fatalf("status = %d, want 206; body=%s", rec.Code, rec.Body.String())
	}
	if rec.Body.String() != "0123" {
		t.Fatalf("range body = %q, want 0123", rec.Body.String())
	}
	if cr := rec.Header().Get("Content-Range"); !strings.HasPrefix(cr, "bytes 0-3/16") {
		t.Fatalf("Content-Range = %q, want bytes 0-3/16", cr)
	}
}

func TestResourceClassificationRepairOnRead(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	root := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: root})

	// Mimic a chunked/bundle upload that persisted octet-stream + "file" kind.
	seedTextResource(t, mem, root, "oct-1", "big.csv", "application/octet-stream", "file", []byte("a,b\n1,2\n"))
	seedTextResource(t, mem, root, "oct-2", "manifest.json", "application/octet-stream", "file", []byte("{}\n"))

	for _, tc := range []struct {
		id       string
		wantType string
		wantKind string
	}{
		{"oct-1", "text/csv", "table"},
		{"oct-2", "application/json", "document"},
	} {
		rec := analysisAuthedGet(t, router, "/v2/resources/"+tc.id, textViewerUser, textViewerOrg)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s status = %d, body=%s", tc.id, rec.Code, rec.Body.String())
		}
		var resp struct {
			Resource struct {
				ContentType  string `json:"content_type"`
				ResourceKind string `json:"resource_kind"`
			} `json:"resource"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode %s: %v", tc.id, err)
		}
		if !strings.HasPrefix(resp.Resource.ContentType, tc.wantType) {
			t.Fatalf("%s content_type = %q, want prefix %q", tc.id, resp.Resource.ContentType, tc.wantType)
		}
		if resp.Resource.ResourceKind != tc.wantKind {
			t.Fatalf("%s resource_kind = %q, want %q", tc.id, resp.Resource.ResourceKind, tc.wantKind)
		}
	}
}

func TestResourceTextHeadCrossUserDenied(t *testing.T) {
	t.Parallel()
	mem := store.NewMemoryStore()
	root := t.TempDir()
	router := NewRouter(ServerDeps{Version: "test", Store: mem, UploadRoot: root})

	seedTextResource(t, mem, root, "priv-1", "secret.csv", "text/csv", "table", []byte("a,b\n1,2\n"))

	rec := analysisAuthedGet(t, router, "/v2/resources/priv-1/text-head", "intruder", "other-org")
	if rec.Code == http.StatusOK {
		t.Fatalf("cross-user text-head should not return 200; got body=%s", rec.Body.String())
	}
}

func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}
