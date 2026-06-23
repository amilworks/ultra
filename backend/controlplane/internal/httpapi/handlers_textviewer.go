package httpapi

import (
	"bytes"
	"encoding/csv"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/go-chi/chi/v5"
)

// This file backs the text/data resource viewer (CSV/JSON/YAML/XML/Markdown/text).
// Both endpoints read only a BOUNDED window of the file with O(1) memory regardless
// of file size, so a 3GB CSV costs the same as a 5KB JSON for first paint. They
// reuse the exact auth + path resolution of handleDownloadResource (per-user/org
// scoping via GetResourceForUser, traversal-guarded resolveCatalogResourcePath).

const (
	// defaultTextHeadBytes is the head window served when the client does not ask
	// for a specific size. Large enough for a meaningful preview, small enough to
	// keep memory and latency flat.
	defaultTextHeadBytes = 1 << 20 // 1 MiB
	// maxTextHeadBytes caps any single text-head read so a client cannot defeat the
	// O(1)-memory guarantee by requesting an enormous window.
	maxTextHeadBytes = 4 << 20 // 4 MiB

	// defaultCsvRows / maxCsvRows bound a single CSV page.
	defaultCsvRows = 200
	maxCsvRows     = 2000
	// maxCsvScanBytes caps how far one CSV page request may scan, so a pathological
	// file (e.g. one multi-GB line) cannot make a single request unbounded.
	maxCsvScanBytes = 32 << 20 // 32 MiB
)

// openCatalogResourceForRequest resolves, authorizes, and opens a catalog resource
// for reading. It mirrors handleDownloadResource exactly. On any failure it writes
// the appropriate error response and returns ok=false; otherwise the caller owns
// (and must Close) the returned file.
func (deps ServerDeps) openCatalogResourceForRequest(w http.ResponseWriter, r *http.Request) (*os.File, os.FileInfo, domain.ResourceRecord, bool) {
	root, err := deps.resolvedUploadRoot()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return nil, nil, domain.ResourceRecord{}, false
	}
	catalog, ok := deps.resourceCatalogStore()
	if !ok {
		writeStoreError(w, store.ErrNotFound)
		return nil, nil, domain.ResourceRecord{}, false
	}
	if err := deps.ensureUploadCatalogMigrated(r.Context(), root); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return nil, nil, domain.ResourceRecord{}, false
	}
	fileID := chi.URLParam(r, "file_id")
	principal := deps.principalFromRequest(r, "")
	resource, err := catalog.GetResourceForUser(r.Context(), fileID, principal.UserID, principal.OrgID)
	if err != nil {
		writeStoreError(w, err)
		return nil, nil, domain.ResourceRecord{}, false
	}
	path, err := resolveCatalogResourcePath(root, resource)
	if err != nil {
		writeStoreError(w, err)
		return nil, nil, domain.ResourceRecord{}, false
	}
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			writeStoreError(w, store.ErrNotFound)
			return nil, nil, domain.ResourceRecord{}, false
		}
		writeError(w, http.StatusInternalServerError, err)
		return nil, nil, domain.ResourceRecord{}, false
	}
	info, err := file.Stat()
	if err != nil {
		_ = file.Close()
		writeError(w, http.StatusInternalServerError, err)
		return nil, nil, domain.ResourceRecord{}, false
	}
	if info.IsDir() {
		_ = file.Close()
		writeStoreError(w, store.ErrNotFound)
		return nil, nil, domain.ResourceRecord{}, false
	}
	return file, info, resource, true
}

type resourceTextHeadResponse struct {
	FileID           string `json:"file_id"`
	OriginalName     string `json:"original_name"`
	ContentType      string `json:"content_type"`
	Format           string `json:"format"`
	TotalSizeBytes   int64  `json:"total_size_bytes"`
	Offset           int64  `json:"offset"`
	ReturnedBytes    int64  `json:"returned_bytes"`
	NextOffset       int64  `json:"next_offset"`
	Truncated        bool   `json:"truncated"`
	Encoding         string `json:"encoding"`
	EOL              string `json:"eol"`
	LineCount        int    `json:"line_count"`
	ApproxTotalLines int64  `json:"approx_total_lines"`
	Text             string `json:"text"`
}

// handleResourceTextHead returns a bounded, UTF-8-safe head window of a text/data
// resource plus the metadata the viewer needs (total size, truncation, encoding,
// line estimate). O(maxBytes) memory; never reads the whole file.
func (deps ServerDeps) handleResourceTextHead(w http.ResponseWriter, r *http.Request) {
	file, info, resource, ok := deps.openCatalogResourceForRequest(w, r)
	if !ok {
		return
	}
	defer file.Close()

	maxBytes := clampInt64(parseInt64Query(r, "max_bytes", defaultTextHeadBytes), 1, maxTextHeadBytes)
	offset := parseInt64Query(r, "offset", 0)
	if offset < 0 {
		offset = 0
	}
	totalSize := info.Size()
	if offset > totalSize {
		offset = totalSize
	}

	raw := make([]byte, maxBytes)
	n, readErr := readAtOffset(file, offset, raw)
	if readErr != nil && !errors.Is(readErr, io.EOF) {
		writeError(w, http.StatusInternalServerError, readErr)
		return
	}
	raw = raw[:n]
	truncated := offset+int64(n) < totalSize

	text, encoding, keepRaw := decodeHeadWindow(raw, offset == 0, truncated)
	contentType := strings.TrimSpace(resource.ContentType)
	if contentType == "" || contentType == "application/octet-stream" {
		contentType = contentTypeForUpload(resource.OriginalName, contentType)
	}

	lineCount := countLines(text)
	approxTotalLines := int64(lineCount)
	if truncated && keepRaw > 0 && lineCount > 0 {
		approxTotalLines = int64(float64(lineCount) * (float64(totalSize) / float64(offset+int64(keepRaw))))
		if approxTotalLines < int64(lineCount) {
			approxTotalLines = int64(lineCount)
		}
	}

	writeJSON(w, http.StatusOK, resourceTextHeadResponse{
		FileID:           resource.ResourceID,
		OriginalName:     resource.OriginalName,
		ContentType:      contentType,
		Format:           detectTextFormat(resource.OriginalName, contentType, raw),
		TotalSizeBytes:   totalSize,
		Offset:           offset,
		ReturnedBytes:    int64(keepRaw),
		NextOffset:       offset + int64(keepRaw),
		Truncated:        truncated,
		Encoding:         encoding,
		EOL:              detectEOL(text),
		LineCount:        lineCount,
		ApproxTotalLines: approxTotalLines,
		Text:             text,
	})
}

type resourceCsvRowsResponse struct {
	FileID          string     `json:"file_id"`
	OriginalName    string     `json:"original_name"`
	Delimiter       string     `json:"delimiter"`
	Columns         []string   `json:"columns,omitempty"`
	Rows            [][]string `json:"rows"`
	OffsetBytes     int64      `json:"offset_bytes"`
	NextOffsetBytes int64      `json:"next_offset_bytes"`
	ReturnedRows    int        `json:"returned_rows"`
	HasMore         bool       `json:"has_more"`
	ApproxTotalRows int64      `json:"approx_total_rows"`
	TotalSizeBytes  int64      `json:"total_size_bytes"`
}

// handleResourceCsvRows returns one quote-aware page of CSV/TSV rows using cursor
// (byte-offset) pagination. Each response carries next_offset_bytes so the client
// can fetch the following page; offset_bytes==0 also returns the header row as
// columns. O(page_size) memory; never reads the whole file.
func (deps ServerDeps) handleResourceCsvRows(w http.ResponseWriter, r *http.Request) {
	file, info, resource, ok := deps.openCatalogResourceForRequest(w, r)
	if !ok {
		return
	}
	defer file.Close()

	limit := int(clampInt64(parseInt64Query(r, "limit", defaultCsvRows), 1, maxCsvRows))
	offsetBytes := parseInt64Query(r, "offset_bytes", 0)
	if offsetBytes < 0 {
		offsetBytes = 0
	}
	totalSize := info.Size()
	if offsetBytes > totalSize {
		offsetBytes = totalSize
	}

	delim, explicit := parseDelimiterParam(r.URL.Query().Get("delimiter"))
	if !explicit {
		if strings.EqualFold(filepath.Ext(resource.OriginalName), ".tsv") {
			delim = '\t'
		} else {
			// Sniff from the window we are about to read (offset 0 or a continuation
			// cursor) so a ';'/'|'/tab file paginates with the right delimiter on
			// every page, not just the first.
			sample := make([]byte, 16<<10)
			sn, _ := readAtOffset(file, offsetBytes, sample)
			delim = sniffCsvDelimiter(sample[:sn])
		}
	}

	if _, err := file.Seek(offsetBytes, io.SeekStart); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	// Read directly from the file (not an io.LimitReader): the real EOF marks real
	// record boundaries, so a record at the true end of the file is parsed whole. A
	// LimitReader would cut the last record mid-row, corrupting it and leaving the
	// cursor (InputOffset) pointing inside a record. The scan cap below is enforced
	// AFTER each complete record instead, so it can never split one.
	reader := csv.NewReader(file)
	reader.Comma = delim
	reader.LazyQuotes = true
	reader.FieldsPerRecord = -1
	reader.ReuseRecord = false

	var columns []string
	if offsetBytes == 0 {
		header, err := reader.Read()
		if err != nil && !errors.Is(err, io.EOF) {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		columns = header
	}

	rows := make([][]string, 0, limit)
	for len(rows) < limit {
		record, err := reader.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			// Tolerate a malformed record by stopping cleanly with what we have.
			break
		}
		rows = append(rows, record)
		// Backstop: cap how far one page scans (e.g. pathological multi-MB rows).
		// Checked after a complete record, so the cursor stays record-aligned.
		if reader.InputOffset() >= maxCsvScanBytes {
			break
		}
	}

	consumed := reader.InputOffset()
	nextOffset := offsetBytes + consumed
	if nextOffset > totalSize {
		nextOffset = totalSize
	}
	// nextOffset is always a true record boundary now, so "more remains" is simply
	// "the cursor has not reached the end".
	hasMore := nextOffset < totalSize

	approxTotalRows := int64(len(rows))
	if consumed > 0 {
		recordsConsumed := len(rows)
		if offsetBytes == 0 {
			recordsConsumed += 1 // header
		}
		if recordsConsumed > 0 {
			avg := float64(consumed) / float64(recordsConsumed)
			est := int64(float64(totalSize) / avg)
			if offsetBytes == 0 && est > 0 {
				est -= 1 // exclude header from the data-row estimate
			}
			if est > approxTotalRows {
				approxTotalRows = est
			}
		}
	}
	if !hasMore && offsetBytes == 0 {
		approxTotalRows = int64(len(rows)) // exact: whole file fit in one page
	}

	writeJSON(w, http.StatusOK, resourceCsvRowsResponse{
		FileID:          resource.ResourceID,
		OriginalName:    resource.OriginalName,
		Delimiter:       string(delim),
		Columns:         columns,
		Rows:            rows,
		OffsetBytes:     offsetBytes,
		NextOffsetBytes: nextOffset,
		ReturnedRows:    len(rows),
		HasMore:         hasMore,
		ApproxTotalRows: approxTotalRows,
		TotalSizeBytes:  totalSize,
	})
}

// --- helpers ---------------------------------------------------------------

func parseInt64Query(r *http.Request, key string, fallback int64) int64 {
	raw := strings.TrimSpace(r.URL.Query().Get(key))
	if raw == "" {
		return fallback
	}
	v, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return fallback
	}
	return v
}

func clampInt64(v, lo, hi int64) int64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// readAtOffset fills buf from the given offset, tolerating short reads. It returns
// the number of bytes read (io.EOF is not treated as an error by callers).
func readAtOffset(file *os.File, offset int64, buf []byte) (int, error) {
	if _, err := file.Seek(offset, io.SeekStart); err != nil {
		return 0, err
	}
	n, err := io.ReadFull(file, buf)
	if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
		return n, io.EOF
	}
	return n, err
}

// decodeHeadWindow turns a raw byte window into valid UTF-8 text plus the detected
// encoding and the number of raw bytes consumed (for cursor advance). When the
// window is truncated it drops the trailing partial line so a CSV row / token is
// never split mid-record.
func decodeHeadWindow(raw []byte, atStart bool, truncated bool) (text string, encoding string, keepRaw int) {
	if atStart && len(raw) >= 2 && raw[0] == 0xFF && raw[1] == 0xFE {
		// Consume only whole 16-bit code units so the cursor stays aligned (a
		// trailing odd byte is left for the next window).
		evenLen := len(raw) - ((len(raw) - 2) % 2)
		s := string(utf16.Decode(bytesToUint16(raw[2:evenLen], false)))
		if truncated {
			s = dropPartialLineString(s)
		}
		return s, "utf-16le", evenLen
	}
	if atStart && len(raw) >= 2 && raw[0] == 0xFE && raw[1] == 0xFF {
		evenLen := len(raw) - ((len(raw) - 2) % 2)
		s := string(utf16.Decode(bytesToUint16(raw[2:evenLen], true)))
		if truncated {
			s = dropPartialLineString(s)
		}
		return s, "utf-16be", evenLen
	}

	body := raw
	bomLen := 0
	encoding = "utf-8"
	if atStart && len(body) >= 3 && body[0] == 0xEF && body[1] == 0xBB && body[2] == 0xBF {
		body = body[3:]
		bomLen = 3
		encoding = "utf-8-bom"
	}

	keep := body
	if truncated {
		if idx := bytes.LastIndexByte(body, '\n'); idx >= 0 {
			keep = body[:idx+1]
		}
	}
	if utf8.Valid(keep) {
		return string(keep), encoding, bomLen + len(keep)
	}
	// Trim a trailing partial rune (no newline boundary was available).
	for len(keep) > 0 && !utf8.Valid(keep) {
		keep = keep[:len(keep)-1]
	}
	if utf8.Valid(keep) && len(keep) > 0 {
		return string(keep), encoding, bomLen + len(keep)
	}
	// Not valid UTF-8 — fall back to Latin-1 (every byte maps to a rune).
	latin := body
	if truncated {
		if idx := bytes.LastIndexByte(body, '\n'); idx >= 0 {
			latin = body[:idx+1]
		}
	}
	return latin1ToUTF8(latin), "latin-1", bomLen + len(latin)
}

func bytesToUint16(b []byte, bigEndian bool) []uint16 {
	out := make([]uint16, 0, len(b)/2)
	for i := 0; i+1 < len(b); i += 2 {
		if bigEndian {
			out = append(out, uint16(b[i])<<8|uint16(b[i+1]))
		} else {
			out = append(out, uint16(b[i+1])<<8|uint16(b[i]))
		}
	}
	return out
}

func latin1ToUTF8(b []byte) string {
	var sb strings.Builder
	sb.Grow(len(b))
	for _, c := range b {
		sb.WriteRune(rune(c))
	}
	return sb.String()
}

func dropPartialLineString(s string) string {
	if idx := strings.LastIndexByte(s, '\n'); idx >= 0 {
		return s[:idx+1]
	}
	return s
}

func countLines(text string) int {
	if text == "" {
		return 0
	}
	n := strings.Count(text, "\n")
	if !strings.HasSuffix(text, "\n") {
		n++
	}
	return n
}

func detectEOL(text string) string {
	if strings.Contains(text, "\r\n") {
		return "crlf"
	}
	if strings.Contains(text, "\n") {
		return "lf"
	}
	return "none"
}

func detectTextFormat(originalName, contentType string, sample []byte) string {
	switch strings.ToLower(filepath.Ext(originalName)) {
	case ".csv", ".tsv":
		return "csv"
	case ".json", ".jsonl", ".ndjson", ".geojson":
		return "json"
	case ".yaml", ".yml":
		return "yaml"
	case ".xml", ".xsd", ".xslt":
		return "xml"
	case ".md", ".markdown", ".mdx":
		return "markdown"
	case ".log", ".txt", ".text":
		return "text"
	}
	ct := strings.ToLower(contentType)
	switch {
	case strings.Contains(ct, "csv"), strings.Contains(ct, "tab-separated"):
		return "csv"
	case strings.Contains(ct, "json"):
		return "json"
	case strings.Contains(ct, "yaml"):
		return "yaml"
	case strings.Contains(ct, "xml"):
		return "xml"
	case strings.Contains(ct, "markdown"):
		return "markdown"
	}
	trimmed := bytes.TrimSpace(sample)
	if len(trimmed) > 0 {
		switch trimmed[0] {
		case '{', '[':
			return "json"
		case '<':
			return "xml"
		}
	}
	return "text"
}

func parseDelimiterParam(s string) (rune, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "auto":
		return 0, false
	case "comma", ",":
		return ',', true
	case "tab", "\\t", "\t":
		return '\t', true
	case "semicolon", ";":
		return ';', true
	case "pipe", "|":
		return '|', true
	case "space":
		return ' ', true
	}
	rs := []rune(strings.TrimSpace(s))
	if len(rs) == 1 {
		return rs[0], true
	}
	return 0, false
}

// sniffCsvDelimiter picks the most consistent delimiter across the first lines of
// a sample. A delimiter that yields a constant field count per line wins.
func sniffCsvDelimiter(sample []byte) rune {
	candidates := []rune{',', '\t', ';', '|'}
	lines := strings.Split(string(sample), "\n")
	if len(lines) > 20 {
		lines = lines[:20]
	}
	best := ','
	bestScore := -1
	for _, c := range candidates {
		counts := make([]int, 0, len(lines))
		for _, ln := range lines {
			if strings.TrimSpace(ln) == "" {
				continue
			}
			counts = append(counts, strings.Count(ln, string(c)))
		}
		if len(counts) == 0 {
			continue
		}
		minC, maxC, sum := counts[0], counts[0], 0
		for _, n := range counts {
			if n < minC {
				minC = n
			}
			if n > maxC {
				maxC = n
			}
			sum += n
		}
		if maxC == 0 {
			continue
		}
		score := sum
		if minC == maxC {
			score += 1000
		}
		if score > bestScore {
			bestScore = score
			best = c
		}
	}
	return best
}
