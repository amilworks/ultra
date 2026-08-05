package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

type tileReq struct{ level, col, row int }

// panZoomTrace simulates a DeepZoom interaction: a viewport panning across a level
// (consecutive viewports overlap heavily), a zoom-in to a finer level, then a zoom-out
// that revisits already-seen coarse tiles. The result is a request stream with many
// repeats — exactly the pattern a tile cache eliminates.
func panZoomTrace() []tileReq {
	var trace []tileReq
	viewport := func(level, c0, r0, w, h int) {
		for r := r0; r < r0+h; r++ {
			for c := c0; c < c0+w; c++ {
				trace = append(trace, tileReq{level, c, r})
			}
		}
	}
	for step := 0; step < 10; step++ { // pan a 5x4 viewport right by 1 col each step
		viewport(2, step, 3, 5, 4)
	}
	for step := 0; step < 6; step++ { // zoom in: pan a 4x3 viewport at the finer level
		viewport(1, 4+step, 6, 4, 3)
	}
	for step := 0; step < 5; step++ { // zoom back out: revisits coarse tiles already seen
		viewport(2, step, 3, 5, 4)
	}
	return trace
}

func uniqueTileCount(trace []tileReq) int {
	seen := map[tileReq]struct{}{}
	for _, t := range trace {
		seen[t] = struct{}{}
	}
	return len(seen)
}

// TestImageCacheBeforeAfterPanZoom is the concrete before/after: it replays the SAME
// realistic pan/zoom tile trace through the control plane with the response cache OFF then
// ON, against a counting + latency "image service". It proves the cache eliminates the
// repeated engine decodes (the metric that matters) and is byte-correct.
func TestImageCacheBeforeAfterPanZoom(t *testing.T) {
	// Default the cache OFF via env so NewRouter only enables it when we inject one.
	t.Setenv("ULTRA_CONTROL_IMAGE_CACHE_BYTES", "0")

	const perTileDecode = 5 * time.Millisecond // representative bounded-tile decode latency
	var engineCalls int64
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/viewerinfo" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"axis_sizes": map[string]any{"T": 1, "C": 3, "Z": 1, "Y": 4, "X": 4},
			})
			return
		}
		if r.URL.Path != "/tile" {
			http.NotFound(w, r)
			return
		}
		atomic.AddInt64(&engineCalls, 1)
		time.Sleep(perTileDecode)
		q := r.URL.Query()
		// Body is a pure function of the tile coords, so a cached response must equal
		// what a fresh engine decode would have returned.
		body := []byte(fmt.Sprintf("TILE-%s-%s-%s", q.Get("level"), q.Get("col"), q.Get("row")))
		w.Header().Set("Content-Type", "image/png")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	}))
	defer imageSvc.Close()

	mem := store.NewMemoryStore()
	uploadRoot := t.TempDir()
	build := func(cache *imageResponseCache) http.Handler {
		return NewRouter(ServerDeps{
			Version:         "test-version",
			Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
			Store:           mem,
			UploadRoot:      uploadRoot,
			ImageServiceURL: imageSvc.URL,
			imageCache:      cache,
		})
	}

	// Upload a file + drop a derived pyramid so tile path resolution + the cache stat-stamp work.
	uploadRouter := build(nil)
	fileID := uploadNamedFileForProxyTest(t, uploadRouter, "slide.png", testPNGBytes(t, 4, 4))
	derivedDir := filepath.Join(uploadRoot, "derived")
	if err := os.MkdirAll(derivedDir, 0o755); err != nil {
		t.Fatalf("mkdir derived: %v", err)
	}
	if err := os.WriteFile(filepath.Join(derivedDir, derivedPyramidName(fileID)), []byte("PYRAMID-BYTES"), 0o644); err != nil {
		t.Fatalf("write pyramid: %v", err)
	}

	trace := panZoomTrace()
	uniques := uniqueTileCount(trace)

	replay := func(router http.Handler) (bodies [][]byte, elapsed time.Duration) {
		atomic.StoreInt64(&engineCalls, 0)
		start := time.Now()
		for _, tr := range trace {
			url := fmt.Sprintf("/v2/uploads/%s/tiles/z/%d/%d/%d?size=256", fileID, tr.level, tr.col, tr.row)
			req := httptest.NewRequest(http.MethodGet, url, nil)
			setProxyOwnerHeaders(req)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Fatalf("tile %v -> %d: %s", tr, rec.Code, rec.Body.String())
			}
			bodies = append(bodies, append([]byte(nil), rec.Body.Bytes()...))
		}
		return bodies, time.Since(start)
	}

	// BEFORE: cache disabled — every request hits the engine.
	bodiesOff, elapsedOff := replay(build(nil))
	callsOff := atomic.LoadInt64(&engineCalls)

	// AFTER: cache enabled.
	cache := newImageResponseCache(64 << 20)
	bodiesOn, elapsedOn := replay(build(cache))
	callsOn := atomic.LoadInt64(&engineCalls)
	hits, misses, _, _ := cache.stats()

	t.Logf("trace=%d requests, %d unique tiles", len(trace), uniques)
	t.Logf("BEFORE (no cache): engine calls=%d, wall=%v", callsOff, elapsedOff.Round(time.Millisecond))
	t.Logf("AFTER  (cache on): engine calls=%d, wall=%v, cache hits=%d misses=%d", callsOn, elapsedOn.Round(time.Millisecond), hits, misses)
	t.Logf("engine-call reduction=%.1f%%, speedup=%.1fx", 100*(1-float64(callsOn)/float64(callsOff)), float64(elapsedOff)/float64(elapsedOn))

	// Hard, deterministic guarantees (the real win):
	if callsOff != int64(len(trace)) {
		t.Fatalf("BEFORE should hit the engine for every request: got %d want %d", callsOff, len(trace))
	}
	if callsOn != int64(uniques) {
		t.Fatalf("AFTER should hit the engine only for unique tiles: got %d want %d", callsOn, uniques)
	}
	if callsOn*2 >= callsOff {
		t.Fatalf("expected a substantial engine-call reduction: %d -> %d", callsOff, callsOn)
	}
	if hits != uint64(len(trace)-uniques) || misses != uint64(uniques) {
		t.Fatalf("cache stats off: hits=%d misses=%d (want hits=%d misses=%d)", hits, misses, len(trace)-uniques, uniques)
	}
	// Byte-for-byte correctness: cached responses match the engine's.
	for i := range bodiesOff {
		if string(bodiesOff[i]) != string(bodiesOn[i]) {
			t.Fatalf("request %d body mismatch: %q (no cache) vs %q (cache)", i, bodiesOff[i], bodiesOn[i])
		}
	}
	// Soft latency check (the structural win; loose to avoid CI-timing flakiness).
	if elapsedOn >= elapsedOff {
		t.Fatalf("cache did not reduce wall-clock: before=%v after=%v", elapsedOff, elapsedOn)
	}
}

// TestImageCacheInvalidatesOnReDerive proves the stat-stamped key never serves stale tiles:
// after a source is re-derived (the served file changes), the next request is a fresh miss.
func TestImageCacheInvalidatesOnReDerive(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_IMAGE_CACHE_BYTES", "0")
	var engineCalls int64
	viewerInfo := derivativeViewerInfoForTest(1, 3, 1, 4, 4, 512)
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/viewerinfo" {
			_ = json.NewEncoder(w).Encode(viewerInfo)
			return
		}
		atomic.AddInt64(&engineCalls, 1)
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("X"))
	}))
	defer imageSvc.Close()

	mem := store.NewMemoryStore()
	uploadRoot := t.TempDir()
	router := NewRouter(ServerDeps{
		Version:         "test-version",
		Runs:            runcontrol.NewService(mem, eventbus.NewMemoryBus()),
		Store:           mem,
		UploadRoot:      uploadRoot,
		ImageServiceURL: imageSvc.URL,
		imageCache:      newImageResponseCache(8 << 20),
	})
	fileID := uploadNamedFileForProxyTest(t, router, "slide.png", testPNGBytes(t, 4, 4))
	record := uploadedResourceRecordForTest(t, mem, fileID)
	sourcePath := uploadedSourcePathForTest(t, uploadRoot, fileID)
	capabilities := derivativeCapabilities{Tile: true, TileT: true, TileZ: true, Slice: true, Thumbnail: true, OrderedChannels: true, LUT: true}
	writeStrictDerivativeForTest(t, uploadRoot, sourcePath, record, viewerInfo, capabilities, []byte("PYRAMID-V1"))

	get := func() {
		req := httptest.NewRequest(http.MethodGet, "/v2/uploads/"+fileID+"/tiles/z/0/0/0", nil)
		setProxyOwnerHeaders(req)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("tile status %d", rec.Code)
		}
	}

	get() // miss -> engine
	get() // hit  -> no engine
	if got := atomic.LoadInt64(&engineCalls); got != 1 {
		t.Fatalf("after warm cache, engine calls=%d want 1", got)
	}
	// Re-derive: manifest-last publication points at a new immutable artifact, so the
	// served path and cache key both change.
	writeStrictDerivativeForTest(t, uploadRoot, sourcePath, record, viewerInfo, capabilities, []byte("PYRAMID-VERSION-2-LARGER"))
	get() // must be a fresh miss, not a stale hit
	if got := atomic.LoadInt64(&engineCalls); got != 2 {
		t.Fatalf("after re-derive, engine calls=%d want 2 (stale tile served!)", got)
	}
}

func TestImageCacheKeyRejectsSameSizeSameMtimeReplacement(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "image.tif")
	fixedTime := time.Unix(1_700_000_000, 123_000_000)
	if err := os.WriteFile(path, []byte("AAAA"), 0o600); err != nil {
		t.Fatalf("write initial file: %v", err)
	}
	if err := os.Chtimes(path, fixedTime, fixedTime); err != nil {
		t.Fatalf("stamp initial file: %v", err)
	}
	query := url.Values{"path": {path}, "z": {"0"}}
	initialKey, ok := imageCacheKey("/slice", query)
	if !ok {
		t.Fatal("initial file did not produce a cache key")
	}

	replacement := filepath.Join(root, "replacement.tif")
	if err := os.WriteFile(replacement, []byte("BBBB"), 0o600); err != nil {
		t.Fatalf("write replacement: %v", err)
	}
	if err := os.Chtimes(replacement, fixedTime, fixedTime); err != nil {
		t.Fatalf("stamp replacement: %v", err)
	}
	if err := os.Rename(replacement, path); err != nil {
		t.Fatalf("replace file: %v", err)
	}
	replacementKey, ok := imageCacheKey("/slice", query)
	if !ok {
		t.Fatal("replacement file did not produce a cache key")
	}
	if replacementKey == initialKey {
		t.Fatal("same-size/same-mtime replacement reused the stale cache key")
	}
}

func TestCachedViewerInfoCoalescesConcurrentMisses(t *testing.T) {
	path := filepath.Join(t.TempDir(), "source.czi")
	if err := os.WriteFile(path, []byte("source"), 0o600); err != nil {
		t.Fatalf("write source: %v", err)
	}
	var calls int64
	release := make(chan struct{})
	called := make(chan struct{}, 1)
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/viewerinfo" {
			http.NotFound(w, r)
			return
		}
		atomic.AddInt64(&calls, 1)
		select {
		case called <- struct{}{}:
		default:
		}
		<-release
		_ = json.NewEncoder(w).Encode(map[string]any{
			"axis_sizes": map[string]any{"T": 1, "C": 3, "Z": 1},
		})
	}))
	defer imageSvc.Close()

	deps := ServerDeps{ImageServiceURL: imageSvc.URL}
	cache := newImageResponseCache(8 << 20)
	const concurrency = 16
	start := make(chan struct{})
	var ready sync.WaitGroup
	var done sync.WaitGroup
	errs := make(chan error, concurrency)
	ready.Add(concurrency)
	done.Add(concurrency)
	for range concurrency {
		go func() {
			defer done.Done()
			ready.Done()
			<-start
			info, err := deps.cachedImageServiceViewerInfoVia(context.Background(), path, cache)
			if err == nil {
				if _, _, _, ok := sourceViewerAxes(info); !ok {
					errs <- fmt.Errorf("viewer info axes are malformed: %#v", info)
				}
			} else {
				errs <- err
			}
		}()
	}
	ready.Wait()
	close(start)
	<-called
	close(release)
	done.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if got := atomic.LoadInt64(&calls); got != 1 {
		t.Fatalf("viewer-info upstream calls = %d, want 1", got)
	}
}
