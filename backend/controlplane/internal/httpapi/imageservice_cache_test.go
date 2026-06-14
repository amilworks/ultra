package httpapi

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
	imageSvc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	derivedDir := filepath.Join(uploadRoot, "derived")
	_ = os.MkdirAll(derivedDir, 0o755)
	derivedPath := filepath.Join(derivedDir, derivedPyramidName(fileID))
	if err := os.WriteFile(derivedPath, []byte("PYRAMID-V1"), 0o644); err != nil {
		t.Fatalf("write pyramid: %v", err)
	}

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
	// Re-derive: the served file changes size (and mtime), so the key must change.
	if err := os.WriteFile(derivedPath, []byte("PYRAMID-VERSION-2-LARGER"), 0o644); err != nil {
		t.Fatalf("re-derive: %v", err)
	}
	get() // must be a fresh miss, not a stale hit
	if got := atomic.LoadInt64(&engineCalls); got != 2 {
		t.Fatalf("after re-derive, engine calls=%d want 2 (stale tile served!)", got)
	}
}
