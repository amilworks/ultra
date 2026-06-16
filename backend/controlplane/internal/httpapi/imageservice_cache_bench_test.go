package httpapi

import (
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// Benchmarks for the image response cache hot path (tile/atlas/slice/thumbnail serving).
// These quantify C3 (cache-hit latency target << 5ms) and C4 (behavior under many
// concurrent viewers): the cache uses a single Mutex and get() mutates the LRU
// (MoveToFront + hits++), so every hit takes the write lock; imageCacheKey() also does an
// os.Stat + sha256 on every request. Run: go test ./internal/httpapi -run=^$ -bench=ImageCache -benchmem

func benchCachedResponse(n int) *cachedResponse {
	return &cachedResponse{
		status:      http.StatusOK,
		contentType: "image/png",
		headers:     http.Header{},
		body:        make([]byte, n),
	}
}

// BenchmarkImageCacheGetHit — raw cache-hit latency (serial). C3: must be far under 5ms.
func BenchmarkImageCacheGetHit(b *testing.B) {
	cache := newImageResponseCache(64 << 20)
	cache.put("hot", benchCachedResponse(64<<10), 64<<10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := cache.get("hot"); !ok {
			b.Fatal("expected hit")
		}
	}
}

// BenchmarkImageCacheGetHitParallel — concurrent hits on a hot key. Because get() mutates
// the LRU under the single Mutex, all concurrent hits serialize here; this measures whether
// the cache scales with cores (C4) or becomes a lock bottleneck under many viewers.
func BenchmarkImageCacheGetHitParallel(b *testing.B) {
	cache := newImageResponseCache(64 << 20)
	cache.put("hot", benchCachedResponse(64<<10), 64<<10)
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, ok := cache.get("hot"); !ok {
				b.Fatal("expected hit")
			}
		}
	})
}

// BenchmarkImageCacheMixedParallel — concurrent get over a working set larger than one
// entry (the realistic multi-viewer pattern: overlapping but distinct tiles).
func BenchmarkImageCacheMixedParallel(b *testing.B) {
	cache := newImageResponseCache(256 << 20)
	const keys = 1024
	keyList := make([]string, keys)
	for i := 0; i < keys; i++ {
		keyList[i] = "k" + strconv.Itoa(i)
		cache.put(keyList[i], benchCachedResponse(32<<10), 32<<10)
	}
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			k := keyList[i&(keys-1)]
			if _, ok := cache.get(k); !ok {
				cache.put(k, benchCachedResponse(32<<10), 32<<10)
			}
			i++
		}
	})
}

// BenchmarkImageCacheKey — per-request key derivation cost (os.Stat + sha256), paid on
// EVERY tile request including cache hits.
func BenchmarkImageCacheKey(b *testing.B) {
	dir := b.TempDir()
	p := filepath.Join(dir, "pyramid.tif")
	if err := os.WriteFile(p, make([]byte, 1<<20), 0o644); err != nil {
		b.Fatal(err)
	}
	q := url.Values{"path": {p}, "size": {"256"}, "level": {"3"}, "col": {"12"}, "row": {"7"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := imageCacheKey("/tile", q); !ok {
			b.Fatal("expected key")
		}
	}
}

// BenchmarkImageCacheKeyParallel — key derivation under concurrency (os.Stat syscall +
// sha256 on every concurrent tile request).
func BenchmarkImageCacheKeyParallel(b *testing.B) {
	dir := b.TempDir()
	p := filepath.Join(dir, "pyramid.tif")
	if err := os.WriteFile(p, make([]byte, 1<<20), 0o644); err != nil {
		b.Fatal(err)
	}
	q := url.Values{"path": {p}, "size": {"256"}, "level": {"3"}, "col": {"12"}, "row": {"7"}}
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, ok := imageCacheKey("/tile", q); !ok {
				b.Fatal("expected key")
			}
		}
	})
}
