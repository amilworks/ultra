package httpapi

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
)

// imageResponseCache is a bounded in-process LRU over the image service's repeatable
// read responses (tiles, atlases, thumbnails, slices). A DeepZoom pan/zoom and repeated
// 3D loads re-request the SAME engine outputs constantly; without this every one hits the
// Python sidecar's global decode lock. A cache hit is a memory copy — the engine is skipped
// entirely. Bounded by total bytes (LRU eviction) and a per-entry cap so an oversized
// response self-skips. Keyed on the upstream request + the served file's stat stamp, so a
// re-derived pyramid (new file) never serves stale bytes.
type imageResponseCache struct {
	mu       sync.Mutex
	ll       *list.List               // front = most-recently-used
	items    map[string]*list.Element // key -> *cacheItem element
	curBytes int64
	maxBytes int64
	maxEntry int64 // responses larger than this are served but not cached
	hits     uint64
	misses   uint64
}

type cacheItem struct {
	key  string
	resp *cachedResponse
	size int64
}

// cachedResponse is the captured engine response: status, content-type, the forwarded
// x-volume-*/x-image-* sidecar headers, and the body bytes.
type cachedResponse struct {
	status      int
	contentType string
	headers     http.Header
	body        []byte
}

// newImageResponseCache builds a cache bounded to maxBytes total. Per-entry cap is
// max(1MiB, maxBytes/16) so one large atlas can't evict the whole working set.
func newImageResponseCache(maxBytes int64) *imageResponseCache {
	if maxBytes <= 0 {
		return nil
	}
	maxEntry := maxBytes / 16
	if maxEntry < (1 << 20) {
		maxEntry = 1 << 20
	}
	return &imageResponseCache{
		ll:       list.New(),
		items:    make(map[string]*list.Element),
		maxBytes: maxBytes,
		maxEntry: maxEntry,
	}
}

// newImageResponseCacheFromEnv reads ULTRA_CONTROL_IMAGE_CACHE_BYTES (default 512MiB;
// "0" disables, returning nil so the proxy streams uncached).
func newImageResponseCacheFromEnv() *imageResponseCache {
	maxBytes := int64(512 << 20)
	if raw := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_IMAGE_CACHE_BYTES")); raw != "" {
		if v, err := strconv.ParseInt(raw, 10, 64); err == nil {
			maxBytes = v
		}
	}
	return newImageResponseCache(maxBytes)
}

func (c *imageResponseCache) get(key string) (*cachedResponse, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[key]; ok {
		c.ll.MoveToFront(el)
		c.hits++
		return el.Value.(*cacheItem).resp, true
	}
	c.misses++
	return nil, false
}

func (c *imageResponseCache) put(key string, resp *cachedResponse, size int64) {
	if size > c.maxEntry {
		return // oversized: served, not cached
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[key]; ok {
		item := el.Value.(*cacheItem)
		c.curBytes += size - item.size
		item.resp, item.size = resp, size
		c.ll.MoveToFront(el)
	} else {
		el := c.ll.PushFront(&cacheItem{key: key, resp: resp, size: size})
		c.items[key] = el
		c.curBytes += size
	}
	for c.curBytes > c.maxBytes && c.ll.Len() > 0 {
		oldest := c.ll.Back()
		item := oldest.Value.(*cacheItem)
		c.ll.Remove(oldest)
		delete(c.items, item.key)
		c.curBytes -= item.size
	}
}

// stats returns (hits, misses, entries, bytes) for observability/tests.
func (c *imageResponseCache) stats() (hits, misses uint64, entries int, bytes int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits, c.misses, c.ll.Len(), c.curBytes
}

// imageCacheKey derives a stable cache key from the upstream endpoint+query and the served
// file's stat stamp. Returns ("", false) when the file can't be stamped (then we don't cache,
// the safe default). The stamp (size+mtime) means a re-derived pyramid never serves stale tiles.
func imageCacheKey(endpoint string, query url.Values) (string, bool) {
	path := query.Get("path")
	if path == "" {
		return "", false
	}
	fi, err := os.Stat(path)
	if err != nil || fi.IsDir() {
		return "", false
	}
	stamp := strconv.FormatInt(fi.Size(), 10) + ":" + strconv.FormatInt(fi.ModTime().UnixNano(), 10)
	sum := sha256.Sum256([]byte(endpoint + "?" + query.Encode() + "|" + stamp))
	return hex.EncodeToString(sum[:]), true
}

// proxyImageServiceCached serves a repeatable image-read endpoint through the response cache:
// a cache hit replays captured bytes without touching the engine; a miss fetches, serves, and
// stores a successful response. Falls back to the plain streaming proxy when the cache is
// disabled or the request can't be keyed.
func (deps ServerDeps) proxyImageServiceCached(w http.ResponseWriter, r *http.Request, endpoint string, query url.Values, fallback ...http.HandlerFunc) {
	cache := deps.imageCache
	if cache == nil {
		deps.proxyImageService(w, r, endpoint, query, fallback...)
		return
	}
	key, ok := imageCacheKey(endpoint, query)
	if !ok {
		deps.proxyImageService(w, r, endpoint, query, fallback...)
		return
	}
	if resp, hit := cache.get(key); hit {
		writeCachedResponse(w, resp, "hit")
		return
	}

	base := strings.TrimRight(strings.TrimSpace(deps.ImageServiceURL), "/")
	target := base + endpoint
	if encoded := query.Encode(); encoded != "" {
		target += "?" + encoded
	}
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, target, nil)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	resp, err := imageServiceHTTPClient.Do(req)
	if err != nil {
		if len(fallback) > 0 && fallback[0] != nil {
			fallback[0](w, r) // image service unreachable -> legacy native path
			return
		}
		writeError(w, http.StatusBadGateway, fmt.Errorf("image service request failed: %w", err))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Never cache or forward an upstream error body (internal paths / tracebacks);
		// reply with a clean, bounded error at the same status. See the streaming proxy.
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		writeImageServiceUpstreamError(r.Context(), w, endpoint, resp.StatusCode, detail)
		return
	}

	contentType := resp.Header.Get("Content-Type")
	forwarded := http.Header{}
	for k, values := range resp.Header {
		if lk := strings.ToLower(k); strings.HasPrefix(lk, "x-volume-") || strings.HasPrefix(lk, "x-image-") {
			forwarded[k] = values
		}
	}
	// Read at most maxEntry+1 so an unexpectedly large body bounds memory and self-skips caching.
	body, _ := io.ReadAll(io.LimitReader(resp.Body, cache.maxEntry+1))

	if contentType != "" {
		w.Header().Set("Content-Type", contentType)
	}
	for k, values := range forwarded {
		for _, v := range values {
			w.Header().Add(k, v)
		}
	}
	if resp.StatusCode == http.StatusOK {
		w.Header().Set("Cache-Control", "private, max-age=3600")
	}
	w.Header().Set("X-Ultra-Cache", "miss")

	if int64(len(body)) > cache.maxEntry {
		// Too large to cache: serve the buffered head + stream the remainder.
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(body)
		_, _ = io.Copy(w, resp.Body)
		return
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(body)
	if resp.StatusCode == http.StatusOK {
		cache.put(key, &cachedResponse{status: resp.StatusCode, contentType: contentType, headers: forwarded, body: body}, int64(len(body)))
	}
}

func writeCachedResponse(w http.ResponseWriter, resp *cachedResponse, marker string) {
	if resp.contentType != "" {
		w.Header().Set("Content-Type", resp.contentType)
	}
	for k, values := range resp.headers {
		for _, v := range values {
			w.Header().Add(k, v)
		}
	}
	w.Header().Set("Cache-Control", "private, max-age=3600")
	w.Header().Set("X-Ultra-Cache", marker)
	w.WriteHeader(resp.status)
	_, _ = w.Write(resp.body)
}
