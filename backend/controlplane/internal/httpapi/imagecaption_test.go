package httpapi

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeTestPNG(t *testing.T, path string, w, h int) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x % 256), G: uint8(y % 256), B: 128, A: 255})
		}
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create png: %v", err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
}

func TestPrepareCaptionImageDownscales(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	p := filepath.Join(dir, "fig.png")
	writeTestPNG(t, p, 2400, 1000)

	b64, err := prepareCaptionImageBase64(p, 1280)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if b64 == "" {
		t.Fatal("empty base64")
	}
	// Decode the produced JPEG back and confirm the long edge was bounded.
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		t.Fatalf("decode b64: %v", err)
	}
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("decode jpeg: %v", err)
	}
	if b := img.Bounds(); b.Dx() > 1280 || b.Dy() > 1280 {
		t.Fatalf("long edge not bounded: %dx%d", b.Dx(), b.Dy())
	}
}

func TestCaptionerDisabledIsNoOp(t *testing.T) {
	t.Parallel()
	c := &imageCaptioner{enabled: false}
	caption, err := c.captionForFile(context.Background(), "/nonexistent.png")
	if err != nil || caption != "" {
		t.Fatalf("disabled captioner should no-op, got (%q, %v)", caption, err)
	}
}

func TestCaptionCacheHitSkipsVLM(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	cacheDir := filepath.Join(dir, ".captions")
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		t.Fatal(err)
	}
	fig := filepath.Join(dir, "fig.png")
	writeTestPNG(t, fig, 64, 64)

	c := &imageCaptioner{
		enabled:  true,
		baseURL:  "http://127.0.0.1:1", // would fail if called — proves the cache short-circuits
		model:    "test",
		maxEdge:  1280,
		timeout:  time.Second,
		cacheDir: cacheDir,
		sem:      make(chan struct{}, 1),
	}
	// Pre-seed the cache for this file's stamp.
	info, _ := os.Stat(fig)
	key := captionCacheKey(fig, info)
	want := "A scatter plot of X versus Y for two classes."
	if err := os.WriteFile(filepath.Join(cacheDir, key+".txt"), []byte(want), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := c.captionForFile(context.Background(), fig)
	if err != nil {
		t.Fatalf("cache hit errored: %v", err)
	}
	if got != want {
		t.Fatalf("caption = %q, want cached %q", got, want)
	}
}

func TestCaptionCacheKeyChangesWithContent(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	fig := filepath.Join(dir, "fig.png")
	writeTestPNG(t, fig, 32, 32)
	info1, _ := os.Stat(fig)
	k1 := captionCacheKey(fig, info1)

	time.Sleep(10 * time.Millisecond)
	writeTestPNG(t, fig, 48, 48) // re-rendered figure (new size/mtime)
	info2, _ := os.Stat(fig)
	k2 := captionCacheKey(fig, info2)
	if k1 == k2 {
		t.Fatal("cache key should change when the figure is re-rendered")
	}
}

// TestLiveCaptionAgainstVLM hits the real VLM when ULTRA_CONTROL_VLM_BASE_URL is set
// (env-gated, skipped in CI) to confirm the Go request shape + thinking-OFF parse work
// end to end and produce a faithful, non-empty caption.
func TestLiveCaptionAgainstVLM(t *testing.T) {
	base := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_VLM_BASE_URL"))
	fig := strings.TrimSpace(os.Getenv("CAPTION_TEST_FIGURE"))
	if base == "" || fig == "" {
		t.Skip("set ULTRA_CONTROL_VLM_BASE_URL + CAPTION_TEST_FIGURE to run the live VLM caption test")
	}
	t.Setenv("ULTRA_CONTROL_VLM_ENABLED", "1")
	c := newImageCaptionerFromEnv(t.TempDir())
	if !c.enabled {
		t.Fatal("captioner should be enabled with env set")
	}
	caption, err := c.captionForFile(context.Background(), fig)
	if err != nil {
		t.Fatalf("live caption failed: %v", err)
	}
	if len(strings.Fields(caption)) < 5 {
		t.Fatalf("caption too short / empty: %q", caption)
	}
	t.Logf("LIVE CAPTION: %s", caption)
}
