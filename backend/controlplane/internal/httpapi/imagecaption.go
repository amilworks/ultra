package httpapi

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif" // register decoders for image.Decode
	"image/jpeg"
	_ "image/png" // register the PNG decoder (matplotlib figures)
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	xdraw "golang.org/x/image/draw"
	"golang.org/x/sync/singleflight"
)

// imageCaptioner generates calm, academic-style captions for run-output figures via
// a grounded (thinking-OFF) vision-language model. It is LAZY (only when a figure is
// viewed), CACHED on disk (generated once per figure, stat-stamped so a re-rendered
// figure re-captions), CONCURRENCY-BOUNDED, and DEGRADES GRACEFULLY: if the VLM is
// disabled or unreachable, captioning is a no-op and the figure simply shows no
// caption — it can never block or break figure display.
type imageCaptioner struct {
	enabled    bool
	baseURL    string
	apiKey     string
	model      string
	maxEdge    int
	timeout    time.Duration
	cacheDir   string
	httpClient *http.Client
	sem        chan struct{}
	group      singleflight.Group
}

// The caption prompt: describe ONLY what is visibly labeled, never invent values or
// conclusions. Thinking-OFF + this prompt is the validated low-confabulation read.
const captionPrompt = "You are writing a figure caption for a scientific report, in the neutral style of an academic paper. " +
	"In ONE or TWO sentences, describe what this figure shows: the plot type, what is on the axes, and the groups/series in the legend. " +
	"Write ONLY what is visibly labeled in the figure (title, axis labels, legend, annotations). " +
	"Do NOT invent numeric values, statistics, or conclusions that are not printed on the figure. " +
	"Begin with the subject (not 'This figure shows'). Be concise and neutral."

func readSecretMaybeFile(value, fileEnv string) string {
	if v := strings.TrimSpace(value); v != "" {
		return v
	}
	if path := strings.TrimSpace(os.Getenv(fileEnv)); path != "" {
		if b, err := os.ReadFile(path); err == nil {
			return strings.TrimSpace(string(b))
		}
	}
	return ""
}

// newImageCaptionerFromEnv builds the captioner. Disabled (a no-op) unless
// ULTRA_CONTROL_VLM_ENABLED is truthy AND a base URL + model are configured.
func newImageCaptionerFromEnv(artifactRoot string) *imageCaptioner {
	enabled := false
	switch strings.ToLower(strings.TrimSpace(os.Getenv("ULTRA_CONTROL_VLM_ENABLED"))) {
	case "1", "true", "yes", "on":
		enabled = true
	}
	baseURL := strings.TrimRight(strings.TrimSpace(os.Getenv("ULTRA_CONTROL_VLM_BASE_URL")), "/")
	model := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_VLM_MODEL"))
	if model == "" {
		model = "Qwen3.6-27B"
	}
	maxEdge := 1280
	if v, err := strconv.Atoi(strings.TrimSpace(os.Getenv("ULTRA_CONTROL_VLM_MAX_EDGE"))); err == nil && v > 0 {
		maxEdge = v
	}
	timeout := 60 * time.Second
	if v, err := strconv.Atoi(strings.TrimSpace(os.Getenv("ULTRA_CONTROL_VLM_TIMEOUT_S"))); err == nil && v > 0 {
		timeout = time.Duration(v) * time.Second
	}
	concurrency := 3
	if v, err := strconv.Atoi(strings.TrimSpace(os.Getenv("ULTRA_CONTROL_VLM_MAX_CONCURRENCY"))); err == nil && v > 0 {
		concurrency = v
	}
	cacheDir := strings.TrimSpace(os.Getenv("ULTRA_CONTROL_CAPTION_CACHE_DIR"))
	if cacheDir == "" && strings.TrimSpace(artifactRoot) != "" {
		cacheDir = filepath.Join(artifactRoot, ".captions")
	}
	if enabled && (baseURL == "" || cacheDir == "") {
		enabled = false // misconfigured → safe no-op
	}
	return &imageCaptioner{
		enabled:    enabled,
		baseURL:    baseURL,
		apiKey:     readSecretMaybeFile(os.Getenv("ULTRA_CONTROL_VLM_API_KEY"), "ULTRA_CONTROL_VLM_API_KEY_FILE"),
		model:      model,
		maxEdge:    maxEdge,
		timeout:    timeout,
		cacheDir:   cacheDir,
		httpClient: &http.Client{Timeout: timeout},
		sem:        make(chan struct{}, concurrency),
	}
}

func captionCacheKey(path string, info os.FileInfo) string {
	stamp := strconv.FormatInt(info.Size(), 10) + ":" + strconv.FormatInt(info.ModTime().UnixNano(), 10)
	sum := sha256.Sum256([]byte(path + "|" + stamp))
	return hex.EncodeToString(sum[:])
}

// captionForFile returns the cached caption, or generates+caches one. Returns
// ("", nil) when captioning is disabled or the file can't be read — never an error
// the caller must surface (degrade gracefully).
func (c *imageCaptioner) captionForFile(ctx context.Context, path string) (string, error) {
	if c == nil || !c.enabled {
		return "", nil
	}
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return "", nil
	}
	key := captionCacheKey(path, info)
	cacheFile := filepath.Join(c.cacheDir, key+".txt")
	if b, err := os.ReadFile(cacheFile); err == nil {
		return strings.TrimSpace(string(b)), nil
	}
	// Single-flight by key: concurrent viewers of the same fresh figure generate once.
	v, err, _ := c.group.Do(key, func() (any, error) {
		// Re-check the cache inside the flight (a prior holder may have written it).
		if b, rerr := os.ReadFile(cacheFile); rerr == nil {
			return strings.TrimSpace(string(b)), nil
		}
		select {
		case c.sem <- struct{}{}:
			defer func() { <-c.sem }()
		case <-ctx.Done():
			return "", ctx.Err()
		}
		b64, perr := prepareCaptionImageBase64(path, c.maxEdge)
		if perr != nil {
			return "", perr
		}
		caption, gerr := c.generate(ctx, b64)
		if gerr != nil {
			return "", gerr
		}
		caption = strings.TrimSpace(caption)
		if caption != "" {
			_ = os.MkdirAll(c.cacheDir, 0o755)
			tmp := cacheFile + ".tmp"
			if werr := os.WriteFile(tmp, []byte(caption), 0o644); werr == nil {
				_ = os.Rename(tmp, cacheFile)
			}
		}
		return caption, nil
	})
	if err != nil {
		return "", err
	}
	caption, _ := v.(string)
	return caption, nil
}

// prepareCaptionImageBase64 decodes an image, downscales its long edge to <= maxEdge
// (so a large figure never balloons the VLM request), and JPEG-encodes it.
func prepareCaptionImageBase64(path string, maxEdge int) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if info.Size() > 64<<20 { // never decode a pathologically large file just to caption it
		return "", errors.New("image too large to caption")
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return "", err
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return "", errors.New("empty image")
	}
	if w > maxEdge || h > maxEdge {
		scale := float64(maxEdge) / float64(maxInt(w, h))
		nw, nh := maxInt(1, int(float64(w)*scale)), maxInt(1, int(float64(h)*scale))
		dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
		xdraw.CatmullRom.Scale(dst, dst.Bounds(), img, b, xdraw.Over, nil)
		img = dst
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85}); err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// --- VLM call (OpenAI-compatible chat/completions, thinking OFF) ------------------

type vlmChatRequest struct {
	Model             string           `json:"model"`
	Messages          []vlmChatMessage `json:"messages"`
	Temperature       float64          `json:"temperature"`
	TopP              float64          `json:"top_p"`
	MaxTokens         int              `json:"max_tokens"`
	ChatTemplateKwarg map[string]any   `json:"chat_template_kwargs"`
}

type vlmChatMessage struct {
	Role    string            `json:"role"`
	Content []vlmContentBlock `json:"content"`
}

type vlmContentBlock struct {
	Type     string       `json:"type"`
	Text     string       `json:"text,omitempty"`
	ImageURL *vlmImageURL `json:"image_url,omitempty"`
}

type vlmImageURL struct {
	URL string `json:"url"`
}

type vlmChatResponse struct {
	Choices []struct {
		Message struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"message"`
	} `json:"choices"`
}

func (c *imageCaptioner) generate(ctx context.Context, imageB64 string) (string, error) {
	reqBody := vlmChatRequest{
		Model:       c.model,
		Temperature: 0.7,
		TopP:        0.8,
		MaxTokens:   512,
		// Thinking OFF: the validated low-confabulation "grounded" read — extended
		// thinking makes the model reason itself into claims not in the figure.
		ChatTemplateKwarg: map[string]any{"enable_thinking": false},
		Messages: []vlmChatMessage{{
			Role: "user",
			Content: []vlmContentBlock{
				{Type: "text", Text: captionPrompt},
				{Type: "image_url", ImageURL: &vlmImageURL{URL: "data:image/jpeg;base64," + imageB64}},
			},
		}},
	}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("vlm caption status %d", resp.StatusCode)
	}
	var out vlmChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 {
		return "", errors.New("vlm returned no choices")
	}
	text := strings.TrimSpace(out.Choices[0].Message.Content)
	if idx := strings.LastIndex(text, "</think>"); idx >= 0 {
		text = strings.TrimSpace(text[idx+len("</think>"):])
	}
	return text, nil
}

type runArtifactCaptionResponse struct {
	Caption string `json:"caption"`
	Enabled bool   `json:"enabled"`
}

// handleRunArtifactCaption returns a calm academic caption for a run-output figure,
// generating (and caching) it lazily on first view. Always 200 with a (possibly
// empty) caption so the frontend never has to special-case failures.
func (deps ServerDeps) handleRunArtifactCaption(w http.ResponseWriter, r *http.Request) {
	if !deps.ready(w) {
		return
	}
	enabled := deps.captioner != nil && deps.captioner.enabled
	if !enabled || strings.TrimSpace(deps.ArtifactRoot) == "" {
		writeJSON(w, http.StatusOK, runArtifactCaptionResponse{Caption: "", Enabled: false})
		return
	}
	runID := chi.URLParam(r, "run_id")
	principal := deps.principalFromRequest(r, "")
	path := strings.TrimSpace(r.URL.Query().Get("path"))
	if path == "" {
		writeError(w, http.StatusBadRequest, errors.New("path query parameter is required"))
		return
	}
	artifacts, err := deps.Store.ListRunArtifactsForUser(r.Context(), runID, principal.UserID, 5000)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	for _, artifact := range artifacts {
		if !artifactPathMatches(artifact, path) {
			continue
		}
		resolved, rerr := resolveArtifactDownloadPath(deps.ArtifactRoot, artifact)
		if rerr != nil {
			writeJSON(w, http.StatusOK, runArtifactCaptionResponse{Caption: "", Enabled: true})
			return
		}
		caption, cerr := deps.captioner.captionForFile(r.Context(), resolved)
		if cerr != nil {
			// VLM unreachable / decode failed: degrade to no caption, never error out.
			writeJSON(w, http.StatusOK, runArtifactCaptionResponse{Caption: "", Enabled: true})
			return
		}
		writeJSON(w, http.StatusOK, runArtifactCaptionResponse{Caption: caption, Enabled: true})
		return
	}
	writeError(w, http.StatusNotFound, errors.New("artifact path was not found for run"))
}
