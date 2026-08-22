package httpapi

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/go-chi/chi/v5"
)

const (
	videoExportSchema             = "ultra.image-video-export-manifest.v1"
	videoExportRecipeSchema       = "ultra.image-video-export-recipe.v1"
	videoExportRendererRevision   = "ultra.slice-video-renderer.v1"
	videoExportFPS                = 24
	videoExportPreviewFrameLimit  = 240
	videoExportCompleteFrameLimit = 1200
	videoExportMarkerMaxBytes     = 1 << 20
	videoExportRequeueAfter       = 10 * time.Minute
)

var videoExportRenderIDPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

type videoExportCreateRequest struct {
	Mode                      string   `json:"mode"`
	Profile                   string   `json:"profile"`
	FixedZ                    int      `json:"fixed_z"`
	FixedT                    int      `json:"fixed_t"`
	Channels                  []int    `json:"channels"`
	ChannelColors             []string `json:"channel_colors,omitempty"`
	Enhancement               string   `json:"enhancement,omitempty"`
	Negative                  bool     `json:"negative,omitempty"`
	ScalarRenderMode          string   `json:"scalar_render_mode"`
	ScalarThresholdValue      *float64 `json:"scalar_threshold_value,omitempty"`
	ScalarThresholdForeground string   `json:"scalar_threshold_foreground,omitempty"`
}

type videoExportSourceIdentity struct {
	SHA256    string `json:"sha256"`
	SizeBytes int64  `json:"size_bytes"`
}

type videoExportAxes struct {
	T int `json:"T"`
	C int `json:"C"`
	Z int `json:"Z"`
}

type videoExportRecipe struct {
	Schema                    string                    `json:"schema"`
	RendererRevision          string                    `json:"renderer_revision"`
	ResourceID                string                    `json:"resource_id"`
	Source                    videoExportSourceIdentity `json:"source"`
	Axes                      videoExportAxes           `json:"axes"`
	Mode                      string                    `json:"mode"`
	Profile                   string                    `json:"profile"`
	FPS                       int                       `json:"fps"`
	SourceFrameCount          int                       `json:"source_frame_count"`
	FrameIndices              []int                     `json:"frame_indices"`
	FixedZ                    int                       `json:"fixed_z"`
	FixedT                    int                       `json:"fixed_t"`
	StrictScalarSlice         bool                      `json:"strict_scalar_slice"`
	Channels                  []int                     `json:"channels"`
	ChannelColors             []string                  `json:"channel_colors"`
	Enhancement               string                    `json:"enhancement"`
	Negative                  bool                      `json:"negative"`
	ScalarRenderMode          string                    `json:"scalar_render_mode"`
	ScalarThresholdValue      *float64                  `json:"scalar_threshold_value"`
	ScalarThresholdForeground string                    `json:"scalar_threshold_foreground"`
	FullResolution            bool                      `json:"full_resolution"`
	MaxFrameEdge              int                       `json:"max_frame_edge"`
}

type videoExportArtifact struct {
	Basename   string `json:"basename"`
	SHA256     string `json:"sha256"`
	SizeBytes  int64  `json:"size_bytes"`
	MediaType  string `json:"media_type"`
	Width      int    `json:"width"`
	Height     int    `json:"height"`
	FrameCount int    `json:"frame_count"`
	FPS        int    `json:"fps"`
}

type videoExportManifest struct {
	Schema    string                    `json:"schema"`
	RenderID  string                    `json:"render_id"`
	CreatedAt string                    `json:"created_at"`
	Source    videoExportSourceIdentity `json:"source"`
	Recipe    videoExportRecipe         `json:"recipe"`
	Artifact  videoExportArtifact       `json:"artifact"`
}

type videoExportMarker struct {
	Schema          string                    `json:"schema"`
	RenderID        string                    `json:"render_id"`
	ResourceID      string                    `json:"resource_id"`
	Source          videoExportSourceIdentity `json:"source"`
	Mode            string                    `json:"mode"`
	Profile         string                    `json:"profile"`
	SourceFrames    int                       `json:"source_frame_count"`
	FramesTotal     int                       `json:"frames_total"`
	FramesCompleted int                       `json:"frames_completed"`
	UpdatedAt       string                    `json:"updated_at"`
	Code            string                    `json:"code,omitempty"`
}

type videoExportResponse struct {
	RenderID        string `json:"render_id"`
	Status          string `json:"status"`
	Mode            string `json:"mode"`
	Profile         string `json:"profile"`
	FPS             int    `json:"fps"`
	SourceFrames    int    `json:"source_frame_count"`
	FramesTotal     int    `json:"frames_total"`
	FramesCompleted int    `json:"frames_completed"`
	Sampled         bool   `json:"sampled"`
	DownloadURL     string `json:"download_url,omitempty"`
	Filename        string `json:"filename,omitempty"`
	ErrorCode       string `json:"error_code,omitempty"`
}

func videoExportBase(resourceID, renderID string) string {
	return resourceID + "__video." + renderID
}

func videoExportManifestName(resourceID, renderID string) string {
	return videoExportBase(resourceID, renderID) + ".manifest.json"
}

func videoExportArtifactName(resourceID, renderID string) string {
	return videoExportBase(resourceID, renderID) + ".mp4"
}

func videoExportMarkerName(resourceID, renderID, state string) string {
	return videoExportBase(resourceID, renderID) + "." + state + ".json"
}

func uniformlySampledFrameIndices(sourceCount, limit int) []int {
	if sourceCount <= 0 || limit <= 0 {
		return nil
	}
	count := min(sourceCount, limit)
	indices := make([]int, count)
	if count == 1 {
		return indices
	}
	for index := range indices {
		indices[index] = int(math.Round(float64(index) * float64(sourceCount-1) / float64(count-1)))
	}
	return indices
}

func completeFrameIndices(sourceCount int) []int {
	indices := make([]int, max(0, sourceCount))
	for index := range indices {
		indices[index] = index
	}
	return indices
}

func decodeStrictVideoExportRequest(w http.ResponseWriter, r *http.Request, target *videoExportCreateRequest) bool {
	defer r.Body.Close()
	reader := http.MaxBytesReader(w, r.Body, 64<<10)
	data, err := io.ReadAll(reader)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return false
	}
	strict := json.NewDecoder(bytes.NewReader(data))
	strict.UseNumber()
	if err := consumeStrictJSONValue(strict); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return false
	}
	if _, err := strict.Token(); err != io.EOF {
		writeError(w, http.StatusBadRequest, errors.New("request body must contain one JSON object"))
		return false
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return false
	}
	return true
}

func validateVideoExportChannels(channels []int, colors []string, channelCount int) error {
	if len(channels) < 1 || len(channels) > maxCompositeImageChannels {
		return fmt.Errorf("select between 1 and %d channels", maxCompositeImageChannels)
	}
	seen := map[int]struct{}{}
	for _, channel := range channels {
		if channel < 0 || channel >= channelCount {
			return errors.New("channel selection is out of range")
		}
		if _, duplicate := seen[channel]; duplicate {
			return errors.New("channel selection contains duplicates")
		}
		seen[channel] = struct{}{}
	}
	if len(colors) != 0 && len(colors) != len(channels) {
		return errors.New("channel colors must match the selected channel order")
	}
	for _, color := range colors {
		if len(color) != 7 || color[0] != '#' {
			return errors.New("channel colors must be #RRGGBB")
		}
		for _, char := range color[1:] {
			if !((char >= '0' && char <= '9') || (char >= 'a' && char <= 'f') || (char >= 'A' && char <= 'F')) {
				return errors.New("channel colors must be #RRGGBB")
			}
		}
	}
	return nil
}

func canonicalVideoExportJSON(value any) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	// Match the worker's UTF-8 canonicalization. The recipe hash is an
	// interoperable identity, not Go's default HTML-safe presentation JSON.
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buffer.Bytes(), []byte{'\n'}), nil
}

func (deps ServerDeps) resolveVideoExportAxes(ctx context.Context, authorization uploadServingAuthorization, sourcePath string) (videoExportAxes, error) {
	if t, c, z, ok := catalogImageSelectorLimits(authorization.record); ok {
		return videoExportAxes{T: t, C: c, Z: z}, nil
	}
	if isNiftiUpload(authorization.record.OriginalName, authorization.record.ContentType) {
		geometry, err := readNiftiHeaderGeometry(sourcePath)
		if err != nil {
			return videoExportAxes{}, err
		}
		return videoExportAxes{T: geometry.timeCount, C: geometry.channelCount, Z: geometry.depth}, nil
	}
	var info map[string]any
	var err error
	if deps.servesViaNgff(authorization.record, sourcePath) {
		info, err = deps.ngffDeps().cachedImageServiceViewerInfo(ctx, sourcePath)
	} else {
		info, _, _, _, err = deps.sourceImageServiceViewerInfo(ctx, sourcePath)
	}
	if err != nil {
		return videoExportAxes{}, err
	}
	t, c, z, ok := sourceViewerAxes(info)
	if !ok {
		return videoExportAxes{}, errMalformedImageViewerAxes
	}
	return videoExportAxes{T: t, C: c, Z: z}, nil
}

func canonicalVideoExportRecipe(record resourceRecord, axes videoExportAxes, request videoExportCreateRequest) (videoExportRecipe, string, error) {
	request.Mode = strings.ToLower(strings.TrimSpace(request.Mode))
	request.Profile = strings.ToLower(strings.TrimSpace(request.Profile))
	request.ScalarRenderMode = strings.ToLower(strings.TrimSpace(request.ScalarRenderMode))
	request.ScalarThresholdForeground = strings.ToLower(strings.TrimSpace(request.ScalarThresholdForeground))
	if request.Mode != "z_sweep" && request.Mode != "time_series" {
		return videoExportRecipe{}, "", errors.New("video mode must be z_sweep or time_series")
	}
	if request.Profile != "preview" && request.Profile != "complete" {
		return videoExportRecipe{}, "", errors.New("video profile must be preview or complete")
	}
	if request.FixedZ < 0 || request.FixedZ >= axes.Z || request.FixedT < 0 || request.FixedT >= axes.T {
		return videoExportRecipe{}, "", errors.New("fixed slice or time index is out of range")
	}
	if err := validateVideoExportChannels(request.Channels, request.ChannelColors, axes.C); err != nil {
		return videoExportRecipe{}, "", err
	}
	for index := range request.ChannelColors {
		request.ChannelColors[index] = strings.ToLower(request.ChannelColors[index])
	}
	strictScalar := isNiftiUpload(record.OriginalName, record.ContentType)
	if strictScalar && len(request.Channels) != 1 {
		return videoExportRecipe{}, "", errors.New("this scalar image requires exactly one channel")
	}
	if request.ScalarRenderMode == "" {
		request.ScalarRenderMode = "intensity"
	}
	if request.ScalarRenderMode != "intensity" && request.ScalarRenderMode != "mask" {
		return videoExportRecipe{}, "", errors.New("scalar render mode must be intensity or mask")
	}
	if request.ScalarRenderMode == "mask" {
		if strictScalar {
			return videoExportRecipe{}, "", errors.New("mask video export is unsupported for NIfTI sources")
		}
		if request.ScalarThresholdValue == nil || math.IsNaN(*request.ScalarThresholdValue) || math.IsInf(*request.ScalarThresholdValue, 0) {
			return videoExportRecipe{}, "", errors.New("mask video export requires a finite threshold")
		}
		if request.ScalarThresholdForeground != "above" {
			return videoExportRecipe{}, "", errors.New("mask video foreground must be above")
		}
	} else {
		request.ScalarThresholdValue = nil
		request.ScalarThresholdForeground = ""
	}
	if len(request.Enhancement) > 128 {
		return videoExportRecipe{}, "", errors.New("enhancement selector is too long")
	}
	if !strictScalar {
		// Generic scientific planes use the image service's stable, source-wide
		// calibration. These fields are NIfTI-only and must not fork identical exports.
		request.Enhancement = ""
		request.Negative = false
	}
	sourceFrameCount := axes.Z
	if request.Mode == "time_series" {
		sourceFrameCount = axes.T
	}
	if sourceFrameCount < 2 {
		return videoExportRecipe{}, "", errors.New("the selected video axis has fewer than two frames")
	}
	var frameIndices []int
	if request.Profile == "complete" {
		if sourceFrameCount > videoExportCompleteFrameLimit {
			return videoExportRecipe{}, "", fmt.Errorf("complete video exports are limited to %d frames", videoExportCompleteFrameLimit)
		}
		frameIndices = completeFrameIndices(sourceFrameCount)
	} else {
		frameIndices = uniformlySampledFrameIndices(sourceFrameCount, videoExportPreviewFrameLimit)
	}
	sourceSHA := strings.ToLower(strings.TrimSpace(record.SHA256))
	if !lowercaseSHA256Pattern.MatchString(sourceSHA) || record.SizeBytes < 0 {
		return videoExportRecipe{}, "", errors.New("resource has no immutable source identity for video export")
	}
	recipe := videoExportRecipe{
		Schema:                    videoExportRecipeSchema,
		RendererRevision:          videoExportRendererRevision,
		ResourceID:                record.FileID,
		Source:                    videoExportSourceIdentity{SHA256: sourceSHA, SizeBytes: record.SizeBytes},
		Axes:                      axes,
		Mode:                      request.Mode,
		Profile:                   request.Profile,
		FPS:                       videoExportFPS,
		SourceFrameCount:          sourceFrameCount,
		FrameIndices:              frameIndices,
		FixedZ:                    request.FixedZ,
		FixedT:                    request.FixedT,
		StrictScalarSlice:         strictScalar,
		Channels:                  append([]int(nil), request.Channels...),
		ChannelColors:             append([]string(nil), request.ChannelColors...),
		Enhancement:               strings.TrimSpace(request.Enhancement),
		Negative:                  request.Negative,
		ScalarRenderMode:          request.ScalarRenderMode,
		ScalarThresholdValue:      request.ScalarThresholdValue,
		ScalarThresholdForeground: request.ScalarThresholdForeground,
		FullResolution:            false,
		MaxFrameEdge:              1024,
	}
	canonical, err := canonicalVideoExportJSON(recipe)
	if err != nil {
		return videoExportRecipe{}, "", err
	}
	digest := sha256.Sum256(canonical)
	return recipe, hex.EncodeToString(digest[:]), nil
}

func videoExportResponseForRecipe(renderID, status string, recipe videoExportRecipe) videoExportResponse {
	return videoExportResponse{
		RenderID:     renderID,
		Status:       status,
		Mode:         recipe.Mode,
		Profile:      recipe.Profile,
		FPS:          recipe.FPS,
		SourceFrames: recipe.SourceFrameCount,
		FramesTotal:  len(recipe.FrameIndices),
		Sampled:      len(recipe.FrameIndices) < recipe.SourceFrameCount,
	}
}

func atomicWriteVideoExportJSON(path string, value any) error {
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return err
	}
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	temporary, err := os.CreateTemp(directory, "."+filepath.Base(path)+".tmp-")
	if err != nil {
		return err
	}
	temporaryName := temporary.Name()
	defer os.Remove(temporaryName)
	if err := temporary.Chmod(0o600); err != nil {
		temporary.Close()
		return err
	}
	if _, err := temporary.Write(data); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryName, path); err != nil {
		return err
	}
	directoryFile, err := os.Open(directory)
	if err != nil {
		return err
	}
	defer directoryFile.Close()
	return directoryFile.Sync()
}

func readStrictVideoExportJSON(path string, target any) error {
	info, err := regularFileInfo(path)
	if err != nil {
		return err
	}
	if info.Size() < 2 || info.Size() > videoExportMarkerMaxBytes {
		return errors.New("video export record has an invalid size")
	}
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	data, readErr := io.ReadAll(io.LimitReader(file, videoExportMarkerMaxBytes+1))
	closeErr := file.Close()
	if readErr != nil || closeErr != nil {
		return errors.Join(readErr, closeErr)
	}
	strict := json.NewDecoder(bytes.NewReader(data))
	strict.UseNumber()
	if err := consumeStrictJSONValue(strict); err != nil {
		return err
	}
	if _, err := strict.Token(); err != io.EOF {
		return errors.New("video export record contains trailing JSON")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	return decoder.Decode(target)
}

func validVideoExportManifest(manifest videoExportManifest, record resourceRecord, renderID string) bool {
	if manifest.Schema != videoExportSchema || manifest.RenderID != renderID || manifest.Recipe.Schema != videoExportRecipeSchema || manifest.Recipe.RendererRevision != videoExportRendererRevision {
		return false
	}
	sourceSHA := strings.ToLower(strings.TrimSpace(record.SHA256))
	if manifest.Source.SHA256 != sourceSHA || manifest.Source.SizeBytes != record.SizeBytes || manifest.Recipe.Source != manifest.Source || manifest.Recipe.ResourceID != record.FileID {
		return false
	}
	if manifest.Artifact.Basename != videoExportArtifactName(record.FileID, renderID) || filepath.Base(manifest.Artifact.Basename) != manifest.Artifact.Basename {
		return false
	}
	rebuilt, rebuiltID, err := canonicalVideoExportRecipe(record, manifest.Recipe.Axes, videoExportCreateRequest{
		Mode:                      manifest.Recipe.Mode,
		Profile:                   manifest.Recipe.Profile,
		FixedZ:                    manifest.Recipe.FixedZ,
		FixedT:                    manifest.Recipe.FixedT,
		Channels:                  append([]int(nil), manifest.Recipe.Channels...),
		ChannelColors:             append([]string(nil), manifest.Recipe.ChannelColors...),
		Enhancement:               manifest.Recipe.Enhancement,
		Negative:                  manifest.Recipe.Negative,
		ScalarRenderMode:          manifest.Recipe.ScalarRenderMode,
		ScalarThresholdValue:      manifest.Recipe.ScalarThresholdValue,
		ScalarThresholdForeground: manifest.Recipe.ScalarThresholdForeground,
	})
	if err != nil || rebuiltID != renderID || !reflect.DeepEqual(rebuilt, manifest.Recipe) {
		return false
	}
	return manifest.Artifact.MediaType == "video/mp4" && manifest.Artifact.SizeBytes > 0 && manifest.Artifact.Width > 0 && manifest.Artifact.Height > 0 && manifest.Artifact.FrameCount == len(manifest.Recipe.FrameIndices) && manifest.Artifact.FPS == videoExportFPS && lowercaseSHA256Pattern.MatchString(manifest.Artifact.SHA256)
}

func (deps ServerDeps) videoExportStatus(root string, record resourceRecord, renderID string) (videoExportResponse, bool) {
	derived := filepath.Join(root, resourceDerivedDir)
	manifestPath := filepath.Join(derived, videoExportManifestName(record.FileID, renderID))
	var manifest videoExportManifest
	if readStrictVideoExportJSON(manifestPath, &manifest) == nil && validVideoExportManifest(manifest, record, renderID) {
		artifactPath := filepath.Join(derived, manifest.Artifact.Basename)
		if info, err := regularFileInfo(artifactPath); err == nil && info.Size() == manifest.Artifact.SizeBytes && verifiedFileDigest(artifactPath, manifest.Artifact.SHA256, info) {
			response := videoExportResponseForRecipe(renderID, "ready", manifest.Recipe)
			response.FramesCompleted = response.FramesTotal
			response.DownloadURL = "/v2/uploads/" + url.PathEscape(record.FileID) + "/video-exports/" + renderID + "/download"
			response.Filename = strings.TrimSuffix(filepath.Base(record.OriginalName), filepath.Ext(record.OriginalName)) + "-" + strings.ReplaceAll(manifest.Recipe.Mode, "_", "-") + ".mp4"
			return response, true
		}
	}
	for _, state := range []string{"failed", "progress", "queued"} {
		var marker videoExportMarker
		if readStrictVideoExportJSON(filepath.Join(derived, videoExportMarkerName(record.FileID, renderID, state)), &marker) != nil {
			continue
		}
		if marker.RenderID != renderID || marker.ResourceID != record.FileID || marker.Source.SHA256 != strings.ToLower(strings.TrimSpace(record.SHA256)) || marker.Source.SizeBytes != record.SizeBytes || marker.FramesTotal < 1 || marker.SourceFrames < marker.FramesTotal {
			continue
		}
		response := videoExportResponse{
			RenderID:        renderID,
			Status:          state,
			Mode:            marker.Mode,
			Profile:         marker.Profile,
			FPS:             videoExportFPS,
			SourceFrames:    marker.SourceFrames,
			FramesTotal:     marker.FramesTotal,
			FramesCompleted: min(max(0, marker.FramesCompleted), marker.FramesTotal),
			Sampled:         marker.FramesTotal < marker.SourceFrames,
			ErrorCode:       marker.Code,
		}
		return response, true
	}
	return videoExportResponse{}, false
}

func videoExportMarkerFresh(path string, now time.Time) bool {
	info, err := regularFileInfo(path)
	return err == nil && now.Sub(info.ModTime()) < videoExportRequeueAfter
}

func (deps ServerDeps) handleCreateUploadVideoExport(w http.ResponseWriter, r *http.Request) {
	authorization, ok := deps.authorizeUploadServingRequest(w, r)
	if !ok {
		return
	}
	sourcePath, ok := resolveAuthorizedUploadStorage(w, authorization)
	if !ok {
		return
	}
	if deps.DataAgentJobs == nil {
		deps.handleNotConfigured("video export requires the background image job queue")(w, r)
		return
	}
	if !isNiftiUpload(authorization.record.OriginalName, authorization.record.ContentType) {
		if deps.servesViaNgff(authorization.record, sourcePath) {
			if strings.TrimSpace(deps.NgffServiceURL) == "" {
				writeError(w, http.StatusServiceUnavailable, errNgffServiceNotConfigured)
				return
			}
		} else if !deps.imageServiceConfigured() {
			deps.handleNotConfigured("video export requires the configured scientific image service")(w, r)
			return
		}
	}
	var request videoExportCreateRequest
	if !decodeStrictVideoExportRequest(w, r, &request) {
		return
	}
	axes, err := deps.resolveVideoExportAxes(r.Context(), authorization, sourcePath)
	if err != nil {
		writeImageSourceAuthorityError(w, err)
		return
	}
	recipe, renderID, err := canonicalVideoExportRecipe(authorization.record, axes, request)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, err)
		return
	}
	if response, found := deps.videoExportStatus(authorization.root, authorization.record, renderID); found && response.Status == "ready" {
		writeJSON(w, http.StatusOK, response)
		return
	}
	rootHandle, err := os.OpenRoot(authorization.root)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	defer rootHandle.Close()
	lifecycleLock, err := acquireResourceLifecycleLock(r.Context(), rootHandle, authorization.record.FileID, "")
	if err != nil {
		writeError(w, http.StatusConflict, errors.New("resource is no longer available for video export"))
		return
	}
	defer lifecycleLock.release()
	derived := filepath.Join(authorization.root, resourceDerivedDir)
	queuedPath := filepath.Join(derived, videoExportMarkerName(authorization.record.FileID, renderID, "queued"))
	progressPath := filepath.Join(derived, videoExportMarkerName(authorization.record.FileID, renderID, "progress"))
	now := time.Now().UTC()
	if videoExportMarkerFresh(queuedPath, now) || videoExportMarkerFresh(progressPath, now) {
		response := videoExportResponseForRecipe(renderID, "queued", recipe)
		if existing, found := deps.videoExportStatus(authorization.root, authorization.record, renderID); found {
			response = existing
		}
		writeJSON(w, http.StatusAccepted, response)
		return
	}
	for _, state := range []string{"queued", "progress", "failed"} {
		_ = os.Remove(filepath.Join(derived, videoExportMarkerName(authorization.record.FileID, renderID, state)))
	}
	marker := videoExportMarker{
		Schema:       "ultra.image-video-export-status.v1",
		RenderID:     renderID,
		ResourceID:   authorization.record.FileID,
		Source:       recipe.Source,
		Mode:         recipe.Mode,
		Profile:      recipe.Profile,
		SourceFrames: recipe.SourceFrameCount,
		FramesTotal:  len(recipe.FrameIndices),
		UpdatedAt:    now.Format(time.RFC3339Nano),
	}
	if err := atomicWriteVideoExportJSON(queuedPath, marker); err != nil {
		writeError(w, http.StatusInternalServerError, errors.New("video export queue marker could not be published"))
		return
	}
	principal := deps.principalFromRequest(r, "")
	jobID := domain.NewID("vidjob")
	envelope := eventbus.DataAgentJob{
		JobID:         jobID,
		OwnerUserID:   principal.UserID,
		OwnerOrgID:    principal.OrgID,
		JobType:       "image.render_video",
		ResourceIDs:   []string{authorization.record.FileID},
		ResourceCount: 1,
		Metadata: domain.JSONMap{
			"resource_id":       authorization.record.FileID,
			"src_path":          sourcePath,
			"source_sha256":     recipe.Source.SHA256,
			"source_size_bytes": recipe.Source.SizeBytes,
			"render_id":         renderID,
			"recipe":            recipe,
			"output_path":       filepath.Join(derived, videoExportArtifactName(authorization.record.FileID, renderID)),
			"manifest_path":     filepath.Join(derived, videoExportManifestName(authorization.record.FileID, renderID)),
			"queued_path":       queuedPath,
			"progress_path":     progressPath,
			"failed_path":       filepath.Join(derived, videoExportMarkerName(authorization.record.FileID, renderID, "failed")),
			"owner_role":        principal.Role,
		},
	}
	if err := deps.DataAgentJobs.PublishDataAgentJob(r.Context(), envelope); err != nil {
		_ = os.Remove(queuedPath)
		writeError(w, http.StatusBadGateway, fmt.Errorf("failed to enqueue video export: %w", err))
		return
	}
	writeJSON(w, http.StatusAccepted, videoExportResponseForRecipe(renderID, "queued", recipe))
}

func (deps ServerDeps) handleGetUploadVideoExport(w http.ResponseWriter, r *http.Request) {
	authorization, ok := deps.authorizeUploadServingRequest(w, r)
	if !ok {
		return
	}
	renderID := strings.ToLower(strings.TrimSpace(chi.URLParam(r, "render_id")))
	if !videoExportRenderIDPattern.MatchString(renderID) {
		writeError(w, http.StatusUnprocessableEntity, errors.New("invalid video export id"))
		return
	}
	response, found := deps.videoExportStatus(authorization.root, authorization.record, renderID)
	if !found {
		writeError(w, http.StatusNotFound, errors.New("video export not found"))
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func (deps ServerDeps) handleDownloadUploadVideoExport(w http.ResponseWriter, r *http.Request) {
	authorization, ok := deps.authorizeUploadServingRequest(w, r)
	if !ok {
		return
	}
	sourcePath, ok := resolveAuthorizedUploadStorage(w, authorization)
	if !ok {
		return
	}
	renderID := strings.ToLower(strings.TrimSpace(chi.URLParam(r, "render_id")))
	if !videoExportRenderIDPattern.MatchString(renderID) {
		writeError(w, http.StatusUnprocessableEntity, errors.New("invalid video export id"))
		return
	}
	manifestPath := filepath.Join(authorization.root, resourceDerivedDir, videoExportManifestName(authorization.record.FileID, renderID))
	var manifest videoExportManifest
	if readStrictVideoExportJSON(manifestPath, &manifest) != nil || !validVideoExportManifest(manifest, authorization.record, renderID) {
		writeError(w, http.StatusNotFound, errors.New("video export not found"))
		return
	}
	sourceInfo, err := regularFileInfo(sourcePath)
	if err != nil || sourceInfo.Size() != manifest.Source.SizeBytes || !verifiedFileDigest(sourcePath, manifest.Source.SHA256, sourceInfo) {
		writeError(w, http.StatusConflict, errors.New("video export source failed integrity verification"))
		return
	}
	artifactPath := filepath.Join(authorization.root, resourceDerivedDir, manifest.Artifact.Basename)
	info, err := regularFileInfo(artifactPath)
	if err != nil || info.Size() != manifest.Artifact.SizeBytes || !verifiedFileDigest(artifactPath, manifest.Artifact.SHA256, info) {
		writeError(w, http.StatusConflict, errors.New("video export failed integrity verification"))
		return
	}
	file, err := os.Open(artifactPath)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	defer file.Close()
	stem := strings.TrimSuffix(filepath.Base(authorization.record.OriginalName), filepath.Ext(authorization.record.OriginalName))
	filename := stem + "-" + strings.ReplaceAll(manifest.Recipe.Mode, "_", "-") + ".mp4"
	disposition := mime.FormatMediaType("attachment", map[string]string{"filename": filename})
	w.Header().Set("Content-Type", "video/mp4")
	w.Header().Set("Content-Disposition", disposition)
	w.Header().Set("Cache-Control", "private, max-age=31536000, immutable")
	http.ServeContent(w, r, filename, info.ModTime(), file)
}

// handleServeUploadVideoFrame is worker-only. It deliberately delegates to the
// same slice handler as Lens so exports cannot drift from the calibrated 2D view.
func (deps ServerDeps) handleServeUploadVideoFrame(w http.ResponseWriter, r *http.Request) {
	if deps.workerRequestAuth(r) != workerAuthValid || strings.TrimSpace(r.Header.Get("X-Ultra-User-Id")) == "" {
		writeError(w, http.StatusUnauthorized, errors.New("valid worker authentication is required"))
		return
	}
	deps.handleServeUploadSliceService(w, r)
}
