package httpapi

import (
	"bytes"
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/sync/singleflight"
)

const (
	derivedPyramidManifestSchemaV1  = "ultra.image-derived-pyramid-manifest.v1"
	derivedPyramidManifestSchema    = "ultra.image-derived-pyramid-manifest.v2"
	derivedPyramidConversionSchema  = "ultra.image-pyramid.v1"
	derivedPyramidProducerRevision  = "ultra-deepagents.image-pyramid-publisher.v1"
	derivedPyramidConverterRevision = "libbioimage.imgcnv-pyramid.v1"
	maxDerivedPyramidManifestBytes  = 1 << 20
	maxDerivativeCacheEntries       = 2048
)

var lowercaseSHA256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
var derivativeForceIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)

type derivativeManifest struct {
	Schema             string                     `json:"schema"`
	ConversionContract string                     `json:"conversion_contract"`
	Request            *derivativeManifestRequest `json:"request,omitempty"`
	ConversionSpec     derivativePublicationSpec  `json:"conversion_spec"`
	Producer           derivativeProducer         `json:"producer"`
	Source             derivativeManifestSource   `json:"source"`
	Semantics          derivativeSemantics        `json:"semantics"`
	Artifact           derivativeArtifact         `json:"artifact"`
	Capabilities       derivativeCapabilities     `json:"capabilities"`
}

type derivativeManifestRequest struct {
	ForceID string `json:"force_id"`
}

type derivativeManifestSource struct {
	SHA256    string `json:"sha256"`
	SizeBytes int64  `json:"size_bytes"`
}

type derivativeSourceStat struct {
	Device    int64 `json:"device"`
	Inode     int64 `json:"inode"`
	SizeBytes int64 `json:"size_bytes"`
	MtimeNS   int64 `json:"mtime_ns"`
	CtimeNS   int64 `json:"ctime_ns"`
}

type derivativeConversionSpec struct {
	TileSize    int    `json:"tile_size"`
	Compression string `json:"compression"`
	Layout      string `json:"layout"`
	Format      string `json:"fmt"`
}

type derivativePublicationSpec struct {
	Requested         derivativeConversionSpec `json:"requested"`
	Effective         derivativeConversionSpec `json:"effective"`
	ProducerRevision  string                   `json:"producer_revision"`
	ConverterRevision string                   `json:"converter_revision"`
}

type derivativeProducer struct {
	Reader      string `json:"reader"`
	SeriesCount int    `json:"series_count"`
	SeriesIndex int    `json:"series_index"`
	SeriesName  string `json:"series_name"`
}

type derivativeSemantics struct {
	DimsOrder string                    `json:"dims_order"`
	AxisSizes derivativeAxisSizes       `json:"axis_sizes"`
	DType     string                    `json:"dtype"`
	Scene     derivativeScene           `json:"scene"`
	Channels  []derivativeChannel       `json:"channels"`
	Spacing   derivativePhysicalSpacing `json:"spacing"`
	Display   derivativeDisplay         `json:"display"`
}

type derivativeAxisSizes struct {
	T int `json:"T"`
	C int `json:"C"`
	Z int `json:"Z"`
	Y int `json:"Y"`
	X int `json:"X"`
}

type derivativeScene struct {
	Count int     `json:"count"`
	ID    *string `json:"id"`
	Index int     `json:"index"`
}

type derivativeDisplay struct {
	RenderPolicy    string `json:"render_policy"`
	ChannelMode     string `json:"channel_mode"`
	DefaultChannels []int  `json:"default_channels"`
}

type derivativeChannel struct {
	Index int    `json:"index"`
	Name  string `json:"name"`
}

type derivativePhysicalSpacing struct {
	X derivativeSpacing `json:"x"`
	Y derivativeSpacing `json:"y"`
	Z derivativeSpacing `json:"z"`
}

type derivativeSpacing struct {
	Value float64 `json:"value"`
	Unit  string  `json:"unit"`
}

type derivativeArtifact struct {
	Basename  string `json:"basename"`
	SizeBytes int64  `json:"size_bytes"`
	SHA256    string `json:"sha256"`
}

type derivativeCapabilities struct {
	Atlas           bool `json:"atlas"`
	AtlasT          bool `json:"atlas_t"`
	LUT             bool `json:"lut"`
	OrderedChannels bool `json:"ordered_channels"`
	Slice           bool `json:"slice"`
	Thumbnail       bool `json:"thumbnail"`
	Tile            bool `json:"tile"`
	TileT           bool `json:"tile_t"`
	TileZ           bool `json:"tile_z"`
}

type derivativeUse struct {
	capability      string
	requireT        bool
	requireZ        bool
	requireChannels bool
	requireLUT      bool
}

func (cap derivativeCapabilities) supports(use derivativeUse) bool {
	switch use.capability {
	case "tile":
		if !cap.Tile || use.requireT && !cap.TileT || use.requireZ && !cap.TileZ {
			return false
		}
	case "slice":
		if !cap.Slice {
			return false
		}
	case "atlas":
		if !cap.Atlas || use.requireT && !cap.AtlasT {
			return false
		}
	case "thumbnail":
		if !cap.Thumbnail {
			return false
		}
	default:
		return false
	}
	return (!use.requireChannels || cap.OrderedChannels) && (!use.requireLUT || cap.LUT)
}

func derivedPyramidManifestName(fileID string) string {
	return fileID + "__pyramid.manifest.json"
}

func derivedPyramidManifestPath(root, fileID string) string {
	return filepath.Join(root, "derived", derivedPyramidManifestName(fileID))
}

func exactJSONKeys(value any, expected ...string) error {
	object, ok := value.(map[string]any)
	if !ok || len(object) != len(expected) {
		return errors.New("manifest object has missing or unknown fields")
	}
	for _, key := range expected {
		if _, present := object[key]; !present {
			return errors.New("manifest object has missing or unknown fields")
		}
	}
	return nil
}

// consumeStrictJSONValue rejects duplicate object keys, a case encoding/json's
// normal struct decoder intentionally accepts. It also consumes exactly one value.
func consumeStrictJSONValue(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delim, isDelim := token.(json.Delim)
	if !isDelim {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]struct{}{}
		for decoder.More() {
			keyToken, keyErr := decoder.Token()
			if keyErr != nil {
				return keyErr
			}
			key, ok := keyToken.(string)
			if !ok {
				return errors.New("manifest object key is not a string")
			}
			if _, duplicate := seen[key]; duplicate {
				return fmt.Errorf("duplicate manifest key %q", key)
			}
			seen[key] = struct{}{}
			if err := consumeStrictJSONValue(decoder); err != nil {
				return err
			}
		}
		closing, closeErr := decoder.Token()
		if closeErr != nil || closing != json.Delim('}') {
			return errors.New("malformed manifest object")
		}
	case '[':
		for decoder.More() {
			if err := consumeStrictJSONValue(decoder); err != nil {
				return err
			}
		}
		closing, closeErr := decoder.Token()
		if closeErr != nil || closing != json.Delim(']') {
			return errors.New("malformed manifest array")
		}
	default:
		return errors.New("unexpected manifest delimiter")
	}
	return nil
}

func validateDerivativeManifestJSON(data []byte) error {
	strict := json.NewDecoder(bytes.NewReader(data))
	strict.UseNumber()
	if err := consumeStrictJSONValue(strict); err != nil {
		return err
	}
	if _, err := strict.Token(); err != io.EOF {
		if err == nil {
			return errors.New("manifest contains trailing JSON")
		}
		return err
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	schema, ok := raw["schema"].(string)
	if !ok {
		return errors.New("manifest schema is invalid")
	}
	rootKeys := []string{"schema", "conversion_contract", "conversion_spec", "producer", "source", "semantics", "artifact", "capabilities"}
	if schema == derivedPyramidManifestSchema {
		rootKeys = append(rootKeys, "request")
	} else if schema != derivedPyramidManifestSchemaV1 {
		return errors.New("manifest schema is unsupported")
	}
	if err := exactJSONKeys(raw, rootKeys...); err != nil {
		return err
	}
	if schema == derivedPyramidManifestSchema {
		if err := exactJSONKeys(raw["request"], "force_id"); err != nil {
			return err
		}
		request := raw["request"].(map[string]any)
		forceID, forceIDOK := request["force_id"].(string)
		if !forceIDOK || forceID != "" && !derivativeForceIDPattern.MatchString(forceID) {
			return errors.New("manifest force request id is invalid")
		}
	}
	if err := exactJSONKeys(raw["conversion_spec"], "requested", "effective", "producer_revision", "converter_revision"); err != nil {
		return err
	}
	conversionSpec := raw["conversion_spec"].(map[string]any)
	if err := exactJSONKeys(conversionSpec["requested"], "tile_size", "compression", "layout", "fmt"); err != nil {
		return err
	}
	if err := exactJSONKeys(conversionSpec["effective"], "tile_size", "compression", "layout", "fmt"); err != nil {
		return err
	}
	if err := exactJSONKeys(raw["producer"], "reader", "series_count", "series_index", "series_name"); err != nil {
		return err
	}
	source, _ := raw["source"].(map[string]any)
	semantics, _ := raw["semantics"].(map[string]any)
	artifact, _ := raw["artifact"].(map[string]any)
	capabilities, _ := raw["capabilities"].(map[string]any)
	if err := exactJSONKeys(source, "sha256", "size_bytes"); err != nil {
		return err
	}
	if err := exactJSONKeys(semantics, "dims_order", "axis_sizes", "dtype", "scene", "channels", "spacing", "display"); err != nil {
		return err
	}
	if err := exactJSONKeys(semantics["axis_sizes"], "T", "C", "Z", "Y", "X"); err != nil {
		return err
	}
	if err := exactJSONKeys(semantics["scene"], "count", "id", "index"); err != nil {
		return err
	}
	if err := exactJSONKeys(semantics["spacing"], "x", "y", "z"); err != nil {
		return err
	}
	if err := exactJSONKeys(semantics["display"], "render_policy", "channel_mode", "default_channels"); err != nil {
		return err
	}
	spacing := semantics["spacing"].(map[string]any)
	for _, axis := range []string{"x", "y", "z"} {
		if err := exactJSONKeys(spacing[axis], "value", "unit"); err != nil {
			return err
		}
	}
	channels, ok := semantics["channels"].([]any)
	if !ok {
		return errors.New("manifest channels is not an array")
	}
	for _, channel := range channels {
		if err := exactJSONKeys(channel, "index", "name"); err != nil {
			return err
		}
	}
	if err := exactJSONKeys(artifact, "basename", "size_bytes", "sha256"); err != nil {
		return err
	}
	return exactJSONKeys(
		capabilities,
		"atlas", "atlas_t", "lut", "ordered_channels", "slice", "thumbnail", "tile", "tile_t", "tile_z",
	)
}

func validateDerivativeSemantics(semantics derivativeSemantics) error {
	if strings.TrimSpace(semantics.DimsOrder) == "" || strings.TrimSpace(semantics.DType) == "" {
		return errors.New("manifest semantic identity is incomplete")
	}
	axes := semantics.AxisSizes
	if axes.T < 1 || axes.C < 1 || axes.Z < 1 || axes.Y < 1 || axes.X < 1 || len(semantics.Channels) != axes.C {
		return errors.New("manifest semantic axes or channel count is invalid")
	}
	for index, channel := range semantics.Channels {
		if channel.Index != index {
			return errors.New("manifest channels are not in canonical order")
		}
	}
	for _, spacing := range []derivativeSpacing{semantics.Spacing.X, semantics.Spacing.Y, semantics.Spacing.Z} {
		if math.IsNaN(spacing.Value) || math.IsInf(spacing.Value, 0) || spacing.Value <= 0 || strings.TrimSpace(spacing.Unit) == "" {
			return errors.New("manifest physical spacing is invalid")
		}
	}
	if semantics.Scene.Count < 1 || semantics.Scene.Index < 0 || semantics.Scene.Index >= semantics.Scene.Count || semantics.Scene.ID != nil && *semantics.Scene.ID == "" {
		return errors.New("manifest scene identity is invalid")
	}
	if strings.TrimSpace(semantics.Display.RenderPolicy) == "" || strings.TrimSpace(semantics.Display.ChannelMode) == "" {
		return errors.New("manifest display provenance is invalid")
	}
	seenChannels := map[int]struct{}{}
	for _, channel := range semantics.Display.DefaultChannels {
		if channel < 0 || channel >= axes.C {
			return errors.New("manifest display channel is out of range")
		}
		if _, duplicate := seenChannels[channel]; duplicate {
			return errors.New("manifest display channels contain duplicates")
		}
		seenChannels[channel] = struct{}{}
	}
	return nil
}

func validateDerivativeProduction(manifest derivativeManifest) error {
	if manifest.Schema == derivedPyramidManifestSchema {
		if manifest.Request == nil || manifest.Request.ForceID != "" && !derivativeForceIDPattern.MatchString(manifest.Request.ForceID) {
			return errors.New("manifest force request identity is invalid")
		}
	} else if manifest.Schema != derivedPyramidManifestSchemaV1 || manifest.Request != nil {
		return errors.New("manifest request contract is invalid")
	}
	if manifest.ConversionSpec.Requested != (derivativeConversionSpec{
		TileSize: 512, Compression: "lzw", Layout: "topdirs", Format: "auto",
	}) {
		return errors.New("manifest requested conversion spec is not the control-plane contract")
	}
	effective := manifest.ConversionSpec.Effective
	if effective.TileSize != manifest.ConversionSpec.Requested.TileSize ||
		effective.Compression != manifest.ConversionSpec.Requested.Compression ||
		effective.Layout != manifest.ConversionSpec.Requested.Layout ||
		(effective.Format != "bigtiff" && effective.Format != "ome-bigtiff") {
		return errors.New("manifest effective conversion spec is invalid")
	}
	if manifest.ConversionSpec.ProducerRevision != derivedPyramidProducerRevision || manifest.ConversionSpec.ConverterRevision != derivedPyramidConverterRevision {
		return errors.New("manifest conversion revision is unsupported")
	}
	producer := manifest.Producer
	if strings.TrimSpace(producer.Reader) == "" || producer.SeriesCount < 1 || producer.SeriesIndex < 0 || producer.SeriesIndex >= producer.SeriesCount {
		return errors.New("manifest producer provenance is invalid")
	}
	if producer.SeriesCount != manifest.Semantics.Scene.Count || producer.SeriesIndex != manifest.Semantics.Scene.Index {
		return errors.New("manifest producer does not match selected source scene")
	}
	wantName := ""
	if manifest.Semantics.Scene.ID != nil {
		wantName = *manifest.Semantics.Scene.ID
	}
	if producer.SeriesName != wantName {
		return errors.New("manifest producer series name does not match selected source scene")
	}
	return nil
}

type boundedDerivativeCacheEntry[T any] struct {
	key   string
	value T
}

type boundedDerivativeCache[T any] struct {
	mu         sync.Mutex
	order      *list.List
	entries    map[string]*list.Element
	maxEntries int
}

func newBoundedDerivativeCache[T any](maxEntries int) *boundedDerivativeCache[T] {
	return &boundedDerivativeCache[T]{
		order:      list.New(),
		entries:    make(map[string]*list.Element),
		maxEntries: maxEntries,
	}
}

func (cache *boundedDerivativeCache[T]) get(key string) (T, bool) {
	cache.mu.Lock()
	defer cache.mu.Unlock()
	element, ok := cache.entries[key]
	if !ok {
		var zero T
		return zero, false
	}
	cache.order.MoveToFront(element)
	return element.Value.(*boundedDerivativeCacheEntry[T]).value, true
}

func (cache *boundedDerivativeCache[T]) put(key string, value T) {
	cache.mu.Lock()
	defer cache.mu.Unlock()
	if cache.maxEntries <= 0 {
		return
	}
	if element, ok := cache.entries[key]; ok {
		element.Value.(*boundedDerivativeCacheEntry[T]).value = value
		cache.order.MoveToFront(element)
		return
	}
	element := cache.order.PushFront(&boundedDerivativeCacheEntry[T]{key: key, value: value})
	cache.entries[key] = element
	for len(cache.entries) > cache.maxEntries {
		oldest := cache.order.Back()
		if oldest == nil {
			break
		}
		cache.order.Remove(oldest)
		delete(cache.entries, oldest.Value.(*boundedDerivativeCacheEntry[T]).key)
	}
}

func (cache *boundedDerivativeCache[T]) len() int {
	cache.mu.Lock()
	defer cache.mu.Unlock()
	return len(cache.entries)
}

var derivativeDigestCache = newBoundedDerivativeCache[string](maxDerivativeCacheEntries)

var derivativeDigestFlights singleflight.Group

type derivativeAdmissionEntry struct {
	manifest           derivativeManifest
	artifactPath       string
	artifactGeneration derivativeSourceStat
}

var derivativeAdmissionCache = newBoundedDerivativeCache[derivativeAdmissionEntry](maxDerivativeCacheEntries)

func generationCacheKey(path string, generation derivativeSourceStat) string {
	return fmt.Sprintf(
		"%s:%d:%d:%d:%d:%d",
		path,
		generation.Device,
		generation.Inode,
		generation.SizeBytes,
		generation.MtimeNS,
		generation.CtimeNS,
	)
}

func reflectedStatInt64(value reflect.Value, field string) (int64, bool) {
	if value.Kind() == reflect.Pointer {
		value = value.Elem()
	}
	if !value.IsValid() || value.Kind() != reflect.Struct {
		return 0, false
	}
	member := value.FieldByName(field)
	if !member.IsValid() {
		return 0, false
	}
	switch member.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return member.Int(), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		unsigned := member.Uint()
		if unsigned > math.MaxInt64 {
			return 0, false
		}
		return int64(unsigned), true
	default:
		return 0, false
	}
}

func fileGeneration(info os.FileInfo) (derivativeSourceStat, bool) {
	if info == nil || !info.Mode().IsRegular() || info.Sys() == nil {
		return derivativeSourceStat{}, false
	}
	system := reflect.ValueOf(info.Sys())
	device, deviceOK := reflectedStatInt64(system, "Dev")
	inode, inodeOK := reflectedStatInt64(system, "Ino")
	if system.Kind() == reflect.Pointer {
		system = system.Elem()
	}
	var ctime int64
	var ctimeOK bool
	for _, fieldName := range []string{"Ctim", "Ctimespec"} {
		field := system.FieldByName(fieldName)
		if !field.IsValid() {
			continue
		}
		seconds, secondsOK := reflectedStatInt64(field, "Sec")
		nanoseconds, nanosOK := reflectedStatInt64(field, "Nsec")
		if secondsOK && nanosOK && seconds <= (math.MaxInt64-nanoseconds)/int64(time.Second) {
			ctime = seconds*int64(time.Second) + nanoseconds
			ctimeOK = true
			break
		}
	}
	if !deviceOK || !inodeOK || !ctimeOK {
		return derivativeSourceStat{}, false
	}
	return derivativeSourceStat{
		Device:    device,
		Inode:     inode,
		SizeBytes: info.Size(),
		MtimeNS:   info.ModTime().UnixNano(),
		CtimeNS:   ctime,
	}, true
}

func openRegularNoFollow(path string) (*os.File, derivativeSourceStat, error) {
	before, err := regularFileInfo(path)
	if err != nil {
		return nil, derivativeSourceStat{}, err
	}
	beforeGeneration, ok := fileGeneration(before)
	if !ok {
		return nil, derivativeSourceStat{}, errors.New("file generation identity is unavailable")
	}
	file, err := os.OpenFile(path, os.O_RDONLY|syscall.O_NOFOLLOW, 0)
	if err != nil {
		return nil, derivativeSourceStat{}, err
	}
	opened, err := file.Stat()
	openedGeneration, generationOK := fileGeneration(opened)
	if err != nil || !generationOK || openedGeneration != beforeGeneration {
		_ = file.Close()
		return nil, derivativeSourceStat{}, errors.New("file generation changed while opening")
	}
	return file, openedGeneration, nil
}

func regularFileInfo(path string) (os.FileInfo, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return nil, errors.New("path is not a regular file")
	}
	return info, nil
}

func verifiedFileDigest(path, expected string, initial os.FileInfo) bool {
	initialGeneration, ok := fileGeneration(initial)
	if !ok {
		return false
	}
	cacheKey := fmt.Sprintf(
		"%s:%d:%d:%d:%d:%d",
		path,
		initialGeneration.Device,
		initialGeneration.Inode,
		initialGeneration.SizeBytes,
		initialGeneration.MtimeNS,
		initialGeneration.CtimeNS,
	)
	cached, cachedOK := derivativeDigestCache.get(cacheKey)
	if cachedOK {
		return cached == expected
	}
	value, err, _ := derivativeDigestFlights.Do(cacheKey, func() (any, error) {
		// A concurrent waiter may have populated the cache before this flight began.
		digest, digestOK := derivativeDigestCache.get(cacheKey)
		if digestOK {
			return digest, nil
		}
		file, openedGeneration, openErr := openRegularNoFollow(path)
		if openErr != nil {
			return "", openErr
		}
		if openedGeneration != initialGeneration {
			_ = file.Close()
			return "", errors.New("derivative artifact generation changed while opening")
		}
		hasher := sha256.New()
		_, copyErr := io.Copy(hasher, file)
		closeErr := file.Close()
		after, statErr := regularFileInfo(path)
		afterGeneration, afterOK := fileGeneration(after)
		if copyErr != nil || closeErr != nil || statErr != nil || !afterOK || afterGeneration != initialGeneration {
			return "", errors.New("derivative artifact changed while hashing")
		}
		digest = hex.EncodeToString(hasher.Sum(nil))
		derivativeDigestCache.put(cacheKey, digest)
		return digest, nil
	})
	if err != nil {
		return false
	}
	digest, ok := value.(string)
	return ok && digest == expected
}

func readDerivativeManifest(root string, record resourceRecord, sourcePath string) (derivativeManifest, string, bool) {
	manifestPath := derivedPyramidManifestPath(root, record.FileID)
	manifestInfo, err := regularFileInfo(manifestPath)
	if err != nil || manifestInfo.Size() <= 0 || manifestInfo.Size() > maxDerivedPyramidManifestBytes {
		return derivativeManifest{}, "", false
	}
	manifestGeneration, manifestGenerationOK := fileGeneration(manifestInfo)
	sourceInfo, sourceErr := regularFileInfo(sourcePath)
	sourceGeneration, sourceGenerationOK := fileGeneration(sourceInfo)
	if !manifestGenerationOK || sourceErr != nil || !sourceGenerationOK {
		return derivativeManifest{}, "", false
	}
	admissionKey := strings.Join([]string{
		generationCacheKey(manifestPath, manifestGeneration),
		generationCacheKey(sourcePath, sourceGeneration),
		strings.ToLower(strings.TrimSpace(record.SHA256)),
		strconv.FormatInt(record.SizeBytes, 10),
	}, "|")
	cached, cachedOK := derivativeAdmissionCache.get(admissionKey)
	if cachedOK {
		artifactInfo, artifactErr := regularFileInfo(cached.artifactPath)
		artifactGeneration, artifactOK := fileGeneration(artifactInfo)
		if artifactErr == nil && artifactOK && artifactGeneration == cached.artifactGeneration {
			return cached.manifest, cached.artifactPath, true
		}
	}
	manifestFile, openedManifestGeneration, err := openRegularNoFollow(manifestPath)
	if err != nil || openedManifestGeneration != manifestGeneration {
		return derivativeManifest{}, "", false
	}
	data, readErr := io.ReadAll(io.LimitReader(manifestFile, maxDerivedPyramidManifestBytes+1))
	closeErr := manifestFile.Close()
	manifestAfter, afterErr := regularFileInfo(manifestPath)
	manifestAfterGeneration, manifestAfterOK := fileGeneration(manifestAfter)
	if readErr != nil || closeErr != nil || afterErr != nil || !manifestAfterOK || manifestAfterGeneration != manifestGeneration || int64(len(data)) != manifestInfo.Size() || validateDerivativeManifestJSON(data) != nil {
		return derivativeManifest{}, "", false
	}
	var manifest derivativeManifest
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&manifest); err != nil ||
		(manifest.Schema != derivedPyramidManifestSchema && manifest.Schema != derivedPyramidManifestSchemaV1) ||
		manifest.ConversionContract != derivedPyramidConversionSchema {
		return derivativeManifest{}, "", false
	}
	if validateDerivativeSemantics(manifest.Semantics) != nil || validateDerivativeProduction(manifest) != nil || !lowercaseSHA256Pattern.MatchString(manifest.Source.SHA256) || manifest.Source.SizeBytes < 0 {
		return derivativeManifest{}, "", false
	}
	if manifest.Source.SHA256 != strings.ToLower(strings.TrimSpace(record.SHA256)) || manifest.Source.SizeBytes != record.SizeBytes {
		return derivativeManifest{}, "", false
	}
	if sourceGeneration.SizeBytes != manifest.Source.SizeBytes || !verifiedFileDigest(sourcePath, manifest.Source.SHA256, sourceInfo) {
		return derivativeManifest{}, "", false
	}
	if !lowercaseSHA256Pattern.MatchString(manifest.Artifact.SHA256) || manifest.Artifact.SizeBytes <= 0 {
		return derivativeManifest{}, "", false
	}
	expectedBasename := fmt.Sprintf("%s__pyramid.sha256-%s.tif", record.FileID, manifest.Artifact.SHA256)
	if manifest.Artifact.Basename != expectedBasename || filepath.Base(manifest.Artifact.Basename) != manifest.Artifact.Basename {
		return derivativeManifest{}, "", false
	}
	artifactPath := filepath.Join(root, "derived", manifest.Artifact.Basename)
	artifactInfo, err := regularFileInfo(artifactPath)
	if err != nil || artifactInfo.Size() != manifest.Artifact.SizeBytes || !verifiedFileDigest(artifactPath, manifest.Artifact.SHA256, artifactInfo) {
		return derivativeManifest{}, "", false
	}
	artifactGeneration, artifactGenerationOK := fileGeneration(artifactInfo)
	manifestFinal, err := regularFileInfo(manifestPath)
	manifestFinalGeneration, manifestFinalOK := fileGeneration(manifestFinal)
	if !artifactGenerationOK || err != nil || !manifestFinalOK || manifestFinalGeneration != manifestGeneration {
		return derivativeManifest{}, "", false
	}
	derivativeAdmissionCache.put(admissionKey, derivativeAdmissionEntry{
		manifest: manifest, artifactPath: artifactPath, artifactGeneration: artifactGeneration,
	})
	return manifest, artifactPath, true
}

func viewerDerivativeSemantics(info map[string]any) (derivativeSemantics, bool) {
	axisRaw, ok := info["axis_sizes"].(map[string]any)
	if !ok {
		return derivativeSemantics{}, false
	}
	axisValues := make([]int, 5)
	for index, axis := range []string{"T", "C", "Z", "Y", "X"} {
		value, valid := jsonInt(axisRaw[axis])
		if !valid || value < 1 {
			return derivativeSemantics{}, false
		}
		axisValues[index] = value
	}
	dimsOrder, dimsOK := info["dims_order"].(string)
	dtype, dtypeOK := info["dtype"].(string)
	namesRaw, namesOK := info["channel_names"].([]any)
	spacingRaw, spacingOK := info["physical_spacing"].(map[string]any)
	metadata, metadataOK := info["metadata"].(map[string]any)
	if !dimsOK || dimsOrder == "" || !dtypeOK || dtype == "" || !namesOK || len(namesRaw) != axisValues[1] || !spacingOK || !metadataOK {
		return derivativeSemantics{}, false
	}
	unitsRaw, unitsOK := metadata["spacing_units"].(map[string]any)
	if !unitsOK {
		return derivativeSemantics{}, false
	}
	channels := make([]derivativeChannel, len(namesRaw))
	for index, raw := range namesRaw {
		name, valid := raw.(string)
		if !valid {
			return derivativeSemantics{}, false
		}
		channels[index] = derivativeChannel{Index: index, Name: name}
	}
	spacingValues := make([]derivativeSpacing, 3)
	for index, axis := range []string{"x", "y", "z"} {
		value, valueOK := strictJSONFloat(spacingRaw[axis])
		unit, unitOK := unitsRaw[axis].(string)
		if !valueOK || value <= 0 || !unitOK || unit == "" {
			return derivativeSemantics{}, false
		}
		spacingValues[index] = derivativeSpacing{Value: value, Unit: unit}
	}
	scene := derivativeScene{Count: 1, Index: 0}
	sceneCountRaw, hasSceneCount := info["scene_count"]
	if !hasSceneCount {
		sceneCountRaw, hasSceneCount = metadata["scene_count"]
	}
	if hasSceneCount {
		sceneCount, valid := jsonInt(sceneCountRaw)
		if !valid || sceneCount < 1 {
			return derivativeSemantics{}, false
		}
		scene.Count = sceneCount
	}
	sceneIDRaw, hasSceneID := info["selected_scene_id"]
	if !hasSceneID {
		sceneIDRaw, hasSceneID = metadata["selected_scene_id"]
	}
	if hasSceneID && sceneIDRaw != nil {
		sceneID, valid := sceneIDRaw.(string)
		if !valid || sceneID == "" {
			return derivativeSemantics{}, false
		}
		scene.ID = &sceneID
	}
	sceneIndexRaw, hasSceneIndex := info["selected_scene_index"]
	if !hasSceneIndex {
		sceneIndexRaw, hasSceneIndex = metadata["selected_scene_index"]
	}
	if hasSceneIndex && sceneIndexRaw != nil {
		sceneIndex, valid := jsonInt(sceneIndexRaw)
		if !valid || sceneIndex < 0 || sceneIndex >= scene.Count {
			return derivativeSemantics{}, false
		}
		scene.Index = sceneIndex
	}
	viewer, _ := info["viewer"].(map[string]any)
	renderPolicy, _ := viewer["render_policy"].(string)
	if renderPolicy == "" {
		renderPolicy = "scalar"
	}
	channelMode, _ := viewer["channel_mode"].(string)
	if channelMode == "" {
		channelMode = "single"
		if axisValues[1] > 1 {
			channelMode = "composite"
		}
	}
	defaultChannels := make([]int, 0, min(axisValues[1], 3))
	displayDefaults, _ := info["display_defaults"].(map[string]any)
	if displayDefaults == nil {
		displayDefaults, _ = viewer["display_defaults"].(map[string]any)
	}
	if rawDefaults, present := displayDefaults["channels"]; present {
		rawChannels, ok := rawDefaults.([]any)
		if !ok {
			if typed, typedOK := rawDefaults.([]int); typedOK {
				rawChannels = make([]any, len(typed))
				for index, value := range typed {
					rawChannels[index] = value
				}
			} else {
				return derivativeSemantics{}, false
			}
		}
		seen := map[int]struct{}{}
		for _, raw := range rawChannels {
			channel, valid := jsonInt(raw)
			if !valid || channel < 0 || channel >= axisValues[1] {
				return derivativeSemantics{}, false
			}
			if _, duplicate := seen[channel]; duplicate {
				return derivativeSemantics{}, false
			}
			seen[channel] = struct{}{}
			defaultChannels = append(defaultChannels, channel)
		}
	} else {
		for channel := 0; channel < min(axisValues[1], 3); channel++ {
			defaultChannels = append(defaultChannels, channel)
		}
	}
	semantics := derivativeSemantics{
		DimsOrder: dimsOrder,
		AxisSizes: derivativeAxisSizes{T: axisValues[0], C: axisValues[1], Z: axisValues[2], Y: axisValues[3], X: axisValues[4]},
		DType:     dtype,
		Scene:     scene,
		Channels:  channels,
		Spacing:   derivativePhysicalSpacing{X: spacingValues[0], Y: spacingValues[1], Z: spacingValues[2]},
		Display: derivativeDisplay{
			RenderPolicy: renderPolicy, ChannelMode: channelMode, DefaultChannels: defaultChannels,
		},
	}
	return semantics, validateDerivativeSemantics(semantics) == nil
}

func strictJSONFloat(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, !math.IsNaN(typed) && !math.IsInf(typed, 0)
	case float32:
		converted := float64(typed)
		return converted, !math.IsNaN(converted) && !math.IsInf(converted, 0)
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case json.Number:
		converted, err := typed.Float64()
		return converted, err == nil && !math.IsNaN(converted) && !math.IsInf(converted, 0)
	default:
		return 0, false
	}
}

func derivativeSemanticsMatch(info map[string]any, expected derivativeSemantics) bool {
	actual, ok := viewerDerivativeSemantics(info)
	return ok && reflect.DeepEqual(actual, expected)
}

func derivativeArtifactSemanticsMatch(info map[string]any, expected derivativeSemantics) bool {
	actual, ok := viewerDerivativeSemantics(info)
	if !ok {
		return false
	}
	actual.Scene = derivativeScene{}
	expected.Scene = derivativeScene{}
	return reflect.DeepEqual(actual, expected)
}

func derivativeCapabilitiesForViewer(info map[string]any, semantics derivativeSemantics) derivativeCapabilities {
	viewer, _ := info["viewer"].(map[string]any)
	_, nestedTile := viewer["tile_scheme"].(map[string]any)
	_, topLevelTile := info["tile_scheme"].(map[string]any)
	tile := nestedTile || topLevelTile
	_, atlas := viewer["atlas_scheme"].(map[string]any)
	atlas = atlas && semantics.AxisSizes.Z > 1
	return derivativeCapabilities{
		Atlas:           atlas,
		AtlasT:          atlas,
		LUT:             true,
		OrderedChannels: true,
		Slice:           true,
		Thumbnail:       true,
		Tile:            tile,
		TileT:           tile && semantics.AxisSizes.T == 1,
		TileZ:           tile && semantics.AxisSizes.Z == 1,
	}
}
