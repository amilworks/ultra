package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

const (
	sensorFormatBindingSchema      = "ultra.sensor-format-binding.v1"
	sensorSeriesSchema             = "ultra.sensor-series.v1"
	maxSensorRootAttributesBytes   = 1 << 20
	sensorFormatDetectionAuthority = "bounded_root_attributes"
)

// authorizeRunFileIDs turns selected resource ids into a tenant-scoped run
// capability. An id the caller cannot read (nonexistent OR foreign — the two are
// intentionally indistinguishable) is DROPPED from the selection rather than
// failing the whole run: a stale/deleted/unshared leftover selection in the
// client must not block run creation. Only a genuine store error propagates. The
// returned catalog records are also the sole source for immutable binding
// descriptors; caller-supplied descriptors never provide checksum authority.
func authorizeRunFileIDs(
	ctx context.Context,
	catalog resourceCatalogStore,
	principal requestPrincipal,
	fileIDs []string,
) ([]string, []domain.ResourceRecord, error) {
	fileIDs = uniqueTrimmedStringValues(fileIDs)
	authorized := make([]string, 0, len(fileIDs))
	resources := make([]domain.ResourceRecord, 0, len(fileIDs))
	for _, fileID := range fileIDs {
		resource, err := catalog.GetResourceForUser(ctx, fileID, principal.UserID, principal.OrgID)
		if err != nil {
			if errors.Is(err, store.ErrNotFound) {
				// Skip an unreadable selection; do not leak whether it was
				// nonexistent or foreign, and do not fail the run.
				continue
			}
			return nil, nil, err
		}
		authorized = append(authorized, fileID)
		resources = append(resources, resource)
	}
	return authorized, resources, nil
}

func withAuthorizedSelectedResourceDescriptors(
	_ []domain.JSONMap,
	resources []domain.ResourceRecord,
	principal requestPrincipal,
) []domain.JSONMap {
	// Every descriptor arriving in the request body is caller-authored. Do not
	// preserve any of them: selected-resource capabilities are reconstructed
	// below from tenant-authorized catalog rows, and prior-artifact capabilities
	// are added later by runcontrol from owner-scoped durable records.
	out := make([]domain.JSONMap, 0, len(resources))
	for _, resource := range resources {
		governanceScope := "read_only_usage"
		if resource.OwnerUserID == principal.UserID &&
			(strings.TrimSpace(resource.OwnerOrgID) == "" || resource.OwnerOrgID == principal.OrgID) {
			governanceScope = "owner_validation"
		}
		descriptor := domain.JSONMap{
			"type":                     "selected_resource",
			"binding_schema":           "ultra.selected_resource.v1",
			"authority":                "control_resource_catalog",
			"resource_id":              resource.ResourceID,
			"file_id":                  resource.ResourceID,
			"original_name":            resource.OriginalName,
			"content_type":             resource.ContentType,
			"resource_kind":            resource.ResourceKind,
			"source_type":              resource.SourceType,
			"sha256":                   resource.SHA256,
			"size_bytes":               resource.SizeBytes,
			"calphad_governance_scope": governanceScope,
		}
		if databaseFormat, ok := domain.CalphadDatabaseFormatFromName(resource.OriginalName); ok {
			descriptor["database_format"] = databaseFormat
		}
		if metadata := projectRunResourceMetadata(resource.Metadata); len(metadata) > 0 {
			descriptor["metadata"] = metadata
		}
		if treeIdentity := projectCatalogTreeIdentity(resource); len(treeIdentity) > 0 {
			descriptor["tree_identity"] = treeIdentity
		}
		if sensorFormat := projectCatalogSensorFormat(resource); len(sensorFormat) > 0 {
			descriptor["sensor_format"] = sensorFormat
		}
		out = append(out, descriptor)
	}
	return out
}

// projectCatalogTreeIdentity exposes a directory identity only when the
// catalog's immutable top-level digest belongs to the server's canonical
// bundle layout. The path is used solely as a type/shape check and is never
// returned. This remains durable across a catalog display-name rename.
func projectCatalogTreeIdentity(resource domain.ResourceRecord) domain.JSONMap {
	digest := strings.ToLower(strings.TrimSpace(resource.SHA256))
	if !isSHA256Hex(digest) || !catalogResourceIsBundleDirectory(resource) {
		return nil
	}
	return resourceTreeIdentityDescriptor(digest)
}

// projectCatalogSensorFormat emits a closed type marker only after bounded inspection of
// catalog-bound root attributes. A generic .zarr suffix or OME-NGFF multiscales declaration
// is intentionally insufficient: those resources must not be routed through sensor tools.
func projectCatalogSensorFormat(resource domain.ResourceRecord) domain.JSONMap {
	digest := strings.ToLower(strings.TrimSpace(resource.SHA256))
	root, ok := catalogResourceBundleDirectoryPath(resource)
	if !ok || !isSHA256Hex(digest) || !catalogRootDeclaresSensorSeries(root) {
		return nil
	}
	return domain.JSONMap{
		"schema":          sensorFormatBindingSchema,
		"authority":       "control_resource_catalog",
		"container":       "zarr",
		"sensor_schema":   sensorSeriesSchema,
		"resource_sha256": digest,
		"detection":       sensorFormatDetectionAuthority,
	}
}

func catalogRootDeclaresSensorSeries(root string) bool {
	for _, candidate := range []struct {
		name       string
		attributes bool
	}{
		{name: ".zattrs"},
		{name: "zarr.json", attributes: true},
	} {
		path := filepath.Join(root, candidate.name)
		info, err := os.Lstat(path)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return false
		}
		if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() ||
			info.Size() < 0 || info.Size() > maxSensorRootAttributesBytes {
			return false
		}
		file, err := os.Open(path)
		if err != nil {
			return false
		}
		payload, readErr := io.ReadAll(io.LimitReader(file, maxSensorRootAttributesBytes+1))
		closeErr := file.Close()
		if readErr != nil || closeErr != nil || len(payload) > maxSensorRootAttributesBytes {
			return false
		}
		var rootObject map[string]any
		if err := json.Unmarshal(payload, &rootObject); err != nil {
			return false
		}
		attributes := rootObject
		if candidate.attributes {
			var attributesOK bool
			attributes, attributesOK = rootObject["attributes"].(map[string]any)
			if !attributesOK {
				return false
			}
		}
		return attributesDeclareSensorSeries(attributes)
	}
	return false
}

func attributesDeclareSensorSeries(attributes map[string]any) bool {
	if ultra, ok := attributes["ultra"].(map[string]any); ok {
		if series, ok := ultra["sensor_series"].(map[string]any); ok {
			if schema, ok := series["schema"].(string); ok && schema == sensorSeriesSchema {
				return true
			}
		}
	}
	if series, ok := attributes["ultra.sensor_series"].(map[string]any); ok {
		schema, ok := series["schema"].(string)
		return ok && schema == sensorSeriesSchema
	}
	return false
}

func catalogResourceIsBundleDirectory(resource domain.ResourceRecord) bool {
	_, ok := catalogResourceBundleDirectoryPath(resource)
	return ok
}

func catalogResourceBundleDirectoryPath(resource domain.ResourceRecord) (string, bool) {
	candidates := make([]string, 0, 2)
	if path := strings.TrimSpace(resource.StoragePath); path != "" {
		candidates = append(candidates, path)
	}
	if raw := strings.TrimSpace(resource.StorageURI); raw != "" {
		if parsed, err := url.Parse(raw); err == nil && parsed.Scheme == "file" {
			candidates = append(candidates, parsed.Path)
		}
	}
	for _, candidate := range candidates {
		path := filepath.Clean(candidate)
		resourceDir := filepath.Dir(path)
		if filepath.Base(resourceDir) != resource.ResourceID ||
			filepath.Base(filepath.Dir(resourceDir)) != bundlesDirName {
			continue
		}
		info, err := os.Lstat(path)
		if err == nil && info.Mode()&os.ModeSymlink == 0 && info.IsDir() {
			return path, true
		}
	}
	return "", false
}

// trustedRunOrgID returns the organization stamped by metadataWithPrincipal at
// run creation. Worker-supplied organization headers are deliberately not an
// input. A contradictory nested principal fails closed to the empty org scope.
func trustedRunOrgID(run domain.RunRecord) string {
	orgID, _ := safeMetadataString(run.Metadata["org_id"], 512)
	principal, ok := jsonMapValue(run.Metadata["principal"])
	if !ok {
		return orgID
	}
	principalOrgID, _ := safeMetadataString(principal["org_id"], 512)
	if principalOrgID != "" && principalOrgID != orgID {
		return ""
	}
	return orgID
}

// projectRunResourceMetadata is the complete model-visible metadata allowlist
// for run resource search/resolve. Unknown keys are denied by default. In
// particular, raw license text, credentials, download URLs, vendor terms, and
// arbitrary nested metadata never cross the model boundary. CALPHAD fields are
// labeled as owner declarations; owner-writable validation and nested content-
// identity claims are omitted in favor of the catalog record's top-level hash.
func projectRunResourceMetadata(metadata domain.JSONMap) domain.JSONMap {
	if len(metadata) == 0 {
		return nil
	}
	out := domain.JSONMap{}
	if source, ok := safeOwnerDeclaration(metadata["source"], 1024); ok {
		out["source"] = source
	}
	copySafeScalarKeys(out, metadata, []string{
		"caption",
		"description",
	})
	copySafeStringListKeys(out, metadata, []string{
		"scientific_descriptors",
	})
	if header, ok := jsonMapValue(metadata["image_header"]); ok {
		if projected := projectRunImageHeader(header); len(projected) > 0 {
			out["image_header"] = projected
		}
	}
	if calphad, ok := jsonMapValue(metadata["calphad"]); ok {
		if projected := projectRunCALPHADMetadata(calphad); len(projected) > 0 {
			out["calphad"] = projected
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func projectRunCALPHADMetadata(metadata domain.JSONMap) domain.JSONMap {
	out := domain.JSONMap{}
	// These fields are sufficient to select immutable database bytes and assess
	// their scientific/provenance scope without exposing license bodies or
	// private commercial terms.
	for _, field := range []struct {
		key       string
		maxLength int
	}{
		{key: "database_id", maxLength: 512},
		{key: "database_name", maxLength: 512},
		{key: "database_version", maxLength: 256},
		{key: "title", maxLength: 1024},
		{key: "citation", maxLength: 1024},
		{key: "publication_doi", maxLength: 512},
		{key: "source", maxLength: 1024},
		{key: "authorized_scope", maxLength: 512},
		{key: "assessment_scope", maxLength: 1024},
		{key: "reference_state", maxLength: 512},
	} {
		if declaration, ok := safeOwnerDeclaration(metadata[field.key], field.maxLength); ok {
			out[field.key] = declaration
		}
	}
	for _, key := range []string{"license_id", "license_identifier"} {
		if identifier, ok := safeLicenseIdentifier(metadata[key]); ok {
			out[key] = identifier
		}
	}
	copySafeScalarKeys(out, metadata, []string{"component_count", "phase_count"})
	for _, key := range []string{"source_uri", "license_uri"} {
		if uri, ok := safeMetadataURI(metadata[key]); ok {
			out[key] = uri
		}
	}
	copySafeStringListKeys(out, metadata, []string{
		"authors",
		"components",
		"elements",
		"phases",
		"required_provenance_fields",
	})
	copySafeScalarListKeys(out, metadata, []string{
		"tdb_temperature_limits_K",
		"assessment_temperature_limits_K",
	})
	if pressureLimits, ok := normalizedCalphadAssessmentPressureLimits(
		metadata[domain.CalphadAssessmentPressureLimitsMetadataKey],
	); ok {
		out[domain.CalphadAssessmentPressureLimitsMetadataKey] = []float64{
			pressureLimits[0], pressureLimits[1],
		}
	}
	if limits, ok := jsonMapValue(metadata["limits"]); ok {
		if projected := projectRunCALPHADLimits(limits); len(projected) > 0 {
			out["limits"] = projected
		}
	}
	if len(out) > 0 {
		// Resource metadata is owner-writable. Provenance fields are useful input
		// declarations, but never a validation attestation. Immutable content
		// identity is carried separately by the catalog descriptor's top-level
		// sha256/size_bytes fields.
		out["declaration_authority"] = "resource_owner"
	}
	return out
}

func normalizedCalphadAssessmentPressureLimits(value any) ([2]float64, bool) {
	values, ok := metadataList(value)
	if !ok || len(values) != 2 {
		return [2]float64{}, false
	}
	minimum, minimumOK := calphadDescriptorFloat64(values[0])
	maximum, maximumOK := calphadDescriptorFloat64(values[1])
	limits := [2]float64{minimum, maximum}
	if !minimumOK || !maximumOK || minimum < domain.CalphadMinimumPressurePa ||
		maximum > domain.CalphadMaximumPressurePa || minimum > maximum {
		return [2]float64{}, false
	}
	return limits, true
}

func projectRunCALPHADLimits(limits domain.JSONMap) domain.JSONMap {
	out := domain.JSONMap{}
	copySafeScalarKeys(out, limits, []string{
		"max_source_bytes",
		"max_file_bytes",
		"max_result_bytes",
		"max_components",
		"max_phases",
		"max_conditions",
		"max_grid_points",
		"temperature_min_k",
		"temperature_max_k",
		"pressure_min_pa",
		"pressure_max_pa",
		"composition_min",
		"composition_max",
		"total_amount_mol",
	})
	return out
}

func projectRunImageHeader(metadata domain.JSONMap) domain.JSONMap {
	out := domain.JSONMap{}
	copySafeScalarKeys(out, metadata, []string{
		"reader",
		"width",
		"height",
		"depth",
		"time_count",
		"channel_count",
		"dims_order",
		"array_dtype",
		"bytes_per_voxel",
		"physical_spacing_unit",
		"voxel_offset",
		"rescale_slope",
		"rescale_intercept",
		"affine_method",
		"content_type",
		"size_bytes",
		"sha256",
		"source_metadata_reader",
	})
	copySafeScalarListKeys(out, metadata, []string{"array_shape"})
	if spacing, ok := jsonMapValue(metadata["physical_spacing"]); ok {
		projected := domain.JSONMap{}
		copySafeScalarKeys(projected, spacing, []string{"x", "y", "z"})
		if len(projected) > 0 {
			out["physical_spacing"] = projected
		}
	}
	return out
}

func copySafeScalarKeys(out domain.JSONMap, source domain.JSONMap, keys []string) {
	for _, key := range keys {
		if value, ok := safeMetadataScalar(source[key]); ok {
			out[key] = value
		}
	}
}

func copySafeStringListKeys(out domain.JSONMap, source domain.JSONMap, keys []string) {
	for _, key := range keys {
		if value, ok := safeMetadataStringList(source[key]); ok {
			out[key] = value
		}
	}
}

func copySafeScalarListKeys(out domain.JSONMap, source domain.JSONMap, keys []string) {
	for _, key := range keys {
		if value, ok := safeMetadataScalarList(source[key]); ok {
			out[key] = value
		}
	}
}

func safeMetadataScalar(value any) (any, bool) {
	switch typed := value.(type) {
	case string:
		return safeMetadataString(typed, 4096)
	case bool:
		return typed, true
	case float64:
		return typed, !math.IsNaN(typed) && !math.IsInf(typed, 0)
	case float32:
		value := float64(typed)
		return value, !math.IsNaN(value) && !math.IsInf(value, 0)
	case int:
		return typed, true
	case int8:
		return typed, true
	case int16:
		return typed, true
	case int32:
		return typed, true
	case int64:
		return typed, true
	case uint:
		return typed, true
	case uint8:
		return typed, true
	case uint16:
		return typed, true
	case uint32:
		return typed, true
	case uint64:
		return typed, true
	default:
		return nil, false
	}
}

func safeMetadataString(value any, maxLength int) (string, bool) {
	text, ok := value.(string)
	if !ok {
		return "", false
	}
	text = strings.TrimSpace(text)
	if text == "" || len(text) > maxLength {
		return "", false
	}
	return text, true
}

func safeMetadataURI(value any) (string, bool) {
	raw, ok := safeMetadataString(value, 4096)
	if !ok {
		return "", false
	}
	parsed, err := url.Parse(raw)
	if err != nil || (parsed.Scheme != "https" && parsed.Scheme != "http") || parsed.Host == "" {
		return "", false
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", false
	}
	return parsed.String(), true
}

func safeLicenseIdentifier(value any) (string, bool) {
	return domain.SafeCalphadLicenseIdentifier(value)
}

func safeOwnerDeclaration(value any, maxLength int) (string, bool) {
	return domain.SafeCalphadOwnerDeclaration(value, maxLength)
}

func safeMetadataStringList(value any) ([]string, bool) {
	values, ok := metadataList(value)
	if !ok || len(values) == 0 || len(values) > 256 {
		return nil, false
	}
	out := make([]string, 0, len(values))
	for _, item := range values {
		text, ok := safeMetadataString(item, 512)
		if !ok {
			return nil, false
		}
		out = append(out, text)
	}
	return out, true
}

func safeMetadataScalarList(value any) ([]any, bool) {
	values, ok := metadataList(value)
	if !ok || len(values) == 0 || len(values) > 64 {
		return nil, false
	}
	out := make([]any, 0, len(values))
	for _, item := range values {
		scalar, ok := safeMetadataScalar(item)
		if !ok {
			return nil, false
		}
		out = append(out, scalar)
	}
	return out, true
}

func metadataList(value any) ([]any, bool) {
	switch typed := value.(type) {
	case []any:
		return typed, true
	case []string:
		out := make([]any, len(typed))
		for index, item := range typed {
			out[index] = item
		}
		return out, true
	case []float64:
		out := make([]any, len(typed))
		for index, item := range typed {
			out[index] = item
		}
		return out, true
	case []int:
		out := make([]any, len(typed))
		for index, item := range typed {
			out[index] = item
		}
		return out, true
	default:
		return nil, false
	}
}

func jsonMapValue(value any) (domain.JSONMap, bool) {
	switch typed := value.(type) {
	case domain.JSONMap:
		return typed, true
	case map[string]any:
		return domain.JSONMap(typed), true
	default:
		return nil, false
	}
}
