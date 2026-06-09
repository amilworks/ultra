package store

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func mergeResourceMetadata(existing domain.JSONMap, patch domain.JSONMap) domain.JSONMap {
	merged := domain.JSONMap{}
	if existing != nil {
		merged = cloneResourceMetadataValue(existing).(domain.JSONMap)
	}
	for key, value := range patch {
		patchMap, patchIsMap := resourceMetadataMap(value)
		if patchIsMap {
			if existingMap, existingIsMap := resourceMetadataMap(merged[key]); existingIsMap {
				merged[key] = mergeResourceMetadata(existingMap, patchMap)
			} else {
				merged[key] = cloneResourceMetadataValue(patchMap)
			}
			continue
		}
		merged[key] = cloneResourceMetadataValue(value)
	}
	return merged
}

func resourceMetadataMap(value any) (domain.JSONMap, bool) {
	switch typed := value.(type) {
	case domain.JSONMap:
		return typed, true
	case map[string]any:
		out := make(domain.JSONMap, len(typed))
		for key, item := range typed {
			out[key] = item
		}
		return out, true
	default:
		return nil, false
	}
}

func cloneResourceMetadataValue(value any) any {
	switch typed := value.(type) {
	case nil:
		return nil
	case domain.JSONMap:
		out := make(domain.JSONMap, len(typed))
		for key, item := range typed {
			out[key] = cloneResourceMetadataValue(item)
		}
		return out
	case map[string]any:
		out := make(domain.JSONMap, len(typed))
		for key, item := range typed {
			out[key] = cloneResourceMetadataValue(item)
		}
		return out
	case []any:
		out := make([]any, len(typed))
		for index, item := range typed {
			out[index] = cloneResourceMetadataValue(item)
		}
		return out
	default:
		return typed
	}
}

func normalizeResourceTags(tags []string) []string {
	out := make([]string, 0, len(tags))
	seen := map[string]bool{}
	for _, tag := range tags {
		trimmed := strings.TrimSpace(tag)
		if trimmed == "" {
			continue
		}
		key := strings.ToLower(trimmed)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, trimmed)
	}
	return out
}

func resourceTagKeys(tags []string) []string {
	normalized := normalizeResourceTags(tags)
	keys := make([]string, 0, len(normalized))
	for _, tag := range normalized {
		keys = append(keys, strings.ToLower(tag))
	}
	return keys
}

func resourceTagsFromMetadata(metadata domain.JSONMap) []string {
	if metadata == nil {
		return nil
	}
	switch tags := metadata["tags"].(type) {
	case []string:
		return normalizeResourceTags(tags)
	case []any:
		values := make([]string, 0, len(tags))
		for _, item := range tags {
			if text, ok := item.(string); ok {
				values = append(values, text)
			}
		}
		return normalizeResourceTags(values)
	case string:
		return normalizeResourceTags([]string{tags})
	default:
		return nil
	}
}

func resourceMetadataWithTags(metadata domain.JSONMap, tags []string) domain.JSONMap {
	normalized := normalizeResourceTags(tags)
	out := domain.JSONMap{}
	if metadata != nil {
		out = cloneResourceMetadataValue(metadata).(domain.JSONMap)
	}
	if len(normalized) == 0 {
		delete(out, "tags")
		delete(out, "tag_keys")
		return out
	}
	out["tags"] = append([]string(nil), normalized...)
	tagKeys := make([]string, 0, len(normalized))
	tagKeys = append(tagKeys, resourceTagKeys(normalized)...)
	out["tag_keys"] = tagKeys
	return out
}

func tagsForResource(resource domain.ResourceRecord) []string {
	if len(resource.Tags) > 0 {
		return normalizeResourceTags(resource.Tags)
	}
	return resourceTagsFromMetadata(resource.Metadata)
}

func resourceWithNormalizedTags(resource domain.ResourceRecord) domain.ResourceRecord {
	tags := tagsForResource(resource)
	resource.Tags = append([]string(nil), tags...)
	resource.Metadata = resourceMetadataWithTags(resource.Metadata, tags)
	return resource
}

func mergeResourceTags(existing []string, added []string) []string {
	merged := append([]string(nil), existing...)
	seen := map[string]bool{}
	for _, tag := range normalizeResourceTags(merged) {
		seen[strings.ToLower(tag)] = true
	}
	for _, tag := range normalizeResourceTags(added) {
		key := strings.ToLower(tag)
		if seen[key] {
			continue
		}
		seen[key] = true
		merged = append(merged, tag)
	}
	return normalizeResourceTags(merged)
}

func resourceMatchesTags(resource domain.ResourceRecord, required []string) bool {
	normalizedRequired := normalizeResourceTags(required)
	if len(normalizedRequired) == 0 {
		return true
	}
	have := map[string]bool{}
	for _, tag := range tagsForResource(resource) {
		have[strings.ToLower(tag)] = true
	}
	for _, tag := range normalizedRequired {
		if !have[strings.ToLower(tag)] {
			return false
		}
	}
	return true
}

func resourceSearchDocument(resource domain.ResourceRecord) string {
	parts := make([]string, 0, 16)
	appendSearchPart := func(value string) {
		trimmed := strings.TrimSpace(value)
		if trimmed != "" {
			parts = append(parts, strings.ToLower(trimmed))
		}
	}
	appendSearchPart(resource.ResourceID)
	appendSearchPart(resource.OriginalName)
	appendSearchPart(resource.SourceURI)
	appendSearchPart(resource.ContentType)
	appendSearchPart(resource.ResourceKind)
	appendSearchPart(resource.SourceType)
	appendSearchPart(resource.ProjectID)
	appendSearchPart(resource.SHA256)
	appendResourceMetadataSearchValues(resource.Metadata, appendSearchPart)
	return strings.Join(parts, "\n")
}

func appendResourceMetadataSearchValues(value any, appendSearchPart func(string)) {
	switch typed := value.(type) {
	case nil:
		return
	case string:
		appendSearchPart(typed)
	case bool:
		appendSearchPart(strconv.FormatBool(typed))
	case int:
		appendSearchPart(strconv.Itoa(typed))
	case int8:
		appendSearchPart(strconv.FormatInt(int64(typed), 10))
	case int16:
		appendSearchPart(strconv.FormatInt(int64(typed), 10))
	case int32:
		appendSearchPart(strconv.FormatInt(int64(typed), 10))
	case int64:
		appendSearchPart(strconv.FormatInt(typed, 10))
	case uint:
		appendSearchPart(strconv.FormatUint(uint64(typed), 10))
	case uint8:
		appendSearchPart(strconv.FormatUint(uint64(typed), 10))
	case uint16:
		appendSearchPart(strconv.FormatUint(uint64(typed), 10))
	case uint32:
		appendSearchPart(strconv.FormatUint(uint64(typed), 10))
	case uint64:
		appendSearchPart(strconv.FormatUint(typed, 10))
	case float32:
		appendSearchPart(strconv.FormatFloat(float64(typed), 'f', -1, 32))
	case float64:
		appendSearchPart(strconv.FormatFloat(typed, 'f', -1, 64))
	case domain.JSONMap:
		for _, item := range typed {
			appendResourceMetadataSearchValues(item, appendSearchPart)
		}
	case map[string]any:
		for _, item := range typed {
			appendResourceMetadataSearchValues(item, appendSearchPart)
		}
	case []string:
		for _, item := range typed {
			appendSearchPart(item)
		}
	case []any:
		for _, item := range typed {
			appendResourceMetadataSearchValues(item, appendSearchPart)
		}
	default:
		appendSearchPart(fmt.Sprint(typed))
	}
}

var resourceDescriptorMetadataPaths = [][]string{
	{"label"},
	{"labels"},
	{"descriptor"},
	{"descriptors"},
	{"scientific_descriptor"},
	{"scientific_descriptors"},
	{"diagnosis"},
	{"diagnoses"},
	{"modality"},
	{"organism"},
	{"species"},
	{"data_agent", "caption_resources", "caption"},
	{"data_agent", "caption_resources", "summary"},
	{"data_agent", "extract_metadata", "caption"},
	{"data_agent", "extract_metadata", "summary"},
	{"data_agent", "extract_metadata", "label"},
	{"data_agent", "extract_metadata", "labels"},
	{"data_agent", "extract_metadata", "descriptor"},
	{"data_agent", "extract_metadata", "descriptors"},
	{"data_agent", "extract_metadata", "scientific_descriptor"},
	{"data_agent", "extract_metadata", "scientific_descriptors"},
	{"data_agent", "quality_check_resources", "summary"},
	{"data_agent", "organize_resources", "summary"},
}

func normalizeResourceDescriptors(descriptors []string) []string {
	out := make([]string, 0, len(descriptors))
	seen := map[string]bool{}
	for _, descriptor := range descriptors {
		trimmed := strings.TrimSpace(descriptor)
		if trimmed == "" {
			continue
		}
		key := strings.ToLower(trimmed)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, trimmed)
	}
	return out
}

func resourceMatchesDescriptors(resource domain.ResourceRecord, required []string) bool {
	normalizedRequired := normalizeResourceDescriptors(required)
	if len(normalizedRequired) == 0 {
		return true
	}
	values := make([]string, 0, len(resourceDescriptorMetadataPaths)+len(resource.Tags))
	values = append(values, tagsForResource(resource)...)
	for _, path := range resourceDescriptorMetadataPaths {
		value, ok := metadataValueAtPath(resource.Metadata, path)
		if !ok {
			continue
		}
		appendResourceDescriptorValues(&values, value)
	}
	haystack := strings.ToLower(strings.Join(values, "\x00"))
	for _, descriptor := range normalizedRequired {
		if !strings.Contains(haystack, strings.ToLower(strings.TrimSpace(descriptor))) {
			return false
		}
	}
	return true
}

func appendResourceDescriptorValues(values *[]string, value any) {
	switch typed := value.(type) {
	case nil:
		return
	case string:
		if strings.TrimSpace(typed) != "" {
			*values = append(*values, typed)
		}
	case []string:
		for _, item := range typed {
			appendResourceDescriptorValues(values, item)
		}
	case []any:
		for _, item := range typed {
			appendResourceDescriptorValues(values, item)
		}
	case domain.JSONMap:
		for _, item := range typed {
			appendResourceDescriptorValues(values, item)
		}
	case map[string]any:
		for _, item := range typed {
			appendResourceDescriptorValues(values, item)
		}
	default:
		text := strings.TrimSpace(fmt.Sprint(typed))
		if text != "" {
			*values = append(*values, text)
		}
	}
}

func resourceMatchesCreatedRange(resource domain.ResourceRecord, createdAfter time.Time, createdBefore time.Time) bool {
	createdAt := resource.CreatedAt.UTC()
	if !createdAfter.IsZero() && createdAt.Before(createdAfter.UTC()) {
		return false
	}
	if !createdBefore.IsZero() && createdAt.After(createdBefore.UTC()) {
		return false
	}
	return true
}

var resourceProcessingReadyJobTypes = []string{
	"caption_resources",
	"extract_metadata",
	"batch_tag_resources",
	"quality_check_resources",
	"deduplicate_resources",
	"organize_resources",
}

func resourceMatchesProcessingStatus(resource domain.ResourceRecord, processingStatus string) bool {
	switch strings.ToLower(strings.TrimSpace(processingStatus)) {
	case "", "all":
		return true
	case "caption_ready":
		return resourceDataAgentJobSucceeded(resource.Metadata, "caption_resources")
	case "metadata_ready":
		return resourceDataAgentJobSucceeded(resource.Metadata, "extract_metadata")
	case "tags_ready":
		return resourceDataAgentJobSucceeded(resource.Metadata, "batch_tag_resources")
	case "qc_complete":
		return resourceDataAgentJobSucceeded(resource.Metadata, "quality_check_resources")
	case "dedupe_checked":
		return resourceDataAgentJobSucceeded(resource.Metadata, "deduplicate_resources")
	case "organization_ready":
		return resourceDataAgentJobSucceeded(resource.Metadata, "organize_resources")
	case "data_agent_ready":
		for _, jobType := range resourceProcessingReadyJobTypes {
			if resourceDataAgentJobSucceeded(resource.Metadata, jobType) {
				return true
			}
		}
		return false
	case "needs_caption":
		return !resourceDataAgentJobSucceeded(resource.Metadata, "caption_resources")
	case "needs_metadata":
		return !resourceDataAgentJobSucceeded(resource.Metadata, "extract_metadata")
	case "data_agent_failed":
		for _, jobType := range resourceProcessingReadyJobTypes {
			if resourceDataAgentJobFailed(resource.Metadata, jobType) {
				return true
			}
		}
		return false
	default:
		return false
	}
}

func resourceDataAgentJobSucceeded(metadata domain.JSONMap, jobType string) bool {
	status := resourceDataAgentJobStatus(metadata, jobType)
	return status == "succeeded" || status == "completed"
}

func resourceDataAgentJobFailed(metadata domain.JSONMap, jobType string) bool {
	status := resourceDataAgentJobStatus(metadata, jobType)
	return status == "failed" || status == "error"
}

func resourceDataAgentJobStatus(metadata domain.JSONMap, jobType string) string {
	dataAgentMetadata, ok := resourceMetadataMap(metadata["data_agent"])
	if !ok {
		return ""
	}
	statusRecord, ok := resourceMetadataMap(dataAgentMetadata[jobType])
	if !ok {
		return ""
	}
	return strings.ToLower(strings.TrimSpace(fmt.Sprint(statusRecord["status"])))
}

func resourceMatchesMetadataFilters(resource domain.ResourceRecord, filters []domain.ResourceMetadataFilter) bool {
	if len(filters) == 0 {
		return true
	}
	for _, filter := range filters {
		if !resourceMatchesMetadataFilter(resource.Metadata, filter) {
			return false
		}
	}
	return true
}

func resourceMetadataFilterSpecs(filters []domain.ResourceMetadataFilter) []string {
	specs := make([]string, 0, len(filters))
	for _, filter := range filters {
		path := strings.TrimSpace(filter.Path)
		operator := strings.ToLower(strings.TrimSpace(filter.Operator))
		value := strings.TrimSpace(filter.Value)
		if path == "" || operator == "" {
			continue
		}
		specs = append(specs, path+":"+operator+":"+value)
	}
	return specs
}

func resourceMatchesMetadataFilter(metadata domain.JSONMap, filter domain.ResourceMetadataFilter) bool {
	path := metadataFilterPath(filter.Path)
	operator := strings.ToLower(strings.TrimSpace(filter.Operator))
	value := strings.TrimSpace(filter.Value)
	if len(path) == 0 || operator == "" {
		return false
	}
	actual, ok := metadataValueAtPath(metadata, path)
	switch operator {
	case "exists":
		return ok
	case "eq":
		return ok && metadataValueEquals(actual, value)
	case "contains":
		return ok && metadataValueContains(actual, value)
	case "lt", "lte", "gt", "gte":
		if !ok {
			return false
		}
		actualNumber, actualOK := metadataValueFloat(actual)
		wantNumber, wantOK := strconv.ParseFloat(value, 64)
		if actualOK != nil || wantOK != nil {
			return false
		}
		switch operator {
		case "lt":
			return actualNumber < wantNumber
		case "lte":
			return actualNumber <= wantNumber
		case "gt":
			return actualNumber > wantNumber
		case "gte":
			return actualNumber >= wantNumber
		}
	}
	return false
}

func metadataFilterPath(path string) []string {
	parts := strings.Split(strings.TrimSpace(path), ".")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil
		}
		out = append(out, part)
	}
	return out
}

func metadataValueAtPath(metadata domain.JSONMap, path []string) (any, bool) {
	if metadata == nil || len(path) == 0 {
		return nil, false
	}
	var current any = metadata
	for _, part := range path {
		currentMap, ok := resourceMetadataMap(current)
		if !ok {
			return nil, false
		}
		value, ok := currentMap[part]
		if !ok {
			return nil, false
		}
		current = value
	}
	return current, true
}

func metadataValueEquals(value any, want string) bool {
	switch typed := value.(type) {
	case nil:
		return want == ""
	case string:
		return strings.EqualFold(strings.TrimSpace(typed), want)
	case bool:
		return strings.EqualFold(strconv.FormatBool(typed), want)
	case int:
		return metadataNumberEquals(float64(typed), want)
	case int64:
		return metadataNumberEquals(float64(typed), want)
	case float64:
		return metadataNumberEquals(typed, want)
	case json.Number:
		number, err := typed.Float64()
		return err == nil && metadataNumberEquals(number, want)
	default:
		return strings.EqualFold(strings.TrimSpace(fmt.Sprint(typed)), want)
	}
}

func metadataNumberEquals(value float64, want string) bool {
	wantNumber, err := strconv.ParseFloat(want, 64)
	return err == nil && value == wantNumber
}

func metadataValueContains(value any, want string) bool {
	needle := strings.ToLower(strings.TrimSpace(want))
	if needle == "" {
		return true
	}
	switch typed := value.(type) {
	case string:
		return strings.Contains(strings.ToLower(typed), needle)
	case []string:
		for _, item := range typed {
			if strings.Contains(strings.ToLower(item), needle) {
				return true
			}
		}
		return false
	case []any:
		for _, item := range typed {
			if metadataValueContains(item, needle) {
				return true
			}
		}
		return false
	default:
		encoded, err := json.Marshal(typed)
		if err != nil {
			return strings.Contains(strings.ToLower(fmt.Sprint(typed)), needle)
		}
		return strings.Contains(strings.ToLower(string(encoded)), needle)
	}
}

func metadataValueFloat(value any) (float64, error) {
	switch typed := value.(type) {
	case int:
		return float64(typed), nil
	case int64:
		return float64(typed), nil
	case float64:
		return typed, nil
	case json.Number:
		return typed.Float64()
	case string:
		return strconv.ParseFloat(strings.TrimSpace(typed), 64)
	default:
		return 0, fmt.Errorf("metadata value is not numeric")
	}
}
