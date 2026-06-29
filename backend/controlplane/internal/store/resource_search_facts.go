package store

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

type resourceSearchFact struct {
	Key    string
	Text   string
	Number *float64
	Source string
}

type resourceSearchNumericPredicate struct {
	Key    string
	Op     string
	Number float64
}

type resourceSearchTextPredicate struct {
	Key  string
	Text string
}

type parsedResourceSearchQuery struct {
	ResidualText      string
	NumericPredicates []resourceSearchNumericPredicate
	TextPredicates    []resourceSearchTextPredicate
}

func (query parsedResourceSearchQuery) hasFactPredicates() bool {
	return len(query.NumericPredicates) > 0 || len(query.TextPredicates) > 0
}

var (
	resourceSearchNumericPredicateRE = regexp.MustCompile(`(?i)\b([a-z][a-z0-9_.-]*)\s*(>=|<=|>|<|=|:)\s*([+-]?[0-9]+(?:\.[0-9]+)?)\b`)
	resourceSearchFilenameAgeRE      = regexp.MustCompile(`(?i)(?:^|[^a-z0-9])([0-9]{1,3})\s*(?:yo|y/o|yrs?|years? old|years old|year old)(?:$|[^a-z0-9])`)
	resourceSearchExplicitAgeRE      = regexp.MustCompile(`(?i)(?:^|[^a-z0-9])age[_\-\s]*([0-9]{1,3})(?:$|[^a-z0-9])`)
)

func parseResourceSearchQuery(query string) parsedResourceSearchQuery {
	query = strings.TrimSpace(query)
	if query == "" {
		return parsedResourceSearchQuery{}
	}
	remaining := []byte(query)
	numericMatches := resourceSearchNumericPredicateRE.FindAllStringSubmatchIndex(query, -1)
	predicates := make([]resourceSearchNumericPredicate, 0, len(numericMatches))
	for _, match := range numericMatches {
		key := canonicalResourceSearchFactKey(query[match[2]:match[3]])
		if key == "" {
			continue
		}
		number, err := strconv.ParseFloat(query[match[6]:match[7]], 64)
		if err != nil {
			continue
		}
		predicates = append(predicates, resourceSearchNumericPredicate{
			Key:    key,
			Op:     normalizeResourceSearchOperator(query[match[4]:match[5]]),
			Number: number,
		})
		for index := match[0]; index < match[1]; index++ {
			remaining[index] = ' '
		}
	}

	fields := strings.Fields(string(remaining))
	residualParts := make([]string, 0, len(fields))
	textPredicates := make([]resourceSearchTextPredicate, 0, 1)
	for _, field := range fields {
		trimmed := strings.Trim(field, " \t\r\n,;")
		if strings.HasPrefix(trimmed, "*.") && len(trimmed) > 2 {
			extension := normalizeResourceSearchText(strings.TrimPrefix(trimmed, "*."))
			if extension != "" {
				textPredicates = append(textPredicates, resourceSearchTextPredicate{
					Key:  "extension",
					Text: extension,
				})
				continue
			}
		}
		residualParts = append(residualParts, field)
	}

	return parsedResourceSearchQuery{
		ResidualText:      strings.ToLower(strings.Join(residualParts, " ")),
		NumericPredicates: predicates,
		TextPredicates:    textPredicates,
	}
}

func normalizeResourceSearchOperator(operator string) string {
	switch strings.TrimSpace(operator) {
	case ">":
		return "gt"
	case ">=":
		return "gte"
	case "<":
		return "lt"
	case "<=":
		return "lte"
	case ":", "=":
		return "eq"
	default:
		return strings.TrimSpace(operator)
	}
}

func resourceMatchesParsedSearchQuery(resource domain.ResourceRecord, parsed parsedResourceSearchQuery) bool {
	facts := resourceSearchFacts(resource)
	for _, predicate := range parsed.NumericPredicates {
		if !resourceSearchFactsMatchNumericPredicate(facts, predicate) {
			return false
		}
	}
	for _, predicate := range parsed.TextPredicates {
		if !resourceSearchFactsMatchTextPredicate(facts, predicate) {
			return false
		}
	}
	if parsed.ResidualText != "" && !strings.Contains(resourceSearchDocument(resource), parsed.ResidualText) {
		return false
	}
	return true
}

func resourceSearchFactsMatchNumericPredicate(facts []resourceSearchFact, predicate resourceSearchNumericPredicate) bool {
	for _, fact := range facts {
		if fact.Key != predicate.Key || fact.Number == nil {
			continue
		}
		if compareResourceSearchNumber(*fact.Number, predicate.Op, predicate.Number) {
			return true
		}
	}
	return false
}

func resourceSearchFactsMatchTextPredicate(facts []resourceSearchFact, predicate resourceSearchTextPredicate) bool {
	for _, fact := range facts {
		if fact.Key == predicate.Key && fact.Text == predicate.Text {
			return true
		}
	}
	return false
}

func compareResourceSearchNumber(actual float64, operator string, expected float64) bool {
	switch operator {
	case "gt":
		return actual > expected
	case "gte":
		return actual >= expected
	case "lt":
		return actual < expected
	case "lte":
		return actual <= expected
	case "eq":
		return actual == expected
	default:
		return false
	}
}

func resourceSearchFacts(resource domain.ResourceRecord) []resourceSearchFact {
	facts := make([]resourceSearchFact, 0, 16)
	factKeys := map[string]struct{}{}
	addTextFact := func(key string, value string, source string) {
		key = canonicalResourceSearchFactKey(key)
		value = normalizeResourceSearchText(value)
		if key == "" || value == "" {
			return
		}
		dedupeKey := key + "\x00text\x00" + value
		if _, ok := factKeys[dedupeKey]; ok {
			return
		}
		factKeys[dedupeKey] = struct{}{}
		facts = append(facts, resourceSearchFact{Key: key, Text: value, Source: source})
	}
	addNumberFact := func(key string, value float64, source string) {
		key = canonicalResourceSearchFactKey(key)
		if key == "" {
			return
		}
		dedupeKey := key + "\x00number\x00" + strconv.FormatFloat(value, 'g', -1, 64)
		if _, ok := factKeys[dedupeKey]; ok {
			return
		}
		factKeys[dedupeKey] = struct{}{}
		number := value
		facts = append(facts, resourceSearchFact{Key: key, Number: &number, Source: source})
	}

	for _, name := range []string{resource.OriginalName, resource.SourceURI, resource.StorageURI} {
		for _, extension := range resourceSearchExtensions(name) {
			addTextFact("extension", extension, "filename")
		}
		for _, age := range resourceSearchAgesFromName(name) {
			addNumberFact("age", age, "filename")
			addNumberFact("subject_age", age, "filename")
		}
	}
	appendResourceMetadataFacts(resource.Metadata, nil, addTextFact, addNumberFact)
	return facts
}

func appendResourceMetadataFacts(
	value any,
	path []string,
	addTextFact func(string, string, string),
	addNumberFact func(string, float64, string),
) {
	switch typed := value.(type) {
	case nil:
		return
	case string:
		if len(path) == 0 {
			return
		}
		for _, key := range resourceSearchFactKeysForPath(path) {
			addTextFact(key, typed, "metadata")
		}
		if number, ok := parseLooseResourceSearchNumber(typed); ok {
			for _, key := range resourceSearchFactKeysForPath(path) {
				addNumberFact(key, number, "metadata")
			}
		}
	case bool:
		if len(path) == 0 {
			return
		}
		for _, key := range resourceSearchFactKeysForPath(path) {
			addTextFact(key, strconv.FormatBool(typed), "metadata")
		}
	case int:
		appendResourceMetadataNumberFacts(float64(typed), path, addNumberFact)
	case int8:
		appendResourceMetadataNumberFacts(float64(typed), path, addNumberFact)
	case int16:
		appendResourceMetadataNumberFacts(float64(typed), path, addNumberFact)
	case int32:
		appendResourceMetadataNumberFacts(float64(typed), path, addNumberFact)
	case int64:
		appendResourceMetadataNumberFacts(float64(typed), path, addNumberFact)
	case uint:
		appendResourceMetadataNumberFacts(float64(typed), path, addNumberFact)
	case uint8:
		appendResourceMetadataNumberFacts(float64(typed), path, addNumberFact)
	case uint16:
		appendResourceMetadataNumberFacts(float64(typed), path, addNumberFact)
	case uint32:
		appendResourceMetadataNumberFacts(float64(typed), path, addNumberFact)
	case uint64:
		appendResourceMetadataNumberFacts(float64(typed), path, addNumberFact)
	case float32:
		appendResourceMetadataNumberFacts(float64(typed), path, addNumberFact)
	case float64:
		appendResourceMetadataNumberFacts(typed, path, addNumberFact)
	case domain.JSONMap:
		for key, item := range typed {
			appendResourceMetadataFacts(item, append(path, key), addTextFact, addNumberFact)
		}
	case map[string]any:
		for key, item := range typed {
			appendResourceMetadataFacts(item, append(path, key), addTextFact, addNumberFact)
		}
	case []string:
		for _, item := range typed {
			appendResourceMetadataFacts(item, path, addTextFact, addNumberFact)
		}
	case []any:
		for _, item := range typed {
			appendResourceMetadataFacts(item, path, addTextFact, addNumberFact)
		}
	}
}

func appendResourceMetadataNumberFacts(value float64, path []string, addNumberFact func(string, float64, string)) {
	if len(path) == 0 {
		return
	}
	for _, key := range resourceSearchFactKeysForPath(path) {
		addNumberFact(key, value, "metadata")
	}
}

func resourceSearchFactKeysForPath(path []string) []string {
	if len(path) == 0 {
		return nil
	}
	leaf := normalizeResourceSearchKey(path[len(path)-1])
	full := normalizeResourceSearchKey(strings.Join(path, "_"))
	canonicalLeaf := canonicalResourceSearchFactKey(leaf)
	keys := make([]string, 0, 3)
	seen := map[string]struct{}{}
	for _, key := range []string{canonicalLeaf, leaf, full} {
		key = canonicalResourceSearchFactKey(key)
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		keys = append(keys, key)
	}
	return keys
}

func canonicalResourceSearchFactKey(key string) string {
	key = normalizeResourceSearchKey(key)
	if key == "" {
		return ""
	}
	parts := strings.Split(key, "_")
	leaf := key
	if len(parts) > 0 {
		leaf = parts[len(parts)-1]
	}
	switch key {
	case "subject_age", "patient_age", "participant_age", "age_years", "years_old":
		return "age"
	case "focal_length_mm", "focallength", "focal_length_35mm":
		return "focal_length"
	case "image_width", "pixel_width", "size_x", "x_size":
		return "width"
	case "image_height", "pixel_height", "size_y", "y_size":
		return "height"
	case "image_depth", "pixel_depth", "size_z", "z_size", "slices":
		return "depth"
	case "channel", "channels", "size_c", "c_size":
		return "channels"
	case "time_points", "timepoint", "timepoints", "size_t", "t_size", "frames":
		return "timepoints"
	case "iso_speed", "iso_speed_ratings", "photographic_sensitivity":
		return "iso"
	}
	switch leaf {
	case "age":
		return "age"
	case "width":
		return "width"
	case "height":
		return "height"
	case "depth":
		return "depth"
	case "iso":
		return "iso"
	}
	if strings.HasSuffix(key, "_age") {
		return "age"
	}
	if strings.HasSuffix(key, "_width") {
		return "width"
	}
	if strings.HasSuffix(key, "_height") {
		return "height"
	}
	if strings.HasSuffix(key, "_focal_length") || strings.HasSuffix(key, "_focal_length_mm") {
		return "focal_length"
	}
	return key
}

func normalizeResourceSearchKey(key string) string {
	key = strings.ToLower(strings.TrimSpace(key))
	replacer := strings.NewReplacer(
		".", "_",
		"-", "_",
		" ", "_",
		"/", "_",
		"(", "",
		")", "",
	)
	key = replacer.Replace(key)
	key = strings.Trim(key, "_")
	for strings.Contains(key, "__") {
		key = strings.ReplaceAll(key, "__", "_")
	}
	return key
}

func normalizeResourceSearchText(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.Trim(value, ".")
	return value
}

func resourceSearchExtensions(name string) []string {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		return nil
	}
	if index := strings.IndexAny(name, "?#"); index >= 0 {
		name = name[:index]
	}
	name = strings.TrimRight(name, "/")
	parts := strings.Split(name, "/")
	name = parts[len(parts)-1]
	if !strings.Contains(name, ".") {
		return nil
	}
	segments := strings.Split(name, ".")
	extensions := make([]string, 0, 3)
	if len(segments) >= 2 {
		extensions = append(extensions, segments[len(segments)-1])
	}
	if len(segments) >= 3 {
		extensions = append(extensions, segments[len(segments)-2]+"."+segments[len(segments)-1])
	}
	if strings.HasSuffix(name, ".nii.gz") {
		extensions = append(extensions, "nii")
	}
	return uniqueResourceSearchTexts(extensions)
}

func resourceSearchAgesFromName(name string) []float64 {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil
	}
	ages := make([]float64, 0, 1)
	for _, match := range resourceSearchFilenameAgeRE.FindAllStringSubmatch(name, -1) {
		if len(match) < 2 {
			continue
		}
		if age, ok := parseResourceSearchAge(match[1]); ok {
			ages = append(ages, age)
		}
	}
	for _, match := range resourceSearchExplicitAgeRE.FindAllStringSubmatch(name, -1) {
		if len(match) < 2 {
			continue
		}
		if age, ok := parseResourceSearchAge(match[1]); ok {
			ages = append(ages, age)
		}
	}
	if age, ok := resourceSearchCohortBareAgeFromName(name); ok {
		ages = append(ages, age)
	}
	return ages
}

func resourceSearchCohortBareAgeFromName(name string) (float64, bool) {
	name = strings.ToLower(strings.TrimSpace(name))
	if index := strings.IndexAny(name, "?#"); index >= 0 {
		name = name[:index]
	}
	name = strings.TrimRight(name, "/")
	parts := strings.Split(name, "/")
	name = parts[len(parts)-1]
	switch {
	case strings.HasSuffix(name, ".nii.gz"):
		name = strings.TrimSuffix(name, ".nii.gz")
	case strings.Contains(name, "."):
		segments := strings.Split(name, ".")
		name = strings.Join(segments[:len(segments)-1], ".")
	}
	tokens := strings.FieldsFunc(name, func(char rune) bool {
		return !((char >= 'a' && char <= 'z') || (char >= '0' && char <= '9'))
	})
	if len(tokens) < 3 {
		return 0, false
	}
	hasCohortToken := false
	numericTokenCount := 0
	for _, token := range tokens {
		if token == "old" || token == "young" {
			hasCohortToken = true
		}
		if _, err := strconv.Atoi(token); err == nil {
			numericTokenCount++
		}
	}
	if !hasCohortToken || numericTokenCount < 2 {
		return 0, false
	}
	last := tokens[len(tokens)-1]
	if len(last) < 2 {
		return 0, false
	}
	return parseResourceSearchAge(last)
}

func parseResourceSearchAge(value string) (float64, bool) {
	age, err := strconv.ParseFloat(value, 64)
	if err != nil || age <= 0 || age > 130 {
		return 0, false
	}
	return age, true
}

func parseLooseResourceSearchNumber(value string) (float64, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}
	number, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, false
	}
	return number, true
}

func uniqueResourceSearchTexts(values []string) []string {
	out := make([]string, 0, len(values))
	seen := map[string]struct{}{}
	for _, value := range values {
		value = normalizeResourceSearchText(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}
