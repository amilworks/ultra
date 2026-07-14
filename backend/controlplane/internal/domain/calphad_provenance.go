package domain

import (
	"net/url"
	"regexp"
	"strings"
)

var calphadSafeLicenseIdentifierPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:+/() -]{0,127}$`)

// SafeCalphadOwnerDeclaration applies the same deny-by-default projection used
// for model-visible selected-resource metadata before provenance is copied into
// the immutable CALPHAD ledger.
func SafeCalphadOwnerDeclaration(value any, maximum int) (string, bool) {
	text, ok := value.(string)
	if !ok || text != strings.TrimSpace(text) || text == "" || len(text) > maximum ||
		strings.ContainsAny(text, "\r\n\t") {
		return "", false
	}
	for _, character := range text {
		if character < 32 || character == 127 {
			return "", false
		}
	}
	lower := strings.ToLower(text)
	for _, sensitive := range []string{
		"password", "api key", "api_key", "access token", "bearer ", "secret",
		"credential", "confidential", "proprietary", "non-disclosure",
		"nondisclosure", "do not distribute", "private license", "license agreement",
		"license text", "commercial terms", "vendor terms", "eula",
	} {
		if strings.Contains(lower, sensitive) {
			return "", false
		}
	}
	if strings.Contains(text, "://") || strings.HasPrefix(lower, "www.") {
		parsed, err := url.Parse(text)
		if err != nil || (parsed.Scheme != "https" && parsed.Scheme != "http") ||
			parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
			return "", false
		}
		return parsed.String(), true
	}
	return text, true
}

func SafeCalphadLicenseIdentifier(value any) (string, bool) {
	identifier, ok := SafeCalphadOwnerDeclaration(value, 128)
	return identifier, ok && calphadSafeLicenseIdentifierPattern.MatchString(identifier)
}

func CalphadDatabaseFormatFromName(name string) (string, bool) {
	name = strings.ToLower(strings.TrimSpace(name))
	switch {
	case strings.HasSuffix(name, ".tdb"):
		return CalphadDatabaseFormatTDB, true
	case strings.HasSuffix(name, ".dat"):
		return CalphadDatabaseFormatDAT, true
	default:
		return "", false
	}
}
