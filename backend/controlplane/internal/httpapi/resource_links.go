package httpapi

import (
	"net/url"
	"strings"
)

// lensDeepLinkPath is the in-app route the frontend's navUrl.ts parses to open
// the Lens viewer for a catalog resource: "/?view=lens&resource=<file_id>".
// The resource id is the same file_id GET /v2/resources/{file_id} serves, so a
// link built here always opens the exact record the agent was shown.
const lensDeepLinkPath = "/?view=lens&resource="

// lensURLForResource builds the Lens deep link the agent copies verbatim into
// its answer. With a configured public URL the link is absolute (the agent's
// output is also read outside the SPA — reports, exports), otherwise it is the
// relative form the frontend resolves against its own origin. An empty id has
// no viewer target and yields "" so the JSON field is omitted.
func lensURLForResource(publicURL, resourceID string) string {
	if resourceID == "" {
		return ""
	}
	base := strings.TrimRight(strings.TrimSpace(publicURL), "/")
	return base + lensDeepLinkPath + url.QueryEscape(resourceID)
}
