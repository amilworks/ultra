package httpapi

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestResourceListLimitIsCapped pins the contract behind the resource-management
// list endpoints (resources, collections, collection-resources, dataset snapshots,
// share grants): a client-supplied limit is bounded so ?limit=10000000 cannot turn
// into an unbounded SQL scan + JSON serialization (limit32 in the store is a
// default-filler, not a cap, so the cap must be applied at the handler). The
// handlers compose clampLimit(parseLimit(r, 200), 1000); this verifies that
// composition caps oversized values, passes in-range values, and defaults when absent.
func TestResourceListLimitIsCapped(t *testing.T) {
	const def, max = 200, 1000
	cases := []struct {
		query string
		want  int
	}{
		{"", def},                // absent -> default page size
		{"?limit=50", 50},        // in-range -> honored
		{"?limit=1000", max},     // exactly the cap -> honored
		{"?limit=1001", max},     // just over -> capped
		{"?limit=10000000", max}, // abusive -> capped, not an unbounded scan
		{"?limit=0", def},        // invalid (<1) -> default
		{"?limit=-5", def},       // invalid -> default
		{"?limit=abc", def},      // non-numeric -> default
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("limit%s", tc.query), func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/v2/resources"+tc.query, nil)
			if got := clampLimit(parseLimit(req, def), max); got != tc.want {
				t.Fatalf("clampLimit(parseLimit(%q)) = %d, want %d", tc.query, got, tc.want)
			}
		})
	}
}
