package httpapi

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func TestSeqCursorRoundTripAndRejectsGarbage(t *testing.T) {
	t.Parallel()
	for _, seq := range []int64{0, 1, 42, 1 << 40} {
		if got, ok := decodeSeqCursor(encodeSeqCursor(seq)); !ok || got != seq {
			t.Fatalf("roundtrip seq=%d -> ok=%v got=%d", seq, ok, got)
		}
	}
	// A malformed, foreign, or out-of-range token must decode to ok=false so a stale cursor
	// degrades to "from the start" rather than erroring the request.
	garbage := []string{
		"",
		"not-base64!!!",
		base64.RawURLEncoding.EncodeToString([]byte("x:5")),   // foreign prefix
		base64.RawURLEncoding.EncodeToString([]byte("s:-1")),  // negative
		base64.RawURLEncoding.EncodeToString([]byte("s:abc")), // non-numeric
		base64.RawURLEncoding.EncodeToString([]byte("5")),     // no prefix
	}
	for _, bad := range garbage {
		if _, ok := decodeSeqCursor(bad); ok {
			t.Fatalf("expected reject for %q", bad)
		}
	}
}

func TestRunEventsCursorPaginatesFullPages(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "local-user", Title: "cursor"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	run, err := service.CreateRun(ctx, runcontrol.CreateRunRequest{
		ThreadID: thread.ThreadID,
		UserID:   "local-user",
		Goal:     "cursor",
		Messages: []domain.ThreadMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	// Seed events beyond the auto run.accepted one so total > one page.
	for i := 0; i < 5; i++ {
		if _, err := mem.AppendRunEvent(ctx, domain.AppendRunEventInput{
			RunID:     run.RunID,
			EventKind: "message.delta",
		}); err != nil {
			t.Fatalf("AppendRunEvent: %v", err)
		}
	}

	getPage := func(query string) runEventsResponse {
		req := httptest.NewRequest(http.MethodGet, "/v2/runs/"+run.RunID+"/events?"+query, nil).WithContext(ctx)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d for %q: %s", rec.Code, query, rec.Body.String())
		}
		var page runEventsResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &page); err != nil {
			t.Fatalf("decode: %v", err)
		}
		return page
	}

	// A forward drain begins at after_sequence=0; a full page returns a cursor + has_more.
	page1 := getPage("after_sequence=0&limit=2")
	if page1.Count != 2 || !page1.HasMore || page1.NextCursor == "" {
		t.Fatalf("page1 = count=%d has_more=%v cursor=%q", page1.Count, page1.HasMore, page1.NextCursor)
	}
	lastSeq1 := page1.Events[len(page1.Events)-1].Sequence

	// Second page via the opaque ?cursor= starts strictly after page1's last sequence.
	page2 := getPage("limit=2&cursor=" + url.QueryEscape(page1.NextCursor))
	if page2.Count == 0 || page2.Events[0].Sequence <= lastSeq1 {
		t.Fatalf("page2 did not advance past seq %d: first=%d", lastSeq1, func() int64 {
			if len(page2.Events) > 0 {
				return page2.Events[0].Sequence
			}
			return -1
		}())
	}

	// Drain to the final (partial/empty) page: no cursor, has_more omitted/false. Total events =
	// run.accepted + 5 = 6 -> pages of 2 give exactly 3 full pages, so the 4th page is empty.
	cursor := page1.NextCursor
	pages := 1
	for {
		page := getPage("limit=2&cursor=" + url.QueryEscape(cursor))
		if !page.HasMore {
			if page.NextCursor != "" {
				t.Fatalf("final page must not carry a cursor, got %q", page.NextCursor)
			}
			break
		}
		cursor = page.NextCursor
		pages++
		if pages > 10 {
			t.Fatalf("pagination did not terminate")
		}
	}
}
