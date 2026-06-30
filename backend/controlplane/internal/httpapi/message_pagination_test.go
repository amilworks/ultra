package httpapi

import (
	"context"
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

func TestListThreadMessagesPaginatesEarlierPages(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	service := runcontrol.NewService(mem, bus)
	router := NewRouter(ServerDeps{Version: "test-version", Runs: service, Store: mem, Bus: bus})

	thread, err := service.CreateThread(ctx, runcontrol.CreateThreadRequest{UserID: "local-user", Title: "paginate"})
	if err != nil {
		t.Fatalf("CreateThread: %v", err)
	}
	for i := 0; i < 6; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		if _, err := mem.AppendThreadMessage(ctx, domain.ThreadMessage{
			ThreadID:  thread.ThreadID,
			MessageID: "msg-" + string(rune('a'+i)),
			Role:      role,
			Content:   "m",
		}); err != nil {
			t.Fatalf("AppendThreadMessage: %v", err)
		}
	}

	get := func(query string) threadMessagesResponse {
		path := "/v2/threads/" + thread.ThreadID + "/messages"
		if query != "" {
			path += "?" + query
		}
		req := httptest.NewRequest(http.MethodGet, path, nil).WithContext(ctx)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d for %q: %s", rec.Code, query, rec.Body.String())
		}
		var resp threadMessagesResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode: %v", err)
		}
		return resp
	}

	// No params: full thread (back-compat), no pagination fields.
	full := get("")
	if full.Count != 6 || full.HasMore || full.NextCursor != "" {
		t.Fatalf("full = count %d has_more %v cursor %q", full.Count, full.HasMore, full.NextCursor)
	}

	// limit=2: newest tail (msg-e, msg-f), older remain, cursor = oldest in page (msg-e).
	page1 := get("limit=2")
	if page1.Count != 2 || !page1.HasMore || page1.NextCursor != "msg-e" {
		t.Fatalf("page1 = count %d has_more %v cursor %q", page1.Count, page1.HasMore, page1.NextCursor)
	}
	if page1.Messages[0].MessageID != "msg-e" || page1.Messages[1].MessageID != "msg-f" {
		t.Fatalf("page1 ids = %s,%s", page1.Messages[0].MessageID, page1.Messages[1].MessageID)
	}

	// Load earlier before the cursor.
	page2 := get("limit=2&before=" + url.QueryEscape(page1.NextCursor))
	if page2.Count != 2 || !page2.HasMore || page2.Messages[0].MessageID != "msg-c" {
		t.Fatalf("page2 = count %d has_more %v first %s", page2.Count, page2.HasMore, page2.Messages[0].MessageID)
	}

	// The final (oldest) page has no more.
	page3 := get("limit=2&before=" + url.QueryEscape(page2.NextCursor))
	if page3.Count != 2 || page3.HasMore || page3.NextCursor != "" || page3.Messages[0].MessageID != "msg-a" {
		t.Fatalf("page3 = count %d has_more %v cursor %q first %s", page3.Count, page3.HasMore, page3.NextCursor, page3.Messages[0].MessageID)
	}
}
