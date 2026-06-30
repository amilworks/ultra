package store

import (
	"testing"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
)

func msgs(ids ...string) []domain.ThreadMessage {
	out := make([]domain.ThreadMessage, len(ids))
	for i, id := range ids {
		out[i] = domain.ThreadMessage{MessageID: id}
	}
	return out
}

func first(page []domain.ThreadMessage) string {
	if len(page) == 0 {
		return ""
	}
	return page[0].MessageID
}

func last(page []domain.ThreadMessage) string {
	if len(page) == 0 {
		return ""
	}
	return page[len(page)-1].MessageID
}

func TestPageThreadMessagesTail(t *testing.T) {
	all := msgs("m0", "m1", "m2", "m3", "m4", "m5", "m6") // ascending

	// Newest tail (no cursor), limit 3 -> m4,m5,m6 and older remain.
	page, hasMore := pageThreadMessagesTail(all, "", 3)
	if first(page) != "m4" || last(page) != "m6" || !hasMore {
		t.Fatalf("tail page = %v..%v hasMore=%v, want m4..m6 true", first(page), last(page), hasMore)
	}

	// Load earlier before m4 -> m1,m2,m3 and older remain.
	page, hasMore = pageThreadMessagesTail(all, "m4", 3)
	if first(page) != "m1" || last(page) != "m3" || !hasMore {
		t.Fatalf("before-m4 page = %v..%v hasMore=%v, want m1..m3 true", first(page), last(page), hasMore)
	}

	// Load earlier before m1 -> m0 only, no older remain.
	page, hasMore = pageThreadMessagesTail(all, "m1", 3)
	if first(page) != "m0" || last(page) != "m0" || hasMore {
		t.Fatalf("before-m1 page = %v..%v hasMore=%v, want m0..m0 false", first(page), last(page), hasMore)
	}

	// Window >= total -> everything, no more.
	page, hasMore = pageThreadMessagesTail(all, "", 100)
	if len(page) != 7 || hasMore {
		t.Fatalf("full page len=%d hasMore=%v, want 7 false", len(page), hasMore)
	}
}
