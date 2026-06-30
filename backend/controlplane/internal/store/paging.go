package store

import "github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"

const defaultThreadMessagePageSize = 30

// pageThreadMessagesTail returns the most-recent `limit` messages from an ascending message slice,
// ending just BEFORE `beforeMessageID` (exclusive) — i.e. the "load earlier" page for infinite
// scroll-up. The page is returned in ascending (chronological) order, plus whether older messages
// remain before it. An empty beforeMessageID pages the newest tail. The API contract (limit +
// before-cursor + hasMore) is keyset-friendly, so the in-memory slice here can be swapped for a
// keyset DB query without changing callers once threads grow large enough to warrant it.
func pageThreadMessagesTail(
	all []domain.ThreadMessage,
	beforeMessageID string,
	limit int,
) (page []domain.ThreadMessage, hasMore bool) {
	if limit <= 0 {
		limit = defaultThreadMessagePageSize
	}
	end := len(all)
	if beforeMessageID != "" {
		for i := range all {
			if all[i].MessageID == beforeMessageID {
				end = i
				break
			}
		}
	}
	start := end - limit
	hasMore = start > 0
	if start < 0 {
		start = 0
	}
	page = append([]domain.ThreadMessage(nil), all[start:end]...)
	return page, hasMore
}
