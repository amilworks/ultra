package httpapi

import (
	"context"
	"log/slog"
	"time"
)

type NoteProposalExpiryStore interface {
	ExpireNoteAppendProposals(context.Context, time.Time, int) (int, error)
	ExpireNoteReadGrants(context.Context, time.Time, int) (int, error)
}

// RunNoteProposalExpiryGC guarantees that exact proposed Note text is erased
// shortly after its approval window closes. This is independent of the
// opt-in resource-retention collector because proposal expiry is a privacy
// boundary, not an operator-controlled deletion policy.
func RunNoteProposalExpiryGC(ctx context.Context, store NoteProposalExpiryStore, interval time.Duration, batch int) {
	if interval <= 0 {
		interval = time.Minute
	}
	if batch <= 0 {
		batch = 200
	}
	sweepProposals := func() {
		for {
			expired, err := store.ExpireNoteAppendProposals(ctx, time.Now().UTC(), batch)
			if err != nil {
				if ctx.Err() == nil {
					slog.WarnContext(ctx, "note proposal expiry sweep failed", "error", err)
				}
				return
			}
			if expired < batch {
				return
			}
		}
	}
	sweepReadGrants := func() {
		for {
			expired, err := store.ExpireNoteReadGrants(ctx, time.Now().UTC(), batch)
			if err != nil {
				if ctx.Err() == nil {
					slog.WarnContext(ctx, "note read-grant expiry sweep failed", "error", err)
				}
				return
			}
			if expired < batch {
				return
			}
		}
	}
	sweep := func() {
		sweepProposals()
		sweepReadGrants()
	}
	sweep()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			sweep()
		}
	}
}
