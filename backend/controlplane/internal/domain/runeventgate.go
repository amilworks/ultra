package domain

import "context"

// runEventGateBypassKey marks a context whose run-event ingest should skip
// the source_sequence predecessor gate. The eventbus ingest worker sets it
// after a bounded wait for a predecessor that never arrived (a permanent
// producer-side gap); storing the successor with a gap is strictly better
// than terminating it, because readers order by the control-plane-assigned
// sequence_number, not source_sequence.
type runEventGateBypassKey struct{}

func WithRunEventGateBypass(ctx context.Context) context.Context {
	return context.WithValue(ctx, runEventGateBypassKey{}, true)
}

func RunEventGateBypassed(ctx context.Context) bool {
	bypassed, ok := ctx.Value(runEventGateBypassKey{}).(bool)
	return ok && bypassed
}
