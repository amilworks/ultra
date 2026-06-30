package httpapi

import (
	"encoding/base64"
	"strconv"
	"strings"
)

// Keyset-pagination cursor convention shared by the list endpoints (run events today, thread
// messages next). A cursor is an opaque base64url token the client treats as a black box and
// passes back verbatim via ?cursor=. Opacity lets the underlying key shape evolve — a bare
// monotonic sequence today, a composite (created_at,id) for messages — without changing the
// client contract. A malformed or foreign token degrades to "from the start" rather than
// erroring the request, so a stale cursor never wedges a client.

const seqCursorPrefix = "s:"

// encodeSeqCursor encodes a monotonic sequence keyset position into an opaque cursor token.
func encodeSeqCursor(seq int64) string {
	return base64.RawURLEncoding.EncodeToString([]byte(seqCursorPrefix + strconv.FormatInt(seq, 10)))
}

// decodeSeqCursor decodes a sequence cursor; ok=false for any malformed/foreign token.
func decodeSeqCursor(token string) (seq int64, ok bool) {
	raw, err := base64.RawURLEncoding.DecodeString(strings.TrimSpace(token))
	if err != nil {
		return 0, false
	}
	payload := string(raw)
	if !strings.HasPrefix(payload, seqCursorPrefix) {
		return 0, false
	}
	seq, err = strconv.ParseInt(payload[len(seqCursorPrefix):], 10, 64)
	if err != nil || seq < 0 {
		return 0, false
	}
	return seq, true
}
