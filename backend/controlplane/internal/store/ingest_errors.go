package store

import (
	"context"
	"errors"
	"net"
	"strings"

	"github.com/jackc/pgx/v5/pgconn"
)

// IsTransientIngestError reports whether an ingest failure is an
// infrastructure condition that will heal on its own (Postgres restart,
// failover, pool exhaustion, network blips). The ingest retry loop treats
// these as retry-indefinitely: terminating the message would permanently
// lose a real event, and with MaxAckPending=1 nothing else progresses during
// the outage anyway, so waiting costs nothing extra.
func IsTransientIngestError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	if pgconn.Timeout(err) {
		return true
	}
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		code := pgErr.Code
		// Class 08: connection exception. Class 57: operator intervention
		// (admin shutdown, crash shutdown). Class 53: insufficient resources
		// (too many connections, out of memory). 40001/40P01: serialization
		// failure / deadlock — safe to retry by definition.
		if strings.HasPrefix(code, "08") || strings.HasPrefix(code, "57") || strings.HasPrefix(code, "53") {
			return true
		}
		if code == "40001" || code == "40P01" {
			return true
		}
	}
	// pgx surfaces "closed pool" / broken conn conditions as plain errors.
	message := err.Error()
	if strings.Contains(message, "closed pool") || strings.Contains(message, "conn closed") {
		return true
	}
	return false
}

// IsDeterministicIngestError reports whether an ingest failure is inherent to
// the event itself (bad data, schema violation, statement limits) and will
// fail identically on every retry. These get a short bounded retry, then the
// message is terminated to keep the partition flowing. Note: 23505 unique
// collisions never reach this classification — the append path resolves them
// as duplicate/drop outcomes upstream.
func IsDeterministicIngestError(err error) bool {
	if err == nil {
		return false
	}
	var pgErr *pgconn.PgError
	if !errors.As(err, &pgErr) {
		return false
	}
	code := pgErr.Code
	// 22: data exception. 42: syntax/access-rule violation. 54: program
	// limits exceeded. 23: integrity violations that escaped the upstream
	// conflict resolution (e.g. FK violations for vanished runs).
	return strings.HasPrefix(code, "22") ||
		strings.HasPrefix(code, "42") ||
		strings.HasPrefix(code, "54") ||
		strings.HasPrefix(code, "23")
}
