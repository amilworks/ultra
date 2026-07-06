package httpapi

import (
	"fmt"
	"log/slog"
	"net/http"
	"runtime/debug"
)

// recoverPanics converts a panic in any downstream handler into a structured log
// line plus a clean JSON 500, replacing net/http's default handling — which lets
// the process survive but drops the client's connection without a response and
// dumps a raw stack trace to stderr. With this, a poison request that trips a
// nil-deref or out-of-range panic in a handler fails as one bounded 500 for THAT
// request: the service stays up, the socket isn't dropped, and no stack trace
// leaks to the client (C2 robustness: "a poison input affects only that request";
// C6 observability: the panic is logged with method/path/stack via slog).
//
// http.ErrAbortHandler is re-panicked so the server's deliberate stream-abort path
// (flushing/SSE handlers that abort a hijacked or streaming response) is preserved.
// guardedResponseWriter makes WriteHeader idempotent: the first status wins and
// later WriteHeader calls are silently ignored instead of tripping net/http's
// "superfluous response.WriteHeader call" log (and instead of a second status
// clobbering the response). This matches the behaviour of chi's and the stdlib's
// own recorder wrappers. A handler reaching an error path after it has already
// begun the response — e.g. writeJSON after a stream started, or recoverPanics'
// best-effort 500 after a partial write — is the source of that log. Only
// WriteHeader is guarded; Header, Write, and Flush (SSE) delegate straight
// through. Flush is always present so an SSE handler's w.(http.Flusher) check
// still succeeds; it no-ops when the underlying writer is not flushable.
type guardedResponseWriter struct {
	http.ResponseWriter
	wroteHeader bool
}

func (g *guardedResponseWriter) WriteHeader(status int) {
	if g.wroteHeader {
		return
	}
	g.wroteHeader = true
	g.ResponseWriter.WriteHeader(status)
}

func (g *guardedResponseWriter) Write(b []byte) (int, error) {
	g.wroteHeader = true
	return g.ResponseWriter.Write(b)
}

func (g *guardedResponseWriter) Flush() {
	if flusher, ok := g.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// guardResponseWriter wraps every response so a duplicate WriteHeader is a no-op
// rather than a superfluous-header log + response corruption. Wire it right after
// recoverPanics so the recovery path's fallback write is also guarded.
func guardResponseWriter(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		next.ServeHTTP(&guardedResponseWriter{ResponseWriter: w}, r)
	})
}

func recoverPanics(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			rec := recover()
			if rec == nil {
				return
			}
			if rec == http.ErrAbortHandler {
				panic(rec) // honor a deliberate handler abort; not an error to mask
			}
			slog.ErrorContext(r.Context(), "recovered panic in HTTP handler",
				"method", r.Method,
				"path", r.URL.Path,
				"panic", fmt.Sprintf("%v", rec),
				"stack", string(debug.Stack()),
			)
			// Best-effort clean error. If the handler already began streaming the
			// response before panicking, WriteHeader is a no-op (and writeJSON's
			// encode may fail) — guard so a secondary panic here can't escape.
			defer func() { _ = recover() }()
			writeError(w, http.StatusInternalServerError, fmt.Errorf("internal error"))
		}()
		next.ServeHTTP(w, r)
	})
}
