package httpapi

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRecoverPanicsReturnsCleanJSON500(t *testing.T) {
	h := recoverPanics(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		panic("boom: simulated nil pointer dereference")
	}))
	rec := httptest.NewRecorder()
	// Must not propagate the panic out of ServeHTTP.
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v2/uploads/x/viewer", nil))

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"error"`) || !strings.Contains(body, "internal error") {
		t.Fatalf("body = %q, want clean JSON error", body)
	}
	// The client must never see the panic value or a stack trace.
	if strings.Contains(body, "boom") || strings.Contains(body, "goroutine") {
		t.Fatalf("body leaked panic/stack detail: %q", body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("content-type = %q, want application/json", ct)
	}
}

func TestRecoverPanicsHappyPathUnchanged(t *testing.T) {
	h := recoverPanics(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTeapot)
		_, _ = w.Write([]byte("ok"))
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))

	if rec.Code != http.StatusTeapot || rec.Body.String() != "ok" {
		t.Fatalf("happy path altered: code=%d body=%q", rec.Code, rec.Body.String())
	}
}

func TestRecoverPanicsRepanicsAbortHandler(t *testing.T) {
	// http.ErrAbortHandler is the sentinel net/http uses to abort a response
	// silently (streaming/flush paths). The recoverer must NOT swallow it.
	h := recoverPanics(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		panic(http.ErrAbortHandler)
	}))
	defer func() {
		if got := recover(); got != http.ErrAbortHandler {
			t.Fatalf("recover = %v, want ErrAbortHandler re-panicked", got)
		}
	}()
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/", nil))
	t.Fatal("expected ErrAbortHandler to propagate")
}
