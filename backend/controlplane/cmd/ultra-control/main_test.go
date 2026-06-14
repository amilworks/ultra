package main

import (
	"net/http"
	"testing"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
)

func TestControlCommandRecognizesMigrate(t *testing.T) {
	t.Parallel()

	if got := controlCommand([]string{"ultra-control", "migrate"}); got != "migrate" {
		t.Fatalf("controlCommand() = %q, want migrate", got)
	}
}

func TestControlCommandDefaultsToServe(t *testing.T) {
	t.Parallel()

	if got := controlCommand([]string{"ultra-control"}); got != "serve" {
		t.Fatalf("controlCommand() = %q, want serve", got)
	}
}

func TestNewControlHTTPServerUsesHeaderTimeout(t *testing.T) {
	t.Parallel()

	server := newControlHTTPServer(config.Config{
		HTTPAddr:          "127.0.0.1:0",
		ReadHeaderTimeout: 7 * time.Second,
		ReadTimeout:       0,
		WriteTimeout:      0,
		IdleTimeout:       30 * time.Second,
	}, http.NewServeMux())

	if server.ReadHeaderTimeout != 7*time.Second {
		t.Fatalf("ReadHeaderTimeout = %s, want 7s", server.ReadHeaderTimeout)
	}
	if server.ReadTimeout != 0 {
		t.Fatalf("ReadTimeout = %s, want no whole-body timeout", server.ReadTimeout)
	}
}
