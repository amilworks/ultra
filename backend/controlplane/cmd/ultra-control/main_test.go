package main

import "testing"

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
