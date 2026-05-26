package config

import "testing"

func TestLoadDefaults(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_APP_VERSION", "test-version")
	cfg := Load()
	if cfg.AppVersion != "test-version" {
		t.Fatalf("AppVersion = %q", cfg.AppVersion)
	}
	if cfg.HTTPAddr == "" {
		t.Fatalf("HTTPAddr must have default")
	}
}
