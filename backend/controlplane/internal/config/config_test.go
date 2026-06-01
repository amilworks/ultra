package config

import (
	"strings"
	"testing"
)

func TestLoadDefaults(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_APP_VERSION", "test-version")
	cfg := Load()
	if cfg.Environment != "development" {
		t.Fatalf("Environment = %q, want development default", cfg.Environment)
	}
	if cfg.AppVersion != "test-version" {
		t.Fatalf("AppVersion = %q", cfg.AppVersion)
	}
	if cfg.HTTPAddr == "" {
		t.Fatalf("HTTPAddr must have default")
	}
	if cfg.NATSRareSpotJobsSubject != "ultra.runs.rarespot.jobs" {
		t.Fatalf("NATSRareSpotJobsSubject = %q, want RareSpot subject", cfg.NATSRareSpotJobsSubject)
	}
	if cfg.NATSCancelSubject != "ultra.runs.cancel" {
		t.Fatalf("NATSCancelSubject = %q, want cancel subject", cfg.NATSCancelSubject)
	}
	if cfg.NATSWorkerDurable != "ultra-deepagents-worker" {
		t.Fatalf("NATSWorkerDurable = %q, want Deep Agents durable", cfg.NATSWorkerDurable)
	}
	if cfg.NATSRareSpotWorkerDurable != "rarespot-ecology-worker" {
		t.Fatalf("NATSRareSpotWorkerDurable = %q, want RareSpot durable", cfg.NATSRareSpotWorkerDurable)
	}
	if cfg.ArtifactRoot == "" {
		t.Fatalf("ArtifactRoot must have default")
	}
	if !cfg.DevAdminEnabled {
		t.Fatalf("DevAdminEnabled should default on for the local control-plane dashboard")
	}
	if !cfg.RunRecoveryEnabled {
		t.Fatalf("RunRecoveryEnabled should default on so expired worker leases are recovered")
	}
	if cfg.RunRecoveryInterval <= 0 {
		t.Fatalf("RunRecoveryInterval must have a positive default")
	}
	if cfg.RunRecoveryBatchLimit <= 0 {
		t.Fatalf("RunRecoveryBatchLimit must have a positive default")
	}
}

func TestLoadDevAdminEnabledCanBeDisabled(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_DEV_ADMIN_ENABLED", "false")
	cfg := Load()
	if cfg.DevAdminEnabled {
		t.Fatalf("DevAdminEnabled = true, want false when ULTRA_CONTROL_DEV_ADMIN_ENABLED=false")
	}
}

func TestLoadRunRecoveryCanBeConfigured(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_RUN_RECOVERY_ENABLED", "false")
	t.Setenv("ULTRA_CONTROL_RUN_RECOVERY_INTERVAL_SECONDS", "17")
	t.Setenv("ULTRA_CONTROL_RUN_RECOVERY_BATCH_LIMIT", "123")

	cfg := Load()
	if cfg.RunRecoveryEnabled {
		t.Fatalf("RunRecoveryEnabled = true, want false")
	}
	if cfg.RunRecoveryInterval.String() != "17s" {
		t.Fatalf("RunRecoveryInterval = %s, want 17s", cfg.RunRecoveryInterval)
	}
	if cfg.RunRecoveryBatchLimit != 123 {
		t.Fatalf("RunRecoveryBatchLimit = %d, want 123", cfg.RunRecoveryBatchLimit)
	}
}

func TestLoadUsesRunStorePathAsDatabaseFallback(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_DATABASE_URL", "")
	t.Setenv("RUN_STORE_PATH", "postgresql://postgres:postgres@127.0.0.1:55432/ultra")

	cfg := Load()
	if cfg.DatabaseURL != "postgresql://postgres:postgres@127.0.0.1:55432/ultra" {
		t.Fatalf("DatabaseURL = %q, want RUN_STORE_PATH fallback", cfg.DatabaseURL)
	}
}

func TestLoadPrefersControlDatabaseURLOverRunStorePath(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:55432/control")
	t.Setenv("RUN_STORE_PATH", "postgresql://postgres:postgres@127.0.0.1:55432/legacy")

	cfg := Load()
	if cfg.DatabaseURL != "postgresql://postgres:postgres@127.0.0.1:55432/control" {
		t.Fatalf("DatabaseURL = %q, want ULTRA_CONTROL_DATABASE_URL", cfg.DatabaseURL)
	}
}

func TestValidateRequiresDurableBackendsInProduction(t *testing.T) {
	cfg := Config{Environment: "production"}

	err := cfg.Validate()
	if err == nil {
		t.Fatalf("Validate() error = nil, want production backend error")
	}
	text := err.Error()
	for _, want := range []string{"ULTRA_CONTROL_DATABASE_URL", "RUN_STORE_PATH", "ULTRA_CONTROL_NATS_URL"} {
		if !strings.Contains(text, want) {
			t.Fatalf("Validate() error = %q, want mention %s", text, want)
		}
	}
}

func TestValidateAllowsProductionWithPostgresAndNATS(t *testing.T) {
	cfg := Config{
		Environment: "production",
		DatabaseURL: "postgresql://postgres:postgres@127.0.0.1:55432/ultra",
		NATSURL:     "nats://127.0.0.1:4222",
	}

	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() error = %v, want nil", err)
	}
}
