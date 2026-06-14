package config

import (
	"strings"
	"testing"
	"time"
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
	if cfg.ReadTimeout != 0 {
		t.Fatalf("ReadTimeout = %s, want no whole-body default timeout", cfg.ReadTimeout)
	}
	if cfg.ReadHeaderTimeout <= 0 {
		t.Fatalf("ReadHeaderTimeout must have a positive default")
	}
	if cfg.NATSRareSpotJobsSubject != "ultra.runs.rarespot.jobs" {
		t.Fatalf("NATSRareSpotJobsSubject = %q, want RareSpot subject", cfg.NATSRareSpotJobsSubject)
	}
	if cfg.NATSDataAgentJobsSubject != "ultra.data_agent.jobs" {
		t.Fatalf("NATSDataAgentJobsSubject = %q, want Data Agent subject", cfg.NATSDataAgentJobsSubject)
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
	if cfg.NATSDataAgentWorkerDurable != "ultra-data-agent-worker" {
		t.Fatalf("NATSDataAgentWorkerDurable = %q, want Data Agent durable", cfg.NATSDataAgentWorkerDurable)
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

func TestLoadBisqueIntegrationConfig(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_BISQUE_ROOT_URL", "https://bisque.example.org")
	t.Setenv("ULTRA_CONTROL_BISQUE_USERNAME", "ada")
	t.Setenv("ULTRA_CONTROL_BISQUE_PASSWORD", "secret")
	t.Setenv("ULTRA_CONTROL_BISQUE_MAX_IMPORT_BYTES", "12345")
	t.Setenv("ULTRA_CONTROL_SECRET_ENCRYPTION_KEY", "01234567890123456789012345678901")
	t.Setenv("ULTRA_CONTROL_SECRET_ENCRYPTION_KEY_ID", "bisque-test-key")

	cfg := Load()
	if cfg.BisqueRootURL != "https://bisque.example.org" {
		t.Fatalf("BisqueRootURL = %q", cfg.BisqueRootURL)
	}
	if cfg.BisqueUsername != "ada" || cfg.BisquePassword != "secret" {
		t.Fatalf("Bisque credentials were not loaded")
	}
	if cfg.BisqueMaxImportBytes != 12345 {
		t.Fatalf("BisqueMaxImportBytes = %d, want 12345", cfg.BisqueMaxImportBytes)
	}
	if cfg.SecretEncryptionKey != "01234567890123456789012345678901" || cfg.SecretEncryptionKeyID != "bisque-test-key" {
		t.Fatalf("secret encryption config = %q/%q, want explicit test key", cfg.SecretEncryptionKey, cfg.SecretEncryptionKeyID)
	}
}

func TestLoadBisqueIntegrationConfigAcceptsSharedBisqueUserFallback(t *testing.T) {
	t.Setenv("BISQUE_ROOT", "https://bisque.example.org")
	t.Setenv("BISQUE_USER", "shared-user")
	t.Setenv("BISQUE_PASSWORD", "shared-secret")

	cfg := Load()
	if cfg.BisqueRootURL != "https://bisque.example.org" {
		t.Fatalf("BisqueRootURL = %q", cfg.BisqueRootURL)
	}
	if cfg.BisqueUsername != "shared-user" || cfg.BisquePassword != "shared-secret" {
		t.Fatalf("BisQue fallback credentials = %q/%t, want shared env values", cfg.BisqueUsername, cfg.BisquePassword != "")
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

func TestLoadPostgresPoolConfig(t *testing.T) {
	t.Setenv("ULTRA_CONTROL_DATABASE_MAX_CONNS", "12")
	t.Setenv("ULTRA_CONTROL_DATABASE_MIN_CONNS", "3")
	t.Setenv("ULTRA_CONTROL_DATABASE_STATEMENT_TIMEOUT_SECONDS", "45")

	cfg := Load()
	if cfg.DatabaseMaxConns != 12 {
		t.Fatalf("DatabaseMaxConns = %d, want 12", cfg.DatabaseMaxConns)
	}
	if cfg.DatabaseMinConns != 3 {
		t.Fatalf("DatabaseMinConns = %d, want 3", cfg.DatabaseMinConns)
	}
	if cfg.DatabaseStatementTimeout != 45*time.Second {
		t.Fatalf("DatabaseStatementTimeout = %v, want 45s", cfg.DatabaseStatementTimeout)
	}
}

func TestLoadDefaultsDatabaseStatementTimeoutOff(t *testing.T) {
	// Off by default (like ReadTimeout/WriteTimeout) so adding the knob changes no
	// existing deployment's behavior until an operator opts in.
	t.Setenv("ULTRA_CONTROL_DATABASE_STATEMENT_TIMEOUT_SECONDS", "")
	if got := Load().DatabaseStatementTimeout; got != 0 {
		t.Fatalf("default DatabaseStatementTimeout = %v, want 0 (disabled)", got)
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
		Environment:          "production",
		DatabaseURL:          "postgresql://postgres:postgres@127.0.0.1:55432/ultra",
		NATSURL:              "nats://127.0.0.1:4222",
		AuthProvider:         "workos",
		WorkOSClientID:       "client_test",
		WorkOSAPIKey:         "sk_test",
		WorkOSRedirectURI:    "https://ultra.example.org/v2/auth/workos/callback",
		WorkOSCookiePassword: "workos-cookie-password-for-production-test",
	}

	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() error = %v, want nil", err)
	}
}

func TestValidateRequiresWorkOSInProduction(t *testing.T) {
	cfg := Config{
		Environment: "production",
		DatabaseURL: "postgresql://postgres:postgres@127.0.0.1:55432/ultra",
		NATSURL:     "nats://127.0.0.1:4222",
	}

	err := cfg.Validate()
	if err == nil {
		t.Fatalf("Validate() error = nil, want WorkOS auth config error")
	}
	text := err.Error()
	for _, want := range []string{
		"ULTRA_CONTROL_AUTH_PROVIDER=workos",
		"ULTRA_CONTROL_WORKOS_CLIENT_ID",
		"ULTRA_CONTROL_WORKOS_API_KEY",
		"ULTRA_CONTROL_WORKOS_REDIRECT_URI",
		"ULTRA_CONTROL_WORKOS_COOKIE_PASSWORD",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("Validate() error = %q, want mention %s", text, want)
		}
	}
}

func TestValidateAllowsExplicitDevAuthInProduction(t *testing.T) {
	cfg := Config{
		Environment:              "production",
		DatabaseURL:              "postgresql://postgres:postgres@127.0.0.1:55432/ultra",
		NATSURL:                  "nats://127.0.0.1:4222",
		AuthProvider:             "dev",
		AllowDevAuthInProduction: true,
		DevAdminEnabled:          true,
		SecretEncryptionKey:      "01234567890123456789012345678901",
		SecretEncryptionKeyID:    "dev-auth-production-cutover-test",
	}

	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() error = %v, want nil with explicit dev-auth production override", err)
	}
}

func TestValidateRequiresSecretEncryptionKeyWhenBisqueIsConfiguredInProduction(t *testing.T) {
	cfg := Config{
		Environment:          "production",
		DatabaseURL:          "postgresql://postgres:postgres@127.0.0.1:55432/ultra",
		NATSURL:              "nats://127.0.0.1:4222",
		BisqueRootURL:        "https://bisque.example.org",
		AuthProvider:         "workos",
		WorkOSClientID:       "client_test",
		WorkOSAPIKey:         "sk_test",
		WorkOSRedirectURI:    "https://ultra.example.org/v2/auth/workos/callback",
		WorkOSCookiePassword: "workos-cookie-password-for-production-test",
	}

	err := cfg.Validate()
	if err == nil {
		t.Fatalf("Validate() error = nil, want BisQue secret-key error")
	}
	if !strings.Contains(err.Error(), "ULTRA_CONTROL_SECRET_ENCRYPTION_KEY") {
		t.Fatalf("Validate() error = %q, want secret encryption key requirement", err.Error())
	}

	cfg.SecretEncryptionKey = "01234567890123456789012345678901"
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() error = %v, want nil with BisQue secret key", err)
	}
}
