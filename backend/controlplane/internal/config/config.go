package config

import (
	"errors"
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	AppName                    string
	AppVersion                 string
	Environment                string
	HTTPAddr                   string
	ReadHeaderTimeout          time.Duration
	ReadTimeout                time.Duration
	WriteTimeout               time.Duration
	IdleTimeout                time.Duration
	DatabaseURL                string
	DatabaseMaxConns           int
	DatabaseMinConns           int
	DatabaseStatementTimeout   time.Duration
	NATSURL                    string
	NATSStream                 string
	NATSJobsSubject            string
	NATSDataAgentJobsSubject   string
	NATSEventsSubject          string
	NATSCancelSubject          string
	NATSEventConsumer          string
	NATSEventIngestConcurrency int
	NATSWorkerDurable          string
	NATSDataAgentWorkerDurable string
	ArtifactRoot               string
	UploadRoot                 string
	ImageServiceURL            string
	BisqueRootURL              string
	BisqueUsername             string
	BisquePassword             string
	BisqueMaxImportBytes       int64
	SecretEncryptionKey        string
	SecretEncryptionKeyID      string
	AuthProvider               string
	WorkOSClientID             string
	WorkOSAPIKey               string
	WorkOSRedirectURI          string
	WorkOSPostLoginRedirectURI string
	WorkOSLogoutRedirectURI    string
	WorkOSCookiePassword       string
	WorkOSBaseURL              string
	WorkOSCookieSecure         bool
	DevAdminEnabled            bool
	AllowDevAuthInProduction   bool
	RunRecoveryEnabled         bool
	RunRecoveryInterval        time.Duration
	RunRecoveryBatchLimit      int
	RetentionGCEnabled         bool
	RetentionGCInterval        time.Duration
	RetentionGCBatch           int
	WorkerToken                string
}

func Load() Config {
	return Config{
		AppName:           envString("ULTRA_CONTROL_APP_NAME", "BisQue Ultra Control Plane"),
		AppVersion:        envString("ULTRA_CONTROL_APP_VERSION", "dev"),
		Environment:       strings.ToLower(envString("ULTRA_CONTROL_ENVIRONMENT", envString("ENVIRONMENT", "development"))),
		HTTPAddr:          envString("ULTRA_CONTROL_HTTP_ADDR", "127.0.0.1:8088"),
		ReadHeaderTimeout: envDurationSeconds("ULTRA_CONTROL_READ_HEADER_TIMEOUT_SECONDS", 10),
		ReadTimeout:       envDurationSeconds("ULTRA_CONTROL_READ_TIMEOUT_SECONDS", 0),
		WriteTimeout:      envDurationSeconds("ULTRA_CONTROL_WRITE_TIMEOUT_SECONDS", 0),
		IdleTimeout:       envDurationSeconds("ULTRA_CONTROL_IDLE_TIMEOUT_SECONDS", 120),
		DatabaseURL:       envString("ULTRA_CONTROL_DATABASE_URL", envString("RUN_STORE_PATH", "")),
		DatabaseMaxConns:  envInt("ULTRA_CONTROL_DATABASE_MAX_CONNS", 8),
		DatabaseMinConns:  envInt("ULTRA_CONTROL_DATABASE_MIN_CONNS", 0),
		// Per-query server-side timeout for the SERVING pool so one stuck/runaway query
		// can't hold a connection from the small pool forever. Off by default (0), like
		// ReadTimeout/WriteTimeout — operators opt in per their workload. Migrations are
		// exempt (slow DDL is legitimate).
		DatabaseStatementTimeout:   envDurationSeconds("ULTRA_CONTROL_DATABASE_STATEMENT_TIMEOUT_SECONDS", 0),
		NATSURL:                    envString("ULTRA_CONTROL_NATS_URL", ""),
		NATSStream:                 envString("ULTRA_CONTROL_NATS_STREAM", "ULTRA_RUNS"),
		NATSJobsSubject:            envString("ULTRA_CONTROL_NATS_JOBS_SUBJECT", "ultra.runs.jobs"),
		NATSDataAgentJobsSubject:   envString("ULTRA_CONTROL_NATS_DATA_AGENT_JOBS_SUBJECT", "ultra.data_agent.jobs"),
		NATSEventsSubject:          envString("ULTRA_CONTROL_NATS_EVENTS_SUBJECT", "ultra.runs.events"),
		NATSCancelSubject:          envString("ULTRA_CONTROL_NATS_CANCEL_SUBJECT", "ultra.runs.cancel"),
		NATSEventConsumer:          envString("ULTRA_CONTROL_NATS_EVENT_CONSUMER", "ultra-control-event-ingest"),
		NATSEventIngestConcurrency: envInt("ULTRA_CONTROL_NATS_EVENT_INGEST_CONCURRENCY", 4),
		NATSWorkerDurable:          envString("ULTRA_CONTROL_NATS_WORKER_DURABLE", "ultra-deepagents-worker"),
		NATSDataAgentWorkerDurable: envString("ULTRA_CONTROL_NATS_DATA_AGENT_WORKER_DURABLE", "ultra-data-agent-worker"),
		ArtifactRoot:               envString("ULTRA_CONTROL_ARTIFACT_ROOT", envString("ARTIFACT_ROOT", "data/artifacts")),
		UploadRoot:                 envString("ULTRA_CONTROL_UPLOAD_ROOT", envString("ULTRA_RESOURCE_ROOT", envString("UPLOAD_STORE_ROOT", "data/uploads"))),
		ImageServiceURL:            envString("ULTRA_CONTROL_IMAGE_SERVICE_URL", ""),
		BisqueRootURL:              envString("ULTRA_CONTROL_BISQUE_ROOT_URL", envString("BISQUE_ROOT", envString("BISQUE_SERVER", ""))),
		BisqueUsername:             envString("ULTRA_CONTROL_BISQUE_USERNAME", envString("BISQUE_USERNAME", envString("BISQUE_USER", ""))),
		BisquePassword:             envString("ULTRA_CONTROL_BISQUE_PASSWORD", envString("BISQUE_PASSWORD", "")),
		BisqueMaxImportBytes:       int64(envInt("ULTRA_CONTROL_BISQUE_MAX_IMPORT_BYTES", 512<<20)),
		SecretEncryptionKey:        envString("ULTRA_CONTROL_SECRET_ENCRYPTION_KEY", envString("ULTRA_SECRET_ENCRYPTION_KEY", "")),
		SecretEncryptionKeyID:      envString("ULTRA_CONTROL_SECRET_ENCRYPTION_KEY_ID", "local-dev-key"),
		AuthProvider:               strings.ToLower(envString("ULTRA_CONTROL_AUTH_PROVIDER", "dev")),
		WorkOSClientID:             envString("ULTRA_CONTROL_WORKOS_CLIENT_ID", ""),
		WorkOSAPIKey:               envString("ULTRA_CONTROL_WORKOS_API_KEY", ""),
		WorkOSRedirectURI:          envString("ULTRA_CONTROL_WORKOS_REDIRECT_URI", ""),
		WorkOSPostLoginRedirectURI: envString("ULTRA_CONTROL_WORKOS_POST_LOGIN_REDIRECT_URI", "/"),
		WorkOSLogoutRedirectURI:    envString("ULTRA_CONTROL_WORKOS_LOGOUT_REDIRECT_URI", "/"),
		WorkOSCookiePassword:       envString("ULTRA_CONTROL_WORKOS_COOKIE_PASSWORD", ""),
		WorkOSBaseURL:              envString("ULTRA_CONTROL_WORKOS_BASE_URL", ""),
		WorkOSCookieSecure:         envBool("ULTRA_CONTROL_WORKOS_COOKIE_SECURE", true),
		DevAdminEnabled:            envBool("ULTRA_CONTROL_DEV_ADMIN_ENABLED", true),
		AllowDevAuthInProduction:   envBool("ULTRA_CONTROL_ALLOW_DEV_AUTH_IN_PRODUCTION", false),
		RunRecoveryEnabled:         envBool("ULTRA_CONTROL_RUN_RECOVERY_ENABLED", true),
		RunRecoveryInterval:        envDurationSeconds("ULTRA_CONTROL_RUN_RECOVERY_INTERVAL_SECONDS", 30),
		RunRecoveryBatchLimit:      envInt("ULTRA_CONTROL_RUN_RECOVERY_BATCH_LIMIT", 1000),
		// Off by default — it permanently deletes resources past their 30-day undelete window.
		RetentionGCEnabled:  envBool("ULTRA_CONTROL_RETENTION_GC_ENABLED", false),
		RetentionGCInterval: envDurationSeconds("ULTRA_CONTROL_RETENTION_GC_INTERVAL_SECONDS", 3600),
		RetentionGCBatch:    envInt("ULTRA_CONTROL_RETENTION_GC_BATCH", 100),
		WorkerToken:         envString("ULTRA_CONTROL_WORKER_TOKEN", ""),
	}
}

func (c Config) Validate() error {
	if c.Environment != "production" {
		return nil
	}
	var missing []string
	if strings.TrimSpace(c.DatabaseURL) == "" {
		missing = append(missing, "ULTRA_CONTROL_DATABASE_URL or RUN_STORE_PATH")
	}
	if strings.TrimSpace(c.NATSURL) == "" {
		missing = append(missing, "ULTRA_CONTROL_NATS_URL")
	}
	if strings.TrimSpace(c.BisqueRootURL) != "" && strings.TrimSpace(c.SecretEncryptionKey) == "" {
		missing = append(missing, "ULTRA_CONTROL_SECRET_ENCRYPTION_KEY")
	}
	devAuthAllowed := c.AllowDevAuthInProduction && strings.ToLower(strings.TrimSpace(c.AuthProvider)) == "dev"
	if strings.ToLower(strings.TrimSpace(c.AuthProvider)) != "workos" && !devAuthAllowed {
		missing = append(missing, "ULTRA_CONTROL_AUTH_PROVIDER=workos")
	}
	if !devAuthAllowed {
		if strings.TrimSpace(c.WorkOSClientID) == "" {
			missing = append(missing, "ULTRA_CONTROL_WORKOS_CLIENT_ID")
		}
		if strings.TrimSpace(c.WorkOSAPIKey) == "" {
			missing = append(missing, "ULTRA_CONTROL_WORKOS_API_KEY")
		}
		if strings.TrimSpace(c.WorkOSRedirectURI) == "" {
			missing = append(missing, "ULTRA_CONTROL_WORKOS_REDIRECT_URI")
		}
		if strings.TrimSpace(c.WorkOSCookiePassword) == "" {
			missing = append(missing, "ULTRA_CONTROL_WORKOS_COOKIE_PASSWORD")
		}
	}
	if len(missing) > 0 {
		return errors.New("production control plane requires durable backends: set " + strings.Join(missing, " and "))
	}
	return nil
}

func envString(key string, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envDurationSeconds(key string, fallback int) time.Duration {
	raw := os.Getenv(key)
	if raw == "" {
		return time.Duration(fallback) * time.Second
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 0 {
		return time.Duration(fallback) * time.Second
	}
	return time.Duration(value) * time.Second
}

func envBool(key string, fallback bool) bool {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	value, err := strconv.ParseBool(raw)
	if err != nil {
		return fallback
	}
	return value
}

func envInt(key string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}
