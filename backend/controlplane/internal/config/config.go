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
	PublicURL                  string
	DefaultUserTimezone        string
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
	NATSEventPartitions        int
	NATSStreamMaxAge           time.Duration
	NATSStreamMaxBytes         int64
	NATSEventIngestConcurrency int
	NATSWorkerDurable          string
	NATSDataAgentWorkerDurable string
	NATSTrainingJobsSubject    string
	NATSTrainingWorkerDurable  string
	ArtifactRoot               string
	UploadRoot                 string
	ImageServiceURL            string
	NgffServiceURL             string
	BisqueRootURL              string
	BisqueUsername             string
	BisquePassword             string
	BisqueMaxImportBytes       int64
	GoogleClientID             string
	GoogleClientSecret         string
	GoogleRedirectURL          string
	GooglePickerAPIKey         string
	GoogleMaxImportBytes       int64
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
	// WorkOSAdminEmails grants the admin role to these (case-insensitive) emails
	// after WorkOS authentication, regardless of the WorkOS-reported role. This
	// is the bootstrap path for the first admin (you can't self-promote in-app
	// until you're already an admin). From ULTRA_CONTROL_ADMIN_EMAILS.
	WorkOSAdminEmails         []string
	DevAdminEnabled           bool
	AllowDevAuthInProduction  bool
	RunRecoveryEnabled        bool
	RunRecoveryInterval       time.Duration
	RunRecoveryFirstPassDelay time.Duration
	RunRecoveryBatchLimit     int
	RetentionGCEnabled        bool
	RetentionGCInterval       time.Duration
	RetentionGCBatch          int
	RunEventDeltaRetention    time.Duration
	WorkerToken               string
}

func Load() Config {
	return Config{
		AppName:     envString("ULTRA_CONTROL_APP_NAME", "BisQue Ultra Control Plane"),
		AppVersion:  envString("ULTRA_CONTROL_APP_VERSION", "dev"),
		Environment: strings.ToLower(envString("ULTRA_CONTROL_ENVIRONMENT", envString("ENVIRONMENT", "development"))),
		PublicURL:   envString("ULTRA_CONTROL_PUBLIC_URL", ""),
		DefaultUserTimezone: envString(
			"ULTRA_CONTROL_DEFAULT_USER_TIMEZONE",
			envString("TZ", "UTC"),
		),
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
		DatabaseStatementTimeout: envDurationSeconds("ULTRA_CONTROL_DATABASE_STATEMENT_TIMEOUT_SECONDS", 0),
		NATSURL:                  envString("ULTRA_CONTROL_NATS_URL", ""),
		NATSStream:               envString("ULTRA_CONTROL_NATS_STREAM", "ULTRA_RUNS"),
		NATSJobsSubject:          envString("ULTRA_CONTROL_NATS_JOBS_SUBJECT", "ultra.runs.jobs"),
		NATSDataAgentJobsSubject: envString("ULTRA_CONTROL_NATS_DATA_AGENT_JOBS_SUBJECT", "ultra.data_agent.jobs"),
		NATSEventsSubject:        envString("ULTRA_CONTROL_NATS_EVENTS_SUBJECT", "ultra.runs.events"),
		NATSCancelSubject:        envString("ULTRA_CONTROL_NATS_CANCEL_SUBJECT", "ultra.runs.cancel"),
		NATSEventConsumer:        envString("ULTRA_CONTROL_NATS_EVENT_CONSUMER", "ultra-control-event-ingest"),
		// Must match the worker's partition count, or events published to a
		// partition the control plane has no consumer for are stored but never
		// ingested (runs look hung). The worker (config.py) reads
		// ULTRA_NATS_EVENT_PARTITIONS first, then ULTRA_CONTROL_NATS_EVENT_PARTITIONS;
		// mirror that precedence exactly so a single shared var keeps both sides
		// in agreement.
		NATSEventPartitions:        envInt("ULTRA_NATS_EVENT_PARTITIONS", envInt("ULTRA_CONTROL_NATS_EVENT_PARTITIONS", 64)),
		NATSEventIngestConcurrency: envInt("ULTRA_CONTROL_NATS_EVENT_INGEST_CONCURRENCY", 4),
		// Retention bounds for the JetStream stream (events are archived in
		// Postgres; the stream is transport). MaxAge is clamped to >= the 24h
		// duplicate window in eventbus.
		NATSStreamMaxAge:           time.Duration(envInt("ULTRA_CONTROL_NATS_STREAM_MAX_AGE_HOURS", 72)) * time.Hour,
		NATSStreamMaxBytes:         int64(envInt("ULTRA_CONTROL_NATS_STREAM_MAX_BYTES", 8<<30)),
		NATSWorkerDurable:          envString("ULTRA_CONTROL_NATS_WORKER_DURABLE", "ultra-deepagents-worker"),
		NATSDataAgentWorkerDurable: envString("ULTRA_CONTROL_NATS_DATA_AGENT_WORKER_DURABLE", "ultra-data-agent-worker"),
		NATSTrainingJobsSubject:    envString("ULTRA_CONTROL_NATS_TRAINING_JOBS_SUBJECT", "ultra.training.jobs"),
		NATSTrainingWorkerDurable:  envString("ULTRA_CONTROL_NATS_TRAINING_WORKER_DURABLE", "ultra-training-worker"),
		ArtifactRoot:               envString("ULTRA_CONTROL_ARTIFACT_ROOT", envString("ARTIFACT_ROOT", "data/artifacts")),
		UploadRoot:                 envString("ULTRA_CONTROL_UPLOAD_ROOT", envString("ULTRA_RESOURCE_ROOT", envString("UPLOAD_STORE_ROOT", "data/uploads"))),
		ImageServiceURL:            envString("ULTRA_CONTROL_IMAGE_SERVICE_URL", ""),
		NgffServiceURL:             envString("ULTRA_CONTROL_NGFF_SERVICE_URL", ""),
		BisqueRootURL:              envString("ULTRA_CONTROL_BISQUE_ROOT_URL", envString("BISQUE_ROOT", envString("BISQUE_SERVER", ""))),
		BisqueUsername:             envString("ULTRA_CONTROL_BISQUE_USERNAME", envString("BISQUE_USERNAME", envString("BISQUE_USER", ""))),
		BisquePassword:             envString("ULTRA_CONTROL_BISQUE_PASSWORD", envString("BISQUE_PASSWORD", "")),
		BisqueMaxImportBytes:       int64(envInt("ULTRA_CONTROL_BISQUE_MAX_IMPORT_BYTES", 512<<20)),
		GoogleClientID:             envString("ULTRA_CONTROL_GOOGLE_CLIENT_ID", ""),
		GoogleClientSecret:         envString("ULTRA_CONTROL_GOOGLE_CLIENT_SECRET", ""),
		GoogleRedirectURL:          envString("ULTRA_CONTROL_GOOGLE_REDIRECT_URL", ""),
		GooglePickerAPIKey:         envString("ULTRA_CONTROL_GOOGLE_PICKER_API_KEY", ""),
		GoogleMaxImportBytes:       int64(envInt("ULTRA_CONTROL_GOOGLE_MAX_IMPORT_BYTES", 10<<30)),
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
		WorkOSAdminEmails:          envStringList("ULTRA_CONTROL_ADMIN_EMAILS"),
		DevAdminEnabled:            envBool("ULTRA_CONTROL_DEV_ADMIN_ENABLED", true),
		AllowDevAuthInProduction:   envBool("ULTRA_CONTROL_ALLOW_DEV_AUTH_IN_PRODUCTION", false),
		RunRecoveryEnabled:         envBool("ULTRA_CONTROL_RUN_RECOVERY_ENABLED", true),
		RunRecoveryInterval:        envDurationSeconds("ULTRA_CONTROL_RUN_RECOVERY_INTERVAL_SECONDS", 30),
		RunRecoveryBatchLimit:      envInt("ULTRA_CONTROL_RUN_RECOVERY_BATCH_LIMIT", 1000),
		// Off by default — it permanently deletes resources past their 30-day undelete window.
		RetentionGCEnabled:  envBool("ULTRA_CONTROL_RETENTION_GC_ENABLED", false),
		RetentionGCInterval: envDurationSeconds("ULTRA_CONTROL_RETENTION_GC_INTERVAL_SECONDS", 3600),
		RetentionGCBatch:    envInt("ULTRA_CONTROL_RETENTION_GC_BATCH", 100),
		// Per-token delta events are pruned for runs that completed more than this long ago (0 = keep
		// forever / disabled). The durable answer is in control_thread_messages, so this is lossless.
		RunEventDeltaRetention: envDurationSeconds("ULTRA_CONTROL_RUN_EVENT_DELTA_TTL_SECONDS", 0),
		WorkerToken:            envString("ULTRA_CONTROL_WORKER_TOKEN", ""),
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

// envStringList parses a comma-separated env value into a trimmed, non-empty
// slice. Missing or blank yields an empty slice.
func envStringList(key string) []string {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	values := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			values = append(values, trimmed)
		}
	}
	return values
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
