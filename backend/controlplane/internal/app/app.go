package app

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/domain"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/httpapi"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/worker"
	"github.com/jackc/pgx/v5/pgxpool"
)

type App struct {
	Handler            http.Handler
	Store              runcontrol.Store
	Bus                eventbus.Bus
	RunEvents          eventbus.RunEventSource
	Runs               *runcontrol.Service
	JobSource          <-chan eventbus.Job
	Worker             *worker.StubWorker
	DataAgentJobSource <-chan eventbus.DataAgentJob
	DataAgentWorker    *worker.DataAgentWorker
	Start              func(context.Context) error
	closeFns           []func()
}

type dataAgentLeaseRecoveryStore interface {
	RecoverExpiredDataAgentJobLeases(context.Context, domain.RecoverExpiredDataAgentJobLeasesInput) (domain.RecoverExpiredDataAgentJobLeasesResult, error)
	AppendDataAgentJobEvent(context.Context, domain.AppendDataAgentJobEventInput) (domain.DataAgentJobEventRecord, error)
}

func New(cfg config.Config) (*App, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	ctx := context.Background()
	var controlStore runcontrol.Store
	var closeFns []func()
	storeBackend := "memory"
	if cfg.DatabaseURL != "" {
		poolConfig, err := postgresPoolConfig(cfg)
		if err != nil {
			return nil, err
		}
		// Serving pool only: bound every query server-side so one stuck/runaway query
		// (missing index, lock wait) can't hold a connection from the small pool
		// indefinitely and starve the service. Migrations go through postgresPoolConfig
		// WITHOUT this, so slow DDL (index builds) isn't aborted.
		applyStatementTimeout(poolConfig, cfg.DatabaseStatementTimeout)
		pool, err := pgxpool.NewWithConfig(ctx, poolConfig)
		if err != nil {
			return nil, err
		}
		if err := pingPostgres(ctx, pool); err != nil {
			pool.Close()
			return nil, err
		}
		if err := store.VerifyPostgresSchema(ctx, pool); err != nil {
			pool.Close()
			return nil, err
		}
		closeFns = append(closeFns, pool.Close)
		controlStore = store.NewPostgresStore(pool)
		storeBackend = "postgres"
	} else {
		controlStore = store.NewMemoryStore()
	}

	var bus eventbus.Bus
	var runEvents eventbus.RunEventSource
	var jobSource <-chan eventbus.Job
	var dataAgentJobSource <-chan eventbus.DataAgentJob
	var dataAgentJobs eventbus.DataAgentJobPublisher
	var natsBus *eventbus.NATSBus
	runtime := httpapi.RuntimeSummary{
		AppVersion:               cfg.AppVersion,
		StoreBackend:             storeBackend,
		DispatchMode:             "local_memory",
		JobTransport:             "local_memory",
		EventTransport:           "local_memory",
		StubWorkerEnabled:        true,
		NATSConfigured:           false,
		NATSStream:               cfg.NATSStream,
		NATSJobsSubject:          cfg.NATSJobsSubject,
		NATSDataAgentJobsSubject: cfg.NATSDataAgentJobsSubject,
		NATSEventsSubject:        cfg.NATSEventsSubject,
		NATSCancelSubject:        cfg.NATSCancelSubject,
		NATSEventConsumer:        cfg.NATSEventConsumer,
		ArtifactRoot:             cfg.ArtifactRoot,
		UploadRoot:               cfg.UploadRoot,
		RunRecoveryEnabled:       cfg.RunRecoveryEnabled,
		RunRecoveryIntervalSecs:  cfg.RunRecoveryInterval.Seconds(),
		RunRecoveryBatchLimit:    cfg.RunRecoveryBatchLimit,
	}
	if cfg.NATSURL != "" {
		var err error
		natsBus, err = eventbus.NewNATSBus(ctx, eventbus.NATSConfig{
			URL:                    cfg.NATSURL,
			Stream:                 cfg.NATSStream,
			JobsSubject:            cfg.NATSJobsSubject,
			DataAgentJobsSubject:   cfg.NATSDataAgentJobsSubject,
			EventsSubject:          cfg.NATSEventsSubject,
			CancelSubject:          cfg.NATSCancelSubject,
			EventConsumer:          cfg.NATSEventConsumer,
			EventIngestConcurrency: cfg.NATSEventIngestConcurrency,
			ConsumerTargets: []eventbus.QueueConsumerTarget{
				{Name: cfg.NATSWorkerDurable, Role: "deepagents", Subject: cfg.NATSJobsSubject},
				{Name: cfg.NATSDataAgentWorkerDurable, Role: "data_agent", Subject: cfg.NATSDataAgentJobsSubject},
				{Name: cfg.NATSEventConsumer, Role: "event_ingest", Subject: cfg.NATSEventsSubject},
			},
		})
		if err != nil {
			for _, closeFn := range closeFns {
				closeFn()
			}
			return nil, err
		}
		closeFns = append(closeFns, natsBus.Close)
		localEvents := eventbus.NewMemoryBus()
		bus = eventbus.NewSplitBus(natsBus, localEvents)
		runEvents = localEvents
		dataAgentJobs = natsBus
		runtime.DispatchMode = "nats_jetstream"
		runtime.JobTransport = "nats_jetstream"
		runtime.EventTransport = "nats_jetstream_to_local_fanout"
		runtime.StubWorkerEnabled = false
		runtime.NATSConfigured = true
	} else {
		memBus := eventbus.NewMemoryBus()
		bus = memBus
		runEvents = memBus
		jobSource = memBus.Jobs()
		dataAgentJobSource = memBus.DataAgentJobs()
		dataAgentJobs = memBus
	}

	runService := runcontrol.NewService(controlStore, bus)
	var stubWorker *worker.StubWorker
	if jobSource != nil {
		stubWorker = worker.NewStubWorker(controlStore, bus)
	}
	var dataAgentWorker *worker.DataAgentWorker
	if dataAgentJobSource != nil {
		if dataAgentStore, ok := controlStore.(worker.DataAgentWorkerStore); ok {
			dataAgentWorker = worker.NewDataAgentWorker(dataAgentStore, worker.DataAgentWorkerConfig{
				WorkerID: cfg.NATSDataAgentWorkerDurable,
			})
		}
	}
	var startFns []func(context.Context) error
	if natsBus != nil {
		startFns = append(startFns, func(ctx context.Context) error {
			return natsBus.SubscribeAllRunEvents(ctx, func(ctx context.Context, input domain.AppendRunEventInput) error {
				_, err := runService.IngestRunEvent(ctx, input)
				return err
			})
		})
	}
	if cfg.RunRecoveryEnabled {
		startFns = append(startFns, func(ctx context.Context) error {
			startRunRecoveryLoop(ctx, runService, cfg.RunRecoveryInterval, cfg.RunRecoveryBatchLimit)
			if recoveryStore, ok := controlStore.(dataAgentLeaseRecoveryStore); ok && dataAgentJobs != nil {
				startDataAgentRecoveryLoop(ctx, recoveryStore, dataAgentJobs, cfg.RunRecoveryInterval, cfg.RunRecoveryBatchLimit)
			}
			return nil
		})
	}
	// Retention GC permanently reclaims artifacts of resources past their undelete window.
	// Off by default (it deletes data); an operator enables it after watching the
	// retention_backlog the admin overview reports.
	if cfg.RetentionGCEnabled {
		if gcStore, ok := controlStore.(httpapi.RetentionGCStore); ok {
			startFns = append(startFns, func(ctx context.Context) error {
				go httpapi.RunRetentionGC(ctx, gcStore, cfg.UploadRoot, cfg.RetentionGCInterval, cfg.RetentionGCBatch)
				return nil
			})
		}
	}
	bisqueService := httpapi.NewBisqueService(httpapi.BisqueServiceConfig{
		RootURL:       cfg.BisqueRootURL,
		DevUsername:   cfg.BisqueUsername,
		DevPassword:   cfg.BisquePassword,
		AllowedRoots:  []string{cfg.BisqueRootURL},
		UploadRoot:    cfg.UploadRoot,
		MaxImportSize: cfg.BisqueMaxImportBytes,
	})
	bisqueCredentialStore := httpapi.NewBisqueCredentialStore()
	if strings.TrimSpace(cfg.SecretEncryptionKey) != "" {
		cipher, err := httpapi.NewBisqueCredentialCipherFromString(cfg.SecretEncryptionKey, cfg.SecretEncryptionKeyID)
		if err != nil {
			for _, closeFn := range closeFns {
				closeFn()
			}
			return nil, err
		}
		persistentCredentials, ok := controlStore.(httpapi.BisquePersistentCredentialStore)
		if !ok {
			for _, closeFn := range closeFns {
				closeFn()
			}
			return nil, fmt.Errorf("control store does not support persistent BisQue credentials")
		}
		bisqueCredentialStore = httpapi.NewPersistentBisqueCredentialStore(persistentCredentials, cipher, cfg.BisqueRootURL)
	}
	var workOSAuth *httpapi.WorkOSAuth
	if strings.EqualFold(strings.TrimSpace(cfg.AuthProvider), "workos") {
		auth, err := httpapi.NewWorkOSAuth(httpapi.WorkOSAuthConfig{
			Enabled:              true,
			ClientID:             cfg.WorkOSClientID,
			APIKey:               cfg.WorkOSAPIKey,
			RedirectURI:          cfg.WorkOSRedirectURI,
			PostLoginRedirectURI: cfg.WorkOSPostLoginRedirectURI,
			LogoutRedirectURI:    cfg.WorkOSLogoutRedirectURI,
			CookiePassword:       cfg.WorkOSCookiePassword,
			CookieSecure:         cfg.WorkOSCookieSecure,
			BaseURL:              cfg.WorkOSBaseURL,
		})
		if err != nil {
			for _, closeFn := range closeFns {
				closeFn()
			}
			return nil, err
		}
		workOSAuth = auth
	}
	var start func(context.Context) error
	if len(startFns) > 0 {
		start = func(ctx context.Context) error {
			for _, startFn := range startFns {
				if err := startFn(ctx); err != nil {
					return err
				}
			}
			return nil
		}
	}
	handler := httpapi.NewRouter(httpapi.ServerDeps{
		Version:           cfg.AppVersion,
		Runs:              runService,
		Store:             controlStore,
		Bus:               runEvents,
		ArtifactRoot:      cfg.ArtifactRoot,
		UploadRoot:        cfg.UploadRoot,
		ImageServiceURL:   cfg.ImageServiceURL,
		DevAdminEnabled:   cfg.DevAdminEnabled,
		Runtime:           runtime,
		QueueDiagnostics:  natsBus,
		DataAgentJobs:     dataAgentJobs,
		Bisque:            bisqueService,
		BisqueCredentials: bisqueCredentialStore,
		WorkOS:            workOSAuth,
		WorkerToken:       cfg.WorkerToken,
	})
	return &App{
		Handler:            handler,
		Store:              controlStore,
		Bus:                bus,
		RunEvents:          runEvents,
		Runs:               runService,
		JobSource:          jobSource,
		Worker:             stubWorker,
		DataAgentJobSource: dataAgentJobSource,
		DataAgentWorker:    dataAgentWorker,
		Start:              start,
		closeFns:           closeFns,
	}, nil
}

func startRunRecoveryLoop(ctx context.Context, runs *runcontrol.Service, interval time.Duration, limit int) {
	if interval <= 0 {
		interval = 30 * time.Second
	}
	if limit <= 0 {
		limit = 1000
	}
	go func() {
		recoverExpiredRunLeases(ctx, runs, limit)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				recoverExpiredRunLeases(ctx, runs, limit)
			}
		}
	}()
}

func recoverExpiredRunLeases(ctx context.Context, runs *runcontrol.Service, limit int) {
	result, err := runs.RecoverExpiredRunLeases(ctx, runcontrol.RecoverExpiredRunLeasesRequest{
		Reason: "automatic expired run lease recovery",
		Limit:  limit,
	})
	if err != nil {
		slog.Error("run lease recovery pass failed", "error", err, "checked", result.Checked)
		return
	}
	if len(result.RequeuedRuns) > 0 {
		runIDs := make([]string, 0, len(result.RequeuedRuns))
		for _, run := range result.RequeuedRuns {
			runIDs = append(runIDs, run.RunID)
		}
		slog.Info("run lease recovery requeued runs", "checked", result.Checked, "requeued", runIDs)
	}
}

func startDataAgentRecoveryLoop(ctx context.Context, store dataAgentLeaseRecoveryStore, publisher eventbus.DataAgentJobPublisher, interval time.Duration, limit int) {
	if interval <= 0 {
		interval = 30 * time.Second
	}
	if limit <= 0 {
		limit = 1000
	}
	go func() {
		recoverExpiredDataAgentJobLeases(ctx, store, publisher, limit)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				recoverExpiredDataAgentJobLeases(ctx, store, publisher, limit)
			}
		}
	}()
}

func recoverExpiredDataAgentJobLeases(ctx context.Context, store dataAgentLeaseRecoveryStore, publisher eventbus.DataAgentJobPublisher, limit int) {
	result, err := store.RecoverExpiredDataAgentJobLeases(ctx, domain.RecoverExpiredDataAgentJobLeasesInput{
		Reason: "automatic expired data-agent lease recovery",
		Limit:  limit,
	})
	if err != nil {
		slog.Error("data-agent lease recovery pass failed", "error", err)
		return
	}
	for _, job := range result.RequeuedJobs {
		dispatchID := domain.NewID("dispatch")
		envelope := eventbus.DataAgentJob{
			JobID:         job.JobID,
			DispatchID:    dispatchID,
			OwnerUserID:   job.OwnerUserID,
			OwnerOrgID:    job.OwnerOrgID,
			ProjectID:     job.ProjectID,
			JobType:       job.JobType,
			ResourceIDs:   dataAgentRecoveryResourceIDs(job.InputSelector),
			ResourceCount: job.ResourceCount,
			InputSelector: cloneAppJSONMap(job.InputSelector),
			Metadata:      cloneAppJSONMap(job.Metadata),
		}
		if err := publisher.PublishDataAgentJob(ctx, envelope); err != nil {
			continue
		}
		_, _ = store.AppendDataAgentJobEvent(ctx, domain.AppendDataAgentJobEventInput{
			JobID:       job.JobID,
			EventType:   "data_agent.job.dispatched",
			ActorUserID: job.OwnerUserID,
			ActorOrgID:  job.OwnerOrgID,
			TS:          domain.Now(),
			Message:     "Data Agent job dispatched after expired lease recovery.",
			Metadata: domain.JSONMap{
				"dispatch_id": dispatchID,
				"job_type":    job.JobType,
				"recovery":    "expired_data_agent_job_lease",
			},
		})
	}
}

func dataAgentRecoveryResourceIDs(selector domain.JSONMap) []string {
	if selector == nil {
		return nil
	}
	value, ok := selector["resource_ids"]
	if !ok {
		return nil
	}
	switch typed := value.(type) {
	case []string:
		return uniqueAppStrings(typed)
	case []any:
		values := make([]string, 0, len(typed))
		for _, item := range typed {
			if text, ok := item.(string); ok {
				values = append(values, text)
			}
		}
		return uniqueAppStrings(values)
	default:
		return nil
	}
}

func uniqueAppStrings(values []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func cloneAppJSONMap(value domain.JSONMap) domain.JSONMap {
	if value == nil {
		return nil
	}
	out := domain.JSONMap{}
	for key, item := range value {
		out[key] = item
	}
	return out
}

func pingPostgres(ctx context.Context, pool *pgxpool.Pool) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := pool.Ping(ctx); err != nil {
		return fmt.Errorf("postgres backend is not reachable: %w", err)
	}
	return nil
}

func postgresPoolConfig(cfg config.Config) (*pgxpool.Config, error) {
	poolConfig, err := pgxpool.ParseConfig(cfg.DatabaseURL)
	if err != nil {
		return nil, err
	}
	if cfg.DatabaseMaxConns > 0 {
		poolConfig.MaxConns = int32(cfg.DatabaseMaxConns)
	}
	if cfg.DatabaseMinConns > 0 {
		poolConfig.MinConns = int32(cfg.DatabaseMinConns)
	}
	if poolConfig.MinConns > poolConfig.MaxConns {
		return nil, fmt.Errorf("ULTRA_CONTROL_DATABASE_MIN_CONNS (%d) cannot exceed ULTRA_CONTROL_DATABASE_MAX_CONNS (%d)", poolConfig.MinConns, poolConfig.MaxConns)
	}
	return poolConfig, nil
}

// applyStatementTimeout sets a server-side per-query timeout on every connection the
// pool opens, via the standard Postgres statement_timeout runtime parameter. This
// bounds a single stuck/runaway query so it cannot hold a pooled connection forever
// and starve the small serving pool (MaxConns default 8) — the failure mode where a
// handful of wedged queries take the whole service offline. A no-op when timeout<=0
// (disabled, the default), so it never changes behavior unless an operator opts in.
// Applied to the serving pool only; never to migrations, whose DDL is legitimately slow.
func applyStatementTimeout(poolConfig *pgxpool.Config, timeout time.Duration) {
	if timeout <= 0 {
		return
	}
	if poolConfig.ConnConfig.RuntimeParams == nil {
		poolConfig.ConnConfig.RuntimeParams = map[string]string{}
	}
	poolConfig.ConnConfig.RuntimeParams["statement_timeout"] = strconv.FormatInt(timeout.Milliseconds(), 10)
}

func MigratePostgres(ctx context.Context, cfg config.Config) error {
	if strings.TrimSpace(cfg.DatabaseURL) == "" {
		return fmt.Errorf("ULTRA_CONTROL_DATABASE_URL or RUN_STORE_PATH is required to migrate the control-plane Postgres schema")
	}
	poolConfig, err := postgresPoolConfig(cfg)
	if err != nil {
		return err
	}
	pool, err := pgxpool.NewWithConfig(ctx, poolConfig)
	if err != nil {
		return err
	}
	defer pool.Close()
	if err := pingPostgres(ctx, pool); err != nil {
		return err
	}
	if err := store.ApplyPostgresSchema(ctx, pool); err != nil {
		return err
	}
	return store.VerifyPostgresSchema(ctx, pool)
}

func NewHTTPHandler(cfg config.Config) http.Handler {
	application, err := New(cfg)
	if err != nil {
		panic(err)
	}
	return application.Handler
}

func (a *App) Close() {
	for index := len(a.closeFns) - 1; index >= 0; index-- {
		a.closeFns[index]()
	}
}
