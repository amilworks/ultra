package app

import (
	"context"
	"fmt"
	"net/http"
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
	Handler   http.Handler
	Store     runcontrol.Store
	Bus       eventbus.Bus
	RunEvents eventbus.RunEventSource
	Runs      *runcontrol.Service
	JobSource <-chan eventbus.Job
	Worker    *worker.StubWorker
	Start     func(context.Context) error
	closeFns  []func()
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
		pool, err := pgxpool.New(ctx, cfg.DatabaseURL)
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
	var natsBus *eventbus.NATSBus
	runtime := httpapi.RuntimeSummary{
		AppVersion:              cfg.AppVersion,
		StoreBackend:            storeBackend,
		DispatchMode:            "local_memory",
		JobTransport:            "local_memory",
		EventTransport:          "local_memory",
		StubWorkerEnabled:       true,
		NATSConfigured:          false,
		NATSStream:              cfg.NATSStream,
		NATSJobsSubject:         cfg.NATSJobsSubject,
		NATSRareSpotJobsSubject: cfg.NATSRareSpotJobsSubject,
		NATSEventsSubject:       cfg.NATSEventsSubject,
		NATSCancelSubject:       cfg.NATSCancelSubject,
		NATSEventConsumer:       cfg.NATSEventConsumer,
		ArtifactRoot:            cfg.ArtifactRoot,
		UploadRoot:              cfg.UploadRoot,
		RunRecoveryEnabled:      cfg.RunRecoveryEnabled,
		RunRecoveryIntervalSecs: cfg.RunRecoveryInterval.Seconds(),
		RunRecoveryBatchLimit:   cfg.RunRecoveryBatchLimit,
	}
	if cfg.NATSURL != "" {
		var err error
		natsBus, err = eventbus.NewNATSBus(ctx, eventbus.NATSConfig{
			URL:                 cfg.NATSURL,
			Stream:              cfg.NATSStream,
			JobsSubject:         cfg.NATSJobsSubject,
			RareSpotJobsSubject: cfg.NATSRareSpotJobsSubject,
			EventsSubject:       cfg.NATSEventsSubject,
			CancelSubject:       cfg.NATSCancelSubject,
			EventConsumer:       cfg.NATSEventConsumer,
			ConsumerTargets: []eventbus.QueueConsumerTarget{
				{Name: cfg.NATSWorkerDurable, Role: "deepagents", Subject: cfg.NATSJobsSubject},
				{Name: cfg.NATSRareSpotWorkerDurable, Role: "rarespot", Subject: cfg.NATSRareSpotJobsSubject},
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
	}

	runService := runcontrol.NewService(controlStore, bus)
	var stubWorker *worker.StubWorker
	if jobSource != nil {
		stubWorker = worker.NewStubWorker(controlStore, bus)
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
			return nil
		})
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
		DevAdminEnabled:   cfg.DevAdminEnabled,
		Runtime:           runtime,
		QueueDiagnostics:  natsBus,
		Bisque:            bisqueService,
		BisqueCredentials: bisqueCredentialStore,
		WorkOS:            workOSAuth,
	})
	return &App{
		Handler:   handler,
		Store:     controlStore,
		Bus:       bus,
		RunEvents: runEvents,
		Runs:      runService,
		JobSource: jobSource,
		Worker:    stubWorker,
		Start:     start,
		closeFns:  closeFns,
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
	_, _ = runs.RecoverExpiredRunLeases(ctx, runcontrol.RecoverExpiredRunLeasesRequest{
		Reason: "automatic expired run lease recovery",
		Limit:  limit,
	})
}

func pingPostgres(ctx context.Context, pool *pgxpool.Pool) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := pool.Ping(ctx); err != nil {
		return fmt.Errorf("postgres backend is not reachable: %w", err)
	}
	return nil
}

func MigratePostgres(ctx context.Context, cfg config.Config) error {
	if strings.TrimSpace(cfg.DatabaseURL) == "" {
		return fmt.Errorf("ULTRA_CONTROL_DATABASE_URL or RUN_STORE_PATH is required to migrate the control-plane Postgres schema")
	}
	pool, err := pgxpool.New(ctx, cfg.DatabaseURL)
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
