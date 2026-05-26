package app

import (
	"net/http"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/httpapi"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/worker"
)

type App struct {
	Handler http.Handler
	Store   *store.MemoryStore
	Bus     *eventbus.MemoryBus
	Worker  *worker.StubWorker
}

func New(cfg config.Config) (*App, error) {
	memStore := store.NewMemoryStore()
	memBus := eventbus.NewMemoryBus()
	runService := runcontrol.NewService(memStore, memBus)
	stubWorker := worker.NewStubWorker(memStore, memBus)
	handler := httpapi.NewRouter(httpapi.ServerDeps{
		Version: cfg.AppVersion,
		Runs:    runService,
		Store:   memStore,
		Bus:     memBus,
	})
	return &App{
		Handler: handler,
		Store:   memStore,
		Bus:     memBus,
		Worker:  stubWorker,
	}, nil
}

func NewHTTPHandler(cfg config.Config) http.Handler {
	application, err := New(cfg)
	if err != nil {
		panic(err)
	}
	return application.Handler
}
