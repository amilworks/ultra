package app

import (
	"net/http"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/eventbus"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/httpapi"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/runcontrol"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/store"
)

func NewHTTPHandler(cfg config.Config) http.Handler {
	mem := store.NewMemoryStore()
	bus := eventbus.NewMemoryBus()
	runs := runcontrol.NewService(mem, bus)
	return httpapi.NewRouter(httpapi.ServerDeps{
		Version: cfg.AppVersion,
		Runs:    runs,
		Store:   mem,
	})
}
