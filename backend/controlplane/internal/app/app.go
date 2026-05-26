package app

import (
	"net/http"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/httpapi"
)

func NewHTTPHandler(cfg config.Config) http.Handler {
	return httpapi.NewRouter(httpapi.ServerDeps{
		Version: cfg.AppVersion,
	})
}
