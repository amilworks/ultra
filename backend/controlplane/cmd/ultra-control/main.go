package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/app"
	"github.com/amilworks/bisque-ultra/backend/controlplane/internal/config"
)

func main() {
	cfg := config.Load()
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	application, err := app.New(cfg)
	if err != nil {
		logger.Error("application setup failed", "error", err)
		os.Exit(1)
	}

	server := &http.Server{
		Addr:         cfg.HTTPAddr,
		Handler:      application.Handler,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		IdleTimeout:  cfg.IdleTimeout,
	}

	errs := make(chan error, 1)
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case job := <-application.Bus.Jobs():
				if err := application.Worker.RunJob(ctx, job); err != nil && !errors.Is(err, context.Canceled) {
					logger.Error("worker job failed", "run_id", job.RunID, "error", err)
				}
			}
		}
	}()
	go func() {
		logger.Info("starting control plane", "addr", cfg.HTTPAddr)
		errs <- server.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		logger.Info("shutting down", "signal", ctx.Err())
	case err := <-errs:
		if !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server failed", "error", err)
			os.Exit(1)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		logger.Error("shutdown failed", "error", err)
		os.Exit(1)
	}
}
