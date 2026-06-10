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

	switch controlCommand(os.Args) {
	case "migrate":
		if err := app.MigratePostgres(ctx, cfg); err != nil {
			logger.Error("control-plane migration failed", "error", err)
			os.Exit(1)
		}
		logger.Info("control-plane migration completed")
		return
	case "serve":
	default:
		logger.Error("unknown command", "command", os.Args[1])
		os.Exit(2)
	}

	application, err := app.New(cfg)
	if err != nil {
		logger.Error("application setup failed", "error", err)
		os.Exit(1)
	}
	defer application.Close()

	if application.Start != nil {
		if err := application.Start(ctx); err != nil {
			logger.Error("application background setup failed", "error", err)
			os.Exit(1)
		}
	}

	server := &http.Server{
		Addr:         cfg.HTTPAddr,
		Handler:      application.Handler,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		IdleTimeout:  cfg.IdleTimeout,
	}

	errs := make(chan error, 1)
	if application.JobSource != nil && application.Worker != nil {
		go func() {
			for {
				select {
				case <-ctx.Done():
					return
				case job := <-application.JobSource:
					if err := application.Worker.RunJob(ctx, job); err != nil && !errors.Is(err, context.Canceled) {
						logger.Error("worker job failed", "run_id", job.RunID, "error", err)
					}
				}
			}
		}()
	}
	if application.DataAgentJobSource != nil && application.DataAgentWorker != nil {
		go func() {
			for {
				select {
				case <-ctx.Done():
					return
				case job := <-application.DataAgentJobSource:
					if err := application.DataAgentWorker.RunJob(ctx, job); err != nil && !errors.Is(err, context.Canceled) {
						logger.Error("data agent worker job failed", "job_id", job.JobID, "error", err)
					}
				}
			}
		}()
	}
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

func controlCommand(args []string) string {
	if len(args) < 2 {
		return "serve"
	}
	return args[1]
}
