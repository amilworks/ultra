package config

import (
	"os"
	"strconv"
	"time"
)

type Config struct {
	AppName      string
	AppVersion   string
	HTTPAddr     string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	IdleTimeout  time.Duration
}

func Load() Config {
	return Config{
		AppName:      envString("ULTRA_CONTROL_APP_NAME", "BisQue Ultra Control Plane"),
		AppVersion:   envString("ULTRA_CONTROL_APP_VERSION", "dev"),
		HTTPAddr:     envString("ULTRA_CONTROL_HTTP_ADDR", "127.0.0.1:8088"),
		ReadTimeout:  envDurationSeconds("ULTRA_CONTROL_READ_TIMEOUT_SECONDS", 10),
		WriteTimeout: envDurationSeconds("ULTRA_CONTROL_WRITE_TIMEOUT_SECONDS", 0),
		IdleTimeout:  envDurationSeconds("ULTRA_CONTROL_IDLE_TIMEOUT_SECONDS", 120),
	}
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
