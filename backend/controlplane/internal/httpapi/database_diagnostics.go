package httpapi

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	defaultDatabaseTopQueryLimit = 8
	maxDatabaseQueryPreviewBytes = 480
)

// DatabaseDiagnosticsProvider is the admin-overview hook for read-only database
// observability. Implementations should be cheap and best-effort.
type DatabaseDiagnosticsProvider interface {
	DatabaseDiagnostics(context.Context, int) (adminDatabaseDiagnostics, error)
}

type adminDatabaseDiagnostics struct {
	Available  bool                      `json:"available"`
	Pool       adminDatabasePoolStats    `json:"pool"`
	TopQueries []adminDatabaseQueryStats `json:"top_queries"`
	Error      string                    `json:"error,omitempty"`
}

type adminDatabasePoolStats struct {
	MaxConns                int32   `json:"max_conns"`
	TotalConns              int32   `json:"total_conns"`
	AcquiredConns           int32   `json:"acquired_conns"`
	IdleConns               int32   `json:"idle_conns"`
	ConstructingConns       int32   `json:"constructing_conns"`
	AcquireCount            int64   `json:"acquire_count"`
	EmptyAcquireCount       int64   `json:"empty_acquire_count"`
	CanceledAcquireCount    int64   `json:"canceled_acquire_count"`
	NewConnsCount           int64   `json:"new_conns_count"`
	MaxLifetimeDestroyCount int64   `json:"max_lifetime_destroy_count"`
	MaxIdleDestroyCount     int64   `json:"max_idle_destroy_count"`
	AcquireDurationSeconds  float64 `json:"acquire_duration_seconds"`
	EmptyAcquireWaitSeconds float64 `json:"empty_acquire_wait_seconds"`
	Saturation              float64 `json:"saturation"`
	WaitRatio               float64 `json:"wait_ratio"`
}

type adminDatabaseQueryStats struct {
	QueryID           string  `json:"query_id"`
	Calls             int64   `json:"calls"`
	MeanExecMs        float64 `json:"mean_exec_ms"`
	TotalExecMs       float64 `json:"total_exec_ms"`
	Rows              int64   `json:"rows"`
	SharedBlocksHit   int64   `json:"shared_blocks_hit"`
	SharedBlocksRead  int64   `json:"shared_blocks_read"`
	TempBlocksWritten int64   `json:"temp_blocks_written"`
	Query             string  `json:"query"`
}

// PostgresDatabaseDiagnostics samples pgxpool counters and pg_stat_statements.
type PostgresDatabaseDiagnostics struct {
	pool *pgxpool.Pool
}

func NewPostgresDatabaseDiagnostics(pool *pgxpool.Pool) *PostgresDatabaseDiagnostics {
	if pool == nil {
		return nil
	}
	return &PostgresDatabaseDiagnostics{pool: pool}
}

func (d *PostgresDatabaseDiagnostics) DatabaseDiagnostics(ctx context.Context, limit int) (adminDatabaseDiagnostics, error) {
	if d == nil || d.pool == nil {
		return adminDatabaseDiagnostics{Available: false}, nil
	}
	if limit <= 0 || limit > 25 {
		limit = defaultDatabaseTopQueryLimit
	}
	diagnostics := adminDatabaseDiagnostics{
		Available: true,
		Pool:      adminDatabasePoolStatsFromPGX(d.pool.Stat()),
	}
	queries, err := d.topQueries(ctx, limit)
	if err != nil {
		diagnostics.Error = err.Error()
		return diagnostics, nil
	}
	diagnostics.TopQueries = queries
	return diagnostics, nil
}

func (d *PostgresDatabaseDiagnostics) topQueries(ctx context.Context, limit int) ([]adminDatabaseQueryStats, error) {
	rows, err := d.pool.Query(ctx, `
SELECT
	COALESCE(queryid::text, '') AS query_id,
	calls::bigint,
	total_exec_time::double precision,
	mean_exec_time::double precision,
	rows::bigint,
	shared_blks_hit::bigint,
	shared_blks_read::bigint,
	temp_blks_written::bigint,
	query
FROM pg_stat_statements
WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
ORDER BY mean_exec_time DESC
LIMIT $1
`, limit)
	if err != nil {
		return nil, fmt.Errorf("query pg_stat_statements: %w", err)
	}
	defer rows.Close()

	queries := make([]adminDatabaseQueryStats, 0, limit)
	for rows.Next() {
		var q adminDatabaseQueryStats
		if err := rows.Scan(
			&q.QueryID,
			&q.Calls,
			&q.TotalExecMs,
			&q.MeanExecMs,
			&q.Rows,
			&q.SharedBlocksHit,
			&q.SharedBlocksRead,
			&q.TempBlocksWritten,
			&q.Query,
		); err != nil {
			return nil, fmt.Errorf("scan pg_stat_statements: %w", err)
		}
		q.Query = compactDatabaseQueryPreview(q.Query)
		queries = append(queries, q)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate pg_stat_statements: %w", err)
	}
	return queries, nil
}

func adminDatabasePoolStatsFromPGX(stat *pgxpool.Stat) adminDatabasePoolStats {
	if stat == nil {
		return adminDatabasePoolStats{}
	}
	maxConns := stat.MaxConns()
	acquireCount := stat.AcquireCount()
	emptyAcquireCount := stat.EmptyAcquireCount()
	stats := adminDatabasePoolStats{
		MaxConns:                maxConns,
		TotalConns:              stat.TotalConns(),
		AcquiredConns:           stat.AcquiredConns(),
		IdleConns:               stat.IdleConns(),
		ConstructingConns:       stat.ConstructingConns(),
		AcquireCount:            acquireCount,
		EmptyAcquireCount:       emptyAcquireCount,
		CanceledAcquireCount:    stat.CanceledAcquireCount(),
		NewConnsCount:           stat.NewConnsCount(),
		MaxLifetimeDestroyCount: stat.MaxLifetimeDestroyCount(),
		MaxIdleDestroyCount:     stat.MaxIdleDestroyCount(),
		AcquireDurationSeconds:  seconds(stat.AcquireDuration()),
		EmptyAcquireWaitSeconds: seconds(stat.EmptyAcquireWaitTime()),
	}
	if maxConns > 0 {
		stats.Saturation = float64(stats.AcquiredConns) / float64(maxConns)
	}
	if acquireCount > 0 {
		stats.WaitRatio = float64(emptyAcquireCount) / float64(acquireCount)
	}
	return stats
}

func (deps ServerDeps) adminDatabaseDiagnostics(ctx context.Context) adminDatabaseDiagnostics {
	if deps.DatabaseDiagnostics == nil {
		return adminDatabaseDiagnostics{Available: false}
	}
	diagnostics, err := deps.DatabaseDiagnostics.DatabaseDiagnostics(ctx, defaultDatabaseTopQueryLimit)
	if err != nil && diagnostics.Error == "" {
		diagnostics.Error = err.Error()
	}
	return diagnostics
}

func compactDatabaseQueryPreview(query string) string {
	query = strings.Join(strings.Fields(query), " ")
	if len(query) <= maxDatabaseQueryPreviewBytes {
		return query
	}
	return query[:maxDatabaseQueryPreviewBytes]
}

func seconds(d time.Duration) float64 {
	return d.Seconds()
}
