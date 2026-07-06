# Ultra Perf Engineer Memory

Purpose: persistent operating memory for `ultra_perf_engineer`. Read this
before every performance, benchmark, or optimization task. Numbers here are
only trustworthy with their date and source; re-measure before using any
baseline in an argument about today's code.

## Baseline Sources

- `planning/2026-07-02-production-load-benchmark.md` — live 5-node production
  stress test: chat TTFT and token throughput, static serving ceiling, and
  the network-edge bottleneck analysis. This is the canonical envelope
  document for "how much load can prod take".
- `planning/2026-07-01-prod-perf-robustness-rebuild.md` — frontend bundle
  reduction, imaging parallel-decode latency work, and edge/NFS restructuring
  from the production perf round.
- `backend/controlplane/internal/httpapi/imageservice_cache_bench_test.go` —
  the executable Go-side cache benchmark; run it before and after any change
  to imageservice caching.
- `backend/deepagents_runtime/src/ultra_deepagents/imaging/benchmark.py` —
  imaging decode/serve benchmark harness.
- `scripts/benchmark_code_execution_service.py` — sandbox/code-execution
  latency harness.

## Measurement Protocol

- Record every accepted measurement as: date, command, environment (local
  stack vs prod-like vs prod), dataset scale, p50/p95/p99, and the baseline it
  was compared against.
- A win without a named regression gate is not done: name the test, bench, or
  make target that catches the regression, or propose the narrowest new one.
- Percentiles over averages; realistic scale over toy data; one variable per
  comparison on the same hardware.

## Review Traps

- Frontend: anything that grows per token-delta is a leak; structural events
  and artifacts are the durable surface. Re-render counts matter as much as
  bytes.
- Imaging: latency regressions hide in cache-miss paths and cold pyramids;
  measure both warm and cold.
- Transport: ingest throughput claims that bypass JetStream (direct Postgres
  benchmarks) do not prove drain-time budgets after an outage.
- Local-stack numbers do not transfer to prod (different disks, NICs, and
  cache states); label the environment on every number.

## Durable Lessons

- 2026-07-05: Memory initialized. Append dated, sourced measurements and
  optimization outcomes here via parent-approved "Memory updates"; each entry
  needs the exact command and environment so future sessions can re-run it.
