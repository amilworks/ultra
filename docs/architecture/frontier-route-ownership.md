# Frontier Route Ownership

This repo is moving toward slice ownership so a second team can extend it without reopening the monolith entry points by default.

## Backend slices

- `auth`
  - owner modules: `src/auth/*`
  - API surface: session bootstrap, guest sign-in, BisQue-auth context
  - integration boundary: request-scoped auth context plus cookie/session transport
- `chat/runtime`
  - owner modules: `src/api/v3.py`, `src/agno_backend/v3_services.py`, `src/agentic/*`
  - API surface: `/v3/sessions`, `/v3/runs`, `/v3/runs/{run_id}/events`, `/v3/runs/{run_id}/artifacts`
  - integration boundary: typed session/run/artifact contract
- `resources/viewer`
  - owner modules: `src/science/viewer.py`, upload/viewer endpoints in `src/api/main.py`
  - API surface: upload previews, slices, atlas delivery, HDF5 inspection
  - integration boundary: viewer manifests and artifact-backed previews
- `admin/ops`
  - owner modules: admin route handlers in `src/api/main.py`, request logging and metrics in `src/logger.py`
  - API surface: `/v1/admin/*`, `/v1/metrics`
  - integration boundary: admin-session auth plus structured request metadata
- `training`
  - owner modules: `src/training/*`
  - API surface: `/v1/training/*`, prairie active-learning workflows
  - integration boundary: typed job/domain/lineage records

## Frontend slices

- `frontend/src/features/auth`
  - browser auth and BisQue navigation helpers
- `frontend/src/features/chat`
  - conversation paging, run-event hydration, v3 session/run calls
- `frontend/src/features/resources`
  - library and composer resource lookups
- `frontend/src/features/admin`
  - overview, users, runs, and issues
- `frontend/src/features/training`
  - prairie/training dashboard snapshots and lineage discovery

## Current rule

`frontend/src/App.tsx` and `src/api/main.py` remain compatibility shells. New feature-specific data access should land in slice-owned modules first, then be wired into the shell.
