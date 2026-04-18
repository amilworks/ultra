# Review-Ready Trust Upgrade

This wave hardens the repo around four stable slices so new work can land outside the app shell and API monolith:

- `auth`
  - Browser session bootstrap, guest/BisQue auth, admin session checks, and request auth context.
- `chat/runtime`
  - Conversation execution, streaming, run ownership, and v3 runtime entry points.
- `resources/uploads/viewer`
  - Resource browsing, upload metadata, thumbnails, preview rendering, and viewer assets.
- `admin/ops`
  - Platform overview, run moderation, issue surfacing, model-health inspection, and operational metrics.

## Auth Model

- Browser UI uses same-origin cookies for interactive auth.
- `X-API-Key` remains valid for automation and service clients.
- Query-string `api_key` is a temporary compatibility bridge controlled by `ALLOW_QUERY_API_KEY_COMPAT`.
- Admin endpoints require a signed-in BisQue session that resolves to an allowed admin username.
- Browser-facing viewer and resource asset routes require an authenticated guest or BisQue session and do not rely on URL secrets.

## Dependency Map

- `src/api/main.py`
  - App wiring, middleware, compatibility shims, and router registration.
- `src/auth.py`
  - Session persistence and request-scoped BisQue auth helpers.
- `src/orchestration/store.py`
  - Conversation, run, admin, and artifact persistence.
- `src/agno_backend/runtime.py`
  - Chat/runtime execution engine and v3-adjacent orchestration.
- `frontend/src/App.tsx`
  - App shell and cross-slice composition only.
- `frontend/src/features/auth/*`
  - Browser auth and BisQue navigation helpers.
- `frontend/src/lib/api.ts`
  - Shared API client; browser URL builders must stay free of `api_key` query params.

## Guardrails For This Wave

- Do not add new feature logic to `src/api/main.py` or `frontend/src/App.tsx` when a slice module can own it.
- Do not persist browser secrets in local storage, session storage, or URLs.
- Do not add broad `except Exception` blocks in touched code unless the exception is intentionally translated into a typed API response.
