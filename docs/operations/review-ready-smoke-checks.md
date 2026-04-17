# Review-Ready Smoke Checks

Use this checklist after auth, frontend shell, or backend routing changes.

## Local Verification

1. Backend quality
   - `uv sync --frozen --extra dev`
   - `uv run ruff check src tests`
   - `uv run ruff format --check src tests`
   - `uv run mypy src/config.py`
2. Backend tests
   - `uv run pytest -q`
3. Frontend quality
   - `pnpm --dir frontend lint`
   - `pnpm --dir frontend typecheck`
   - `pnpm --dir frontend test:unit`
4. Frontend smoke
   - `pnpm --dir frontend build`
   - `pnpm --dir frontend test:smoke`

## Auth Regression Checklist

- `/v1/auth/session` works with no API key.
- Browser login/guest/logout flows work with cookies only.
- Browser-facing thumbnail, preview, viewer, atlas, tile, slice, and artifact download links contain no `api_key` query param.
- `X-API-Key` still works for automation-only requests.
- Query-string `api_key` is accepted only when compatibility is intentionally enabled.
- Admin endpoints reject API-key-only callers and guest sessions.

## Rollback Points

- If browser auth breaks after rollout, revert the frontend build first and verify `/v1/auth/session` plus `/v1/config/public`.
- If service clients break, confirm `X-API-Key` handling before re-enabling query-string compatibility.
- If viewer assets fail, verify authenticated session cookies, then verify `/v1/resources/{file_id}/thumbnail` and `/v1/uploads/{file_id}/viewer`.
- If observability causes noise, keep request IDs enabled and temporarily disable metrics scraping before removing middleware.
