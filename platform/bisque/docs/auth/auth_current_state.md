# BisQue Auth Current State (Milestone 4)

Date: 2026-02-17

## Active Authentication Paths

1. Browser session cookie (`authtkt`)
- Entry: `/auth_service/login`
- Legacy mode: local login form remains active.
- Dual/OIDC mode: login entrypoint redirects to `/auth_service/oidc_login` unless `legacy=1`.
- If `bisque.oidc.login_page_enabled=true`, the BisQue login page is rendered first with a primary OIDC button.
- Default OIDC compose setup now uses Keycloak realm login theme `bisque-llm` for a standardized hosted login page.
- OIDC logout defaults to returning users to `/client_service/` (configurable via `bisque.oidc.post_logout_redirect_uri`).
- Session bridge: successful OIDC/Firebase login mints the same `authtkt` cookie path used by legacy TurboGears flow.

2. API Basic auth
- Header: `Authorization: Basic ...`
- Routed through `repoze.who` middleware.
- Allowed in `legacy` and `dual`.
- Blocked in `oidc` mode with deterministic `401` + JSON error (`basic_auth_disabled`).

3. API Bearer auth
- Header: `Authorization: Bearer <token>`
- Routed through `repoze.who` bearer plugin.
- Token validation supports:
  - Local BisQue tokens (`/auth_service/token`) when enabled.
  - OIDC tokens validated via issuer metadata + JWKS.
- Available in all modes, but local token issuance can be disabled by mode/config.

4. Module/engine MEX auth
- Header: `Authorization: Mex ...`
- Remains unchanged in Milestone 4 to preserve module dispatch/callback compatibility.

5. Firebase social auth
- Endpoints under `/auth_service/firebase_*`.
- Now uses the same cookie-bridge helper as OIDC for post-login session creation.

## Auth Mode Semantics

`bisque.auth.mode=legacy`
- Browser: legacy login form.
- API: Basic + Bearer(local token) + Mex.
- `/auth_service/token`: enabled.

`bisque.auth.mode=dual`
- Browser: OIDC redirect by default, legacy fallback with `?legacy=1`.
- API: Basic + Bearer(local/OIDC) + Mex.
- `/auth_service/token`: enabled by default.

`bisque.auth.mode=oidc`
- Browser: OIDC required.
- API: Basic disabled, Bearer(OIDC) + Mex.
- `/auth_service/token`: disabled unless explicitly re-enabled via `bisque.auth.local_token.enabled=true`.

## Request Routing Summary

`ConditionalAuthMiddleware` behavior:
- `Authorization: Basic|Bearer|Mex` -> route through `repoze.who`.
- No auth header -> route through TurboGears cookie/session path.
- In `oidc` mode, Basic requests return deterministic `401` JSON before auth plugin processing.

## User Mapping Rules

OIDC claim mapping:
- Username priority: configured username claim, then `preferred_username`, `email`, `sub`.
- Groups claim: configurable (`bisque.oidc.groups_claim`, default `groups`).
- On first OIDC login, local user is auto-created if no match exists; BisQue tags are upserted:
  - `oidc_sub`, `oidc_issuer`, `oidc_provider`, `oidc_groups`.

## Known Constraints

- OIDC JWT verification requires `cryptography` runtime dependency.
- `oidc` mode intentionally blocks local password token grant to prevent silent downgrade.
- MEX auth is intentionally not replaced in Milestone 4 to avoid engine/module regressions.
