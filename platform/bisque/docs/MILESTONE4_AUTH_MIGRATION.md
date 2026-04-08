# Milestone 4 Auth Migration Runbook

Date: 2026-02-17

## Goal

Roll BisQue authentication from legacy-only behavior to modern OIDC-compatible auth without breaking:
- browser login/session behavior,
- API access (`bqapi`, scripts, notebooks),
- module execution (`Mex`, engine callbacks).

## Modes

1. `legacy`
- Existing behavior only.
- Basic + local token + cookie + Mex.

2. `dual` (recommended transition default)
- Browser defaults to OIDC entrypoint.
- Legacy login can still be used with `?legacy=1`.
- Basic + Bearer + Mex remain available.

3. `oidc`
- Browser/API require OIDC flow.
- Basic disabled.
- Local password token grant disabled by default.
- Mex remains active for module runtime.

## Local Near-Production Stack

Start BisQue + engine + Keycloak:

```bash
docker compose -f docker-compose.with-engine.yml -f docker-compose.oidc.yml up -d --build
```

Confirm health:

```bash
docker inspect --format '{{.State.Health.Status}}' bisque-server
curl -fsS http://127.0.0.1:18080/realms/bisque/.well-known/openid-configuration
curl -fsS http://127.0.0.1:8080/services | rg 'type="client_service"|type="module_service"|type="image_service"'
```

## Mode Switching (Local Config)

```bash
./scripts/dev/auth_mode_switch.sh dual
./scripts/dev/test_auth_baseline.sh --expect-dual

./scripts/dev/auth_mode_switch.sh oidc
./scripts/dev/test_auth_baseline.sh --expect-oidc

./scripts/dev/auth_mode_switch.sh legacy
./scripts/dev/test_auth_baseline.sh --expect-legacy
```

## OIDC Token Retrieval

```bash
source .venv/bin/activate
python scripts/dev/oidc_get_token.py --provider local --user admin --password admin --print-access-token
```

## API/`bqapi` Validation

1. Local token (`legacy`/`dual`):

```bash
curl -X POST http://127.0.0.1:8080/auth_service/token \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin","grant_type":"password"}'
```

2. OIDC token (`dual`/`oidc`):

```bash
OIDC_TOKEN="$(python scripts/dev/oidc_get_token.py --provider local --user admin --password admin --print-access-token)"
curl -H "Authorization: Bearer ${OIDC_TOKEN}" http://127.0.0.1:8080/auth_service/whoami
```

3. Module execution via bearer token:

```bash
python scripts/dev/bqapi_module_runner.py \
  --bisque-root http://127.0.0.1:8080 \
  --token "${OIDC_TOKEN}" \
  --module EdgeDetection \
  --images test_images/00-db8XD5ccsYdeysUv2Jr9fc.ome-tiff.ome.tif \
  --summary-json artifacts/m4_auth/edge_oidc_summary.json
```

## Browser/OIDC Verification

1. Visit:
- `http://localhost:8080/auth_service/login`
- In `dual/oidc`, confirm redirect chain enters `/auth_service/oidc_login` and Keycloak login UI.

2. After successful login (`admin/admin` in local realm), verify:
- redirected back to `came_from`,
- `http://localhost:8080/auth_service/session` shows authenticated user,
- module pages load without login loop.

## Rollback

Immediate rollback path:

```bash
./scripts/dev/auth_mode_switch.sh legacy
./scripts/dev/test_auth_baseline.sh --expect-legacy
```

For containerized rollout, set:

```bash
BISQUE_AUTH_MODE=legacy
BISQUE_AUTH_LOCAL_TOKEN_ENABLED=true
```

then restart BisQue.

## Production Hardening Checklist

Before production cutover:
- replace dev IdP client secret and admin credentials,
- enable TLS for BisQue and IdP endpoints,
- set strict redirect URIs/web origins,
- decide audience verification policy and enable where required,
- monitor auth and engine callback logs during first staged rollout,
- keep `dual` mode first, then move to `oidc` after successful soak period.
