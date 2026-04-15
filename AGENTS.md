# AGENTS.md

## Purpose
This production repo is the operator-facing source of truth for BisQue Ultra deploys. Prefer precise, low-risk runbook steps over generic advice. When production guidance conflicts with development habits, follow the production guidance here.

## Current UCSB Production Topology

Public hostname:

- `https://ultra.ece.ucsb.edu`

Current node responsibilities:

- `nail01.ece.ucsb.edu`
  - public edge via `Caddy`
  - static Ultra frontend
  - `ultra-backend@1`
  - `ultra-backend@2`
- `nail04.ece.ucsb.edu`
  - BisQue platform stack
  - internal platform `Caddy`
  - `bisque-server`
  - `Keycloak`
  - `Postgres`
- model/inference services
  - external to ordinary app deploys
  - do not assume routine app rollouts need model-node changes

## SSH Entry Points

Use these exact production SSH paths:

- app node: `ssh bisque_amil@nail01.ece.ucsb.edu`
- platform node: `ssh amil@nail04.ece.ucsb.edu`

Prefer direct SSH to `nail04`; the older jump-host flow is no longer the default path.

## Production File And Env Locations

### App node (`nail01`)

- backend env: `/etc/ultra/ultra-backend.env`
- active backend symlink: `/srv/ultra/current`
- active frontend symlink: `/srv/ultra/frontend-current`
- release root: `/mnt/barrel-data/ultra/releases/<git-sha>/`
- local runtime roots:
  - `/srv/ultra/models`
  - `/srv/ultra/runtime`
  - `/srv/ultra/shared`

### Platform node (`nail04`)

- platform env: `/etc/ultra/platform.env`
- active platform checkout/symlink: `/srv/ultra/platform-current`
- platform releases: `/srv/ultra/platform-releases/<git-sha>/`
- BisQue config on shared storage: `/mnt/barrel-data/ultra/platform/bisque-config/site.cfg`
- BisQue file storage root: `/mnt/barrel-data/ultra/platform/`

## What Runs Where

### `nail01`

Treat `nail01` as the ordinary app deploy target.

It owns:

- frontend asset updates
- FastAPI backend rollouts
- Pro Mode app/runtime changes
- app-side env changes in `/etc/ultra/ultra-backend.env`

It does not own:

- BisQue container rebuilds
- Keycloak admin/bootstrap config
- Postgres lifecycle

### `nail04`

Treat `nail04` as the platform deploy target.

It owns:

- `docker compose` platform rollouts
- BisQue auth/platform behavior
- Keycloak realm/client state
- Postgres and platform storage config
- platform-side env changes in `/etc/ultra/platform.env`

## Production Deploy Workflow

### App rollout (`nail01`)

Use committed release snapshots for production deploys. Do not deploy a dirty working tree by accident.

Recommended flow:

1. Build/test locally from the exact committed SHA you intend to ship.
2. Stage backend release contents to:
   - `/mnt/barrel-data/ultra/releases/<git-sha>/backend`
3. Stage built frontend assets to:
   - `/mnt/barrel-data/ultra/releases/<git-sha>/frontend`
4. On `nail01`, deploy backend:

   ```bash
   sudo ULTRA_RELEASE_ROOT=/srv/ultra \
     /mnt/barrel-data/ultra/releases/<git-sha>/backend/scripts/deploy_ultra_backend.sh <git-sha>
   ```

5. On `nail01`, deploy frontend:

   ```bash
   sudo ULTRA_RELEASE_ROOT=/srv/ultra \
     /mnt/barrel-data/ultra/releases/<git-sha>/backend/scripts/deploy_ultra_frontend.sh <git-sha>
   ```

6. Verify:
   - `readlink -f /srv/ultra/current`
   - `readlink -f /srv/ultra/frontend-current`
   - `systemctl is-active ultra-backend@1 ultra-backend@2`
   - `systemctl is-active caddy`
   - `curl -fsS https://ultra.ece.ucsb.edu/v1/health`

### Platform rollout (`nail04`)

Use `nail04` only when the change touches BisQue, Keycloak, Postgres, or platform proxying.

Typical flow:

1. Ensure `/etc/ultra/platform.env` and `/etc/ultra/ultra-backend.env` are both current.
2. Sync the committed platform release to `/srv/ultra/platform-releases/<git-sha>`.
3. Point `/srv/ultra/platform-current` at that release if needed.
4. Run the platform deploy script from the release checkout:

   ```bash
   cd /srv/ultra/platform-current
   ENV_FILE=/etc/ultra/platform.env \
   ULTRA_ENV=/etc/ultra/ultra-backend.env \
   PLATFORM_DEPLOY_MODE=platform-node \
   ./scripts/deploy_platform_manual.sh up
   ```

5. Verify:
   - `docker compose ps`
   - `curl -fsS https://ultra.ece.ucsb.edu/auth/realms/bisque/.well-known/openid-configuration`
   - `curl -fsS https://ultra.ece.ucsb.edu/image_service/formats`

## Deployment Gotchas

- `nail01` uses `Caddy`, not `nginx`. Frontend deploy helpers must validate/reload `Caddy`.
- Do not `source /etc/ultra/ultra-backend.env` blindly for ad hoc checks; values like `BISQUE_AUTH_OIDC_SCOPE=openid profile email` contain spaces.
- If Opus appears configured but production silently falls back to `gpt-oss-120b`, verify the exact `PRO_MODE_API_KEY` contents, not just that the variable exists.
- The UCSB Bedrock-published API is healthy on the `/api` path, not the host root.
- Keep Postgres and Keycloak on local Docker volumes on `nail04`. Do not move them back onto the barrel NFS mount.
- In split-node routing, BisQue admin UI requires `/admin*` on both the public `nail01` edge and the internal `nail04` platform proxy.

## Fast Verification Matrix

After app deploys:

- `https://ultra.ece.ucsb.edu/`
- `https://ultra.ece.ucsb.edu/v1/health`
- one Pro Mode direct-response prompt
- one Pro Mode tool workflow

After platform deploys:

- `https://ultra.ece.ucsb.edu/auth/realms/bisque/.well-known/openid-configuration`
- `https://ultra.ece.ucsb.edu/client_service/`
- `https://ultra.ece.ucsb.edu/auth_service/whoami`
- `https://ultra.ece.ucsb.edu/image_service/formats`
- one browser login
- one BisQue upload/search flow

## GitHub Actions

GitHub Actions are verification-first now.

- backend/frontend workflows run CI checks
- production rollout is still a manual operator action
- do not assume a successful GitHub run means production has been updated
