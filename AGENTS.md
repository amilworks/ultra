# AGENTS.md

## Purpose
This repo may be shared publicly. Keep production guidance precise, low-risk, and reusable, but do not publish private hostnames, usernames, credentials, or environment-specific endpoints here. Store concrete operator values in private runbooks outside the repo.

## Public Production Topology Template

Public hostname:

- `https://<public-host>`

Example node responsibilities:

- `<app-node>`
  - public edge via `Caddy`
  - static Ultra frontend
  - `ultra-backend@1`
  - `ultra-backend@2`
- `<platform-node>`
  - BisQue platform stack
  - internal platform proxy
  - `bisque-server`
  - `Keycloak`
  - `Postgres`
- model or inference services
  - external to ordinary app deploys
  - do not assume routine app rollouts need model-node changes

## SSH Entry Points

Use your private operator inventory for the exact SSH targets. In a split-node deployment the common shape is:

- app node: `ssh <app-user>@<app-node>`
- platform node: `ssh <platform-user>@<platform-node>`

Prefer direct SSH to the platform node when you need to inspect the platform stack instead of assuming a jump-host flow.

## Production File And Env Locations

### App node

- backend env: `/etc/ultra/ultra-backend.env`
- active backend symlink: `/srv/ultra/current`
- active frontend symlink: `/srv/ultra/frontend-current`
- release root: `<release-root>/releases/<git-sha>/`
- local runtime roots:
  - `/srv/ultra/models`
  - `/srv/ultra/runtime`
  - `/srv/ultra/shared`

### Platform node

- platform env: `/etc/ultra/platform.env`
- active platform checkout or symlink: `/srv/ultra/platform-current`
- platform releases: `<release-root>/platform-releases/<git-sha>/`
- BisQue config on shared storage: `<platform-data-root>/bisque-config/site.cfg`
- BisQue file storage root: `<platform-data-root>/`

## What Runs Where

### App node

Treat the app node as the ordinary target for:

- frontend asset updates
- FastAPI backend rollouts
- Pro Mode app/runtime changes
- app-side env changes in `/etc/ultra/ultra-backend.env`

It does not own:

- BisQue container rebuilds
- Keycloak admin/bootstrap config
- Postgres lifecycle

### Platform node

Treat the platform node as the ordinary target for:

- `docker compose` platform rollouts
- BisQue auth/platform behavior
- Keycloak realm/client state
- Postgres and platform storage config
- platform-side env changes in `/etc/ultra/platform.env`

## Production Deploy Workflow

### App rollout

Use committed release snapshots for production deploys. Do not deploy a dirty working tree by accident.

Recommended flow:

1. Build and test locally from the exact committed SHA you intend to ship.
2. Stage backend release contents to:
   - `<release-root>/releases/<git-sha>/backend`
3. Stage built frontend assets to:
   - `<release-root>/releases/<git-sha>/frontend`
4. On the app node, deploy backend:

   ```bash
   sudo ULTRA_RELEASE_ROOT=/srv/ultra \
     <release-root>/releases/<git-sha>/backend/scripts/deploy_ultra_backend.sh <git-sha>
   ```

5. On the app node, deploy frontend:

   ```bash
   sudo ULTRA_RELEASE_ROOT=/srv/ultra \
     <release-root>/releases/<git-sha>/backend/scripts/deploy_ultra_frontend.sh <git-sha>
   ```

6. Verify:
   - `readlink -f /srv/ultra/current`
   - `readlink -f /srv/ultra/frontend-current`
   - `systemctl is-active ultra-backend@1 ultra-backend@2`
   - `systemctl is-active caddy`
   - `curl -fsS https://<public-host>/v1/health`

### Platform rollout

Use the platform node only when the change touches BisQue, Keycloak, Postgres, or platform proxying.

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
   - `curl -fsS https://<public-host>/auth/realms/bisque/.well-known/openid-configuration`
   - `curl -fsS https://<public-host>/image_service/formats`

## Deployment Gotchas

- The app node may use `Caddy` rather than `nginx`. Frontend deploy helpers must validate and reload the active web server.
- Do not `source /etc/ultra/ultra-backend.env` blindly for ad hoc checks; values like `BISQUE_AUTH_OIDC_SCOPE=openid profile email` contain spaces.
- If Opus appears configured but production silently falls back to `gpt-oss-120b`, verify the exact Pro Mode API key contents, not just that the variable exists.
- If a Bedrock-published API is fronted by an API Gateway wrapper, verify whether the health and conversation endpoints live under `/api` instead of the host root.
- Keep Postgres and Keycloak on local Docker volumes on the platform node. Do not move them onto shared network storage.
- In split-node routing, BisQue admin UI requires `/admin*` on both the public edge and the internal platform proxy.

## Fast Verification Matrix

After app deploys:

- `https://<public-host>/`
- `https://<public-host>/v1/health`
- one Pro Mode direct-response prompt
- one Pro Mode tool workflow

After platform deploys:

- `https://<public-host>/auth/realms/bisque/.well-known/openid-configuration`
- `https://<public-host>/client_service/`
- `https://<public-host>/auth_service/whoami`
- `https://<public-host>/image_service/formats`
- one browser login
- one BisQue upload/search flow

## GitHub Actions

GitHub Actions are verification-first.

- backend and frontend workflows run CI checks
- production rollout is still a manual operator action
- do not assume a successful GitHub run means production has been updated
