# Production Deployment

BisQue Ultra runs cleanly in production when you separate three concerns:

1. public traffic and TLS
2. long-lived application processes
3. stateful platform services

This repo now includes the deployment scaffolding for that split.

## Recommended Topology

Use two machines if you have them. If you only have one public DNS name, the clean production split is still:

- **App node**
  - external `Caddy` or `nginx`
  - `ultra-backend@1`
  - `ultra-backend@2`
  - static frontend assets
- **Platform node**
  - BisQue
  - Keycloak
  - Postgres
  - internal `Caddy`
- **Model node**
  - `vLLM` or `Ollama`
  - private OpenAI-compatible endpoint

The app node keeps the public edge small and responsive. The platform node absorbs the legacy BisQue/UI bootstrap load. The model node absorbs inference and can be replaced without changing the web stack.

## Single-Hostname Production Layout

For the single-host deployment where only `ultra.example.com` exists publicly, keep one public edge and split by path:

- `https://ultra.example.com/`
  - static Ultra frontend
- `https://ultra.example.com/v1/*`
  - Ultra backend API
- `https://ultra.example.com/v3/*`
  - Ultra run/session API
- `https://ultra.example.com/auth/*`
  - Keycloak on the platform node
- `https://ultra.example.com/client_service/*`
  - BisQue browser UI on the platform node
- `https://ultra.example.com/auth_service/*`
  - BisQue auth endpoints
- `https://ultra.example.com/data_service/*`
  - BisQue data APIs
- `https://ultra.example.com/image_service/*`
  - BisQue image APIs

Do not put BisQue behind `/bisque`.

BisQue is an older platform with many root-path assumptions baked into redirects, templates, JS, and generated URLs. Keep BisQue on its native top-level service namespaces and let the public edge dispatch those namespaces to the platform node. Keycloak is the component that should live under a path prefix, because it supports a relative path cleanly.

## Repo Assets Added For Production

- `deploy/env/ultra-backend.env.example`
- `deploy/env/platform.env.example`
- `deploy/caddy/Caddyfile.platform-node.template`
- `deploy/nginx/*.template`
- `deploy/systemd/ultra-backend@.service`
- `deploy/systemd/ultra-backend.target`
- `deploy/systemd/ultra-platform.service`
- `platform/bisque/docker-compose.production.yml`
- `platform/bisque/docker-compose.platform-node.yml`
- `platform/bisque/docker/postgres/init/10-create-app-databases.sh`
- `scripts/render_production_templates.py`
- `scripts/bootstrap_production_host.sh`
- `scripts/deploy_ultra_backend.sh`
- `scripts/deploy_ultra_frontend.sh`
- `scripts/deploy_platform_manual.sh`
- `.github/workflows/deploy-*.yml`

## Runtime Layout

The deployment scripts assume this filesystem layout on the app node:

```text
/srv/ultra/
  current -> /srv/ultra/releases/<git-sha>/backend
  frontend-current -> /srv/ultra/releases/<git-sha>/frontend
  releases/
    <git-sha>/
      backend/
      frontend/
  ops/
  models/
    yolo/
    medsam2/
      checkpoints/
    sam3/
      facebook-sam3/
  runtime/
    MedSAM2/
  shared/
    artifacts/
    uploads/
    sessions/
    science/
```

On the platform node, use the shared barrel mount only for BisQue file storage:

```text
/mnt/barrel-data/ultra/platform/
  bisque-config/
  bisque-data/
  bisque-public/
  bisque-reports/
  bisque-staging/
```

Postgres and Keycloak should use local Docker volumes on the platform node. Do
not place Postgres data on the barrel NFS mount: transient NFS stalls can leave
postgres processes in uninterruptible sleep and prevent fresh connections.

The backend continues to use local-disk roots, so both backend instances stay
on the same app node and share the same filesystem.

## Server-Side Environment Files

Do not deploy from the repo `.env`.

Create:

- `/etc/ultra/ultra-backend.env`
- `/etc/ultra/platform.env`

Start from the examples in `deploy/env/`.

### `ultra-backend.env`

This file is for the FastAPI app and the backend deploy scripts. It should contain:

- `ENVIRONMENT=production`
- `ULTRA_PUBLIC_HOST`
- `ULTRA_RELEASE_ROOT`
- `BISQUE_ROOT`

For deterministic scientific tooling, also point the backend at stable local
model/runtime paths on the app node instead of release-relative paths:

- `YOLO_DEFAULT_MODEL=/srv/ultra/models/yolo/yolo26x.pt`
- `YOLOV5_RARESPOT_WEIGHTS=/srv/ultra/models/yolo/RareSpotWeights.pt`
- `MEDSAM2_RUNTIME_ROOT=/srv/ultra/runtime/MedSAM2`
- `MEDSAM2_CHECKPOINT_DIR=/srv/ultra/models/medsam2/checkpoints`
- `SAM3_MODEL_ID=/srv/ultra/models/sam3/facebook-sam3`

Use `scripts/sync_science_model_assets.sh --source <old-local-repo> --remote-host <app-node>`
to seed those local assets without committing weights into Git.

### `platform.env`

This file is for BisQue, Keycloak, Postgres, and split-node routing. In addition to hostnames and database credentials, keep the BisQue SQLAlchemy pool settings here. The authenticated first-load burst for `client_service` is large enough that the legacy defaults can exhaust the DB pool behind a reverse proxy.

Recommended starting values:

- `BISQUE_SQLALCHEMY_POOL_SIZE=25`
- `BISQUE_SQLALCHEMY_MAX_OVERFLOW=25`
- `BISQUE_SQLALCHEMY_POOL_TIMEOUT=60`
- `BISQUE_SQLALCHEMY_POOL_PRE_PING=true`
- `PLATFORM_DEPLOY_MODE`
- `PLATFORM_DATA_ROOT`
- `PLATFORM_BIND_HOST`
- `PLATFORM_PROXY_PORT`
- `PLATFORM_DB_BIND_HOST`
- `AUTH_UPSTREAM`
- `BISQUE_UPSTREAM`

This file is also for Docker Compose, BisQue, Keycloak, Postgres bootstrap, and proxy template rendering. It should contain:

- `BISQUE_PUBLIC_HOST`
- `AUTH_PUBLIC_HOST`
- `KEYCLOAK_REALM_IMPORT_FILE`
- `POSTGRES_PASSWORD`
- `BISQUE_DB_*`
- `KEYCLOAK_DB_*`
- `ULTRA_DB_*`
- BisQue OIDC variables
- Keycloak admin bootstrap values

The platform environment owns the BisQue browser client in Keycloak. The Ultra
API/web client is loaded from `ultra-backend.env` during platform deploy so the
realm can reconcile both intended production clients together.

For hardened production auth, keep these values in the platform environment:

- `BISQUE_AUTH_MODE=oidc`
- `BISQUE_AUTH_LOCAL_TOKEN_ENABLED=false`
- `BISQUE_AUTH_COOKIE_SECURE=true`
- `BISQUE_BEAKER_SESSION_SECURE=true`
- `KEYCLOAK_CLIENT_DIRECT_ACCESS_GRANTS=false`
- `KEYCLOAK_ADMIN_SERVER_URL=http://127.0.0.1:8080/auth`

### Production Keycloak Client Model

This repo now treats the Keycloak realm as having two intentional confidential
web clients:

- `bisque-web`
  - BisQue browser UI
  - callback: `/auth_service/oidc_callback`
- `ultra-web`
  - Ultra backend/browser login bridge
  - callback: `/v1/auth/oidc/callback`

`scripts/deploy_platform_manual.sh` loads both `/etc/ultra/platform.env` and
`/etc/ultra/ultra-backend.env` before it renders and reconciles the realm, so
the live Keycloak client configuration stays aligned with both services.

The production realm renderer intentionally strips the dev users from
`platform/bisque/docker/keycloak/realm-bisque-dev.json` and replaces the dev
client list with only the configured production client(s).

## Initial Host Bootstrap

1. Install system packages:
   - `caddy` or `nginx` on the app node
   - `docker` + Compose plugin
   - `python3`
   - `uv`
   - `rsync`
   - `libgl1` and `libglib2.0-0` on Ubuntu/Debian app nodes so `opencv-python` / `ultralytics` can import for YOLO and segmentation tools
2. Copy the env examples into `/etc/ultra/` and fill them with real values.
3. Create the release layout:

   ```bash
   sudo ./scripts/bootstrap_production_host.sh \
     --ultra-env /etc/ultra/ultra-backend.env \
     --platform-env /etc/ultra/platform.env
   ```

   On Ubuntu/Debian, the bootstrap script now installs `libgl1` and
   `libglib2.0-0` automatically when it is run as root and those packages are
   missing.

4. Render the proxy configs into a staging directory:

   ```bash
   python3 scripts/render_production_templates.py \
     --ultra-env /etc/ultra/ultra-backend.env \
     --platform-env /etc/ultra/platform.env \
     --output-dir .tmp/rendered-production
   ```

5. On the app node, install `Caddyfile.single-host` or `ultra-single-host.conf` as the active edge config and reload the service.
6. On the platform node, render and install `Caddyfile.platform-node` to the path referenced by `PLATFORM_CADDYFILE`.

### Fresh Install Reset

If you need a clean rebuild of the platform node, wipe the platform containers
and the local Docker volumes, but keep the BisQue barrel-backed file store if
you want to preserve uploaded images:

```bash
cd /srv/ultra/platform-current
docker compose \
  --env-file /etc/ultra/platform.env \
  -f platform/bisque/docker-compose.with-engine.yml \
  -f platform/bisque/docker-compose.production.yml \
  -f platform/bisque/docker-compose.platform-node.yml \
  down -v --remove-orphans

docker volume rm -f ultra-platform-postgres ultra-platform-keycloak || true
```

If you truly want a full wipe, remove `/mnt/barrel-data/ultra/platform/bisque-*`
after the containers are down and then redeploy.
7. Install the `systemd` units from `deploy/systemd/` into `/etc/systemd/system/` and run `sudo systemctl daemon-reload`.
8. Enable the backend target on the app node:

   ```bash
   sudo systemctl enable ultra-backend.target
   ```

## Platform Bring-Up

The production platform stack uses the base BisQue compose file plus the production override:

```bash
ENV_FILE=/etc/ultra/platform.env \
ULTRA_ENV=/etc/ultra/ultra-backend.env \
make platform-up-prod
```

For the split-node platform layout, add the platform-node override:

```bash
ENV_FILE=/etc/ultra/platform.env \
ULTRA_ENV=/etc/ultra/ultra-backend.env \
PLATFORM_DEPLOY_MODE=platform-node \
./scripts/deploy_platform_manual.sh up
```

The production overrides:

- switch all stateful services to barrel-backed bind mounts
- switch Keycloak out of `start-dev`
- render a production realm import with public redirect URIs and web origins
- strip dev users and dev-only client baggage from the rendered realm
- reconcile the live Keycloak client(s) on every deploy so existing realms do not drift
- keep Keycloak under `/auth`
- initialize separate `bisque`, `keycloak`, and `ultra` databases in one Postgres instance
- add a small internal Caddy that exposes one platform ingress port
- keep BisQue and Keycloak off the public internet

Use this only for the platform. Ultra backend and frontend still deploy separately on the app node.

## Ultra Backend Rollout

The backend deploy script assumes the release has already been copied to:

`/srv/ultra/releases/<git-sha>/backend`

Then it:

1. installs Python deps with `uv sync --frozen`
2. atomically switches `/srv/ultra/current`
3. restarts `ultra-backend@1`
4. waits for `http://127.0.0.1:8001/v1/health`
5. restarts `ultra-backend@2`
6. waits for `http://127.0.0.1:8002/v1/health`

Manual rollout:

```bash
sudo ULTRA_RELEASE_ROOT=/srv/ultra ./scripts/deploy_ultra_backend.sh <git-sha>
```

## Ultra Frontend Rollout

The frontend deploy script assumes the built assets already exist at:

`/srv/ultra/releases/<git-sha>/frontend`

Then it:

1. atomically switches `/srv/ultra/frontend-current`
2. runs `nginx -t`
3. reloads `nginx`

Manual rollout:

```bash
sudo ULTRA_RELEASE_ROOT=/srv/ultra ./scripts/deploy_ultra_frontend.sh <git-sha>
```

## GitHub Actions

GitHub Actions are currently split into lightweight verification plus manual platform operations:

- `deploy-ultra-frontend.yml`
  - triggers on frontend changes for pushes and pull requests
  - installs dependencies and builds the frontend
- `deploy-ultra-backend.yml`
  - triggers on backend changes for pushes and pull requests
  - syncs the Python environment, compiles key modules, and runs focused backend tests
- `deploy-platform-manual.yml`
  - runs only when you dispatch it manually or trigger `repository_dispatch`
  - syncs BisQue/Keycloak/platform changes and runs the platform deploy script

Normal pushes do not auto-roll production anymore. App deploys happen manually from an operator shell, and the platform workflow targets the platform node explicitly when you choose to run it.

## GitHub Secrets And Variables

Set these in GitHub:

### Secrets

- `PLATFORM_DEPLOY_SSH_PRIVATE_KEY`
- `PLATFORM_DEPLOY_SSH_HOST`
- `PLATFORM_DEPLOY_SSH_USER`
- `PLATFORM_DEPLOY_SSH_KNOWN_HOSTS`
- `PLATFORM_DEPLOY_SSH_JUMP_HOST`

### Variables

- `ULTRA_DEPLOY_ROOT`
  - default if omitted: `/srv/ultra`

Do not put the full app runtime `.env` into GitHub Secrets unless you intentionally want GitHub to own runtime configuration.

## Verification

### Before a live deploy

- `make verify-platform-smoke`
- `make verify-integration`
- `pnpm --dir frontend build`
- `bash -n scripts/deploy_ultra_backend.sh scripts/deploy_ultra_frontend.sh scripts/deploy_platform_manual.sh scripts/bootstrap_production_host.sh`
- `python3 scripts/render_production_templates.py --help`
- `docker compose --env-file deploy/env/platform.env.example -f platform/bisque/docker-compose.with-engine.yml -f platform/bisque/docker-compose.production.yml config`

### After a live deploy

- `https://ultra.example.com/`
- `https://ultra.example.com/v1/health`
- `https://ultra.example.com/v1/config/public`
- `https://ultra.example.com/image_service/formats`
- `https://ultra.example.com/auth/realms/bisque/.well-known/openid-configuration`
- one browser login
- one upload
- one `/v1/chat`
- one BisQue resource search
- one Pro Mode run
- one artifact-producing tool run

## Rollback

- frontend rollback:
  - point `/srv/ultra/frontend-current` back to the previous frontend release
  - `sudo systemctl reload nginx`
- backend rollback:
  - point `/srv/ultra/current` back to the previous backend release
  - restart `ultra-backend@1` then `ultra-backend@2`
- platform rollback:
  - manual only
  - do not couple it to ordinary Ultra deploys
