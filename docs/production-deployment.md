# Production Deployment

BisQue Ultra runs cleanly in production when you separate three concerns:

1. public traffic and TLS
2. long-lived application processes
3. stateful platform services

This repo now includes the deployment scaffolding for that split.

## Recommended Topology

Use two machines:

- **App node**
  - external `nginx`
  - BisQue
  - Keycloak
  - Postgres
  - `ultra-backend@1`
  - `ultra-backend@2`
  - static frontend assets
- **Model node**
  - `vLLM` or `Ollama`
  - private OpenAI-compatible endpoint

The app node keeps browser traffic, storage, and auth local. The model node absorbs the heavy inference load and can be replaced without changing the web stack.

## Repo Assets Added For Production

- `deploy/env/ultra-backend.env.example`
- `deploy/env/platform.env.example`
- `deploy/nginx/*.template`
- `deploy/systemd/ultra-backend@.service`
- `deploy/systemd/ultra-backend.target`
- `platform/bisque/docker-compose.production.yml`
- `platform/bisque/docker/postgres/init/10-create-app-databases.sh`
- `scripts/render_production_templates.py`
- `scripts/bootstrap_production_host.sh`
- `scripts/deploy_ultra_backend.sh`
- `scripts/deploy_ultra_frontend.sh`
- `scripts/deploy_platform_manual.sh`
- `.github/workflows/deploy-*.yml`

## Runtime Layout

The deployment scripts assume this filesystem layout:

```text
/srv/ultra/
  current -> /srv/ultra/releases/<git-sha>/backend
  frontend-current -> /srv/ultra/releases/<git-sha>/frontend
  releases/
    <git-sha>/
      backend/
      frontend/
  ops/
  shared/
    artifacts/
    uploads/
    sessions/
    science/
```

The backend continues to use local-disk roots, so both backend instances stay on the same node and share the same filesystem.

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
- `RUN_STORE_PATH`
- `ARTIFACT_ROOT`
- `SESSION_UPLOAD_ROOT`
- `SCIENCE_DATA_ROOT`
- `UPLOAD_STORE_ROOT`
- all Ultra-side OIDC variables
- model endpoint and model name

### `platform.env`

This file is for Docker Compose, BisQue, Keycloak, Postgres bootstrap, and `nginx` template rendering. It should contain:

- `BISQUE_PUBLIC_HOST`
- `AUTH_PUBLIC_HOST`
- `POSTGRES_PASSWORD`
- `BISQUE_DB_*`
- `KEYCLOAK_DB_*`
- `ULTRA_DB_*`
- BisQue OIDC variables
- Keycloak admin bootstrap values

## Initial Host Bootstrap

1. Install system packages:
   - `nginx`
   - `docker` + Compose plugin
   - `python3`
   - `uv`
   - `rsync`
2. Copy the env examples into `/etc/ultra/` and fill them with real values.
3. Create the release layout:

   ```bash
   sudo ./scripts/bootstrap_production_host.sh \
     --ultra-env /etc/ultra/ultra-backend.env \
     --platform-env /etc/ultra/platform.env
   ```

4. Render the `nginx` configs into a staging directory:

   ```bash
   python3 scripts/render_production_templates.py \
     --ultra-env /etc/ultra/ultra-backend.env \
     --platform-env /etc/ultra/platform.env \
     --output-dir .tmp/rendered-production
   ```

5. Copy the rendered `nginx` files to `/etc/nginx/sites-available/`, enable them, and reload `nginx`.
6. Install the `systemd` units from `deploy/systemd/` into `/etc/systemd/system/` and run `sudo systemctl daemon-reload`.
7. Enable the backend target:

   ```bash
   sudo systemctl enable ultra-backend.target
   ```

## Platform Bring-Up

The production platform stack uses the base BisQue compose file plus the production override:

```bash
ENV_FILE=/etc/ultra/platform.env make platform-up-prod
```

That production override does four things:

- binds BisQue, Keycloak, and Postgres to localhost-only ports
- switches Keycloak out of `start-dev`
- adds persistent Keycloak data
- initializes separate `bisque`, `keycloak`, and `ultra` databases in one Postgres instance

Use this only for the platform. Ultra backend and frontend deploy separately.

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

Automatic deploys are split by concern:

- `deploy-ultra-frontend.yml`
  - triggers on `frontend/**`
  - ships static assets only
- `deploy-ultra-backend.yml`
  - triggers on `src/**`, `pyproject.toml`, `uv.lock`, and backend deploy scripts
  - ships the backend release only
- `deploy-platform-manual.yml`
  - runs only when you dispatch it manually
  - syncs BisQue/Keycloak/platform changes and runs the platform deploy script

Normal pushes to `main` do not redeploy BisQue.

## GitHub Secrets And Variables

Set these in GitHub:

### Secrets

- `DEPLOY_SSH_PRIVATE_KEY`
- `DEPLOY_SSH_HOST`
- `DEPLOY_SSH_USER`
- `DEPLOY_SSH_KNOWN_HOSTS`

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

- `https://ultra.ece.ucsb.edu/`
- `https://ultra.ece.ucsb.edu/v1/health`
- `https://ultra.ece.ucsb.edu/v1/config/public`
- `https://bisque.example.com/image_service/formats`
- `https://auth.example.com/realms/bisque/.well-known/openid-configuration`
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
