# Contributing to BisQue Ultra

Thanks for contributing! A few lightweight conventions keep the repo tidy and the
history readable.

## Branches & pull requests

- Branch off `main` with a descriptive name: `feat/…`, `fix/…`, `chore/…`, `refactor/…`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit subjects —
  `feat(viewer): …`, `fix(control-plane): …`, `chore(ci): …`.
- Open the PR into `main` and fill out the template. CI must be green before merge.

## Labels

Every issue and PR uses a scoped taxonomy so you can filter by *what · where · how urgent*:

- **`type:`** — exactly one per PR: `feature`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`.
- **`area:`** — one or more subsystems: `frontend`, `control-plane`, `worker`, `viewer`, `data`,
  `infra`, `agents`. These are **applied automatically** from your changed paths by the labeler
  bot ([`.github/labeler.yml`](.github/labeler.yml)) — adjust if it guesses wrong.
- **`priority:`** — `high` / `medium` / `low`, mainly for triaging open issues.
- **`breaking-change`** — backwards-incompatible API, schema, or contract changes.

Issue forms apply the right `type:` label for you (bug → `type: fix`, feature → `type: feature`).

## Local development

- **Run the stack:** `make up` starts the production-parity stack (see the [README](README.md)).
- **Go control-plane:** `make -C backend/controlplane test`  ·  `go vet ./...`
- **Frontend:**
  - `pnpm --dir frontend test:unit`
  - `pnpm --dir frontend typecheck`
  - `pnpm --dir frontend lint`
- **Python worker:** see `backend/deepagents_runtime/` (`uv` / `pyproject.toml`).

Please keep changes focused, add tests where it makes sense, and never commit secrets or
generated build artifacts.
