# BisQue Platform

This directory contains the BisQue platform subtree that powers storage, datasets, metadata, modules, and authentication for BisQue Ultra.

For this production cut, treat the repo root as the control point. Start, stop, and verify the platform from the top-level `Makefile` and scripts instead of working inside this folder by hand.

## What lives here

- the BisQue server and supporting services
- the Keycloak-backed auth configuration used by the local stack
- the trimmed production module set:
  - `nphprediction`
  - `Dream3D_UCSB`
  - `CellSegment3DUnet`

## How to run it

From the repository root:

```bash
make platform-up
make verify-platform-smoke
```

That brings up the local BisQue stack and checks the baseline auth flow.

When the stack is healthy, BisQue is available at:

- `http://localhost:8080`

## How module registration works

BisQue discovers modules from the local module service and engine-service configuration that ship with this repo. In normal use, you should not need to enter internal engine URLs by hand.

If you do need to inspect the engine service during local debugging, use the local address:

- `http://localhost:8080/engine_service/`

Do not replace this with lab-specific hostnames in production docs or examples.

## Operational guidance

- Keep this folder trimmed to the modules the release actually supports.
- Prefer local `localhost` examples in docs, scripts, and screenshots.
- If you need broader upstream context, use [README_main.md](README_main.md) as historical background, not as the release runbook.
