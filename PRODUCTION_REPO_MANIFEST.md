# BisQue Ultra Production Repo Manifest

This repo cut contains the live BisQue Ultra stack needed to launch:

- FastAPI backend under `src/`
- runtime eval/support helpers under `src/evals/`
- React frontend under `frontend/`
- BisQue platform + Keycloak auth assets under `platform/bisque/`
  Module set is intentionally trimmed to `nphprediction`, `Dream3D_UCSB`, and `CellSegment3DUnet`.
- root startup orchestration via `Makefile`
- local restart / smoke-check scripts under `scripts/`

Intentionally excluded from this cut:

- iOS app
- tests and Playwright suites
- docs, benchmarks, and most evaluation assets
- retired legacy app scaffolding
- generated runtime state (`data/`, `runs/`, `downloads/`, `outputs/`)
- large local model caches and training artifacts

External/runtime-provisioned assets you may still want locally:

- MedSAM2 checkpoints under `data/models/medsam2/checkpoints/`
- SAM3 checkpoints under `data/models/sam3/`
- YOLO / prairie weights such as `RareSpotWeights.pt` and `yolo26x.pt`
- any production secrets copied into `.env`

Validation target for this cut:

- `make verify-platform-smoke`
- `make verify-integration`
- `pnpm --dir frontend build`
- `uv run python -m py_compile src/api/main.py src/agno_backend/runtime.py src/tools.py`
- optional local restart: `./scripts/restart_dev.sh restart`
