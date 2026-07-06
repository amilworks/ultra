# Ultra Imaging Pipeline Memory

Purpose: persistent operating memory for `ultra_imaging_pipeline`. Read this
before every imaging, viewer, NGFF/OME-Zarr, HDF5, conversion, or tile/slice
serving task.

## Ultra Surface Map

- Python imaging engine:
  `backend/deepagents_runtime/src/ultra_deepagents/imaging/` — engine.py
  (decode), convert.py (derived pyramids), transcode.py, pipelines.py,
  hdf5.py, viewerinfo.py, pool.py (bounded decode concurrency), worker.py,
  special_formats.py, benchmark.py.
- OME-Zarr serving: `backend/deepagents_runtime/src/ultra_deepagents/ngff/` —
  reader.py, render.py, service.py, viewerinfo.py; entrypoint
  `ngff_service.py`.
- Service entrypoints: `image_service.py`, `ngff_service.py`,
  `image_convert_worker.py`; local launch via `scripts/run_image_service.sh`.
- Go proxy, auth, and caching:
  `backend/controlplane/internal/httpapi/imageservice.go`,
  `imageservice_viewer.go`, `imageservice_hdf5.go`, `imageservice_cache.go`,
  with tests including `imageservice_cache_bench_test.go` and
  `hdf5_live_e2e_test.go`.
- Frontend consumers: `frontend/src/components/ScientificViewerPage.tsx`,
  `ResourceBrowser.tsx`, `UploadViewerSheet.tsx`,
  `ResourceThumbnailPreview.tsx`.
- GPU segmentation service: `services/megaseg_service/app.py`.

## Core Contract

- Convert once, read bounded (documented in
  `backend/deepagents_runtime/src/ultra_deepagents/imaging/convert.py`): a non-tiled
  source is converted a single time into a tiled pyramidal OME/BigTIFF via the
  `image.derive_pyramid` NATS job; serving reads bounded regions of the
  derived artifact and must never decode unbounded source data per request.
- viewerinfo responses are a three-language contract: Python computes them,
  Go proxies them, the frontend decides what to render from them. Wrong
  metadata fails silently as a blank or lying viewer.
- The Go layer owns auth for imaging routes: cookie auth plus resource
  ownership before proxying. Worker-token routes are a separate, narrower
  channel.
- Response caching in the Go layer must key by resource identity and request
  parameters; cross-tenant or stale-after-reconversion serving is a blocking
  defect.

## Review Traps

- Unbounded reads hide behind convenience APIs: whole-array `.read()`,
  full-volume numpy materialization, and naive format fallbacks defeat the
  bounded-serving story exactly on the 50GB files that matter.
- Path handling in staging/conversion must reason explicitly about symlinks,
  `..`, and partial-failure cleanup; scientific source data is irreplaceable.
- Interactive paths (tiles, slices, scrubbing, thumbnails) are
  latency-sensitive; changes that add per-request work need percentile
  evidence, not reasoning alone.
- Decode pools bound concurrency for a reason; bypassing the pool or the bulk
  semaphore turns one heavy viewer into a denial of service for everyone.
- Conversion is a NATS job: redelivery and duplicate dispatch must be safe
  (idempotent target paths, no partial artifacts promoted). Pair with
  `ultra_nats_expert` when dispatch semantics change.

## Durable Lessons

- 2026-07-05: Memory initialized from repository evidence. Append durable,
  dated lessons here via parent-approved "Memory updates"; keep entries
  specific (files, symbols, commands, measured numbers) so future sessions
  can re-verify them.
- 2026-07-05: Batch analysis source resolution currently handles top-level
  regular upload files only. OME-Zarr Resources can live as directory bundles
  under `uploads/bundles/{fileID}/{name}` and need an explicit packaging path
  or product-level rejection before claiming "select any Resource" support.
- 2026-07-05: MegaSeg `.zarr.tar.gz` extraction must be both path-safe and
  expected-root scoped. The service wrapper should reject tar members outside
  the archive's root directory, reject links/special files, and stream members
  with incremental member/byte caps before writing.
- 2026-07-05: RareSpot/MegaSeg batch outputs currently surface as Resource
  collection artifacts, not Lens overlay layers over the original. Overlay
  composition in the viewer requires a separate associated-output/viewerinfo
  contract across worker, control plane, and frontend.
