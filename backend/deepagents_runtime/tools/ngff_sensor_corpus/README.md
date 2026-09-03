# ngff_sensor_corpus

Synthetic OME-NGFF (OME-Zarr) **sensor-data corpus** and **stress harness** for the Ultra
image/data service. It is the regression bed for the `ngff-service`
reader/renderer/viewer-info contract.

## Layout

| Module | What it holds |
|---|---|
| `signals.py` | Deterministic, seeded 2-D signal generators (fluorescence blobs, spectral ramps, CT shells, spectrograms, label fields, terrain, …). |
| `specs.py` | `catalog()` — 29 spec-correct `StoreSpec`s across materials/biology/environmental/medical/geophysics/astronomy/other. |
| `writer.py` | `write_store` / `build_corpus` — turn a `StoreSpec` into a spec-valid OME-Zarr store (NGFF 0.4/0.5, Zarr v2/v3, pyramids, omero, `dimension_names`). |
| `scale.py` | `scale_probes()` — declared-huge, cheap-on-disk lazy stores (gigapixel, long-t, deep-z, 256-channel). |
| `adversarial.py` | `build_adversarial` — 23 malformed/attack stores that must fail closed. |
| `stress.py` | The harness: reader + viewer-info + render + live FastAPI HTTP + concurrency + memory bounds → JSON report. |

## Usage

```bash
cd backend/deepagents_runtime
export PYTHONPATH="$PWD/src:$PWD/tools"   # needs the [ngff] extra deps (zarr, numpy, Pillow, fastapi)

# One store:
python -c "from ngff_sensor_corpus.specs import catalog; from ngff_sensor_corpus.writer import write_store; print(write_store(catalog()[0], '/tmp/c'))"

# Full stress run + report:
python -m ngff_sensor_corpus.stress --out /tmp/corpus --report /tmp/report.json
# flags: --no-http, --no-concurrency
```

## Design constraint (important)

The reader treats any axis whose name is not `t/c/z/y/x` as a *custom* axis that must be
singleton. So every modality models its non-spatial dimensions with **canonical axis names**
and carries scientific identity in the **axis units, omero channel labels, dtype, and
coordinate transforms** — never in a custom axis name. A "wavelength"/"band"/"depth"/"m/z"
dimension is therefore `c` or `z` with a domain unit, not an axis literally named `wavelength`.

Generated stores are dev fixtures; do not commit generated `*.ome.zarr` output — regenerate it.
