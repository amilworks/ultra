# Backend Ruff Ratchet

## Goal

Make the backend read like a team-owned system, not just a founder-owned codebase, by turning Ruff into an architectural ratchet instead of a cosmetic formatter.

## Current Contract

- Global backend baseline:
  - `E,F,I,N,W,UP`
- Strict backend ratchet:
  - `B,RUF,SIM,RET`
- Strict scope in this wave:
  - `src/auth`
  - `src/config.py`
  - `src/api/client.py`
  - `src/api/v3.py`
  - `src/api/main.py`
  - `src/tooling/domains`
  - `src/tooling/engine.py`
  - `src/evals/golden_tasks.py`
  - `src/agno_backend/runtime.py`
  - `src/agno_backend/pro_mode.py`
  - `src/training/adapters.py`

## Intentional Exceptions

- `B008` is ignored for `src/api/main.py` and `src/api/v3.py`.
  - Reason: FastAPI route signatures use `Depends(...)`, `Body(...)`, and `File(...)` as framework conventions rather than accidental default-value side effects.
  - Rule: keep the ignore scoped to route modules only. Do not treat it as a blanket exception for ordinary backend modules.

## What This Wave Cleans Up

- Broad `try/except/pass` patterns were converted into explicit suppression paths where cleanup or parsing is intentionally best-effort.
- Nested boolean/control-flow branches were simplified so review intent is easier to scan.
- Closure capture in the threaded tool engine now binds loop values explicitly instead of relying on late binding.
- Mutable class-level constants are marked as `ClassVar` to make ownership and intent explicit.
- User-facing dimension strings were normalized to ASCII-safe output.

## Why This Matters

The goal is not “maximum rule count.” The goal is that new backend slices are easy to review, easy to reason about, and hard to accidentally regress. A strict-clean module boundary is also a forcing function for architecture: if a file is too large or too stateful to satisfy `B,RUF,SIM,RET`, that is usually a signal to split the file rather than to weaken the rules.

## Next Ratchet

The largest remaining strict debt still lives in `src/tools.py` and other legacy monolith surfaces outside this wave. The next pass should:

1. Extract one more backend slice out of `src/tools.py` or `src/api/main.py`.
2. Make that extracted slice strict-clean under `B,RUF,SIM,RET` from day one.
3. Expand the strict scope only after the new slice is clean.

This keeps the lint policy honest: we raise standards by shrinking monoliths and proving new ownership boundaries, not by papering over legacy debt.
