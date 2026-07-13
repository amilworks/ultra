# Domain-correctness invariants

Executable "canonical-output invariant" checks for the scientific domain skills
(materials, biology, ecology). Each test encodes a property that **must hold if
the analysis is done with the correct field-standard library** and would **fail
for the plausible-but-wrong hand-rolled shortcut** the skills warn against — e.g.
"the cubic IPF-Z colour key maps 001→red, 101→green, 111→blue" (a hand-rolled
`R=x,G=y,B=z` mapping paints it blue and fails this test).

## What these guard
1. **The recipes we ship are correct** — every recipe in a skill's
   `references/*-recipe.md` has a matching invariant here, so a wrong recipe (or a
   library API change) is caught.
2. **A regression tripwire** for the domain libraries themselves.

They do **not** run the agent — the runtime backstop against a fresh hallucination
is the `self-check` block embedded in each recipe plus the machine-readable
`/outputs/materials_validation.json` record. These tests validate the reference
the agent copies from. The independent MatTools gate evaluates the complete Ultra
runtime separately.

## Running
These need the scientific stack (orix, scanpy, geopandas, …) which lives in the
**sandbox image**, not the lean worker. In lean tests an absent optional library
skips. Promotion mode fails closed on every skip:

```
# inside the code-execution sandbox image (bisque-ultra-codeexec:py311):
ULTRA_FAIL_ON_DOMAIN_SKIP=1 python -m pytest \
  backend/deepagents_runtime/tests/domain_correctness/test_materials_invariants.py -v
```

The `make materials-domain-test` target performs that full-image run with the
network disabled. A skipped materials invariant makes the target fail.

The materials release set currently contains 13 required validators. The real
CALPHAD validator resolves the embedded, content-addressed NIST Al-Co-W catalog
through Ultra's bounded runtime and checks seven unary phase-transition
checkpoints plus two phase-field points read from Wang et al. (2017): the Al4W
single-phase field in Fig. 8(b) at 1000 K and X(W)=0.20, and the
AlCo-Al4W-Al5Co2 three-phase triangle in Fig. 12(a) at 1173 K and
X(Al,Co,W)=(0.675,0.260,0.065). Every calculation requests all 18 declared
phases. The validator also checks database/source hashes, CC0 provenance,
declared assessment bounds, finite GM/MU/X/NP values, phase and bulk-composition
closure, and vertex-weighted mass balance. Pure-element FCC and BCC checkpoints
accept only their declared ordered/disordered phase-family aliases; this does
not promote the catalog's metastable L12 model to an equilibrium claim for
ternary alloys.

The copyrighted figure bytes are not bundled. The evidence record binds the
exact publisher assets by URL, byte size, and SHA-256: Fig. 8(b) is 857,601
bytes / `ce4ba92e8861bd56cc37ef5b997477780bde1e9edd784797a555f26c063793ad`;
Fig. 12(a) is 822,036 bytes /
`2cbd5bda9493d138442133a4796b4dbd944cdf106dcda197341baa9f935e59e6`.

The two published phase-field points are a cross-engine reproduction check:
the paper reports Thermo-Calc calculations, while Ultra executes the released
TDB with pycalphad 0.11.2. Because the figures and TDB belong to the same
assessment, this is not independent experimental validation and does not
qualify a numeric phase boundary or tie-line composition. Such a benchmark
requires an openly licensed experimental table with explicit equilibrium phase
compositions and uncertainty.
