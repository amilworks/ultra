---
name: materials-structure-thermo
description: Crystal-structure, point-defect, and thermodynamics workflow for materials science. Use for CIF/POSCAR/structure parsing, composition and occupancy checks, spglib/pymatgen space-group and symmetry analysis, ordered-vs-disordered phase identity, conventional/primitive cells, vacancy/interstitial/substitutional defect construction and geometry with pymatgen-analysis-defects, CALPHAD equilibrium and phase diagrams with an authorized Thermo-Calc `.tdb` or ChemSage `.dat` database, energy-above-hull reasoning with compatible energies, and matminer composition featurization. Do not use it to claim DFT, MD, phonon, defect formation energies, or production atomistic simulations without the required engine and energy provenance.
---

# Materials Structure and Thermodynamics

## When to use

Read this skill for CIF or atomistic-structure files, space-group/symmetry questions, ordered phase
identity, CALPHAD equilibrium/phase fractions/phase diagrams, compatible-energy convex hulls, and
materials-informatics composition features. It also covers structural point-defect objects,
symmetry multiplicities, and defect supercell construction with `pymatgen-analysis-defects`. Use
`/skills/materials-characterization/SKILL.md` for
XRD or spectroscopy and `/skills/computational-materials/SKILL.md` for EBSD and 3D microstructure.

The supported lightweight stack is `pymatgen`, `spglib`, `ase`, `pycalphad`, and `matminer` in the
default Python interpreter. Record versions with `importlib.metadata.version()` rather than
assuming a top-level `__version__` exists.

## Hard capability boundary

There is **no production atomistic engine** in the release sandbox: no VASP, Quantum ESPRESSO,
CP2K, LAMMPS, GPAW, xTB, phonopy, MPI workflow, or configured scheduler. ASE is a structure and
calculator interface, not evidence that one of those engines ran. It is acceptable to build,
convert, inspect, or visualize a structure and to exercise an explicitly named toy calculator for
a smoke test. It is not acceptable to label a generated input deck, empirical placeholder, or ASE
object as a DFT/MD/phonon result. Mark such requested capabilities `unsupported` unless a separately
configured execution service is actually present and identified.

The installed defects library can construct and inspect vacancy, interstitial, substitutional,
and related defect structures. That does not supply bulk/defect total energies, chemical
potentials, dielectric tensors, charge corrections, band edges, or Fermi-level conditions.
Therefore a geometrical defect object is supported, while a first-principles defect formation
energy remains `unsupported` unless every required energy and correction input comes from a named,
compatible external calculation or database with provenance.

## Required workflow

### 1. Establish input provenance and chemical meaning

- Stage the original file, retain it unchanged, and record path/artifact ID plus SHA-256.
- Parse with `pymatgen` or ASE; do not regex-parse CIF loops or infer occupancy by eye.
- Report reduced/full composition, site count, lattice parameters/angles, coordinate convention,
  occupancies, oxidation states if supplied, magnetic moments if supplied, and warnings emitted by
  the parser. Missing metadata stays missing.
- Preserve partial occupancy, disorder, and ordering. Ordered Ni3Al L1_2 and a disordered fcc
  representation are not interchangeable even when their average lattice looks similar.
- When writing a converted structure, read it back and compare composition, lattice, fractional
  coordinates modulo periodicity, and occupancies before calling the conversion successful.

### 2. Symmetry and phase identity

Use `pymatgen.symmetry.analyzer.SpacegroupAnalyzer`, which delegates to spglib. Record `symprec`,
`angle_tolerance`, cell setting, and whether the result refers to the input, primitive, or
conventional cell. Sweep a declared tolerance range appropriate to coordinate precision rather
than presenting one default as exact. Report all distinct assignments and mark the space group
unverified if chemically plausible tolerances change the conclusion.

Run the recipe controls in
[structure-thermo-recipes.md](references/structure-thermo-recipes.md): ordered Ni3Al L1_2 must
resolve to 221 (Pm-3m), while an elemental fcc structure resolves to 225 (Fm-3m). These controls
test the analysis path; they do not prove the user's structure is either phase.

Never identify a material phase from space group or lattice parameter alone. Combine composition,
occupancy/order, symmetry, and—when available—independent diffraction/chemistry evidence. State
candidate phase, evidence, alternatives, and the confidence-limiting ambiguity.

### 3. Point-defect structures and geometry

Use the installed `pymatgen-analysis-defects` APIs rather than deleting or inserting sites by
untracked list position. Record the pristine structure hash, defect class (vacancy, interstitial,
substitution/antisite, or complex), host and substituent species, fractional/Cartesian defect
coordinates, symmetry-equivalent site multiplicity, nominal charge state, supercell transform,
and minimum periodic image separation. Preserve the mapping between bulk and defect sites.

Validate every generated defect structure by checking the expected composition delta, lattice/
supercell relationship, site count, minimum interatomic distance, and round-trip serialization.
For interstitial searches, state how candidate sites were generated and deduplicated under crystal
symmetry. A geometrically valid candidate is not automatically a stable defect configuration.

Do not report a defect formation energy from geometry alone. Such a result requires compatible
bulk and defect total energies plus declared chemical potentials, charge state, Fermi-level and
band-edge references, electrostatic/image-charge correction method, dielectric data, and finite-
size convergence. If any required term is absent, enumerate the missing inputs and mark the energy
claim `unsupported` rather than substituting a toy calculator or zero correction.

Start from the version-checked generator and supercell patterns in
[defect-recipes.md](references/defect-recipes.md). The NaCl controls there validate API semantics;
they are not benchmark answers or a substitute for validation on the user's host structure.

### 4. CALPHAD and phase stability

`pycalphad` requires an appropriate thermodynamic database. Ultra ships one open assessed reference,
`nist-al-co-w-wang-2017`, under `$ULTRA_CALPHAD_DATABASE_ROOT` (production default
`/opt/ultra-calphad`). It is the CC0 NIST Al-Co-W reassessment associated with
DOI `10.1016/j.calphad.2017.09.007`; use it only for covered Al-Co-W questions and the declared
assessment domain. It is not a general alloy database. For other chemistries require a staged,
user-supplied Thermo-Calc `.tdb` or ChemSage `.dat` database with explicit license/use
authorization; reject `.db`, which is not a registered pycalphad 0.11.2 input format. Record the
exact database ID/name and parser format, source, version/date, resource or artifact ID, stored and
source SHA-256 when they differ, components, candidate phases, assessment temperature and pressure
bounds, reference state, and any phase exclusions. Record the pressure interval under the canonical
`assessment_pressure_limits_Pa` key as exactly two finite pascal values `[minimum, maximum]` within
`[1e-9, 1e12]`, with `minimum <= maximum`; missing or malformed bounds block both inspection and
equilibrium/Scheil execution. Files under
`pycalphad/tests`, package test-data directories, examples, tutorials, or benchmark fixtures are
**test fixtures**, not scientific databases; never use them for scientific conclusions.
For production agent work, call the first-class `calphad_inspect_database` tool first. Set exactly
one source: a tenant-authorized selected/catalog `resource_id`, or the reviewed embedded
`database_id`. Do not stage a path manually and do not run raw pycalphad through generic
`execute`. The typed tool binds a user resource to its server-authored catalog SHA-256/size and
owner-declared source, license/use authorization, assessment scope, reference state, temperature
limits, and `assessment_pressure_limits_Pa`; it then rehashes the staged bytes inside the immutable networkless
sandbox. Require every requested pressure to lie inside the exact declared assessment interval.
The embedded NIST Al-Co-W reference is fixed to 101325 Pa because no broader pressure assessment is
declared.
Missing provenance, symlinks, copied test fixtures, hash drift, parse failures, or timeouts fail
closed. A successful resource inspection is sent to the control plane as bounded compressed bytes.
The server rehashes and parses the complete pycalphad manifest, rechecks the live catalog binding,
active run lease, server-stamped immutable runtime image, and pinned pycalphad version, retains the
exact raw evidence blob, and appends `input_validated` to the immutable resource revision. This is a
technical input/evidence status, not independent validation of the thermodynamic assessment.
The revision also retains the exact bounded database bytes content-addressed by SHA-256. Owners can
replay those immutable bytes even after the upload/catalog object is deleted. The revision must
also retain the explicit parser format (`tdb` or `dat`) and a bounded, server-normalized snapshot of
the owner-declared database ID, source, license ID, assessment scope, reference state, temperature
limits, and `assessment_pressure_limits_Pa`. Reconstruct replay as `<sha256>.<format>` and pass the
same explicit format to pycalphad; never infer it from retained bytes or default it to TDB. Changed
declarations require a new resource/revision. Missing or corrupt input, format, or declaration
snapshot makes ledger reads and later validation non-promotable.

Choose an explicit component and phase subset from the returned inventory, then call
`calphad_run_equilibrium` with the inspection artifact SHA-256, the same database source, and typed
finite conditions. The equilibrium tool accepts no caller path, code, solver model, parameter
override, or arbitrary pycalphad option. It writes full v2 evidence under
`/outputs/calphad/equilibrium/<sha256>.json`, including phase fractions, per-vertex phase
compositions, chemical potentials, molar Gibbs energy, units, provenance, and warnings, and returns
only a bounded summary. The typed surface supports bounded equilibrium points and Cartesian
condition/composition grids; it is not an adaptive phase-boundary tracer or a full arbitrary
phase-diagram engine. `inspect_calphad_input(...)` and `run_calphad_equilibrium(...)` are
sandbox-internal/replay APIs used by that fixed tool surface and by deterministic validators; they
are not the production agent invocation path. Resolve an embedded database by its reviewed manifest
`database_id`. The internal runtime rejects byte-identical copies of bundled fixtures, not only
paths that still live under `pycalphad/tests`.

For classic Scheil--Gulliver solidification, read
`/skills/materials-processing-kinetics/SKILL.md` and call the governed `calphad_run_scheil` tool
after inspection. Pass the same database identity and inspection SHA, a complete defensible phase
selection including `LIQUID`, every physical component, one scalar independent bulk composition,
a single-phase-liquid start temperature, a bounded temperature step, fixed 101325 Pa, and the
residual-liquid stopping fraction. The host retains inspected `VA` automatically. The surface
accepts no path, caller code, model override, or arbitrary solver option. It writes complete
content-addressed evidence under `/outputs/calphad/scheil/<sha256>.json`; the sandbox, host, and
control plane independently reconstruct every retained elemental inventory from liquid plus solid
increments before accepting it. This is the classic no-solid-diffusion idealization, not back
diffusion, finite-rate transport, precipitation, phase field, or evidence that the selected
thermodynamic assessment is experimentally valid for the alloy.

Use the runtime's canonical dependent-component convention: exclude `VA`, sort the physical
components, and make the first component dependent. Supply mole fractions for the remaining
physical components. Thus the embedded Al-Co-W reference uses `CO` and `W` as the independent
variables and derives `AL`; the 1173 K checkpoint at `X(AL)=0.675`, `X(CO)=0.260`, and
`X(W)=0.065` must be submitted as `CO=0.260`, `W=0.065`. The typed runtime may transparently
reframe an equivalent singleton or binary-axis request before hashing and solving. Reject a
multiaxis Cartesian grid when that reframing would couple two varying axes; do not let a caller's
choice of dependent component select a different equilibrium basin.

When the authenticated inspection inventory lists `VA`, include it in the selected components even
though it has no bulk mole-fraction axis. The typed host automatically retains inspected `VA`, and
the sandbox rejects a direct typed request that omits it. Omitting `VA` can remove substitutional
phase models and yield a higher-Gibbs assemblage that still satisfies numerical closure.

ChemSage MQMQA phases can expose cation/anion constituent groups that do not map one-to-one to TDB
site-ratio arrays. Preserve and report those groups, retain the runtime warning, and do not invent
site ratios to make the inventory look TDB-shaped.

For a tenant resource, equilibrium or Scheil persistence requires the exact retained inspection
artifact from the same revision, run, and immutable runtime image. For equilibrium, the control
plane independently validates the complete v2 point inventory, grid and composition closure, phase
vertices/fractions, bulk reconstruction residuals, units, chemical potentials, Gibbs-Euler
relation, database-manifest hash, and canonical evidence hash before appending
`equilibrium_completed`. For Scheil it independently validates the fixed solver contract,
monotonic path, phase increments/cumulative fractions, complete phase compositions, pointwise
elemental inventory reconstruction, units, limits, lineage, and canonical evidence hash before
appending `scheil_completed`. Missing or legacy-unretained evidence is non-promotable. Treat
`input_validated`, `equilibrium_completed`, `scheil_completed`, and the ledger's technical
`promotable` flag as governance evidence only; they never change scientific status to `verified`
without the task-specific validation record and applicable independent evidence.
An owner-selected resource uses the append-only validation ledger. A read-granted/shared resource
is deliberately `read_only_unpromoted`: its typed calculation and content-addressed artifact may be
used in the current analysis, but it cannot mutate or promote the owner's database ledger.

Do not concatenate independently assessed TDBs, translate their Gibbs-energy functions into ad hoc
Postgres rows, or combine phases solely because the element sets overlap. Database merging requires
expert review of reference states, phase names/models, endmembers, parameter provenance, magnetic/
ordering conventions, and duplicate functions. Modified TDB bytes must be uploaded under a new
resource ID and hash. Link a replacement resource to its prior immutable revision through the
server-managed parent-revision field; never mutate the old revision. The append-only PostgreSQL
ledger retains technical inspection/equilibrium/Scheil events and their exact bytes, but per-run
content-addressed evidence must not be relabeled as independent assessment validation or a universal
global catalog claim.

For every equilibrium or diagram record:

- composition basis and independent composition variables;
- temperature and pressure ranges/units, total amount, and grid/adaptive strategy;
- components including vacancies where required, candidate phases, suspended phases, and
  convergence warnings;
- phase-fraction bounds and closure over equilibrium vertices at every reported condition; and
- database domain limits, extrapolation, and comparison with independent evidence when available.

Use the bounded CALPHAD runtime rather than an unbounded all-phase call. It must cap TDB bytes,
element/phase/parameter counts, condition-grid points, wall time, and serialized result size. Supply
finite `T` in kelvin, `P` in pascal, `N` in mole, and exactly the independent composition variables;
validate composition closure before solving. Report only finite equilibrium vertices with stable
phase names, phase-amount fractions, per-vertex phase compositions, chemical potentials, molar
Gibbs energy, phase and bulk-composition closure, Gibbs-Euler consistency, solver warnings, and the
exact database/runtime versions. A timeout, convergence failure, empty vertex set, non-finite
result, mass-balance/Euler inconsistency, or fraction/composition-closure failure is
`unverified`/`failed`, never a blank phase diagram presented as success.

Numerical closure is necessary but not independent scientific validation. A converged local
solution can satisfy phase-fraction, mass-balance, and Gibbs-Euler checks while missing the lower
global minimum. For the embedded NIST Al-Co-W database, treat the documented 1173 K Al-Co-Al4W-
Al5Co2 checkpoint in
[structure-thermo-recipes.md](references/structure-thermo-recipes.md) as a mandatory regression
when that exact state is requested. For other states, compare against applicable published phase
fields, experimental data, or a predeclared reviewed checkpoint when available. Without such
independent evidence, report the calculation as internally consistent but scientifically
`unverified`; never upgrade it to `verified` from self-consistency alone.

Equilibrium and metastable calculations are different contracts. If phases are suspended or a
metastable phase set is selected, label that result metastable and list the excluded phases. Never
present a restricted-phase result as the global equilibrium. For the embedded Al-Co-W assessment,
preserve the publication's caveat about gamma-prime/L12 metastability rather than inferring phase
stability from the mere presence of an `L12_FCC` model.

If the appropriate database is absent, report that CALPHAD was not run and call
`assess_scientific_status(..., capability_supported=False)` so the canonical module computes
`unsupported` and its exact reason fields; never assign verdict fields manually, substitute an ad
hoc stability score, or rename a toy database.

For pymatgen convex-hull/energy-above-hull work, require compatible energies from a named common
reference and correction scheme. Record energy units and normalization (per atom vs per formula
unit), included chemical system, correction provenance, and entry IDs/hashes. A hull made from
mixed functionals or arbitrary isolated totals is not interpretable.

### 5. Materials informatics

Use named `matminer` featurizers and record preset/version/labels. The Magpie `ElementProperty`
preset yields a structured set of elemental-property statistics; do not replace it with a single
hand-averaged vector. Split train/validation/test data by a scientifically meaningful grouping
when composition families or source studies can leak. A descriptor calculation is not a property
prediction unless a trained, validated model and its applicability domain are supplied.

### 6. Scientific validation record

Every scientific claim must produce `/outputs/materials_validation.json`, distinct from run
success. Use `ultra_deepagents.materials.validation.ValidationCheck`, `EvidenceArtifact`,
`assess_scientific_status`, and `canonical_record_json`. The production image asserts this import
at build time. If it fails, report release-infrastructure failure and do not hand-author a verdict.

Build CALPHAD validation checks directly from the immutable typed equilibrium artifact and
predeclared independent expectations. Do not delegate a successful typed result to a generic code
runner to manufacture a second JSON file and treat that self-authored file as independent evidence.
Never instantiate `ScientificAssessment`, call `canonical_record_json` with keyword fields, or edit
top-level verdict fields after serialization. Use the exact decision path below and parse the final
bytes before reporting a verdict:

```python
import json
from pathlib import Path

from ultra_deepagents.materials.validation import (
    assess_scientific_status,
    canonical_record_json,
    parse_assessment_record,
)

required_ids = tuple(sorted(check.validator_id for check in checks if check.required))
assessment = assess_scientific_status(
    run_status="succeeded",
    checks=checks,
    required_validator_ids=required_ids,
    capability_supported=True,
)
encoded = canonical_record_json(assessment)
parse_assessment_record(json.loads(encoded))
Path("/outputs/materials_validation.json").write_text(encoded + "\n", encoding="utf-8")
```

If the parse fails, the validation artifact is invalid and the scientific claim is `unverified`
regardless of any prose or manually assigned field.

Each check contains `validator_id`, `outcome` (`pass`, `fail`, or `skip`), `observed`, `expected`,
`units`, `tolerance_rationale`, `required`, `critical`, `library_versions`, `evidence` (a list of
objects containing `name`, a 64-hex `sha256`, and `path` or `artifact_id`, with optional
`size_bytes`), and `message`. The top level contains:

```json
{
  "schema_version": "1",
  "run_status": "succeeded",
  "scientific_status": "verified",
  "verified": true,
  "silent_success": false,
  "capability_supported": true,
  "required_validator_ids": ["materials.structure.symmetry_tolerance.v1"],
  "missing_validator_ids": [],
  "critical_failures": [],
  "contradiction_failures": [],
  "reasons": [],
  "checks": [
    {
      "validator_id": "materials.structure.symmetry_tolerance.v1",
      "outcome": "pass",
      "observed": {"space_group_numbers": [221, 221, 221]},
      "expected": {"stable_assignment": true},
      "units": "dimensionless space-group number",
      "tolerance_rationale": "Tolerance sweep spans the CIF coordinate precision",
      "required": true,
      "critical": true,
      "library_versions": {"pymatgen": "<version>", "spglib": "<version>"},
      "evidence": [
        {"name": "symmetry.json", "sha256": "0000000000000000000000000000000000000000000000000000000000000000", "path": "/outputs/symmetry.json"}
      ],
      "message": "Assignment remained stable across the declared sweep"
    }
  ]
}
```

The all-zero digest is a documentation placeholder only. Replace it with the final artifact's
actual lowercase SHA-256; trace inspection matches the durable `/outputs` artifact metadata and
rejects a mismatch. Keep the complete record below the trace inspector's 1,000,000-byte limit.

Declare required validator IDs before assessment. Missing/skipped required checks and incomplete
passing evidence fail closed to `unverified`; invariant failures become `failed`; unavailable
engines/databases become `unsupported` only through
`assess_scientific_status(..., capability_supported=False)`. Report `silent_success` when orchestration succeeded but
scientific checks failed. Hash final evidence artifacts after writing them.

### 7. Outputs and report

Write the original-input manifest/hash, normalized structure (when requested), symmetry sweep,
CALPHAD conditions/results, phase/entry tables, code, package versions, validator evidence, and
`materials_validation.json` under `/outputs`. In the user-facing answer state run status and
scientific status separately, list failed/skipped/missing required checks, and include limitations.
