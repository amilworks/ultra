---
name: materials-crystal-plasticity
description: Convention-explicit crystal-plasticity workflow for FCC/BCC/HCP slip geometry, active crystal-to-sample orientations, Schmid factors, resolved shear stress, declaration-only phase/structure association, CRSS/hardening provenance, and fail-closed CPFE input readiness. Use for slip-system, slip-family, grain-orientation, CRSS, DAMASK, CPFEM/CPFE, twinning/non-Schmid applicability, or constitutive-input requests. The current typed surface analyzes one orientation and validates inputs; it does not independently identify a phase or run a constitutive, finite-element, or spectral solver.
---

# Materials Crystal Plasticity

## Capability boundary

Call `materials_analyze_crystal_slip` or `materials_validate_cpfe_contract` directly. Do not
delegate API discovery to `execute` or the code runner, hand-enter slip tables, or create a
crystal-plasticity subagent. The tools provide two bounded operations:

- canonical DAMASK-3.1.0-transcribed FCC/BCC/HCP slip geometry and, for one orientation, classical
  uniaxial Schmid factors or resolved shear under one symmetric sample-frame Cauchy stress; and
- structural validation of a closed schema-v1 CPFE input contract with caller-declared provenance.

Treat the built-in catalog as geometry, not evidence that a family is active. Composition, phase
state, temperature, strain rate, CRSS, hardening, twinning, and non-Schmid effects govern
constitutive response. The PyPI DAMASK package can independently cross-check crystallographic
kinematics and support pre/post-processing; it is not the DAMASK solver. No constitutive integrator
or FE/spectral backend is bound. A requested CPFE/DAMASK solve is `unsupported`, even when its input
contract is valid; never substitute Schmid factors or a toy stress-strain model.

The first-class analytical tool accepts one phase and at most one orientation per call. The core
library has a bounded batch kernel, but it is not an agent-facing governed tool today. Do not invoke
that internal API through generic code execution. For research-scale EBSD batches, require a
separately governed batch surface that preserves grain/phase identities and enforces output bounds.

A deterministic typed input rejection is terminal for the submitted request. Report the single
error and the missing field; do not repeat the call across seeds, durations, other slip families, or
invented substitute inputs. Exact input validation has no sampling uncertainty, and the general
dynamical-systems replication contract does not apply.

## Required workflow

### 1. Bind phase, state, and frames

Record phase ID, composition, temperature, phase fraction, crystal structure, symmetry, lattice
parameters, and their source. Use only these qualified structure/symmetry pairs:

- FCC: `fcc`, `m-3m`;
- BCC: `bcc`, `m-3m`; and
- HCP: `hcp`, `6/mmm`, with a measured or supplied finite positive `c_over_a`.

Do not infer FCC or BCC from cubic symmetry alone. Partition mixed-phase data and never transfer an
orientation, CRSS, or hardening law between phases without evidence.

For `materials_analyze_crystal_slip`, treat `phase_id` as an opaque caller label. The analytical
tool has no independently bound phase-identification source and must not parse names such as
`alpha`, `gamma`, or `alpha-Ti-hcp` to infer a structure. It evaluates the explicitly selected
structure's geometry, emits
`crystal_plasticity.phase_structure_assignment_bound=skip`, and reports overall
`scientific_status=unverified` even when every geometry check passes. Preserve that separation;
do not upgrade the phase/structure association from naming convention or general alloy knowledge.

Declare the active orientation matrix `R_sc` by `v_sample = R_sc @ v_crystal`. Name the EBSD Euler
or quaternion convention and use its library conversion before submitting the matrix. Do not
transpose it unless the source convention proves an inverse is required. Require a finite,
orthonormal matrix with determinant `+1`.

For resolved shear, provide a finite symmetric Cauchy stress in the sample frame and an explicit
unit from `Pa`, `kPa`, `MPa`, or `GPa`. Never infer stress units from magnitude. For a CPFE contract,
convert stress to SI `Pa`.

### 2. Select systems without assuming activity

Call `materials_analyze_crystal_slip` with the phase ID, structure, selected families, and explicit
HCP `c_over_a`. Use the exact family IDs:

- `fcc-{111}<110>`: 12 octahedral systems;
- `fcc-{110}<110>`: 6 non-octahedral systems;
- `bcc-{110}<111>`: 12 systems;
- `bcc-{112}<111>`: 12 systems;
- `bcc-{123}<111>`: 24 systems;
- `hcp-basal-{0001}<11-20>`: 3 systems;
- `hcp-prismatic-{10-10}<11-20>`: 3 systems;
- `hcp-pyramidal-{10-11}<11-20>`: 6 systems;
- `hcp-pyramidal-{10-11}<11-23>`: 12 first-order c+a systems; and
- `hcp-pyramidal-{11-22}<11-23>`: 6 second-order c+a systems.

Select families from phase-, temperature-, and rate-appropriate evidence, and record its source and
SHA-256. When exact DAMASK 3.1.0 is installed for qualification, compare the complete Schmid-tensor
sets with `cross_validate_slip_systems_with_damask`; version drift is not equivalent evidence.

### 3. Calculate only the quantity permitted by loading

For uniaxial loading, provide `load_axis_sample`. Report the classical absolute factor
`|cos(phi) cos(lambda)|`, bounded by `0.5`. For an arbitrary symmetric stress, provide
`stress_sample` and `stress_unit`; report signed
`tau = d_sample.T @ sigma_sample @ n_sample`. Rank by `abs(tau)` only when the constitutive model is
sign-symmetric. Do not call arbitrary-stress resolved shear a Schmid factor.

Set `hydrostatic_control_stress` in the same unit to run the zero-shear control. Before making a
geometry claim, require:

- unit slip directions and plane normals with zero dot product;
- a proper active crystal-to-sample rotation and a symmetric stress tensor;
- hydrostatic stress resolving to zero shear;
- invariance when crystal orientation and stress are rotated together;
- FCC `<001>` loading on `{111}<110>` giving maximum `1/sqrt(6)`; and
- exact DAMASK-3.1.0 tensor agreement when that reference backend is available.

The largest `abs(tau)` or Schmid factor is a geometric tendency, not proof of observed or active
slip. Confirm activity with compatible CRSS/constitutive parameters and independent microstructure
or deformation evidence.

### 4. Validate CPFE input readiness fail closed

Pass a closed mapping to `materials_validate_cpfe_contract` with schema version `1` and exactly
these top-level objects: `phase`, `frames`, `units`, `orientations`, `slip_families`, `crss`, and
`hardening`.

Use `frames.orientation="crystal_to_sample"`, `frames.stress="sample"`, and
`units={"stress":"Pa","length":"m","time":"s"}`. Provide one finite positive Pa-valued CRSS for
every family. Phase, CRSS, and hardening blocks each need nonblank source ID, supported source type,
citation, and lowercase SHA-256 of the exact source artifact.

Those digest fields are declarations at this tool boundary. The validator checks schema and
lowercase SHA-256 syntax but does not resolve the cited bytes or re-hash them. Consequently, a
structurally valid CPFE contract reports
`crystal_plasticity.source_provenance_bytes_bound=skip` and overall
`scientific_status=unverified` until an independent backend performs byte retrieval and digest
replay. Contract structure can still pass; do not describe that as verified source provenance.

Provide finite hardening scalars with explicit units. Structural completeness does not qualify the
constitutive equations: a future solver adapter must validate model names, equations, interaction
matrices, integration tolerances, and admissible ranges. Set `attempt_execution=true` when asked to
test the boundary; the tool must return contract validity separately from
`execution_supported=false` and must not fabricate stress-strain or convergence output.

### 5. Preserve canonical evidence and scientific status

When the user requests durable outputs, copy `analysis_artifact.canonical_json` exactly to the
requested analysis output and `materials_validation_artifact.canonical_json` exactly to
`/outputs/materials_validation.json`. Do not create unrequested files, reconstruct these objects
from prose, edit verdict fields, or pass an output path to the typed tools.

Include checks for phase/frame identity, proper rotations, stress symmetry and units, slip-family
identity, geometry controls, CRSS/hardening provenance binding, and reference-backend agreement when
available. Analytical geometry checks may pass while the caller-declared phase/structure assignment
and overall result remain unverified. A CPFE contract with declaration-only digests is likewise
unverified, slip activity remains unverified, and CPFE execution remains unsupported. State these
statuses separately.

If the request couples crystal plasticity to fracture, fatigue, creep, oxidation, or corrosion,
also read `/skills/materials-mechanics-degradation/SKILL.md`; do not convert one model's calibrated
parameters into another model's inputs without compatible provenance and domain evidence.
