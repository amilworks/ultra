---
name: materials-processing-kinetics
description: Processing and kinetics workflow for Scheil-Gulliver solidification, selected-resource Kawin transport, post-solidification 1-D back diffusion, binary KWN precipitation, and phase-field requests. Use for casting paths, segregation, kinetic databases, heat-treatment precipitation, and microstructure-evolution questions. Routes each method to its exact qualified runtime and fails closed outside that scope.
---

# Materials Processing and Kinetics

## Route the method before computing

Read this skill for Scheil, solidification, segregation, back diffusion, diffusion mobility,
precipitation, coarsening, phase field, or process-path requests. Also read
`/skills/materials-structure-thermo/SKILL.md` for the CALPHAD database and phase model. Call
`materials_processing_method_support` directly before promising an executable method; do not use
execute or code-runner merely to discover this static support boundary.

The production boundary is intentional:

- `scheil_gulliver`: executable through pinned `scheil==0.3.0` and `pycalphad==0.11.2`.
- `mobility_diffusion`: executable with `materials_transport_coefficients` (single selected phase;
  MF/MQ multicomponent or binary DF/DQ) and `materials_run_diffusion_1d` (isothermal Cartesian 1-D,
  zero flux) in the separately pinned Kawin 0.5 / NumPy-2 image.
- `back_diffusion`: executable with `materials_run_diffusion_1d` only as post-solidification,
  single-phase 1-D diffusion with an explicit physical length-scale source. It is not a coupled
  moving solid/liquid interface or a cooling-path solidification solver.
- `precipitation`: executable with `materials_run_binary_precipitation_kwn` only for binary,
  isothermal, spherical KWN with one matrix/precipitate pair, homogeneous bulk nucleation,
  sourced physical parameters, fixed nonadaptive bins, and infinite precipitate diffusion.
- `phase_field`: requires an external MOOSE/PRISMS-PF-class HPC solver, a governing free-energy
  functional, kinetic coefficients, boundary/initial conditions, and mesh-convergence evidence.

Never relabel Scheil output as back diffusion, finite-rate diffusion, precipitation, or phase
field. Never broaden the three Kawin tools beyond their stated scope. A validated input contract is
not a solver.

## Qualified isolated Kawin workflow

1. Require an explicitly selected, governed `.tdb` whose server binding contains exact SHA-256,
   size, source, license, assessment scope, reference state, and temperature/pressure limits.
2. For coefficients, call `materials_transport_coefficients`; report the reference component,
   volume-fixed frame, MF/MQ versus DF/DQ family, coefficient units, and fixed 101325 Pa. A binary
   DF/DQ assessment returns only the assessed solute tracer; do not invent a solvent coefficient.
3. For single-phase diffusion/back diffusion, call `materials_run_diffusion_1d`; supply the phase,
   isotherm, duration, domain, mesh, linear independent-composition profile, and profile source. For
   back diffusion also supply the measured/modelled physical length-scale source and label the
   result post-solidification-only.
4. For precipitation, call `materials_run_binary_precipitation_kwn`; supply sources for temperature,
   molar volumes, unit-cell atom counts, bulk site density, grain-boundary/interfacial energies, and
   elastic strain-energy density. Report the final discrete number density as `per bin per m3`, not
   as a radius probability density.
5. Retain the content-addressed evidence artifact. Require the exact Kawin/NumPy/pycalphad/SciPy
   versions and database identity, independently verified composition/mass closure, finite ordered
   grids, bounded solver steps, and an immutable no-network runtime image. A single mesh/bin run is
   not a convergence study or experimental validation.

## Qualified Scheil workflow

1. Call `calphad_inspect_database` on an explicitly selected governed TDB or the reviewed embedded
   registry. Retain its content-addressed inspection SHA and inventory. Record resource identity,
   exact SHA-256/size, source, license, assessment scope, reference state, temperature range, and
   pressure range.
2. Call `calphad_run_scheil` with that inspection SHA and the same database identity. Choose every
   physical component (the typed host retains `VA` automatically when the inventory declares it),
   the complete defensible phase set including `LIQUID`, one scalar independent bulk mole-fraction
   composition, a single-phase-liquid start temperature, bounded temperature step, fixed 101325 Pa,
   and a residual-liquid stopping fraction. No path, code, model, or arbitrary option is accepted.
3. Do not call generic `execute`, the upstream package, or the lower-level Python wrapper for a
   product result. The typed host authenticates inspection lineage, owner/release authority, and
   immutable database/runtime identities; the fixed CLI then calls the Ultra wrapper, which
   preflights the all-liquid state, fixes amount, enforces assessment and wall/result/2048-step
   bounds, and verifies phase, composition, and elemental inventory closure. The host independently
   rechecks the entire retained path before returning its bounded summary.
4. Require `result.converged=true`. The wrapper rejects upstream nonconverged final-fill paths and
   discards only the precisely detectable same-temperature terminal point appended after the
   residual-liquid criterion, recording that decision in the evidence.
5. Report the four assumptions verbatim: perfectly mixed liquid, local solid/liquid equilibrium,
   no solid diffusion after formation, and constant 101325 Pa on a one-mole basis. Report database
   validity separately from numerical convergence.
6. Retain the returned `scheil_artifact`; its canonical JSON contains the full
   temperature/phase-fraction and phase-composition paths, exact package versions, limits,
   assumptions, warnings, provenance, and mass-closure evidence. Derivative tables/plots must cite
   that artifact. Never substitute guessed provenance or assessment limits.

## Requests outside the qualified paths

Refuse coupled solidification/back-diffusion, moving interfaces, nonisothermal diffusion, arbitrary
boundary conditions, heterogeneous KWN nucleation, multicomponent/multiphase precipitation,
adaptive bins, finite precipitate diffusion, or an initial particle distribution as unsupported by
these typed tools. For phase field, require the free-energy functional, gradient and kinetic
coefficients, domain/mesh, boundary/initial conditions, solver identity, and a PFHub or analytical
convergence benchmark.

If any required evidence or runtime is absent, return `unsupported` or `unverified` with the exact
missing items. Do not create a toy finite-difference or empirical curve and present it as the
requested research solver.

## Accuracy and regression gates

Before accepting a Scheil result, test a known converged case, a non-liquid start, a nonconverged
eutectic/final-fill case, mass/composition closure, assessment-range escape, nonfinite input,
pressure mismatch, timeout, and result bounds. Run a pilot timing case before a composition sweep.
For Kawin, require selected-resource/hash mismatch tests, dependency/version checks, analytical
diffusion and mesh-refinement controls, KWN solute closure and bin refinement, nonfinite/malformed
result rejection, unsupported-scope refusals, and a full typed-tool/container acceptance run.
Compare any scientific conclusion against equilibrium bounds or independent experimental data;
self-consistency is necessary but not experimental validation.

Write `/outputs/materials_validation.json` through
`ultra_deepagents.materials.validation.assess_scientific_status`. Separate run success from
scientific status and include validator IDs, expected/observed values, units, tolerance rationale,
library versions, and content-addressed evidence. See
[the execution boundary](references/execution-boundary.md) for dependency and model details.
