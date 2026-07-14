---
name: materials-mechanics-degradation
description: Provenance-declaration- and domain-explicit workflow for bounded Mode-I LEFM screening, held-out Paris-law calibration, secondary Norton-Arrhenius creep-rate evaluation, single-temperature linear/parabolic oxidation mass gain, and Faraday-law uniform-corrosion conversion. Use for fracture, fatigue crack growth, creep, oxidation, corrosion, environmental damage, or coating/service degradation. Never present these analytical reducers as component life, ASTM compliance, localized damage, byte-verified provenance, or transferable predictions outside their calibrated domains.
---

# Materials Mechanics and Degradation

## Capability boundary

Call the matching first-class typed tool directly; do not delegate API discovery to `execute` or
the code runner:

- `materials_evaluate_mode_i_lefm`: evaluate `K_I = Y sigma sqrt(pi a)` with a
  caller-declared geometry calibration and caller-cited small-scale-yielding criterion;
- `materials_fit_paris_law`: fit calibration rows, score disjoint held-out interpolation rows, and
  predict only inside the calibrated `Delta K` interval under identical conditions;
- `materials_evaluate_norton_arrhenius_creep`: evaluate a normalized-stress secondary-creep rate
  inside closed stress, temperature, material-state, and environment domains;
- `materials_evaluate_oxidation_mass_gain`: evaluate a linear or parabolic areal mass-gain law
  inside a closed time domain at one exact calibrated isothermal temperature; and
- `materials_convert_uniform_corrosion`: convert electrochemical current density to average
  uniform penetration using explicit equivalent mass, density, efficiency, duration, and sources.

No fracture-growth, finite-element, phase-field, variable-amplitude fatigue-life, creep-damage or
rupture, oxide-diffusion, spallation, or localized-corrosion solver is bound. Prepare inputs and
bounded analyses, identify the missing engine, and return `unsupported` for a requested full solve.
Never substitute an analytical screen or fitted curve for component life or degradation physics.

## Required workflow

### 1. Bind evidence and applicability before calculation

Identify the governing model and require specimen geometry, load/history, temperature,
environment, material state, parameters with units/provenance, initial conditions, and applicable
held-out experimental evidence. Declare every observation, parameter, criterion, and geometry factor
with `EvidenceProvenance`: artifact ID, lowercase SHA-256, locator, and citation. The typed tool
checks this declaration's structure only; it does not fetch the locator or re-hash source bytes.
A caller-supplied 64-hex digest is therefore not byte-verified provenance. Express each
numerical validity domain as a finite inclusive `ClosedInterval` with quantity and unit. Never fill
missing parameters from typical alloy values. Never fabricate placeholder or all-zero hashes, demo
citations/locators, synthetic material states or environments, or guessed validity intervals merely
to satisfy the typed schema. If a required value is absent and the prompt did not explicitly declare
a complete synthetic fixture, stop and list the missing fields instead of calling the tool. Treat a
deterministic typed input rejection as terminal: do not retry with substitute inputs, and do not
create unrequested output files.

Distinguish observations from model outputs. A crack-initiation observation is not fatigue life; a
Paris fit is not valid below threshold or outside its load-ratio/environment domain; a
room-temperature strength value is not a creep law; equilibrium thermodynamics is not oxidation
kinetics; and a polarization curve does not by itself predict long-term penetration.

### 2. Run a bounded Mode-I LEFM screen

Construct `GeometryFactorCalibration` with dimensionless `Y`, the named dimensionless coordinate
`crack_length_over_crack_plus_remaining_ligament`, its evaluated value, and source. The runtime
derives the coordinate from supplied dimensions and rejects a mismatch. Bind the exact crack-length
and nominal-stress definitions; surface depth, half-length, net-section stress, and remote-stress
center-crack definitions are not interchangeable.

Call `materials_evaluate_mode_i_lefm` only with positive SI stress, crack length, ligament,
thickness, yield strength, a declared plane-stress or plane-strain convention, and a positive
caller-supplied minimum-dimension/plastic-zone ratio with a cited criterion. Report every
applicability check. This is not toughness, residual-stress analysis, ASTM E399, or failure
prediction. Do not compare against toughness data without compatible constraint, temperature,
environment, rate, orientation, and provenance.

### 3. Fit Paris data with a real holdout

Create one exact `ParisTestConditions` record with material state, environment, force ratio `R`,
temperature, frequency, waveform, thickness/geometry, applied or effective `Delta K` definition,
and `da/dN` reduction method. Call `materials_fit_paris_law` with at least three calibration rows
and two held-out rows that are unique, disjoint, and together cover all observations. Every held-out
`Delta K` must be inside the calibration interval so its residual tests interpolation.

The fit is unweighted ordinary least squares in natural-log space and does not propagate
measurement uncertainty. Use optional predictions only inside the calibrated interval and under
conditions exactly equal to the calibration record. Never use it for initiation, threshold,
short-crack or closure behavior, overload/sequence effects, terminal instability,
variable-amplitude life, or component failure.

### 4. Evaluate secondary creep only inside calibration

Construct `NortonArrheniusCreepModel` with `A` in `s^-1`, reference stress and stress bounds in
`Pa`, exponent `n`, activation energy in `J/mol`, temperature bounds in `K`, exact material state
and environment, named scalar stress measure, and parameter provenance. Call
`materials_evaluate_norton_arrhenius_creep` only inside every closed domain. The scalar result is an
effective secondary rate, not primary/tertiary creep, rupture, multiaxial flow, damage, remaining
life, or an oxidation-coupled solve.

### 5. Evaluate only supported oxidation regimes

Construct `OxidationKineticsModel` only when measurements support a linear or parabolic areal
mass-gain law. Supply the required `rate_constant_unit` field as exactly `kg*m^-2*s^-1` for a
linear constant or `kg^2*m^-4*s^-1` for a parabolic constant. Missing units, slash-style aliases,
and a unit belonging to the other law must fail closed; never infer the unit from the selected law
or from magnitude. Bind time, temperature, material state, environment, initial mass gain,
exposed-area normalization, and provenance declaration. Because this constant-law schema has no
Arrhenius or other temperature-dependence term, set the temperature interval to the same positive
Kelvin value at both bounds and evaluate at exactly that value. A two-temperature interval must
fail closed; do not reuse the same constant at another temperature.

Do not convert mass gain to oxide thickness or metal loss without reaction stoichiometry,
phase-resolved scale density, morphology, and independent validation. Reject transient, breakaway,
spalling, volatilizing, cyclic, or mixed regimes.

### 6. Convert only average uniform corrosion

Populate `CorrosionPenetrationInputs` with measured current density in `A/m^2`, effective
equivalent mass in `kg/mol electron` for stated dissolution valences, density in `kg/m^3`, current
efficiency in `(0,1]`, duration in `s`, exact material state/environment, current-density area
basis, and separate provenance for all measured or assumed quantities.

Call `materials_convert_uniform_corrosion` and label the result average uniform penetration. It is
not ASTM G102 compliance and cannot predict pits, crevices, galvanic coupling, passivation,
transport limitation, time-varying current, or service life.

### 7. Preserve canonical evidence and scientific status

For every typed tool, copy `analysis_artifact.canonical_json` exactly to the requested analysis
output and `materials_validation_artifact.canonical_json` exactly to
`/outputs/materials_validation.json`. Do not reconstruct objects from prose, edit verdict fields,
or pass output paths to the tools.

Include validator IDs for provenance binding, units, domain containment, geometry/applicability, exact
calibration-versus-holdout identity, finite results, and independent comparisons. A bounded
calculation can pass its internal numerical check while the overall scientific status remains
`unverified`: the current typed surface emits a required
`materials.bounded_tool.provenance_bytes_bound=skip` check for caller-declared digests. Promote that
check only after an independent backend resolves the exact bytes, replays SHA-256, and confirms the
match. State run, reducer-validation, provenance-binding, and requested-conclusion status separately.

When a backend is absent, provide an input-readiness audit, dimensional checks, sensitivity plan,
and experiment/validation design. Mark the requested prediction `unsupported` instead of
fabricating a number.
