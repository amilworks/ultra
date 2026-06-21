---
name: computational-experiment-rigor
description: Rigor protocol for the analysis itself — uncertainty quantification, convergence checks, classification thresholds, literature grounding, and reproducibility records. Use when designing or running a simulation, numerical experiment, or parameter study, when computing estimates (exponents, rates, fits, spectra, regime classifications), or before making any quantitative claim — i.e. while producing the numbers, not while writing them up.
---

# Computational Experiment Rigor

## When to use
Read this skill before finalizing any run that estimates numbers from simulation
or data (exponents, rates, fits, spectra, regime classifications, benchmark
metrics). Apply it proportionally: a quick demo plot needs only items 1 and 6;
a regime classification or quantitative claim needs all of them.

## Protocol

### 1. Recognize canonical systems and ground in known results
If the system under study is a standard one (e.g. driven pendulum, Lorenz,
logistic map, Ising, double pendulum, Duffing, Hodgkin–Huxley), say so by name,
state the commonly reported behavior for the studied parameter values from
literature you are confident about, and compare your results against it.
- When your result disagrees with the commonly reported one, do not silently
  report yours: flag the discrepancy and investigate the usual causes —
  initial conditions and multistability (coexisting attractors), transient
  length, integrator step, finite observation time.
- Name sources at the textbook/author level only when confident (e.g.
  "Baker & Gollub, *Chaotic Dynamics*"). Never fabricate precise citations,
  page numbers, DOIs, or numeric reference values you are unsure of; prefer
  "commonly reported as ..." with an explicit confidence statement.

### 2. Quantify uncertainty on every decision-relevant estimate
A point estimate near a decision boundary is not a result.
- Repeat estimates across ≥3 random seeds and ≥2 observation durations for
  every quantity that drives a classification or conclusion — not just one
  showcase case.
- Report mean ± sample standard deviation (or min–max range) in the table and
  CSV, not bare point values.
- Classification threshold rule: if |estimate| < 3× its spread, the value is
  indistinguishable from zero — label it "marginal / not resolved at this
  precision" and either extend the run or use an independent discriminator.
  Do not assign a definitive class from an unresolved estimate.

### 3. Convergence and validity checks
- Verify at least one representative case per regime class against a halved
  timestep and a doubled duration; state the relative change.
- For Hamiltonian-leaning or long integrations with fixed-step methods, check
  a conserved/slowly-varying quantity (energy drift, phase-space volume) and
  report it.
- State integrator, step size, transient discarded, and observation window in
  the report.

### 4. Triangulate classifications with independent discriminators
Never classify dynamical regimes from a single statistic. Cheap corroborations
you usually already have:
- Strobe/Poincaré point counting → identifies period-n orbits explicitly
  ("3 distinct strobe points → period-3"), distinguishes quasiperiodic
  (closed curve) from chaotic (folded scatter).
- Power spectrum of a long sample → discrete lines vs broadband.
- Sensitivity probe → rerun from a 1e-8-perturbed initial condition and
  report divergence or convergence of trajectories.
State which discriminators agree and which were not run.

### 5. Reproducibility record
The report and code must let a colleague regenerate every number: record
seeds, initial conditions, all parameter values, step sizes, durations, and
the exact commands run. Idempotent script entrypoints beat notebook-style
ad-hoc cells.

### 6. Honest accounting
- Report wall-clock time and compute time separately and label them; never
  imply the whole study took only the inner-loop compute time.
- Failed or dead-end attempts that shaped the method belong in one short
  "what didn't work" line, not hidden.
- Every conclusion gets a confidence level (high/medium/low) tied to the
  evidence above.

## Delegation note
When a verification subtask is delegated to a subagent, give it this skill's
standards explicitly in the task description (seeds, durations, threshold
rule), and reconcile by comparing its numbers to yours with the spread, not
just "consistent".

## Results contract (Intelligence: Pro)
On Pro-intelligence runs the harness enforces a results contract on the final
answer (mean ± spread on every decision-relevant estimate, the decision rule
stated and applied per row, a classification table, projection-aliasing noted,
basin/IC-dependence for borderline cases, and a Limitations paragraph) and sends
the run back for revision when any is missing. The authoritative wording is
injected into the system prompt on those runs; follow it verbatim and meet it on
the first pass to avoid a revision round-trip. The items above (UQ, the 3×-spread
rule, triangulation, limitations) already satisfy it.
