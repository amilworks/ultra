---
name: computational-experiment-rigor
description: Domain-specific rigor protocol for computational nonlinear-dynamical-system regime studies — Lyapunov estimation, convergence, IC and basin checks, independent discriminators, literature grounding, and reproducibility. Use only when a request explicitly studies dynamical regimes in systems such as logistic, Lorenz, Duffing, or pendulum models; generic quantitative, simulation, statistics, spectrum, metric, and classification work routes elsewhere.
---

# Computational Experiment Rigor

## When to use
Read this skill only when the request explicitly asks for a computational study
of nonlinear or dynamical-system regimes, with a strong anchor such as a
Lyapunov exponent, bifurcation, Poincare section, phase portrait, basin of
attraction, return map, or a named canonical system. Generic simulations,
estimates, classifications, statistics, spectra, scaling exponents, and
scientific-image analyses do not use this protocol. Apply it proportionally: a
quick canonical-system demo needs only items 1 and 6; a regime classification
needs all of them.

## Protocol

### 1. Recognize canonical systems and ground in known results
If the system under study is a standard one (e.g. driven pendulum, Lorenz,
logistic map, double pendulum, Duffing), say so by name,
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

## Dynamical-systems results contract (Intelligence: Pro)
On Pro-intelligence runs whose request is positively identified as a
computational dynamical-regime study, the harness enforces a results contract on
the final answer (mean ± spread on every decision-relevant estimate, the
decision rule stated and applied per row, a classification table,
projection-aliasing noted, basin/IC-dependence for borderline cases, and a
Limitations paragraph) and sends the run back for revision when any is missing.
The authoritative wording is injected only on those dynamics runs; follow it
verbatim and meet it on the first pass to avoid a revision round-trip. The items
above (UQ, the 3×-spread rule, triangulation, limitations) already satisfy it.
