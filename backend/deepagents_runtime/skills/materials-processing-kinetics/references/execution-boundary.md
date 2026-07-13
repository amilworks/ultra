# Processing and kinetics execution boundary

## Qualified now

The production wrapper uses `scheil==0.3.0` with `pycalphad==0.11.2`. Classic Scheil-Gulliver
assumes perfect liquid mixing, local solid/liquid equilibrium, and no solid diffusion. The upstream
documentation is <https://scheil.readthedocs.io/en/latest/>. Pycalphad quantities are SI, molar
where applicable, and compositions are mole fractions: <https://pycalphad.org/docs/latest/faq.html>.

Ultra adds database hash/provenance checks, a single-phase-liquid equilibrium preflight, fixed
101325 Pa and one-mole basis, assessment limits, process/result bounds, convergence checks,
monotonic temperature and solid fraction, solid/liquid closure, cumulative solid-phase closure,
and phase-composition closure. `scheil 0.3.0` can append a same-temperature nominally fully-solid
terminal point whose phase increments do not close after the stopping criterion has already been
met. Ultra discards only that exact detectable terminal shape and retains the last residual-liquid
point. It rejects nonconverged partial/final-fill paths and internal length mismatches.

## Qualified isolated Kawin runtime

Kawin provides CALPHAD-coupled mobility, diffusion, and precipitation modeling, including MF/MQ
mobility and DF/DQ diffusivity parameters: <https://kawin.org/docs/03-overview/mobility_modeling/>.
Ultra pins Kawin 0.5.0 with NumPy 2.4.6, pycalphad 0.11.2, and SciPy 1.17.1 in a separate immutable,
network-disabled image; the shared imaging/RareSpot NumPy-1.26 sandbox is unchanged. The public
surface is limited to selected-resource transport coefficients, isothermal single-phase Cartesian
1-D zero-flux diffusion, post-solidification-only 1-D back diffusion, and binary isothermal
spherical homogeneous-bulk-nucleated fixed-bin KWN precipitation with infinite precipitate
diffusion. Each result is request-, database-, version-, limit-, and content-address bound.

Back diffusion also needs an explicit diffusion length scale and source. Neither a Scheil curve nor
a mobility import supplies that choice. Moving-interface or coupled solidification/back-diffusion
remains unsupported, as do nonisothermal/arbitrary-boundary diffusion and broader KWN models.

## External HPC runtime required

Quantitative phase field is not a generic Python plot. Use a qualified engine such as PRISMS-PF or
MOOSE with the actual PDE/free-energy model, coefficients, discretization, boundaries, restart
state, and mesh/time-step convergence. PRISMS-PF documents its parallel finite-element runtime and
prebuilt applications at <https://prisms-center.github.io/phaseField/>. Benchmark the configured
model against a PFHub or analytical reference relevant to the governing equations.
