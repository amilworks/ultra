---
name: computational-materials
description: Domain toolkit and rigor for computational materials science — superalloy microstructure, EBSD/crystallographic orientation analysis, 3D serial-section/tomography (TriBeam) segmentation and quantification (grains, phases, precipitates, porosity), CALPHAD phase stability, and materials-informatics featurization. Use when a task involves EBSD orientation/misorientation/texture, 3D microstructure segmentation from serial-section or tomography volumes, grain/phase/precipitate/porosity quantification, crystal structure or space-group analysis, phase-diagram / phase-fraction / phase-stability (gamma-prime, TCP) prediction, or ML over alloy composition/process space. Read it before writing code that would otherwise reach for a hand-rolled surrogate (a home-made quaternion misorientation, a hand-rolled IPF colour mapping, a bare intensity threshold for grains, an averaged-atomic-property feature vector, or an ad-hoc phase-stability heuristic).
---

# Computational Materials

## When to use
Read this before any microstructure, crystallographic-orientation, phase-thermodynamics, or
materials-informatics analysis. It is biased toward the Ni/Co-base superalloy, TriBeam-tomography,
and ICME research line (single-crystal turbine-blade alloys, γ/γ′, TCP phases, rafting,
creep/fatigue, thermal-barrier coatings, high-throughput/ML alloy + additive-manufacturing
process design). The point is to use the field's actual named tools and validation methods
instead of re-deriving weaker surrogates from numpy, and to prove any microstructural quantity
or phase you report is real and stable rather than an artifact of a parameter choice. This is
domain tooling; the rigor protocol in `/skills/computational-experiment-rigor/SKILL.md` still
applies on top of it.

## Environment — one interpreter, no isolation
The entire materials stack lives in the default `python` (numpy 1.26.4) — no isolated env (the
contrast with computational-biology). Available: `orix`, `kikuchipy` (+hyperspy/rosettasciio),
`diffsims`, `defdap`, `porespy` for characterization; `pymatgen` (+`spglib`), `pycalphad`,
`matminer`, `ase` for crystallography/thermodynamics/informatics; plus baked scikit-image,
scipy, scikit-learn, networkx, pandas, matplotlib, dask, zarr, h5py, xarray, and the
SimpleITK/nibabel 3D-imaging stack. State the interpreter and versions. Two caveats: **`pymatgen`
and `defdap` expose no top-level `__version__`** (record their versions from pip metadata), and
**`pycalphad` needs a user-supplied TDB thermodynamic database** — none is bundled, so name the
database you used and never fabricate one.

## Protocol
Apply proportionally: a quick orientation-map render needs only §1 and §6; a reported grain-size
distribution, texture strength, or predicted phase fraction that drives a conclusion needs all.

### 1. Use the field-standard named tool, name the method, pin its parameters
Do not substitute a generic surrogate when a canonical materials tool exists.
**Before writing analysis code, copy the vetted recipe** — correct call + a runnable self-check +
the named anti-pattern — from **[references/materials-recipes.md](references/materials-recipes.md)**
(misorientation vs Mackenzie, grain segmentation + stereology, CALPHAD phase fractions, space-group ID,
Magpie featurization, porosity) and **[references/ebsd-ipf-recipe.md](references/ebsd-ipf-recipe.md)**
(IPF colouring). Hand-rolling these is how wrong-but-plausible results ship; run the self-check first.
- **EBSD / orientation:** use **orix** for orientations, symmetry, misorientation, and
  IPF/pole-figure/ODF plotting — never a hand-written quaternion, Euler misorientation, **or IPF
  RGB mapping**. The naive "sort the vector so x≤y≤z, then R=x,G=y,B=z" paints the whole triangle
  blue/cyan — it is WRONG. The correct cubic IPF-Z key is **001=red, 101=green, 111=blue**, which
  `orix.plot.IPFColorKeyTSL` renders (and colours orientations) for you. Copy the vetted recipe —
  colour key, per-pixel map, and a self-check — from
  **[references/ebsd-ipf-recipe.md](references/ebsd-ipf-recipe.md)**. This holds even for a quick
  *illustrative/teaching* figure: a wrong colour key shown to a user is worse than none. State the
  crystal symmetry (point group / Laue class), which sample direction the IPF is for (ND=IPF-Z /
  RD=IPF-X / TD=IPF-Y), and the orientation representation. Use **kikuchipy** from raw Kikuchi
  patterns (state the indexing — Hough or dictionary — and its parameters); **diffsims** to build
  the simulated-diffraction dictionary.
- **3D microstructure segmentation (TriBeam volumes):** use **scikit-image** named methods —
  marker-controlled `watershed` for grains, `label`+`regionprops_table` for per-grain/phase/
  precipitate/pore metrics, `morphology` for cleanup — not a bare intensity threshold. State the
  full pipeline (denoise → threshold/method → markers → watershed → connectivity), every
  parameter, and the physical voxel size (TriBeam z-slice spacing and in-plane pixel size). Use
  **porespy** for named porosity/pore-size/tortuosity/two-point-correlation metrics.
- **Crystal structure / phase symmetry:** **pymatgen** (`SpacegroupAnalyzer`, via **spglib**) to
  identify space group and phase — γ FCC (Fm-3m), γ′ L1₂ (Pm-3m), TCP σ/μ/Laves — tolerance
  stated; **ase** for structure I/O.
- **Phase stability / thermodynamics:** **pycalphad** (`equilibrium`/`calculate`) against a NAMED
  TDB for phase fractions and phase diagrams; pymatgen phase-diagram / energy-above-hull for
  DFT-energy stability. Never invent an ad-hoc "stability score".
- **Materials informatics:** **matminer** featurizers (name them — Magpie ElementProperty,
  oxidation-state, stoichiometry) to turn composition/structure into descriptors, not a
  hand-averaged atomic-property vector; then baked scikit-learn.
Name the method at the level you are confident in; never fabricate a citation, database name, or number.

### 2. Validate against a principled null or ground truth (the hallucination test)
A segmentation or orientation map returns *something* on any input; a number is not a result
until it beats a principled baseline.
- **Segmentation:** sweep the decisive parameter (threshold, watershed markers, min-object size)
  and report how grain count, mean grain size, and phase/pore volume fraction move — a quantity
  that swings wildly with a threshold is not resolved. Cross-check a phase volume fraction against
  an independent estimate (e.g. composition + a CALPHAD phase-fraction prediction) and reconcile.
- **Texture:** report ODF/pole-figure intensity in **multiples-of-random-distribution (MRD)** or
  the texture index relative to a UNIFORM reference; "strong texture" with no random baseline is
  not a claim.
- **Misorientation / boundaries:** compare the grain-boundary misorientation-angle distribution
  against the random-orientation **Mackenzie distribution** for the crystal symmetry; deviations
  are what indicate real texture / special boundaries.
- **Phase prediction:** state the database and conditions; corroborate a predicted phase against
  what EBSD/EDS/structure actually shows, and reconcile disagreements rather than reporting only
  the model.

### 3. Quantify uncertainty and stability on every decision-relevant estimate
Report grain-size, aspect-ratio, and volume-fraction as **distributions with N and spread**
(mean, median, P10–P90, count), not a single mean; state whether stereological correction was
applied, and flag/exclude boundary-touching grains (report how many). Re-run the segmentation
across the decisive parameter range and across sub-volumes; report each metric as mean ± spread.
If a metric changes by more than its spread across reasonable parameters, label it "not resolved
at this precision." Report the number of grains/particles/pores analyzed and the analyzed volume
+ voxel size so representativeness can be judged (a 20-grain volume does not support a grain-size
distribution claim).

### 4. Respect the microstructure and the physics of the data
- **3D reconstruction integrity:** TriBeam serial sectioning is destructive — slice-to-slice
  registration/drift, anisotropic voxels (z-spacing vs in-plane), and ablation artifacts
  propagate into 3D grain shapes. State the voxel dimensions, whether slices were registered, and
  any anisotropy; a 3D grain shape from unregistered or strongly anisotropic slices is suspect.
- **Multimodal alignment:** TriBeam produces co-registered SE (morphology), EBSD (orientation/
  phase), and EDS (chemistry) channels. State which modality drives a quantity and confirm the
  channels share a grid before fusing.
- **Crystallography conventions:** state symmetry / point group, Euler convention (e.g. Bunge
  ZXZ), and reference frame for every orientation quantity; a misorientation or IPF colour is only
  meaningful with symmetry applied. For superalloys respect the γ/γ′ cube-cube orientation
  relationship and L1₂ ordering.
- **Preprocessing you must state:** denoising, orientation cleanup / grain dilation,
  grain-reconstruction misorientation tolerance, pattern-quality masking — each changes grain
  count and boundary character.

### 5. Reproducibility record
Record the interpreter, package versions (pip metadata for pymatgen/defdap, which lack
`__version__`), the exact segmentation pipeline and parameters, crystal symmetry / Euler
convention / reference frame, voxel dimensions, the CALPHAD **database name/version and conditions
(composition, T, P)**, matminer featurizer names, seeds, and the parameter sweeps. Prefer
idempotent script entrypoints. Write intermediate artifacts (segmented label volumes, orientation
maps, per-grain regionprops tables) to `/workspace` / `/outputs`.

### 6. Honest accounting
If a real named tool was unavailable and you used a surrogate, say so and name what was lost — do
not present a numpy reimplementation as orix or an averaged-atomic-property vector as a matminer
featurizer. If pycalphad had no appropriate TDB, say the phase prediction was not run rather than
substitute a heuristic. Every conclusion (this is a TCP σ phase; the texture is ⟨110⟩ fiber; γ′
volume fraction is X%; this composition is single-phase) gets a confidence level tied to §2–§3 —
the null/ground-truth comparison, the parameter stability, the sampling adequacy — not just that
a tool returned something. Report which validations you ran and which you did not; report
wall-clock vs compute time separately for long tomography or CALPHAD sweeps.

## Cross-reference and delegation note
`/skills/scientific-reporting` is the write-up contract. TriBeam volumes are 3D image stacks
structurally like medical volumes — reuse the SimpleITK/nibabel stack and the
`/skills/medical-volume-slices` tool to render correctly-proportioned single-panel slices for
the vision-reasoner instead of hand-writing montage code. When delegating a verification subtask
(e.g. "confirm this grain-size distribution / texture / phase fraction survives a parameter sweep
and beats the random baseline"), give the subagent this skill's null (Mackenzie / MRD /
independent phase-fraction), the sweep, and the sampling count, and reconcile to yours against
the spread.
