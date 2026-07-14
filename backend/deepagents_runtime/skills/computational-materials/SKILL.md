---
name: computational-materials
description: Compatibility skill for materials microstructure and EBSD analysis — crystallographic orientations, IPF/pole-figure/ODF and misorientation work, raw Kikuchi indexing, TriBeam or tomography segmentation, grain/phase/precipitate morphology, stereology, and porosity. Use for EBSD maps, DREAM.3D-style microstructures, serial-section volumes, grain statistics, texture, boundary character, and pore networks. For CIF/space-group/CALPHAD work use materials-structure-thermo; for XRD/Raman/XPS/EDS use materials-characterization.
---

# Materials Microstructure and EBSD

## Scope and compatibility

This path is retained for compatibility with prior runs and prompts that name
`computational-materials`. Its production scope is now deliberately narrow:

- EBSD orientations, IPF maps, pole figures, ODF/texture, misorientation, and boundaries;
- raw Kikuchi-pattern indexing and simulated diffraction dictionaries;
- TriBeam, serial-section, and tomography segmentation and 3D reconstruction;
- grain, phase, precipitate, inclusion, and pore morphology and stereology; and
- microstructure-derived datasets and uncertainty-aware statistics.

Read `/skills/materials-structure-thermo/SKILL.md` for CIF, crystal symmetry, phase diagrams,
CALPHAD, and composition featurization. Read `/skills/materials-characterization/SKILL.md` for
XRD, SAED, Raman, XPS, or EDS. A task spanning raw EBSD patterns and orientation maps should read
this skill plus the characterization skill. Do not turn this compatibility path back into a
monolithic materials agent.

The default interpreter includes `orix`, `kikuchipy`, `diffsims`, `defdap`, `porespy`,
scikit-image, scipy, h5py, xarray, dask, zarr, SimpleITK, and nibabel. Record versions with
`importlib.metadata.version()`; `defdap` has no reliable top-level `__version__`.

## Required workflow

### 1. Preserve data and conventions

- Stage the selected input and identify it by path/artifact ID and SHA-256. Never overwrite it.
- Record array shape, dtype, channel names, physical voxel size `(dz, dy, dx)`, axis order,
  coordinate units, and whether the volume is anisotropic.
- For EBSD, record phase, point group/Laue class, Euler convention, degrees vs radians,
  active/passive convention, crystal-to-sample direction, and the sample direction used by each
  IPF (ND/IPF-Z, RD/IPF-X, TD/IPF-Y).
- For multimodal TriBeam data, identify the modality driving each quantity and verify SE, EBSD,
  and EDS channels share a registered physical grid before fusion.

### 2. Use the named domain implementation

Copy and run the vetted recipes before adapting them:

- [EBSD/IPF color and map recipe](references/ebsd-ipf-recipe.md)
- [Misorientation, texture, segmentation, stereology, and porosity recipes](references/materials-recipes.md)

Required boundaries:

- Use `orix` symmetry operations for orientation, disorientation, IPF color, pole figures, and
  ODF work. Never hand-roll quaternion symmetry reduction, Euler misorientation, or IPF RGB.
- The cubic TSL IPF key invariant is **001=red, 101=green, 111=blue**. Assert the 001/cube red
  condition in code before promoting an IPF artifact.
- Use `kikuchipy` for raw Kikuchi patterns and state Hough vs dictionary indexing, detector
  geometry/calibration, phase library, and every indexing/cleanup parameter. Use `diffsims` for
  a simulated diffraction dictionary; do not label a hand-made spot field as a dictionary.
- Use anisotropy-aware distance transforms and marker-controlled `skimage.segmentation.watershed`
  for touching grains when justified, then `label`/`regionprops_table` for named measurements.
  A bare threshold is a foreground mask, not a grain segmentation.
- Use `porespy` for named pore-network metrics and preserve its **True = void** convention.

### 3. Validate the quantity, not only the file

- IPF: require the programmatic 001-red invariant and record the point group and sample direction.
- Misorientation: compare the symmetry-reduced boundary distribution with a random-orientation
  Mackenzie baseline for cubic material (or the appropriate named null for another symmetry).
- Texture: report MRD or texture index relative to a uniform reference; raw pole density alone
  does not establish texture.
- Segmentation: sweep the decisive threshold, marker spacing, or cleanup size over a defensible
  neighborhood. Report how grain count, equivalent diameter, aspect ratio, and phase/pore volume
  fraction change. If the parameter effect exceeds sampling spread, mark the quantity not resolved
  at the stated precision.
- Stereology: check physical-volume recovery on a synthetic anisotropic object before analyzing
  the specimen. Report boundary-touching exclusions and results both before and after exclusion.
- Porosity: verify True=void on a labeled synthetic pore and compare the observed dense/porous
  direction with acquisition knowledge; `porosity == void.mean()` is only an identity, not a
  convention check.

### 4. Quantify sampling and uncertainty

Report distributions with count, mean, median, spread, and P10-P90 for decision-relevant grain,
particle, or pore quantities. Record analyzed physical volume and sub-volume layout. Use spatial
sub-volumes or specimen-level replicates for representativeness; pixels from one field of view are
not independent replicates. Seeds are relevant only when an algorithm is stochastic, and never
replace parameter stability or independent specimens.

### 5. Emit scientific validation separately from run success

Every analysis that makes a scientific claim must write
`/outputs/materials_validation.json`. A successfully completed process is not scientific
validation. Generate the record with
`ultra_deepagents.materials.validation.assess_scientific_status` and
`canonical_record_json`. The production sandbox guarantees this import at build time. If the
canonical module cannot be imported, treat that as release-infrastructure failure: do not
hand-author top-level verdict fields or claim verification. The top level contains `schema_version`,
`run_status`, `scientific_status` (`verified`, `failed`, `unsupported`, or `unverified`),
`verified`, `silent_success`, `capability_supported`, `required_validator_ids`,
`missing_validator_ids`, `critical_failures`, `contradiction_failures`, `reasons`, and `checks`.
For a supported task with no prose/artifact contradiction, the corresponding fields are
`"capability_supported": true` and `"contradiction_failures": []`; do not infer them from run
success. Every check follows this shape:

```json
{
  "validator_id": "materials.microstructure.ipf_001_red.v1",
  "outcome": "pass",
  "observed": {"rgb": [1.0, 0.0, 0.0]},
  "expected": {"red_min": 0.8, "green_max": 0.3, "blue_max": 0.3},
  "units": "unitless RGB",
  "tolerance_rationale": "TSL cubic IPF key invariant with rendering tolerance",
  "required": true,
  "critical": true,
  "library_versions": {"orix": "<installed-version>"},
  "evidence": [
    {"name": "ipf_map.png", "sha256": "0000000000000000000000000000000000000000000000000000000000000000", "path": "/outputs/ipf_map.png"}
  ],
  "message": "Cube/ND maps to the red TSL corner"
}
```

The all-zero digest above is a syntactically valid documentation placeholder, not evidence. It
must be replaced with the final file's actual lowercase SHA-256; trace validation resolves the
durable artifact and rejects a digest or size mismatch. Keep the complete validation record below
1,000,000 bytes so the bounded trace inspector can parse it.

Use only `pass`, `fail`, or `skip` for `outcome`. Declare the required validator IDs before
assessment; a missing or skipped required check must fail closed to `unverified`, a deterministic
failure to `failed`, and a missing release capability to `unsupported`. Hash the exact final
evidence artifact, not a temporary predecessor. Include validator IDs for convention checks,
synthetic controls, parameter sweeps, and independent/null comparisons.

### 6. Required outputs and accounting

Write final code, parameters, per-object tables, labels/orientation maps, figures, and
`materials_validation.json` under `/outputs`; keep exploratory diagnostics under `/workspace`.
State package versions, compute time and run wall-clock separately, failed or skipped checks, and
the evidence supporting each conclusion. If an input lacks voxel calibration, phase identity,
orientation convention, or enough sampled objects, report the bounded result that remains and
label the affected conclusion `unverified`; do not manufacture metadata.

For user-facing scientific status, summarize the JSON status and distinguish it explicitly from
run success. A correct-looking artifact without the required invariant record is not promotable.
