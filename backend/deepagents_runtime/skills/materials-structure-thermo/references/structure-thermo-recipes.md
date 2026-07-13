# Structure and thermodynamics vetted recipes

## Space-group analysis with ordered/disordered controls

**Trap:** identifying a phase by lattice parameter or running a single symmetry tolerance. This
can collapse ordered L1_2 Ni3Al to an fcc average or promote coordinate noise to lower symmetry.

```python
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def assignment(structure, symprec, angle_tolerance=5.0):
    analyzer = SpacegroupAnalyzer(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    return {
        "symprec": symprec,
        "angle_tolerance_deg": angle_tolerance,
        "number": analyzer.get_space_group_number(),
        "symbol": analyzer.get_space_group_symbol(),
    }


l12 = Structure(
    Lattice.cubic(3.57),
    ["Al", "Ni", "Ni", "Ni"],
    [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
)
fcc_ni = Structure.from_spacegroup(
    "Fm-3m",
    Lattice.cubic(3.52),
    ["Ni"],
    [[0, 0, 0]],
)

tolerances = (0.001, 0.01, 0.1)
l12_results = [assignment(l12, value) for value in tolerances]
fcc_results = [assignment(fcc_ni, value) for value in tolerances]
assert {row["number"] for row in l12_results} == {221}
assert {row["number"] for row in fcc_results} == {225}
```

Choose the user's tolerance range from coordinate precision and expected positional noise; the
control range above is a regression test, not a universal specimen tolerance. For the actual
structure, preserve occupancies/order and record results across the chosen sweep.

## CALPHAD equilibrium with an authorized `.tdb` or `.dat` database

**Trap:** using an internal pycalphad test fixture, omitting the vacancy component, or reporting
phase fractions without testing convergence and closure.

Inspect first and preserve the returned parser format plus both assessment intervals. `.tdb` uses
the Thermo-Calc parser and `.dat` uses the ChemSage parser; reject `.db`. Require every requested
temperature and pressure to lie within its declared interval. The pressure declaration must be the
canonical `assessment_pressure_limits_Pa` field with exactly two finite pascal values
`[minimum, maximum]` in `[1e-9, 1e12]`; missing or malformed bounds block inspection and
equilibrium. For the embedded NIST Al-Co-W
reference, use exactly 101325 Pa. Treat MQMQA constituent-group warnings as provenance to report,
not permission to fabricate site ratios.

Do not write or execute raw Python for this workflow. First call `calphad_inspect_database` with
exactly one of these typed sources:

```json
{"resource_id":"<server-authorized resource id>","embedded_database_id":"","components":null,"phases":null}
```

or, for Ultra's reviewed CC0 reference:

```json
{"resource_id":"","embedded_database_id":"nist-al-co-w-wang-2017","components":null,"phases":null}
```

The resource variant reads the SHA-256, byte size, and identity from the server-authored catalog
binding and the source/license/scope/reference-state/temperature limits plus
`assessment_pressure_limits_Pa` from explicitly labeled owner
declarations. Never copy those values from a filename or prompt prose. Select components and phases
from the returned inventory. For a global-equilibrium claim include every applicable database phase;
if phases are intentionally excluded, label the result restricted/metastable and list them.

Use the canonical dependent-component convention before forming conditions: remove `VA`, sort the
physical components, and derive the first component. For Al-Co-W, submit `CO` and `W` and derive
`AL`. Equivalent singleton or one-axis requests can be reframed by the typed runtime. A Cartesian
request with multiple varying axes must be rejected if canonical reframing would couple those axes.
This convention is part of the evidence identity and prevents dependent-component choice from
changing which local minimum pycalphad returns.

If the retained inventory lists `VA`, keep it in `components`; do not create `X(VA)`. The typed host
authenticates the retained inspection artifact and adds `VA` before request hashing when necessary,
while the sandbox rejects a forged/direct omission. For the Al-Co-W checkpoint, omitting `VA`
produces the same known higher-Gibbs `LIQUID` + `AL5CO2` solution described below.

The immutable resource revision snapshots those normalized declarations and the exact parser format.
After catalog garbage collection, replay only as `<sha256>.tdb` or `<sha256>.dat` using that retained
format and explicit pycalphad parser dispatch. Never guess from bytes, MIME type, or a default parser;
if the retained format or declaration snapshot is absent, stop and mark the replay non-promotable.

Then call `calphad_run_equilibrium` with the inspection artifact hash and typed conditions:

```json
{
  "inspection_artifact_sha256":"<inspection artifact sha256>",
  "resource_id":"<same resource id>",
  "embedded_database_id":"",
  "components":["NI","AL","VA"],
  "phases":["FCC_A1","L12_FCC","LIQUID"],
  "temperatures_K":[1073.0],
  "pressures_Pa":[101325.0],
  "independent_compositions":{"AL":[0.15]}
}
```

The typed primitive fixes `N=1 mol`, condition/result limits, wall time, pycalphad models, and output
surface. Preserve its content-addressed `/outputs/calphad/...` artifact. Verify phase-fraction and
per-vertex composition closure, chemical-potential units, and `phase_selection` scope from the v2
evidence. The restricted Ni-Al list above is only a request-shape example, not a global-equilibrium
claim or recommendation for an arbitrary TDB. The raw `inspect_calphad_input` and
`run_calphad_equilibrium` Python APIs are reserved for the trusted CLI implementation and replay
validators, not agent-authored production execution.

### Embedded Al-Co-W global-minimum checkpoint

For the exact embedded-reference state `T=1173 K`, `P=101325 Pa`, `N=1 mol`,
`X(AL)=0.675`, `X(CO)=0.260`, and `X(W)=0.065`, request all 18 assessed phases and submit the
canonical independent conditions `CO=0.260`, `W=0.065`. Require:

- stable phases `AL4W`, `AL5CO2`, and `BCC_B2`, matching Wang et al. (2017), Fig. 12(a)'s interior
  Al4W-Al5Co2-AlCo three-phase field;
- phase fractions approximately `0.3249914280`, `0.3487946716`, and `0.3262139004`, respectively;
- molar Gibbs energy approximately `-85970.067462 J/mol`; and
- the normal phase-fraction, bulk-composition, per-vertex composition, and Gibbs-Euler closure
  checks.

The retained published-figure asset has SHA-256
`2cbd5bda9493d138442133a4796b4dbd944cdf106dcda197341baa9f935e59e6`; bind the validation to the
retained asset and its source locator, not merely this prose. Phase identity is the independent
published check; the fractions and Gibbs energy are deterministic runtime regression values. A
`LIQUID` + `AL5CO2` result near `-85512.61 J/mol` is a known higher-Gibbs local solution and must
fail the checkpoint even if every closure residual passes. Do not label an arbitrary CALPHAD point
scientifically verified from closure alone.

## Magpie composition features

**Trap:** replacing the named featurizer with a hand-averaged atomic-property vector.

```python
import numpy as np
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

featurizer = ElementProperty.from_preset("magpie")
labels = featurizer.feature_labels()
values = np.asarray(featurizer.featurize(Composition("Ni3Al")), dtype=float)

assert len(labels) == 132
assert values.shape == (132,)
assert np.isfinite(values).all()
assert any("avg_dev" in label for label in labels)
assert any("range" in label for label in labels)
assert any("mode" in label for label in labels)
```

Magpie uses its own reference tables. A tiny discrepancy against another library's element table
is not automatically an error; record the preset/version and validate dimensionality, labels, and
finiteness.
