# Point-defect structure recipes

These recipes use public `pymatgen-analysis-defects` APIs and a simple NaCl regression control.
They validate object/generator/supercell semantics only. They do not calculate formation energies
and do not encode MatTools questions, verifier values, or expected benchmark answers.

## Symmetry-distinct vacancies and explicit supercells

**Trap:** deleting a raw site index and treating it as a symmetry-distinct defect, or comparing a
defect supercell with the unit-cell atom count.

```python
import numpy as np
from pymatgen.analysis.defects.generators import VacancyGenerator
from pymatgen.core import Lattice, Structure

bulk = Structure.from_spacegroup(
    "Fm-3m",
    Lattice.cubic(5.64),
    ["Na", "Cl"],
    [[0, 0, 0], [0.5, 0.5, 0.5]],
)
vacancies = list(VacancyGenerator(symprec=0.01).generate(bulk))

# This conventional control has one symmetry-distinct vacancy for each species.
assert len(bulk) == 8
assert len(vacancies) == 2
assert {defect.name for defect in vacancies} == {"v_Na", "v_Cl"}
assert {defect.multiplicity for defect in vacancies} == {4}
assert all(len(defect.defect_structure) == len(bulk) - 1 for defect in vacancies)

transform = np.diag([2, 2, 2])
for defect in vacancies:
    supercell = defect.get_supercell_structure(sc_mat=transform)
    assert len(supercell) == len(bulk) * int(round(np.linalg.det(transform))) - 1
```

For a user structure, derive the supercell transform from a declared convergence/minimum-image
criterion. Record `symprec`, multiplicity, defect coordinates, transform, site mapping, minimum
periodic image separation, and composition delta. A large supercell is not evidence of convergence
without an energy/geometry study.

## Substitutions, antisites, and interstitial candidates

**Trap:** reversing the substitution dictionary or calling every raw Voronoi node an independent
interstitial.

```python
from pymatgen.analysis.defects.generators import (
    AntiSiteGenerator,
    SubstitutionGenerator,
    VoronoiInterstitialGenerator,
)

# Reuse `bulk` from the NaCl control above. The mapping means host Na -> inserted K.
substitutions = list(SubstitutionGenerator().generate(bulk, {"Na": "K"}))
assert len(substitutions) == 1
assert substitutions[0].name == "K_Na"
assert substitutions[0].multiplicity == 4
assert len(substitutions[0].defect_structure) == len(bulk)

antisites = list(AntiSiteGenerator().generate(bulk))
assert {defect.name for defect in antisites} == {"Cl_Na", "Na_Cl"}

interstitials = list(
    VoronoiInterstitialGenerator(min_dist=1.0).generate(bulk, {"Li"})
)
assert interstitials
assert all(len(defect.defect_structure) == len(bulk) + 1 for defect in interstitials)
```

For real interstitial work, record generator type, inserted species, clustering and minimum-distance
tolerances, symmetry-equivalent positions/multiplicity, and the distance to host atoms. Sweep the
decisive geometric tolerances. Candidate generation is not a relaxation or stability prediction.

## Formation-energy input contract

`DefectEntry` and `FormationEnergyDiagram` are bookkeeping/analysis objects; their existence does
not supply physics inputs. Before constructing a formation-energy result, require compatible bulk
and defect `ComputedStructureEntry` energies, charge state, correction terms and metadata, chemical
potential phase-diagram entries, VBM, band gap, dielectric/correction provenance, and finite-size
evidence. If those are absent, emit an `unsupported` assessment through
`assess_scientific_status(..., capability_supported=False)` and list the missing inputs.
