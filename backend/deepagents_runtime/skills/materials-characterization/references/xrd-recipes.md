# XRD vetted recipes

## Simulated Cu K-alpha powder pattern (`pymatgen`)

**Trap:** hand-coding Bragg peaks, omitting the wavelength/occupancies, or presenting calculated
stick intensities as an experimental/refined pattern.

```python
import csv
import json
from importlib.metadata import version
from pathlib import Path

from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure


def peak_hkls(group):
    return [
        {
            "hkl": list(item["hkl"]),
            "multiplicity": int(item["multiplicity"]),
        }
        for item in group
    ]


# Replace this regression control with the staged user structure for the actual result.
fcc_ni = Structure.from_spacegroup(
    "Fm-3m",
    Lattice.cubic(3.52),
    ["Ni"],
    [[0, 0, 0]],
)
calculator = XRDCalculator(wavelength="CuKa")
pattern = calculator.get_pattern(fcc_ni, two_theta_range=(20, 100))

rows = []
for two_theta, intensity, hkl_group, d_hkl in zip(
    pattern.x,
    pattern.y,
    pattern.hkls,
    pattern.d_hkls,
    strict=True,
):
    rows.append(
        {
            "two_theta_deg": float(two_theta),
            "relative_intensity": float(intensity),
            "d_spacing_angstrom": float(d_hkl),
            "hkls": peak_hkls(hkl_group),
        }
    )

first_hkls = {tuple(item["hkl"]) for item in rows[0]["hkls"]}
assert 44.0 < rows[0]["two_theta_deg"] < 45.2
assert (1, 1, 1) in first_hkls

Path("/outputs/xrd_metadata.json").write_text(
    json.dumps(
        {
            "kind": "simulated powder XRD",
            "structure": "fcc Ni regression control",
            "lattice_parameter_angstrom": 3.52,
            "radiation": "CuKa",
            "wavelength_angstrom": float(calculator.wavelength),
            "two_theta_range_deg": [20, 100],
            "broadening_model": None,
            "intensity_normalization": "maximum=100",
            "pymatgen_version": version("pymatgen"),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

with Path("/outputs/xrd_peaks.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=("two_theta_deg", "relative_intensity", "d_spacing_angstrom", "hkls"),
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "hkls": json.dumps(row["hkls"], sort_keys=True)})
```

For a user CIF, parse/stability-check the structure under the structure/thermodynamics skill and
replace the control structure. Record the actual wavelength value, not only the source label.
Pymatgen emits idealized peak positions/intensities; add a broadening function only with explicit
instrument/sample assumptions and keep the unbroadened peak table.

## Experimental peak fitting checklist

There is no universal fitting snippet: the correct likelihood, background, doublet treatment,
instrument broadening, and constraints depend on acquisition metadata. Before fitting, require:

1. calibrated two-theta or scattering-vector axis and units;
2. raw counts or an explicit variance model;
3. wavelength/optics, scan step, and resolution information;
4. declared background and peak-shape models with bounded parameters;
5. residual diagnostics and parameter covariance/intervals; and
6. held-out/diagnostic reflections for any phase-identification claim.

Save both raw and fitted arrays. A high R2 can coexist with biased peak centers and is not by
itself a fit validator. Do not report Rietveld-derived phase fractions without an installed,
executed, and validated refinement engine.
