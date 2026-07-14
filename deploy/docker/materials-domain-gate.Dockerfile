# syntax=docker/dockerfile:1.7

# Multi-architecture index pinned to the published Python 3.11.13 slim-bookworm
# image. The immutable index resolves to the matching amd64/arm64 manifest.
FROM python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1

ENV HOME=/tmp \
    LC_ALL=C.UTF-8 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/ultra/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    TZ=UTC \
    XDG_CACHE_HOME=/tmp/cache

WORKDIR /workspace

COPY deploy/docker/materials-requirements.txt /opt/ultra/materials-requirements.txt

# Pin the installer/test harness as well as every direct materials distribution.
# `pip check` makes dependency conflicts a build failure, while the runtime gate
# independently compares every direct installed version against the source pins.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && python -m pip install \
      pip==25.1.1 \
      setuptools==80.9.0 \
      wheel==0.45.1 \
    && python -m pip install \
      pytest==8.4.2 \
      -r /opt/ultra/materials-requirements.txt \
    && python -m pip check \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/deepagents_runtime/src/ultra_deepagents/materials /opt/ultra/src/ultra_deepagents/materials
COPY backend/deepagents_runtime/src/ultra_deepagents/sensors /opt/ultra/src/ultra_deepagents/sensors
COPY backend/deepagents_runtime/materials_data/calphad /opt/ultra-calphad
ENV ULTRA_CALPHAD_DATABASE_ROOT=/opt/ultra-calphad

RUN python - <<'PY'
import hashlib
import json
import math
from pathlib import Path

from pycalphad import Database
import damask
import scheil
from ultra_deepagents.materials import (
    assess_scientific_status,
    calculate_diffraction_profile_metrics,
    canonical_slip_systems,
    convert_corrosion_current_to_uniform_penetration,
    evaluate_mode_i_lefm,
    evaluate_norton_arrhenius_creep_rate,
    evaluate_oxidation_mass_gain,
    fit_rigid_registration,
    fit_paris_law,
    inspect_calphad_input,
    run_scheil_solidification,
)
from ultra_deepagents.materials.calphad_cli import main as typed_calphad_main
from ultra_deepagents.sensors import open_sensor_series, parse_sensor_series

assert callable(assess_scientific_status) and callable(inspect_calphad_input)
assert callable(canonical_slip_systems) and callable(run_scheil_solidification)
assert callable(calculate_diffraction_profile_metrics) and callable(fit_rigid_registration)
assert callable(evaluate_mode_i_lefm) and callable(fit_paris_law)
assert callable(evaluate_norton_arrhenius_creep_rate)
assert callable(evaluate_oxidation_mass_gain)
assert callable(convert_corrosion_current_to_uniform_penetration)
assert callable(open_sensor_series) and callable(parse_sensor_series)
assert callable(typed_calphad_main)
assert damask.__version__ == "3.1.0"
assert scheil.__version__ == "0.3.0"
root = Path("/opt/ultra-calphad")
manifest_path = root / "manifest.json"
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == "a5dfb3aac68f119a8fe0ee751255a16f31fe8e8515cfa9739320ba21aa28fb09"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for record in manifest["databases"]:
    path = root / record["filename"]
    payload = path.read_bytes()
    assert len(payload) == record["size_bytes"]
    assert hashlib.sha256(payload).hexdigest() == record["sha256"]
    database_format = str(record.get("format") or "").casefold()
    assert database_format in {"tdb", "dat"}
    assert path.suffix.casefold() == f".{database_format}"
    pressure_limits = record.get("assessment_pressure_limits_Pa")
    assert isinstance(pressure_limits, list) and len(pressure_limits) == 2
    assert all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        for value in pressure_limits
    )
    assert 1e-9 <= float(pressure_limits[0]) <= float(pressure_limits[1]) <= 1e12
    database = Database.from_file(str(path), fmt=database_format)
    components = sorted(element for element in database.elements if element != "/-")
    assert components == sorted(record["components"])
    assert sorted(element for element in components if element != "VA") == sorted(record["physical_elements"])
    assert sorted(element for element in database.elements if element == "/-") == sorted(record["pseudo_elements"])
    assert sorted(database.phases) == sorted(record["phases"])
    inspection = inspect_calphad_input(
        root,
        database_id=record["database_id"],
        components=record["components"],
        phases=record["phases"],
    )
    assert inspection["sha256"] == record["sha256"]
    assert inspection["format"] == database_format
    assert inspection["assessment_pressure_limits_Pa"] == [
        float(value) for value in pressure_limits
    ]
PY

COPY scripts/materials_domain_gate.py /opt/ultra/materials_domain_gate.py
COPY scripts/calphad_experimental_benchmark.py /opt/ultra/calphad_experimental_benchmark.py

ARG VCS_REF=unknown

# Revision-specific metadata stays after the expensive dependency layer so a new
# commit SHA does not invalidate the reusable materials environment cache.
LABEL org.opencontainers.image.title="Ultra deterministic materials domain gate" \
      org.opencontainers.image.description="Pinned scientific environment for non-skipping materials invariants" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/amilworks/ultra"

ENTRYPOINT ["python", "/opt/ultra/materials_domain_gate.py"]
