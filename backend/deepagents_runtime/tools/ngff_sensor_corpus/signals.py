"""Deterministic synthetic sensor-signal generators.

Every generator is seeded and pure: the same ``(shape, seed)`` always yields the same
array, so the corpus is reproducible across machines and CI runs. Signals are shaped to
*look* like the real modality they stand in for — a fluorescence blob field, a spectral
ramp across bands, a tomographic shell, a spectrogram of drifting tones — so a human
inspecting a rendered slice sees something recognisable, and so auto-contrast / windowing
has real structure to work against.

The functions here only ever produce a single 2-D ``(Y, X)`` plane for a given
non-spatial coordinate ``(t, c, z)``. The writer composes planes into the full
n-dimensional store, which keeps peak memory at one plane regardless of store size.
"""

from __future__ import annotations

import numpy as np

__all__ = ["plane"]


def _rng(seed: int, t: int, c: int, z: int) -> np.random.Generator:
    """A per-plane deterministic RNG so noise is stable yet varies across t/c/z."""
    return np.random.default_rng((seed * 1_000_003 + t * 9_176 + c * 131 + z * 17) & 0xFFFFFFFF)


def _coords(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    return yy, xx


def _blobs(h: int, w: int, rng: np.random.Generator, n: int, sharpness: float) -> np.ndarray:
    """Sum of Gaussian blobs — fluorescence puncta, mineral grains, calorimeter hits."""
    yy, xx = _coords(h, w)
    field = np.zeros((h, w), dtype=np.float64)
    for _ in range(min(n, 300)):  # cap: past a few hundred blobs the field just saturates
        cy, cx = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
        amp = rng.uniform(0.4, 1.0)
        sig = rng.uniform(0.03, 0.16)
        field += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig**2)) * sharpness)
    return field


def _ramp(h: int, w: int, angle: float) -> np.ndarray:
    yy, xx = _coords(h, w)
    return 0.5 * (1.0 + np.cos(angle) * xx + np.sin(angle) * yy)


def _radial(h: int, w: int, rings: float) -> np.ndarray:
    yy, xx = _coords(h, w)
    r = np.sqrt(yy**2 + xx**2)
    return 0.5 * (1.0 + np.cos(r * rings * np.pi))


def _shell(h: int, w: int, radius: float, thickness: float) -> np.ndarray:
    """A soft spherical shell cross-section — a CT / tomography slice through a part."""
    yy, xx = _coords(h, w)
    r = np.sqrt(yy**2 + xx**2)
    return np.exp(-((r - radius) ** 2) / (2 * thickness**2))


def _spectrogram(h: int, w: int, rng: np.random.Generator, tones: int) -> np.ndarray:
    """Frequency (Y) x time (X): drifting sinusoidal tones + broadband texture."""
    field = np.zeros((h, w), dtype=np.float64)
    tvec = np.linspace(0.0, 1.0, w)
    for _ in range(tones):
        f0 = rng.uniform(0.05, 0.9)
        drift = rng.uniform(-0.15, 0.15)
        track = np.clip(f0 + drift * tvec, 0.02, 0.98) * (h - 1)
        rows = track.astype(int)
        field[rows, np.arange(w)] += rng.uniform(0.6, 1.0)
    # Bleed the tone tracks vertically a little so they read as bands, not 1px lines.
    field = 0.6 * field + 0.4 * np.vstack([np.roll(field, k, axis=0) for k in (-1, 0, 1)]).reshape(
        3, h, w
    ).mean(0)
    return field + 0.05 * rng.standard_normal((h, w))


def _labels(h: int, w: int, rng: np.random.Generator, n: int) -> np.ndarray:
    """Integer label field (segmentation mask) via nearest-seed Voronoi tessellation."""
    yy, xx = np.mgrid[0:h, 0:w]
    seeds_y = rng.integers(0, h, size=n)
    seeds_x = rng.integers(0, w, size=n)
    out = np.zeros((h, w), dtype=np.int64)
    best = np.full((h, w), np.inf)
    for i in range(n):
        d = (yy - seeds_y[i]) ** 2 + (xx - seeds_x[i]) ** 2
        closer = d < best
        best = np.where(closer, d, best)
        out = np.where(closer, i + 1, out)  # label 0 reserved for background
    return out.astype(np.float64)


def _rgb_tissue(h: int, w: int, rng: np.random.Generator, channel: int) -> np.ndarray:
    """One channel of a faux H&E slide: eosin (pink) + hematoxylin (nuclei) texture."""
    nuclei = _blobs(h, w, _rng(0, 0, 7, 0), n=max(8, (h * w) // 900), sharpness=6.0)
    stroma = _ramp(h, w, 0.7) * 0.5 + 0.3 * _radial(h, w, 3.0)
    # R, G, B weightings that read as pink tissue with purple nuclei.
    mix = (
        (0.85 * stroma + 0.15 * (1 - nuclei)),
        (0.55 * stroma + 0.05),
        (0.80 * stroma + 0.35 * nuclei),
    )
    return np.clip(mix[channel % 3], 0.0, 1.0)


def plane(
    kind: str,
    h: int,
    w: int,
    *,
    seed: int,
    t: int = 0,
    c: int = 0,
    z: int = 0,
    num_c: int = 1,
    num_z: int = 1,
    num_t: int = 1,
) -> np.ndarray:
    """Return a float64 ``(h, w)`` plane in roughly [0, 1] for the given coordinate.

    ``kind`` selects the modality's spatial signature; ``t/c/z`` and the ``num_*`` counts
    let the signature evolve across the non-spatial axes (channels separate spectrally,
    z sweeps a volume, t advances a process).
    """
    rng = _rng(seed, t, c, z)

    if kind == "fluor_puncta":
        # Each channel = a different fluorophore with its own blob population; brighter
        # near a channel-specific "structure". z sweeps focus (blur via fewer/softer blobs).
        density = max(6, (h * w) // 1600)
        base = _blobs(h, w, _rng(seed, 0, c, 0), n=density, sharpness=2.0 + c)
        focus = 1.0 - abs((z + 0.5) / max(1, num_z) - 0.5) * 1.2
        return np.clip(base * max(0.15, focus), 0.0, 1.0)

    if kind == "em_grayscale":
        tex = _blobs(h, w, rng, n=max(20, (h * w) // 400), sharpness=8.0)
        return np.clip(0.25 + 0.75 * tex + 0.05 * rng.standard_normal((h, w)), 0.0, 1.0)

    if kind == "rgb_tissue":
        return _rgb_tissue(h, w, rng, c)

    if kind == "labels":
        return _labels(h, w, rng, n=max(6, (h * w) // 2500))

    if kind == "calcium":
        # Timelapse: a few neurons flash on a slow oscillation.
        base = _blobs(h, w, _rng(seed, 0, 0, 0), n=max(5, (h * w) // 3000), sharpness=3.0)
        flash = 0.5 + 0.5 * np.sin(2 * np.pi * (t / max(1, num_t)) + base * 6.0)
        return np.clip(base * flash, 0.0, 1.0)

    if kind == "spectral_bands":
        # Hyperspectral / multispectral / EDS / MSI: a smooth spatial scene whose
        # amplitude is modulated by a per-band spectral response, plus band-specific
        # "features" so bands are visually distinct.
        scene = 0.6 * _ramp(h, w, 0.9) + 0.4 * _radial(h, w, 2.0)
        response = 0.35 + 0.65 * np.abs(np.sin(np.pi * (c + 0.5) / max(1, num_c)))
        feature = _blobs(h, w, _rng(seed, 0, c, 0), n=4, sharpness=2.0)
        return np.clip(scene * response + 0.3 * feature, 0.0, 1.0)

    if kind == "ct_volume":
        # Nested soft shells whose radius shrinks toward the volume centre (z sweep).
        depth = (z + 0.5) / max(1, num_z)
        radius = 0.35 + 0.45 * np.sin(np.pi * depth)
        return _shell(h, w, radius=radius, thickness=0.06) + 0.15 * _radial(h, w, 6.0)

    if kind == "mri_sequences":
        # Each channel = a pulse sequence with different tissue contrast.
        anat = _shell(h, w, 0.6, 0.12) + 0.7 * _blobs(h, w, _rng(seed, 0, 0, 0), 6, 2.0)
        contrast = (0.4, 1.0, 0.7, 0.2)[c % 4]
        depth = 1.0 - abs((z + 0.5) / max(1, num_z) - 0.5)
        return np.clip(anat * contrast * (0.4 + 0.6 * depth), 0.0, 1.0)

    if kind == "seismic":
        # Layered reflectors (Y = depth) with lateral (X) structure and a fault.
        yy, xx = _coords(h, w)
        layers = np.sin((yy * 8 + 0.6 * np.sin(xx * 3 + z * 0.2)) * np.pi)
        return 0.5 * (1.0 + layers)

    if kind == "climate_field":
        # A smooth geophysical field (temperature / moisture) drifting over time.
        yy, xx = _coords(h, w)
        phase = 2 * np.pi * (t / max(1, num_t))
        field = np.cos(yy * 2.0 + phase) * np.sin(xx * 2.0 - 0.5 * phase)
        band = np.cos(yy * 3.0)  # latitude banding
        return 0.5 * (1.0 + 0.6 * field + 0.4 * band)

    if kind == "radar_volume":
        # Convective cells in a weather-radar volume; cells intensify with height (z).
        cells = _blobs(h, w, _rng(seed, 0, 0, 0), n=5, sharpness=1.5)
        gain = 0.4 + 0.6 * (z + 0.5) / max(1, num_z)
        return np.clip(cells * gain, 0.0, 1.0)

    if kind == "dem":
        # Terrain: fractal-ish sum of ramps + radial basins.
        yy, xx = _coords(h, w)
        terrain = (
            0.5 * np.sin(xx * 2 + 0.5) + 0.3 * np.sin(yy * 3 - 0.7) + 0.2 * np.cos((xx + yy) * 5)
        )
        return 0.5 * (1.0 + terrain)

    if kind == "spectrogram":
        return np.clip(_spectrogram(h, w, rng, tones=max(3, h // 24)), 0.0, None)

    if kind == "thermal":
        # IR camera: a warm object cooling over time on a cool background.
        hot = _blobs(h, w, _rng(seed, 0, 0, 0), n=3, sharpness=2.5)
        cool = np.exp(-2.5 * t / max(1, num_t))
        return np.clip(0.2 + 0.8 * hot * cool, 0.0, 1.0)

    if kind == "afm_height":
        # AFM/profilometry: terraced surface with atomic-step-like ramps + roughness.
        yy, xx = _coords(h, w)
        steps = np.floor((xx + yy) * 4) / 8.0
        return 0.5 + 0.4 * steps + 0.02 * rng.standard_normal((h, w))

    if kind == "ebsd_ipf":
        # EBSD inverse-pole-figure map: grains (Voronoi) each with a per-grain orientation
        # component; channel selects one RGB IPF component.
        lab = _labels(h, w, _rng(seed, 0, 0, 0), n=max(8, (h * w) // 2000))
        grng = np.random.default_rng(seed * 7 + c)
        colours = grng.uniform(0.1, 1.0, size=int(lab.max()) + 1)
        return colours[lab.astype(int)]

    if kind == "diffraction":
        # 4D-STEM virtual detector image / diffraction: bright central disc + rings.
        return _radial(h, w, 5.0) * 0.7 + _shell(h, w, 0.0, 0.12)

    if kind == "gradient":
        return _ramp(h, w, 0.5 + 0.3 * c)

    raise ValueError(f"unknown signal kind {kind!r}")
