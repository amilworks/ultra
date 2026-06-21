#!/usr/bin/env python3
"""medical-volume-slices: 3D/4D medical volume -> VLM-optimal 2D slices.

Turns a CT/MRI/fMRI/DICOM/tractography volume into a small set of clean, single-panel,
correctly-oriented, anatomically-spaced PNG slices a vision-language model can actually
read — instead of the agent ad-hoc-coding montages that lose resolution after the VLM's
image downscale.

Design choices that matter for VLM reading:
  * ONE anatomical image per PNG (no 3-panel montages) so detail survives the ~1280px cap.
  * Reorient to canonical RAS and display upright with a consistent, LABELED convention.
  * Correct the pixel aspect ratio from the voxel spacing (no anatomical distortion).
  * Modality-appropriate intensity: CT -> clinical windows (brain/bone); MRI -> robust
    1-99 percentile normalization; fMRI(4D) -> temporal mean then MRI-style.
  * Anatomically-spaced slice selection inside the brain/tissue extent (skip empty ends).
  * Cap the count so a follow-up VLM pass stays fast/cheap.

Usage (in the code sandbox; nibabel/SimpleITK/dipy are pre-installed):
  python /opt/medvol/volume_slices.py --input /workspace/staged_uploads/scan.nii.gz \
      --out /outputs/volume_slices --planes axial,coronal,sagittal --n-axial 6 --n-coronal 4 --n-sagittal 3

Prints a JSON manifest (modality, shape, spacing, the slice paths grouped by plane) to stdout
for the agent to read, then delegate the listed slice PNGs to the vision-reasoner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_NIFTI_SUFFIXES = (".nii", ".nii.gz")
_TRACT_SUFFIXES = (".trk", ".tck")


# --------------------------------------------------------------------------- loading

def _is_nifti(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def _is_tractography(p: Path) -> bool:
    return p.suffix.lower() in _TRACT_SUFFIXES


def load_volume(path: Path):
    """Return (kind, data, spacing, meta). kind in {nifti3d, nifti4d, dicom, tractography}.

    data is a float32 numpy array in canonical RAS for nifti/dicom; for tractography it is
    a streamlines object carried in meta."""
    if _is_tractography(path):
        return _load_tractography(path)
    if _is_nifti(path):
        return _load_nifti(path)
    if path.is_dir() or path.suffix.lower() in (".dcm", ".ima"):
        return _load_dicom(path)
    # last resort: let SimpleITK try (handles .mha/.nrrd/.nii too)
    return _load_sitk(path)


def _load_nifti(path: Path):
    import nibabel as nib

    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)  # -> RAS+, so axes are (R, A, S)
    data = np.asanyarray(img.dataobj, dtype=np.float32)  # applies scl_slope/inter
    zooms = img.header.get_zooms()
    meta = {"source": path.name, "affine": img.affine.tolist()}
    if data.ndim == 4:
        spacing = tuple(float(z) for z in zooms[:3])
        meta["n_volumes"] = int(data.shape[3])
        return "nifti4d", data, spacing, meta
    if data.ndim == 3:
        return "nifti3d", data, tuple(float(z) for z in zooms[:3]), meta
    raise ValueError(f"unsupported NIfTI ndim={data.ndim}")


def _load_dicom(path: Path):
    import SimpleITK as sitk  # noqa: N813

    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        # a folder can hold several series (scout, localizer, the main volume, ...);
        # pick the one with the most slices rather than whatever GetGDCMSeriesFileNames
        # returns first, so we always reconstruct the primary 3D volume.
        series = reader.GetGDCMSeriesIDs(str(path))
        if series:
            files = max((reader.GetGDCMSeriesFileNames(str(path), sid) for sid in series),
                        key=len)
        else:
            files = reader.GetGDCMSeriesFileNames(str(path))
        if not files:
            raise ValueError(f"no DICOM series found in {path}")
        reader.SetFileNames(files)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(path))
    data, spacing = _sitk_to_ras(image)
    return "dicom", data, spacing, {"source": path.name}


def _load_sitk(path: Path):
    import SimpleITK as sitk  # noqa: N813

    image = sitk.ReadImage(str(path))
    data, spacing = _sitk_to_ras(image)
    return "dicom", data, spacing, {"source": path.name}


def _sitk_to_ras(image):
    import SimpleITK as sitk  # noqa: N813

    # DICOMOrient to RAS gives a consistent anatomical layout; array is (z,y,x).
    try:
        image = sitk.DICOMOrient(image, "RAS")
    except Exception:
        pass
    arr = sitk.GetArrayFromImage(image).astype(np.float32)  # (z, y, x)
    arr = np.transpose(arr, (2, 1, 0))  # -> (x, y, z) to match nibabel RAS convention
    sp = image.GetSpacing()  # (x, y, z)
    return arr, (float(sp[0]), float(sp[1]), float(sp[2]))


def _load_tractography(path: Path):
    from dipy.io.streamline import load_tractogram

    sft = load_tractogram(str(path), "same", bbox_valid_check=False)
    return "tractography", None, (1.0, 1.0, 1.0), {"source": path.name, "sft": sft}


# --------------------------------------------------------------------------- modality + intensity

def detect_modality(kind: str, data: np.ndarray) -> str:
    if kind == "nifti4d":
        return "fmri"
    if kind == "tractography":
        return "tractography"
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return "unknown"
    # CT is calibrated in Hounsfield units: air ~ -1000, so a substantial negative tail
    # with a plausible HU max is the signature; MRI intensities are non-negative arbitrary.
    p1 = float(np.percentile(finite, 1))
    if p1 < -200 and float(finite.min()) < -500 and float(np.percentile(finite, 99)) < 4000:
        return "ct"
    return "mri"


def _window(slice2d: np.ndarray, center: float, width: float) -> np.ndarray:
    lo, hi = center - width / 2.0, center + width / 2.0
    out = np.clip((slice2d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def _robust_norm(slice2d: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.0,
                 ref: np.ndarray | None = None) -> np.ndarray:
    base = ref if ref is not None else slice2d
    finite = base[np.isfinite(base)]
    if finite.size == 0:
        return np.zeros(slice2d.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, lo_p), np.percentile(finite, hi_p)
    out = np.clip((slice2d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


# --------------------------------------------------------------------------- slice selection + render

def _foreground_extent(volume: np.ndarray, axis: int):
    """[lo, hi] index range along `axis` where the volume actually has tissue (skip empty
    air at the ends), using a low threshold on the per-slice signal."""
    other = tuple(a for a in range(volume.ndim) if a != axis)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0, volume.shape[axis] - 1
    thr = np.percentile(finite, 60)
    prof = (volume > thr).sum(axis=other)
    nz = np.where(prof > prof.max() * 0.02)[0] if prof.max() > 0 else np.array([])
    if nz.size == 0:
        return 0, volume.shape[axis] - 1
    return int(nz[0]), int(nz[-1])


def _pick_indices(volume: np.ndarray, axis: int, n: int):
    lo, hi = _foreground_extent(volume, axis)
    span = hi - lo
    if span <= 0 or n <= 0:
        return [volume.shape[axis] // 2]
    # keep inside the central 80% of the tissue extent (the diagnostic range)
    a, b = lo + int(0.10 * span), hi - int(0.10 * span)
    if b <= a:
        a, b = lo, hi
    return sorted({int(round(x)) for x in np.linspace(a, b, n)})


_PLANE_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}  # canonical RAS axes (R, A, S)


def _extract_plane(volume: np.ndarray, plane: str, idx: int):
    """Return (slice2d, (sp_h, sp_w), caption_side) oriented upright for display.

    RAS axes: 0=R(L->R), 1=A(P->A), 2=S(I->S). We display each plane the conventional way
    with a documented orientation so the VLM (and reader) can interpret left/right."""
    axis = _PLANE_AXIS[plane]
    sl = np.take(volume, idx, axis=axis)
    # canonical RAS -> rot90 puts the conventional axis up. L/R is per RAS (left of the
    # array = patient right); we caption the up-direction and tell the reader to judge
    # symmetry rather than rely on absolute L/R.
    img = np.rot90(sl)
    side = {"axial": "anterior=top", "coronal": "superior=top",
            "sagittal": "superior=top, anterior=left"}[plane]
    return img, side, axis


def _spacing_for_plane(spacing, plane: str):
    # physical mm per pixel for the displayed (rows, cols) after rot90
    sx, sy, sz = spacing
    if plane == "axial":
        return sx, sy   # after rot90 of (R,A): rows~A(sy), cols~R(sx) -> use (sy, sx)
    if plane == "coronal":
        return sz, sx
    return sz, sy


def _render(slice2d_u8: np.ndarray, plane: str, world_mm: float, spacing, *,
            target: int, label: str):
    from PIL import Image, ImageDraw

    im = Image.fromarray(slice2d_u8, mode="L").convert("RGB")
    # correct anatomical aspect from voxel spacing, then scale so the brain is ~target px
    sp_h, sp_w = _spacing_for_plane(spacing, plane)
    h, w = im.size[1], im.size[0]
    phys_h, phys_w = h * sp_h, w * sp_w
    scale = target / max(phys_h, phys_w)
    new_w = max(1, int(round(phys_w * scale)))
    new_h = max(1, int(round(phys_h * scale)))
    im = im.resize((new_w, new_h), Image.LANCZOS)
    # thin caption bar so the VLM knows the plane/level/orientation
    bar = 22
    canvas = Image.new("RGB", (im.size[0], im.size[1] + bar), (0, 0, 0))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((4, im.size[1] + 4), f"{plane} @ {world_mm:+.0f}mm | {label}", fill=(0, 255, 0))
    return canvas


# --------------------------------------------------------------------------- main per-modality

def _world_mm(spacing, plane, idx, shape):
    axis = _PLANE_AXIS[plane]
    return (idx - shape[axis] / 2.0) * spacing[axis]


def slice_scalar_volume(volume3d, spacing, modality, out: Path, args) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    plane_n = {"axial": args.n_axial, "coronal": args.n_coronal, "sagittal": args.n_sagittal}
    for plane in args.planes:
        for idx in _pick_indices(volume3d, _PLANE_AXIS[plane], plane_n.get(plane, 4)):
            sl, side, _ = _extract_plane(volume3d, plane, idx)
            world = _world_mm(spacing, plane, idx, volume3d.shape)
            renders = []
            if modality == "ct":
                renders.append(("brain", _window(sl, 40, 80)))
                if plane == "axial":  # one bone window per axial level is clinically useful
                    renders.append(("bone", _window(sl, 500, 2000)))
            else:  # mri / fmri-mean / unknown -> robust normalization on the whole volume
                renders.append(("", _robust_norm(sl, ref=volume3d)))
            for win, u8 in renders:
                label = (win + " " if win else "") + side.split(";")[0]
                im = _render(u8, plane, world, spacing, target=args.max_edge, label=label)
                name = f"{plane}_{idx:03d}{('_' + win) if win else ''}.png"
                im.save(out / name, "PNG")
                written.append({"plane": plane, "index": idx, "world_mm": round(world, 1),
                                "window": win or "normalized", "path": str(out / name),
                                "size": list(im.size)})
    return written


def slice_tractography(meta, out: Path, args) -> list[dict]:
    """Track-density projection: bin streamline points into a grid and max-project onto
    each plane (headless, no VTK). Gives the VLM a density 'glass-brain' per view."""
    out.mkdir(parents=True, exist_ok=True)
    sft = meta["sft"]
    sft.to_rasmm()
    pts = np.concatenate([s for s in sft.streamlines]) if len(sft.streamlines) else np.zeros((0, 3))
    written: list[dict] = []
    if pts.shape[0] == 0:
        return written
    mn, mx = pts.min(0), pts.max(0)
    res = 1.5  # mm per bin
    dims = np.maximum(((mx - mn) / res).astype(int) + 1, 1)
    grid = np.zeros(dims, dtype=np.float32)
    ij = np.clip(((pts - mn) / res).astype(int), 0, dims - 1)
    np.add.at(grid, (ij[:, 0], ij[:, 1], ij[:, 2]), 1.0)
    grid = np.log1p(grid)
    for plane, axis in (("axial", 2), ("coronal", 1), ("sagittal", 0)):
        proj = grid.max(axis=axis)
        proj = np.rot90(proj)
        u8 = _robust_norm(proj, 0, 99.5)
        im = _render(u8, plane, 0.0, (res, res, res), target=args.max_edge, label="track density (log)")
        name = f"tract_{plane}.png"
        im.save(out / name, "PNG")
        written.append({"plane": plane, "window": "track_density", "path": str(out / name),
                        "size": list(im.size)})
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Medical 3D/4D volume -> VLM-optimal slices.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="/outputs/volume_slices")
    ap.add_argument("--planes", default="axial,coronal,sagittal")
    ap.add_argument("--n-axial", type=int, default=6)
    ap.add_argument("--n-coronal", type=int, default=4)
    ap.add_argument("--n-sagittal", type=int, default=3)
    ap.add_argument("--max-edge", type=int, default=896)
    ap.add_argument("--modality", default="auto", choices=["auto", "ct", "mri", "fmri", "tractography"])
    args = ap.parse_args(argv)
    args.planes = [p.strip() for p in args.planes.split(",") if p.strip() in _PLANE_AXIS]

    path = Path(args.input).expanduser()
    if not path.exists():
        print(json.dumps({"ok": False, "error": f"input not found: {args.input}"}))
        return 2
    out = Path(args.out).expanduser()

    kind, data, spacing, meta = load_volume(path)
    modality = args.modality if args.modality != "auto" else detect_modality(kind, data if data is not None else np.zeros(1))

    if kind == "tractography" or modality == "tractography":
        slices = slice_tractography(meta, out, args)
        summary = {"modality": "tractography", "n_streamlines": len(meta["sft"].streamlines)}
    else:
        vol = data
        notes = []
        if kind == "nifti4d":  # fMRI / DWI time-series -> show anatomy via the temporal mean
            vol = np.nanmean(data, axis=3)
            notes.append(f"4D volume ({meta.get('n_volumes')} timepoints); showing the temporal MEAN "
                         "(anatomical reference) — fMRI/DWI statistics are not assessable from raw slices.")
        slices = slice_scalar_volume(vol, spacing, modality, out, args)
        summary = {"modality": modality, "shape": list(vol.shape),
                   "spacing_mm": [round(s, 3) for s in spacing], "notes": notes}

    by_plane: dict[str, list[str]] = {}
    for s in slices:
        by_plane.setdefault(s["plane"], []).append(s["path"])
    manifest = {"ok": True, "source": meta.get("source"), **summary,
                "slice_count": len(slices), "out_dir": str(out),
                "slices_by_plane": by_plane, "slices": slices,
                "orientation": "canonical RAS; views rendered upright (see each slice caption)"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
