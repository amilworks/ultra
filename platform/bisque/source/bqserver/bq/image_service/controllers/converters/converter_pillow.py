"""Pillow-based fallback converter for basic image formats.

This converter is intended for development environments where imgcnv is not
available (for example local macOS uv workflows). It supports metadata reads
and simple raster operations for common formats.
"""

from __future__ import annotations

import logging
import os
import os.path
from typing import Iterable

from PIL import Image

from bq.image_service.controllers.converter_base import ConverterBase, Format
from bq.image_service.controllers.defaults import block_reads
from bq.image_service.controllers.process_token import ProcessToken
from bq.util.compat import OrderedDict
from bq.util.locks import Locks

log = logging.getLogger("bq.image_service.converter_pillow")


def _pil_format_name(fmt: str | None) -> str:
    if not fmt:
        return "TIFF"
    f = fmt.lower()
    if f in ("jpg", "jpeg"):
        return "JPEG"
    if f in ("tif", "tiff", "ome-tif", "ome-tiff", "bigtiff"):
        return "TIFF"
    if f == "png":
        return "PNG"
    if f == "gif":
        return "GIF"
    if f == "bmp":
        return "BMP"
    if f == "webp":
        return "WEBP"
    return f.upper()


def _mode_depth_bits(mode: str) -> int:
    return {
        "1": 1,
        "L": 8,
        "P": 8,
        "RGB": 8,
        "RGBA": 8,
        "CMYK": 8,
        "YCbCr": 8,
        "LAB": 8,
        "HSV": 8,
        "I;16": 16,
        "I;16L": 16,
        "I;16B": 16,
        "I": 32,
        "F": 32,
    }.get(mode, 8)


def _open_image(path: str) -> Image.Image | None:
    try:
        return Image.open(path)
    except Exception:
        return None


class ConverterPillow(ConverterBase):
    installed = False
    version = None
    installed_formats = None
    name = "pillow"
    required_version = "0.0.0"

    @classmethod
    def get_version(cls):
        try:
            import PIL  # noqa: PLC0415
        except Exception:
            return None
        parts = [int(p) if p.isdigit() else 0 for p in PIL.__version__.split(".")]
        while len(parts) < 3:
            parts.append(0)
        return {
            "full": PIL.__version__,
            "numeric": parts[:3],
            "major": parts[0],
            "minor": parts[1],
            "build": parts[2],
        }

    @classmethod
    def get_formats(cls):
        if cls.installed_formats is not None:
            return
        cls.installed_formats = OrderedDict()
        cls.installed_formats["jpeg"] = Format(
            name="jpeg",
            fullname="JPEG",
            ext=["jpg", "jpeg"],
            reading=True,
            writing=True,
            metadata=True,
        )
        cls.installed_formats["png"] = Format(
            name="png",
            fullname="PNG",
            ext=["png"],
            reading=True,
            writing=True,
            metadata=True,
        )
        cls.installed_formats["tiff"] = Format(
            name="tiff",
            fullname="TIFF",
            ext=["tif", "tiff"],
            reading=True,
            writing=True,
            multipage=True,
            metadata=True,
        )
        # Keep a writable alias used by image service defaults.
        cls.installed_formats["bigtiff"] = Format(
            name="bigtiff",
            fullname="BigTIFF",
            ext=["tif", "tiff"],
            reading=True,
            writing=True,
            multipage=True,
            metadata=True,
        )
        cls.installed_formats["bmp"] = Format(
            name="bmp",
            fullname="BMP",
            ext=["bmp"],
            reading=True,
            writing=True,
            metadata=True,
        )
        cls.installed_formats["gif"] = Format(
            name="gif",
            fullname="GIF",
            ext=["gif"],
            reading=True,
            writing=True,
            metadata=True,
        )
        cls.installed_formats["webp"] = Format(
            name="webp",
            fullname="WEBP",
            ext=["webp"],
            reading=True,
            writing=True,
            metadata=True,
        )

    @classmethod
    def supported(cls, token, **kw):
        if not cls.installed:
            return False
        ifnm = token.first_input_file()
        if not ifnm or not os.path.exists(ifnm):
            return False
        img = _open_image(ifnm)
        if img is None:
            return False
        img.close()
        return True

    @classmethod
    def _frame_index(cls, img: Image.Image, token: ProcessToken, zt: tuple | None = None) -> int:
        n_frames = int(getattr(img, "n_frames", 1) or 1)
        if n_frames <= 1:
            return 0
        idx = 0
        if zt is not None:
            z, t = zt
            t0 = int(t[0]) if isinstance(t, (tuple, list)) and len(t) > 0 else int(t or 0)
            z0 = int(z[0]) if isinstance(z, (tuple, list)) and len(z) > 0 else int(z or 0)
            if t0 > 0:
                idx = t0 - 1
            elif z0 > 0:
                idx = z0 - 1
        elif token.series is not None:
            idx = max(0, int(token.series))
        return min(max(0, idx), n_frames - 1)

    @classmethod
    def _collect_info(cls, img: Image.Image, filename: str) -> dict:
        fmt = (img.format or os.path.splitext(filename)[1].lstrip(".") or "UNKNOWN").upper()
        bands = img.getbands() or ()
        info = {
            "format": fmt,
            "image_num_series": 0,
            "image_series_index": 0,
            "image_num_x": int(img.size[0]),
            "image_num_y": int(img.size[1]),
            "image_num_z": 1,
            "image_num_t": int(getattr(img, "n_frames", 1) or 1),
            "image_num_c": max(1, len(bands) if isinstance(bands, Iterable) else 1),
            "image_pixel_format": "unsigned integer",
            "image_pixel_depth": _mode_depth_bits(img.mode),
            "image_mode": img.mode,
            "image_num_resolution_levels": 1,
            "tile_num_x": 0,
            "tile_num_y": 0,
        }
        try:
            info["filesize"] = int(os.path.getsize(filename))
        except OSError:
            pass
        dpi = img.info.get("dpi")
        if isinstance(dpi, tuple) and len(dpi) == 2:
            try:
                info["pixel_resolution_x"] = float(dpi[0])
                info["pixel_resolution_y"] = float(dpi[1])
                info["pixel_resolution_unit_x"] = "dpi"
                info["pixel_resolution_unit_y"] = "dpi"
            except (TypeError, ValueError):
                pass
        return info

    @classmethod
    def info(cls, token, **kw):
        if not cls.installed:
            return {}
        ifnm = token.first_input_file()
        if not ifnm or not os.path.exists(ifnm):
            return {}
        with Locks(ifnm, failonread=(not block_reads)) as l:
            if l.locked is False:
                return {}
            img = _open_image(ifnm)
            if img is None:
                return {}
            try:
                return cls._collect_info(img, ifnm)
            finally:
                img.close()

    @classmethod
    def meta(cls, token, **kw):
        return cls.info(token, **kw)

    @classmethod
    def _prepare_frame(
        cls,
        token: ProcessToken,
        z: tuple | None = None,
        t: tuple | None = None,
        roi: tuple[int, int, int, int] | None = None,
    ) -> Image.Image | None:
        ifnm = token.first_input_file()
        if not ifnm or not os.path.exists(ifnm):
            return None
        img = _open_image(ifnm)
        if img is None:
            return None

        try:
            idx = cls._frame_index(img, token, zt=(z, t) if z is not None and t is not None else None)
            try:
                img.seek(idx)
            except EOFError:
                pass
            frame = img.copy()
        finally:
            img.close()

        if roi is not None:
            x1, x2, y1, y2 = [int(v or 0) for v in roi]
            left = max(0, x1 - 1) if x1 > 0 else 0
            upper = max(0, y1 - 1) if y1 > 0 else 0
            right = min(frame.size[0], x2) if x2 > 0 else frame.size[0]
            lower = min(frame.size[1], y2) if y2 > 0 else frame.size[1]
            if right > left and lower > upper:
                frame = frame.crop((left, upper, right, lower))
        return frame

    @classmethod
    def _save(cls, img: Image.Image, ofnm: str, fmt: str | None = None):
        out_dir = os.path.dirname(ofnm)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pil_fmt = _pil_format_name(fmt or os.path.splitext(ofnm)[1].lstrip("."))
        save_img = img
        if pil_fmt in ("JPEG", "BMP") and save_img.mode not in ("L", "RGB"):
            save_img = save_img.convert("RGB")
        save_img.save(ofnm, format=pil_fmt)

    @classmethod
    def convert(cls, token, ofnm, fmt=None, extra=None, **kw):
        frame = cls._prepare_frame(token)
        if frame is None:
            return None
        try:
            cls._save(frame, ofnm, fmt=fmt)
            return ofnm
        except Exception:
            log.exception("Pillow convert failed for %s", token.first_input_file())
            return None
        finally:
            frame.close()

    @classmethod
    def convertToOmeTiff(cls, token, ofnm, extra=None, **kw):
        return cls.convert(token, ofnm, fmt="tiff")

    @classmethod
    def thumbnail(cls, token, ofnm, width, height, **kw):
        frame = cls._prepare_frame(token)
        if frame is None:
            return None
        try:
            method = (kw.get("method") or "BC").upper()
            resample = {
                "NN": Image.Resampling.NEAREST,
                "BL": Image.Resampling.BILINEAR,
                "BC": Image.Resampling.BICUBIC,
            }.get(method, Image.Resampling.BICUBIC)
            frame.thumbnail((int(width), int(height)), resample=resample)
            cls._save(frame, ofnm, fmt=kw.get("fmt", "jpeg"))
            return ofnm
        except Exception:
            log.exception("Pillow thumbnail failed for %s", token.first_input_file())
            return None
        finally:
            frame.close()

    @classmethod
    def slice(cls, token, ofnm, z, t, roi=None, **kw):
        frame = cls._prepare_frame(token, z=z, t=t, roi=roi)
        if frame is None:
            return None
        try:
            cls._save(frame, ofnm, fmt=kw.get("fmt", "tiff"))
            return ofnm
        except Exception:
            log.exception("Pillow slice failed for %s", token.first_input_file())
            return None
        finally:
            frame.close()

    @classmethod
    def tile(cls, token, ofnm, level=None, x=None, y=None, sz=None, **kw):
        frame = cls._prepare_frame(token)
        if frame is None:
            return None
        try:
            lvl = max(0, int(level or 0))
            tile_x = max(0, int(x or 0))
            tile_y = max(0, int(y or 0))
            tile_sz = max(1, int(sz or 512))

            if lvl > 0:
                scale = 2 ** lvl
                new_size = (max(1, frame.size[0] // scale), max(1, frame.size[1] // scale))
                frame = frame.resize(new_size, resample=Image.Resampling.BILINEAR)

            left = tile_x * tile_sz
            upper = tile_y * tile_sz
            if left >= frame.size[0] or upper >= frame.size[1]:
                return None
            right = min(frame.size[0], left + tile_sz)
            lower = min(frame.size[1], upper + tile_sz)
            tile_img = frame.crop((left, upper, right, lower))
            cls._save(tile_img, ofnm, fmt=kw.get("fmt", "tiff"))
            tile_img.close()
            return ofnm
        except Exception:
            log.exception("Pillow tile failed for %s", token.first_input_file())
            return None
        finally:
            frame.close()


try:
    ConverterPillow.init()
except Exception:
    log.warning("Pillow converter unavailable")
