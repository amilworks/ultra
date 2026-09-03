"""The synthetic sensor-data catalog.

Every entry is one OME-Zarr store standing in for a real STEM sensor modality. The point
is breadth: the app claims to support "sensor data" across materials science, biology,
environmental/remote sensing, medical, geophysics, and astronomy, and this catalog puts a
concrete, spec-correct store behind each of those claims so the image/data service can be
exercised against all of them.

Design constraints baked in (from the OME-NGFF reader's validators):
  * Axis *names* are always drawn from the canonical set t/c/z/y/x — the reader treats any
    other name as a custom axis that must be singleton, so a "wavelength"/"band"/"depth"
    axis is modelled as c or z and its scientific identity is carried by the axis UNIT,
    the per-channel omero labels, the dtype, and the physical scale.
  * Exactly one y and one x axis; rank 2..5.
  * For NGFF 0.5: types are time/channel/space, ordered time -> channel -> space, with 2 or
    3 space axes, stored in Zarr v3 with dimension_names bound to the axis names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = ["DOMAINS", "ChannelSpec", "StoreSpec", "catalog"]

DOMAINS = (
    "materials",
    "biology",
    "environmental",
    "medical",
    "geophysics",
    "astronomy",
    "other",
)


@dataclass(frozen=True)
class ChannelSpec:
    label: str
    color: str  # RRGGBB hex, no '#'


@dataclass(frozen=True)
class StoreSpec:
    domain: str
    modality: str  # slug; becomes the store directory name
    title: str
    instrument: str  # the real sensor/instrument this mimics
    axes: tuple[tuple[str, str], ...]  # (canonical name, unit) in STORED order
    dtype: str
    base: dict[str, int]  # base (level-0) size per present axis
    signal: str  # signals.plane kind
    scale: dict[str, float]  # physical units-per-pixel at level 0
    ngff_version: Literal["0.4", "0.5"] = "0.4"
    zarr_version: Literal[2, 3] = 2
    levels: int = 1
    chunks: dict[str, int] = field(default_factory=dict)
    value_range: tuple[float, float] = (0.0, 1.0)
    channels: tuple[ChannelSpec, ...] = ()
    emit_omero: bool = True
    translation: dict[str, float] = field(default_factory=dict)
    lazy_fill: bool = False  # declare a large shape but write no pixels (scale probe)
    notes: str = ""

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.axes)

    @property
    def store_name(self) -> str:
        return f"{self.domain}__{self.modality}.ome.zarr"


def _fluor_channels() -> tuple[ChannelSpec, ...]:
    return (
        ChannelSpec("DAPI (nucleus)", "0033FF"),
        ChannelSpec("Alexa488 (actin)", "00FF00"),
        ChannelSpec("TRITC (mito)", "FF3300"),
        ChannelSpec("Cy5 (membrane)", "FF00FF"),
    )


def _sentinel_bands() -> tuple[ChannelSpec, ...]:
    names = [
        "B01 aerosol",
        "B02 blue",
        "B03 green",
        "B04 red",
        "B05 rededge1",
        "B06 rededge2",
        "B07 rededge3",
        "B08 nir",
        "B8A narrownir",
        "B09 watervapor",
        "B11 swir1",
        "B12 swir2",
    ]
    greys = ["888888"] * len(names)
    return tuple(ChannelSpec(n, g) for n, g in zip(names, greys, strict=True))


def _mri_channels() -> tuple[ChannelSpec, ...]:
    return (
        ChannelSpec("T1", "FFFFFF"),
        ChannelSpec("T2", "FFCC66"),
        ChannelSpec("FLAIR", "66CCFF"),
        ChannelSpec("DWI", "FF6666"),
    )


def catalog() -> list[StoreSpec]:
    """The full valid-store catalog (spec-correct across all domains)."""
    specs: list[StoreSpec] = []

    # ------------------------------------------------------------------ materials
    specs += [
        StoreSpec(
            domain="materials",
            modality="microct_alloy",
            title="X-ray micro-CT of a Ni-superalloy coupon",
            instrument="lab micro-CT / synchrotron tomography",
            axes=(("z", "micrometer"), ("y", "micrometer"), ("x", "micrometer")),
            dtype="uint16",
            base={"z": 24, "y": 512, "x": 512},
            signal="ct_volume",
            scale={"z": 3.0, "y": 1.1, "x": 1.1},
            levels=3,
            value_range=(0, 60000),
            emit_omero=False,
            notes="Isotropic-ish attenuation volume, 3-level pyramid.",
        ),
        StoreSpec(
            domain="materials",
            modality="eds_elemental_map",
            title="SEM-EDS hyperspectral elemental map",
            instrument="SEM energy-dispersive X-ray spectroscopy",
            axes=(("c", "electronvolt"), ("y", "micrometer"), ("x", "micrometer")),
            dtype="uint16",
            base={"c": 8, "y": 384, "x": 384},
            signal="spectral_bands",
            scale={"c": 1.0, "y": 0.25, "x": 0.25},
            levels=2,
            value_range=(0, 8000),
            channels=(
                ChannelSpec("Al-K", "FF0000"),
                ChannelSpec("Ti-K", "FF9900"),
                ChannelSpec("Cr-K", "FFFF00"),
                ChannelSpec("Fe-K", "66FF00"),
                ChannelSpec("Co-K", "00FFCC"),
                ChannelSpec("Ni-K", "0099FF"),
                ChannelSpec("Mo-L", "6633FF"),
                ChannelSpec("W-M", "FF00FF"),
            ),
            notes="Per-element count maps as channels; counts in uint16.",
        ),
        StoreSpec(
            domain="materials",
            modality="ebsd_ipf",
            title="EBSD inverse-pole-figure orientation map",
            instrument="SEM electron backscatter diffraction",
            axes=(("c", "index"), ("y", "micrometer"), ("x", "micrometer")),
            dtype="uint8",
            base={"c": 3, "y": 400, "x": 400},
            signal="ebsd_ipf",
            scale={"c": 1.0, "y": 0.5, "x": 0.5},
            levels=2,
            value_range=(0, 255),
            channels=(
                ChannelSpec("IPF-R", "FF0000"),
                ChannelSpec("IPF-G", "00FF00"),
                ChannelSpec("IPF-B", "0000FF"),
            ),
            notes="Grain map; RGB IPF colouring stored as 3 channels (uint8).",
        ),
        StoreSpec(
            domain="materials",
            modality="afm_topography",
            title="AFM surface topography",
            instrument="atomic force microscope",
            axes=(("y", "micrometer"), ("x", "micrometer")),
            dtype="float32",
            base={"y": 256, "x": 256},
            signal="afm_height",
            scale={"y": 0.02, "x": 0.02},
            value_range=(0.0, 3.0e-7),  # metres of height
            emit_omero=False,
            notes="Single-channel height field, nanometre-scale z-values.",
        ),
        StoreSpec(
            domain="materials",
            modality="4dstem_virtual",
            title="4D-STEM virtual-detector images",
            instrument="scanning transmission electron microscope (pixelated detector)",
            axes=(("c", "index"), ("y", "nanometer"), ("x", "nanometer")),
            dtype="float32",
            base={"c": 3, "y": 256, "x": 256},
            signal="diffraction",
            scale={"c": 1.0, "y": 0.8, "x": 0.8},
            value_range=(0.0, 1.0),
            channels=(
                ChannelSpec("bright-field", "FFFFFF"),
                ChannelSpec("annular-DF", "FFAA00"),
                ChannelSpec("HAADF", "00AAFF"),
            ),
            ngff_version="0.5",
            zarr_version=3,
            notes="NGFF 0.5 / Zarr v3 with dimension_names binding.",
        ),
        StoreSpec(
            domain="materials",
            modality="insitu_graingrowth",
            title="In-situ heating grain-growth time series",
            instrument="in-situ SEM / hot-stage optical microscopy",
            axes=(("t", "second"), ("y", "micrometer"), ("x", "micrometer")),
            dtype="uint8",
            base={"t": 12, "y": 320, "x": 320},
            signal="labels",
            scale={"t": 30.0, "y": 0.7, "x": 0.7},
            value_range=(0, 255),
            emit_omero=False,
            notes="Time axis in seconds; grain evolution.",
        ),
    ]

    # ------------------------------------------------------------------- biology
    specs += [
        StoreSpec(
            domain="biology",
            modality="confocal_4ch_zstack",
            title="Confocal 4-channel z-stack",
            instrument="laser-scanning confocal microscope",
            axes=(("c", "index"), ("z", "micrometer"), ("y", "micrometer"), ("x", "micrometer")),
            dtype="uint16",
            base={"c": 4, "z": 16, "y": 384, "x": 384},
            signal="fluor_puncta",
            scale={"c": 1.0, "z": 0.5, "y": 0.16, "x": 0.16},
            levels=3,
            value_range=(0, 4095),
            channels=_fluor_channels(),
            notes="Classic multichannel confocal volume (12-bit).",
        ),
        StoreSpec(
            domain="biology",
            modality="livecell_timelapse",
            title="Live-cell 2-channel timelapse",
            instrument="widefield fluorescence microscope",
            axes=(("t", "minute"), ("c", "index"), ("y", "micrometer"), ("x", "micrometer")),
            dtype="uint16",
            base={"t": 20, "c": 2, "y": 320, "x": 320},
            signal="fluor_puncta",
            scale={"t": 5.0, "c": 1.0, "y": 0.32, "x": 0.32},
            levels=2,
            value_range=(0, 4095),
            channels=(ChannelSpec("GFP", "00FF00"), ChannelSpec("mCherry", "FF3366")),
            ngff_version="0.5",
            zarr_version=3,
            notes="5D TCZYX-minus-Z (TCYX) time series, NGFF 0.5.",
        ),
        StoreSpec(
            domain="biology",
            modality="histology_he_rgb",
            title="H&E whole-slide histology (RGB)",
            instrument="brightfield slide scanner",
            axes=(("y", "micrometer"), ("x", "micrometer"), ("c", "index")),
            dtype="uint8",
            base={"y": 2560, "x": 2560, "c": 3},
            signal="rgb_tissue",
            scale={"y": 0.25, "x": 0.25, "c": 1.0},
            levels=4,
            value_range=(0, 255),
            channels=(
                ChannelSpec("R", "FF0000"),
                ChannelSpec("G", "00FF00"),
                ChannelSpec("B", "0000FF"),
            ),
            notes="Channels-last (YXC) arbitrary axis order; 4-level pyramid; gigapixel-style tiling.",
        ),
        StoreSpec(
            domain="biology",
            modality="calcium_imaging",
            title="Neuronal calcium-imaging timelapse",
            instrument="two-photon calcium imaging (GCaMP)",
            axes=(("t", "second"), ("y", "micrometer"), ("x", "micrometer")),
            dtype="uint16",
            base={"t": 30, "y": 256, "x": 256},
            signal="calcium",
            scale={"t": 0.5, "y": 1.0, "x": 1.0},
            value_range=(0, 3000),
            emit_omero=False,
            notes="Single-channel functional time series.",
        ),
        StoreSpec(
            domain="biology",
            modality="tem_grayscale",
            title="TEM micrograph",
            instrument="transmission electron microscope",
            axes=(("y", "nanometer"), ("x", "nanometer")),
            dtype="uint8",
            base={"y": 512, "x": 512},
            signal="em_grayscale",
            scale={"y": 0.5, "x": 0.5},
            levels=2,
            value_range=(0, 255),
            emit_omero=False,
            notes="Single-channel grayscale, nanometre pixels.",
        ),
        StoreSpec(
            domain="biology",
            modality="segmentation_labels",
            title="3D nuclei segmentation label mask",
            instrument="derived (segmentation of a light-sheet volume)",
            axes=(("z", "micrometer"), ("y", "micrometer"), ("x", "micrometer")),
            dtype="uint32",
            base={"z": 12, "y": 320, "x": 320},
            signal="labels",
            scale={"z": 1.0, "y": 0.4, "x": 0.4},
            value_range=(0, 0),  # pass-through labels
            emit_omero=False,
            notes="Integer label volume (uint32); no omero.",
        ),
        StoreSpec(
            domain="biology",
            modality="spatial_transcriptomics",
            title="Spatial-transcriptomics gene-expression stack",
            instrument="imaging-based spatial transcriptomics (MERFISH-like)",
            axes=(("c", "index"), ("y", "micrometer"), ("x", "micrometer")),
            dtype="float32",
            base={"c": 20, "y": 256, "x": 256},
            signal="spectral_bands",
            scale={"c": 1.0, "y": 0.11, "x": 0.11},
            value_range=(0.0, 12.0),
            emit_omero=False,
            notes="20 gene channels as float32 expression; no omero (too many).",
        ),
    ]

    # -------------------------------------------------------------- environmental
    specs += [
        StoreSpec(
            domain="environmental",
            modality="sentinel2_multispectral",
            title="Sentinel-2-style 12-band multispectral scene",
            instrument="spaceborne multispectral imager",
            axes=(("c", "index"), ("y", "meter"), ("x", "meter")),
            dtype="uint16",
            base={"c": 12, "y": 512, "x": 512},
            signal="spectral_bands",
            scale={"c": 1.0, "y": 10.0, "x": 10.0},
            levels=3,
            value_range=(0, 10000),
            channels=_sentinel_bands(),
            translation={"y": 4.20e6, "x": 5.40e5},
            notes="Reflectance*10000 in uint16; UTM-style translation; metre GSD.",
        ),
        StoreSpec(
            domain="environmental",
            modality="hyperspectral_cube",
            title="Airborne hyperspectral reflectance cube",
            instrument="AVIRIS-style imaging spectrometer",
            axes=(("c", "nanometer"), ("y", "meter"), ("x", "meter")),
            dtype="float32",
            base={"c": 120, "y": 128, "x": 128},
            signal="spectral_bands",
            scale={"c": 4.2, "y": 3.0, "x": 3.0},
            value_range=(0.0, 1.0),
            emit_omero=False,
            notes="120 contiguous wavelength bands (c, nm); float reflectance.",
        ),
        StoreSpec(
            domain="environmental",
            modality="climate_reanalysis",
            title="Climate reanalysis 2-m temperature field",
            instrument="numerical weather reanalysis (ERA5-like)",
            axes=(("t", "hour"), ("y", "degree"), ("x", "degree")),
            dtype="float32",
            base={"t": 24, "y": 181, "x": 360},
            signal="climate_field",
            scale={"t": 1.0, "y": 1.0, "x": 1.0},
            value_range=(233.0, 313.0),  # Kelvin
            translation={"y": -90.0, "x": -180.0},
            emit_omero=False,
            ngff_version="0.5",
            zarr_version=3,
            notes="Global lat/lon grid over 24 hours; degrees; NGFF 0.5.",
        ),
        StoreSpec(
            domain="environmental",
            modality="weather_radar_volume",
            title="Weather-radar reflectivity volume",
            instrument="dual-pol Doppler weather radar",
            axes=(("z", "meter"), ("y", "meter"), ("x", "meter")),
            dtype="float32",
            base={"z": 10, "y": 320, "x": 320},
            signal="radar_volume",
            scale={"z": 500.0, "y": 1000.0, "x": 1000.0},
            value_range=(-15.0, 70.0),  # dBZ
            emit_omero=False,
            notes="Stacked elevation scans as a z-volume; dBZ.",
        ),
        StoreSpec(
            domain="environmental",
            modality="lidar_canopy_dem",
            title="LiDAR canopy-height / terrain model",
            instrument="airborne LiDAR (derived raster)",
            axes=(("y", "meter"), ("x", "meter")),
            dtype="float32",
            base={"y": 512, "x": 512},
            signal="dem",
            scale={"y": 1.0, "x": 1.0},
            levels=3,
            value_range=(0.0, 45.0),  # metres height
            emit_omero=False,
            translation={"y": 4.10e6, "x": 5.55e5},
            notes="Single-band elevation raster with projected translation.",
        ),
    ]

    # -------------------------------------------------------------------- medical
    specs += [
        StoreSpec(
            domain="medical",
            modality="ct_thorax",
            title="Thoracic CT volume (Hounsfield)",
            instrument="clinical X-ray CT scanner",
            axes=(("z", "millimeter"), ("y", "millimeter"), ("x", "millimeter")),
            dtype="int16",
            base={"z": 20, "y": 512, "x": 512},
            signal="ct_volume",
            scale={"z": 1.5, "y": 0.7, "x": 0.7},
            levels=3,
            value_range=(-1024, 3071),
            emit_omero=False,
            notes="Signed int16 Hounsfield units; anisotropic z spacing.",
        ),
        StoreSpec(
            domain="medical",
            modality="mri_multisequence",
            title="Brain MRI multi-sequence volume",
            instrument="clinical MRI scanner",
            axes=(("c", "index"), ("z", "millimeter"), ("y", "millimeter"), ("x", "millimeter")),
            dtype="uint16",
            base={"c": 4, "z": 18, "y": 256, "x": 256},
            signal="mri_sequences",
            scale={"c": 1.0, "z": 1.0, "y": 0.9, "x": 0.9},
            levels=2,
            value_range=(0, 4095),
            channels=_mri_channels(),
            notes="T1/T2/FLAIR/DWI as channels; CZYX.",
        ),
        StoreSpec(
            domain="medical",
            modality="ultrasound_cine",
            title="Cardiac ultrasound cine loop",
            instrument="B-mode ultrasound",
            axes=(("t", "second"), ("y", "millimeter"), ("x", "millimeter")),
            dtype="uint8",
            base={"t": 30, "y": 384, "x": 384},
            signal="calcium",
            scale={"t": 0.033, "y": 0.2, "x": 0.2},
            value_range=(0, 255),
            emit_omero=False,
            notes="Grayscale cine; 30 Hz frame cadence.",
        ),
    ]

    # ------------------------------------------------------------------ geophysics
    specs += [
        StoreSpec(
            domain="geophysics",
            modality="seismic_volume",
            title="3-D seismic reflection amplitude volume",
            instrument="reflection seismic survey",
            axes=(("z", "millisecond"), ("y", "meter"), ("x", "meter")),
            dtype="float32",
            base={"z": 24, "y": 384, "x": 384},
            signal="seismic",
            scale={"z": 4.0, "y": 25.0, "x": 25.0},
            levels=2,
            value_range=(-1.0, 1.0),
            emit_omero=False,
            translation={"y": 6.20e6, "x": 4.30e5},
            notes="Two-way-time (ms) as z; inline/crossline in metres; signed amplitude.",
        ),
        StoreSpec(
            domain="geophysics",
            modality="gpr_bscan_timelapse",
            title="Ground-penetrating-radar B-scan survey",
            instrument="ground-penetrating radar",
            axes=(("t", "index"), ("y", "nanosecond"), ("x", "meter")),
            dtype="float32",
            base={"t": 8, "y": 384, "x": 512},
            signal="seismic",
            scale={"t": 1.0, "y": 0.2, "x": 0.05},
            value_range=(-1.0, 1.0),
            emit_omero=False,
            notes="Survey-line index as t; travel-time (ns) as y.",
        ),
    ]

    # ------------------------------------------------------------------ astronomy
    specs += [
        StoreSpec(
            domain="astronomy",
            modality="radio_spectral_cube",
            title="Radio-interferometer spectral cube",
            instrument="ALMA/VLA-style radio interferometer",
            axes=(("c", "hertz"), ("y", "degree"), ("x", "degree")),
            dtype="float32",
            base={"c": 32, "y": 256, "x": 256},
            signal="spectral_bands",
            scale={"c": 3.9e6, "y": 2.7e-4, "x": 2.7e-4},
            value_range=(0.0, 1.0),
            emit_omero=False,
            translation={"y": -30.0, "x": 187.7},
            notes="Frequency channels (Hz) as c; sky coords (deg) as y/x.",
        ),
        StoreSpec(
            domain="astronomy",
            modality="optical_survey_ugriz",
            title="Optical multi-band survey cutout (ugriz)",
            instrument="ground-based optical survey camera",
            axes=(("c", "index"), ("y", "degree"), ("x", "degree")),
            dtype="float32",
            base={"c": 5, "y": 384, "x": 384},
            signal="spectral_bands",
            scale={"c": 1.0, "y": 7.3e-5, "x": 7.3e-5},
            value_range=(0.0, 1.0),
            channels=(
                ChannelSpec("u", "9933FF"),
                ChannelSpec("g", "33CC33"),
                ChannelSpec("r", "FFCC00"),
                ChannelSpec("i", "FF6600"),
                ChannelSpec("z", "CC0000"),
            ),
            notes="Five broadband filters as channels; arcsecond-scale pixels.",
        ),
        StoreSpec(
            domain="astronomy",
            modality="solar_euv_timelapse",
            title="Solar EUV multi-wavelength timelapse",
            instrument="space solar observatory (SDO/AIA-like)",
            axes=(("t", "second"), ("c", "angstrom"), ("y", "arcsec"), ("x", "arcsec")),
            dtype="uint16",
            base={"t": 8, "c": 3, "y": 256, "x": 256},
            signal="radar_volume",
            scale={"t": 12.0, "c": 1.0, "y": 0.6, "x": 0.6},
            levels=2,
            value_range=(0, 16000),
            channels=(
                ChannelSpec("171A", "FFCC00"),
                ChannelSpec("193A", "CC6600"),
                ChannelSpec("211A", "AA00AA"),
            ),
            ngff_version="0.5",
            zarr_version=3,
            notes="Full 5D TCZYX-minus-Z (TCYX); NGFF 0.5.",
        ),
    ]

    # ----------------------------------------------------------------------- other
    specs += [
        StoreSpec(
            domain="other",
            modality="audio_spectrogram",
            title="Acoustic sensor spectrogram",
            instrument="microphone / hydrophone array (STFT)",
            axes=(("y", "hertz"), ("x", "second")),
            dtype="float32",
            base={"y": 256, "x": 512},
            signal="spectrogram",
            scale={"y": 86.13, "x": 0.011},
            value_range=(-80.0, 0.0),  # dB
            emit_omero=False,
            notes="Non-image sensor stream: frequency (y, Hz) x time (x, s); power in dB.",
        ),
        StoreSpec(
            domain="other",
            modality="ir_thermography",
            title="Infrared thermography timelapse",
            instrument="uncooled microbolometer IR camera",
            axes=(("t", "second"), ("y", "millimeter"), ("x", "millimeter")),
            dtype="float32",
            base={"t": 16, "y": 240, "x": 320},
            signal="thermal",
            scale={"t": 2.0, "y": 1.0, "x": 1.0},
            value_range=(18.0, 85.0),  # Celsius
            emit_omero=False,
            notes="Temperature (deg C) field cooling over time.",
        ),
        StoreSpec(
            domain="other",
            modality="msi_maldi",
            title="MALDI mass-spectrometry imaging",
            instrument="MALDI imaging mass spectrometer",
            axes=(("c", "thomson"), ("y", "micrometer"), ("x", "micrometer")),
            dtype="float32",
            base={"c": 16, "y": 200, "x": 200},
            signal="spectral_bands",
            scale={"c": 1.0, "y": 20.0, "x": 20.0},
            value_range=(0.0, 5.0e5),
            emit_omero=False,
            notes="m/z bins as channels; ion intensity as float.",
        ),
    ]

    return specs
