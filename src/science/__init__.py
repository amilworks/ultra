"""Science-specific libraries (statistics, reporting, domain adapters)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "summary_statistics": ("src.science.stats", "summary_statistics"),
    "confidence_interval_mean": ("src.science.stats", "confidence_interval_mean"),
    "bootstrap_confidence_interval": ("src.science.stats", "bootstrap_confidence_interval"),
    "cohen_d": ("src.science.stats", "cohen_d"),
    "cliffs_delta": ("src.science.stats", "cliffs_delta"),
    "compare_two_groups": ("src.science.stats", "compare_two_groups"),
    "list_curated_stat_tools": ("src.science.stats", "list_curated_stat_tools"),
    "run_stat_tool": ("src.science.stats", "run_stat_tool"),
    "generate_repro_report": ("src.science.reporting", "generate_repro_report"),
    "load_scientific_image": ("src.science.imaging", "load_scientific_image"),
    "segment_array_with_medsam2": ("src.science.medsam2", "segment_array_with_medsam2"),
    "segment_array_with_sam3": ("src.science.sam3", "segment_array_with_sam3"),
    "segment_array_with_sam3_concept": ("src.science.sam3", "segment_array_with_sam3_concept"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
