"""Science-specific libraries (statistics, reporting, domain adapters)."""

from src.science.imaging import load_scientific_image
from src.science.medsam2 import segment_array_with_medsam2
from src.science.sam3 import segment_array_with_sam3, segment_array_with_sam3_concept
from src.science.reporting import generate_repro_report
from src.science.stats import (
    bootstrap_confidence_interval,
    cohen_d,
    compare_two_groups,
    confidence_interval_mean,
    cliffs_delta,
    list_curated_stat_tools,
    run_stat_tool,
    summary_statistics,
)

__all__ = [
    "summary_statistics",
    "confidence_interval_mean",
    "bootstrap_confidence_interval",
    "cohen_d",
    "cliffs_delta",
    "compare_two_groups",
    "list_curated_stat_tools",
    "run_stat_tool",
    "generate_repro_report",
    "load_scientific_image",
    "segment_array_with_medsam2",
    "segment_array_with_sam3",
    "segment_array_with_sam3_concept",
]
