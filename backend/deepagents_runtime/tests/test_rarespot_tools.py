"""Tests for the surviving RareSpot helpers after the dispatch tool was retired.

The ``rarespot_ecology_inference`` dispatch tool (and its create/wait/auth plumbing)
was replaced by the ``prairie-dog-detection`` Skill, so what remains to cover is:
  * ``looks_report_only_rarespot_goal`` — the report-only follow-up classifier the
    agent still uses to avoid re-running inference on synthesis requests; and
  * ``summarize_detection_confidences`` — the class-wise confidence summary the
    inference pipeline (now run by the Skill) produces.
"""

from ultra_deepagents.rarespot.inference import summarize_detection_confidences
from ultra_deepagents.rarespot.tools import looks_report_only_rarespot_goal


def test_report_only_synthesis_goal_is_classified_report_only():
    assert looks_report_only_rarespot_goal(
        "Write a combined quantitative report across all RareSpot runs in this chat. "
        "Include thresholds, counts by class, artifact links, and limitations."
    )


def test_negated_rerun_goal_is_classified_report_only():
    assert looks_report_only_rarespot_goal(
        "Write a combined quantitative report across the RareSpot runs in this chat. "
        "Do not rerun inference; use the prior RareSpot outputs."
    )


def test_explicit_new_threshold_goal_is_not_report_only():
    assert not looks_report_only_rarespot_goal(
        "Write a combined report, then run another RareSpot pass at confidence threshold 0.15."
    )


def test_goal_without_rarespot_is_not_report_only():
    assert not looks_report_only_rarespot_goal("Detect prairie dogs and burrows in these survey images.")


def test_summarize_detection_confidences_reports_classwise_stats():
    summary = summarize_detection_confidences(
        [
            {
                "boxes": [
                    {"class_name": "prairie_dog", "confidence": 0.5},
                    {"class_name": "prairie_dog", "confidence": 0.7},
                    {"class_name": "burrow", "confidence": 0.9},
                ]
            },
            {"boxes": [{"class_name": "prairie_dog", "confidence": 0.9}]},
        ]
    )

    assert summary == {
        "burrow": {"count": 1, "min": 0.9, "mean": 0.9, "max": 0.9},
        "prairie_dog": {"count": 3, "min": 0.5, "mean": 0.7, "max": 0.9},
    }
