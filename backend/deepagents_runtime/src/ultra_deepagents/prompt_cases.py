from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotWorkflowSmokeCase:
    case_id: str
    prompt: str
    requires_code_execution: bool = True
    requires_inline_captions: bool = True
    is_follow_up: bool = False
    requires_existing_plot_update: bool = False


PLOT_WORKFLOW_SMOKE_CASES = (
    PlotWorkflowSmokeCase(
        case_id="bubble_sort_inline_plots",
        prompt=(
            "Write code to show how bubble sort works and create some plots to show how the code works."
        ),
    ),
    PlotWorkflowSmokeCase(
        case_id="dummy_neural_network_training_plots",
        prompt=(
            "Show training a simple neural network on dummy data, plot the training loss and "
            "predictions, and explain each plot with its caption inline."
        ),
    ),
    PlotWorkflowSmokeCase(
        case_id="matplotlib_error_bars_followup",
        prompt="Change the plot to include error bars",
        is_follow_up=True,
        requires_existing_plot_update=True,
    ),
)
