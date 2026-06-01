from __future__ import annotations

from ultra_deepagents.prompt_cases import PLOT_WORKFLOW_SMOKE_CASES


def test_real_plot_workflow_prompt_cases_cover_inline_captions_and_matplotlib_updates() -> None:
    case_by_id = {case.case_id: case for case in PLOT_WORKFLOW_SMOKE_CASES}

    bubble_sort = case_by_id["bubble_sort_inline_plots"]
    assert (
        "Write code to show how bubble sort works and create some plots to show how the code works."
        == bubble_sort.prompt
    )
    assert bubble_sort.requires_code_execution is True
    assert bubble_sort.requires_inline_captions is True

    neural_net = case_by_id["dummy_neural_network_training_plots"]
    assert "training a simple neural network on dummy data" in neural_net.prompt
    assert neural_net.requires_inline_captions is True

    follow_up = case_by_id["matplotlib_error_bars_followup"]
    assert follow_up.prompt == "Change the plot to include error bars"
    assert follow_up.is_follow_up is True
    assert follow_up.requires_existing_plot_update is True
