"""Vision-reasoner tooling: a host-side bridge to the Qwen3.6-27B VLM.

The vision-reasoner subagent orchestrates with the inherited text model and calls
``inspect_images`` to actually look at pixels. The tool loads/crops/resizes images
host-side (the api key never enters the network-none sandbox), sends them to the
on-prem Qwen VLM with thinking on, and returns the model's analysis + reasoning.
"""

from ultra_deepagents.vision.tools import build_vision_tools

__all__ = ["build_vision_tools"]
