from __future__ import annotations

import time

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.model import build_chat_model


def main() -> None:
    settings = RuntimeSettings.from_env()
    model = build_chat_model(settings)
    started = time.perf_counter()
    response = model.invoke("Reply with one short sentence confirming the model is reachable.")
    elapsed_ms = (time.perf_counter() - started) * 1000
    content = str(response.content or "")
    print(
        f"model={settings.openai_model} "
        f"base_url={settings.openai_base_url} "
        f"latency_ms={elapsed_ms:.1f} "
        f"response_chars={len(content)}"
    )


if __name__ == "__main__":
    main()
