from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSettings:
    openai_base_url: str
    openai_model: str
    openai_api_key: str = "EMPTY"
    request_timeout_seconds: float = 120.0
    max_retries: int = 1
    sandbox_image: str = "bisque-ultra-codeexec:py311"
    sandbox_network: str = "none"
    sandbox_cpus: float = 2.0
    sandbox_memory: str = "4g"
    sandbox_pids_limit: int = 256
    sandbox_timeout_seconds: int = 900
    sandbox_output_limit_bytes: int = 200_000

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            openai_base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "deepseek_v4"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
            request_timeout_seconds=float(os.getenv("ULTRA_DEEPAGENTS_TIMEOUT_SECONDS", "120")),
            max_retries=int(os.getenv("ULTRA_DEEPAGENTS_MAX_RETRIES", "1")),
            sandbox_image=os.getenv(
                "ULTRA_DEEPAGENTS_SANDBOX_IMAGE",
                "bisque-ultra-codeexec:py311",
            ),
            sandbox_network=os.getenv("ULTRA_DEEPAGENTS_SANDBOX_NETWORK", "none"),
            sandbox_cpus=float(os.getenv("ULTRA_DEEPAGENTS_SANDBOX_CPUS", "2.0")),
            sandbox_memory=os.getenv("ULTRA_DEEPAGENTS_SANDBOX_MEMORY", "4g"),
            sandbox_pids_limit=int(os.getenv("ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT", "256")),
            sandbox_timeout_seconds=int(
                os.getenv("ULTRA_DEEPAGENTS_SANDBOX_TIMEOUT_SECONDS", "900")
            ),
            sandbox_output_limit_bytes=int(
                os.getenv("ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES", "200000")
            ),
        )
