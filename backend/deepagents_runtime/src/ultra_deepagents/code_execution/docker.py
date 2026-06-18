from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from ultra_deepagents.code_execution.paths import resolve_workspace_file

MATPLOTLIBRC = """\
backend: Agg
figure.dpi: 300
savefig.dpi: 300
"""


@dataclass(frozen=True)
class DockerSandboxConfig:
    image: str
    network: str = "none"
    cpus: float = 0.0
    memory: str = ""
    pids_limit: int = 0
    timeout_seconds: int = 0
    output_limit_bytes: int = 0


class DockerSandboxBackend(BaseSandbox):
    """Deep Agents sandbox backend that executes commands in isolated Docker containers."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path,
        config: DockerSandboxConfig,
        outputs_dir: str | Path | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.outputs_dir = Path(outputs_dir) if outputs_dir is not None else None
        self.config = config
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        if self.outputs_dir is not None:
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_matplotlib_config()
        try:
            os.chmod(self.workspace_dir, 0o777)
            if self.outputs_dir is not None:
                os.chmod(self.outputs_dir, 0o777)
        except OSError:
            pass

    @property
    def id(self) -> str:
        return f"docker:{self.workspace_dir.resolve()}"

    def _ensure_matplotlib_config(self) -> None:
        config_dir = self.workspace_dir / ".cache" / "matplotlib"
        config_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.workspace_dir / "matplotlibrc", config_dir / "matplotlibrc"):
            path.write_text(MATPLOTLIBRC)
        for path in (self.workspace_dir / ".cache", config_dir):
            try:
                os.chmod(path, 0o777)
            except OSError:
                pass

    def build_docker_command(self, command: str) -> list[str]:
        workspace_mount = f"{self.workspace_dir.resolve()}:/workspace:rw"
        docker_command = [
            "docker",
            "run",
            "--rm",
            "--network",
            self.config.network,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=512m",
            "--volume",
            workspace_mount,
            "--workdir",
            "/workspace",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "MPLCONFIGDIR=/workspace/.cache/matplotlib",
            "--env",
            "XDG_CACHE_HOME=/workspace/.cache",
        ]
        if self.outputs_dir is not None:
            docker_command.extend(
                [
                    "--volume",
                    f"{self.outputs_dir.resolve()}:/outputs:rw",
                ]
            )
        if self.config.cpus > 0:
            docker_command.extend(["--cpus", str(self.config.cpus)])
        if self.config.memory.strip():
            docker_command.extend(["--memory", self.config.memory])
        if self.config.pids_limit > 0:
            docker_command.extend(["--pids-limit", str(self.config.pids_limit)])
        docker_command.extend([self.config.image, "bash", "-lc", command])
        return docker_command

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        violation = validate_sandbox_command(command)
        if violation is not None:
            return ExecuteResponse(output=violation, exit_code=126)

        _ = timeout
        timeout_seconds = self.config.timeout_seconds
        try:
            completed = subprocess.run(
                self.build_docker_command(command),
                capture_output=True,
                text=True,
                timeout=timeout_seconds if timeout_seconds > 0 else None,
                check=False,
            )
        except FileNotFoundError:
            return ExecuteResponse(output="Docker executable not found.", exit_code=127)
        except subprocess.TimeoutExpired as exc:
            output = _combine_output(exc.stdout, exc.stderr)
            truncated_output, truncated = _truncate_output(
                f"Command timed out after {timeout_seconds} seconds.\n{output}",
                self.config.output_limit_bytes,
            )
            return ExecuteResponse(
                output=truncated_output,
                exit_code=124,
                truncated=truncated,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            # A host-level launch failure (PermissionError on the docker binary,
            # ENOMEM/EAGAIN on fork, etc.) must surface as a structured result the
            # model can react to, not an unhandled graph error.
            return ExecuteResponse(
                output=f"Sandbox launch failed: {exc}",
                exit_code=127,
            )

        output, truncated = _truncate_output(
            _combine_output(completed.stdout, completed.stderr),
            self.config.output_limit_bytes,
        )
        return ExecuteResponse(
            output=output,
            exit_code=completed.returncode,
            truncated=truncated,
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for requested_path, content in files:
            try:
                target = resolve_workspace_file(self.workspace_dir, requested_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            except ValueError:
                responses.append(FileUploadResponse(path=requested_path, error="permission_denied"))
            except OSError as exc:
                responses.append(FileUploadResponse(path=requested_path, error=str(exc)))
            else:
                responses.append(FileUploadResponse(path=requested_path, error=None))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for requested_path in paths:
            try:
                target = resolve_workspace_file(self.workspace_dir, requested_path)
                if not target.exists():
                    responses.append(
                        FileDownloadResponse(path=requested_path, error="file_not_found")
                    )
                    continue
                if not target.is_file():
                    responses.append(FileDownloadResponse(path=requested_path, error="not_a_file"))
                    continue
                content = target.read_bytes()
            except ValueError:
                responses.append(
                    FileDownloadResponse(path=requested_path, error="permission_denied")
                )
            except OSError as exc:
                responses.append(FileDownloadResponse(path=requested_path, error=str(exc)))
            else:
                responses.append(
                    FileDownloadResponse(path=requested_path, content=content, error=None)
                )
        return responses


def validate_sandbox_command(command: str) -> str | None:
    normalized = " ".join(str(command or "").split())
    if not normalized:
        return "Command is empty."
    if _uses_shell_timeout_wrapper(normalized):
        return _shell_timeout_message()
    if _searches_root_with_python_glob(normalized):
        return _root_search_message()
    if re.search(r"\bfind\s+/(?:\s|$)", normalized):
        return _root_search_message()
    if re.search(r"\bfind\s+/(?!workspace(?:/|\s|$))", normalized):
        return _root_search_message()
    if re.search(r"\bos\.walk\(\s*['\"]/\s*['\"]\s*\)", normalized):
        return _root_search_message()
    if re.search(r"Path\(\s*['\"]/\s*['\"]\s*\)\.rglob\(", normalized):
        return _root_search_message()
    if re.search(r"\b(?:grep|rg)\b[^;&|]*\s+/(?:\s|$)", normalized) and "-R" in normalized:
        return _root_search_message()
    return None


def _uses_shell_timeout_wrapper(command: str) -> bool:
    return bool(re.search(r"(?:^|(?:&&|\|\||;|\|)\s*)g?timeout\b", command))


def _searches_root_with_python_glob(command: str) -> bool:
    if "glob" not in command and "rglob" not in command:
        return False
    return bool(re.search(r"['\"]/\*\*", command))


def _root_search_message() -> str:
    return (
        "Recursive searches must stay under /workspace. Uploaded files are app "
        "storage handles; stage exact IDs with stage_uploaded_files_for_analysis "
        "before reading them, and use /workspace paths only."
    )


def _shell_timeout_message() -> str:
    return (
        "Do not wrap sandbox commands with shell timeout. Long-running analysis is "
        "allowed; run the command directly and let the platform's operator-configured "
        "sandbox policy handle any hard limit."
    )


def _combine_output(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    parts = []
    for value in (stdout, stderr):
        if isinstance(value, bytes):
            parts.append(value.decode("utf-8", errors="replace"))
        elif value:
            parts.append(value)
    return "".join(parts)


def _truncate_output(output: str, limit_bytes: int) -> tuple[str, bool]:
    if limit_bytes <= 0:
        return output, False
    encoded = output.encode("utf-8", errors="replace")
    if len(encoded) <= limit_bytes:
        return output, False
    truncated = encoded[:limit_bytes].decode("utf-8", errors="replace")
    return f"{truncated}\n...[truncated]", True
