"""Compatibility shim for YOLOv5's legacy pkg_resources usage.

The vendored YOLOv5 runtime expects ``pkg_resources`` from setuptools, but our
runtime environment may not include that module. This shim provides the small
subset YOLOv5 uses via ``importlib.metadata`` and ``packaging`` so prairie
inference can run without depending on setuptools internals.
"""

from __future__ import annotations

from types import SimpleNamespace

try:  # pragma: no cover - prefer the original implementation when available
    import pkg_resources as pkg  # type: ignore
except Exception:  # pragma: no cover - exercised in runtime when setuptools is absent
    from importlib.metadata import PackageNotFoundError, version as distribution_version

    from packaging.requirements import Requirement
    from packaging.version import parse as parse_version

    class DistributionNotFound(Exception):
        """Raised when a required distribution is not installed."""

    class VersionConflict(Exception):
        """Raised when an installed distribution misses the required specifier."""

    def parse_requirements(requirements):
        if hasattr(requirements, "read"):
            raw_requirements = requirements.read().splitlines()
        else:
            raw_requirements = requirements
        for raw_requirement in raw_requirements:
            line = str(raw_requirement or "").strip()
            if not line or line.startswith("#"):
                continue
            if " #" in line:
                line = line.split(" #", 1)[0].strip()
            if not line:
                continue
            yield Requirement(line)

    def require(requirement_text: str):
        requirement = Requirement(str(requirement_text))
        try:
            installed_version = distribution_version(requirement.name)
        except PackageNotFoundError as exc:
            raise DistributionNotFound(str(exc)) from exc
        if requirement.specifier and not requirement.specifier.contains(installed_version, prereleases=True):
            raise VersionConflict(
                f"{requirement.name} {installed_version} does not satisfy {requirement.specifier}"
            )
        return [requirement]

    pkg = SimpleNamespace(
        parse_version=parse_version,
        parse_requirements=parse_requirements,
        require=require,
        VersionConflict=VersionConflict,
        DistributionNotFound=DistributionNotFound,
    )

__all__ = ["pkg"]
