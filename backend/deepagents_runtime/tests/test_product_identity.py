"""The agent must know who made it, and must not be free to invent an answer.

Background: asked "who created BisQue Ultra?", the agent produced a confident,
plausible, entirely invented name. It had no self-knowledge in its system prompt,
and the anti-fabrication guidance it did have was written about images, files, and
datasets — so naming a person never registered as the same kind of error.

These tests pin both halves of the fix, and pin ABOUT.md as the single source of
truth so the prompt cannot quietly drift away from it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ultra_deepagents.agent import (
    PRODUCT_IDENTITY_GUIDANCE,
    SYSTEM_PROMPT,
    _GROUNDING_SYSTEM_GUIDANCE,
)

# tests/ -> deepagents_runtime/ -> backend/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
ABOUT_PATH = _REPO_ROOT / "ABOUT.md"

# Every fact the assistant is allowed to state about its own provenance. Each entry
# must appear verbatim in BOTH ABOUT.md and the injected identity block, so the two
# can never disagree about a name, a title, or an address.
CANONICAL_FACTS = (
    "Amil Khan",
    "Electrical and Computer Engineering",
    "University of California, Santa Barbara",
    "B.S. Manjunath",
    "Vision Research Lab",
    "amil@ucsb.edu",
    "https://github.com/amilworks/ultra",
    "https://amilworks.github.io/ultra_website/",
)


def test_no_release_version_baked_into_the_prompt() -> None:
    """Release numbers go stale the moment a new one ships.

    ABOUT.md may name the current release for human readers, but the prompt is baked
    into the worker image and would keep asserting an old version long after it was
    superseded. The block must point at the website instead.
    """
    assert not re.search(r"\b20\d{2}\.\d{2}\b", PRODUCT_IDENTITY_GUIDANCE), (
        "A release version leaked into the injected identity block. Remove it — the "
        "website is the live source for release news."
    )
    assert "ultra_website" in PRODUCT_IDENTITY_GUIDANCE


def test_about_file_exists() -> None:
    assert ABOUT_PATH.is_file(), (
        f"{ABOUT_PATH} is the canonical source of product-identity facts and is "
        "referenced by the agent's system prompt; it must not be deleted or moved "
        "without updating PRODUCT_IDENTITY_GUIDANCE."
    )


@pytest.mark.parametrize("fact", CANONICAL_FACTS)
def test_fact_present_in_about_file(fact: str) -> None:
    assert fact in ABOUT_PATH.read_text(encoding="utf-8"), (
        f"{fact!r} is asserted by the agent's identity block but is missing from "
        "ABOUT.md. Fix ABOUT.md, or drop the claim from PRODUCT_IDENTITY_GUIDANCE — "
        "the assistant must not state a fact this repo does not record."
    )


@pytest.mark.parametrize("fact", CANONICAL_FACTS)
def test_fact_present_in_identity_block(fact: str) -> None:
    assert fact in PRODUCT_IDENTITY_GUIDANCE, (
        f"{fact!r} is recorded in ABOUT.md but never reaches the model. Facts only "
        "prevent fabrication if they are in the prompt."
    )


def test_identity_block_reaches_the_system_prompt() -> None:
    """A constant that is defined but never interpolated changes nothing."""
    assert PRODUCT_IDENTITY_GUIDANCE in SYSTEM_PROMPT


def test_creator_attribution_is_unambiguous() -> None:
    """'created by Amil Khan' must survive rewording of the surrounding block."""
    assert re.search(r"created by Amil Khan", PRODUCT_IDENTITY_GUIDANCE)


def test_identity_block_tells_the_model_what_to_do_when_it_does_not_know() -> None:
    """Facts cover the common questions; this covers every other one.

    Without an explicit escape hatch the model still has to produce *something* for
    "who else works on Ultra?" — and a plausible guess is exactly the failure mode.
    """
    lowered = PRODUCT_IDENTITY_GUIDANCE.lower()
    assert "do not know" in lowered or "don't know" in lowered
    assert "amil@ucsb.edu" in lowered


def test_grounding_guidance_covers_people_not_only_data() -> None:
    """The original guidance enumerated images, video, datasets, and files only.

    That scoping is precisely why naming a person slipped through: the model was not
    describing an attached resource, so none of the rules felt applicable.
    """
    lowered = _GROUNDING_SYSTEM_GUIDANCE.lower()
    assert "person's name" in lowered, (
        "Grounding guidance no longer names people explicitly. Without that, "
        "provenance questions fall outside every rule in the block and the "
        "invented-collaborator bug returns."
    )
    for concept in ("affiliation", "authorship"):
        assert concept in lowered, f"grounding guidance lost coverage of {concept!r}"


def test_grounding_guidance_still_covers_resources() -> None:
    """Guard against a future edit trading data grounding for people grounding."""
    lowered = _GROUNDING_SYSTEM_GUIDANCE.lower()
    for concept in ("image", "dataset", "file"):
        assert concept in lowered
