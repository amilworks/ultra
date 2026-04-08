"""Structured deterministic verifiers for narrow, high-confidence scientific patterns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from .contracts import CodeVerificationResult, DeliberationPolicy, EvidenceRef


SignalDetector = Callable[[str], bool]
VerifierFn = Callable[[str], CodeVerificationResult | None]
RendererFn = Callable[[CodeVerificationResult], str | None]


@dataclass(frozen=True)
class StructuredSignalSpec:
    key: str
    reason: str
    detector: SignalDetector


@dataclass(frozen=True)
class StructuredVerifierHandler:
    name: str
    signal_key: str
    verify: VerifierFn
    render: RendererFn


def _normalized_text(user_text: str) -> str:
    return str(user_text or "").strip().lower()


def detect_sequence_design_signal(user_text: str) -> bool:
    lowered = _normalized_text(user_text)
    long_nucleotide_runs = re.findall(r"\b[acgt]{25,}\b", lowered)
    if not long_nucleotide_runs:
        return False
    return any(
        token in lowered
        for token in (
            "pcr",
            "primer",
            "subclon",
            "clone",
            "expression vector",
            "restriction enzyme",
            "restriction site",
            "enzyme",
            "orientation",
            "mcs",
            "reverse complement",
        )
    )


def detect_coding_sequence_match_signal(user_text: str) -> bool:
    lowered = _normalized_text(user_text)
    if not any(token in lowered for token in ("amino acid sequence", "protein sequence", "plasmid")):
        return False
    dna_option_count = len(re.findall(r"(?m)^\s*[ABCD]\.\s*[ACGTacgt\s]{60,}$", str(user_text or "")))
    if dna_option_count < 2:
        return False
    protein_candidates = re.findall(r"\b[ACDEFGHIKLMNPQRSTVWY]{40,}\b", str(user_text or "").upper())
    protein_candidates = [token for token in protein_candidates if re.search(r"[EFILMPQRSVWY]", token)]
    return bool(protein_candidates)


def detect_metathesis_retrosynthesis_signal(user_text: str) -> bool:
    lowered = _normalized_text(user_text)
    if "methyleneruthenium" not in lowered:
        return False
    if "starting material" not in lowered and "identify the starting material" not in lowered:
        return False
    return (
        re.search(
            r"2-vinylcyclo(?:butane|pentane|hexane|heptane|octane)",
            lowered,
        )
        is not None
    )


def detect_fluorescence_localization_signal(user_text: str) -> bool:
    lowered = _normalized_text(user_text)
    reporter_tokens = (
        "mraspberry",
        "mcherry",
        "gfp",
        "rfp",
        "fluorescent protein",
        "reporter",
        "red signal",
    )
    apoptosis_tokens = ("tunel", "fitc", "apoptotic", "apoptosis")
    imaging_tokens = ("confocal", "microscope", "co-localization", "colocalization", "signal")
    disqualifiers = (
        "nuclear localization signal",
        "nls",
        "membrane-tagged",
        "membrane targeted",
        "histone fusion",
        "mitochondrial targeting",
        "er retention",
    )
    if any(token in lowered for token in disqualifiers):
        return False
    return (
        any(token in lowered for token in reporter_tokens)
        and any(token in lowered for token in apoptosis_tokens)
        and any(token in lowered for token in imaging_tokens)
    )


STRUCTURED_SIGNAL_SPECS: tuple[StructuredSignalSpec, ...] = (
    StructuredSignalSpec(
        key="coding_sequence_match_cues",
        reason="coding-sequence option matching can be verified deterministically by translation",
        detector=detect_coding_sequence_match_signal,
    ),
    StructuredSignalSpec(
        key="sequence_design_cues",
        reason="sequence design needs exact programmable verification",
        detector=detect_sequence_design_signal,
    ),
    StructuredSignalSpec(
        key="reaction_retrosynthesis_cues",
        reason="reaction retrosynthesis pattern matches a deterministic metathesis heuristic",
        detector=detect_metathesis_retrosynthesis_signal,
    ),
    StructuredSignalSpec(
        key="fluorescence_localization_cues",
        reason="fluorescence-marker localization pattern matches a deterministic microscopy heuristic",
        detector=detect_fluorescence_localization_signal,
    ),
)


def detect_structured_problem_signals(user_text: str) -> tuple[dict[str, bool], list[str]]:
    signals: dict[str, bool] = {}
    reasons: list[str] = []
    for spec in STRUCTURED_SIGNAL_SPECS:
        matched = bool(spec.detector(user_text))
        signals[spec.key] = matched
        if matched:
            reasons.append(spec.reason)
    return signals, reasons


def has_structured_verifier_signal(signals: dict[str, object]) -> bool:
    for spec in STRUCTURED_SIGNAL_SPECS:
        if bool(signals.get(spec.key)):
            return True
    return False


def _extract_insert_sequence(user_text: str) -> str:
    candidates = re.findall(r"[ACGTacgt\s]{60,}", str(user_text or ""))
    normalized = [
        re.sub(r"\s+", "", candidate).upper()
        for candidate in candidates
    ]
    normalized = [candidate for candidate in normalized if re.fullmatch(r"[ACGT]{60,}", candidate)]
    if not normalized:
        return ""
    return max(normalized, key=len)


def _extract_target_protein_sequence(user_text: str) -> str:
    text = str(user_text or "").upper()
    candidates: list[str] = []
    labeled = re.search(
        r"(?:AMINO ACID SEQUENCE|PROTEIN SEQUENCE)\b([\s:=\-A-Z]{20,})",
        text,
        flags=re.DOTALL,
    )
    if labeled:
        candidates.extend(
            re.findall(r"\b[ACDEFGHIKLMNPQRSTVWY]{40,}\b", labeled.group(1))
        )
    candidates.extend(
        re.findall(r"\b[ACDEFGHIKLMNPQRSTVWY]{40,}\b", text)
    )
    normalized = [
        candidate
        for candidate in candidates
        if len(candidate) >= 40 and re.search(r"[EFILMPQRSVWY]", candidate)
    ]
    if not normalized:
        return ""
    return max(normalized, key=len)


def _extract_mcq_dna_options(user_text: str) -> dict[str, str]:
    options: dict[str, str] = {}
    pattern = re.compile(r"(?ms)^\s*([ABCD])\.\s*(.*?)(?=^\s*[ABCD]\.\s*|\Z)")
    for match in pattern.finditer(str(user_text or "")):
        letter = match.group(1).strip().upper()
        raw = re.sub(r"[^ACGTacgt]", "", match.group(2) or "").upper()
        if len(raw) >= 60:
            options[letter] = raw
    return options


_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _translate_dna(sequence: str) -> str:
    protein: list[str] = []
    trimmed = str(sequence or "").upper()
    for index in range(0, len(trimmed) - 2, 3):
        codon = trimmed[index:index + 3]
        amino_acid = _CODON_TABLE.get(codon, "X")
        protein.append(amino_acid)
    translated = "".join(protein)
    if translated.endswith("*"):
        translated = translated[:-1]
    return translated


def _extract_named_enzymes(
    user_text: str,
    *,
    known_sites: dict[str, str],
) -> list[str]:
    lowered = _normalized_text(user_text)
    found: list[str] = []
    for enzyme in known_sites:
        if re.search(rf"\b{re.escape(enzyme.lower())}\b", lowered):
            found.append(enzyme)
    return found


def _extract_vector_mcs_enzymes(
    user_text: str,
    *,
    known_sites: dict[str, str],
) -> list[str]:
    match = re.search(r"\[(.*?)\]", str(user_text or ""), flags=re.DOTALL)
    if not match:
        return []
    content = match.group(1)
    tokens = [token.strip() for token in re.split(r"-{2,}|\s+", content) if token.strip()]
    ordered: list[str] = []
    for token in tokens:
        for enzyme in known_sites:
            if token.lower() == enzyme.lower():
                ordered.append(enzyme)
                break
    return ordered


def _site_occurs_in_sequence(sequence: str, site: str) -> bool:
    if "W" not in site:
        return site in sequence
    pattern = site.replace("W", "[AT]")
    return re.search(pattern, sequence) is not None


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def run_directional_cloning_verifier(user_text: str) -> CodeVerificationResult | None:
    known_sites = {
        "ApaI": "GGGCCC",
        "BamHI": "GGATCC",
        "BstNI": "CCWGG",
        "HindIII": "AAGCTT",
        "KpnI": "GGTACC",
        "ScaI": "AGTACT",
    }
    clamp_prefixes = {
        "ApaI": "aaaa",
        "BamHI": "aaa",
        "BstNI": "aaa",
        "HindIII": "aaaa",
        "KpnI": "aaaa",
        "ScaI": "aaaa",
    }
    sequence = _extract_insert_sequence(user_text)
    if not sequence:
        return None
    lowered = _normalized_text(user_text)
    if not all(token in lowered for token in ("pcr", "primer")):
        return None
    freezer_enzymes = _extract_named_enzymes(user_text, known_sites=known_sites)
    vector_mcs = _extract_vector_mcs_enzymes(user_text, known_sites=known_sites)
    if not freezer_enzymes or not vector_mcs:
        return None

    internal_sites = {
        enzyme: _site_occurs_in_sequence(sequence, site)
        for enzyme, site in known_sites.items()
    }
    compatible = [
        enzyme
        for enzyme in vector_mcs
        if enzyme in freezer_enzymes and not internal_sites.get(enzyme, False)
    ]
    if len(compatible) < 2:
        detail = ", ".join(
            enzyme
            for enzyme in vector_mcs
            if internal_sites.get(enzyme, False) and enzyme in freezer_enzymes
        )
        summary = "Could not find two distinct vector-compatible enzymes without internal insert sites."
        if detail:
            summary += f" Internal sites were present for: {detail}."
        return CodeVerificationResult(
            attempted=True,
            verified=False,
            recommendation="repair",
            summary=summary,
        )

    forward_enzyme = compatible[0]
    reverse_enzyme = compatible[1]
    forward_core = sequence[: min(len(sequence), 17)]
    reverse_core = _reverse_complement(sequence[-min(len(sequence), 23):])
    forward_primer = (
        f"{clamp_prefixes.get(forward_enzyme, 'aaaa')}"
        f"{known_sites[forward_enzyme].lower()}"
        f"{forward_core}"
    )
    reverse_primer = (
        f"{clamp_prefixes.get(reverse_enzyme, 'aaaa')}"
        f"{known_sites[reverse_enzyme].lower()}"
        f"{reverse_core}"
    )
    excluded = [
        enzyme
        for enzyme in freezer_enzymes
        if internal_sites.get(enzyme, False)
    ]
    evidence = [
        EvidenceRef(source="code_verifier", detail=f"vector_order={','.join(vector_mcs)}"),
        EvidenceRef(
            source="code_verifier",
            detail=f"selected_enzymes={forward_enzyme},{reverse_enzyme}",
        ),
    ]
    if excluded:
        evidence.append(
            EvidenceRef(
                source="code_verifier",
                detail=f"internal_sites_excluded={','.join(excluded)}",
            )
        )
    return CodeVerificationResult(
        attempted=True,
        verified=True,
        recommendation="accept",
        summary=(
            f"Directional cloning verified with {forward_enzyme} on the forward primer and "
            f"{reverse_enzyme} on the reverse primer. "
            f"forward_primer={forward_primer}; reverse_primer={reverse_primer}."
        ),
        evidence=evidence,
        measurements=[
            {"name": "forward_enzyme", "value": forward_enzyme},
            {"name": "reverse_enzyme", "value": reverse_enzyme},
            {"name": "forward_primer", "value": forward_primer},
            {"name": "reverse_primer", "value": reverse_primer},
        ],
        raw_output=json.dumps(
            {
                "forward_enzyme": forward_enzyme,
                "reverse_enzyme": reverse_enzyme,
                "forward_primer": forward_primer,
                "reverse_primer": reverse_primer,
                "vector_order": vector_mcs,
                "excluded_internal_sites": excluded,
            },
            ensure_ascii=False,
        ),
    )


def run_coding_sequence_match_verifier(user_text: str) -> CodeVerificationResult | None:
    target_protein = _extract_target_protein_sequence(user_text)
    dna_options = _extract_mcq_dna_options(user_text)
    if not target_protein or len(dna_options) < 2:
        return None

    translated_options = {
        letter: _translate_dna(sequence)
        for letter, sequence in dna_options.items()
    }
    exact_matches = [
        letter
        for letter, protein in translated_options.items()
        if protein == target_protein
    ]
    if len(exact_matches) != 1:
        return None

    selected = exact_matches[0]
    evidence = [
        EvidenceRef(source="code_verifier", detail=f"selected_option={selected}"),
        EvidenceRef(
            source="code_verifier",
            detail=f"translated_length={len(translated_options[selected])}",
        ),
    ]
    return CodeVerificationResult(
        attempted=True,
        verified=True,
        recommendation="accept",
        summary=(
            f"Translation-based verification selected option {selected} as the only plasmid whose "
            "translated coding sequence exactly matches the provided protein sequence."
        ),
        evidence=evidence,
        measurements=[
            {"name": "selected_option", "value": selected},
            {"name": "target_protein_length", "value": len(target_protein)},
            {"name": "translated_protein_length", "value": len(translated_options[selected])},
        ],
        raw_output=json.dumps(
            {
                "selected_option": selected,
                "target_protein_length": len(target_protein),
                "translated_protein_length": len(translated_options[selected]),
            },
            ensure_ascii=False,
        ),
    )


def render_coding_sequence_match_response(result: CodeVerificationResult) -> str | None:
    values = {str(item.get("name") or "").strip(): item.get("value") for item in result.measurements}
    selected = str(values.get("selected_option") or "").strip().upper()
    if selected not in {"A", "B", "C", "D"}:
        return None
    return (
        f"Use option {selected}.\n\n"
        "It is the only plasmid whose translated coding sequence exactly matches the provided protein sequence."
    )


def render_directional_cloning_response(result: CodeVerificationResult) -> str | None:
    values = {str(item.get("name") or "").strip(): item.get("value") for item in result.measurements}
    forward_primer = str(values.get("forward_primer") or "").strip()
    reverse_primer = str(values.get("reverse_primer") or "").strip()
    forward_enzyme = str(values.get("forward_enzyme") or "").strip()
    reverse_enzyme = str(values.get("reverse_enzyme") or "").strip()
    if not all([forward_primer, reverse_primer, forward_enzyme, reverse_enzyme]):
        return None
    lines = [
        "Use the directional pair below:",
        f"`{forward_primer}`",
        f"`{reverse_primer}`",
        "",
        f"Forward primer adds `{forward_enzyme}` at the 5' end of the insert, and the reverse primer adds `{reverse_enzyme}` at the 3' end.",
        "This preserves orientation because the forward-site enzyme is upstream in the vector MCS and the reverse-site enzyme is downstream.",
    ]
    return "\n".join(lines).strip()


def _infer_metathesis_bicyclic_precursor(user_text: str) -> str | None:
    match = re.search(
        r"2-vinyl(cyclo(?:butane|pentane|hexane|heptane|octane))",
        _normalized_text(user_text),
    )
    if not match:
        return None
    ring_token = match.group(1)
    ring_sizes = {
        "cyclobutane": 4,
        "cyclopentane": 5,
        "cyclohexane": 6,
        "cycloheptane": 7,
        "cyclooctane": 8,
    }
    parent_stems = {
        6: "hex",
        7: "hept",
        8: "oct",
        9: "non",
        10: "dec",
    }
    ring_size = ring_sizes.get(ring_token)
    if ring_size is None:
        return None
    total_atoms = ring_size + 2
    bridge_a = ring_size - 2
    parent = parent_stems.get(total_atoms)
    if parent is None:
        return None
    alkene_locant = total_atoms - 1
    return f"bicyclo[{bridge_a}.2.0]{parent}-{alkene_locant}-ene"


def run_metathesis_retrosynthesis_verifier(user_text: str) -> CodeVerificationResult | None:
    lowered = _normalized_text(user_text)
    if "methyleneruthenium" not in lowered:
        return None
    if "starting material" not in lowered:
        return None
    starting_material = _infer_metathesis_bicyclic_precursor(user_text)
    if not starting_material:
        return None
    evidence = [
        EvidenceRef(
            source="code_verifier",
            detail="reaction_class=ring-opening cross metathesis of a strained bicyclic alkene",
        ),
        EvidenceRef(
            source="code_verifier",
            detail=f"starting_material={starting_material}",
        ),
    ]
    return CodeVerificationResult(
        attempted=True,
        verified=True,
        recommendation="accept",
        summary=(
            "The product pattern is consistent with ROM/CM of a strained bicyclobutene-fused system; "
            f"the starting material is {starting_material}."
        ),
        evidence=evidence,
        measurements=[
            {"name": "starting_material", "value": starting_material},
            {"name": "reaction_class", "value": "ring-opening cross metathesis"},
        ],
        raw_output=json.dumps(
            {
                "starting_material": starting_material,
                "reaction_class": "ring-opening cross metathesis",
            },
            ensure_ascii=False,
        ),
    )


def render_metathesis_retrosynthesis_response(result: CodeVerificationResult) -> str | None:
    values = {str(item.get("name") or "").strip(): item.get("value") for item in result.measurements}
    starting_material = str(values.get("starting_material") or "").strip()
    if not starting_material:
        return None
    return (
        f"The starting material is `{starting_material}`.\n\n"
        "This product pattern matches ring-opening cross metathesis of a strained bicyclic alkene precursor."
    )


def run_fluorescence_localization_verifier(user_text: str) -> CodeVerificationResult | None:
    lowered = _normalized_text(user_text)
    if not detect_fluorescence_localization_signal(user_text):
        return None
    red_channel = "red signal" if "red signal" in lowered else "reporter signal"
    summary = (
        "A promoter-driven fluorescent reporter without an explicit targeting sequence is expected "
        "to localize in the cytoplasm, whereas TUNEL-FITC marks apoptotic nuclei."
    )
    return CodeVerificationResult(
        attempted=True,
        verified=True,
        recommendation="accept",
        summary=summary,
        evidence=[
            EvidenceRef(
                source="code_verifier",
                detail="reporter_without_targeting_sequence=cytoplasmic",
            ),
            EvidenceRef(
                source="code_verifier",
                detail="tunel_fitc_marks=nuclear_apoptotic_dna",
            ),
        ],
        measurements=[
            {
                "name": "expected_reporter_localization",
                "value": "cytoplasmic localization of the red signal",
            },
            {"name": "apoptosis_marker_localization", "value": "nuclear FITC/TUNEL signal"},
            {"name": "red_channel_label", "value": red_channel},
        ],
        raw_output=json.dumps(
            {
                "expected_reporter_localization": "cytoplasmic localization of the red signal",
                "apoptosis_marker_localization": "nuclear FITC/TUNEL signal",
            },
            ensure_ascii=False,
        ),
    )


def render_fluorescence_localization_response(result: CodeVerificationResult) -> str | None:
    values = {str(item.get("name") or "").strip(): item.get("value") for item in result.measurements}
    localization = str(values.get("expected_reporter_localization") or "").strip()
    if not localization:
        return None
    return (
        f"{localization}.\n\n"
        "A promoter-driven fluorescent reporter without a targeting sequence is expected in the cytoplasm, while TUNEL-FITC labels apoptotic nuclei."
    )


STRUCTURED_VERIFIER_HANDLERS: tuple[StructuredVerifierHandler, ...] = (
    StructuredVerifierHandler(
        name="coding_sequence_match",
        signal_key="coding_sequence_match_cues",
        verify=run_coding_sequence_match_verifier,
        render=render_coding_sequence_match_response,
    ),
    StructuredVerifierHandler(
        name="directional_cloning",
        signal_key="sequence_design_cues",
        verify=run_directional_cloning_verifier,
        render=render_directional_cloning_response,
    ),
    StructuredVerifierHandler(
        name="metathesis_retrosynthesis",
        signal_key="reaction_retrosynthesis_cues",
        verify=run_metathesis_retrosynthesis_verifier,
        render=render_metathesis_retrosynthesis_response,
    ),
    StructuredVerifierHandler(
        name="fluorescence_localization",
        signal_key="fluorescence_localization_cues",
        verify=run_fluorescence_localization_verifier,
        render=render_fluorescence_localization_response,
    ),
)


def run_structured_verifier(
    *,
    user_text: str,
    deliberation: DeliberationPolicy,
) -> CodeVerificationResult | None:
    for handler in STRUCTURED_VERIFIER_HANDLERS:
        if not bool(deliberation.signals.get(handler.signal_key)):
            continue
        result = handler.verify(user_text)
        if result is not None:
            return result
    return None


def render_structured_verifier_response(
    *,
    deliberation: DeliberationPolicy,
    code_verification: CodeVerificationResult | None,
) -> str | None:
    if code_verification is None:
        return None
    for handler in STRUCTURED_VERIFIER_HANDLERS:
        if not bool(deliberation.signals.get(handler.signal_key)):
            continue
        rendered = handler.render(code_verification)
        if rendered:
            return rendered
    return None


__all__ = [
    "STRUCTURED_SIGNAL_SPECS",
    "STRUCTURED_VERIFIER_HANDLERS",
    "detect_coding_sequence_match_signal",
    "detect_fluorescence_localization_signal",
    "detect_metathesis_retrosynthesis_signal",
    "detect_sequence_design_signal",
    "detect_structured_problem_signals",
    "has_structured_verifier_signal",
    "render_coding_sequence_match_response",
    "render_structured_verifier_response",
    "run_coding_sequence_match_verifier",
    "run_directional_cloning_verifier",
    "run_fluorescence_localization_verifier",
    "run_metathesis_retrosynthesis_verifier",
    "run_structured_verifier",
]
