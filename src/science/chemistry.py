"""Deterministic chemistry helpers powered by RDKit."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Literal

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdFMCS, rdMolDescriptors
except Exception as exc:  # pragma: no cover - import guard for runtime use
    Chem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    Lipinski = None  # type: ignore[assignment]
    rdFMCS = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(exc)
else:  # pragma: no cover - import exercised indirectly by tool tests
    _IMPORT_ERROR = None


StructureInputFormat = Literal["smiles", "inchi", "molblock", "smarts"]
ParsedStructureInputFormat = Literal["smiles", "inchi", "molblock"]


_FUNCTIONAL_GROUP_SMARTS: dict[str, str] = {
    "alkyl_halide": "[CX4][F,Cl,Br,I]",
    "alcohol": "[OX2H][CX4]",
    "phenol": "[OX2H]c",
    "ether": "[OD2]([#6])[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "amide": "[NX3][CX3](=O)[#6]",
    "amine": "[NX3;!$(NC=O)]",
    "nitrile": "[CX2]#N",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "epoxide": "[OX2r3]1[CX4r3][CX4r3]1",
    "carbonyl": "[CX3]=[OX1]",
}


def _build_patterns() -> dict[str, Any]:
    if Chem is None:
        return {}
    return {
        name: Chem.MolFromSmarts(smarts)
        for name, smarts in _FUNCTIONAL_GROUP_SMARTS.items()
    }


_PATTERNS: dict[str, Any] = _build_patterns()


def _require_rdkit() -> None:
    if Chem is None:
        raise RuntimeError(
            "RDKit is not available. Install RDKit in the runtime that executes the tool. "
            f"Original import error: {_IMPORT_ERROR}"
        )


def _parse_formula(formula: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for symbol, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[symbol] += int(count) if count else 1
    return counts


def _format_formula_delta(delta: Counter[str]) -> dict[str, int]:
    return {key: value for key, value in sorted(delta.items()) if value != 0}


def _signed_counter_delta(a: Counter[str], b: Counter[str]) -> Counter[str]:
    keys = set(a) | set(b)
    return Counter({key: a.get(key, 0) - b.get(key, 0) for key in keys})


def _parse_mol(structure: str, input_format: StructureInputFormat) -> Any:
    _require_rdkit()
    token = str(structure or "").strip()
    if not token:
        raise ValueError("Structure string is required.")
    if input_format == "smiles":
        mol = Chem.MolFromSmiles(token)
    elif input_format == "inchi":
        mol = Chem.MolFromInchi(token)
    elif input_format == "molblock":
        mol = Chem.MolFromMolBlock(token, sanitize=True)
    elif input_format == "smarts":
        mol = Chem.MolFromSmarts(token)
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    if mol is None:
        raise ValueError(f"Could not parse structure as {input_format!r}: {token[:120]}")
    return mol


def _canonical_smiles(mol: Any) -> str:
    return str(Chem.MolToSmiles(mol, canonical=True))


def _ring_sizes(mol: Any) -> list[int]:
    ring_info = mol.GetRingInfo()
    return sorted(len(ring) for ring in ring_info.AtomRings())


def _ring_size_counter(mol: Any) -> Counter[int]:
    return Counter(_ring_sizes(mol))


def _formula_and_counts(mol: Any) -> tuple[str, Counter[str]]:
    formula = str(rdMolDescriptors.CalcMolFormula(mol))
    return formula, _parse_formula(formula)


def _dbe_from_formula_counts(counts: Counter[str]) -> float:
    c_count = counts.get("C", 0)
    h_count = counts.get("H", 0)
    n_count = counts.get("N", 0)
    x_count = counts.get("F", 0) + counts.get("Cl", 0) + counts.get("Br", 0) + counts.get("I", 0)
    return (2 * c_count + 2 + n_count - h_count - x_count) / 2


def _count_functional_groups(mol: Any) -> dict[str, int]:
    output: dict[str, int] = {}
    for name, pattern in _PATTERNS.items():
        if pattern is None:
            continue
        output[name] = len(mol.GetSubstructMatches(pattern, uniquify=True))
    return output


def _strain_flags(mol: Any) -> list[str]:
    flags: list[str] = []
    ring_sizes = _ring_sizes(mol)
    if any(size == 3 for size in ring_sizes):
        flags.append("contains three-membered ring (high ring strain)")
    if any(size == 4 for size in ring_sizes):
        flags.append("contains four-membered ring (ring strain; rearrangements/ring expansion may be favored)")
    if rdMolDescriptors.CalcNumBridgeheadAtoms(mol) > 0:
        flags.append("contains bridgehead atom(s)")
    if rdMolDescriptors.CalcNumSpiroAtoms(mol) > 0:
        flags.append("contains spiro atom(s)")
    return flags


def _conditions_to_classes(conditions_text: str) -> list[str]:
    text = str(conditions_text or "").lower()
    classes: set[str] = set()
    if any(token in text for token in ["pcc", "pdc", "dess-martin", "dmp", "swern", "jones", "cro3", "oxid"]):
        classes.add("oxidation")
    if any(token in text for token in ["nah", "lda", "dbu", "tbuok", "base", "alkoxide"]):
        classes.add("base")
    if any(token in text for token in ["h+", "acid", "h2so4", "tsoh", "hcl", "hbr", "bf3"]):
        classes.add("acid")
    if any(token in text for token in ["h2o", "water", "aq", "hydrolysis"]):
        classes.add("water")
    if any(token in text for token in ["nabh4", "lah", "lialh4", "dibal", "h2", "pd/c", "pt", "raney"]):
        classes.add("reduction")
    if any(token in text for token in ["wittig", "ylide", "ph3p=ch2", "h2cpph3", "phosphorane"]):
        classes.add("wittig")
    if any(token in text for token in ["sn1", "sn2", "substitution"]):
        classes.add("substitution")
    if any(token in text for token in ["e1", "e2", "elimination", "dehydration"]):
        classes.add("elimination")
    return sorted(classes)


def _top_sites(site_scores: dict[int, dict[str, Any]], mol: Any, top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(site_scores.items(), key=lambda item: (-item[1]["score"], item[0]))[:top_k]
    output: list[dict[str, Any]] = []
    for atom_index, payload in ranked:
        atom = mol.GetAtomWithIdx(atom_index)
        output.append(
            {
                "atom_index": atom_index,
                "symbol": atom.GetSymbol(),
                "score": round(float(payload["score"]), 2),
                "reasons": sorted(payload["reasons"]),
                "in_ring": bool(atom.IsInRing()),
                "aromatic": bool(atom.GetIsAromatic()),
                "formal_charge": int(atom.GetFormalCharge()),
            }
        )
    return output


def _infer_transformation_labels(
    substrate_functional_groups: dict[str, int],
    product_functional_groups: dict[str, int],
    formula_delta: Counter[str],
    substrate_rings: Counter[int],
    product_rings: Counter[int],
) -> list[str]:
    labels: list[str] = []
    delta = _format_formula_delta(formula_delta)
    if (
        substrate_functional_groups.get("alcohol", 0) > product_functional_groups.get("alcohol", 0)
        and (
            product_functional_groups.get("ketone", 0)
            + product_functional_groups.get("aldehyde", 0)
            + product_functional_groups.get("carboxylic_acid", 0)
            > substrate_functional_groups.get("ketone", 0)
            + substrate_functional_groups.get("aldehyde", 0)
            + substrate_functional_groups.get("carboxylic_acid", 0)
        )
        and delta.get("O", 0) == 0
    ):
        labels.append("alcohol oxidation")
    if (
        substrate_functional_groups.get("alkyl_halide", 0) > product_functional_groups.get("alkyl_halide", 0)
        and product_functional_groups.get("alcohol", 0) > substrate_functional_groups.get("alcohol", 0)
    ):
        labels.append("halide-to-alcohol substitution or hydrolysis")
    if (
        substrate_functional_groups.get("carbonyl", 0) > product_functional_groups.get("carbonyl", 0)
        and product_functional_groups.get("alkene", 0) > substrate_functional_groups.get("alkene", 0)
    ):
        labels.append("carbonyl-to-alkene conversion (olefination / methylenation / deoxygenation class)")
    if substrate_rings.get(4, 0) > product_rings.get(4, 0) and product_rings.get(5, 0) >= substrate_rings.get(5, 0):
        labels.append("strain-relieving ring expansion or rearrangement")
    if delta.get("H", 0) < 0 and delta.get("O", 0) < 0 and product_functional_groups.get("alkene", 0) > substrate_functional_groups.get("alkene", 0):
        labels.append("elimination / dehydration")
    return labels


def structure_report(
    structure: str,
    input_format: StructureInputFormat = "smiles",
) -> dict[str, Any]:
    """Return deterministic structural facts for a molecule."""

    mol = _parse_mol(structure, input_format)
    formula, counts = _formula_and_counts(mol)
    functional_groups = _count_functional_groups(mol)
    ring_sizes = _ring_sizes(mol)
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    return {
        "success": True,
        "canonical_smiles": _canonical_smiles(mol),
        "formula": formula,
        "exact_mass": round(float(rdMolDescriptors.CalcExactMolWt(mol)), 6),
        "molecular_weight": round(float(Descriptors.MolWt(mol)), 6),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "hbond_donors": int(Lipinski.NumHDonors(mol)),
        "hbond_acceptors": int(Lipinski.NumHAcceptors(mol)),
        "topological_polar_surface_area": round(float(rdMolDescriptors.CalcTPSA(mol)), 6),
        "rings": {
            "count": len(ring_sizes),
            "sizes": ring_sizes,
            "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "aliphatic_ring_count": int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
            "bridgehead_atom_count": int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
            "spiro_atom_count": int(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
        },
        "degree_of_unsaturation": _dbe_from_formula_counts(counts),
        "stereochemistry": {
            "chiral_center_count": len(chiral_centers),
            "centers": [
                {"atom_index": int(atom_index), "assignment": str(tag)}
                for atom_index, tag in chiral_centers
            ],
        },
        "functional_groups": functional_groups,
        "strain_flags": _strain_flags(mol),
    }


def compare_structures(
    substrate: str,
    product: str,
    input_format: ParsedStructureInputFormat = "smiles",
) -> dict[str, Any]:
    """Compare two molecules and summarize objective structural changes."""

    substrate_mol = _parse_mol(substrate, input_format)
    product_mol = _parse_mol(product, input_format)

    substrate_formula, substrate_counts = _formula_and_counts(substrate_mol)
    product_formula, product_counts = _formula_and_counts(product_mol)
    formula_delta = _signed_counter_delta(product_counts, substrate_counts)

    substrate_functional_groups = _count_functional_groups(substrate_mol)
    product_functional_groups = _count_functional_groups(product_mol)
    functional_group_delta = {
        key: product_functional_groups.get(key, 0) - substrate_functional_groups.get(key, 0)
        for key in sorted(set(substrate_functional_groups) | set(product_functional_groups))
    }
    functional_group_delta = {
        key: value for key, value in functional_group_delta.items() if value != 0
    }

    substrate_rings = _ring_size_counter(substrate_mol)
    product_rings = _ring_size_counter(product_mol)
    ring_size_delta = {
        str(key): product_rings.get(key, 0) - substrate_rings.get(key, 0)
        for key in sorted(set(substrate_rings) | set(product_rings))
    }
    ring_size_delta = {key: value for key, value in ring_size_delta.items() if value != 0}

    mcs = rdFMCS.FindMCS(
        [substrate_mol, product_mol],
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        timeout=5,
    )

    return {
        "success": True,
        "substrate": {
            "canonical_smiles": _canonical_smiles(substrate_mol),
            "formula": substrate_formula,
            "degree_of_unsaturation": _dbe_from_formula_counts(substrate_counts),
        },
        "product": {
            "canonical_smiles": _canonical_smiles(product_mol),
            "formula": product_formula,
            "degree_of_unsaturation": _dbe_from_formula_counts(product_counts),
        },
        "formula_delta_product_minus_substrate": _format_formula_delta(formula_delta),
        "functional_group_delta_product_minus_substrate": functional_group_delta,
        "ring_size_delta_product_minus_substrate": ring_size_delta,
        "mcs": {
            "num_atoms": int(mcs.numAtoms),
            "num_bonds": int(mcs.numBonds),
            "smarts": str(mcs.smartsString or ""),
        },
        "heuristic_transformation_labels": _infer_transformation_labels(
            substrate_functional_groups,
            product_functional_groups,
            formula_delta,
            substrate_rings,
            product_rings,
        ),
    }


def propose_reactive_sites(
    structure: str,
    conditions_text: str = "",
    input_format: ParsedStructureInputFormat = "smiles",
    top_k: int = 8,
) -> dict[str, Any]:
    """Score likely reactive atoms and motifs under supplied conditions."""

    mol = _parse_mol(structure, input_format)
    conditions = _conditions_to_classes(conditions_text)
    site_scores: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"score": 0.0, "reasons": set()}
    )
    global_motifs: list[str] = []

    def add(atom_index: int, score: float, reason: str) -> None:
        site_scores[atom_index]["score"] += score
        site_scores[atom_index]["reasons"].add(reason)

    for ring in mol.GetRingInfo().AtomRings():
        size = len(ring)
        for atom_index in ring:
            if size <= 4:
                add(atom_index, 1.25, f"atom lies in strained {size}-membered ring")

    for name, pattern in _PATTERNS.items():
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern, uniquify=True)
        if matches:
            global_motifs.append(name)

        if name == "alkyl_halide":
            for match in matches:
                carbon_index, halogen_index = match[0], match[1]
                add(carbon_index, 2.5, "sp3 carbon bearing halide: substitution/elimination handle")
                add(halogen_index, 1.0, "leaving group halogen")
        elif name in {"alcohol", "phenol"}:
            for match in matches:
                oxygen_index, carbon_index = match[0], match[1]
                add(oxygen_index, 1.75, "heteroatom that can be protonated/deprotonated or activated")
                add(carbon_index, 1.25, "carbon attached to oxygen")
        elif name == "carbonyl":
            for match in matches:
                carbon_index, oxygen_index = match[0], match[1]
                add(carbon_index, 3.0, "electrophilic carbonyl carbon")
                add(oxygen_index, 1.5, "basic carbonyl oxygen")
                carbon = mol.GetAtomWithIdx(carbon_index)
                for neighbor in carbon.GetNeighbors():
                    if neighbor.GetIdx() == oxygen_index or neighbor.GetAtomicNum() != 6:
                        continue
                    if neighbor.GetTotalNumHs() > 0:
                        add(neighbor.GetIdx(), 1.5, "alpha carbon to carbonyl (enolization / deprotonation site)")
        elif name == "alkene":
            for match in matches:
                add(match[0], 1.5, "alkene carbon: electrophilic addition / protonation site")
                add(match[1], 1.5, "alkene carbon: electrophilic addition / protonation site")
        elif name == "epoxide":
            for match in matches:
                for atom_index in match:
                    add(atom_index, 2.0, "epoxide atom: strained ring-opening motif")
        elif name == "nitrile":
            for match in matches:
                add(match[0], 1.0, "nitrile carbon")
                add(match[1], 0.75, "nitrile nitrogen")

    if "oxidation" in conditions:
        for atom_index, payload in list(site_scores.items()):
            if any(
                "attached to oxygen" in reason or "protonated/deprotonated" in reason
                for reason in payload["reasons"]
            ):
                add(atom_index, 1.5, "oxidation conditions increase relevance of alcohol/adjacent carbon")

    if "reduction" in conditions or "wittig" in conditions:
        for atom_index, payload in list(site_scores.items()):
            if any("carbonyl carbon" in reason for reason in payload["reasons"]):
                bonus = 2.5 if "wittig" in conditions else 1.75
                add(atom_index, bonus, "carbonyl carbon is prioritized by the stated conditions")

    if "acid" in conditions:
        for atom_index, payload in list(site_scores.items()):
            if any("oxygen" in reason or "alkene carbon" in reason for reason in payload["reasons"]):
                add(atom_index, 1.5, "acidic conditions enhance protonation/cation formation")
            if any("strained" in reason for reason in payload["reasons"]):
                add(atom_index, 1.75, "acid plus strain can favor rearrangement or ring expansion")

    if "base" in conditions or "elimination" in conditions:
        for atom_index, payload in list(site_scores.items()):
            if any("alpha carbon" in reason or "bearing halide" in reason for reason in payload["reasons"]):
                add(atom_index, 1.5, "base/elimination conditions increase relevance")

    if "water" in conditions or "substitution" in conditions:
        for atom_index, payload in list(site_scores.items()):
            if any("bearing halide" in reason or "electrophilic carbonyl carbon" in reason for reason in payload["reasons"]):
                add(atom_index, 1.0, "water/substitution conditions increase relevance")

    return {
        "success": True,
        "canonical_smiles": _canonical_smiles(mol),
        "condition_classes": conditions,
        "global_motifs": sorted(set(global_motifs)),
        "top_reactive_sites": _top_sites(site_scores, mol, max(1, min(int(top_k), 20))),
        "strain_flags": _strain_flags(mol),
    }


def formula_balance_check(
    reactants: list[str],
    products: list[str],
    input_format: ParsedStructureInputFormat = "smiles",
    reactant_coefficients: list[int] | None = None,
    product_coefficients: list[int] | None = None,
) -> dict[str, Any]:
    """Check elemental balance between reactant and product structure lists."""

    reactant_coefficients = reactant_coefficients or [1] * len(reactants)
    product_coefficients = product_coefficients or [1] * len(products)
    if len(reactants) != len(reactant_coefficients):
        raise ValueError("reactant_coefficients must have the same length as reactants")
    if len(products) != len(product_coefficients):
        raise ValueError("product_coefficients must have the same length as products")

    lhs: Counter[str] = Counter()
    rhs: Counter[str] = Counter()

    for coefficient, structure in zip(reactant_coefficients, reactants):
        mol = _parse_mol(structure, input_format)
        _, counts = _formula_and_counts(mol)
        for element, count in counts.items():
            lhs[element] += int(coefficient) * int(count)

    for coefficient, structure in zip(product_coefficients, products):
        mol = _parse_mol(structure, input_format)
        _, counts = _formula_and_counts(mol)
        for element, count in counts.items():
            rhs[element] += int(coefficient) * int(count)

    imbalance = _signed_counter_delta(rhs, lhs)
    return {
        "success": True,
        "balanced": all(value == 0 for value in imbalance.values()),
        "reactant_element_counts": dict(sorted(lhs.items())),
        "product_element_counts": dict(sorted(rhs.items())),
        "product_minus_reactant": _format_formula_delta(imbalance),
    }


__all__ = [
    "formula_balance_check",
    "compare_structures",
    "propose_reactive_sites",
    "structure_report",
]
