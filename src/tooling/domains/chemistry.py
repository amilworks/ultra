"""Chemistry workbench tool schemas."""

STRUCTURE_REPORT_TOOL = {
    "type": "function",
    "function": {
        "name": "structure_report",
        "description": (
            "Return deterministic structure facts for a molecule: canonical SMILES, formula, exact mass, "
            "ring sizes, bridgehead/spiro counts, functional groups, and strain cues. "
            "Use this to ground organic chemistry reasoning before proposing mechanisms or products."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "structure": {
                    "type": "string",
                    "description": "SMILES, InChI, MolBlock, or SMARTS for the molecule to analyze.",
                },
                "input_format": {
                    "type": "string",
                    "enum": ["smiles", "inchi", "molblock", "smarts"],
                    "default": "smiles",
                    "description": "How to interpret the structure string.",
                },
            },
            "required": ["structure"],
        },
    },
}


COMPARE_STRUCTURES_TOOL = {
    "type": "function",
    "function": {
        "name": "compare_structures",
        "description": (
            "Compare two molecules and report objective structural deltas: formula change, gained/lost "
            "functional groups, ring-size changes, maximum common substructure, and heuristic transformation labels."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "substrate": {
                    "type": "string",
                    "description": "Starting molecule as SMILES, InChI, or MolBlock.",
                },
                "product": {
                    "type": "string",
                    "description": "Product or candidate product as SMILES, InChI, or MolBlock.",
                },
                "input_format": {
                    "type": "string",
                    "enum": ["smiles", "inchi", "molblock"],
                    "default": "smiles",
                    "description": "Shared input format used for both structures.",
                },
            },
            "required": ["substrate", "product"],
        },
    },
}


PROPOSE_REACTIVE_SITES_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_reactive_sites",
        "description": (
            "Score likely reactive atoms and motifs under supplied conditions. This is a deterministic helper "
            "for reaction-site candidates, not a black-box reaction predictor."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "structure": {
                    "type": "string",
                    "description": "Molecule as SMILES, InChI, or MolBlock.",
                },
                "conditions_text": {
                    "type": "string",
                    "default": "",
                    "description": "Free-text reagents or conditions, for example 'PDC in CH2Cl2' or 'TsOH, heat'.",
                },
                "input_format": {
                    "type": "string",
                    "enum": ["smiles", "inchi", "molblock"],
                    "default": "smiles",
                    "description": "Format of the structure string.",
                },
                "top_k": {
                    "type": "integer",
                    "default": 8,
                    "minimum": 1,
                    "maximum": 20,
                    "description": "How many of the highest-scoring reactive atoms to return.",
                },
            },
            "required": ["structure"],
        },
    },
}


FORMULA_BALANCE_CHECK_TOOL = {
    "type": "function",
    "function": {
        "name": "formula_balance_check",
        "description": (
            "Check elemental balance between reactant and product sides. Use this to eliminate impossible "
            "candidate products or verify whether a proposed transformation preserves atoms aside from omitted reagents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reactants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Reactant molecules as a list of structure strings.",
                },
                "products": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Product molecules as a list of structure strings.",
                },
                "input_format": {
                    "type": "string",
                    "enum": ["smiles", "inchi", "molblock"],
                    "default": "smiles",
                    "description": "Format shared by all structure strings.",
                },
                "reactant_coefficients": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional stoichiometric coefficients for reactants. Defaults to all 1s.",
                },
                "product_coefficients": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional stoichiometric coefficients for products. Defaults to all 1s.",
                },
            },
            "required": ["reactants", "products"],
        },
    },
}


CHEMISTRY_TOOL_SCHEMAS = [
    STRUCTURE_REPORT_TOOL,
    COMPARE_STRUCTURES_TOOL,
    PROPOSE_REACTIVE_SITES_TOOL,
    FORMULA_BALANCE_CHECK_TOOL,
]


__all__ = [
    "CHEMISTRY_TOOL_SCHEMAS",
    "COMPARE_STRUCTURES_TOOL",
    "FORMULA_BALANCE_CHECK_TOOL",
    "PROPOSE_REACTIVE_SITES_TOOL",
    "STRUCTURE_REPORT_TOOL",
]
