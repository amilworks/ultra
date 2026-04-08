from __future__ import annotations

import ast
from typing import Any

import numpy as np


_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "abs": np.abs,
    "arange": np.arange,
    "arccos": np.arccos,
    "arcsin": np.arcsin,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "array": np.array,
    "ceil": np.ceil,
    "cos": np.cos,
    "cosh": np.cosh,
    "deg2rad": np.deg2rad,
    "diag": np.diag,
    "dot": np.dot,
    "exp": np.exp,
    "floor": np.floor,
    "linspace": np.linspace,
    "log": np.log,
    "log10": np.log10,
    "log2": np.log2,
    "matmul": np.matmul,
    "max": np.max,
    "mean": np.mean,
    "min": np.min,
    "norm": np.linalg.norm,
    "outer": np.outer,
    "prod": np.prod,
    "rad2deg": np.rad2deg,
    "round": np.round,
    "sign": np.sign,
    "sin": np.sin,
    "sinh": np.sinh,
    "sqrt": np.sqrt,
    "std": np.std,
    "sum": np.sum,
    "tan": np.tan,
    "tanh": np.tanh,
    "trace": np.trace,
    "var": np.var,
}

_ALLOWED_CONSTANTS: dict[str, Any] = {
    "e": float(np.e),
    "pi": float(np.pi),
    "tau": float(np.pi * 2.0),
    "hbar_c": 197.3269804,
    "hbarc": 197.3269804,
    "m_e_c2": 0.51099895,
    "electron_rest_energy_mev": 0.51099895,
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.Call,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.MatMult,
    ast.UAdd,
    ast.USub,
    ast.List,
    ast.Tuple,
    ast.Subscript,
    ast.Slice,
)


def _ensure_identifier(name: str) -> str:
    token = str(name or "").strip()
    if not token:
        raise ValueError("Variable names must be non-empty.")
    if not token.isidentifier():
        raise ValueError(f"Invalid variable name: {token!r}")
    if token.startswith("_"):
        raise ValueError(f"Variable names cannot start with '_': {token!r}")
    return token


def _normalize_variable_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, complex):
        return complex(value)
    if isinstance(value, list):
        return [_normalize_variable_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_variable_value(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _normalize_variable_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise ValueError(f"Unsupported calculator variable type: {type(value).__name__}")


def _validate_expression(expression: str, allowed_names: set[str]) -> ast.Expression:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported calculator syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed in numpy_calculator.")
            if node.func.id not in allowed_names:
                raise ValueError(f"Unknown calculator function: {node.func.id}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unknown calculator symbol: {node.id}")
    return tree


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_ready(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _result_type(value: Any) -> str:
    if isinstance(value, dict) and {"real", "imag"} <= set(value.keys()):
        return "complex"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "scalar"
    return type(value).__name__


def _format_result(value: Any) -> str:
    if isinstance(value, dict) and {"real", "imag"} <= set(value.keys()):
        return f"{value['real']} + {value['imag']}j"
    if isinstance(value, list):
        return np.array2string(np.asarray(value), precision=12, separator=", ")
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def numpy_calculator(
    expression: str,
    variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a deterministic NumPy-backed numeric expression."""

    raw_expression = str(expression or "").strip()
    if not raw_expression:
        return {
            "success": False,
            "error": "Expression is required.",
        }

    normalized_variables: dict[str, Any] = {}
    for key, value in dict(variables or {}).items():
        normalized_variables[_ensure_identifier(str(key))] = _normalize_variable_value(value)

    environment: dict[str, Any] = {
        **_ALLOWED_FUNCTIONS,
        **_ALLOWED_CONSTANTS,
        **normalized_variables,
    }
    allowed_names = set(environment.keys())

    try:
        tree = _validate_expression(raw_expression, allowed_names)
        value = eval(
            compile(tree, filename="<numpy_calculator>", mode="eval"),
            {"__builtins__": {}},
            environment,
        )
    except Exception as exc:
        return {
            "success": False,
            "expression": raw_expression,
            "error": str(exc),
        }

    serialized = _json_ready(value)
    return {
        "success": True,
        "expression": raw_expression,
        "result": serialized,
        "result_type": _result_type(serialized),
        "formatted_result": _format_result(serialized),
        "variables_used": sorted(normalized_variables.keys()),
    }
