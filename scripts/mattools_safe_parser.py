#!/usr/bin/env python3
"""Fail-closed parser for candidate-controlled MatTools function output.

The pinned upstream ``ComplexDictParser`` evaluates text extracted from a
candidate's stdout.  Qualification must never import or execute that parser on
the host.  This module implements the small data grammar the benchmark needs
without ``eval``, attribute traversal, imports, comprehensions, or user code.
"""

from __future__ import annotations

import ast
import io
import math
import re
import tokenize
from dataclasses import dataclass
from typing import Any

MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_AST_NODES = 200_000
MAX_AST_DEPTH = 64
MAX_COLLECTION_ITEMS = 1_000_000
MAX_STRING_BYTES = 256 * 1024
MAX_INTEGER_BITS = 4096
MAX_ARRAY_ELEMENTS = 1_000_000
MAX_ARRAY_DIMENSIONS = 16
MAX_ARRAY_BYTES = 32 * 1024 * 1024

_ELEMENT_SYMBOL_RE = re.compile(r"^[A-Z][a-z]?$", flags=re.ASCII)
_ALLOWED_DTYPE_ATTRIBUTES = {
    "bool_",
    "bytes_",
    "complex64",
    "complex128",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "str_",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
}
_ALLOWED_DTYPE_NAMES = {"bool", "complex", "float", "int"}
_SAFE_SCALARS = (str, bytes, bool, int, float, complex, type(None))


class SafeParseError(ValueError):
    """Raised when candidate output is outside the reviewed data grammar."""


@dataclass
class _Budget:
    collection_items: int = 0
    string_bytes: int = 0

    def add_items(self, count: int) -> None:
        self.collection_items += count
        if self.collection_items > MAX_COLLECTION_ITEMS:
            raise SafeParseError("candidate output contains too many collection items")

    def add_string(self, value: str | bytes) -> None:
        encoded = value.encode("utf-8") if isinstance(value, str) else value
        self.string_bytes += len(encoded)
        if self.string_bytes > MAX_STRING_BYTES:
            raise SafeParseError("candidate output contains too much string data")


def _legacy_element_syntax(source: str) -> str:
    """Translate only the upstream-supported ``Element Mg`` token form."""

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (IndentationError, tokenize.TokenError) as exc:
        raise SafeParseError("candidate output is not valid Python expression text") from exc
    rewritten: list[tokenize.TokenInfo] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == tokenize.NAME and token.string == "Element":
            next_index = index + 1
            while next_index < len(tokens) and tokens[next_index].type in {
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            }:
                next_index += 1
            if next_index < len(tokens):
                symbol = tokens[next_index]
                if symbol.type == tokenize.NAME and _ELEMENT_SYMBOL_RE.fullmatch(symbol.string):
                    rewritten.extend(
                        (
                            token,
                            tokenize.TokenInfo(
                                tokenize.OP, "(", token.end, token.end, token.line
                            ),
                            tokenize.TokenInfo(
                                tokenize.STRING,
                                repr(symbol.string),
                                symbol.start,
                                symbol.end,
                                symbol.line,
                            ),
                            tokenize.TokenInfo(
                                tokenize.OP, ")", symbol.end, symbol.end, symbol.line
                            ),
                        )
                    )
                    index = next_index + 1
                    continue
        rewritten.append(token)
        index += 1
    return tokenize.untokenize(rewritten)


def _validate_ast_size(tree: ast.AST) -> None:
    stack: list[tuple[ast.AST, int]] = [(tree, 1)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_AST_NODES:
            raise SafeParseError("candidate output AST is too large")
        if depth > MAX_AST_DEPTH:
            raise SafeParseError("candidate output AST is too deeply nested")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


def _safe_constant(value: Any, budget: _Budget) -> Any:
    if not isinstance(value, _SAFE_SCALARS):
        raise SafeParseError(f"unsupported literal type: {type(value).__name__}")
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        if value.bit_length() > MAX_INTEGER_BITS:
            raise SafeParseError("integer literal is too large")
        return value
    if isinstance(value, (float, complex)):
        parts = (value.real, value.imag) if isinstance(value, complex) else (value,)
        if not all(math.isfinite(float(part)) for part in parts):
            raise SafeParseError("non-finite numeric literals are not accepted")
        return value
    if isinstance(value, (str, bytes)):
        budget.add_string(value)
        return value
    raise SafeParseError("unsupported literal")


def _dtype_from_node(node: ast.AST, np: Any, budget: _Budget) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        name = _safe_constant(node.value, budget)
        if len(name) > 32:
            raise SafeParseError("dtype string is too long")
        try:
            dtype = np.dtype(name)
        except (TypeError, ValueError) as exc:
            raise SafeParseError("dtype string is not supported") from exc
    elif isinstance(node, ast.Name) and node.id in _ALLOWED_DTYPE_NAMES:
        dtype = np.dtype({"bool": bool, "complex": complex, "float": float, "int": int}[node.id])
    elif isinstance(node, ast.Name) and node.id == "float128":
        # Preserve the pinned upstream parser's one intentional normalization:
        # it maps NumPy's platform-specific float128 repr to float64.
        dtype = np.dtype(np.float64)
    elif (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "np"
        and node.attr in _ALLOWED_DTYPE_ATTRIBUTES
    ):
        dtype = np.dtype(getattr(np, node.attr))
    else:
        raise SafeParseError("dtype must be an allowlisted scalar dtype")
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise SafeParseError("object, structured, and subarray dtypes are not accepted")
    if dtype.kind not in "biufcSU" or dtype.itemsize <= 0 or dtype.itemsize > 256:
        raise SafeParseError("dtype is outside the reviewed scalar set")
    return dtype


def _numeric_array_data(value: Any) -> tuple[int, int]:
    """Return leaf count and maximum depth before NumPy allocation."""

    stack: list[tuple[Any, int]] = [(value, 1)]
    leaves = 0
    maximum_depth = 0
    while stack:
        item, depth = stack.pop()
        maximum_depth = max(maximum_depth, depth)
        if maximum_depth > MAX_ARRAY_DIMENSIONS + 1:
            raise SafeParseError("array data is too deeply nested")
        if isinstance(item, (list, tuple)):
            stack.extend((child, depth + 1) for child in item)
            continue
        if isinstance(item, bool):
            leaves += 1
        elif isinstance(item, (int, float, complex)):
            leaves += 1
        else:
            raise SafeParseError("arrays may contain only finite numeric scalar literals")
        if leaves > MAX_ARRAY_ELEMENTS:
            raise SafeParseError("array contains too many elements")
    return leaves, maximum_depth


def _safe_array(node: ast.Call, budget: _Budget, np: Any, element_type: Any) -> Any:
    function_is_array = isinstance(node.func, ast.Name) and node.func.id == "array"
    function_is_np_array = (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr == "array"
    )
    if not (function_is_array or function_is_np_array):
        raise SafeParseError("only array, np.array, Element, and empty set calls are accepted")
    if not 1 <= len(node.args) <= 2:
        raise SafeParseError("array requires one data argument and at most one dtype argument")
    if any(keyword.arg is None or keyword.arg != "dtype" for keyword in node.keywords):
        raise SafeParseError("array accepts only the dtype keyword")
    if len(node.keywords) > 1 or (len(node.args) == 2 and node.keywords):
        raise SafeParseError("array dtype may be supplied only once")
    data = _evaluate(node.args[0], budget, np, element_type, top_level=False)
    leaves, _ = _numeric_array_data(data)
    dtype_node = node.args[1] if len(node.args) == 2 else None
    if node.keywords:
        dtype_node = node.keywords[0].value
    dtype = _dtype_from_node(dtype_node, np, budget) if dtype_node is not None else None
    estimated_itemsize = int(dtype.itemsize) if dtype is not None else 16
    if leaves * estimated_itemsize > MAX_ARRAY_BYTES:
        raise SafeParseError("array allocation would exceed the byte limit")
    try:
        array_value = np.array(data, dtype=dtype)
    except (MemoryError, TypeError, ValueError) as exc:
        raise SafeParseError("array data is ragged or incompatible with its dtype") from exc
    if array_value.dtype.hasobject:
        raise SafeParseError("object arrays are not accepted")
    if array_value.ndim > MAX_ARRAY_DIMENSIONS:
        raise SafeParseError("array has too many dimensions")
    if array_value.size > MAX_ARRAY_ELEMENTS or array_value.nbytes > MAX_ARRAY_BYTES:
        raise SafeParseError("array exceeds the reviewed resource limits")
    if array_value.dtype.kind in "fc" and not bool(np.isfinite(array_value).all()):
        raise SafeParseError("non-finite array values are not accepted")
    return array_value


def _insert_unique(mapping: dict[Any, Any], key: Any, value: Any) -> None:
    try:
        duplicate = key in mapping
    except (TypeError, ValueError) as exc:
        raise SafeParseError("dictionary key is not safely hashable") from exc
    if duplicate:
        raise SafeParseError("duplicate dictionary keys are not accepted")
    mapping[key] = value


def _evaluate(
    node: ast.AST,
    budget: _Budget,
    np: Any,
    element_type: Any,
    *,
    top_level: bool,
) -> Any:
    if isinstance(node, ast.Constant):
        return _safe_constant(node.value, budget)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand, budget, np, element_type, top_level=False)
        if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
            raise SafeParseError("unary signs apply only to numeric literals")
        result = value if isinstance(node.op, ast.UAdd) else -value
        return _safe_constant(result, budget)
    if isinstance(node, ast.List):
        budget.add_items(len(node.elts))
        return [_evaluate(item, budget, np, element_type, top_level=False) for item in node.elts]
    if isinstance(node, ast.Tuple):
        budget.add_items(len(node.elts))
        return tuple(
            _evaluate(item, budget, np, element_type, top_level=False) for item in node.elts
        )
    if isinstance(node, ast.Set):
        budget.add_items(len(node.elts))
        values = [_evaluate(item, budget, np, element_type, top_level=False) for item in node.elts]
        try:
            result = set(values)
        except TypeError as exc:
            raise SafeParseError("set values must be safely hashable") from exc
        if len(result) != len(values):
            raise SafeParseError("duplicate set values are not accepted")
        return result
    if isinstance(node, ast.Dict):
        budget.add_items(len(node.keys))
        result: dict[Any, Any] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                raise SafeParseError("dictionary unpacking is not accepted")
            key = _evaluate(key_node, budget, np, element_type, top_level=False)
            if top_level and not isinstance(key, str):
                raise SafeParseError("top-level material-property keys must be strings")
            value = _evaluate(value_node, budget, np, element_type, top_level=False)
            _insert_unique(result, key, value)
        return result
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "set":
            if node.args or node.keywords:
                raise SafeParseError("only an empty set() call is accepted")
            return set()
        if isinstance(node.func, ast.Name) and node.func.id == "Element":
            if len(node.args) != 1 or node.keywords:
                raise SafeParseError("Element requires exactly one symbol literal")
            symbol = _evaluate(node.args[0], budget, np, element_type, top_level=False)
            if not isinstance(symbol, str) or not _ELEMENT_SYMBOL_RE.fullmatch(symbol):
                raise SafeParseError("Element symbol is invalid")
            try:
                return element_type(symbol)
            except (KeyError, TypeError, ValueError) as exc:
                raise SafeParseError("Element symbol is unknown") from exc
        return _safe_array(node, budget, np, element_type)
    raise SafeParseError(f"unsupported expression node: {type(node).__name__}")


class SafeComplexDictParser:
    """Drop-in safe replacement for upstream ``ComplexDictParser``."""

    ultra_safe_parser = True

    def __init__(self, *, element_type: Any | None = None) -> None:
        self._element_type = element_type

    def parse(self, input_str: str) -> dict[str, Any] | None:
        if not isinstance(input_str, str):
            return None
        try:
            encoded = input_str.encode("utf-8", errors="strict")
            if not encoded or len(encoded) > MAX_INPUT_BYTES:
                raise SafeParseError("candidate output is empty or exceeds the byte limit")
            normalized = _legacy_element_syntax(input_str)
            tree = ast.parse(normalized, mode="eval")
            _validate_ast_size(tree)
            import numpy as np

            element_type = self._element_type
            needs_element = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Element"
                for node in ast.walk(tree)
            )
            if element_type is None and needs_element:
                from pymatgen.core import Element

                element_type = Element

            value = _evaluate(tree.body, _Budget(), np, element_type, top_level=True)
            if not isinstance(value, dict) or not value:
                raise SafeParseError("candidate output must be one non-empty dictionary")
            return value
        except (SafeParseError, SyntaxError, UnicodeError, ValueError):
            return None


def parse_complex_string(input_str: str) -> dict[str, Any] | None:
    """Compatibility helper matching the pinned upstream module surface."""

    return SafeComplexDictParser().parse(input_str)
