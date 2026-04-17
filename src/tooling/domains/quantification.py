"""Quantification + statistical reasoning + reporting tool domain."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    pd = None  # type: ignore[assignment]

from src.science.reporting import generate_repro_report
from src.science.stats import (
    compare_two_groups,
    list_curated_stat_tools,
    run_stat_tool,
    summary_statistics,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_counts(condition: dict[str, Any]) -> dict[str, int]:
    counts = condition.get("counts_by_class")
    if isinstance(counts, dict):
        counts_by_class: dict[str, int] = {}
        for key, value in counts.items():
            try:
                counts_by_class[str(key)] = int(value)
            except Exception:
                continue
        return counts_by_class

    summary_rows = condition.get("class_summary")
    if isinstance(summary_rows, list):
        summary_counts: dict[str, int] = {}
        for row in summary_rows:
            if not isinstance(row, dict):
                continue
            cls = str(row.get("class") or row.get("class_name") or "")
            if not cls:
                continue
            try:
                summary_counts[cls] = int(row.get("count", 0))
            except Exception:
                continue
        return summary_counts
    return {}


def _extract_total(condition: dict[str, Any], counts: dict[str, int]) -> int:
    for candidate in (
        condition.get("total_objects"),
        condition.get("total_boxes"),
        (condition.get("metrics") or {}).get("total_boxes")
        if isinstance(condition.get("metrics"), dict)
        else None,
    ):
        try:
            if candidate is not None:
                return int(candidate)
        except Exception:
            continue
    return int(sum(counts.values()))


def _extract_object_rows(condition: dict[str, Any]) -> list[dict[str, Any]]:
    rows = condition.get("object_table")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _load_object_table_input(
    *,
    object_table_path: str | None = None,
    object_table: list[dict[str, Any]] | None = None,
    predictions_json_path: str | None = None,
    predictions: list[dict[str, Any]] | None = None,
    yolo_result: dict[str, Any] | None = None,
    max_object_rows: int = 50000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(object_table, list) and object_table:
        return [row for row in object_table if isinstance(row, dict)], {
            "source": "inline_object_table"
        }

    table_path = str(object_table_path or "").strip()
    if table_path:
        path = Path(table_path).expanduser()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"object_table_path does not exist: {table_path}")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, list):
            raise ValueError("object_table_path must contain a JSON array of per-object rows.")
        return [row for row in parsed if isinstance(row, dict)], {
            "source": "object_table_path",
            "path": str(path.resolve()),
        }

    quantified = quantify_objects(
        predictions_json_path=predictions_json_path,
        predictions=predictions,
        yolo_result=yolo_result,
        max_object_rows=max_object_rows,
    )
    if not bool(quantified.get("success")):
        raise ValueError(
            str(quantified.get("error") or "Failed to quantify detections for plotting.")
        )
    rows = _extract_object_rows(quantified)
    return rows, {
        "source": "quantify_objects",
        "upstream": quantified.get("source"),
    }


def _metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        if metric not in row:
            continue
        raw_value = row.get(metric)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except Exception:
            continue
        if math.isfinite(value):
            vals.append(value)
    return vals


_CSV_LIKE_SUFFIXES = (
    ".csv",
    ".tsv",
    ".tab",
    ".txt",
    ".csv.gz",
    ".tsv.gz",
)

_CSV_DEFAULT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_CSV_DEFAULT_DELIMITERS = (",", "\t", ";", "|")
_CSV_ALLOWED_FILTER_OPS = {
    "==",
    "eq",
    "!=",
    "ne",
    ">",
    "gt",
    ">=",
    "gte",
    "<",
    "lt",
    "<=",
    "lte",
    "in",
    "not_in",
    "contains",
    "startswith",
    "endswith",
    "between",
    "isnull",
    "notnull",
}
_CSV_ALLOWED_AGG_FUNCS = {"sum", "mean", "median", "min", "max", "count", "nunique", "std", "var"}
_CSV_ALLOWED_FILL_METHODS = {"ffill", "bfill", "pad", "backfill"}
_CSV_ALLOWED_KEEP_VALUES = {"first", "last", False}
_CSV_ALLOWED_DROPNA_HOW = {"any", "all"}
_CSV_SAFE_EXPRESSION_RE = re.compile(r"^[A-Za-z0-9_\s+\-*/%().<>=!&|]+$")


def _safe_int(
    value: Any,
    default: int = 0,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if minimum is not None:
        parsed = max(int(minimum), parsed)
    if maximum is not None:
        parsed = min(int(maximum), parsed)
    return parsed


def _is_csv_like_path(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in _CSV_LIKE_SUFFIXES)


def _expand_csv_inputs(file_paths: list[str], *, max_files: int = 200) -> list[Path]:
    expanded: list[Path] = []
    for raw in file_paths:
        if raw is None:
            continue
        path = Path(str(raw)).expanduser()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and _is_csv_like_path(child):
                    expanded.append(child)
                    if len(expanded) >= max_files:
                        return expanded
        elif path.is_file() and _is_csv_like_path(path):
            expanded.append(path)
            if len(expanded) >= max_files:
                return expanded
    return expanded


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _sniff_delimiter(sample_text: str) -> str | None:
    if not sample_text.strip():
        return None
    try:
        sniffed = csv.Sniffer().sniff(sample_text, delimiters="," + "\t" + ";|")
        delimiter = str(getattr(sniffed, "delimiter", "") or "")
        return delimiter or None
    except Exception:
        return None


def _candidate_delimiters(sample_text: str, explicit_delimiter: str | None = None) -> list[str]:
    candidates: list[str] = []
    if explicit_delimiter:
        candidates.append(str(explicit_delimiter))
    sniffed = _sniff_delimiter(sample_text)
    if sniffed:
        candidates.append(sniffed)
    candidates.extend(_CSV_DEFAULT_DELIMITERS)
    return _dedupe_strings([c for c in candidates if c])


def _candidate_encodings(explicit_encoding: str | None = None) -> list[str]:
    candidates: list[str] = []
    if explicit_encoding:
        candidates.append(str(explicit_encoding))
    candidates.extend(_CSV_DEFAULT_ENCODINGS)
    return _dedupe_strings([c for c in candidates if c])


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, bool, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        if pd is not None and isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass
    try:
        as_float = float(value)
        if math.isfinite(as_float):
            return as_float
    except Exception:
        pass
    return str(value)


def _sample_row_widths(sample_text: str, delimiter: str) -> dict[str, Any]:
    widths: Counter[int] = Counter()
    if not sample_text.strip():
        return {"distribution": {}, "is_consistent": True}
    try:
        reader = csv.reader(sample_text.splitlines()[:600], delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            widths[len(row)] += 1
    except Exception:
        return {"distribution": {}, "is_consistent": True}
    ordered = sorted(widths.items(), key=lambda item: (-item[1], item[0]))
    return {
        "distribution": [{"columns": int(k), "rows": int(v)} for k, v in ordered],
        "is_consistent": len(widths) <= 1,
    }


def _delimiter_choice_looks_invalid(frame: Any, sample_text: str, delimiter: str) -> bool:
    lines = [line for line in sample_text.splitlines() if line.strip()]
    if not lines:
        return False
    header = lines[0]
    this_hits = header.count(delimiter)
    alt_hits = max([header.count(d) for d in _CSV_DEFAULT_DELIMITERS if d != delimiter] + [0])
    return this_hits == 0 and alt_hits > 0 and int(frame.shape[1]) <= 1


def _normalize_column_names(frame: Any) -> tuple[Any, list[str], list[str]]:
    current = [str(col).strip() for col in frame.columns]
    issues: list[str] = []
    fixes: list[str] = []
    renamed: list[str] = []
    seen: Counter[str] = Counter()

    duplicates = [name for name, count in Counter(current).items() if name and count > 1]
    if duplicates:
        issues.append("Duplicate column names detected: " + ", ".join(sorted(duplicates)[:8]))

    for index, original in enumerate(current):
        name = original
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{index + 1}"
            fixes.append(f"Renamed blank/unnamed column '{original or '<blank>'}' to '{name}'.")
        seen[name] += 1
        if seen[name] > 1:
            deduped = f"{name}_{seen[name]}"
            fixes.append(f"Renamed duplicate column '{name}' to '{deduped}'.")
            name = deduped
        renamed.append(name)

    if renamed != current:
        frame.columns = renamed
    return frame, issues, fixes


def _read_csv_with_repair(
    path: Path,
    *,
    work_dir: Path,
    delimiter: str | None = None,
    encoding: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    if pd is None:
        raise RuntimeError("pandas is required for analyze_csv but is not installed.")

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"CSV path does not exist: {path}")

    raw_bytes = path.read_bytes()
    sample_bytes = raw_bytes[: min(len(raw_bytes), 262144)]
    line_count = int(raw_bytes.count(b"\n")) + (1 if raw_bytes else 0)

    parse_issues: list[str] = []
    fixes_applied: list[str] = []
    parse_attempts: list[str] = []
    working_path = path
    source_variant = "original"

    if b"\x00" in raw_bytes:
        cleaned_path = work_dir / f"{path.stem}__nul_cleaned.csv"
        cleaned_path.write_bytes(raw_bytes.replace(b"\x00", b""))
        working_path = cleaned_path
        source_variant = "nul_cleaned"
        fixes_applied.append("Removed NUL bytes prior to parsing.")

    sample_text = ""
    decode_notes: list[str] = []
    for enc in _candidate_encodings(encoding):
        try:
            sample_text = sample_bytes.decode(enc)
            break
        except Exception:
            decode_notes.append(f"{enc}: decode error")
    if not sample_text:
        sample_text = sample_bytes.decode("utf-8", errors="ignore")
        decode_notes.append("utf-8(ignore-errors): fallback decode")
        parse_issues.append("Could not cleanly decode sample bytes with preferred encodings.")

    delim_candidates = _candidate_delimiters(sample_text, delimiter)
    enc_candidates = _candidate_encodings(encoding)

    selected_encoding: str | None = None
    selected_delimiter: str | None = None
    parse_mode = "strict"
    dataframe = None

    for enc in enc_candidates:
        for delim in delim_candidates:
            parse_attempts.append(f"strict encoding={enc} delimiter={delim!r}")
            try:
                dataframe = pd.read_csv(
                    working_path,
                    encoding=enc,
                    sep=delim,
                    engine="python",
                    on_bad_lines="error",
                )
                if _delimiter_choice_looks_invalid(dataframe, sample_text, delim):
                    parse_attempts.append(
                        "rejected: delimiter likely incorrect based on header structure"
                    )
                    dataframe = None
                    continue
                selected_encoding = enc
                selected_delimiter = delim
                break
            except Exception as exc:
                parse_attempts.append(f"failed: {type(exc).__name__}: {exc}")
        if dataframe is not None:
            break

    if dataframe is None:
        parse_mode = "tolerant"
        parse_issues.append("Strict CSV parsing failed; fallback parser skipped malformed rows.")
        for enc in enc_candidates:
            for delim in delim_candidates:
                parse_attempts.append(f"tolerant encoding={enc} delimiter={delim!r}")
                try:
                    dataframe = pd.read_csv(
                        working_path,
                        encoding=enc,
                        sep=delim,
                        engine="python",
                        on_bad_lines="skip",
                    )
                    if _delimiter_choice_looks_invalid(dataframe, sample_text, delim):
                        parse_attempts.append(
                            "rejected: delimiter likely incorrect based on header structure"
                        )
                        dataframe = None
                        continue
                    selected_encoding = enc
                    selected_delimiter = delim
                    fixes_applied.append(
                        "Used tolerant parser with on_bad_lines='skip' to recover malformed rows."
                    )
                    break
                except Exception as exc:
                    parse_attempts.append(f"failed: {type(exc).__name__}: {exc}")
            if dataframe is not None:
                break

    if dataframe is None:
        raise ValueError(
            "Failed to parse CSV after strict+tolerant attempts. " + "; ".join(parse_attempts[-6:])
        )

    row_widths = _sample_row_widths(sample_text, selected_delimiter or ",")
    if not row_widths.get("is_consistent", True):
        parse_issues.append("Inconsistent row field counts detected in sampled CSV lines.")

    dataframe, column_issues, column_fixes = _normalize_column_names(dataframe)
    parse_issues.extend(column_issues)
    fixes_applied.extend(column_fixes)

    malformed_rows_estimate: int | None = None
    if line_count > 0:
        malformed_rows_estimate = max(0, line_count - (len(dataframe) + 1))
        if parse_mode == "tolerant" and malformed_rows_estimate > 0:
            parse_issues.append(
                f"Estimated malformed/skipped rows: {malformed_rows_estimate} (approximate)."
            )

    metadata = {
        "source_path": str(path),
        "source_variant": source_variant,
        "working_path": str(working_path),
        "selected_encoding": selected_encoding,
        "selected_delimiter": selected_delimiter,
        "parse_mode": parse_mode,
        "line_count_estimate": line_count,
        "malformed_rows_estimate": malformed_rows_estimate,
        "decode_notes": decode_notes,
        "row_width_distribution": row_widths.get("distribution", []),
        "issues_detected": _dedupe_strings(
            [str(item) for item in parse_issues if str(item).strip()]
        ),
        "fixes_applied": _dedupe_strings(
            [str(item) for item in fixes_applied if str(item).strip()]
        ),
        "parse_attempts_tail": parse_attempts[-12:],
    }
    return dataframe, metadata


def _build_filter_mask(frame: Any, filters: list[dict[str, Any]], logic: str) -> Any:
    if pd is None:
        raise RuntimeError("pandas is required for analyze_csv but is not installed.")
    if not filters:
        return pd.Series([True] * len(frame), index=frame.index)

    masks: list[Any] = []
    for raw_filter in filters:
        if not isinstance(raw_filter, dict):
            raise ValueError("Each filter entry must be an object.")
        column = str(raw_filter.get("column") or "").strip()
        if not column:
            raise ValueError("Each filter must include a non-empty 'column'.")
        if column not in frame.columns:
            raise ValueError(f"Filter column does not exist: {column}")
        op = str(raw_filter.get("op") or raw_filter.get("operator") or "eq").strip().lower()
        if op not in _CSV_ALLOWED_FILTER_OPS:
            raise ValueError(f"Unsupported filter operator '{op}'.")
        series = frame[column]
        value = raw_filter.get("value")

        if op in {"==", "eq"}:
            mask = series == value
        elif op in {"!=", "ne"}:
            mask = series != value
        elif op in {">", "gt"}:
            mask = series > value
        elif op in {">=", "gte"}:
            mask = series >= value
        elif op in {"<", "lt"}:
            mask = series < value
        elif op in {"<=", "lte"}:
            mask = series <= value
        elif op == "in":
            if not isinstance(value, list):
                raise ValueError("Operator 'in' requires list 'value'.")
            mask = series.isin(value)
        elif op == "not_in":
            if not isinstance(value, list):
                raise ValueError("Operator 'not_in' requires list 'value'.")
            mask = ~series.isin(value)
        elif op == "contains":
            mask = series.astype(str).str.contains(str(value), na=False, regex=False)
        elif op == "startswith":
            mask = series.astype(str).str.startswith(str(value), na=False)
        elif op == "endswith":
            mask = series.astype(str).str.endswith(str(value), na=False)
        elif op == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("Operator 'between' requires value=[lower, upper].")
            lower, upper = value
            mask = (series >= lower) & (series <= upper)
        elif op == "isnull":
            mask = series.isna()
        elif op == "notnull":
            mask = series.notna()
        else:
            raise ValueError(f"Unsupported filter operator '{op}'.")

        masks.append(mask.fillna(False))

    if logic == "or":
        final_mask = masks[0]
        for extra in masks[1:]:
            final_mask = final_mask | extra
        return final_mask

    final_mask = masks[0]
    for extra in masks[1:]:
        final_mask = final_mask & extra
    return final_mask


def _apply_csv_operations(
    frame: Any,
    operations: list[dict[str, Any]] | None,
    *,
    fail_on_error: bool = False,
) -> tuple[Any, list[dict[str, Any]], list[str]]:
    if pd is None:
        raise RuntimeError("pandas is required for analyze_csv but is not installed.")

    if not operations:
        return frame, [], []

    operation_logs: list[dict[str, Any]] = []
    operation_issues: list[str] = []
    transformed = frame.copy()

    for index, raw_op in enumerate(operations):
        before_shape = [int(transformed.shape[0]), int(transformed.shape[1])]
        if not isinstance(raw_op, dict):
            issue = f"Operation #{index + 1} is not an object."
            operation_logs.append(
                {
                    "index": index,
                    "operation": "unknown",
                    "status": "error",
                    "details": issue,
                    "shape_before": before_shape,
                    "shape_after": before_shape,
                }
            )
            operation_issues.append(issue)
            if fail_on_error:
                raise ValueError(issue)
            continue

        op_name = str(raw_op.get("operation") or raw_op.get("op") or "").strip().lower()
        if not op_name:
            issue = f"Operation #{index + 1} missing 'operation' field."
            operation_logs.append(
                {
                    "index": index,
                    "operation": "unknown",
                    "status": "error",
                    "details": issue,
                    "shape_before": before_shape,
                    "shape_after": before_shape,
                }
            )
            operation_issues.append(issue)
            if fail_on_error:
                raise ValueError(issue)
            continue

        try:
            detail = "applied"
            if op_name in {"select_columns", "select"}:
                columns = [str(item) for item in (raw_op.get("columns") or []) if str(item).strip()]
                if not columns:
                    raise ValueError("select_columns requires a non-empty 'columns' list.")
                missing = [col for col in columns if col not in transformed.columns]
                if missing:
                    raise ValueError("select_columns missing columns: " + ", ".join(missing[:10]))
                transformed = transformed.loc[:, columns].copy()
                detail = f"Selected {len(columns)} columns."
            elif op_name in {"rename_columns", "rename"}:
                mapping = raw_op.get("mapping")
                if not isinstance(mapping, dict) or not mapping:
                    raise ValueError("rename_columns requires a non-empty 'mapping' object.")
                transformed = transformed.rename(
                    columns={str(k): str(v) for k, v in mapping.items()}
                )
                detail = f"Renamed {len(mapping)} columns."
            elif op_name in {"filter_rows", "filter"}:
                logic = str(raw_op.get("logic") or "and").strip().lower()
                if logic not in {"and", "or"}:
                    raise ValueError("filter_rows supports logic='and' or logic='or'.")
                filters = raw_op.get("filters")
                query = raw_op.get("query")
                if isinstance(query, str) and query.strip():
                    if not _CSV_SAFE_EXPRESSION_RE.fullmatch(query.strip()):
                        raise ValueError("filter_rows query contains unsupported characters.")
                    transformed = transformed.query(query.strip(), engine="python")
                    detail = "Filtered rows via query."
                else:
                    if not isinstance(filters, list) or not filters:
                        raise ValueError(
                            "filter_rows requires non-empty 'filters' or a 'query' string."
                        )
                    mask = _build_filter_mask(transformed, filters, logic=logic)
                    transformed = transformed.loc[mask].copy()
                    detail = f"Filtered rows with {len(filters)} condition(s) ({logic})."
            elif op_name in {"sort_values", "sort"}:
                by = raw_op.get("by")
                if isinstance(by, str):
                    by = [by]
                if not isinstance(by, list) or not by:
                    raise ValueError("sort_values requires 'by' column(s).")
                by_cols = [str(item) for item in by]
                missing = [col for col in by_cols if col not in transformed.columns]
                if missing:
                    raise ValueError("sort_values missing columns: " + ", ".join(missing[:10]))
                ascending = raw_op.get("ascending", True)
                transformed = transformed.sort_values(by=by_cols, ascending=ascending)
                detail = f"Sorted by {', '.join(by_cols[:6])}."
            elif op_name in {"dropna"}:
                how = str(raw_op.get("how") or "any").strip().lower()
                if how not in _CSV_ALLOWED_DROPNA_HOW:
                    raise ValueError("dropna supports how='any' or how='all'.")
                subset_raw = raw_op.get("subset")
                subset = None
                if isinstance(subset_raw, list) and subset_raw:
                    subset = [str(item) for item in subset_raw]
                transformed = transformed.dropna(how=how, subset=subset)
                detail = "Dropped NA rows."
            elif op_name in {"fillna"}:
                values = raw_op.get("values")
                method = raw_op.get("method")
                if isinstance(values, dict) and values:
                    transformed = transformed.fillna(value=values)
                    detail = f"Filled NA values for {len(values)} column(s)."
                elif method is not None:
                    method_name = str(method).strip().lower()
                    if method_name not in _CSV_ALLOWED_FILL_METHODS:
                        raise ValueError(
                            "fillna method must be one of: ffill, bfill, pad, backfill."
                        )
                    transformed = transformed.fillna(method=method_name)
                    detail = f"Filled NA values with method '{method_name}'."
                else:
                    raise ValueError("fillna requires either 'values' map or 'method'.")
            elif op_name in {"drop_duplicates", "dedupe"}:
                keep = raw_op.get("keep", "first")
                if keep not in _CSV_ALLOWED_KEEP_VALUES:
                    raise ValueError("drop_duplicates keep must be 'first', 'last', or false.")
                subset_raw = raw_op.get("subset")
                subset = None
                if isinstance(subset_raw, list) and subset_raw:
                    subset = [str(item) for item in subset_raw]
                transformed = transformed.drop_duplicates(subset=subset, keep=keep)
                detail = "Dropped duplicate rows."
            elif op_name in {"append_rows", "append"}:
                rows = raw_op.get("rows")
                if not isinstance(rows, list) or not rows:
                    raise ValueError("append_rows requires non-empty 'rows'.")
                append_df = pd.DataFrame([row for row in rows if isinstance(row, dict)])
                if append_df.empty:
                    raise ValueError("append_rows contained no valid row objects.")
                transformed = pd.concat([transformed, append_df], ignore_index=True, sort=False)
                detail = f"Appended {len(append_df)} row(s)."
            elif op_name in {"add_column", "assign_column"}:
                column_name = str(raw_op.get("name") or "").strip()
                if not column_name:
                    raise ValueError("add_column requires a non-empty 'name'.")
                if "expression" in raw_op:
                    expression = str(raw_op.get("expression") or "").strip()
                    if not expression:
                        raise ValueError("add_column expression is empty.")
                    if not _CSV_SAFE_EXPRESSION_RE.fullmatch(expression):
                        raise ValueError("add_column expression contains unsupported characters.")
                    transformed[column_name] = transformed.eval(expression, engine="python")
                    detail = f"Added column '{column_name}' from expression."
                elif "from_column" in raw_op:
                    source = str(raw_op.get("from_column") or "").strip()
                    if source not in transformed.columns:
                        raise ValueError(f"add_column source column not found: {source}")
                    transformed[column_name] = transformed[source]
                    detail = f"Added column '{column_name}' copied from '{source}'."
                elif "value" in raw_op:
                    transformed[column_name] = raw_op.get("value")
                    detail = f"Added constant column '{column_name}'."
                else:
                    raise ValueError("add_column requires one of: expression, from_column, value.")
            elif op_name in {"groupby_agg", "groupby"}:
                by = raw_op.get("by")
                if isinstance(by, str):
                    by = [by]
                if not isinstance(by, list) or not by:
                    raise ValueError("groupby_agg requires non-empty 'by' list.")
                by_cols = [str(item) for item in by]
                missing = [col for col in by_cols if col not in transformed.columns]
                if missing:
                    raise ValueError(
                        "groupby_agg missing group columns: " + ", ".join(missing[:10])
                    )
                aggregations = raw_op.get("aggregations")
                if not isinstance(aggregations, dict) or not aggregations:
                    raise ValueError("groupby_agg requires non-empty 'aggregations'.")
                safe_aggs: dict[str, Any] = {}
                for column, funcs in aggregations.items():
                    column_name = str(column)
                    if column_name not in transformed.columns:
                        raise ValueError(f"groupby_agg missing aggregation column: {column_name}")
                    if isinstance(funcs, str):
                        funcs_list = [funcs]
                    elif isinstance(funcs, list):
                        funcs_list = [str(item) for item in funcs]
                    else:
                        raise ValueError(
                            f"groupby_agg aggregations for '{column_name}' must be str/list."
                        )
                    filtered = [f for f in funcs_list if f in _CSV_ALLOWED_AGG_FUNCS]
                    if not filtered:
                        raise ValueError(
                            f"groupby_agg for '{column_name}' has no allowed funcs ({sorted(_CSV_ALLOWED_AGG_FUNCS)})."
                        )
                    safe_aggs[column_name] = filtered if len(filtered) > 1 else filtered[0]
                transformed = (
                    transformed.groupby(by_cols, dropna=False).agg(safe_aggs).reset_index()
                )
                if hasattr(transformed.columns, "to_flat_index"):
                    flattened = []
                    for col in transformed.columns.to_flat_index():
                        if isinstance(col, tuple):
                            flattened.append(
                                "_".join([str(part) for part in col if str(part) and part != ""])
                            )
                        else:
                            flattened.append(str(col))
                    transformed.columns = flattened
                detail = "Grouped and aggregated data."
            elif op_name in {"limit_rows", "head", "tail"}:
                n = _safe_int(raw_op.get("n", 10), default=10, minimum=1, maximum=1000000)
                transformed = transformed.tail(n) if op_name == "tail" else transformed.head(n)
                detail = f"Kept {n} row(s) via {op_name}."
            else:
                raise ValueError(
                    "Unsupported operation '"
                    + op_name
                    + "'. Supported: select_columns, rename_columns, filter_rows, sort_values, "
                    + "dropna, fillna, drop_duplicates, append_rows, add_column, groupby_agg, limit_rows/head/tail."
                )

            after_shape = [int(transformed.shape[0]), int(transformed.shape[1])]
            operation_logs.append(
                {
                    "index": index,
                    "operation": op_name,
                    "status": "applied",
                    "details": detail,
                    "shape_before": before_shape,
                    "shape_after": after_shape,
                }
            )
        except Exception as exc:
            after_shape = [int(transformed.shape[0]), int(transformed.shape[1])]
            issue = f"Operation '{op_name}' failed: {exc}"
            operation_logs.append(
                {
                    "index": index,
                    "operation": op_name,
                    "status": "error",
                    "details": issue,
                    "shape_before": before_shape,
                    "shape_after": after_shape,
                }
            )
            operation_issues.append(issue)
            if fail_on_error:
                raise ValueError(issue) from exc

    return transformed, operation_logs, operation_issues


def _column_profile(frame: Any, *, max_columns: int = 40) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    total_rows = len(frame)
    for column in list(frame.columns)[:max_columns]:
        series = frame[column]
        non_null = int(series.notna().sum())
        missing = int(total_rows - non_null)
        unique = int(series.nunique(dropna=True))
        sample_values: list[Any] = []
        for value in series.dropna().head(3).tolist():
            sample_values.append(_json_safe(value))
        profiles.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "non_null": non_null,
                "missing": missing,
                "missing_fraction": round((missing / total_rows), 6) if total_rows else 0.0,
                "unique_values": unique,
                "sample_values": sample_values,
            }
        )
    return profiles


def _numeric_summary(frame: Any, *, max_columns: int = 20) -> list[dict[str, Any]]:
    if pd is None:
        return []
    numeric_cols = [str(col) for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])][
        :max_columns
    ]
    if not numeric_cols:
        return []

    summary_rows: list[dict[str, Any]] = []
    for column in numeric_cols:
        series = frame[column]
        finite = series.dropna()
        if finite.empty:
            continue
        summary_rows.append(
            {
                "column": column,
                "count": int(finite.shape[0]),
                "mean": _json_safe(float(finite.mean())),
                "median": _json_safe(float(finite.median())),
                "std": _json_safe(float(finite.std(ddof=1))) if finite.shape[0] > 1 else 0.0,
                "min": _json_safe(float(finite.min())),
                "max": _json_safe(float(finite.max())),
            }
        )
    return summary_rows


def _preview_rows(frame: Any, *, n: int = 12, max_columns: int = 24) -> list[dict[str, Any]]:
    cols = list(frame.columns)[:max_columns]
    preview = frame.loc[:, cols].head(max(1, n))
    rows: list[dict[str, Any]] = []
    for record in preview.to_dict(orient="records"):
        rows.append({str(key): _json_safe(value) for key, value in record.items()})
    return rows


def analyze_csv(
    file_paths: list[str],
    operations: list[dict[str, Any]] | None = None,
    preview_rows: int = 12,
    output_dir: str | None = None,
    write_output: bool = True,
    fail_on_operation_error: bool = False,
    delimiter: str | None = None,
    encoding: str | None = None,
) -> dict[str, Any]:
    """Load, validate, repair, and transform CSV/TSV tabular files with common operations."""
    if pd is None:
        return {
            "success": False,
            "error": "pandas is not installed in this runtime. Install pandas>=2.x for analyze_csv.",
        }

    if not file_paths:
        return {"success": False, "error": "file_paths is required."}

    preview_rows = _safe_int(preview_rows, default=12, minimum=1, maximum=200)
    operation_list = [item for item in (operations or []) if isinstance(item, dict)]

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    if output_dir:
        output_root = Path(output_dir).expanduser()
    else:
        output_root = Path("data") / "csv_analysis" / f"run-{run_id}"
    output_root.mkdir(parents=True, exist_ok=True)

    expanded_paths = _expand_csv_inputs([str(path) for path in file_paths], max_files=400)
    if not expanded_paths:
        return {
            "success": False,
            "error": "No CSV-like files found in file_paths. Supported suffixes: .csv, .tsv, .tab, .txt.",
        }

    results: list[dict[str, Any]] = []
    top_level_issues: list[str] = []
    aggregated_ui_artifacts: list[dict[str, Any]] = []

    for source_path in expanded_paths:
        file_result: dict[str, Any] = {
            "source_path": str(source_path),
            "success": False,
        }
        try:
            file_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", source_path.stem).strip("._") or "table"
            file_work_dir = output_root / file_slug
            file_work_dir.mkdir(parents=True, exist_ok=True)

            loaded_frame, load_meta = _read_csv_with_repair(
                source_path,
                work_dir=file_work_dir,
                delimiter=delimiter,
                encoding=encoding,
            )
            rows_before = int(loaded_frame.shape[0])
            cols_before = int(loaded_frame.shape[1])

            transformed_frame, operation_logs, operation_issues = _apply_csv_operations(
                loaded_frame,
                operation_list,
                fail_on_error=bool(fail_on_operation_error),
            )

            rows_after = int(transformed_frame.shape[0])
            cols_after = int(transformed_frame.shape[1])
            duplicate_rows = int(transformed_frame.duplicated().sum()) if rows_after > 0 else 0

            output_csv_path = None
            if write_output:
                suffix = "transformed" if operation_logs else "cleaned"
                output_csv = file_work_dir / f"{file_slug}__{suffix}.csv"
                transformed_frame.to_csv(output_csv, index=False)
                output_csv_path = str(output_csv)

            issues_detected = _dedupe_strings(
                [
                    *[str(item) for item in load_meta.get("issues_detected", [])],
                    *[str(item) for item in operation_issues],
                ]
            )
            fixes_applied = _dedupe_strings(
                [str(item) for item in load_meta.get("fixes_applied", [])]
            )

            if duplicate_rows > 0:
                issues_detected.append(
                    f"Duplicate rows detected after transformations: {duplicate_rows}."
                )

            preview_payload = _preview_rows(transformed_frame, n=preview_rows)
            numeric_summary = _numeric_summary(transformed_frame)
            column_profile = _column_profile(transformed_frame)

            ui_artifacts: list[dict[str, Any]] = [
                {
                    "type": "metrics",
                    "title": f"CSV summary: {source_path.name}",
                    "payload": {
                        "rows_before": rows_before,
                        "rows_after": rows_after,
                        "columns_before": cols_before,
                        "columns_after": cols_after,
                        "operations_applied": len(
                            [row for row in operation_logs if row.get("status") == "applied"]
                        ),
                        "issues_detected": len(issues_detected),
                    },
                },
                {
                    "type": "table",
                    "title": f"CSV preview: {source_path.name}",
                    "payload": preview_payload,
                },
            ]
            if numeric_summary:
                ui_artifacts.append(
                    {
                        "type": "table",
                        "title": f"Numeric summary: {source_path.name}",
                        "payload": numeric_summary,
                    }
                )
            if numeric_summary and len(numeric_summary) >= 1:
                chart_rows = [
                    {"column": row["column"], "mean": row["mean"]}
                    for row in numeric_summary[:12]
                    if isinstance(row.get("mean"), (int, float))
                ]
                if chart_rows:
                    ui_artifacts.append(
                        {
                            "type": "chart",
                            "kind": "bar",
                            "title": f"Column mean values: {source_path.name}",
                            "x": "column",
                            "y": "mean",
                            "data": chart_rows,
                        }
                    )

            file_result.update(
                {
                    "success": True,
                    "rows_before": rows_before,
                    "columns_before": cols_before,
                    "rows_after": rows_after,
                    "columns_after": cols_after,
                    "duplicate_rows_after": duplicate_rows,
                    "parse": {
                        "selected_encoding": load_meta.get("selected_encoding"),
                        "selected_delimiter": load_meta.get("selected_delimiter"),
                        "parse_mode": load_meta.get("parse_mode"),
                        "line_count_estimate": load_meta.get("line_count_estimate"),
                        "malformed_rows_estimate": load_meta.get("malformed_rows_estimate"),
                        "row_width_distribution": load_meta.get("row_width_distribution"),
                        "decode_notes": load_meta.get("decode_notes"),
                    },
                    "issues_detected": issues_detected,
                    "fixes_applied": fixes_applied,
                    "operations": operation_logs,
                    "output_csv_path": output_csv_path,
                    "preview_rows": preview_payload,
                    "numeric_summary": numeric_summary,
                    "column_profile": column_profile,
                    "ui_artifacts": ui_artifacts,
                }
            )
            aggregated_ui_artifacts.extend(ui_artifacts[:6])
        except Exception as exc:
            message = str(exc)
            file_result.update(
                {
                    "success": False,
                    "error": message,
                }
            )
            top_level_issues.append(f"{source_path.name}: {message}")
        results.append(file_result)

    successes = [row for row in results if row.get("success")]
    failures = [row for row in results if not row.get("success")]

    return {
        "success": len(successes) > 0,
        "processed_files": len(successes),
        "failed_files": len(failures),
        "results": results,
        "issues_detected": top_level_issues,
        "output_directory": str(output_root),
        "message": (
            f"Processed {len(successes)} CSV file(s) with validation/repair and "
            f"{len(operation_list)} operation(s)."
            if successes
            else "No CSV files were successfully processed."
        ),
        "ui_artifacts": aggregated_ui_artifacts[:16],
    }


def _load_prediction_payload(
    predictions_json_path: str | None = None,
    predictions: list[dict[str, Any]] | None = None,
    yolo_result: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata: dict[str, Any] = {}

    if isinstance(yolo_result, dict):
        embedded_predictions = yolo_result.get("predictions")
        if isinstance(embedded_predictions, list) and embedded_predictions:
            predictions = embedded_predictions
        embedded_json_path = yolo_result.get("predictions_json")
        if not predictions_json_path and isinstance(embedded_json_path, str):
            predictions_json_path = embedded_json_path

    if predictions_json_path:
        path = Path(predictions_json_path)
        if not path.exists():
            raise FileNotFoundError(f"predictions_json_path does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("predictions_json must contain a JSON object")
        data_predictions = payload.get("predictions")
        if isinstance(data_predictions, list):
            predictions = data_predictions
        metadata = {
            "source_path": str(path),
            "model_name": payload.get("model_name"),
            "model_path": payload.get("model_path"),
            "run_id": payload.get("run_id"),
        }

    if not predictions:
        raise ValueError(
            "No predictions were provided. Supply predictions_json_path, predictions, or yolo_result."
        )

    cleaned = [p for p in predictions if isinstance(p, dict)]
    if not cleaned:
        raise ValueError("Predictions payload did not contain valid prediction objects.")
    return cleaned, metadata


def quantify_objects(
    predictions_json_path: str | None = None,
    predictions: list[dict[str, Any]] | None = None,
    yolo_result: dict[str, Any] | None = None,
    include_classes: list[str] | None = None,
    min_confidence: float = 0.0,
    pixel_size: float | None = None,
    pixel_unit: str = "px",
    max_object_rows: int = 50000,
) -> dict[str, Any]:
    """Summarize detection outputs into measurement-ready scientific tables."""
    try:
        prediction_rows, metadata = _load_prediction_payload(
            predictions_json_path=predictions_json_path,
            predictions=predictions,
            yolo_result=yolo_result,
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    threshold = min(max(_safe_float(min_confidence, default=0.0), 0.0), 1.0)
    class_filter = {str(c).strip() for c in (include_classes or []) if str(c).strip()}
    use_filter = len(class_filter) > 0
    max_object_rows = max(100, min(int(max_object_rows), 250000))

    per_image: list[dict[str, Any]] = []
    counts_by_class: dict[str, int] = {}
    confidences_by_class: dict[str, list[float]] = {}
    area_px_by_class: dict[str, list[float]] = {}
    all_confidences: list[float] = []
    object_rows: list[dict[str, Any]] = []

    for pred in prediction_rows:
        image_path = str(pred.get("path") or "")
        boxes = pred.get("boxes")
        if not isinstance(boxes, list):
            boxes = []

        per_class_local: dict[str, int] = {}
        kept = 0
        for idx, box in enumerate(boxes):
            if not isinstance(box, dict):
                continue
            class_name = str(box.get("class_name") or box.get("class_id") or "unknown")
            if use_filter and class_name not in class_filter:
                continue
            confidence = _safe_float(box.get("confidence"), default=0.0)
            if confidence < threshold:
                continue

            xyxy = box.get("xyxy")
            if not (isinstance(xyxy, list) and len(xyxy) == 4):
                continue
            x1, y1, x2, y2 = [_safe_float(v) for v in xyxy]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area_px = width * height
            aspect_ratio = (width / height) if height > 1e-12 else None
            eq_diameter_px = math.sqrt((4.0 * area_px) / math.pi) if area_px > 0 else 0.0
            centroid_x = (x1 + x2) / 2.0
            centroid_y = (y1 + y2) / 2.0

            kept += 1
            counts_by_class[class_name] = counts_by_class.get(class_name, 0) + 1
            per_class_local[class_name] = per_class_local.get(class_name, 0) + 1
            all_confidences.append(confidence)
            confidences_by_class.setdefault(class_name, []).append(confidence)
            area_px_by_class.setdefault(class_name, []).append(area_px)

            row = {
                "object_id": f"{Path(image_path).stem}_{idx}",
                "image_path": image_path,
                "class": class_name,
                "confidence": round(confidence, 6),
                "width_px": round(width, 6),
                "height_px": round(height, 6),
                "bbox_area_px": round(area_px, 6),
                "aspect_ratio": round(aspect_ratio, 6) if aspect_ratio is not None else None,
                "equivalent_diameter_px": round(eq_diameter_px, 6),
                "centroid_x_px": round(centroid_x, 6),
                "centroid_y_px": round(centroid_y, 6),
            }
            if pixel_size is not None and _safe_float(pixel_size, default=0.0) > 0:
                scale = _safe_float(pixel_size, default=1.0)
                row[f"bbox_area_{pixel_unit}2"] = round(area_px * (scale * scale), 6)
                row[f"equivalent_diameter_{pixel_unit}"] = round(eq_diameter_px * scale, 6)

            if len(object_rows) < max_object_rows:
                object_rows.append(row)

        per_image.append(
            {
                "image_path": image_path,
                "objects": int(kept),
                "counts_by_class": per_class_local,
            }
        )

    total_objects = int(sum(counts_by_class.values()))
    classes_detected = len(counts_by_class)
    mean_conf = round(sum(all_confidences) / len(all_confidences), 6) if all_confidences else 0.0
    median_conf = round(median(all_confidences), 6) if all_confidences else 0.0

    class_summary: list[dict[str, Any]] = []
    for class_name, count in sorted(counts_by_class.items(), key=lambda item: (-item[1], item[0])):
        cls_conf = confidences_by_class.get(class_name, [])
        conf_mean = round(sum(cls_conf) / len(cls_conf), 6) if cls_conf else 0.0
        summary_row: dict[str, Any] = {
            "class": class_name,
            "count": int(count),
            "mean_confidence": conf_mean,
            "median_confidence": round(median(cls_conf), 6) if cls_conf else 0.0,
        }

        areas = area_px_by_class.get(class_name, [])
        if areas:
            area_mean_px = sum(areas) / len(areas)
            summary_row["mean_box_area_px"] = round(area_mean_px, 3)
            if pixel_size is not None and _safe_float(pixel_size, default=0.0) > 0:
                scale = _safe_float(pixel_size, default=1.0)
                area_units = area_mean_px * (scale * scale)
                summary_row[f"mean_box_area_{pixel_unit}2"] = round(area_units, 6)
        class_summary.append(summary_row)

    distribution_summary: dict[str, Any] = {}
    for metric in (
        "bbox_area_px",
        "width_px",
        "height_px",
        "aspect_ratio",
        "equivalent_diameter_px",
    ):
        vals = _metric_values(object_rows, metric)
        if len(vals) >= 2:
            distribution_summary[metric] = summary_statistics(vals)

    ui_artifacts: list[dict[str, Any]] = [
        {
            "type": "metrics",
            "title": "Quantification summary",
            "payload": {
                "total_objects": total_objects,
                "classes_detected": classes_detected,
                "mean_confidence": mean_conf,
                "median_confidence": median_conf,
            },
        },
        {
            "type": "chart",
            "kind": "bar",
            "title": "Object counts by class",
            "data": [{"class": row["class"], "count": row["count"]} for row in class_summary],
            "x": "class",
            "y": "count",
        },
        {
            "type": "table",
            "title": "Class quantification",
            "payload": class_summary,
        },
    ]

    measurements = [
        {"name": "total_objects", "value": total_objects, "unit": "count"},
        {"name": "classes_detected", "value": classes_detected, "unit": "count"},
        {"name": "mean_confidence", "value": mean_conf, "unit": "score"},
    ]

    return {
        "success": True,
        "source": metadata,
        "filters": {
            "min_confidence": threshold,
            "include_classes": sorted(class_filter) if class_filter else [],
        },
        "total_objects": total_objects,
        "counts_by_class": counts_by_class,
        "class_summary": class_summary,
        "per_image": per_image,
        "object_table": object_rows,
        "object_row_count": len(object_rows),
        "distribution_summary": distribution_summary,
        "confidence": {
            "mean": mean_conf,
            "median": median_conf,
            "n": len(all_confidences),
        },
        "measurements": measurements,
        "ui_artifacts": ui_artifacts,
    }


def plot_quantified_detections(
    object_table_path: str | None = None,
    object_table: list[dict[str, Any]] | None = None,
    predictions_json_path: str | None = None,
    predictions: list[dict[str, Any]] | None = None,
    yolo_result: dict[str, Any] | None = None,
    confidence_threshold: float = 0.6,
    source_image_path: str | None = None,
    output_dir: str | None = None,
    max_object_rows: int = 50000,
) -> dict[str, Any]:
    """Create deterministic matplotlib/seaborn plots from quantified detection rows."""
    try:
        rows, source_meta = _load_object_table_input(
            object_table_path=object_table_path,
            object_table=object_table,
            predictions_json_path=predictions_json_path,
            predictions=predictions,
            yolo_result=yolo_result,
            max_object_rows=max_object_rows,
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    if not rows:
        return {"success": False, "error": "No quantified detections were available for plotting."}
    if pd is None:  # pragma: no cover - dependency guard
        return {"success": False, "error": "pandas is required for plot_quantified_detections."}

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - dependency guard
        return {"success": False, "error": f"matplotlib unavailable: {exc}"}

    try:  # pragma: no cover - exercised opportunistically depending on env
        import seaborn as sns  # type: ignore
    except Exception:
        sns = None  # type: ignore[assignment]

    threshold = min(max(_safe_float(confidence_threshold, default=0.6), 0.0), 1.0)
    frame = pd.DataFrame(rows)
    if frame.empty or "confidence" not in frame.columns:
        return {"success": False, "error": "Quantified detections are missing confidence values."}
    if "class" not in frame.columns:
        frame["class"] = "unknown"

    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    frame = frame.dropna(subset=["confidence"]).copy()
    if frame.empty:
        return {
            "success": False,
            "error": "No numeric confidence values were available for plotting.",
        }
    frame["class"] = frame["class"].astype(str)
    frame["is_low_confidence"] = frame["confidence"] < threshold

    source_name = ""
    for candidate in (source_image_path, object_table_path, predictions_json_path):
        token = str(candidate or "").strip()
        if token:
            source_name = Path(token).name
            break
    if not source_name and isinstance(source_meta.get("path"), str):
        source_name = Path(str(source_meta["path"])).name
    source_name = source_name or "detections"

    if output_dir:
        plot_dir = Path(str(output_dir)).expanduser()
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        plot_dir = (
            Path("data") / "quantification_plots" / f"{stamp}-{source_name.replace('.', '_')}"
        )
    plot_dir.mkdir(parents=True, exist_ok=True)

    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    accent = "#ff4fd8"
    accent_secondary = "#00f5ff"
    low_confidence_fill = "#ff5c8a"

    image_title = Path(source_name).name
    total_count = len(frame)
    low_count = int(frame["is_low_confidence"].sum())
    counts_by_class = {
        str(key): int(value) for key, value in frame["class"].value_counts().sort_index().items()
    }
    low_counts_by_class = {
        str(key): int(value)
        for key, value in frame.loc[frame["is_low_confidence"], "class"]
        .value_counts()
        .sort_index()
        .items()
    }

    generated_files: list[dict[str, str]] = []

    def _register_output(path: Path, title: str, caption: str) -> None:
        generated_files.append(
            {
                "path": str(path.resolve()),
                "title": title,
                "caption": caption,
            }
        )

    overall_hist_path = plot_dir / "overall_confidence_histogram.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = max(10, min(20, max(1, total_count // 2)))
    ax.hist(frame["confidence"], bins=bins, color=accent_secondary, alpha=0.85, edgecolor="black")
    ax.axvspan(0.0, threshold, color=low_confidence_fill, alpha=0.18)
    ax.axvline(threshold, color=low_confidence_fill, linestyle="--", linewidth=2)
    ax.set_title(
        f"Detection confidence distribution\n{image_title}", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Detections")
    ax.text(
        0.99,
        0.97,
        f"Low-confidence detections: {low_count}/{total_count}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(overall_hist_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    _register_output(
        overall_hist_path,
        "Overall confidence histogram",
        f"Histogram of all detection confidences for {image_title}. The shaded red band marks scores below {threshold:.2f}.",
    )

    class_distribution_path = plot_dir / "confidence_by_class_distribution.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    order = sorted(frame["class"].dropna().astype(str).unique().tolist())
    palette = [accent if idx % 2 == 0 else accent_secondary for idx, _ in enumerate(order)]
    if sns is not None:
        sns.boxplot(
            data=frame,
            x="class",
            y="confidence",
            order=order,
            palette=palette,
            ax=ax,
        )
    else:
        grouped = [
            frame.loc[frame["class"] == cls, "confidence"].astype(float).tolist() for cls in order
        ]
        box = ax.boxplot(
            grouped,
            patch_artist=True,
            tick_labels=order,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"edgecolor": "black", "linewidth": 1.2},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
        )
        for patch, color in zip(box["boxes"], palette, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
    ax.axhline(threshold, color=low_confidence_fill, linestyle="--", linewidth=2)
    ax.set_title(f"Confidence by class\n{image_title}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Confidence score")
    fig.tight_layout()
    fig.savefig(class_distribution_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    _register_output(
        class_distribution_path,
        "Confidence by class",
        f"Box plot of detection confidence by class for {image_title}. The dashed line marks the {threshold:.2f} review threshold.",
    )

    low_confidence_plot_path = plot_dir / "low_confidence_review.png"
    has_centroids = {"centroid_x_px", "centroid_y_px"}.issubset(set(frame.columns))
    if has_centroids:
        fig, ax = plt.subplots(figsize=(10, 6))
        background_drawn = False
        image_path = (
            Path(str(source_image_path or "")).expanduser()
            if str(source_image_path or "").strip()
            else None
        )
        if image_path is not None and image_path.exists() and image_path.is_file():
            try:
                bg = plt.imread(str(image_path))
                ax.imshow(bg)
                background_drawn = True
            except Exception:
                background_drawn = False
        high = frame.loc[~frame["is_low_confidence"]].copy()
        low = frame.loc[frame["is_low_confidence"]].copy()
        if not high.empty:
            if sns is not None:
                sns.scatterplot(
                    data=high,
                    x="centroid_x_px",
                    y="centroid_y_px",
                    hue="class",
                    palette="cool",
                    alpha=0.7,
                    s=70,
                    ax=ax,
                    legend=True,
                )
            else:
                for idx, cls in enumerate(sorted(high["class"].astype(str).unique().tolist())):
                    cls_rows = high.loc[high["class"] == cls]
                    ax.scatter(
                        cls_rows["centroid_x_px"],
                        cls_rows["centroid_y_px"],
                        s=70,
                        alpha=0.7,
                        c=palette[idx % len(palette)],
                        edgecolors="none",
                        label=cls,
                    )
        if not low.empty:
            ax.scatter(
                low["centroid_x_px"],
                low["centroid_y_px"],
                s=140,
                c=low_confidence_fill,
                marker="X",
                edgecolors="black",
                linewidths=0.7,
                label=f"confidence < {threshold:.2f}",
            )
        if background_drawn:
            ax.set_title(
                f"Low-confidence detections on image\n{image_title}", fontsize=16, fontweight="bold"
            )
        else:
            ax.set_title(
                f"Low-confidence detections in image coordinates\n{image_title}",
                fontsize=16,
                fontweight="bold",
            )
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.invert_yaxis()
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(low_confidence_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        _register_output(
            low_confidence_plot_path,
            "Low-confidence spatial review",
            f"Spatial review plot for {image_title}. Red X markers highlight detections below {threshold:.2f}.",
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        low_frame = pd.DataFrame(
            [
                {
                    "class": cls,
                    "low_confidence_count": int(low_counts_by_class.get(cls, 0)),
                    "total_count": int(counts_by_class.get(cls, 0)),
                }
                for cls in sorted(counts_by_class)
            ]
        )
        if sns is not None:
            sns.barplot(
                data=low_frame,
                x="class",
                y="low_confidence_count",
                color=low_confidence_fill,
                ax=ax,
            )
        else:
            ax.bar(
                low_frame["class"].astype(str).tolist(),
                low_frame["low_confidence_count"].astype(int).tolist(),
                color=low_confidence_fill,
                edgecolor="black",
                linewidth=1.0,
            )
        ax.set_title(
            f"Low-confidence detections by class\n{image_title}", fontsize=16, fontweight="bold"
        )
        ax.set_xlabel("Class")
        ax.set_ylabel(f"Detections below {threshold:.2f}")
        fig.tight_layout()
        fig.savefig(low_confidence_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        _register_output(
            low_confidence_plot_path,
            "Low-confidence review",
            f"Counts of detections below {threshold:.2f} by class for {image_title}.",
        )

    artifacts = [
        {
            "path": item["path"],
            "title": item["title"],
        }
        for item in generated_files
    ]
    ui_artifacts = [
        {
            "type": "image",
            "title": item["title"],
            "path": item["path"],
            "caption": item["caption"],
        }
        for item in generated_files
    ]

    summary = {
        "image_name": image_title,
        "total_detections": total_count,
        "confidence_threshold": threshold,
        "low_confidence_count": low_count,
        "low_confidence_fraction": round(low_count / total_count, 6) if total_count else 0.0,
        "counts_by_class": counts_by_class,
        "low_confidence_by_class": low_counts_by_class,
        "generated_plot_count": len(generated_files),
    }

    return {
        "success": True,
        "source": source_meta,
        "summary": summary,
        "output_directory": str(plot_dir.resolve()),
        "output_files": [item["path"] for item in generated_files],
        "artifacts": artifacts,
        "ui_artifacts": ui_artifacts,
        "plots": generated_files,
    }


def compare_conditions(
    condition_a: dict[str, Any],
    condition_b: dict[str, Any],
    condition_a_name: str = "condition_a",
    condition_b_name: str = "condition_b",
    normalize_to_total: bool = True,
    pseudocount: float = 0.5,
    top_k: int = 30,
    metrics: list[str] | None = None,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compare two quantified conditions with counts + effect sizes + CI."""
    if not isinstance(condition_a, dict) or not isinstance(condition_b, dict):
        return {"success": False, "error": "condition_a and condition_b must both be objects"}

    counts_a = _extract_counts(condition_a)
    counts_b = _extract_counts(condition_b)
    if not counts_a and not counts_b:
        return {
            "success": False,
            "error": "No class counts found. Provide outputs from quantify_objects or yolo_detect.",
        }

    total_a = max(_extract_total(condition_a, counts_a), 1)
    total_b = max(_extract_total(condition_b, counts_b), 1)
    pseudo = max(_safe_float(pseudocount, default=0.5), 1e-9)
    max_rows = max(1, min(int(_safe_float(top_k, default=30)), 500))

    all_classes = sorted(set(counts_a) | set(counts_b))
    comparison_rows: list[dict[str, Any]] = []
    for cls in all_classes:
        a = int(counts_a.get(cls, 0))
        b = int(counts_b.get(cls, 0))
        ratio = (b + pseudo) / (a + pseudo)
        log2_fc = math.log2(ratio)
        row = {
            "class": cls,
            f"{condition_a_name}_count": a,
            f"{condition_b_name}_count": b,
            "delta_count": b - a,
            "fold_change": round(ratio, 6),
            "log2_fold_change": round(log2_fc, 6),
        }
        if normalize_to_total:
            p_a = a / total_a
            p_b = b / total_b
            row[f"{condition_a_name}_proportion"] = round(p_a, 6)
            row[f"{condition_b_name}_proportion"] = round(p_b, 6)
            row["delta_proportion"] = round(p_b - p_a, 6)
        comparison_rows.append(row)

    comparison_rows.sort(
        key=lambda r: abs(_safe_float(r.get("log2_fold_change"), 0.0)), reverse=True
    )
    trimmed = comparison_rows[:max_rows]

    metrics_payload = {
        f"{condition_a_name}_total": int(sum(counts_a.values())),
        f"{condition_b_name}_total": int(sum(counts_b.values())),
        "total_delta": int(sum(counts_b.values()) - sum(counts_a.values())),
        "classes_compared": len(all_classes),
    }

    rows_a = _extract_object_rows(condition_a)
    rows_b = _extract_object_rows(condition_b)
    metric_names = metrics or [
        "confidence",
        "bbox_area_px",
        "width_px",
        "height_px",
        "aspect_ratio",
        "equivalent_diameter_px",
    ]
    statistical_analysis: list[dict[str, Any]] = []
    for metric in metric_names:
        vals_a = _metric_values(rows_a, metric)
        vals_b = _metric_values(rows_b, metric)
        if len(vals_a) < 2 or len(vals_b) < 2:
            continue
        analysis = compare_two_groups(
            vals_a,
            vals_b,
            metric_name=metric,
            alpha=float(alpha),
            test="auto",
            random_seed=int(random_seed),
        )
        statistical_analysis.append(analysis)

    statistical_analysis.sort(
        key=lambda item: abs(_safe_float((item.get("effect_sizes") or {}).get("cohen_d"), 0.0)),
        reverse=True,
    )

    ui_artifacts: list[dict[str, Any]] = [
        {
            "type": "metrics",
            "title": "Condition comparison summary",
            "payload": metrics_payload,
        },
        {
            "type": "table",
            "title": f"{condition_b_name} vs {condition_a_name} (class-level)",
            "payload": trimmed,
        },
        {
            "type": "chart",
            "kind": "bar",
            "title": "Class delta counts",
            "data": [{"class": row["class"], "delta_count": row["delta_count"]} for row in trimmed],
            "x": "class",
            "y": "delta_count",
        },
    ]
    if statistical_analysis:
        ui_artifacts.append(
            {
                "type": "table",
                "title": "Statistical reasoning (effect sizes + CI)",
                "payload": [
                    {
                        "metric": item.get("metric"),
                        "mean_diff": item.get("mean_diff"),
                        "mean_diff_ci95": item.get("mean_diff_ci95"),
                        "cohen_d": (item.get("effect_sizes") or {}).get("cohen_d"),
                        "cliffs_delta": (item.get("effect_sizes") or {}).get("cliffs_delta"),
                        "p_value": (item.get("test") or {}).get("p_value"),
                        "significance": (item.get("test") or {}).get("significance"),
                    }
                    for item in statistical_analysis[:50]
                ],
            }
        )

    top_class = trimmed[0]["class"] if trimmed else None
    summary_parts = [
        f"Compared {len(all_classes)} classes.",
        f"{condition_b_name} total objects: {sum(counts_b.values())}; {condition_a_name}: {sum(counts_a.values())}.",
    ]
    if top_class:
        summary_parts.append(
            f"Largest class-level shift: {top_class} (delta={trimmed[0].get('delta_count')})."
        )
    if statistical_analysis:
        top_metric = statistical_analysis[0]
        summary_parts.append(
            f"Strongest continuous-metric shift: {top_metric.get('metric')} "
            f"(d={(top_metric.get('effect_sizes') or {}).get('cohen_d'):.3f})."
        )

    return {
        "success": True,
        "condition_a_name": condition_a_name,
        "condition_b_name": condition_b_name,
        "normalize_to_total": bool(normalize_to_total),
        "pseudocount": pseudo,
        "alpha": float(alpha),
        "counts": {
            condition_a_name: counts_a,
            condition_b_name: counts_b,
        },
        "class_count_comparison": trimmed,
        "statistical_analysis": statistical_analysis,
        "summary": " ".join(summary_parts),
        "ui_artifacts": ui_artifacts,
    }


def stats_list_curated_tools() -> dict[str, Any]:
    tools = list_curated_stat_tools()
    return {
        "success": True,
        "count": len(tools),
        "tools": tools,
        "ui_artifacts": [
            {
                "type": "table",
                "title": "Curated statistical tools",
                "payload": tools,
            }
        ],
    }


def stats_run_curated_tool(tool_name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    result = run_stat_tool(tool_name, payload or {})
    if not result.get("success"):
        return result
    return {
        **result,
        "ui_artifacts": [
            {
                "type": "write",
                "title": f"Statistical result: {tool_name}",
                "payload": result.get("result"),
            }
        ],
    }


def repro_report(
    run_id: str | None = None,
    title: str | None = None,
    result_summary: str | None = None,
    measurements: list[dict[str, Any]] | None = None,
    statistical_analysis: list[dict[str, Any]] | dict[str, Any] | None = None,
    qc_warnings: list[str] | None = None,
    limitations: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
    next_steps: list[dict[str, Any] | str] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    return generate_repro_report(
        run_id=run_id,
        title=title,
        result_summary=result_summary,
        measurements=measurements,
        statistical_analysis=statistical_analysis,
        qc_warnings=qc_warnings,
        limitations=limitations,
        provenance=provenance,
        next_steps=next_steps,
        output_dir=output_dir,
    )


ANALYZE_CSV_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_csv",
        "description": (
            "Validate and analyze CSV/TSV tabular files with robust parsing/repair fallback. "
            "Reports what is wrong with malformed files, records fixes applied, and supports "
            "common pandas-style dataframe operations (select, rename, filter, sort, fill/drop NA, "
            "dedupe, append rows, add columns, groupby aggregation, row limits)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "description": "CSV/TSV file paths (or directories containing such files).",
                    "items": {"type": "string"},
                },
                "operations": {
                    "type": "array",
                    "description": (
                        "Optional ordered dataframe operations. Supported operation values include: "
                        "select_columns, rename_columns, filter_rows, sort_values, dropna, fillna, "
                        "drop_duplicates, append_rows, add_column, groupby_agg, limit_rows/head/tail."
                    ),
                    "items": {"type": "object"},
                },
                "preview_rows": {
                    "type": "integer",
                    "description": "Number of preview rows to return.",
                    "default": 12,
                    "minimum": 1,
                    "maximum": 200,
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional output directory for cleaned/transformed CSVs.",
                },
                "write_output": {
                    "type": "boolean",
                    "description": "Write cleaned/transformed CSV artifacts to disk.",
                    "default": True,
                },
                "fail_on_operation_error": {
                    "type": "boolean",
                    "description": "If true, stop and fail on the first dataframe operation error.",
                    "default": False,
                },
                "delimiter": {
                    "type": "string",
                    "description": "Optional explicit delimiter override (e.g., ',', '\\t', ';').",
                },
                "encoding": {
                    "type": "string",
                    "description": "Optional explicit file encoding override (e.g., utf-8, cp1252).",
                },
            },
            "required": ["file_paths"],
        },
    },
}


QUANTIFY_OBJECTS_TOOL = {
    "type": "function",
    "function": {
        "name": "quantify_objects",
        "description": (
            "Convert detections into measurement-ready tables. Computes per-object morphology proxies "
            "(area, width, height, aspect ratio, equivalent diameter, centroids), class distributions, "
            "and confidence summaries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "predictions_json_path": {
                    "type": "string",
                    "description": "Path to predictions JSON output produced by yolo_detect.",
                },
                "predictions": {
                    "type": "array",
                    "description": "Inline prediction list (same shape as yolo_detect['predictions']).",
                    "items": {"type": "object"},
                },
                "yolo_result": {
                    "type": "object",
                    "description": "Full output object returned by yolo_detect.",
                    "additionalProperties": True,
                },
                "include_classes": {
                    "type": "array",
                    "description": "Optional class-name filter.",
                    "items": {"type": "string"},
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold in [0,1].",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "pixel_size": {
                    "type": "number",
                    "description": "Optional physical size per pixel for area conversion.",
                },
                "pixel_unit": {
                    "type": "string",
                    "description": "Unit label used with pixel_size (e.g., um, nm).",
                    "default": "px",
                },
                "max_object_rows": {
                    "type": "integer",
                    "description": "Max per-object rows to return to the model.",
                    "default": 50000,
                    "minimum": 100,
                    "maximum": 250000,
                },
            },
            "required": [],
        },
    },
}


PLOT_QUANTIFIED_DETECTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "plot_quantified_detections",
        "description": (
            "Generate deterministic matplotlib/seaborn figures from quantified detection rows or detector predictions. "
            "Produces confidence-distribution plots and highlights detections below a chosen confidence threshold."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_table_path": {
                    "type": "string",
                    "description": "Path to a JSON array of quantified per-object rows (for example from quantify_objects).",
                },
                "object_table": {
                    "type": "array",
                    "description": "Inline quantified per-object rows.",
                    "items": {"type": "object"},
                },
                "predictions_json_path": {
                    "type": "string",
                    "description": "Optional predictions JSON path; used when a quantified object table is not already available.",
                },
                "predictions": {
                    "type": "array",
                    "description": "Inline prediction list (same shape as yolo_detect['predictions']).",
                    "items": {"type": "object"},
                },
                "yolo_result": {
                    "type": "object",
                    "description": "Full output object returned by yolo_detect.",
                    "additionalProperties": True,
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Confidence threshold used to highlight uncertain detections.",
                    "default": 0.6,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "source_image_path": {
                    "type": "string",
                    "description": "Optional source image used as a background for spatial low-confidence review plots.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional output directory for generated PNG plots.",
                },
                "max_object_rows": {
                    "type": "integer",
                    "description": "Maximum per-object rows to materialize when quantifying from raw predictions.",
                    "default": 50000,
                    "minimum": 100,
                    "maximum": 250000,
                },
            },
            "required": [],
        },
    },
}


QUANTIFY_SEGMENTATION_MASKS_TOOL = {
    "type": "function",
    "function": {
        "name": "quantify_segmentation_masks",
        "description": (
            "Quantify binary segmentation masks into per-mask morphology summaries. "
            "Computes coverage, connected-component object counts, and optional overlap metrics "
            "when ground-truth masks are provided. Prefer mask artifacts produced by segmentation tools "
            "(for example preferred_upload_paths), not raw source images."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mask_paths": {
                    "type": "array",
                    "description": "Predicted mask artifact paths (.npy, .tif/.tiff, .nii/.nii.gz, or binary mask images). Do not pass raw source images here.",
                    "items": {"type": "string"},
                },
                "ground_truth_paths": {
                    "type": "array",
                    "description": "Optional ground-truth mask paths for overlap metrics.",
                    "items": {"type": "string"},
                },
                "pair_map": {
                    "type": "object",
                    "description": "Optional explicit mask->ground-truth path map.",
                    "additionalProperties": {"type": "string"},
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold for binarizing non-binary masks.",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "match_by_stem": {
                    "type": "boolean",
                    "description": "When no pair_map is provided, pair by normalized filename stem.",
                    "default": True,
                },
                "min_component_size": {
                    "type": "integer",
                    "description": "Ignore connected components smaller than this size (pixels/voxels).",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 1000000000,
                },
                "pixel_size": {
                    "type": "number",
                    "description": "Optional physical size per pixel/voxel for area/volume conversion.",
                },
                "pixel_unit": {
                    "type": "string",
                    "description": "Unit label used with pixel_size (e.g., um, nm).",
                    "default": "px",
                },
                "stem_strip_tokens": {
                    "type": "array",
                    "description": "Optional list of trailing tokens stripped during stem matching.",
                    "items": {"type": "string"},
                },
            },
            "required": ["mask_paths"],
        },
    },
}


COMPARE_CONDITIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "compare_conditions",
        "description": (
            "Compare two quantified conditions with class-level deltas and statistical reasoning "
            "(effect sizes, CI, and test selection logic) on continuous metrics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "condition_a": {
                    "type": "object",
                    "description": "Baseline quantification output (typically from quantify_objects).",
                    "additionalProperties": True,
                },
                "condition_b": {
                    "type": "object",
                    "description": "Comparison quantification output (typically from quantify_objects).",
                    "additionalProperties": True,
                },
                "condition_a_name": {
                    "type": "string",
                    "description": "Baseline label.",
                    "default": "condition_a",
                },
                "condition_b_name": {
                    "type": "string",
                    "description": "Comparison label.",
                    "default": "condition_b",
                },
                "normalize_to_total": {
                    "type": "boolean",
                    "description": "If true, includes proportion-normalized class deltas.",
                    "default": True,
                },
                "pseudocount": {
                    "type": "number",
                    "description": "Smoothing value used in fold-change calculations.",
                    "default": 0.5,
                    "minimum": 0.0,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max number of class-comparison rows to return.",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 500,
                },
                "metrics": {
                    "type": "array",
                    "description": "Continuous metrics to compare from object_table rows.",
                    "items": {"type": "string"},
                },
                "alpha": {
                    "type": "number",
                    "description": "Significance threshold for test interpretation.",
                    "default": 0.05,
                    "minimum": 0.0001,
                    "maximum": 0.2,
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Seed for bootstrap/approximation procedures.",
                    "default": 42,
                },
            },
            "required": ["condition_a", "condition_b"],
        },
    },
}


STATS_LIST_CURATED_TOOLS_TOOL = {
    "type": "function",
    "function": {
        "name": "stats_list_curated_tools",
        "description": "List curated statistical tools available for deterministic analysis.",
        "parameters": {"type": "object", "properties": {}},
    },
}


STATS_RUN_CURATED_TOOL = {
    "type": "function",
    "function": {
        "name": "stats_run_curated_tool",
        "description": (
            "Run a curated statistical tool by name. Use stats_list_curated_tools first, then run with "
            "explicit payload."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Curated tool name from stats_list_curated_tools.",
                },
                "payload": {
                    "type": "object",
                    "description": "Tool-specific input payload.",
                    "additionalProperties": True,
                },
            },
            "required": ["tool_name"],
        },
    },
}


REPRO_REPORT_TOOL = {
    "type": "function",
    "function": {
        "name": "repro_report",
        "description": (
            "Generate a reproducible scientific report artifact (Markdown + JSON bundle) with methods, "
            "measurements, statistical reasoning, QC warnings, limitations, and provenance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Optional run id for artifact scoping.",
                },
                "title": {"type": "string"},
                "result_summary": {"type": "string"},
                "measurements": {"type": "array", "items": {"type": "object"}},
                "statistical_analysis": {
                    "oneOf": [{"type": "object"}, {"type": "array", "items": {"type": "object"}}],
                },
                "qc_warnings": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "provenance": {"type": "object", "additionalProperties": True},
                "next_steps": {
                    "type": "array",
                    "items": {"oneOf": [{"type": "string"}, {"type": "object"}]},
                },
                "output_dir": {"type": "string"},
            },
            "required": [],
        },
    },
}


ANALYSIS_TOOL_SCHEMAS = [
    ANALYZE_CSV_TOOL,
    QUANTIFY_OBJECTS_TOOL,
    PLOT_QUANTIFIED_DETECTIONS_TOOL,
    QUANTIFY_SEGMENTATION_MASKS_TOOL,
    COMPARE_CONDITIONS_TOOL,
    STATS_LIST_CURATED_TOOLS_TOOL,
    STATS_RUN_CURATED_TOOL,
    REPRO_REPORT_TOOL,
]


__all__ = [
    "ANALYSIS_TOOL_SCHEMAS",
    "ANALYZE_CSV_TOOL",
    "COMPARE_CONDITIONS_TOOL",
    "PLOT_QUANTIFIED_DETECTIONS_TOOL",
    "QUANTIFY_OBJECTS_TOOL",
    "QUANTIFY_SEGMENTATION_MASKS_TOOL",
    "REPRO_REPORT_TOOL",
    "STATS_LIST_CURATED_TOOLS_TOOL",
    "STATS_RUN_CURATED_TOOL",
    "analyze_csv",
    "compare_conditions",
    "plot_quantified_detections",
    "quantify_objects",
    "repro_report",
    "stats_list_curated_tools",
    "stats_run_curated_tool",
]
