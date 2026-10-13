"""Minimal LLM Agents for sculpting, reading, auditing, and source recovery."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from compresstab.evidence import (
    AnswerabilitySignature,
    DerivedObservable,
    DerivedOperand,
    EvidenceResolution,
    EvidenceSlice,
)
from compresstab.llm.client import JsonLLMClient


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


def normalize_structured_answer(value: Any) -> Any:
    """Unwrap a bounded stack of complete serialized containers."""

    current = value
    for _depth in range(3):
        if not isinstance(current, str):
            break
        text = current.strip()
        delimiters = (text[0], text[-1]) if len(text) >= 2 else ("", "")
        if delimiters not in {("[", "]"), ("{", "}"), ('"', '"')}:
            break
        parsed = None
        parsers = (json.loads,) if delimiters == ('"', '"') else (json.loads, ast.literal_eval)
        for parser in parsers:
            try:
                candidate = parser(text)
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(candidate, (list, dict, str)):
                parsed = candidate
                break
        if parsed is None or parsed == current:
            break
        current = parsed
    return current if isinstance(current, (list, dict)) else value


class _LLMAgent:
    def __init__(self, llm: JsonLLMClient, model: str):
        self.llm = llm
        self.model = model

    def _ask(self, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        data, _raw = self.llm.chat_json(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=_json(payload),
            temperature=0.0,
        )
        if not isinstance(data, dict):
            raise ValueError("LLM Agent returned non-object JSON")
        return data


def _valid_id(value: Any, prefix: str, length: int) -> str | None:
    text = str(value or "")
    if not text.startswith(prefix):
        text = f"{prefix}{text}"
    suffix = text[len(prefix):]
    return text if suffix.isdigit() and 0 <= int(suffix) < length else None


def clean_signature(
    data: dict[str, Any],
    *,
    table_sketch: dict[str, Any],
    allowed_ref_ids: set[str] | None = None,
) -> AnswerabilitySignature:
    """Validate an Agent signature without inventing question-specific repairs."""

    sources = [item for item in table_sketch.get("sources", []) if isinstance(item, dict)]
    source_by_id = {str(item.get("id")): item for item in sources if item.get("id")}
    raw_slices = data.get("slices", []) if isinstance(data, dict) else []
    if isinstance(raw_slices, dict):
        raw_slices = [raw_slices]
    cleaned: list[EvidenceSlice] = []
    allowed_modes = {"exact", "identity", "equality", "order", "threshold"}
    allowed_predicates = {
        "eq", "ne", "lt", "le", "gt", "ge", "contains", "starts_with",
        "ends_with", "in", "between", "under",
    }

    for raw in raw_slices or []:
        if not isinstance(raw, dict):
            continue
        source_id = str(raw.get("source", "") or "")
        if source_id not in source_by_id:
            source_id = next(iter(source_by_id), "t0")
        source = source_by_id.get(source_id, {})
        n_rows, n_cols = [int(value or 0) for value in (source.get("shape") or [0, 0])[:2]]
        visible_rows = {
            str(item.get("row")) for item in source.get("evidence", [])
            if isinstance(item, dict) and item.get("row")
        }

        def columns(values: Any, *, symbolic_all: bool = False) -> list[str]:
            result = []
            for value in values or []:
                if symbolic_all and str(value).strip() == "*":
                    if "*" not in result:
                        result.append("*")
                    continue
                if (col_id := _valid_id(value, "c", n_cols)) and col_id not in result:
                    result.append(col_id)
            return result

        selected_cols = columns(raw.get("columns", []), symbolic_all=True)
        if not selected_cols:
            selected_cols = [
                str(item.get("id")) for item in source.get("headers", [])
                if isinstance(item, dict) and item.get("id")
            ]

        rows = []
        for value in raw.get("rows", []) or []:
            row_id = _valid_id(value, "r", n_rows)
            if not row_id:
                continue
            if allowed_ref_ids is None and row_id not in visible_rows:
                continue
            if allowed_ref_ids is not None and not any(
                ref.startswith(f"{source_id}:{row_id}:") for ref in allowed_ref_ids
            ):
                continue
            if row_id not in rows:
                rows.append(row_id)

        where = []
        for item in raw.get("where", []) or []:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                item = {"col": item[0], "op": item[1], "value": item[2]}
            if not isinstance(item, dict):
                continue
            col_id = columns([item.get("col")])
            operator = str(item.get("op", "eq") or "eq").lower()
            if col_id and operator in allowed_predicates:
                where.append({"col": col_id[0], "op": operator, "value": item.get("value")})

        links = []
        for raw_link in raw.get("links", []) or []:
            link = columns(
                raw_link if isinstance(raw_link, list) else [],
                symbolic_all="*" in selected_cols,
            )
            if "*" not in selected_cols:
                link = [col_id for col_id in link if col_id in selected_cols]
            if len(link) >= 2 and link not in links:
                links.append(link)
            elif link == ["*"] and link not in links:
                links.append(link)

        resolutions = []
        seen_resolution: set[
            tuple[str, str, float | None, str | None, str | None, tuple[str, ...]]
        ] = set()
        for item in raw.get("resolution", []) or []:
            if not isinstance(item, dict):
                continue
            col_ids = columns([item.get("column")])
            mode = str(item.get("mode", "identity") or "identity").lower()
            if not col_ids or col_ids[0] not in selected_cols or mode not in allowed_modes:
                continue
            boundary = None
            threshold_operator = None
            order_kind = None
            anchors: list[str] = []
            if mode == "threshold":
                try:
                    boundary = float(item.get("boundary"))
                except (TypeError, ValueError):
                    mode = "exact"
                candidate_operator = str(item.get("operator", "") or "").lower()
                if candidate_operator in {"lt", "le", "gt", "ge"}:
                    threshold_operator = candidate_operator
            elif mode == "equality":
                raw_anchors = item.get("anchors", [])
                if not isinstance(raw_anchors, list):
                    raw_anchors = [raw_anchors]
                anchors = list(dict.fromkeys(
                    str(value).strip() for value in raw_anchors
                    if str(value).strip()
                ))[:8]
            elif mode == "order":
                candidate_order = str(item.get("order_kind", "value") or "value").lower()
                order_kind = candidate_order if candidate_order in {"value", "source"} else "value"
            col_id = col_ids[0]
            resolution_key = (
                col_id,
                mode,
                boundary,
                threshold_operator,
                order_kind,
                tuple(anchors),
            )
            if resolution_key in seen_resolution:
                continue
            seen_resolution.add(resolution_key)
            resolutions.append(EvidenceResolution(
                column=col_id,
                mode=mode,
                boundary=boundary,
                operator=threshold_operator,
                anchors=anchors,
                order_kind=order_kind,
            ))

        derived = []
        seen_observables: set[str] = set()
        raw_derived = [
            item for item in (raw.get("derived", []) or [])
            if isinstance(item, dict)
        ]
        declared_observable_ids = {
            str(item.get("observable_id", "") or "").strip()
            for item in raw_derived
            if re.fullmatch(
                r"v\d+", str(item.get("observable_id", "") or "").strip()
            )
        }
        for raw_observable in raw_derived:
            observable_id = str(raw_observable.get("observable_id", "") or "").strip()
            operator = str(raw_observable.get("operator", "") or "").strip().lower()
            mode = str(raw_observable.get("mode", "order") or "order").strip().lower()
            direction = str(
                raw_observable.get("direction", "descending") or "descending"
            ).strip().lower()
            if (
                not re.fullmatch(r"v\d+", observable_id)
                or observable_id in seen_observables
                or operator not in {
                    "difference", "ratio", "percent_change",
                    "sum", "mean", "min", "max", "range", "magnitude",
                }
                or mode not in {"exact", "order", "threshold"}
                or direction not in {"ascending", "descending"}
            ):
                continue
            operands = []
            for raw_operand in raw_observable.get("operands", []) or []:
                if not isinstance(raw_operand, dict):
                    continue
                raw_node_id = str(raw_operand.get("column", "") or "").strip()
                col_ids = columns([raw_node_id])
                node_id = (
                    col_ids[0]
                    if col_ids and col_ids[0] in selected_cols
                    else raw_node_id
                    if raw_node_id in declared_observable_ids
                    else ""
                )
                role = str(raw_operand.get("role", "") or "").strip().lower()
                value_mode = str(
                    raw_operand.get("value_mode", "signed") or "signed"
                ).strip().lower()
                if (
                    node_id
                    and node_id != observable_id
                    and role in {
                        "value", "input", "left", "right", "numerator",
                        "denominator", "earlier", "later", "group", "rank",
                    }
                ):
                    operands.append(DerivedOperand(
                        column=node_id,
                        role=role,
                        value_mode=(value_mode if value_mode in ANSWER_VALUE_MODES else "signed"),
                    ))
            observed_roles = {item.role for item in operands}
            roles_valid = {
                "difference": (
                    {"earlier", "later"} <= observed_roles
                    or {"left", "right"} <= observed_roles
                ),
                "ratio": {"numerator", "denominator"} <= observed_roles,
                "percent_change": {"earlier", "later"} <= observed_roles,
                "sum": bool(observed_roles & {"value", "input", "group", "rank", "left", "right"}),
                "mean": bool(observed_roles & {"value", "input", "group", "rank", "left", "right"}),
                "min": bool(observed_roles & {"value", "input", "group", "rank", "left", "right"}),
                "max": bool(observed_roles & {"value", "input", "group", "rank", "left", "right"}),
                "range": bool(observed_roles & {"value", "input", "group", "rank", "left", "right"}),
                "magnitude": bool(observed_roles & {"value", "input", "group", "rank", "left", "right"}),
            }[operator]
            if not operands or not roles_valid:
                continue
            boundary = raw_observable.get("boundary")
            threshold_operator = str(
                raw_observable.get("threshold_operator", "") or ""
            ).lower() or None
            if mode == "threshold":
                try:
                    boundary = float(boundary)
                except (TypeError, ValueError):
                    continue
                if threshold_operator not in {"lt", "le", "gt", "ge"}:
                    continue
            else:
                boundary = None
                threshold_operator = None
            try:
                scale = float(raw_observable.get("scale", 1) or 1)
            except (TypeError, ValueError):
                scale = 1.0
            derived.append(DerivedObservable(
                observable_id=observable_id,
                operator=operator,
                operands=operands,
                mode=mode,
                direction=direction,
                boundary=boundary,
                threshold_operator=threshold_operator,
                unit=str(raw_observable.get("unit", "number") or "number"),
                scale=scale,
                required=bool(raw_observable.get("required", True)),
            ))
            seen_observables.add(observable_id)

        where_logic = str(raw.get("where_logic", raw.get("logic", "all")) or "all").lower()
        if where_logic not in {"all", "any"}:
            where_logic = "all"

        cleaned.append(EvidenceSlice(
            source=source_id,
            columns=selected_cols,
            where=where,
            where_logic=where_logic,
            rows=rows,
            links=links,
            resolution=resolutions,
            derived=derived,
        ))

    if not cleaned and sources:
        source = sources[0]
        cols = [
            str(item.get("id")) for item in source.get("headers", [])
            if isinstance(item, dict) and item.get("id")
        ]
        cleaned.append(EvidenceSlice(
            source=str(source.get("id", "t0")),
            columns=cols,
            links=[cols] if len(cols) >= 2 else [],
            resolution=[EvidenceResolution(column=col_id, mode="exact") for col_id in cols],
        ))
    return AnswerabilitySignature(slices=cleaned)


def visible_scale_labels(text: str) -> set[str]:
    """Read explicit numeric scales from Reader-visible metadata and headers."""

    structural_lines = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if (
            re.match(r"^-\s*scale\s*:", stripped, flags=re.I)
            or re.match(r"^-\s*c\d+\s*=", stripped, flags=re.I)
            or stripped.startswith("source_row\t")
            or stripped.startswith("Provenance:")
        ):
            structural_lines.append(stripped)
        elif re.match(r"^r\d+\t", stripped):
            # The first visible field is the row/measure label; units embedded
            # there (for example, "cash generation (£m)") are structural.
            cells = stripped.split("\t")
            if len(cells) >= 2:
                structural_lines.append(cells[1])
    normalized = " ".join(structural_lines).replace("’", "'").replace("`", "'")
    labels: set[str] = set()
    if re.search(
        r"\b(?:thousands?|000s?)\b|(?:[$€£¥]|rmb|usd|eur|gbp)\s*'?000\b|\b(?:usd|eur|gbp)\s*k\b",
        normalized,
        flags=re.I,
    ):
        labels.add("thousand")
    if re.search(
        r"\bmillions?\b|(?:[$€£¥]|rmb|usd|eur|gbp)\s*m\b|\b(?:usd|eur|gbp)\s*mn\b",
        normalized,
        flags=re.I,
    ):
        labels.add("million")
    if re.search(r"\bbillions?\b|\b(?:usd|eur|gbp)\s*bn\b", normalized, flags=re.I):
        labels.add("billion")
    return labels


def answer_lexicalizes_scale(answer: Any, scale: str) -> bool:
    """Return whether the answer surface explicitly names a semantic scale."""

    normalized_scale = str(scale or "").strip().casefold().rstrip("s")
    if normalized_scale not in {"thousand", "million", "billion", "percent"}:
        return False
    values = answer if isinstance(answer, (list, tuple)) else [answer]
    patterns = {
        "thousand": r"(?:\bthousands?\b|\b(?:usd|eur|gbp)\s*k\b|[$€£¥]\s*'?000\b)",
        "million": r"(?:\bmillions?\b|\b(?:usd|eur|gbp)\s*(?:m|mn)\b)",
        "billion": r"(?:\bbillions?\b|\b(?:usd|eur|gbp)\s*bn\b)",
        "percent": r"(?:%|\bpercent(?:age)?\b)",
    }
    return any(
        isinstance(value, str) and re.search(patterns[normalized_scale], value, re.I)
        for value in values
    )


def _violates_expression_invariant(
    candidate: dict[str, Any],
    answer_contract: dict[str, Any] | None,
) -> str:
    """Return a typed-expression violation that is decidable without re-answering."""

    expression = (answer_contract or {}).get("expression", {}) or {}
    operator = str(expression.get("op", "") or "").lower()
    if operator not in {"magnitude", "range"}:
        return ""
    answer = candidate.get("answer")
    if isinstance(answer, bool):
        return ""
    if isinstance(answer, (int, float)):
        negative = answer < 0
    elif isinstance(answer, str):
        text = answer.strip().replace(",", "")
        match = re.fullmatch(r"(-?)\s*(?:[$€£¥]\s*)?(\d+(?:\.\d+)?)\s*%?", text)
        negative = bool(match and match.group(1))
    else:
        negative = False
    if negative:
        return f"{operator} is non-negative by definition, but the candidate is negative"
    return ""


ANSWER_SHAPES = {"value", "list", "row", "decision"}
ANSWER_COVERAGE = {"one", "all"}
ANSWER_MULTIPLICITIES = {"positional", "distinct"}
FIELD_USES = {"filter", "return", "compare", "compute", "derive", "link"}
ANSWER_OPERATORS = {
    "lookup", "count", "sum", "mean", "min", "max", "argmin", "argmax",
    "difference", "ratio", "percent_change", "magnitude", "range", "compare", "topk",
}
ANSWER_TRANSFORMS = {
    "difference", "ratio", "percent_change",
    "sum", "mean", "min", "max", "range", "magnitude",
}
ANSWER_UNITS = {"source", "number", "ratio", "percent", "text"}
ANSWER_ORDERS = {"question", "source", "ascending", "descending"}
ANSWER_ARGUMENT_ROLES = {
    "value", "input", "left", "right", "numerator", "denominator",
    "earlier", "later", "group", "rank", "return",
}
ANSWER_VALUE_MODES = {"signed", "magnitude"}


def _answer_reference(
    raw: Any,
    *,
    sources: dict[str, dict[str, Any]],
    allow_kind: bool = False,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Normalize one source address in a typed answer expression."""

    if not isinstance(raw, dict):
        return None, [{"code": "INVALID_ANSWER_REFERENCE", "message": "An answer-expression reference is not an object."}]
    source_id = _resolve_source(raw.get("source"), sources)
    if source_id is None:
        return None, [{"code": "UNKNOWN_ANSWER_SOURCE", "message": "An answer-expression reference uses an unknown source."}]
    source = sources[source_id]
    if allow_kind:
        kind = str(raw.get("kind", "value") or "value").strip().lower()
        if kind == "column_entity":
            return {"source": source_id, "kind": kind}, []
    col_id = _resolve_column(raw.get("column", raw.get("col")), source)
    if col_id is None:
        return None, [{"code": "UNKNOWN_ANSWER_FIELD", "message": f"An answer-expression reference uses an unknown field in {source_id}."}]
    result: dict[str, Any] = {"source": source_id, "column": col_id}
    row_value = raw.get("row")
    if row_value is not None and row_value != "":
        n_rows = int((source.get("shape") or [0])[0] or 0)
        row_id = _valid_id(row_value, "r", n_rows)
        if row_id is None:
            return None, [{"code": "UNKNOWN_ANSWER_ROW", "message": f"An answer-expression reference uses an unknown row in {source_id}."}]
        result["row"] = row_id
    if allow_kind:
        kind = str(raw.get("kind", "value") or "value").strip().lower()
        result["kind"] = kind if kind in {"value", "entity", "column_entity"} else "value"
    else:
        role_aliases = {
            "minuend": "left",
            "subtrahend": "right",
            "dividend": "numerator",
            "divisor": "denominator",
            "before": "earlier",
            "after": "later",
            "order": "rank",
        }
        role = str(raw.get("role", "value") or "value").strip().lower()
        role = role_aliases.get(role, role)
        result["role"] = role if role in ANSWER_ARGUMENT_ROLES else "value"
        value_mode = str(raw.get("value_mode", "signed") or "signed").strip().lower()
        result["value_mode"] = (
            value_mode if value_mode in ANSWER_VALUE_MODES else "signed"
        )
        pair = str(raw.get("pair", "") or "").strip()
        if pair:
            result["pair"] = pair[:32]
    return result, []


def _answer_expression(
    raw: Any,
    *,
    sources: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Validate the small expression that fixes operation and argument roles."""

    if raw is None or raw == "":
        return None, []
    if not isinstance(raw, dict):
        return None, [{"code": "INVALID_ANSWER_EXPRESSION", "message": "The answer expression is not an object."}]
    aliases = {
        "none": "lookup",
        "identity": "lookup",
        "average": "mean",
        "avg": "mean",
        "subtract": "difference",
        "diff": "difference",
        "divide": "ratio",
        "division": "ratio",
        "percentage_change": "percent_change",
        "arg_min": "argmin",
        "arg_max": "argmax",
        "top_k": "topk",
    }
    op = str(raw.get("op", "lookup") or "lookup").strip().lower()
    op = aliases.get(op, op)
    violations: list[dict[str, str]] = []
    if op not in ANSWER_OPERATORS:
        violations.append({"code": "UNKNOWN_ANSWER_OPERATOR", "message": f"Unknown answer operator {op!r}."})
        op = "lookup"
    transform = str(raw.get("transform", "") or "").strip().lower()
    rowwise_transforms = {
        "difference", "ratio", "percent_change",
        "sum", "mean", "min", "max", "range", "magnitude",
    }
    valid_transform = (
        op == "mean" and transform == "percent_change"
    ) or (
        op in {"argmin", "argmax", "topk"}
        and transform in rowwise_transforms
    )
    if transform and (transform not in ANSWER_TRANSFORMS or not valid_transform):
        violations.append({
            "code": "INVALID_ANSWER_TRANSFORM",
            "message": (
                "Supported transforms are pairwise percent_change under mean, or "
                "a row-bound arithmetic transform under argmin, argmax, or topk."
            ),
        })
        transform = ""

    raw_args = raw.get("args", [])
    if not isinstance(raw_args, list):
        raw_args = [raw_args]
    args = []
    for item in raw_args[:12]:
        reference, problems = _answer_reference(item, sources=sources)
        violations.extend(problems)
        if reference is not None and reference not in args:
            args.append(reference)

    # Normalize temporal and aggregate aliases into each operator's canonical
    # algebra before checking its required roles.
    role_aliases_by_operator = {
        "difference": {"later": "left", "earlier": "right"},
        "sum": {"input": "value", "group": "value"},
        "mean": {} if transform else {"input": "value", "group": "value"},
        "min": {"input": "value"},
        "max": {"input": "value"},
        "magnitude": {"input": "value"},
        "range": {"input": "value"},
    }.get(op, {})
    if op in {"argmin", "argmax", "topk"} and transform == "difference":
        role_aliases_by_operator = {"left": "later", "right": "earlier"}
    elif op in {"argmin", "argmax", "topk"} and transform in {
        "sum", "mean", "min", "max", "range", "magnitude",
    }:
        role_aliases_by_operator = {
            "input": "value", "group": "value", "rank": "value",
        }
    for reference in args:
        role = str(reference.get("role", "value") or "value")
        reference["role"] = role_aliases_by_operator.get(role, role)

    numeric_operators = {
        "count", "sum", "mean", "min", "max", "difference", "ratio",
        "percent_change", "magnitude", "range",
    }
    raw_return = raw.get("return")
    raw_return_kind = (
        str(raw_return.get("kind", "value") or "value").lower()
        if isinstance(raw_return, dict)
        else "value"
    )
    return_ref = None
    # A value-return address is redundant for scalar arithmetic and is often a
    # copied row-label field. Entity returns remain explicit violations below.
    if (
        raw_return is not None
        and raw_return != ""
        and (op not in numeric_operators or raw_return_kind in {"entity", "column_entity"})
    ):
        return_ref, problems = _answer_reference(
            raw_return, sources=sources, allow_kind=True
        )
        violations.extend(problems)

    unit = str(raw.get("unit", "source") or "source").strip().lower()
    unit = unit if unit in ANSWER_UNITS else "source"
    order = str(raw.get("order", "question") or "question").strip().lower()
    order = order if order in ANSWER_ORDERS else "question"
    try:
        scale = float(raw.get("scale", 100 if unit == "percent" else 1))
    except (TypeError, ValueError):
        scale = 100.0 if unit == "percent" else 1.0
    if scale not in {1.0, 100.0}:
        violations.append({"code": "INVALID_ANSWER_SCALE", "message": "Answer-expression scale must be 1 or 100."})
        scale = 100.0 if unit == "percent" else 1.0

    def referenced_value(reference: dict[str, Any]) -> str:
        source = sources.get(str(reference.get("source", "")), {})
        row_id = str(reference.get("row", ""))
        col_id = str(reference.get("column", ""))
        row = next(
            (
                item for item in source.get("evidence", [])
                if isinstance(item, dict) and str(item.get("row")) == row_id
            ),
            {},
        )
        return str((row.get("values") or {}).get(col_id, ""))

    source_values = [referenced_value(item).strip() for item in args]
    source_percentages = bool(source_values) and all(
        value.endswith("%") for value in source_values if value
    ) and all(source_values)
    if op == "difference" and unit == "percent" and source_percentages:
        scale = 1.0
    elif op == "percent_change":
        unit, scale = "percent", 100.0
    elif op == "ratio" and unit == "percent":
        scale = 100.0

    roles = {str(item.get("role")) for item in args}
    required_roles = {
        "difference": {"left", "right"},
        "ratio": {"numerator", "denominator"},
        "percent_change": {"earlier", "later"},
        "magnitude": {"value"},
        "range": {"value"},
        "compare": {"left", "right"},
        "sum": {"value"},
        "mean": {"earlier", "later"} if transform == "percent_change" else {"value"},
        "min": {"value"},
        "max": {"value"},
        "argmin": {"rank"},
        "argmax": {"rank"},
        "topk": {"rank"},
    }.get(op, set())
    if op in {"argmin", "argmax", "topk"} and transform:
        required_roles = {
            "difference": {"earlier", "later"},
            "ratio": {"numerator", "denominator"},
            "percent_change": {"earlier", "later"},
            "sum": {"value"},
            "mean": {"value"},
            "min": {"value"},
            "max": {"value"},
            "range": {"value"},
            "magnitude": {"value"},
        }[transform]
    missing_roles = sorted(required_roles - roles)
    if missing_roles:
        violations.append({
            "code": "INCOMPLETE_ANSWER_ARGUMENTS",
            "message": f"Answer operator {op} is missing roles: {', '.join(missing_roles)}.",
        })
    if op == "mean" and transform == "percent_change":
        pairs: dict[str, set[str]] = {}
        for item in args:
            pair = str(item.get("pair", "") or "")
            if not pair:
                violations.append({
                    "code": "MISSING_TRANSFORM_PAIR",
                    "message": "Every transformed-reduction argument must name its adjacent pair.",
                })
                continue
            pairs.setdefault(pair, set()).add(str(item.get("role", "")))
        if not pairs or any(
            pair_roles != {"earlier", "later"} for pair_roles in pairs.values()
        ):
            violations.append({
                "code": "INCOMPLETE_TRANSFORM_PAIRS",
                "message": "Each pairwise percent_change requires earlier and later operands.",
            })
        unit, scale = "percent", 100.0
    if op in {"argmin", "argmax", "topk"} and return_ref is None:
        violations.append({
            "code": "MISSING_ANSWER_RETURN",
            "message": f"Answer operator {op} must name the returned entity or value field.",
        })
    if (
        return_ref is not None
        and return_ref.get("kind") in {"entity", "column_entity"}
        and op in {
            "count", "sum", "mean", "min", "max", "difference", "ratio",
            "percent_change", "magnitude", "range",
        }
    ):
        violations.append({
            "code": "INVALID_NUMERIC_RETURN",
            "message": f"Answer operator {op} returns a numeric value, not an entity label.",
        })
    rank_args = [item for item in args if item.get("role") == "rank"]
    rank_columns = {str(item.get("column")) for item in rank_args if item.get("column")}
    rank_rows = {str(item.get("row")) for item in rank_args if item.get("row")}
    if return_ref is not None and op in {"argmin", "argmax", "topk"} and not transform:
        return_kind = str(return_ref.get("kind", "value"))
        if return_kind == "column_entity" and len(rank_columns) < 2:
            violations.append({
                "code": "INVALID_RETURN_AXIS",
                "message": "column_entity requires ranking distinct candidate columns.",
            })
        if (
            return_kind == "entity"
            and len(rank_rows) == 1
            and len(rank_columns) >= 2
        ):
            violations.append({
                "code": "INVALID_RETURN_AXIS",
                "message": (
                    "Ranking multiple columns in one row returns the winning column header; "
                    "use return.kind=column_entity instead of a row entity field."
                ),
            })

    expression: dict[str, Any] = {
        "op": op,
        "args": args,
        "unit": unit,
        "scale": int(scale),
        "order": order,
    }
    if transform:
        expression["transform"] = transform
    if return_ref is not None:
        expression["return"] = return_ref
    return expression, violations


def _predicate_value(op: str, value: Any) -> Any:
    """Restore a complete JSON-like list emitted as a quoted string."""

    if op not in {"in", "between"} or not isinstance(value, str):
        return value
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    return value


def _source_map(table_sketch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(source.get("id")): source
        for source in table_sketch.get("sources", [])
        if isinstance(source, dict) and source.get("id")
    }


def _resolve_source(value: Any, sources: dict[str, dict[str, Any]]) -> str | None:
    text = str(value or "").strip()
    if text in sources:
        return text
    if len(text) > 1 and text[0].lower() == "t" and text[1:].isdigit():
        index = int(text[1:])
        ordered = list(sources)
        if 0 <= index < len(ordered):
            return ordered[index]
    folded = text.casefold()
    matches = [
        source_id for source_id, source in sources.items()
        if folded and folded in {
            source_id.casefold(),
            str(source.get("caption", "")).strip().casefold(),
        }
    ]
    return matches[0] if len(matches) == 1 else None


def _resolve_column(value: Any, source: dict[str, Any]) -> str | None:
    n_cols = int((source.get("shape") or [0, 0])[1] or 0)
    if (col_id := _valid_id(value, "c", n_cols)) is not None:
        return col_id
    folded = str(value or "").strip().casefold()
    matches = []
    for header in source.get("headers", []):
        if not isinstance(header, dict) or not header.get("id"):
            continue
        path = [str(part).strip() for part in header.get("path", []) if str(part).strip()]
        names = {str(header.get("id")).casefold(), *(part.casefold() for part in path)}
        if path:
            names.add(".".join(path).casefold())
        if folded and folded in names:
            matches.append(str(header["id"]))
    return matches[0] if len(matches) == 1 else None


def _answer_contract(
    data: dict[str, Any],
    *,
    table_sketch: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    raw = data.get("answer", data.get("contract", {}))
    raw = raw if isinstance(raw, dict) else {}
    shape = str(raw.get("shape", "value") or "value").strip().lower()
    aliases = {
        "scalar": "value",
        "set": "list",
        "tuple": "row",
        "record": "row",
        "boolean": "decision",
        "support": "decision",
    }
    shape = aliases.get(shape, shape)
    coverage = str(raw.get("coverage", "one") or "one").strip().lower()
    multiplicity = str(
        raw.get("multiplicity", "positional") or "positional"
    ).strip().lower()
    if shape not in ANSWER_SHAPES:
        shape = "value"
    if coverage not in ANSWER_COVERAGE:
        coverage = "one"
    if multiplicity not in ANSWER_MULTIPLICITIES:
        multiplicity = "positional"
    history = raw.get("history", [])
    if not isinstance(history, list):
        history = [history]
    contract = {
        "shape": shape,
        "coverage": coverage,
        "history": list(dict.fromkeys(
            str(value).strip() for value in history if str(value).strip()
        ))[:8],
    }
    if shape == "list" or "multiplicity" in raw:
        contract["multiplicity"] = multiplicity
    expression, violations = _answer_expression(
        raw.get("expression"), sources=_source_map(table_sketch)
    )
    if expression is not None:
        contract["expression"] = expression
        scalar_operators = {
            "count", "sum", "mean", "min", "max", "difference", "ratio",
            "percent_change", "magnitude", "range",
        }
        if expression.get("op") in scalar_operators and shape != "value":
            violations.append({
                "code": "INVALID_NUMERIC_SHAPE",
                "message": (
                    f"Answer operator {expression['op']} produces one numeric value, "
                    f"but the declared answer shape is {shape}."
                ),
            })
    return contract, violations


def _augment_requirements_from_expression(
    raw_requirements: list[Any],
    *,
    contract: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Close evidence requirements over every typed expression reference."""

    requirements = [dict(item) for item in raw_requirements if isinstance(item, dict)]
    expression = contract.get("expression")
    if not isinstance(expression, dict):
        return requirements
    refs = [item for item in expression.get("args", []) if isinstance(item, dict)]
    return_ref = expression.get("return")
    if isinstance(return_ref, dict):
        refs.append({**return_ref, "role": "return"})
    op = str(expression.get("op", "lookup"))

    for reference in refs:
        source_id = str(reference.get("source", ""))
        requirement = next(
            (
                item for item in requirements
                if _resolve_source(item.get("source"), sources) == source_id
            ),
            None,
        )
        if requirement is None:
            requirement = {"source": source_id, "scope": {"where_logic": "all", "where": [], "rows": []}, "fields": []}
            requirements.append(requirement)
        scope = requirement.get("scope")
        scope = dict(scope) if isinstance(scope, dict) else {}
        fields = list(requirement.get("fields", []) or [])
        role = str(reference.get("role", "value"))
        if role == "return" and reference.get("kind") == "column_entity":
            continue
        if role == "return":
            use = "return"
        elif op in {"argmin", "argmax", "topk"} and expression.get("transform"):
            use = "derive"
        elif op in {"argmin", "argmax", "topk", "min", "max"} and role in {"rank", "value"}:
            use = "compare"
        else:
            use = "compute"
        field = {"column": reference.get("column"), "use": use}
        if use == "compare":
            field["order"] = "source" if expression.get("order") == "source" else "value"
        if field not in fields:
            fields.append(field)
        if contract.get("coverage") != "all" and reference.get("row") and not scope.get("where"):
            rows = list(scope.get("rows", []) or [])
            if reference["row"] not in rows:
                rows.append(reference["row"])
            scope["rows"] = rows
        requirement["scope"] = scope
        requirement["fields"] = fields
    return requirements


def _field_resolution(
    *,
    col_id: str,
    use: str,
    source: dict[str, Any],
    predicates: list[dict[str, Any]],
    order_kind: str | None = None,
) -> dict[str, Any] | None:
    header = next(
        (
            item for item in source.get("headers", [])
            if isinstance(item, dict) and str(item.get("id")) == col_id
        ),
        {},
    )
    type_hint = str(header.get("type", "text") or "text")
    if use == "derive":
        return None
    if use == "compare":
        return {
            "column": col_id,
            "mode": "order",
            "order_kind": "source" if order_kind == "source" else "value",
        }
    if use == "compute":
        return {"column": col_id, "mode": "exact"}
    if use == "filter":
        relevant = [item for item in predicates if item.get("col") == col_id]
        if relevant and all(item.get("op") in {"eq", "in"} for item in relevant):
            anchors = []
            for item in relevant:
                value = item.get("value")
                values = value if isinstance(value, list) else [value]
                anchors.extend(str(entry).strip() for entry in values if str(entry).strip())
            return {
                "column": col_id,
                "mode": "equality",
                "anchors": list(dict.fromkeys(anchors))[:8],
            }
        if len(relevant) == 1 and relevant[0].get("op") in {"lt", "le", "gt", "ge"}:
            try:
                boundary = float(relevant[0].get("value"))
            except (TypeError, ValueError):
                return {"column": col_id, "mode": "exact"}
            return {
                "column": col_id,
                "mode": "threshold",
                "boundary": boundary,
                "operator": str(relevant[0]["op"]),
            }
        return {"column": col_id, "mode": "exact"}
    return {
        "column": col_id,
        "mode": "exact" if type_hint in {"numeric", "mixed_numeric"} else "identity",
    }


def _lower_rowwise_expression_dag(
    *,
    transform: str,
    expression_args: list[dict[str, Any]],
    direction: str,
    unit: str,
    scale: float,
) -> list[DerivedObservable]:
    """Lower one row-wise algebra into optional shared intermediates and a target."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for argument in expression_args:
        grouped.setdefault(str(argument.get("role", "value")), []).append(argument)

    observables: list[DerivedObservable] = []
    outer_operands: list[DerivedOperand] = []
    compositional_roles = {
        "difference": ("later", "earlier"),
        "ratio": ("numerator", "denominator"),
        "percent_change": ("later", "earlier"),
    }.get(transform, ())

    for role, arguments in grouped.items():
        if role in compositional_roles and len(arguments) > 1:
            helper_id = f"v{len(observables)}"
            observables.append(DerivedObservable(
                observable_id=helper_id,
                operator="sum",
                operands=[
                    DerivedOperand(
                        column=str(argument["column"]),
                        role="value",
                        value_mode=str(argument.get("value_mode", "signed")),
                    )
                    for argument in arguments
                ],
                mode="exact",
                unit="number",
                scale=1.0,
                required=False,
            ))
            outer_operands.append(DerivedOperand(
                column=helper_id,
                role=role,
                value_mode="signed",
            ))
            continue
        outer_operands.extend(
            DerivedOperand(
                column=str(argument["column"]),
                role=role,
                value_mode=str(argument.get("value_mode", "signed")),
            )
            for argument in arguments
        )

    observables.append(DerivedObservable(
        observable_id=f"v{len(observables)}",
        operator=transform,
        operands=outer_operands,
        mode="order",
        direction=direction,
        unit=unit,
        scale=scale,
        required=True,
    ))
    return observables


def lower_answer_requirements(
    data: dict[str, Any],
    *,
    table_sketch: dict[str, Any],
) -> tuple[dict[str, Any], AnswerabilitySignature, list[dict[str, str]]]:
    """Lower compact semantic requirements into an executable Signature."""

    contract, expression_violations = _answer_contract(
        data, table_sketch=table_sketch
    )
    raw_requirements = data.get("requirements")
    if not isinstance(raw_requirements, list):
        # Backward-compatible replay for frozen artifacts and test fixtures.
        signature = clean_signature(data, table_sketch=table_sketch)
        return contract, signature, expression_violations

    sources = _source_map(table_sketch)
    raw_requirements = _augment_requirements_from_expression(
        raw_requirements, contract=contract, sources=sources
    )
    slices = []
    violations: list[dict[str, str]] = list(expression_violations)
    for index, raw in enumerate(raw_requirements):
        if not isinstance(raw, dict):
            continue
        source_id = _resolve_source(raw.get("source"), sources)
        if source_id is None:
            violations.append({
                "code": "UNKNOWN_SOURCE",
                "message": f"Requirement {index} does not identify one visible source.",
            })
            continue
        source = sources[source_id]
        scope = raw.get("scope", {})
        scope = scope if isinstance(scope, dict) else {}

        predicates = []
        for predicate in scope.get("where", raw.get("where", [])) or []:
            if isinstance(predicate, (list, tuple)) and len(predicate) == 3:
                predicate = {"column": predicate[0], "op": predicate[1], "value": predicate[2]}
            if not isinstance(predicate, dict):
                continue
            col_id = _resolve_column(predicate.get("column", predicate.get("col")), source)
            op = str(predicate.get("op", "eq") or "eq").lower()
            if col_id and op in {
                "eq", "ne", "lt", "le", "gt", "ge", "contains", "starts_with",
                "ends_with", "in", "between", "under",
            }:
                predicates.append({
                    "col": col_id,
                    "op": op,
                    "value": _predicate_value(op, predicate.get("value")),
                })
        where_logic = str(scope.get("where_logic", scope.get("logic", "all")) or "all").lower()
        if where_logic not in {"all", "any"}:
            where_logic = "all"

        fields: list[tuple[str, str, str | None]] = []
        for field in raw.get("fields", []) or []:
            if isinstance(field, str):
                field = {"column": field, "use": "return"}
            if not isinstance(field, dict):
                continue
            col_id = _resolve_column(field.get("column", field.get("col")), source)
            use = str(field.get("use", "return") or "return").lower()
            order_kind = str(field.get("order", field.get("order_kind", "")) or "").lower()
            order_kind = order_kind if order_kind in {"value", "source"} else None
            entry = (col_id, use, order_kind)
            if col_id and use in FIELD_USES and entry not in fields:
                fields.append(entry)
        for predicate in predicates:
            pair = (str(predicate["col"]), "filter", None)
            if pair not in fields:
                fields.append(pair)

        rows = scope.get("rows", raw.get("rows", []))
        if not isinstance(rows, list):
            rows = [rows]
        if contract["shape"] == "row":
            columns = ["*"]
            links = [["*"]]
            resolution = []
        else:
            columns = list(dict.fromkeys(col_id for col_id, _use, _order in fields))
            links = [columns] if len(columns) >= 2 else []
            resolution = []
            seen_resolutions: set[tuple[Any, ...]] = set()
            for col_id, use, order_kind in fields:
                item = _field_resolution(
                    col_id=col_id,
                    use=use,
                    source=source,
                    predicates=predicates,
                    order_kind=order_kind,
                )
                if item is None:
                    continue
                key = (
                    item.get("column"),
                    item.get("mode"),
                    item.get("boundary"),
                    item.get("operator"),
                    item.get("order_kind"),
                    tuple(item.get("anchors", [])),
                )
                if key not in seen_resolutions:
                    seen_resolutions.add(key)
                    resolution.append(item)

        derived = []
        expression = contract.get("expression") or {}
        transform = str(expression.get("transform", "") or "")
        if expression.get("op") in {"argmin", "argmax", "topk"} and transform:
            expression_args = [
                arg for arg in expression.get("args", [])
                if isinstance(arg, dict) and arg.get("source") == source_id
            ]
            required_roles = {
                "difference": {"earlier", "later"},
                "ratio": {"numerator", "denominator"},
                "percent_change": {"earlier", "later"},
                "sum": {"value"},
                "mean": {"value"},
                "min": {"value"},
                "max": {"value"},
                "range": {"value"},
                "magnitude": {"value"},
            }.get(transform, set())
            observed_roles = {
                str(arg.get("role", "")) for arg in expression_args
            }
            if (
                expression_args
                and len(expression_args) == len(expression.get("args", []))
                and all(not arg.get("row") for arg in expression_args)
                and required_roles
                and observed_roles == required_roles
            ):
                derived.extend(_lower_rowwise_expression_dag(
                    transform=transform,
                    expression_args=expression_args,
                    direction=(
                        "ascending" if expression.get("op") == "argmin" else "descending"
                    ),
                    unit=str(expression.get("unit", "number") or "number"),
                    scale=float(expression.get("scale", 1) or 1),
                ))

        slices.append({
            "source": source_id,
            "columns": columns,
            "where": predicates,
            "where_logic": where_logic,
            "rows": rows,
            "links": links,
            "resolution": resolution,
            "derived": [item.model_dump(mode="json") for item in derived],
        })

    signature = clean_signature({"slices": slices}, table_sketch=table_sketch)
    if not signature.slices:
        violations.append({"code": "EMPTY_PLAN", "message": "No valid source requirement remains."})

    max_relevance = max(
        (float(source.get("relevance", 0.0) or 0.0) for source in sources.values()),
        default=0.0,
    )
    expression_return = (contract.get("expression") or {}).get("return") or {}
    column_entity_source = (
        str(expression_return.get("source", ""))
        if expression_return.get("kind") == "column_entity"
        else ""
    )
    for evidence_slice in signature.slices:
        source = sources.get(evidence_slice.source, {})
        relevance = float(source.get("relevance", 0.0) or 0.0)
        if max_relevance > 0 and relevance <= 0:
            violations.append({
                "code": "SOURCE_CANDIDATE_CONFLICT",
                "message": f"Selected source {evidence_slice.source} has no query-linked evidence while another source does.",
            })
        if contract["shape"] == "row" and int((source.get("shape") or [0])[0] or 0) == 0:
            violations.append({
                "code": "EMPTY_ROW_SOURCE",
                "message": f"Selected source {evidence_slice.source} has no source tuples for a row-shaped answer.",
            })
        if contract["coverage"] == "all" and evidence_slice.rows and not evidence_slice.where:
            violations.append({
                "code": "SAMPLED_COMPLETE_DOMAIN",
                "message": "An all-results contract cannot be certified by sampled row identifiers.",
            })
        if contract["shape"] != "row" and not evidence_slice.columns:
            violations.append({
                "code": "MISSING_OUTPUT_FIELD",
                "message": f"Source {evidence_slice.source} exposes no answer-bearing field.",
            })
        source_requirements = [
            raw for raw in raw_requirements
            if isinstance(raw, dict)
            and _resolve_source(raw.get("source"), sources) == evidence_slice.source
        ]
        source_uses = {
            str(field.get("use", "return") or "return").lower()
            for raw in source_requirements
            for field in (raw.get("fields", []) or [])
            if isinstance(field, dict)
        }
        if (
            contract["shape"] != "decision"
            and "compare" in source_uses
            and not source_uses.intersection({"return", "compute"})
            and evidence_slice.source != column_entity_source
        ):
            violations.append({
                "code": "MISSING_ANSWER_TARGET",
                "message": (
                    f"Source {evidence_slice.source} compares values but exposes no field "
                    "from which the requested answer can be returned."
                ),
            })
    return contract, signature, violations


class BackwardPlannerAgent(_LLMAgent):
    """Plan answerability requirements backward from the requested answer."""

    PROMPT = """Plan the minimum source evidence and answer semantics needed by a separate Answer Agent.
Return JSON only:
{"answer":{"shape":"value|list|row|decision","coverage":"one|all","multiplicity":"positional|distinct","history":[],"expression":{"op":"lookup|count|sum|mean|min|max|argmin|argmax|difference|ratio|percent_change|magnitude|range|compare|topk","transform":"optional row-wise difference|ratio|percent_change|sum|mean|min|max|range|magnitude for argmin/argmax/topk; percent_change for mean","args":[{"source":"t0","column":"c1","row":"optional r0","role":"value|input|left|right|numerator|denominator|earlier|later|group|rank","value_mode":"signed|magnitude","pair":"required pair id for mean percent_change"}],"return":{"source":"t0","column":"c0 when kind=value|entity","kind":"value|entity|column_entity"},"unit":"source|number|ratio|percent|text","scale":1,"order":"question|source|ascending|descending"}},"requirements":[{"source":"t0","scope":{"where_logic":"all|any","where":[{"column":"c0","op":"eq|ne|lt|le|gt|ge|contains|starts_with|ends_with|in|between|under","value":"source-grounded value"}],"rows":[]},"fields":[{"column":"c0","use":"filter|return|compare|compute|link","order":"value|source when use=compare"}]}]}

Planning policy:
1. Answer form. Identify the requested scalar, list, complete record, or decision. Preserve question order and repeated values for positional lists. Use distinct multiplicity only for unique identities. Use complete coverage for counts, aggregates, extrema, ties, and exhaustive lists.
2. Operation. Prefer a source lookup when the requested measure is explicitly reported. Otherwise record the required operation and every operand without computing the answer. Difference uses left minus right; ratio uses numerator divided by denominator; percent change uses (later-earlier)/earlier times 100. Distinguish absolute change, percentage-point change, and relative change from the question and source units. Do not impose an interpretation from a phrase alone. Use signed operands unless the requested operation explicitly requires magnitude. Preserve requested precision and source scale.
3. Derived quantities. For argmin, argmax, or top-k over a row-wise expression, place its operator in transform and omit operand row IDs. Arguments sharing a role are summed before the outer operation. For mean percent change, identify each earlier/later pair with a pair ID. Preserve all members of a requested aggregate, excluding overlapping subtotals. Do not substitute a single column or sampled tuple for the full expression.
4. Source bindings. Bind each operand to a visible source and full header path. Use a visible row ID for a specific tuple, or a grounded predicate for a population. Across-row ranking returns an entity field; across-column ranking returns column_entity. Preserve keys and links needed to associate values, units, and labels. A stored percentage has scale 1; a ratio converted to percent has scale 100.
5. Scope. The sketch is a sample unless evidence_complete is true. Sample absence does not establish source absence. Preserve predicate direction and conjunction or disjunction. Source prose may define a population, but numeric overlap alone does not establish membership. If prose and a displayed value appear inconsistent, retain both for review rather than silently changing the predicate.
6. Structure and context. Resolve repeated labels through their full row and column paths. Use under only for a hierarchy explicitly present in axis_index. Bind a requested total to its stored subtotal or its non-overlapping descendants. Include any source text needed to interpret units, qualifiers, or definitions. Semantic cues and relevance scores guide inspection but do not prove which rows belong in the answer.
7. Field roles. Mark predicate fields as filter, output fields as return, ordering fields as compare, arithmetic operands as compute, and association keys as link. Retain exact operands for arithmetic and both sides of comparisons. Preserve all fields for a complete-record request.

Use only visible IDs and source values. Do not answer the question or emit SQL, code, physical encodings, benchmark categories, or redundant requirements.
"""

    def plan(self, *, question: str, table_sketch: dict[str, Any]) -> AnswerabilitySignature:
        data = self._ask(self.PROMPT, {"question": question, "table": table_sketch})
        final_requirements = data.get("requirements", [])
        contract, signature, violations = lower_answer_requirements(
            data,
            table_sketch=table_sketch,
        )
        semantic_review = _needs_semantic_plan_review(contract)
        semantic_audit: dict[str, Any] | None = None
        structural_review = any(
            isinstance(source, dict)
            and (
                source.get("axis_index")
                or int(source.get("headers_omitted", 0) or 0) > 0
                or any(
                    isinstance(header, dict) and len(header.get("path", []) or []) > 1
                    for header in source.get("headers", [])
                )
            )
            for source in table_sketch.get("sources", [])
        ) or len(table_sketch.get("sources", [])) > 1
        review_violations = list(violations)
        if semantic_review and not review_violations:
            review_violations.append({
                "code": "NUMERIC_SEMANTIC_REVIEW",
                "message": (
                    "Independently verify the requested numeric operation, all operand "
                    "roles, direct reported measures, hierarchy, and source bindings."
                ),
            })
        elif structural_review and not review_violations:
            review_violations.append({
                "code": "STRUCTURAL_SEMANTIC_REVIEW",
                "message": (
                    "Independently verify answer form, operation, complete operands, "
                    "hierarchical bindings, and evidence closure before compilation."
                ),
            })
        trace: dict[str, Any] = {
            "contract": contract,
            "requirements": data.get("requirements", []),
            "violations": violations,
            "semantic_audit": semantic_audit,
            "audited": False,
        }
        if review_violations:
            repaired_data = SignatureAuditorAgent(self.llm, self.model).repair(
                question=question,
                table_sketch=table_sketch,
                proposal={"answer": contract, "requirements": data.get("requirements", [])},
                violations=review_violations,
            )
            repaired_contract, repaired_signature, repaired_violations = lower_answer_requirements(
                repaired_data,
                table_sketch=table_sketch,
            )
            trace.update({
                "audited": True,
                "audit_requirements": repaired_data.get("requirements", []),
                "audit_violations": repaired_violations,
            })
            if not repaired_violations:
                contract = repaired_contract
                signature = repaired_signature
                violations = repaired_violations
                final_requirements = repaired_data.get("requirements", [])
        context_candidates = _context_candidate_view(table_sketch, final_requirements)
        context_audit = None
        if _needs_context_link_review(contract, context_candidates):
            context_audit = SignatureAuditorAgent(self.llm, self.model).check_context(
                question=question,
                answer_contract=contract,
                requirements=final_requirements,
                candidates=context_candidates,
                planned_sources=_planned_source_view(table_sketch, final_requirements),
            )
            if context_audit["status"] == "ADD":
                revised_requirements = _add_context_requirements(
                    final_requirements, context_audit["additions"]
                )
                revised_contract, revised_signature, revised_violations = lower_answer_requirements(
                    {"answer": contract, "requirements": revised_requirements},
                    table_sketch=table_sketch,
                )
                if not revised_violations:
                    contract = revised_contract
                    signature = revised_signature
                    violations = []
                    final_requirements = revised_requirements
                    context_audit["accepted"] = True
                else:
                    context_audit["accepted"] = False
                    context_audit["violations"] = revised_violations
        trace["context_audit"] = context_audit
        trace["contract"] = contract
        trace["final_requirements"] = final_requirements
        trace["final_violations"] = violations
        self.last_trace = trace
        return signature


def _needs_semantic_plan_review(contract: dict[str, Any]) -> bool:
    """Identify plans whose answer algebra is easy to confuse but cheap to audit."""

    expression = contract.get("expression") or {}
    operator = str(expression.get("op", "lookup") or "lookup").lower()
    return operator in {
        "count", "sum", "mean", "min", "max", "difference", "ratio",
        "percent_change", "magnitude", "range", "argmin", "argmax", "compare", "topk",
    }


def _context_candidate_view(
    table_sketch: dict[str, Any],
    requirements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose compact, source-addressable context outside the planned evidence."""

    selected_sources = {
        str(item.get("source", ""))
        for item in requirements
        if isinstance(item, dict)
    }
    sources = [
        source for source in table_sketch.get("sources", [])
        if isinstance(source, dict)
    ]
    if not selected_sources or len(sources) < 2:
        return []
    candidates = []
    for source in sources:
        source_id = str(source.get("id", "") or "")
        if not source_id or source_id in selected_sources:
            continue
        if float(source.get("relevance", 0.0) or 0.0) <= 0:
            continue
        headers = {
            str(item.get("id")): item.get("path", [])
            for item in source.get("headers", [])
            if isinstance(item, dict) and item.get("id")
        }
        rows = []
        for row in source.get("evidence", [])[:12]:
            if not isinstance(row, dict) or not row.get("row"):
                continue
            values = {
                str(column): value
                for column, value in (row.get("values", {}) or {}).items()
                if str(value).strip()
            }
            if values:
                rows.append({"row": str(row["row"]), "values": values})
        if rows:
            candidates.append({
                "source": source_id,
                "caption": source.get("caption", ""),
                "headers": headers,
                "rows": rows,
            })
    return candidates


def _needs_context_link_review(
    answer_contract: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> bool:
    expression = answer_contract.get("expression") or {}
    operator = str(expression.get("op", "lookup") or "lookup").lower()
    return bool(candidates) and operator in {
        "lookup", "sum", "mean", "min", "max", "difference", "ratio",
        "percent_change", "magnitude", "range",
    }


def _planned_source_view(
    table_sketch: dict[str, Any],
    requirements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = {
        str(item.get("source", ""))
        for item in requirements
        if isinstance(item, dict)
    }
    return [
        {
            "source": str(source.get("id", "")),
            "headers": {
                str(header.get("id")): header.get("path", [])
                for header in source.get("headers", [])
                if isinstance(header, dict) and header.get("id")
            },
            "rows": [
                {"row": row.get("row"), "values": row.get("values", {})}
                for row in source.get("evidence", [])[:12]
                if isinstance(row, dict)
            ],
        }
        for source in table_sketch.get("sources", [])
        if isinstance(source, dict) and str(source.get("id", "")) in selected
    ]


class _ContextCoverageCheck(_LLMAgent):
    """Close only source-grounded unit, definition, and qualifier gaps."""

    PROMPT = """Audit whether a proposed numeric TableQA plan has enough interpretation context for a separate Answer Agent.
Return JSON only:
{"status":"PASS","why":"the selected evidence is interpretable"}
or
{"status":"ADD","why":"one precise interpretation gap","additions":[{"source":"t1","row":"r2","column":"c2"}]}

Check only whether an unselected visible source row explicitly states a unit, scale, denominator meaning, measure definition, or qualifier needed to interpret the already-bound operands. Add the smallest such source cells. A definition may explain that displayed values are already percentages or shares; it must not become a new arithmetic operand. A unit mention is admissible only when the text links it to the same table measure or explicitly names a sibling measure from the selected table in that unit; an unrelated amount is not table context. Do not change the answer shape, operator, operands, return target, source scope, or physical encoding. Use only supplied source, row, and column ids. Return PASS when no candidate explicitly closes an interpretation gap."""

    def audit(
        self,
        *,
        question: str,
        answer_contract: dict[str, Any],
        requirements: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        planned_sources: list[dict[str, Any]],
    ) -> dict[str, Any]:
        data = self._ask(self.PROMPT, {
            "question": question,
            "answer_contract": answer_contract,
            "planned_requirements": requirements,
            "planned_source_preview": planned_sources,
            "candidate_context": candidates,
        })
        valid = {
            (str(source["source"]), str(row["row"]), str(column))
            for source in candidates
            for row in source.get("rows", [])
            for column in row.get("values", {})
        }
        additions = []
        for item in data.get("additions", []) or []:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("source", "")),
                str(item.get("row", "")),
                str(item.get("column", "")),
            )
            if key in valid and key not in additions:
                additions.append(key)
        return {
            "status": "ADD" if additions else "PASS",
            "why": str(data.get("why", "") or "")[:500],
            "additions": [
                {"source": source, "row": row, "column": column}
                for source, row, column in additions[:4]
            ],
        }


def _add_context_requirements(
    requirements: list[dict[str, Any]],
    additions: list[dict[str, str]],
) -> list[dict[str, Any]]:
    revised = [dict(item) for item in requirements if isinstance(item, dict)]
    for item in additions:
        revised.append({
            "source": item["source"],
            "scope": {"where_logic": "all", "where": [], "rows": [item["row"]]},
            "fields": [{"column": item["column"], "use": "link"}],
        })
    return revised


class SignatureAuditorAgent(_LLMAgent):
    """Check and repair semantic requirements before evidence compilation."""

    def check_context(self, **kwargs):
        """Check associated source context using the Signature Auditor's model."""
        return _ContextCoverageCheck(self.llm, self.model).audit(**kwargs)

    PROMPT = """Audit a proposed answerability plan against the question, visible table sketch, and deterministic conflicts.
Return a complete repaired plan as JSON only:
{"answer":{"shape":"value|list|row|decision","coverage":"one|all","multiplicity":"positional|distinct","history":[],"expression":{"op":"lookup|count|sum|mean|min|max|argmin|argmax|difference|ratio|percent_change|magnitude|range|compare|topk","transform":"optional row-wise difference|ratio|percent_change|sum|mean|min|max|range|magnitude for argmin/argmax/topk; percent_change for mean","args":[{"source":"t0","column":"c1","row":"optional r0","role":"value|input|left|right|numerator|denominator|earlier|later|group|rank","value_mode":"signed|magnitude","pair":"required pair id for mean percent_change"}],"return":{"source":"t0","column":"c0 when kind=value|entity","kind":"value|entity|column_entity"},"unit":"source|number|ratio|percent|text","scale":1,"order":"question|source|ascending|descending"}},"requirements":[{"source":"t0","scope":{"where_logic":"all|any","where":[{"column":"c0","op":"eq|ne|lt|le|gt|ge|contains|starts_with|ends_with|in|between|under","value":"source-grounded value"}],"rows":[]},"fields":[{"column":"c0","use":"filter|return|compare|compute|link","order":"value|source when use=compare"}]}]}

For `STRUCTURAL_SEMANTIC_REVIEW`, first derive the minimal answer semantics independently from the question and visible sketch, then compare that shadow plan with the proposal; do not assume the proposal's operation, scope, or bindings are correct. For other listed conflicts, repair only the named defect. Check four dimensions: (1) answer form and complete population, (2) operation, operand roles, return target, scale, and order, (3) complete row/column hierarchy and cross-source bindings, and (4) closure of every filter, comparison, computation, grouping, return field, and external unit or qualifier needed to interpret the answer. A question asking which entity or label returns that label as `shape=value` or `shape=list`, not a full `row`; `shape=row` is only for an explicitly requested complete record. Numeric operators return a value, never an entity. For argmin, argmax, or top-k, infer the return axis from the candidates: candidates in different rows of one value column return `kind=entity` from the row-label field, while candidates in different value columns of one row return `kind=column_entity` with no column field so the Answer Agent returns the winning header label. When the ranked quantity is a row-wise arithmetic expression over sibling fields, preserve that algebra in `transform`, omit operand row ids, and retain the complete bound population so the compiler can materialize one source-aligned observable; use `value` roles for row-wise sum, mean, min, max, range, or magnitude. Use `shape=list` only for values from distinct answer-bearing cells or values that must stay separately aligned to named entities or periods. Keep a coordinated explanatory enumeration from one source sentence as one contiguous `shape=value` span. A definition, meaning, or procedure question must return explanatory text rather than a merely related numeric row; a number is the target only for an explicitly requested amount, value, rate, count, or computation. Match the complete target meaning rather than a candidate that only contains the same words with unsupported modifiers; `total assets` and `total deferred tax asset` are different measures. When `SOURCE_CANDIDATE_CONFLICT` identifies a zero-relevance return source and a positively query-linked source directly contains the requested payload, move the return target and expression argument to that linked source rather than retaining both candidate answers. A lookup is valid only when the question requests the stored measure itself. When the question asks to calculate an average, difference, ratio, or percentage change from named operands or periods, retain those operands and the requested operation even if a rounded summary column such as `%Chg` is present. `Year-on-year change` without `percent`, `percentage`, or `relative` is a later-minus-earlier difference, not percent change. A change between already-stored percentages, rates, or shares is a point difference unless relative percent change is explicit. Multiple arguments with the same role form one summed operand group before an outer difference, ratio, or percent change. If a requested total is an axis parent without a numeric subtotal, bind every numeric descendant leaf to that role; do not bind the empty parent row or one representative child. A request for components under a row-axis parent must use `op=under` on its row-label field, which deterministically selects all strict descendants from `axis_index`, excludes the parent itself, and preserves source order. Use `magnitude` only when the question asks for the amount of a signed decrease, decline, fall, drop, loss, adjustment, or downward revision and no source field directly reports that requested measure; a semantically matching reported change or rate remains a `lookup`. Use `range` for max-minus-min over a population. Never sum an ancestor subtotal with descendants. A total amount over named non-overlapping periods needs every member and a sum, unless the question explicitly asks for separate values such as `respectively`. A union over non-overlapping categories or periods without a matching subtotal needs every member and a sum. An aggregate over `both`, `all`, or an unqualified visible set of periods/categories must preserve every requested member, not one representative. For extrema or top-k over atomic members, exclude ancestor totals, subtotals, and catch-all aggregate rows unless the question explicitly asks to rank aggregates. A pointwise comparison over a population omits argument row ids and returns the matching entity field. A change between previous/revised or other paired measures compares sibling fields under the same requested time header; a sign-change test retains both exact fields over the complete population rather than filtering one sign direction in advance. Stored percentages use scale 1; scale 100 is only for a derived ratio. If another visible source states a required unit, scale, definition, or qualifier, preserve that source field as a linked requirement. The materialization certificate reports what the plan actually selected after a full source scan; it is evidence for review, not permission to infer an answer.
Treat `NUMERIC_SEMANTIC_REVIEW` like `STRUCTURAL_SEMANTIC_REVIEW`: independently verify the complete plan rather than preserving its numeric choices by default. An average of successive percentage or relative changes uses `op=mean`, `transform=percent_change`, and adjacent arguments tagged by shared `pair` ids. A directly reported change/decrease/rate in a matching source field or sentence is a lookup unless the question explicitly asks to recompute it from named observations. When a flat table has several rows with the same requested metric and no explicit parent path, prefer a visible tuple or sentence that co-binds the target entity and metric; do not infer entity ownership from row adjacency alone.
Independently check every arithmetic argument's `value_mode`: `magnitude` is appropriate when a source-negative loss, cost, expense, or decrease is used as the size of that reported quantity; `signed` is appropriate when accounting polarity or net contribution is part of the requested operation. Never simulate this distinction by flipping a computed answer.

For `UNJUSTIFIED_ATOMIC_EXCLUSION`, remove the negative predicate unless the question explicitly excludes that member or the sketch proves it is an ancestor subtotal rather than an atomic candidate. If another visible source explicitly introduces the table as the requested cohort, treat table membership as authoritative and do not reapply a rounded threshold that drops a listed boundary member.

For `COLUMN_AXIS_ENTITY_CONTRACT`, replace a scalar or numeric contract with a list-valued `lookup` whose return is `{"source":"t0","kind":"column_entity"}` and expose every listed answer-bearing axis column as a `return` field. For `PEER_MEMBER_LIST_CONTRACT`, replace the numeric contract with a list-valued source-label lookup over the complete peer-level member scope; preserve source order and exclude only source-marked aggregates or subcomponents. For `NONPEER_LIST_MEMBER`, remove only rows carrying the explicit subcomponent cue while preserving every peer member and source order. For `RECONCILIATION_ENDPOINT_GRANULARITY`, use the two source-named reconciliation endpoints as the complete broad-category set. For `SOURCE_DEFINED_COHORT_REFILTER`, remove the redundant threshold predicate and preserve the complete source-defined table membership.

Use `under` only for a parent-child relation explicitly present in `axis_index`. Treat `row_semantic_signals` only as source-grounded granularity evidence: explicit subcomponent cues are narrower than peer components, and aggregate/summary cues are not atomic members unless the requested granularity includes them. If the source is flat and fully visible, repair a semantic category/type/member scope with one exact `in` predicate over all and only the visible peer members. For questions asking what categories, types, or years a table shows, select row or column labels from the table rather than a same-topic paragraph heading. A list of column-axis labels uses `lookup`, `return.kind=column_entity`, and one `return` field per selected column.

Keep correct parts unchanged. Preserve predicate direction and AND/OR; never replace a complete population with sampled rows. A repeated leaf label must be resolved by its full parent path in `axis_index`. Keep `shape`, `coverage`, `multiplicity`, and explicit history unchanged unless a listed conflict names them. Use only visible ids and source-grounded values. You cannot answer, output SQL/code, invent evidence, choose a physical encoding, or explain the repair."""

    def repair(
        self,
        *,
        question: str,
        table_sketch: dict[str, Any],
        proposal: dict[str, Any],
        violations: list[dict[str, str]],
        materialization_certificate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "table": table_sketch,
            "proposal": proposal,
            "violations": violations,
        }
        if materialization_certificate is not None:
            payload["materialization_certificate"] = materialization_certificate
        return self._ask(self.PROMPT, payload)


class AnswerAgent(_LLMAgent):
    """Own the final answer while seeing only the Reader-facing witness."""

    PROMPT = """Answer the question using only the supplied table evidence.
Return JSON only:
{"status":"answered|blocked","verification":"verified|missing","feedback":"specific missing evidence or empty","basis":"short source-grounded derivation","refs":["visible provenance id"],"answer":null,"scale":"|thousand|million|billion|percent","missing":null}

Evidence. Read the headers, row identities, links, units, and encoding legends together. Decode reversible aliases. A rank preserves ordering and ties, not the original values. A membership code preserves the named predicate, not the identity of a nonmember. Use materialized derived quantities for the stated operation without reconstructing hidden operands. Filter constants are not source records. A complete filter with zero matches represents an empty set.

Reasoning. Follow the answer expression and retain its operand bindings. Arguments with the same arithmetic role are summed before the outer operation. Apply signed or magnitude interpretation to operands as specified, not by flipping the final result. For mean percent change, calculate each named pair before averaging. Check arithmetic, population coverage, ties, and possible subtotal overlap. Show the decisive operation in basis. Retain intermediate precision and follow the question's final precision requirement.

Output. Follow the requested scalar, list, record, or decision shape. Preserve question order and repeated values in positional lists. Deduplicate only when distinct identities are requested. Row-wise extrema return the bound entity field; column-wise extrema return the distinguishing header. For an extractive request, preserve the source span that supplies all requested information. A text-valued answer does not by itself require substring extraction. Do not strip a date, qualifier, or parenthetical unless its non-answer role is established by the source and question.

Units. Preserve displayed values for lookups. Apply conversions only when required by the expression. Keep output scale consistent with the cited source units and the requested operation. Scale metadata is separate from the expression's arithmetic multiplier. Count and entity outputs have no magnitude scale. Do not transfer a unit from an unrelated field or assume a scale from question wording alone.

References. Cite the visible support for the answer. Provenance addresses are metadata: keep them in `refs`, never insert them into `answer` unless the question requests a source field with that name. Return the answer payload without explanatory prose; put reasoning in basis. Return blocked with a specific missing requirement when the evidence cannot support the answer.

Correction. Treat a critique as a claim to check against the evidence, not as an answer to copy. Revise only after checking its cited support. Use neither hidden source content nor external knowledge. Do not generate executable code.
"""

    def answer(
        self,
        *,
        question: str,
        compressed_table: str,
        source_ref_ids: list[str],
        answer_contract: dict[str, Any] | None = None,
        answer_field_names: list[str] | None = None,
        critique: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": question,
            "compressed_table": compressed_table,
            "reference_format": "<table>:<shown source_row>:<shown column_id>",
        }
        if answer_contract:
            payload["answer_contract"] = answer_contract
        if answer_field_names:
            payload["answer_payload_contract"] = {
                "allowed_source_fields": list(dict.fromkeys(answer_field_names)),
                "provenance_only_fields": ["source_row", "table_id", "row_id", "column_id"],
                "instruction": (
                    "For a source-row object, answer keys must be source fields only; "
                    "put provenance addresses in refs."
                ),
            }
        if critique:
            payload["critique"] = critique
        data = self._ask(self.PROMPT, payload)
        status = str(data.get("status", "") or "").lower()
        verification = str(data.get("verification", "") or "").lower()
        if status in {"blocked", "abstain", "need_evidence"} and verification == "verified":
            verification = "missing"
        if verification not in {"verified", "missing"}:
            verification = "missing"
        if status != "answered":
            verification = "missing"
        answer = data.get("answer") if verification == "verified" else None
        contracted_shape = str((answer_contract or {}).get("shape", "") or "").lower()
        if contracted_shape in {"list", "row"}:
            answer = normalize_structured_answer(answer)
        if answer is None:
            # A model cannot certify an absent payload. Preserve its concrete
            # missing target so the source-recovery path can close the gap.
            verification = "missing"
        valid_refs = set(source_ref_ids)
        refs = list(dict.fromkeys(
            str(value) for value in data.get("refs", []) or [] if str(value) in valid_refs
        ))[:8]
        if verification != "verified":
            answer = None
        if isinstance(answer, bool) and contracted_shape == "decision":
            answer = "yes" if answer else "no"
        scale = str(data.get("scale", "") or "").strip().lower()
        scale_aliases = {"%": "percent", "percentage": "percent"}
        scale = scale_aliases.get(scale, scale)
        if scale not in {"", "thousand", "million", "billion", "percent"}:
            scale = ""
        # Scale is semantic answer metadata. It remains meaningful when the
        # source-verbatim answer also lexicalizes the unit; dataset serializers
        # decide how that pair is transported to their official evaluator.
        return {
            "status": "ANSWERED" if verification == "verified" and status not in {"blocked", "abstain", "need_evidence"} else "BLOCKED",
            "verification": verification.upper(),
            "feedback": str(data.get("feedback", "") or "")[:1000],
            "answer": answer,
            "scale": scale,
            "basis": str(data.get("basis", "") or "")[:1500],
            "used_ref_ids": refs,
            "missing": data.get("missing"),
        }


def refinement_obligations(
    answer_contract: dict[str, Any] | None,
) -> dict[str, list[str]]:
    """Compile a typed answer contract into the checks relevant to refinement."""

    contract = answer_contract or {}
    expression = contract.get("expression") if isinstance(contract.get("expression"), dict) else {}
    operator = str(expression.get("op", "lookup") or "lookup").lower()
    shape = str(contract.get("shape", "value") or "value").lower()
    return_kind = str((expression.get("return") or {}).get("kind", "value") or "value").lower()

    checks = ["evidence_closure", "source_binding"]
    modes = ["source_binding", "basis_output_mismatch"]
    if operator in {
        "count", "sum", "mean", "min", "max", "difference", "ratio",
        "percent_change", "magnitude", "range",
    }:
        checks.extend(("operation_and_operands", "numeric_scale", "final_rounding"))
        modes.extend((
            "arithmetic_recompute", "percent_change_recompute", "scale_binding",
            "scale_removal", "numeric_filter_coverage",
        ))
    elif operator in {"argmin", "argmax", "topk", "compare"}:
        checks.extend(("candidate_domain", "comparison_direction", "ties", "return_axis"))
        modes.append("numeric_filter_coverage")
    elif return_kind == "value":
        checks.append("selected_value")

    if shape == "list":
        checks.extend(("member_binding", "answer_cardinality", "serialization_order"))
        modes.append("list_order")
        if str(contract.get("multiplicity", "positional") or "positional").lower() == "distinct":
            checks.append("distinct_member_identity")
            modes.append("list_deduplication")
    elif shape == "row":
        checks.extend(("complete_source_row", "source_field_schema"))
    elif (operator == "lookup" and return_kind == "value"
          and str(expression.get("unit", "") or "").lower() == "text"):
        checks.extend(("extractive_sufficiency", "minimal_span_boundary"))
        modes.append("extractive_span")

    return {
        "checks": list(dict.fromkeys(checks)),
        "certificate_modes": list(dict.fromkeys(modes)),
    }


class EvidenceAuditorAgent(_LLMAgent):
    """Find a falsifiable error in the answer or its evidence boundary."""

    PROMPT = """Check a candidate answer against its question, evidence, and answer requirements.
Return JSON only in one form:
{"status":"pass","why":"all applicable checks passed"}
{"status":"counterexample","kind":"arithmetic|comparison|selection|binding|scale|output","why":"specific contradiction","refs":["visible provenance id"],"certificate":{"mode":"source_binding|basis_output_mismatch|arithmetic_recompute|percent_change_recompute|scale_binding|scale_removal|numeric_filter_coverage|list_order|list_deduplication|extractive_span","claim":"source-grounded check"}}
{"status":"missing","why":"specific evidence gap","missing":"required source information","targets":[{"source":"t0","row":"r2","columns":["c0","c1"]}]}
{"status":"inconclusive","why":"the available evidence does not resolve this check"}

Use audit_obligations to select the relevant checks:
1. Coverage. Verify the requested population, complete filters, ties, and evidence for negative claims. An omitted-candidate preview is a lead for source inspection, not evidence against an answer. If information is missing, identify what must be inspected and copy only visible target addresses.
2. Bindings. Check full header paths, entity-value associations, return axis, and the source fields used by each operation. Do not substitute a different population or operation during answer correction.
3. Reasoning and interpretation. Recompute arithmetic from the cited evidence or check the materialized quantity's stated meaning. Respect signed versus magnitude operands, units, and requested precision. Distinguish a numerical contradiction from an alternative rounding convention. Rank and membership codes cannot support claims about information they do not preserve.
4. Answer form. Check shape, question order, multiplicity, units, and record fields. Preserve repeated values in positional lists. For an extractive request, verify that the cited span retains all required information. Do not request stylistic changes or apply extractive rules merely because an answer is text.

A counterexample must identify visible support and an applicable certificate mode. A sign error is a counterexample when the stated operation and operands establish the sign. An unresolved semantic ambiguity is inconclusive, not a successful verification. If interpretation context is missing, request it. Do not supply a replacement answer or accept an ungrounded alternative. Return pass only after completing the applicable checks without a contradiction or unresolved evidence gap.
"""

    def audit(
        self,
        *,
        question: str,
        signature: AnswerabilitySignature,
        witness: str,
        candidate_answer: dict[str, Any],
        visible_ref_ids: list[str],
        answer_contract: dict[str, Any] | None = None,
        omitted_candidates: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        obligations = refinement_obligations(answer_contract)
        payload = {
            "question": question,
            "answerability_signature": signature.model_dump(),
            "witness": witness,
            "candidate": candidate_answer,
            "audit_obligations": obligations,
            "previous_checks": history or [],
        }
        if answer_contract:
            payload["answer_contract"] = answer_contract
        if omitted_candidates and omitted_candidates.get("sources"):
            payload["omitted_candidate_preview"] = omitted_candidates
        data = self._ask(self.PROMPT, payload)
        status = str(data.get("status", "") or "").lower()
        inconclusive = {
            "status": "INCONCLUSIVE",
            "why": str(data.get("why", "No valid audit decision was supplied."))[:1000],
            "refs": [],
        }
        valid_refs = set(visible_ref_ids)
        refs = list(dict.fromkeys(
            str(value) for value in data.get("refs", []) or [] if str(value) in valid_refs
        ))
        if status == "missing":
            valid_targets = {
                (
                    str(source.get("source", "")),
                    str(row.get("row", "")),
                    str(column),
                )
                for source in (omitted_candidates or {}).get("sources", [])
                if isinstance(source, dict)
                for row in source.get("rows", [])
                if isinstance(row, dict)
                for column in (row.get("values", {}) or {})
            }
            targets = []
            for raw in data.get("targets", []) or []:
                if not isinstance(raw, dict):
                    continue
                source_id = str(raw.get("source", "") or "")
                row_id = str(raw.get("row", "") or "")
                columns = [
                    str(column) for column in raw.get("columns", []) or []
                    if (source_id, row_id, str(column)) in valid_targets
                ]
                if columns:
                    targets.append({"source": source_id, "row": row_id, "columns": columns})
            return {
                "status": "MISSING",
                "why": str(data.get("why", ""))[:1000],
                "missing": str(data.get("missing", "") or data.get("why", ""))[:1000],
                "refs": [],
                "targets": targets[:4],
            }
        if status in {"counterexample", "revise"} and refs:
            certificate = data.get("certificate") if isinstance(data.get("certificate"), dict) else {}
            mode = str(certificate.get("mode", "") or "").lower()
            if mode not in obligations["certificate_modes"]:
                return inconclusive
            return {
                "status": "COUNTEREXAMPLE",
                "kind": str(data.get("kind", data.get("error", "reasoning")))[:100],
                "why": str(data.get("why", ""))[:1000],
                "refs": refs[:8],
                "certificate": {
                    "mode": (mode or "unsupported")[:100],
                    "claim": str(certificate.get("claim", ""))[:500],
                },
            }
        if status == "pass":
            return {"status": "PASS", "why": str(data.get("why", ""))[:1000], "refs": []}
        return inconclusive


    def plan_inspection(self, **kwargs):
        """Select a bounded source inspection for a missing requirement."""
        return _SourceInspector(self.llm, self.model).plan(**kwargs)

    def inspect_findings(self, **kwargs):
        """Ground a requirement extension in returned source observations."""
        return _SourceInspector(self.llm, self.model).judge(**kwargs)


class _SourceInspector(_LLMAgent):
    """Inspect omitted source data only after a concrete missing-evidence audit."""

    PROBE_PROMPT = """Plan a bounded omitted-table inspection for a concrete missing-evidence report.
Return JSON only:
{"status":"pass","why":"why no omitted inspection is needed"}
or
{"status":"probe","why":"answer-changing uncertainty","actions":[{"tool":"search","source":"t0","query":"text","columns":["c0"],"limit":8},{"tool":"slice","source":"t0","rows":["r2"],"columns":["c0","c1"]},{"tool":"filter","source":"t0","where":[{"col":"c0","op":"eq","value":"name"}],"columns":["c0","c1"],"limit":20},{"tool":"profile","source":"t0","columns":["c1"]}]}

Use at most three generic data-access actions. When `missing_report.targets` contains validated omitted-candidate addresses, inspect those exact rows and columns first. Use slice with empty columns to inspect every field of a requested complete row. Use filter when a complete population or zero-match result is required; it scans every source row and supports eq, ne, lt, le, gt, ge, contains, starts_with, ends_with, in, and between. For a requested label reached through an id, inspect the key and label columns in the label-bearing source.

First distinguish a missing operand from missing interpretation context. If the visible witness already contains the values required by its frozen expression, but the Reader asks for a denominator, unit, scale, definition, or qualifier that may describe those values rather than supply a new operand, search descriptive sources such as paragraphs, notes, or captions for the question's measure phrase before probing more numeric rows. A definition can close interpretation without introducing another arithmetic argument. Do not infer context from source names alone; inspect and cite the source text. You cannot answer the benchmark question or revise visible reasoning."""

    JUDGE_PROMPT = """Judge whether omitted-table observations close the reported answer-contract gap or contain a concrete counterexample to the candidate support.
Return JSON only:
{"status":"pass","why":"no answer-changing omitted evidence"}
or
{"status":"counterexample","why":"specific reason","refs":["visible omitted source id"],"slice":{"source":"t0","columns":["c0","c1"],"rows":["r2"],"where":[],"links":[["c0","c1"]],"resolution":[{"column":"c0","mode":"identity"},{"column":"c1","mode":"exact"}]}}

Every non-empty tuple counterexample must cite shown observation ids. A source tuple that explicitly defines the unit, denominator, measure meaning, or qualifier of already-visible operands closes a reported interpretation gap; return it as a counterexample to the missing claim without inventing an additional arithmetic operand. A complete zero-match filter may instead cite the executed filter by reproducing its source and where predicates exactly in the residual slice. Preserve all fields for a complete-row request and both join key and target label for cross-table recovery. Do not emit a final answer. Request only the minimal residual slice."""

    def plan(
        self,
        *,
        question: str,
        candidate_answer: dict[str, Any],
        missing_report: dict[str, Any],
        signature: AnswerabilitySignature,
        visible_witness: str,
        table_lens: dict[str, Any],
        answer_contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "candidate": candidate_answer,
            "missing_report": missing_report,
            "signature": signature.model_dump(),
            "visible_witness": visible_witness,
            "omitted_table_interface": table_lens,
        }
        if answer_contract:
            payload["answer_contract"] = answer_contract
        data = self._ask(self.PROBE_PROMPT, payload)
        if str(data.get("status", "probe") or "probe").lower() == "pass":
            return {"status": "PASS", "why": str(data.get("why", ""))[:500], "actions": []}
        sources = {
            str(item.get("id")): item for item in table_lens.get("sources", [])
            if isinstance(item, dict) and item.get("id")
        }
        actions = []
        for raw in data.get("actions", []) or []:
            if not isinstance(raw, dict) or raw.get("tool") not in {"search", "slice", "filter", "profile"}:
                continue
            source_id = str(raw.get("source", "") or "")
            if source_id not in sources:
                source_id = next(iter(sources), "t0")
            source = sources.get(source_id, {})
            n_rows, n_cols = [int(value or 0) for value in (source.get("shape") or [0, 0])[:2]]
            action: dict[str, Any] = {"tool": raw["tool"], "source": source_id}
            action["columns"] = [
                col_id for value in raw.get("columns", []) or []
                if (col_id := _valid_id(value, "c", n_cols))
            ][:12]
            if raw["tool"] == "search":
                query = str(raw.get("query", "") or "").strip()[:300]
                if not query:
                    continue
                action["query"] = query
                try:
                    action["limit"] = max(1, min(20, int(raw.get("limit", 8) or 8)))
                except (TypeError, ValueError):
                    action["limit"] = 8
            elif raw["tool"] == "slice":
                action["rows"] = [
                    row_id for value in raw.get("rows", []) or []
                    if (row_id := _valid_id(value, "r", n_rows))
                ][:20]
                if not action["rows"]:
                    continue
            elif raw["tool"] == "filter":
                where = []
                allowed_predicates = {
                    "eq", "ne", "lt", "le", "gt", "ge", "contains",
                    "starts_with", "ends_with", "in", "between",
                }
                for item in raw.get("where", []) or []:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        item = {"col": item[0], "op": item[1], "value": item[2]}
                    if not isinstance(item, dict):
                        continue
                    col_id = _valid_id(item.get("col"), "c", n_cols)
                    operator = str(item.get("op", "eq") or "eq").lower()
                    if col_id and operator in allowed_predicates:
                        where.append({"col": col_id, "op": operator, "value": item.get("value")})
                if not where:
                    continue
                action["where"] = where
                try:
                    action["limit"] = max(1, min(20, int(raw.get("limit", 20) or 20)))
                except (TypeError, ValueError):
                    action["limit"] = 20
            elif not action["columns"]:
                continue
            actions.append(action)
            if len(actions) >= 3:
                break
        return {
            "status": "PROBE" if actions else "PASS",
            "why": str(data.get("why", ""))[:500],
            "actions": actions,
        }

    def judge(
        self,
        *,
        question: str,
        candidate_answer: dict[str, Any],
        observations: dict[str, Any],
        table_sketch: dict[str, Any],
        answer_contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "candidate": candidate_answer,
            "omitted_table_observations": observations.get("observations", []),
        }
        if answer_contract:
            payload["answer_contract"] = answer_contract
        data = self._ask(self.JUDGE_PROMPT, payload)
        valid_refs = set(observations.get("valid_ref_ids", []))
        refs = list(dict.fromkeys(
            str(value) for value in data.get("refs", []) or [] if str(value) in valid_refs
        ))
        raw_slice = data.get("slice", {}) if isinstance(data.get("slice"), dict) else {}
        complete_filter = self._matching_complete_filter(raw_slice, observations)
        if (
            str(data.get("status", "pass") or "pass").lower() != "counterexample"
            or (not refs and complete_filter is None)
        ):
            return {
                "status": "PASS",
                "why": str(data.get("why", ""))[:500],
                "refs": [],
                "signature": AnswerabilitySignature(),
                "complete_filter_verified": False,
            }

        # Predicates are accepted only when the deterministic lens executed the
        # exact complete source filter. Tuple residuals remain ref-grounded.
        if raw_slice.get("where") and complete_filter is None:
            if not refs:
                return {
                    "status": "PASS",
                    "why": "The proposed residual predicate was not verified by a complete source filter.",
                    "refs": [],
                    "signature": AnswerabilitySignature(),
                    "complete_filter_verified": False,
                }
            # A cited observed tuple is independently sufficient provenance.
            # Do not let an unnecessary, unexecuted predicate invalidate it.
            raw_slice = {**raw_slice, "where": []}
        residual = clean_signature(
            {"slices": [raw_slice]},
            table_sketch=table_sketch,
            allowed_ref_ids=valid_refs if refs else None,
        )
        by_source: dict[str, list[str]] = {}
        for ref_id in refs:
            by_source.setdefault(ref_id.split(":", 1)[0], []).append(ref_id)
        for evidence_slice in residual.slices:
            exact_cols = {item.column for item in evidence_slice.resolution if item.mode == "exact"}
            for ref_id in by_source.get(evidence_slice.source, []):
                parts = ref_id.split(":")
                if len(parts) != 3:
                    continue
                _, row_id, col_id = parts
                if not evidence_slice.where and row_id not in evidence_slice.rows:
                    evidence_slice.rows.append(row_id)
                if col_id not in evidence_slice.columns:
                    evidence_slice.columns.append(col_id)
                if col_id not in exact_cols:
                    evidence_slice.resolution.append(
                        EvidenceResolution(column=col_id, mode="exact")
                    )
                    exact_cols.add(col_id)
        return {
            "status": "COUNTEREXAMPLE",
            "why": str(data.get("why", ""))[:500],
            "refs": refs,
            "signature": residual,
            "complete_filter_verified": complete_filter is not None,
        }

    @staticmethod
    def _matching_complete_filter(
        raw_slice: dict[str, Any],
        observations: dict[str, Any],
    ) -> dict[str, Any] | None:
        source = str(raw_slice.get("source", "") or "")
        where = raw_slice.get("where", [])
        if not source or not isinstance(where, list) or not where:
            return None
        normalized = []
        for item in where:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                item = {"col": item[0], "op": item[1], "value": item[2]}
            if not isinstance(item, dict):
                return None
            normalized.append({
                "col": str(item.get("col", "")),
                "op": str(item.get("op", "eq") or "eq").lower(),
                "value": item.get("value"),
            })
        target = _json(normalized)
        for observation in observations.get("observations", []):
            if not isinstance(observation, dict):
                continue
            if (
                observation.get("tool") == "filter"
                and observation.get("complete") is True
                and str(observation.get("source", "")) == source
                and _json(observation.get("where", [])) == target
            ):
                return observation
        return None
