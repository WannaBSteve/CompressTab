"""Grounding and answer-validation utilities for the three-stage pipeline."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from compresstab.agents import (
    AnswerAgent,
    SignatureAuditorAgent,
    answer_lexicalizes_scale,
    lower_answer_requirements,
)
from compresstab.compress import CompressedTable, CompressTab
from compresstab.evidence import (
    AnswerabilitySignature,
    EvidenceCompilation,
)
from compresstab.llm.client import JsonLLMClient
from compresstab.table.store import norm_text, parse_number, tokens


def answer_field_names(compressed: CompressedTable) -> list[str]:
    """Return visible source-header names, excluding witness address columns."""

    values = []
    for ref in compressed.evidence_refs:
        path = [str(part).strip() for part in ref.header_path if str(part).strip()]
        if path:
            values.extend((path[-1], ".".join(path)))
    return list(dict.fromkeys(value for value in values if value))


def unexpected_source_row_fields(answer: Any, allowed_fields: list[str]) -> list[str]:
    """Detect metadata mixed into an otherwise source-shaped row object."""

    if not isinstance(answer, dict) or not allowed_fields:
        return []
    allowed = set(allowed_fields)
    keys = {str(key) for key in answer}
    # Constructed JSON objects may legitimately use non-source keys. Enforce
    # this contract only when the object already presents itself as a source row.
    if not keys.intersection(allowed):
        return []
    return sorted(keys - allowed)


def _answer_scale_terms(answer: Any) -> set[str]:
    values = answer if isinstance(answer, (list, tuple)) else [answer]
    found = set()
    for value in values:
        text = str(value or "")
        if "%" in text or re.search(r"\bpercent(?:age)?s?\b", text, re.I):
            found.add("percent")
        if re.search(r"\bthousands?\b", text, re.I):
            found.add("thousand")
        if re.search(r"\bmillions?\b", text, re.I):
            found.add("million")
        if re.search(r"\bbillions?\b|\bbn\b", text, re.I):
            found.add("billion")
    return found


def _is_numeric_answer_atom(answer: Any) -> bool:
    """Return whether every answer member is a standalone numeric surface."""

    values = answer if isinstance(answer, (list, tuple)) else [answer]
    if not values:
        return False
    atom = re.compile(
        r"^\s*(?:[$€£¥]|[A-Z]{2,3}\$)?\s*"
        r"(?:\(\s*)?[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*\))?"
        r"\s*(?:%|percent(?:age)?s?|thousands?|millions?|billions?|bn)?\s*$",
        re.I,
    )
    return all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        or isinstance(value, str) and bool(atom.fullmatch(value))
        for value in values
    )


def answer_bound_scales(
    candidate: dict[str, Any],
    answer_contract: dict[str, Any] | None,
    compressed: CompressedTable,
) -> set[str]:
    """Resolve scales only from source cells cited for the candidate answer."""

    contract = answer_contract or {}
    expression = contract.get("expression") if isinstance(contract.get("expression"), dict) else {}
    return_kind = str((expression.get("return") or {}).get("kind", "value") or "value").lower()
    operator = str(expression.get("op", "lookup") or "lookup").lower()
    if operator == "count" or return_kind in {"entity", "column_entity"}:
        return set()

    ref_by_id = {
        ref.ref_id: ref for ref in getattr(compressed, "evidence_refs", [])
    }
    ref_ids = {
        str(value) for value in candidate.get("used_ref_ids", []) or [] if str(value)
    }
    if not ref_ids:
        for arg in expression.get("args", []) or []:
            if not isinstance(arg, dict):
                continue
            source, row, column = (str(arg.get(key, "") or "") for key in ("source", "row", "column"))
            if source and row and column:
                ref_ids.add(f"{source}:{row}:{column}")
    return {
        str(ref_by_id[ref_id].scale or "").strip().lower()
        for ref_id in ref_ids
        if ref_id in ref_by_id and ref_by_id[ref_id].scale
    }


def normalize_typed_answer_metadata(
    candidate: dict[str, Any],
    answer_contract: dict[str, Any] | None,
    compressed: CompressedTable,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Correct mechanically decidable scale metadata without changing answers."""

    if candidate.get("verification") != "VERIFIED":
        return candidate, None
    contract = answer_contract or {}
    expression = contract.get("expression") if isinstance(contract.get("expression"), dict) else {}
    operator = str(expression.get("op", "lookup") or "lookup").lower()
    return_kind = str((expression.get("return") or {}).get("kind", "value") or "value").lower()
    answer = candidate.get("answer")
    current = str(candidate.get("scale", "") or "").strip().lower()

    expected: str | None = None
    reason = ""
    if operator == "count" or return_kind in {"entity", "column_entity"}:
        expected = ""
        reason = "Entity and count outputs do not carry the ranked measure's scale."
    else:
        lexical = _answer_scale_terms(answer)
        bound = answer_bound_scales(candidate, answer_contract, compressed)
        numeric = _is_numeric_answer_atom(answer)
        if numeric and len(lexical) == 1:
            expected = next(iter(lexical))
            reason = "The source-verbatim answer explicitly lexicalizes one semantic scale."
        elif numeric and len(bound) == 1 and operator in {
            "lookup", "sum", "mean", "min", "max", "difference", "magnitude", "range",
        }:
            expected = next(iter(bound))
            reason = "All cited answer-bearing source cells share one explicit scale."
        elif operator == "percent_change" or (
            str(expression.get("unit", "") or "").lower() == "percent"
            and return_kind not in {"entity", "column_entity"}
        ):
            expected = "percent"
            reason = "The contracted output is a percentage-valued quantity."

    if expected is None or expected == current:
        return candidate, None
    revised = dict(candidate)
    revised["scale"] = expected
    return revised, {
        "status": "typed_metadata_correction",
        "field": "scale",
        "from": current,
        "to": expected,
        "why": reason,
        "answer_unchanged": True,
    }


def answer_surface_contract_violation(
    *,
    question: str,
    candidate: dict[str, Any],
    answer_contract: dict[str, Any] | None,
    compressed: CompressedTable,
) -> dict[str, Any] | None:
    """Find a decidable output-form violation without deriving an answer."""

    if candidate.get("verification") != "VERIFIED":
        return None
    contract = answer_contract or {}
    expression = contract.get("expression") if isinstance(contract.get("expression"), dict) else {}
    shape = str(contract.get("shape", "value") or "value").lower()
    answer = candidate.get("answer")

    expected_types = {
        "value": "one scalar string or number",
        "list": "a JSON list",
        "row": "one JSON object containing source fields",
        "decision": "the requested decision string",
    }
    shape_invalid = bool(contract) and (
        answer is None
        or (shape == "list" and not isinstance(answer, list))
        or (shape in {"value", "decision"} and isinstance(answer, (list, dict)))
        or (shape == "row" and not isinstance(answer, dict))
    )
    if shape_invalid:
        return {
            "code": "answer_shape",
            "why": f"The answer contract requires {expected_types.get(shape, shape)}.",
            "instruction": "Regenerate only the requested payload in the contracted shape.",
        }


    if (
        shape == "list"
        and str(contract.get("multiplicity", "positional") or "positional").lower() == "distinct"
        and isinstance(answer, list)
    ):
        if any(type(item) is type(previous) and item == previous for index, item in enumerate(answer)
               for previous in answer[:index]):
            return {
                "code": "duplicate_distinct_member",
                "why": "A distinct-member list repeats a textually identical source member.",
                "instruction": (
                    "Re-read the same evidence and return each identical semantic member once, "
                    "preserving first source order and leaving similar but non-identical members separate."
                ),
            }


    operator = str(expression.get("op", "lookup") or "lookup").lower()
    scale_preserving = {
        "lookup", "sum", "mean", "min", "max", "difference", "magnitude", "range",
    }
    numeric_values = _numeric_answer_values(answer)
    visible_scales = answer_bound_scales(candidate, answer_contract, compressed)
    if (
        operator in scale_preserving
        and numeric_values is not None
        and len(visible_scales) == 1
        and not answer_lexicalizes_scale(answer, next(iter(visible_scales)))
        and str(candidate.get("scale", "") or "") != next(iter(visible_scales))
    ):
        source_scale = next(iter(visible_scales))
        return {
            "code": "source_scale_binding",
            "why": f"The visible answer-bearing source has one explicit common scale: {source_scale}.",
            "instruction": (
                "Keep the displayed numeric value unchanged and bind that explicit source scale "
                "in the separate scale field."
            ),
        }
    return None


def read_with_answer_contract(
    reader: AnswerAgent,
    *,
    question: str,
    compressed: CompressedTable,
    answer_contract: dict[str, Any] | None = None,
    critique: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Let the Reader correct one decidable output-contract violation."""

    fields = answer_field_names(compressed)
    answer = reader.answer(
        question=question,
        compressed_table=compressed.text,
        source_ref_ids=compressed.evidence_ref_ids,
        answer_contract=answer_contract,
        critique=critique,
    )
    answer, metadata_correction = normalize_typed_answer_metadata(
        answer, answer_contract, compressed
    )
    typed_correction = {
        key: value
        for key, value in {
            "typed_metadata_correction": metadata_correction,
        }.items()
        if value is not None
    } or None
    unexpected = unexpected_source_row_fields(answer.get("answer"), fields)
    violation = None
    if unexpected:
        violation = {
            "code": "source_row_schema",
            "why": "The source-row answer contains fields absent from the source schema.",
            "unexpected_fields": unexpected,
            "allowed_source_fields": fields,
            "instruction": (
                "Regenerate the requested answer from the same witness. Keep only real source "
                "fields in a row object and place provenance addresses in refs."
            ),
        }
    else:
        violation = answer_surface_contract_violation(
            question=question,
            candidate=answer,
            answer_contract=answer_contract,
            compressed=compressed,
        )
    if not violation:
        return answer, typed_correction
    correction = {
        "status": "output_contract_violation",
        **violation,
    }
    revised = reader.answer(
        question=question,
        compressed_table=compressed.text,
        source_ref_ids=compressed.evidence_ref_ids,
        answer_contract=answer_contract,
        answer_field_names=fields,
        critique=correction,
    )
    revised, revised_metadata_correction = normalize_typed_answer_metadata(
        revised, answer_contract, compressed
    )
    revised_unexpected = unexpected_source_row_fields(revised.get("answer"), fields)
    revised_violation = answer_surface_contract_violation(
        question=question,
        candidate=revised,
        answer_contract=answer_contract,
        compressed=compressed,
    )
    if not revised_unexpected and revised_violation is None:
        return revised, {
            "violation": correction,
            "result": revised,
            "typed_metadata_correction": revised_metadata_correction,
        }
    return answer, {
        "violation": correction,
        "result": revised,
        "remaining_violation": revised_violation,
        "status": "unresolved",
    }


def _scope_preview(row_ids: list[str], limit: int = 8) -> list[str]:
    if len(row_ids) <= limit:
        return list(row_ids)
    half = max(1, limit // 2)
    return [*row_ids[:half], "...", *row_ids[-half:]]


def _normalized_answer_contract(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    shape = str(value.get("shape", "value") or "value").lower()
    coverage = str(value.get("coverage", "one") or "one").lower()
    history = value.get("history", [])
    result = {
        "shape": shape if shape in {"value", "list", "row", "decision"} else "value",
        "coverage": coverage if coverage in {"one", "all"} else "one",
        "history": list(dict.fromkeys(
            str(item).strip() for item in (history if isinstance(history, list) else [history])
            if str(item).strip()
        ))[:8],
    }
    if result["shape"] == "list" or "multiplicity" in value:
        multiplicity = str(value.get("multiplicity", "positional") or "positional").lower()
        result["multiplicity"] = (
            multiplicity if multiplicity in {"positional", "distinct"} else "positional"
        )
    expression = value.get("expression")
    if isinstance(expression, dict) and expression.get("op"):
        result["expression"] = expression
    return result


def _expression_erasure_violations(
    incumbent: dict[str, Any] | None,
    revision: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Reject reviewer revisions that silently delete answer operands."""

    old_expression = (incumbent or {}).get("expression") or {}
    new_expression = (revision or {}).get("expression") or {}
    old_args = [item for item in old_expression.get("args", []) if isinstance(item, dict)]
    new_args = [item for item in new_expression.get("args", []) if isinstance(item, dict)]
    if not old_args:
        return []

    def role_counts(items: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            role = str(item.get("role", "value") or "value").lower()
            counts[role] = counts.get(role, 0) + 1
        return counts

    old_counts = role_counts(old_args)
    new_counts = role_counts(new_args)
    erased = {
        role: count - new_counts.get(role, 0)
        for role, count in old_counts.items()
        if new_counts.get(role, 0) < count
    }
    if not erased:
        return []
    detail = ", ".join(f"{role}:{count}" for role, count in sorted(erased.items()))
    return [{
        "code": "ANSWER_ARGUMENT_ERASURE",
        "message": (
            "The materialization revision deleted previously required answer "
            f"operands ({detail}) instead of making their evidence scope executable."
        ),
    }]


def _numeric_answer_values(value: Any) -> list[float] | None:
    values = value if isinstance(value, (list, tuple)) else [value]
    parsed = [parse_number(item) for item in values]
    if not parsed or any(item is None for item in parsed):
        return None
    return [float(item) for item in parsed if item is not None]


def omitted_candidate_preview(
    table_sketch: dict[str, Any],
    compressed: CompressedTable,
    *,
    max_rows: int = 6,
) -> dict[str, Any]:
    """Expose a bounded preview of high-relevance cells omitted by a plan.

    The preview is not Reader evidence.  It only lets the Evidence Auditor
    request a provenance-bearing source probe when Stage A appears to have
    dropped a relevant candidate.
    """

    visible_cells = {
        (ref.table_id, row_id, col_id)
        for ref in compressed.evidence_refs
        for row_id in ref.row_ids
        for col_id in ref.col_ids
    }
    sources = []
    remaining = max(0, int(max_rows))
    for source in table_sketch.get("sources", []) or []:
        if remaining <= 0 or not isinstance(source, dict):
            break
        source_id = str(source.get("id", "") or "")
        rows = []
        for candidate in source.get("evidence", []) or []:
            if remaining <= 0 or not isinstance(candidate, dict):
                break
            row_id = str(candidate.get("row", "") or "")
            omitted = {
                str(col_id): str(value)[:240]
                for col_id, value in (candidate.get("values", {}) or {}).items()
                if (source_id, row_id, str(col_id)) not in visible_cells
                and str(value).strip()
            }
            if not omitted:
                continue
            rows.append({"row": row_id, "values": omitted})
            remaining -= 1
        if rows:
            sources.append({"source": source_id, "rows": rows})
    return {"sources": sources}


def _sketch_column_id(source: dict[str, Any], value: Any) -> str | None:
    text = str(value or "").strip()
    headers = [item for item in source.get("headers", []) if isinstance(item, dict)]
    ids = {str(item.get("id")) for item in headers if item.get("id")}
    if text in ids:
        return text
    folded = text.casefold()
    matches = []
    for header in headers:
        path = [str(part).strip() for part in header.get("path", []) if str(part).strip()]
        names = {part.casefold() for part in path}
        if path:
            names.add(".".join(path).casefold())
            names.add(" / ".join(path).casefold())
        if folded and folded in names:
            matches.append(str(header.get("id")))
    return matches[0] if len(matches) == 1 else None


def _question_linked_header_ids(question: str, headers: dict[str, dict[str, Any]]) -> list[str]:
    """Return headers whose source path is explicitly named by the question."""

    question_text = norm_text(question)
    question_tokens = set(tokens(question))
    phrase_frequency: Counter[str] = Counter()
    header_phrases: dict[str, list[str]] = {}
    for col_id, header in headers.items():
        path = [str(part).strip() for part in header.get("path", []) if str(part).strip()]
        phrases = list(dict.fromkeys(norm_text(part) for part in path if norm_text(part)))
        header_phrases[col_id] = phrases
        phrase_frequency.update(phrases)
    linked = []
    for col_id, phrases in header_phrases.items():
        matched = False
        for phrase in phrases:
            # A repeated leaf such as "Goals" or "2012" does not identify one
            # physical column. Parent-path checks below resolve those cases.
            if phrase_frequency[phrase] != 1:
                continue
            phrase_tokens = set(tokens(phrase))
            if not phrase_tokens:
                continue
            # Multiword paths must be present as a phrase. A single meaningful
            # header token is sufficient, but substring matches are not.
            if " " in phrase:
                matched = phrase in question_text
            else:
                matched = next(iter(phrase_tokens)) in question_tokens
            if matched:
                break
        if matched:
            linked.append(col_id)
    return linked


def materialization_certificate(
    *,
    question: str,
    table_sketch: dict[str, Any],
    answer_contract: dict[str, Any] | None,
    requirements: list[dict[str, Any]],
    signature: AnswerabilitySignature,
    compilation: EvidenceCompilation,
) -> dict[str, Any]:
    """Summarize full-table plan consequences and expose concrete conflicts."""

    contract = _normalized_answer_contract(answer_contract)
    sources = {
        str(item.get("id")): item
        for item in table_sketch.get("sources", [])
        if isinstance(item, dict) and item.get("id")
    }
    slices = {item.source: item for item in signature.slices}
    conflicts: list[dict[str, str]] = []
    advisories: list[dict[str, str]] = []
    source_summaries = []

    for diagnostic in compilation.diagnostics:
        if bool(diagnostic.get("blocking", True)):
            conflicts.append({
                "code": str(diagnostic.get("code", "COMPILER_CONFLICT")),
                "message": str(diagnostic.get("message", "Blocking compiler diagnostic.")),
            })

    raw_by_source: dict[str, list[dict[str, Any]]] = {}
    for raw in requirements or []:
        if not isinstance(raw, dict):
            continue
        source_id = str(raw.get("source", "") or "")
        if source_id in sources:
            raw_by_source.setdefault(source_id, []).append(raw)

    for plan in compilation.tables:
        source = sources.get(plan.table_id, {})
        evidence_slice = slices.get(plan.table_id)
        headers = {
            str(item.get("id")): item
            for item in source.get("headers", [])
            if isinstance(item, dict) and item.get("id")
        }
        selected = set(plan.col_ids)
        question_linked_headers = _question_linked_header_ids(question, headers)
        uses: dict[str, list[str]] = {}
        for raw in raw_by_source.get(plan.table_id, []):
            for field in raw.get("fields", []) or []:
                if not isinstance(field, dict):
                    continue
                col_id = _sketch_column_id(
                    source, field.get("column", field.get("col"))
                )
                if col_id:
                    uses.setdefault(col_id, []).append(
                        str(field.get("use", "return") or "return").lower()
                    )

        missing_fields = sorted(col_id for col_id in uses if col_id not in selected)
        if missing_fields:
            conflicts.append({
                "code": "UNMATERIALIZED_REQUIRED_FIELD",
                "message": (
                    f"Source {plan.table_id} did not materialize required fields "
                    f"{', '.join(missing_fields)}."
                ),
            })

        expression = contract.get("expression", {}) if contract else {}
        expression_refs = [
            item for item in expression.get("args", [])
            if isinstance(item, dict) and str(item.get("source")) == plan.table_id
        ]
        return_ref = expression.get("return")
        if isinstance(return_ref, dict) and str(return_ref.get("source")) == plan.table_id:
            expression_refs.append(return_ref)
        missing_expression_fields = sorted({
            str(item.get("column")) for item in expression_refs
            if item.get("column") and str(item.get("column")) not in selected
        })
        if missing_expression_fields:
            conflicts.append({
                "code": "UNMATERIALIZED_ANSWER_OPERAND",
                "message": (
                    f"Source {plan.table_id} omits answer-expression fields "
                    f"{', '.join(missing_expression_fields)}."
                ),
            })
        missing_expression_rows = sorted({
            str(item.get("row")) for item in expression_refs
            if item.get("row") and str(item.get("row")) not in set(plan.row_ids)
        })
        if missing_expression_rows:
            conflicts.append({
                "code": "UNMATERIALIZED_ANSWER_ARGUMENT",
                "message": (
                    f"Source {plan.table_id} omits answer-expression rows "
                    f"{', '.join(missing_expression_rows)}."
                ),
            })

        omitted_named_headers = sorted(
            col_id for col_id in question_linked_headers if col_id not in selected
        )
        if omitted_named_headers:
            # Lexical overlap identifies an alternative worth recording, but
            # does not prove that the selected answer field is wrong. Treat it
            # as diagnostic context rather than opening an LLM repair call.
            advisories.append({
                "code": "OMITTED_QUESTION_LINKED_FIELD",
                "message": (
                    f"Source {plan.table_id} omits explicitly question-linked fields "
                    f"{', '.join(omitted_named_headers)}."
                ),
            })

        broken_bindings = []
        for dependency in plan.dependencies:
            if dependency.get("kind") != "alignment":
                continue
            endpoints = [str(value) for value in dependency.get("col_ids", [])]
            if any(col_id not in selected for col_id in endpoints):
                broken_bindings.append(endpoints)
        if broken_bindings:
            conflicts.append({
                "code": "UNMATERIALIZED_BINDING",
                "message": f"Source {plan.table_id} has an incomplete tuple binding.",
            })

        if (
            contract
            and contract["coverage"] == "all"
            and evidence_slice is not None
            and evidence_slice.rows
            and not evidence_slice.where
        ):
            conflicts.append({
                "code": "SAMPLED_COMPLETE_DOMAIN",
                "message": (
                    f"Source {plan.table_id} uses explicit sample rows for an all-coverage answer."
                ),
            })


        if (
            contract
            and contract["shape"] == "row"
            and len(selected) < int((source.get("shape") or [0, 0])[1] or 0)
        ):
            conflicts.append({
                "code": "INCOMPLETE_ROW_MATERIALIZATION",
                "message": f"Source {plan.table_id} does not expose every field of a row answer.",
            })

        if (
            contract
            and contract["shape"] != "decision"
            and evidence_slice is not None
            and evidence_slice.where
            and not plan.row_ids
        ):
            advisories.append({
                "code": "EMPTY_ANSWER_DOMAIN",
                "message": (
                    f"The complete predicate on {plan.table_id} materialized zero rows. "
                    "Interpret the empty set according to the requested operation."
                ),
            })

        # Duplicate leaf labels are harmless when the question identifies the
        # selected parent path. They become a concrete review obligation when
        # the selected field conflicts with the uniquely mentioned parent.
        by_leaf: dict[str, list[tuple[str, list[str]]]] = {}
        for col_id, header in headers.items():
            path = [str(part).strip() for part in header.get("path", []) if str(part).strip()]
            if path:
                by_leaf.setdefault(path[-1].casefold(), []).append((col_id, path))
        question_folded = question.casefold()
        header_conflicts = []
        for _leaf, alternatives in by_leaf.items():
            if len(alternatives) < 2 or not any(col_id in selected for col_id, _ in alternatives):
                continue
            parent_mentions = [
                col_id
                for col_id, path in alternatives
                if any(
                    len(parent) > 2 and parent.casefold() in question_folded
                    for parent in path[:-1]
                )
            ]
            if len(parent_mentions) == 1 and parent_mentions[0] not in selected:
                header_conflicts.append({
                    "selected": sorted(col_id for col_id, _ in alternatives if col_id in selected),
                    "question_linked": parent_mentions[0],
                    "paths": [path for _col_id, path in alternatives],
                })
        if header_conflicts:
            # A repeated leaf under another mentioned parent is ambiguous, not
            # by itself contradictory. The full path remains visible to later
            # evidence auditing without forcing a speculative plan rewrite.
            advisories.append({
                "code": "HEADER_PATH_CONFLICT",
                "message": (
                    f"Source {plan.table_id} selected a repeated leaf header under a parent "
                    "different from the uniquely question-linked parent."
                ),
            })

        row_path_conflicts = []
        axis_index = source.get("axis_index", {})
        selected_rows = set(plan.row_ids)
        if contract and contract.get("coverage") == "one" and isinstance(axis_index, dict):
            question_folded = norm_text(question)
            for group in axis_index.get("repeated_row_leaves", []) or []:
                alternatives = [
                    item for item in group.get("paths", [])
                    if isinstance(item, dict) and item.get("row") and item.get("path")
                ]
                if len(alternatives) < 2 or not any(
                    str(item["row"]) in selected_rows for item in alternatives
                ):
                    continue
                parent_mentions = []
                for item in alternatives:
                    path = [str(part).strip() for part in item.get("path", []) if str(part).strip()]
                    if any(
                        len(tokens(parent)) > 0 and norm_text(parent) in question_folded
                        for parent in path[:-1]
                    ):
                        parent_mentions.append(str(item["row"]))
                if len(set(parent_mentions)) == 1 and parent_mentions[0] not in selected_rows:
                    row_path_conflicts.append({
                        "selected": sorted(
                            str(item["row"]) for item in alternatives
                            if str(item["row"]) in selected_rows
                        ),
                        "question_linked": parent_mentions[0],
                        "paths": [item["path"] for item in alternatives],
                    })
        if row_path_conflicts:
            advisories.append({
                "code": "ROW_PATH_CONFLICT",
                "message": (
                    f"Source {plan.table_id} selected a repeated row leaf under a parent "
                    "different from the uniquely question-linked parent."
                ),
            })

        source_summaries.append({
            "source": plan.table_id,
            "source_shape": source.get("shape", []),
            "materialized_rows": len(plan.row_ids),
            "row_ids_preview": _scope_preview(plan.row_ids),
            "selected_headers": [
                {
                    "id": col_id,
                    "path": headers.get(col_id, {}).get("path", []),
                    "uses": sorted(set(uses.get(col_id, []))),
                }
                for col_id in plan.col_ids
            ],
            "complete_predicate": (
                [item.model_dump(mode="json") for item in evidence_slice.where]
                if evidence_slice is not None else []
            ),
            "where_logic": evidence_slice.where_logic if evidence_slice is not None else "all",
            "bindings": [
                list(dependency.get("col_ids", []))
                for dependency in plan.dependencies
                if dependency.get("kind") == "alignment"
            ],
            "omitted_candidate_columns": [
                str(col_id) for col_id in source.get("candidate_columns", [])
                if str(col_id) not in selected
            ][:16],
            "question_linked_headers": question_linked_headers,
            "omitted_question_linked_headers": omitted_named_headers,
            "structural_ambiguities": list(source.get("ambiguities", []))[:8],
            "header_path_conflicts": header_conflicts,
            "row_path_conflicts": row_path_conflicts,
        })

    deduplicated = []
    seen = set()
    for item in conflicts:
        key = (item["code"], item["message"])
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
    return {
        "status": "conflict" if deduplicated else "verified",
        "answer_contract": contract,
        "sources": source_summaries,
        "compiler_diagnostics": list(compilation.diagnostics)[:20],
        "conflicts": deduplicated,
        "advisories": advisories,
    }


def compile_and_encode_with_repair(
    *,
    question: str,
    question_id: str,
    table_sketch: dict[str, Any],
    initial_signature: AnswerabilitySignature,
    compresstab: CompressTab,
    token_budget: int,
    llm: JsonLLMClient,
    repair_model: str,
    answer_contract: dict[str, Any] | None = None,
    planner_requirements: list[dict[str, Any]] | None = None,
    max_plan_verifications: int = 1,
) -> tuple[
    AnswerabilitySignature,
    dict[str, Any] | None,
    Any,
    CompressedTable,
    list[dict[str, Any]],
]:
    """Verify the logical plan, then compile its minimum-token complete view."""

    signature = initial_signature
    events: list[dict[str, Any]] = []
    requirements = list(planner_requirements or [])

    verification_budget = max(0, int(max_plan_verifications))
    fixed_payload = _normalized_answer_contract(answer_contract) or {}
    if requirements and verification_budget > 0:
        for verification_round in range(verification_budget):
            initial_compilation = compresstab.compile_signature(signature)
            initial_certificate = materialization_certificate(
                question=question,
                table_sketch=table_sketch,
                answer_contract=answer_contract,
                requirements=requirements,
                signature=signature,
                compilation=initial_compilation,
            )
            verification_event: dict[str, Any] = {
                "step": "verify_materialized_answerability_plan",
                "round": verification_round,
                "initial_signature": signature.model_dump(mode="json"),
                "certificate": initial_certificate,
                "audited": False,
                "accepted_revision": False,
            }
            if not initial_certificate["conflicts"]:
                events.append(verification_event)
                break

            proposal = {
                "answer": _normalized_answer_contract(answer_contract),
                "requirements": requirements,
            }
            repaired_data = SignatureAuditorAgent(llm, repair_model).repair(
                question=question,
                table_sketch=table_sketch,
                proposal=proposal,
                violations=initial_certificate["conflicts"],
                materialization_certificate=initial_certificate,
            )
            repaired_answer = _normalized_answer_contract(repaired_data.get("answer")) or {}
            for key in ("history", "shape", "coverage", "multiplicity"):
                if key in fixed_payload:
                    repaired_answer[key] = fixed_payload[key]
            repaired_data["answer"] = repaired_answer
            revised_contract, revised_signature, lowering_violations = lower_answer_requirements(
                repaired_data,
                table_sketch=table_sketch,
            )
            lowering_violations.extend(_expression_erasure_violations(
                _normalized_answer_contract(answer_contract),
                _normalized_answer_contract(revised_contract),
            ))
            revised_compilation = compresstab.compile_signature(revised_signature)
            revised_requirements = list(repaired_data.get("requirements", []) or [])
            revised_certificate = materialization_certificate(
                question=question,
                table_sketch=table_sketch,
                answer_contract=revised_contract,
                requirements=revised_requirements,
                signature=revised_signature,
                compilation=revised_compilation,
            )
            revised_conflicts = [*lowering_violations, *revised_certificate["conflicts"]]
            verification_event.update({
                "audited": True,
                "revised_requirements": revised_requirements,
                "revised_signature": revised_signature.model_dump(mode="json"),
                "revised_certificate": revised_certificate,
                "lowering_violations": lowering_violations,
            })
            if not revised_conflicts:
                signature = revised_signature
                answer_contract = _normalized_answer_contract(revised_contract)
                requirements = revised_requirements
                verification_event["accepted_revision"] = True
                events.append(verification_event)
                break

            events.append(verification_event)
            # A still-conflicting but well-typed revision becomes the next
            # proposal only when another bounded verification round remains.
            if verification_round + 1 < verification_budget and not lowering_violations:
                signature = revised_signature
                answer_contract = _normalized_answer_contract(revised_contract)
                requirements = revised_requirements

    compilation = compresstab.compile_signature(signature)
    compressed = compresstab.encode_signature(
        question_id=question_id,
        signature=signature,
        token_budget=token_budget,
    )
    return signature, answer_contract, compilation, compressed, events
