"""Compile evidence requirements into a minimum-token readable table witness."""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from typing import Any, Callable

from compresstab.codec.evidence import (
    CANONICAL_CELL_SEPARATOR,
    CANONICAL_LINE_SEPARATOR,
    EvidencePlanOptimizer,
)
from compresstab.evidence import AnswerabilitySignature, EvidenceCompilation, EvidenceCompiler
from compresstab.table.store import TableStore, approx_token_count, tokens
from compresstab.types import EvidenceRef


def _compact_sketch_value(value: Any, question: str, *, limit: int = 720) -> str:
    """Keep a bounded head plus a query-linked window for planner previews."""

    text = str(value)
    if len(text) <= limit:
        return text
    terms = [
        term for term in re.findall(r"[A-Za-z0-9%$]+", question)
        if len(term) >= 4
    ]
    positions = [
        position
        for term in terms
        if (position := text.casefold().find(term.casefold())) >= 0
    ]
    head_size = min(220, limit // 3)
    tail_size = min(100, limit // 6)
    focus_size = limit - head_size - tail_size - 10
    center = min(positions) if positions else len(text) // 2
    start = max(head_size, center - focus_size // 3)
    end = min(len(text) - tail_size, start + focus_size)
    return f"{text[:head_size]} ... {text[start:end]} ... {text[-tail_size:]}"


@dataclass
class CompressedTable:
    question_id: str
    text: str
    evidence_refs: list[EvidenceRef]
    token_count: int
    token_budget: int
    selected_row_ids: list[str]
    selected_col_ids: list[str]
    selected_encodings: dict[str, str]
    encoding_trace: dict[str, Any]
    dependencies: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    answerability_signature: dict[str, Any]

    @property
    def evidence_ref_ids(self) -> list[str]:
        return [ref.ref_id for ref in self.evidence_refs]

    @property
    def recoverable_source_ref_ids(self) -> list[str]:
        """Source cells whose original values can be decoded from the witness."""

        reversible = {
            "raw",
            "local_dictionary",
            "prefix_factorization",
            "suffix_factorization",
            "exact_delta",
            "exact_delta+exact_patch",
        }
        residual_refs = {
            str(cell.get("source_id"))
            for table in self.encoding_trace.get("tables", {}).values()
            for selected in table.get("selected", {}).values()
            for cell in selected.get("residual_cells", [])
            if cell.get("source_id")
        }
        recoverable = []
        for ref_id in self.evidence_ref_ids:
            parts = ref_id.split(":")
            if len(parts) != 3:
                continue
            table_id, _row_id, col_id = parts
            table_trace = (
                self.encoding_trace.get("tables", {}).get(table_id, {}) or {}
            )
            selected_traces = table_trace.get("selected", {}) or {}
            covering = [
                trace for trace in selected_traces.values()
                if col_id in (trace.get("source_col_ids") or [])
            ]
            source_recoverable = (
                all(bool(trace.get("source_reversible")) for trace in covering)
                if covering else None
            )
            operator = self.selected_encodings.get(
                f"{table_id}:{col_id}",
                self.selected_encodings.get(col_id, "raw"),
            )
            if (
                source_recoverable is True
                or (source_recoverable is None and operator in reversible)
                or ref_id in residual_refs
            ):
                recoverable.append(ref_id)
        return recoverable

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "evidence_refs": [ref.model_dump(mode="json") for ref in self.evidence_refs],
            "evidence_ref_ids": self.evidence_ref_ids,
            "recoverable_source_ref_ids": self.recoverable_source_ref_ids,
            "token_count": self.token_count,
            "token_budget": self.token_budget,
            "selected_row_ids": self.selected_row_ids,
            "selected_col_ids": self.selected_col_ids,
            "selected_encodings": self.selected_encodings,
            "encoding_trace": self.encoding_trace,
            "dependencies": self.dependencies,
            "diagnostics": self.diagnostics,
            "answerability_signature": self.answerability_signature,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompressedTable":
        """Strictly restore the exact Reader-facing witness saved by Stage A."""

        if not isinstance(payload, dict):
            raise TypeError("CompressedTable payload must be a dictionary")
        required = {
            "question_id", "text", "evidence_refs", "evidence_ref_ids",
            "recoverable_source_ref_ids", "token_count", "token_budget",
            "selected_row_ids", "selected_col_ids", "selected_encodings",
            "encoding_trace", "dependencies", "diagnostics",
            "answerability_signature",
        }
        missing = required - set(payload)
        extra = set(payload) - required
        if missing:
            raise ValueError(f"CompressedTable is missing fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"CompressedTable has unknown fields: {sorted(extra)}")
        signature = AnswerabilitySignature.model_validate(payload["answerability_signature"])
        restored = cls(
            question_id=str(payload["question_id"]),
            text=str(payload["text"]),
            evidence_refs=[EvidenceRef.model_validate(item) for item in payload["evidence_refs"]],
            token_count=int(payload["token_count"]),
            token_budget=int(payload["token_budget"]),
            selected_row_ids=[str(value) for value in payload["selected_row_ids"]],
            selected_col_ids=[str(value) for value in payload["selected_col_ids"]],
            selected_encodings={
                str(key): str(value) for key, value in payload["selected_encodings"].items()
            },
            encoding_trace=dict(payload["encoding_trace"]),
            dependencies=[dict(item) for item in payload["dependencies"]],
            diagnostics=[dict(item) for item in payload["diagnostics"]],
            answerability_signature=signature.model_dump(mode="json"),
        )
        if restored.token_count < 0 or restored.token_budget < 1:
            raise ValueError("CompressedTable token counts are invalid")
        if restored.evidence_ref_ids != [str(value) for value in payload["evidence_ref_ids"]]:
            raise ValueError("Serialized evidence_ref_ids do not match EvidenceRefs")
        if restored.recoverable_source_ref_ids != [
            str(value) for value in payload["recoverable_source_ref_ids"]
        ]:
            raise ValueError("Serialized recoverable refs do not match the witness encoding")
        return restored


class CompressTab:
    """Compile an Answerability Signature into a minimum-cost evidence witness."""

    def __init__(
        self,
        store: TableStore,
        *,
        profile_cache_dir: str | None = None,
        evidence_coding: str = "optimized",
        optimization_strategy: str = "global_dp",
        token_counter: Callable[[Any], int] | None = None,
        token_counter_name: str | None = None,
        joint_realizations: bool = True,
        factor_row_addresses: bool = True,
    ):
        self.store = store
        self.profile_cache_dir = profile_cache_dir
        self.evidence_coding = evidence_coding
        self.optimization_strategy = optimization_strategy
        self.joint_realizations = bool(joint_realizations)
        self.factor_row_addresses = bool(factor_row_addresses)
        self.has_exact_token_counter = token_counter is not None
        self.count_tokens = token_counter or approx_token_count
        self.token_counter_name = token_counter_name or (
            "configured_tokenizer" if token_counter is not None else "approximate_counter"
        )
        self.encoder = EvidencePlanOptimizer(
            store,
            profile_cache_dir=profile_cache_dir,
            encoding_mode=evidence_coding,
            optimization_strategy=optimization_strategy,
            token_counter=self.count_tokens,
            token_counter_is_exact=self.has_exact_token_counter,
            joint_realizations=self.joint_realizations,
            factor_row_addresses=self.factor_row_addresses,
        )

    def offline_profile(self) -> dict[str, Any]:
        """Materialize reusable, question-independent table profiles."""

        return {
            table.table_id: self.encoder.profiles.describe(table)
            for table in self.store.artifact.tables
        }

    @staticmethod
    def _hierarchy_rows(
        table: Any,
        question: str,
        candidate_rows: list[str],
        *,
        limit: int,
    ) -> list[str]:
        """Re-rank candidate rows by full axis paths, not leaf text alone."""

        if len(table.row_header_paths) != len(table.rows):
            return candidate_rows[:limit]
        question_tokens = set(tokens(question))
        question_text = " ".join(tokens(question))
        original_rank = {row_id: index for index, row_id in enumerate(candidate_rows)}
        ranked = []
        for row_idx, raw_path in enumerate(table.row_header_paths):
            path = [str(part).strip() for part in raw_path if str(part).strip()]
            if not path:
                continue
            path_tokens = set(tokens(" ".join(path)))
            overlap = len(question_tokens & path_tokens)
            phrase_hits = sum(
                1 for part in path
                if len(tokens(part)) > 1 and " ".join(tokens(part)) in question_text
            )
            row_id = f"r{row_idx}"
            candidate_bonus = 1.0 / (1.0 + original_rank[row_id]) if row_id in original_rank else 0.0
            score = 3.0 * phrase_hits + float(overlap) + candidate_bonus
            if score > 0:
                ranked.append((score, row_id))
        ranked.sort(key=lambda item: (-item[0], int(item[1][1:])))
        ordered = [row_id for _score, row_id in ranked]
        ordered.extend(row_id for row_id in candidate_rows if row_id not in ordered)
        return list(dict.fromkeys(ordered))[:limit]

    @staticmethod
    def _axis_index(table: Any, candidate_rows: list[str]) -> dict[str, Any] | None:
        """Build a bounded source-derived map of complete hierarchical addresses."""

        entries = []
        if len(table.row_header_paths) == len(table.rows):
            for row_id in candidate_rows[:18]:
                row_idx = int(row_id[1:])
                path = [
                    str(part).strip() for part in table.row_header_paths[row_idx]
                    if str(part).strip()
                ]
                if path:
                    entries.append({"row": row_id, "path": path})
        candidate_set = set(candidate_rows[:18])
        persisted_signals = []
        for item in table.structure_candidates:
            if item.kind != "row_role":
                continue
            match = re.search(r":row_role:(r\d+)$", item.candidate_id)
            if not match:
                continue
            row_id = match.group(1)
            row_idx = int(row_id[1:])
            if row_id in candidate_set and 0 <= row_idx < len(table.rows):
                persisted_signals.append({
                    "row": row_id,
                    "label": str(table.rows[row_idx][0]) if table.rows[row_idx] else "",
                    "signals": list(item.signals),
                })
        semantic_signals = persisted_signals or [
            {
                "row": str(item.get("row", "")),
                "label": str(item.get("label", "")),
                "signals": [str(value) for value in item.get("signals", []) or []],
            }
            for item in table.metadata.get("row_semantic_signals", []) or []
            if isinstance(item, dict) and str(item.get("row", "")) in candidate_set
        ][:18]
        hierarchical = bool(entries) and any(len(item["path"]) > 1 for item in entries)
        if not hierarchical and not semantic_signals:
            return None

        by_leaf: dict[str, list[dict[str, Any]]] = {}
        for item in entries:
            by_leaf.setdefault(item["path"][-1].casefold(), []).append(item)
        repeated = []
        for items in by_leaf.values():
            if len(items) < 2:
                continue
            repeated.append({
                "leaf": items[0]["path"][-1],
                "paths": [
                    {"row": item["row"], "path": item["path"]}
                    for item in items[:8]
                ],
            })
            if len(repeated) >= 6:
                break
        result = {"row_semantic_signals": semantic_signals}
        if hierarchical:
            result.update({
                "row_max_depth": max(len(item["path"]) for item in entries),
                "question_linked_row_paths": entries,
                "repeated_row_leaves": repeated,
            })
        return result

    def describe(self, question: str, token_budget: int) -> dict[str, Any]:
        """Expose a bounded MSchema-style sketch plus linked candidate tuples."""

        sources: list[dict[str, Any]] = []
        for table in self.store.artifact.tables:
            hits = self.store.search(
                question,
                top_k=34,
                table_id=table.table_id,
            )
            candidate_rows, candidate_cols = self.store.candidate_rows_and_cols(
                question,
                max_rows=18,
                max_cols=16,
                table_id=table.table_id,
                hits=hits,
            )
            candidate_rows = self._hierarchy_rows(
                table, question, candidate_rows, limit=18
            )
            # A small source should be understood as a complete relation, not
            # as if its query-linked rows were the whole domain. Keep linked
            # rows first so the existing global sketch budget drops unrelated
            # rows before relevant ones when the rendered values are long.
            if len(table.rows) <= 18:
                candidate_rows = list(dict.fromkeys([
                    *candidate_rows,
                    *(f"r{index}" for index in range(len(table.rows))),
                ]))
            question_tokens = set(tokens(question))
            context_overlap = sum(
                len(question_tokens & set(tokens(item.value)))
                for item in table.contexts
                if item.kind in {"caption", "section", "note"}
            )
            relevance = round(
                sum(max(0.0, float(hit.score)) for hit in hits[:8])
                + 1.5 * context_overlap,
                4,
            )
            if len(table.headers) <= 16:
                evidence_cols = [f"c{i}" for i in range(len(table.headers))]
            else:
                evidence_cols = list(dict.fromkeys(["c0", *candidate_cols]))[:16]

            profiles = self.encoder.profiles.table_profiles(table)
            headers = []
            for col_idx in range(len(table.headers)):
                col_id = f"c{col_idx}"
                profile = profiles[col_id]
                header: dict[str, Any] = {
                    "id": col_id,
                    "path": table.header_path(col_idx),
                    "type": profile.type_hint,
                }
                samples = []
                for row_id in candidate_rows:
                    row_idx = int(row_id[1:])
                    if 0 <= row_idx < len(table.rows) and col_idx < len(table.rows[row_idx]):
                        value = str(table.rows[row_idx][col_idx]).strip()
                        if value and value not in samples:
                            samples.append(value)
                # MSchema-style examples are semantic hints, not statistics.
                # Stop as soon as three distinct values are found instead of
                # counting an entire long column for every question.
                for row in table.rows:
                    value = str(row[col_idx]).strip() if col_idx < len(row) else ""
                    if value and value not in samples:
                        samples.append(value)
                    if len(samples) >= 3:
                        break
                compact_samples = []
                for value in samples[:3]:
                    compact_samples.append(value if len(value) <= 48 else value[:47] + "...")
                if compact_samples:
                    header["samples"] = compact_samples
                if profile.numeric_count:
                    header["min"] = profile.numeric_min
                    header["max"] = profile.numeric_max
                elif profile.distinct_count <= 20:
                    header["distinct"] = profile.distinct_count
                if profile.missing_count:
                    header["missing"] = profile.missing_count
                dirty = profile.ambiguous_count + profile.reversible_count
                if dirty:
                    header["dirty"] = dirty
                headers.append(header)

            evidence = [
                {
                    "row": row_id,
                    "values": {
                        col_id: _compact_sketch_value(
                            table.rows[int(row_id[1:])][int(col_id[1:])],
                            question,
                        )
                        for col_id in evidence_cols
                    },
                }
                for row_id in candidate_rows
            ]
            ambiguities = [
                {
                    "id": item.candidate_id,
                    "kind": item.kind,
                    "selected": item.selected,
                    "alternatives": item.alternatives,
                    "signals": item.signals,
                }
                for item in table.structure_candidates
                if not item.selected or item.alternatives
            ]
            context_items = [
                {
                    "id": item.context_id,
                    "kind": item.kind,
                    "value": item.value,
                    "scope": item.scope,
                }
                for item in table.contexts
            ]
            context_items.sort(key=lambda item: (
                -len(question_tokens & set(tokens(item["value"]))),
                0 if item["kind"] in {"scale", "unit", "section", "note"} else 1,
            ))
            source_description = {
                "id": table.table_id,
                "shape": [len(table.rows), len(table.headers)],
                "caption": table.caption,
                "relevance": relevance,
                "candidate_columns": candidate_cols,
                "headers": headers,
                "contexts": context_items,
                "evidence": evidence,
                "evidence_complete": len(evidence) == len(table.rows),
                "evidence_omitted": max(0, len(table.rows) - len(evidence)),
                "ambiguities": ambiguities,
            }
            axis_index = self._axis_index(table, candidate_rows)
            if axis_index is not None:
                source_description["axis_index"] = axis_index
            sources.append(source_description)

        sketch = {"sources": sources}
        max_tokens = max(800, min(3200, int(token_budget)))

        def too_large() -> bool:
            # The planner receives compact JSON. Budget that exact surface with
            # its configured tokenizer instead of undercounting identifiers and
            # punctuation with the generic table estimate.
            rendered = json.dumps(
                sketch,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
            return max(
                self.count_tokens(rendered),
                approx_token_count(sketch),
            ) > max_tokens

        # Bound schema and context text as well as sampled rows.
        while too_large() and any(source["ambiguities"] for source in sources):
            for source in sources:
                if source["ambiguities"]:
                    source["ambiguities"].pop()
        while too_large() and any(
            (source.get("axis_index") or {}).get("repeated_row_leaves")
            for source in sources
        ):
            for source in sources:
                repeated = (source.get("axis_index") or {}).get("repeated_row_leaves", [])
                if repeated:
                    repeated.pop()
        while too_large() and any(
            len((source.get("axis_index") or {}).get("row_semantic_signals", [])) > 6
            for source in sources
        ):
            for source in sources:
                signals = (source.get("axis_index") or {}).get("row_semantic_signals", [])
                if len(signals) > 6:
                    signals.pop()
        while too_large() and any(
            len((source.get("axis_index") or {}).get("question_linked_row_paths", [])) > 6
            for source in sources
        ):
            for source in sources:
                paths = (source.get("axis_index") or {}).get("question_linked_row_paths", [])
                if len(paths) > 6:
                    paths.pop()
        while too_large() and any(len(source["contexts"]) > 2 for source in sources):
            for source in sources:
                if len(source["contexts"]) > 2:
                    source["contexts"].pop()
        while too_large() and any(len(source["evidence"]) > 6 for source in sources):
            for source in sources:
                if len(source["evidence"]) > 6:
                    source["evidence"].pop()
        while too_large() and any(
            len(header.get("samples", [])) > 1
            for source in sources for header in source["headers"]
        ):
            for source in sources:
                for header in reversed(source["headers"]):
                    if len(header.get("samples", [])) > 1:
                        header["samples"].pop()
                        if not too_large():
                            break
                if not too_large():
                    break
        while too_large() and any(len(source["headers"]) > 4 for source in sources):
            changed = False
            for source in sources:
                protected = {"c0", *source.get("candidate_columns", [])}
                removable = [
                    index for index, header in enumerate(source["headers"])
                    if str(header.get("id")) not in protected
                ]
                if removable and len(source["headers"]) > 4:
                    source["headers"].pop(removable[-1])
                    source["headers_omitted"] = int(source.get("headers_omitted", 0)) + 1
                    changed = True
                if not too_large():
                    break
            if not changed:
                break
        while too_large() and any(header.get("samples") for source in sources for header in source["headers"]):
            for source in sources:
                for header in reversed(source["headers"]):
                    if header.get("samples"):
                        header.pop("samples", None)
                        if not too_large():
                            break
                if not too_large():
                    break
        while too_large() and any(len(source["evidence"]) > 2 for source in sources):
            for source in sources:
                if len(source["evidence"]) > 2:
                    source["evidence"].pop()
        while too_large() and any(len(source["headers"]) > 1 for source in sources):
            for source in sources:
                if len(source["headers"]) > 1:
                    source["headers"].pop()
                    source["headers_omitted"] = int(source.get("headers_omitted", 0)) + 1
                if not too_large():
                    break
        if too_large():
            for source in sources:
                source["contexts"] = []
                source["ambiguities"] = []
                source["evidence"] = source["evidence"][:1]
                source.pop("axis_index", None)
                for header in source["headers"]:
                    header.pop("samples", None)
        for source in sources:
            source_rows = int((source.get("shape") or [0])[0] or 0)
            shown_rows = len(source.get("evidence", []))
            source["evidence_complete"] = shown_rows == source_rows
            source["evidence_omitted"] = max(0, source_rows - shown_rows)
        return sketch

    def compile_signature(
        self,
        signature: AnswerabilitySignature | dict[str, Any],
    ) -> EvidenceCompilation:
        """Lower the Answerability Signature into executable coding obligations."""

        return EvidenceCompiler(self.store.artifact).compile(signature)

    def encode_signature(
        self,
        *,
        question_id: str,
        signature: AnswerabilitySignature | dict[str, Any],
        token_budget: int,
    ) -> CompressedTable:
        """Return the minimum-token complete realization of the signature."""

        budget = max(1, int(token_budget))
        compilation = self.compile_signature(signature)
        if compilation.has_blocking_diagnostics:
            problems = "; ".join(
                f"{item.get('code', 'INVALID_PLAN')}: {item.get('message', '')}"
                for item in compilation.diagnostics
                if item.get("blocking", True)
            )
            raise ValueError(f"Uncompilable Answerability Signature: {problems}")
        rendered = self._render_signature(compilation, budget)
        token_count = self.count_tokens(rendered["text"])
        rendered["encoding_trace"].update({
            "target_evidence_tokens": budget,
            "actual_evidence_tokens": token_count,
            "within_target": token_count <= budget,
            "tokens_over_target": max(0, token_count - budget),
        })
        multiple_tables = len(compilation.tables) > 1
        selected_rows = [
            f"{plan.table_id}:{row_id}" if multiple_tables else row_id
            for plan in compilation.tables
            for row_id in plan.row_ids
        ]
        selected_cols = [
            f"{plan.table_id}:{col_id}" if multiple_tables else col_id
            for plan in compilation.tables
            for col_id in plan.col_ids
        ]
        return CompressedTable(
            question_id=question_id,
            text=rendered["text"],
            evidence_refs=rendered["refs"],
            token_count=token_count,
            token_budget=budget,
            selected_row_ids=selected_rows,
            selected_col_ids=selected_cols,
            selected_encodings=rendered["selected_encodings"],
            encoding_trace=rendered["encoding_trace"],
            dependencies=[
                {**dependency, "table_id": plan.table_id}
                for plan in compilation.tables
                for dependency in plan.dependencies
            ],
            diagnostics=list(compilation.diagnostics),
            answerability_signature=compilation.signature.model_dump(),
        )

    def raw_context(self) -> str:
        """Render the full source for the matched raw-reader baseline."""

        lines: list[str] = []
        for table in self.store.artifact.tables:
            view = self.store.slice_table(
                table_id=table.table_id,
                row_ids=[f"r{i}" for i in range(len(table.rows))],
                col_ids=[f"c{i}" for i in range(len(table.headers))],
            )
            if len(self.store.artifact.tables) > 1:
                lines.append(f"TABLE {table.table_id} region={table.region_id}")
            if table.caption:
                lines.append(f"Caption: {table.caption}")
            lines.append("\t".join(["source_row", *[
                "/".join(item["header_path"]) for item in view["headers"]
            ]]))
            lines.extend(
                "\t".join([row["row_id"], *row["values"]])
                for row in view["rows"]
            )
        return "\n".join(lines)

    def _render_signature(
        self,
        compilation: EvidenceCompilation,
        budget: int,
        *,
        extra_diagnostics: list[dict[str, Any]] | None = None,
        encoder: EvidencePlanOptimizer | None = None,
    ) -> dict[str, Any]:
        encoder = encoder or self.encoder
        trace: dict[str, Any] = {
            "objective": "minimize complete rendered tokens subject to distinction preservation",
            "token_estimator": self.token_counter_name,
            "target_evidence_tokens": budget,
            "evidence_coding": encoder.encoding_mode,
            "optimization_strategy": encoder.optimization_strategy,
            "tables": {},
        }
        prepared: list[tuple[Any, Any, dict[str, Any], list[str]]] = []

        for plan in compilation.tables:
            table = self.store.table(plan.table_id)
            view = self.store.slice_table(
                table_id=plan.table_id,
                row_ids=plan.row_ids,
                col_ids=plan.col_ids,
            )
            fixed_lines = self._fixed_lines(plan, table)
            prepared.append((plan, table, view, fixed_lines))

        diagnostics = compilation.diagnostics if extra_diagnostics is None else extra_diagnostics

        def assemble(encoded_by_table: dict[str, dict[str, Any]]) -> dict[str, Any]:
            lines = ["COMPRESSTAB WITNESS"]
            refs: list[EvidenceRef] = []
            selected_encodings: dict[str, str] = {}
            for plan, table, view, fixed_lines in prepared:
                encoded = encoded_by_table[plan.table_id]
                axis = self._axis_addressed_view(plan, table, encoded)
                refs.extend(view.get("evidence_refs", []))
                refs.extend(self._derived_evidence_refs(plan, table, encoded))
                prefix = f"{plan.table_id}:" if len(compilation.tables) > 1 else ""
                selected_encodings.update({
                    f"{prefix}{col_id}": operator
                    for col_id, operator in encoded["selected_ops"].items()
                })
                lines.extend(fixed_lines)
                if encoded["decoder_lines"]:
                    lines.append("Encoding guide:")
                    lines.extend(encoded["decoder_lines"])
                if encoded.get("code_lines"):
                    lines.append(
                        "Coded columns (rows follow grid order):"
                    )
                    lines.extend(encoded["code_lines"])
                lines.extend(axis["lines"])
                paths = {
                    item["col_id"]: (
                        " / ".join(item.get("header_path", [])) or item.get("display", "")
                    )
                    for item in view.get("headers", [])
                }
                logical_owner: dict[str, str] = {}
                derived_expressions: dict[str, str] = {}
                for observable in plan.derived_observables:
                    by_role: dict[str, list[str]] = {}
                    for operand in observable.operands:
                        operand_id = str(operand.column)
                        label = derived_expressions.get(
                            operand_id, paths.get(operand_id, operand_id)
                        )
                        by_role.setdefault(str(operand.role), []).append(label)
                        logical_owner[operand_id] = str(observable.observable_id)

                    def group(*roles: str) -> str:
                        return " + ".join(
                            value for role in roles for value in by_role.get(role, [])
                        )

                    if observable.operator == "difference":
                        expression = (
                            f"{group('later', 'left')} - "
                            f"{group('earlier', 'right')}"
                        )
                    elif observable.operator == "ratio":
                        expression = (
                            f"({group('numerator')}) / ({group('denominator')})"
                        )
                    elif observable.operator == "percent_change":
                        expression = (
                            f"(({group('later')}) - ({group('earlier')})) / "
                            f"({group('earlier')}) * {observable.scale:g}"
                        )
                    else:
                        values = group(
                            "value", "input", "group", "rank", "left", "right"
                        )
                        expression = (
                            values if observable.operator == "sum"
                            else f"{observable.operator}({values})"
                        )
                    observable_id = str(observable.observable_id)
                    derived_expressions[observable_id] = expression
                    paths[observable_id] = f"{observable_id}:{expression}"
                physical_links = []
                for layout in plan.layouts:
                    for link in layout.get("links", []):
                        physical_link = list(dict.fromkeys(
                            logical_owner.get(str(col_id), str(col_id))
                            for col_id in link
                        ))
                        if len(physical_link) >= 2 and physical_link not in physical_links:
                            physical_links.append(physical_link)
                links = [
                    " <-> ".join(paths.get(col_id, col_id) for col_id in link)
                    for link in physical_links
                ]
                if links:
                    if axis["enabled"]:
                        addressed_links = [
                            " <-> ".join(link)
                            for link in physical_links
                        ]
                        lines.append(
                            "Links: "
                            + "; ".join(dict.fromkeys(addressed_links))
                            + " (same source_row)"
                        )
                    else:
                        lines.append("Links: " + "; ".join(dict.fromkeys(links)))
                if axis["grid_header"]:
                    lines.append(
                        EvidencePlanOptimizer._render_grid_line(axis["grid_header"])
                    )
                    lines.extend(
                        EvidencePlanOptimizer._render_grid_line(row)
                        for row in axis["grid_rows"]
                    )
                if encoded["patch_lines"]:
                    lines.extend(["Exact patches:", *encoded["patch_lines"]])
                if axis["enabled"]:
                    lines.append(
                        "Provenance: each value is "
                        f"{plan.table_id}:<source_row>:<column_id>; "
                        "axis entries preserve complete source paths"
                    )
                else:
                    derived_note = (
                        "; derived " + ", ".join(
                            f"{observable_id}={expression}"
                            for observable_id, expression in derived_expressions.items()
                        )
                        if derived_expressions else ""
                    )
                    lines.append(
                        "Provenance: each shown cell is "
                        f"{plan.table_id}:<source_row>:<column_id>; "
                        + ", ".join(
                            f"{col_id}={' / '.join(table.header_path(int(col_id[1:])))}"
                            for col_id in plan.col_ids
                        )
                        + derived_note
                    )
            if diagnostics:
                lines.append("Diagnostics:")
                lines.extend(f"- {item['code']}: {item['message']}" for item in diagnostics)
            return {
                "text": CANONICAL_LINE_SEPARATOR.join(
                    str(line).rstrip(" \r\n") for line in lines
                ),
                "refs": self._dedupe_refs(refs),
                "selected_encodings": selected_encodings,
            }

        raw_encoder = (
            encoder
            if encoder.encoding_mode == "raw"
            else EvidencePlanOptimizer(
                self.store,
                profile_cache_dir=self.profile_cache_dir,
                encoding_mode="raw",
                optimization_strategy="global_dp",
                token_counter=self.count_tokens,
                token_counter_is_exact=self.has_exact_token_counter,
                joint_realizations=False,
                factor_row_addresses=False,
            )
        )
        raw_by_table: dict[str, dict[str, Any]] = {}
        encoded_frontiers: dict[str, dict[str, Any]] = {}
        for plan, _table, view, _fixed_lines in prepared:
            raw_by_table[plan.table_id] = raw_encoder.materialize(
                slice_view=view,
                dependencies=plan.dependencies,
                target_tokens=budget,
                fixed_cost=(0, 0),
                derived_observables=plan.derived_observables,
            )
            encoded_frontiers[plan.table_id] = encoder.materialize(
                slice_view=view,
                dependencies=plan.dependencies,
                target_tokens=budget,
                fixed_cost=(0, 0),
                derived_observables=plan.derived_observables,
            )

        raw_bundle = assemble(raw_by_table)
        raw_lexical = len(tokens(raw_bundle["text"]))
        raw_bytes = len(raw_bundle["text"].encode("utf-8"))
        raw_variable_lexical = sum(
            int(encoded["selection_trace"]["selected_lexical_cost"])
            for encoded in raw_by_table.values()
        )
        raw_variable_bytes = sum(
            int(encoded["selection_trace"]["selected_utf8_bytes"])
            for encoded in raw_by_table.values()
        )
        fixed_cost = (
            max(0, raw_lexical - raw_variable_lexical),
            max(0, raw_bytes - raw_variable_bytes),
        )

        # Retain table alternatives until complete evidence is tokenized.
        global_frontier: list[
            tuple[int, int, int, tuple[str, ...], dict[str, dict[str, Any]]]
        ] = [(0, 0, 0, (), {})]
        for plan, _table, _view, _fixed_lines in prepared:
            table_id = plan.table_id
            expanded = []
            for token_cost, lexical_cost, byte_cost, order, chosen in global_frontier:
                for alternative in encoded_frontiers[table_id]["plan_alternatives"]:
                    expanded.append((
                        token_cost + int(alternative["token_cost"]),
                        lexical_cost + int(alternative["lexical_cost"]),
                        byte_cost + int(alternative["byte_cost"]),
                        (*order, *alternative["plan_order"]),
                        {**chosen, table_id: alternative},
                    ))
            global_frontier = self._distinct_evidence_plans(expanded)

        evaluated = [
            (
                self.count_tokens(assemble(chosen)["text"]),
                token_cost,
                lexical_cost,
                byte_cost,
                order,
                chosen,
            )
            for token_cost, lexical_cost, byte_cost, order, chosen in global_frontier
        ]
        complete_tokens, _tokens, _lexical, _bytes, global_order, chosen_by_table = min(
            evaluated,
            key=lambda item: (item[0], item[1], item[2], item[3], item[4]),
        )
        bundle = assemble(chosen_by_table)

        for plan, _table, _view, _fixed_lines in prepared:
            table_id = plan.table_id
            chosen = chosen_by_table[table_id]
            table_trace = dict(encoded_frontiers[table_id]["selection_trace"])
            table_trace.update({
                "selection_reason": (
                    "GLOBAL_MINIMUM_COST_CERTIFIED_PLAN"
                    if encoder.optimization_strategy == "global_dp"
                    else "LOCAL_GREEDY_CERTIFIED_PLAN"
                ),
                "minimum_required_tokens": int(chosen["token_cost"]),
                "selected_value_tokens": int(chosen["token_cost"]),
                "selected_lexical_cost": int(chosen["lexical_cost"]),
                "selected_utf8_bytes": int(chosen["byte_cost"]),
                "selected": chosen["selected_traces"],
                "shared_sections": chosen["shared_sections"],
                "plan_order": chosen["plan_order"],
                "axis_addressing": self._axis_addressed_view(
                    plan, self.store.table(table_id), chosen,
                )["trace"],
            })
            trace["tables"][table_id] = table_trace

        table_traces = list(trace["tables"].values())
        coding_tokens = sum(
            int(item.get("selected_value_tokens", 0) or 0) for item in table_traces
        )
        raw_equivalent_tokens = sum(
            int(item.get("raw_equivalent_tokens", 0) or 0) for item in table_traces
        )
        trace.update({
            "coding_value_tokens": coding_tokens,
            "raw_value_equivalent_tokens": raw_equivalent_tokens,
            "coding_value_token_savings": raw_equivalent_tokens - coding_tokens,
            "coding_value_compression_ratio": round(
                raw_equivalent_tokens / max(1, coding_tokens), 6,
            ),
            "global_plan_frontier_size": len(global_frontier),
            "global_plan_order": list(global_order),
            "calibrated_fixed_lexical_cost": fixed_cost[0],
            "calibrated_fixed_utf8_bytes": fixed_cost[1],
            "optimized_complete_witness_tokens": complete_tokens,
            "raw_complete_witness_tokens": self.count_tokens(raw_bundle["text"]),
        })
        return {
            **bundle,
            "encoding_trace": trace,
        }

    @classmethod
    def _axis_addressed_view(
        cls,
        plan: Any,
        table: Any,
        encoded: dict[str, Any],
    ) -> dict[str, Any]:
        """Expose hierarchy as reversible row and column source addresses."""

        row_entries: list[tuple[str, list[str]]] = []
        for row_id in plan.row_ids:
            row_idx = int(str(row_id)[1:])
            if row_idx >= len(table.row_header_paths):
                continue
            path = [str(value) for value in table.row_header_paths[row_idx] if str(value)]
            if path:
                row_entries.append((str(row_id), path))

        col_entries = [
            (str(col_id), table.header_path(int(str(col_id)[1:])))
            for col_id in plan.col_ids
            if len(table.header_path(int(str(col_id)[1:]))) > 1
        ]
        row_hierarchy = bool(row_entries) and any(len(path) > 1 for _row_id, path in row_entries)
        if not row_hierarchy and not col_entries:
            return {
                "enabled": False,
                "lines": list(encoded.get("row_axis_lines", [])),
                "grid_header": list(encoded["grid_header"]),
                "grid_rows": [list(row) for row in encoded["grid_rows"]],
                "trace": {
                    "enabled": False,
                    "row_axis_mode": encoded.get("row_axis_mode", "explicit"),
                    "row_axis_certification": encoded.get(
                        "row_axis_certification", {}
                    ),
                },
            }

        row_axis_col = None
        if row_hierarchy and len(row_entries) == len(plan.row_ids):
            for col_id in plan.col_ids:
                if not (str(col_id).startswith("c") and str(col_id)[1:].isdigit()):
                    continue
                col_idx = int(str(col_id)[1:])
                if all(
                    str(table.rows[int(row_id[1:])][col_idx]) == " / ".join(path)
                    for row_id, path in row_entries
                ):
                    row_axis_col = str(col_id)
                    break

        lines: list[str] = list(encoded.get("row_axis_lines", []))
        if row_hierarchy:
            heading = "Row axis"
            if row_axis_col is not None:
                heading += f" {row_axis_col}"
            lines.append(f"{heading} (ordered root-to-leaf paths):")
            lines.extend(cls._factor_axis_paths(row_entries))
        if col_entries:
            lines.append("Column axis (ordered root-to-leaf paths):")
            lines.extend(cls._factor_axis_paths(col_entries))

        # Axis paths are fixed metadata. Keeping the physical grid untouched
        # makes its candidate costs independent of the chosen representation.
        grid_header = list(encoded["grid_header"])
        grid_rows = [list(row) for row in encoded["grid_rows"]]

        row_paths_complete = (
            not row_hierarchy
            or {row_id for row_id, _path in row_entries} == set(plan.row_ids)
        )
        col_paths_complete = all(
            table.header_path(int(col_id[1:])) == path for col_id, path in col_entries
        )
        row_axis_mode = encoded.get("row_axis_mode", "explicit")
        if row_axis_mode == "ordered_range":
            cell_addresses_preserved = (
                len(grid_rows) == len(plan.row_ids)
                and bool(encoded.get("row_axis_certification", {}).get(
                    "row_addresses_recoverable"
                ))
            )
        else:
            cell_addresses_preserved = all(
                row and str(row[0]) == str(row_id)
                for row, row_id in zip(grid_rows, plan.row_ids)
            )
        checks = {
            "row_paths_complete": row_paths_complete,
            "column_paths_complete": col_paths_complete,
            "cell_addresses_preserved": cell_addresses_preserved,
            "omitted_row_values_recoverable": True,
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"Uncertified axis-addressed evidence for {table.table_id}: {checks}"
            )
        return {
            "enabled": True,
            "lines": lines,
            "grid_header": grid_header,
            "grid_rows": grid_rows,
            "trace": {
                "enabled": True,
                "row_axis_column": row_axis_col,
                "row_axis_mode": row_axis_mode,
                "row_path_count": len(row_entries) if row_hierarchy else 0,
                "column_path_count": len(col_entries),
                "inline_semantic_labels": True,
                "certification_checks": checks,
            },
        }

    @staticmethod
    def _inline_axis_label(value: Any) -> str:
        """Keep address aliases readable without changing their source paths."""

        return " ".join(str(value).replace("\t", " ").splitlines()).strip()

    def _derived_evidence_refs(
        self,
        plan: Any,
        table: Any,
        encoded: dict[str, Any],
    ) -> list[EvidenceRef]:
        """Create source-lineage references for visible virtual fields."""

        observables = {
            item.observable_id: item for item in plan.derived_observables
        }
        refs: list[EvidenceRef] = []
        for unit_id, trace in (encoded.get("selected_traces") or {}).items():
            observable = observables.get(str(unit_id))
            if observable is None or trace.get("operator") == "derive_from_operands":
                continue
            values = trace.get("visible_values") or {}
            lineage_by_row = trace.get("lineage_by_row") or {}
            for row_id, lineage in lineage_by_row.items():
                lineage_ids = [str(value) for value in lineage]
                source_cell_ids = []
                for lineage_id in lineage_ids:
                    source_ref = self.store.provenance.get(lineage_id)
                    if source_ref is not None:
                        source_cell_ids.extend(source_ref.source_cell_ids)
                preview_values = values.get(row_id, [])
                preview = str(preview_values[0]) if preview_values else ""
                payload = {
                    "table": table.table_id,
                    "row": row_id,
                    "observable": unit_id,
                    "expression": trace.get("derived_expression", ""),
                    "value": preview,
                    "lineage": lineage_ids,
                }
                source_hash = hashlib.sha256(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                refs.append(EvidenceRef(
                    ref_id=f"{table.table_id}:{row_id}:{unit_id}",
                    table_id=table.table_id,
                    source_type="derived",
                    row_ids=[str(row_id)],
                    col_ids=[
                        str(col_id)
                        for col_id in trace.get("source_col_ids", [])
                    ],
                    header_path=[str(trace.get("display_header", unit_id))],
                    unit=observable.unit or None,
                    scale=str(observable.scale) if observable.scale != 1 else None,
                    value_preview=preview,
                    lineage_ref_ids=lineage_ids,
                    source_cell_ids=list(dict.fromkeys(source_cell_ids)),
                    interpretation_status="derived-certified",
                    source_hash=source_hash,
                ))
        return refs

    @staticmethod
    def _factor_axis_paths(entries: list[tuple[str, list[str]]]) -> list[str]:
        """Factor repeated parents only when the complete text becomes shorter."""

        grouped: dict[tuple[str, ...], list[tuple[str, str]]] = {}
        order: list[tuple[str, ...]] = []
        for address, path in entries:
            parent = tuple(path[:-1])
            if parent not in grouped:
                grouped[parent] = []
                order.append(parent)
            grouped[parent].append((address, path[-1]))

        result: list[str] = []
        for parent in order:
            children = grouped[parent]
            direct = [
                f"- {address} = {json.dumps([*parent, leaf], ensure_ascii=False)}"
                for address, leaf in children
            ]
            if parent and len(children) > 1:
                factored = (
                    f"- {json.dumps(list(parent), ensure_ascii=False)} -> {{"
                    + ", ".join(
                        f"{address}: {json.dumps(leaf, ensure_ascii=False)}"
                        for address, leaf in children
                    )
                    + "}"
                )
                if approx_token_count(factored) < approx_token_count("\n".join(direct)):
                    result.append(factored)
                    continue
            result.extend(direct)
        return result

    @staticmethod
    def _distinct_evidence_plans(
        plans: list[
            tuple[int, int, int, tuple[str, ...], dict[str, dict[str, Any]]]
        ],
    ) -> list[
        tuple[int, int, int, tuple[str, ...], dict[str, dict[str, Any]]]
    ]:
        """Retain distinct cross-table plans for exact complete rendering."""

        unique: list[
            tuple[int, int, int, tuple[str, ...], dict[str, dict[str, Any]]]
        ] = []
        for plan in plans:
            if any(old[4] == plan[4] for old in unique):
                continue
            unique.append(plan)
        return unique

    @staticmethod
    def _fixed_lines(plan: Any, table: Any) -> list[str]:
        caption = f" ({table.caption})" if table.caption else ""
        lines = [
            f"TABLE {plan.table_id}{caption}",
            (
                f"Scope: {len(plan.row_ids)}/{len(table.rows)} rows; "
                f"{len(plan.col_ids)}/{len(table.headers)} columns"
            ),
        ]
        filter_clauses = []
        for layout in plan.layouts:
            predicates = []
            for predicate in layout.get("where", []):
                if not isinstance(predicate, dict):
                    continue
                col_id = str(predicate.get("col", ""))
                try:
                    header = " / ".join(table.header_path(int(col_id[1:])))
                except (ValueError, IndexError):
                    header = col_id
                rendered = (
                    f"{col_id}:{header} {predicate.get('op', 'eq')} "
                    f"{json.dumps(predicate.get('value'), ensure_ascii=False)}"
                )
                if rendered not in predicates:
                    predicates.append(rendered)
            if predicates:
                connective = " OR " if layout.get("where_logic") == "any" else " AND "
                clause = connective.join(predicates)
                filter_clauses.append(f"({clause})" if len(predicates) > 1 else clause)
        if filter_clauses:
            # Separate Signature slices contribute a union of complete source
            # populations, so their clauses are disjunctive at render time.
            lines.append("Complete source filter: " + " OR ".join(filter_clauses))
        contexts = [item for item in table.contexts if item.context_id in plan.context_ids]
        if contexts:
            lines.append("Context:")
            lines.extend(
                f"- {item.kind}: {item.value} (scope={item.scope})"
                for item in contexts
            )
        labels = {
            "exact_value": "exact",
            "identity_recoverable": "identity",
            "equality": "equality",
            "complete_domain": "complete",
            "order": "order",
            "threshold_membership": "threshold",
            "alignment": "alignment",
            "header_hierarchy": "hierarchy",
            "context": "context",
        }
        preserved = list(dict.fromkeys(
            labels[dependency["kind"]]
            for dependency in plan.dependencies
            if dependency.get("kind") in labels
        ))
        preserved.extend(
            f"derived-{observable.mode}({observable.observable_id})"
            for observable in plan.derived_observables
            if observable.required
            if f"derived-{observable.mode}({observable.observable_id})" not in preserved
        )
        if preserved:
            lines.append(f"Required distinctions: {', '.join(preserved)}; source-linked")
        return lines

    @staticmethod
    def _dedupe_refs(refs: list[EvidenceRef]) -> list[EvidenceRef]:
        return list({ref.ref_id: ref for ref in refs}.values())
