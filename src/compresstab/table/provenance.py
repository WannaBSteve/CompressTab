"""Evidence reference construction and novelty checks."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from compresstab.types import EvidenceRef
from compresstab.table.artifact import CanonicalTable


def _scale_terms(text: str) -> set[str]:
    """Return explicit magnitude terms without inferring a unit from values."""

    value = str(text or "")
    terms: set[str] = set()
    if "%" in value or re.search(r"\bpercent(?:age)?s?\b", value, re.I):
        terms.add("percent")
    if re.search(r"\bthousands?\b|(?:[$€£¥]|[A-Z]{2,3}\$)\s*[‘’']?000s?\b", value, re.I):
        terms.add("thousand")
    if re.search(r"\bmillions?\b|(?:[$€£¥]|[A-Z]{2,3}\$)\s*(?:m|mn)\b", value, re.I):
        terms.add("million")
    if re.search(r"\bbillions?\b|\bbn\b|(?:[$€£¥]|[A-Z]{2,3}\$)\s*b\b", value, re.I):
        terms.add("billion")
    return terms


def _cell_semantic_scale(table: CanonicalTable, row_idx: int, col_idx: int) -> str | None:
    """Bind scale to the selected measure before consulting table-wide context."""

    source = table.source_cell(row_idx, col_idx)
    interpretation_scales = {
        str(item.scale or item.unit or "").strip().lower()
        for item in (source.interpretations if source else [])
        if str(item.scale or item.unit or "").strip().lower()
        in {"thousand", "million", "billion", "percent"}
    }
    if len(interpretation_scales) == 1:
        return next(iter(interpretation_scales))

    row_path = (
        table.row_header_paths[row_idx]
        if 0 <= row_idx < len(table.row_header_paths)
        else []
    )
    row_label = table.rows[row_idx][0] if table.rows[row_idx] else ""
    value = table.rows[row_idx][col_idx]
    local_text = " / ".join([*table.header_path(col_idx), *row_path, row_label, value])
    local_scales = _scale_terms(local_text)
    # Percent is an intrinsic measure scale and overrides a broader monetary
    # declaration inherited by a flattened multi-level header.
    if "percent" in local_scales:
        return "percent"
    magnitude_scales = local_scales & {"thousand", "million", "billion"}
    if len(magnitude_scales) == 1:
        return next(iter(magnitude_scales))

    table_scales = {
        str(item.value).strip().lower().rstrip("s")
        for item in table.contexts
        if item.kind == "scale" and item.scope == "table"
    } & {"thousand", "million", "billion", "percent"}
    return next(iter(table_scales)) if len(table_scales) == 1 else None


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class ProvenanceIndex:
    def __init__(self) -> None:
        self._refs: dict[str, EvidenceRef] = {}

    def add(self, ref: EvidenceRef) -> EvidenceRef:
        self._refs[ref.ref_id] = ref
        return ref

    def get(self, ref_id: str) -> EvidenceRef | None:
        return self._refs.get(ref_id)

    def resolve_many(self, ref_ids: Iterable[str]) -> list[EvidenceRef]:
        return [ref for ref_id in ref_ids if (ref := self.get(str(ref_id))) is not None]

    def all_refs(self) -> list[EvidenceRef]:
        return list(self._refs.values())

    def lineage_closure(self, ref_ids: Iterable[str]) -> set[str]:
        out: set[str] = set()
        stack = [str(x) for x in ref_ids]
        while stack:
            ref_id = stack.pop()
            if ref_id in out:
                continue
            out.add(ref_id)
            ref = self.get(ref_id)
            if ref:
                stack.extend(ref.lineage_ref_ids)
        return out

    def has_new_lineage(self, refs: Iterable[EvidenceRef], included_ref_ids: Iterable[str]) -> bool:
        included_lineage = self.lineage_closure(included_ref_ids)
        for ref in refs:
            lineage = set(ref.lineage_ref_ids or [ref.ref_id])
            if lineage - included_lineage:
                return True
        return False


def cell_ref(table: CanonicalTable, row_idx: int, col_idx: int) -> EvidenceRef:
    row_id = f"r{row_idx}"
    col_id = f"c{col_idx}"
    value = table.rows[row_idx][col_idx] if row_idx < len(table.rows) and col_idx < len(table.headers) else ""
    source = table.source_cell(row_idx, col_idx)
    header_path = table.header_path(col_idx)
    semantic_scale = _cell_semantic_scale(table, row_idx, col_idx)
    payload = {
        "table": table.table_id,
        "row": row_id,
        "col": col_id,
        "header_path": header_path,
        "value": value,
        "source_cell": source.cell_id if source else None,
    }
    return EvidenceRef(
        ref_id=f"{table.table_id}:{row_id}:{col_id}",
        table_id=table.table_id,
        source_type="cell",
        row_ids=[row_id],
        col_ids=[col_id],
        header_path=header_path,
        unit="percent" if semantic_scale == "percent" else None,
        scale=semantic_scale,
        value_preview=str(value)[:120],
        source_cell_ids=[source.cell_id] if source else [],
        interpretation_status=table.cell_status(row_idx, col_idx),
        source_hash=stable_hash(payload),
    )


def row_ref(table: CanonicalTable, row_idx: int) -> EvidenceRef:
    row_id = f"r{row_idx}"
    payload = {"table": table.table_id, "row": row_id, "values": table.rows[row_idx]}
    return EvidenceRef(
        ref_id=f"{table.table_id}:{row_id}",
        table_id=table.table_id,
        source_type="row",
        row_ids=[row_id],
        col_ids=[f"c{i}" for i in range(len(table.headers))],
        header_path=["/".join(table.header_path(i)) for i in range(len(table.headers))],
        value_preview=" | ".join(table.rows[row_idx])[:160],
        source_cell_ids=[
            source.cell_id for col_idx in range(len(table.headers))
            if (source := table.source_cell(row_idx, col_idx)) is not None
        ],
        interpretation_status=(
            "ambiguous" if any(table.cell_status(row_idx, i) == "ambiguous" for i in range(len(table.headers)))
            else "missing" if any(table.cell_status(row_idx, i) == "missing" for i in range(len(table.headers)))
            else "reversible" if any(table.cell_status(row_idx, i) == "reversible" for i in range(len(table.headers)))
            else "clean"
        ),
        source_hash=stable_hash(payload),
    )


def column_ref(table: CanonicalTable, col_idx: int) -> EvidenceRef:
    col_id = f"c{col_idx}"
    values = [row[col_idx] for row in table.rows if col_idx < len(row)]
    payload = {"table": table.table_id, "col": col_id, "header": table.headers[col_idx], "values_hash": stable_hash(values)}
    return EvidenceRef(
        ref_id=f"{table.table_id}:{col_id}",
        table_id=table.table_id,
        source_type="column",
        row_ids=[f"r{i}" for i in range(len(table.rows))],
        col_ids=[col_id],
        header_path=table.header_path(col_idx),
        value_preview="/".join(table.header_path(col_idx)),
        source_cell_ids=[f"{table.table_id}:source:header:c{col_idx}"],
        source_hash=stable_hash(payload),
    )


def aggregate_ref(
    table: CanonicalTable,
    *,
    op: str,
    col_idx: int,
    value: Any,
    lineage_ref_ids: list[str],
    label: str | None = None,
) -> EvidenceRef:
    col_id = f"c{col_idx}"
    payload = {
        "table": table.table_id,
        "op": op,
        "col": col_id,
        "value": value,
        "lineage": sorted(lineage_ref_ids),
    }
    ref_id = f"{table.table_id}:agg:{op}:{col_id}:{stable_hash(payload)[:8]}"
    return EvidenceRef(
        ref_id=ref_id,
        table_id=table.table_id,
        source_type="aggregate",
        col_ids=[col_id],
        header_path=table.header_path(col_idx),
        value_preview=f"{label or op}({table.headers[col_idx]}) = {value}",
        lineage_ref_ids=lineage_ref_ids,
        source_hash=stable_hash(payload),
    )


def calculation_ref(
    table: CanonicalTable,
    *,
    op: str,
    col_indices: list[int],
    value: Any,
    lineage_ref_ids: list[str],
    label: str | None = None,
) -> EvidenceRef:
    """Create provenance for a calculation that uses one or more columns."""
    col_ids = [f"c{index}" for index in col_indices]
    headers = [table.headers[index] for index in col_indices]
    payload = {
        "table": table.table_id,
        "op": op,
        "columns": col_ids,
        "value": value,
        "lineage": sorted(lineage_ref_ids),
    }
    ref_id = f"{table.table_id}:calc:{op}:{'-'.join(col_ids)}:{stable_hash(payload)[:8]}"
    return EvidenceRef(
        ref_id=ref_id,
        table_id=table.table_id,
        source_type="derived",
        col_ids=col_ids,
        header_path=["/".join(table.header_path(index)) for index in col_indices],
        value_preview=f"{label or op}({', '.join(headers)}) = {value}",
        lineage_ref_ids=lineage_ref_ids,
        source_hash=stable_hash(payload),
    )
