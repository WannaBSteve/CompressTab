"""Provenance-preserving table artifacts for query-conditioned views."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


InterpretationKind = Literal[
    "text", "number", "percent", "currency", "date", "range", "missing", "unknown"
]
InterpretationStatus = Literal["clean", "reversible", "ambiguous", "missing"]


class TypedInterpretation(BaseModel):
    """A non-destructive interpretation of one raw cell value."""

    model_config = ConfigDict(extra="forbid")

    kind: InterpretationKind
    value: str
    status: InterpretationStatus = "clean"
    transform: str = "identity"
    unit: str | None = None
    scale: str | None = None
    confidence: float = 1.0
    reversible: bool = True


class SourceCell(BaseModel):
    """A source cell whose raw text remains immutable."""

    model_config = ConfigDict(extra="forbid")

    cell_id: str
    source_row: int
    source_col: int
    raw_value: str
    row_span: int = 1
    col_span: int = 1
    kind_candidates: list[str] = Field(default_factory=lambda: ["data"])
    interpretations: list[TypedInterpretation] = Field(default_factory=list)
    format_hint: str = ""


class ContextObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_id: str
    kind: Literal["caption", "section", "unit", "scale", "time", "footnote", "note"]
    value: str
    scope: Literal["table", "region", "row", "column", "cell"] = "table"
    source_cell_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class StructureCandidate(BaseModel):
    """An observed or inferred structural interpretation with visible support."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    kind: Literal[
        "header_row", "orientation", "inheritance", "region", "row_header", "row_role"
    ]
    selected: bool = False
    confidence: float
    signals: list[str]
    alternatives: list[str] = Field(default_factory=list)
    source_cell_ids: list[str] = Field(default_factory=list)


class CanonicalTable(BaseModel):
    """A compatible grid plus the structure needed to form query-local views."""

    model_config = ConfigDict(extra="forbid")

    table_id: str
    headers: list[str]
    rows: list[list[str]]
    caption: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    region_id: str = "region0"
    source_level: int = 2
    header_paths: list[list[str]] = Field(default_factory=list)
    row_header_paths: list[list[str]] = Field(default_factory=list)
    source_row_indices: list[int] = Field(default_factory=list)
    source_cells: list[SourceCell] = Field(default_factory=list)
    contexts: list[ContextObject] = Field(default_factory=list)
    structure_candidates: list[StructureCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def fill_compatibility_fields(self):
        if not self.header_paths:
            self.header_paths = [[header] if header else [] for header in self.headers]
        if not self.source_row_indices:
            self.source_row_indices = list(range(len(self.rows)))
        return self

    def header_path(self, col_idx: int) -> list[str]:
        if 0 <= col_idx < len(self.header_paths):
            return list(self.header_paths[col_idx])
        if 0 <= col_idx < len(self.headers):
            return [self.headers[col_idx]]
        return []

    def source_cell(self, row_idx: int, col_idx: int) -> SourceCell | None:
        source_row = self.source_row_indices[row_idx] if row_idx < len(self.source_row_indices) else row_idx
        for cell in self.source_cells:
            if cell.source_row == source_row and cell.source_col == col_idx:
                return cell
        return None

    def cell_status(self, row_idx: int, col_idx: int) -> str:
        cell = self.source_cell(row_idx, col_idx)
        if cell is None or not cell.interpretations:
            return "clean"
        statuses = {item.status for item in cell.interpretations}
        if "ambiguous" in statuses:
            return "ambiguous"
        if "missing" in statuses:
            return "missing"
        if "reversible" in statuses:
            return "reversible"
        return "clean"


class TableArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    tables: list[CanonicalTable]
    paragraphs: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def _typed_interpretations(raw: str) -> list[TypedInterpretation]:
    """Return conservative interpretations without changing the source text."""

    text = raw.strip()
    if not text:
        return [TypedInterpretation(kind="missing", value="", status="missing", transform="empty")]

    compact = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    accounting_negative = compact.startswith("(") and compact.endswith(")")
    cleaned = compact.strip("()").replace(",", "").strip()
    currency_match = re.fullmatch(r"([$\u00a3\u20ac\u00a5])\s*([-+]?\d+(?:\.\d+)?)", cleaned)
    percent_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*%", cleaned)
    number_match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned.lstrip("$\u00a3\u20ac\u00a5").strip())
    date_match = re.fullmatch(r"\d{4}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?", compact)
    range_match = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?)\s*-\s*([-+]?\d+(?:\.\d+)?)\s*", compact)

    if percent_match:
        value = percent_match.group(1)
        if accounting_negative and not value.startswith("-"):
            value = f"-{value}"
        return [
            TypedInterpretation(
                kind="percent", value=value, status="reversible", transform="parse_percent",
                unit="percent", confidence=1.0,
            )
        ]
    if currency_match:
        symbol, value = currency_match.groups()
        if accounting_negative and not value.startswith("-"):
            value = f"-{value}"
        return [
            TypedInterpretation(
                kind="currency", value=value, status="reversible", transform="parse_currency",
                unit={"$": "currency", "\u00a3": "GBP", "\u20ac": "EUR", "\u00a5": "JPY"}.get(symbol, "currency"),
            )
        ]
    if number_match:
        value = number_match.group(0)
        if accounting_negative and not value.startswith("-"):
            value = f"-{value}"
        status = "reversible" if value != text else "clean"
        return [TypedInterpretation(kind="number", value=value, status=status, transform="parse_number")]
    if date_match:
        return [TypedInterpretation(kind="date", value=compact, status="clean")]
    if range_match:
        return [
            TypedInterpretation(
                kind="range", value=f"{range_match.group(1)}..{range_match.group(2)}",
                status="ambiguous", transform="parse_range",
            )
        ]
    if text.lower() in {"n/a", "na", "null", "none", "-", "--", "not available"}:
        return [
            TypedInterpretation(
                kind="missing", value=text, status="missing", transform="missing_marker",
                confidence=0.95,
            )
        ]
    annotated_number = re.fullmatch(
        r"(?:[$\u00a3\u20ac\u00a5]\s*)?"
        r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
        r"\s*(\*+|[\u2020\u2021]+|\[[^\]]+\]|[a-zA-Z])",
        text,
    )
    if annotated_number:
        return [
            TypedInterpretation(
                kind="number",
                value=annotated_number.group(1).replace(",", ""),
                status="ambiguous",
                transform="strip_annotation_candidate",
                confidence=0.7,
            ),
            TypedInterpretation(
                kind="text",
                value=text,
                status="ambiguous",
                transform="identity",
                confidence=1.0,
            ),
        ]
    return [TypedInterpretation(kind="text", value=text, status="clean")]


def _adjacent_duplicate_runs(headers: list[str]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    while start < len(headers):
        end = start + 1
        while end < len(headers) and headers[end].strip().lower() == headers[start].strip().lower():
            end += 1
        if headers[start].strip() and end - start >= 2:
            runs.append((start, end))
        start = end
    return runs


def _looks_numeric(value: Any) -> bool:
    interpretations = _typed_interpretations(_stringify_cell(value))
    return any(item.kind in {"number", "percent", "currency"} for item in interpretations)


def _header_row_candidate(
    table_id: str,
    headers: list[str],
    rows: list[list[str]],
) -> tuple[StructureCandidate | None, list[list[str]]]:
    """Detect a second header level from structure, never benchmark vocabulary."""

    if not headers or not rows:
        return None, [[header] if header else [] for header in headers]
    first = rows[0]
    comparable = [
        (header.strip().lower(), first[index].strip().lower())
        for index, header in enumerate(headers)
        if index < len(first) and header.strip()
    ]
    repeated_matches = sum(left == right for left, right in comparable)
    if comparable and repeated_matches >= max(2, int(0.8 * len(comparable))):
        candidate = StructureCandidate(
            candidate_id=f"{table_id}:structure:repeated_header_row_0",
            kind="header_row",
            selected=True,
            confidence=0.99,
            signals=["matches_declared_headers", "text_dominant_repeated_header"],
            alternatives=["treat_as_data_row"],
            source_cell_ids=[f"{table_id}:source:r0:c{i}" for i in range(len(first))],
        )
        return candidate, [[header] if header else [] for header in headers]
    runs = _adjacent_duplicate_runs(headers)
    if not runs:
        return None, [[header] if header else [] for header in headers]

    non_empty = sum(bool(cell.strip()) for cell in first)
    numeric = sum(_looks_numeric(cell) for cell in first if cell.strip())
    repeated_width = sum(end - start for start, end in runs)
    distinct_in_runs = sum(
        len({first[index].strip().lower() for index in range(start, end) if first[index].strip()})
        for start, end in runs
    )
    signals = ["adjacent_duplicate_parent_headers"]
    score = 0.55
    if non_empty >= max(2, int(0.6 * len(headers))):
        signals.append("dense_first_row")
        score += 0.12
    if numeric <= max(1, non_empty // 4):
        signals.append("text_dominant_first_row")
        score += 0.14
    if repeated_width >= 2 and distinct_in_runs >= len(runs) + 1:
        signals.append("distinct_children_under_repeated_parents")
        score += 0.16
    score = min(0.99, score)
    # Repeated parents plus a dense row of distinct children is sufficient
    # even when child labels are numeric (years and periods are common).
    selected = score >= 0.82

    paths: list[list[str]] = []
    for top, sub in zip(headers, first):
        top = top.strip()
        sub = sub.strip()
        path = []
        if top:
            path.append(top)
        if sub and (not path or sub.lower() != path[-1].lower()):
            path.append(sub)
        paths.append(path)
    source_ids = [f"{table_id}:source:r0:c{i}" for i in range(len(first))]
    candidate = StructureCandidate(
        candidate_id=f"{table_id}:structure:header_row_0",
        kind="header_row",
        selected=selected,
        confidence=round(score, 4),
        signals=signals,
        alternatives=["declared_headers_only"],
        source_cell_ids=source_ids,
    )
    return candidate, paths


def table_from_rows(
    table_id: str,
    rows: Any,
    *,
    caption: str = "",
    metadata: dict[str, Any] | None = None,
    region_id: str = "region0",
) -> CanonicalTable:
    """Build a structural substrate from a declared grid or raw matrix."""

    source_level = 2
    if hasattr(rows, "headers") and hasattr(rows, "rows"):
        headers = [_stringify_cell(x) for x in rows.headers]
        body = [[_stringify_cell(x) for x in row] for row in rows.rows]
    elif hasattr(rows, "columns") and hasattr(rows, "values"):
        headers = [_stringify_cell(x) for x in rows.columns.tolist()]
        body = [[_stringify_cell(x) for x in row] for row in rows.values.tolist()]
    else:
        source_level = 3
        matrix = list(rows or [])
        headers = [_stringify_cell(x) for x in (matrix[0] if matrix else [])]
        body = [[_stringify_cell(x) for x in row] for row in (matrix[1:] if len(matrix) > 1 else [])]

    width = max([len(headers)] + [len(row) for row in body] + [0])
    if not headers:
        headers = [f"col_{i}" for i in range(width)]
    if len(headers) < width:
        headers.extend(f"col_{i}" for i in range(len(headers), width))
    normalized_rows = []
    for row in body:
        padded = row + [""] * max(0, width - len(row))
        normalized_rows.append(padded[:width])
    headers = headers[:width]

    source_cells: list[SourceCell] = []
    for col_idx, value in enumerate(headers):
        source_cells.append(
            SourceCell(
                cell_id=f"{table_id}:source:header:c{col_idx}",
                source_row=-1,
                source_col=col_idx,
                raw_value=value,
                kind_candidates=["header"],
                interpretations=_typed_interpretations(value),
            )
        )
    for row_idx, row in enumerate(normalized_rows):
        for col_idx, value in enumerate(row):
            source_cells.append(
                SourceCell(
                    cell_id=f"{table_id}:source:r{row_idx}:c{col_idx}",
                    source_row=row_idx,
                    source_col=col_idx,
                    raw_value=value,
                    interpretations=_typed_interpretations(value),
                )
            )

    candidate, candidate_paths = _header_row_candidate(table_id, headers, normalized_rows)
    structure_candidates = [candidate] if candidate is not None else []
    if candidate is not None and candidate.selected:
        header_paths = candidate_paths
        display_headers = ["/".join(path) or f"col_{idx}" for idx, path in enumerate(header_paths)]
        data_rows = normalized_rows[1:]
        source_row_indices = list(range(1, len(normalized_rows)))
        for cell in source_cells:
            if cell.source_row == 0:
                cell.kind_candidates = ["header"]
    else:
        header_paths = [[header] if header else [] for header in headers]
        display_headers = headers
        data_rows = normalized_rows
        source_row_indices = list(range(len(normalized_rows)))

    for item in (metadata or {}).get("row_semantic_signals", []) or []:
        if not isinstance(item, dict):
            continue
        match = re.fullmatch(r"r(\d+)", str(item.get("row", "")))
        if not match:
            continue
        row_idx = int(match.group(1))
        if not 0 <= row_idx < len(data_rows):
            continue
        source_row = source_row_indices[row_idx]
        signals = [
            str(value) for value in item.get("signals", []) or []
            if str(value).strip()
        ]
        if not signals:
            continue
        structure_candidates.append(
            StructureCandidate(
                candidate_id=f"{table_id}:structure:row_role:r{row_idx}",
                kind="row_role",
                selected=True,
                confidence=1.0,
                signals=signals,
                source_cell_ids=[f"{table_id}:source:r{source_row}:c0"],
            )
        )

    contexts = []
    if caption:
        contexts.append(
            ContextObject(
                context_id=f"{table_id}:context:caption",
                kind="caption",
                value=caption,
                scope="table",
            )
        )
    for kind in ("unit", "scale"):
        value = (metadata or {}).get(kind)
        if value is None:
            value = (metadata or {}).get(f"table_{kind}")
        if isinstance(value, (str, int, float)) and str(value).strip():
            contexts.append(
                ContextObject(
                    context_id=f"{table_id}:context:{kind}",
                    kind=kind,
                    value=str(value).strip(),
                    scope="table",
                )
            )
    for index, item in enumerate((metadata or {}).get("semantic_contexts", []) or []):
        if not isinstance(item, dict):
            continue
        value = str(item.get("value", "") or "").strip()
        kind = str(item.get("kind", "note") or "note").strip().lower()
        if not value or kind not in {"caption", "section", "note"}:
            continue
        if any(context.kind == kind and context.value == value for context in contexts):
            continue
        contexts.append(
            ContextObject(
                context_id=f"{table_id}:context:semantic:{index}",
                kind=kind,
                value=value,
                scope="table",
                source_cell_ids=[
                    str(cell_id) for cell_id in item.get("source_cell_ids", []) or []
                    if str(cell_id).strip()
                ],
            )
        )
    return CanonicalTable(
        table_id=table_id,
        headers=display_headers,
        rows=data_rows,
        caption=caption,
        metadata=metadata or {},
        region_id=region_id,
        source_level=source_level,
        header_paths=header_paths,
        source_row_indices=source_row_indices,
        source_cells=source_cells,
        contexts=contexts,
        structure_candidates=structure_candidates,
    )


def artifact_from_dataset_row(row: dict[str, Any]) -> TableArtifact:
    qid = str(row.get("question_id") or row.get("id") or row.get("idx") or "example")
    metadata = dict(row.get("metadata", {}) or {})
    caption = str(metadata.get("caption") or metadata.get("title") or "")
    table_inputs = row.get("tables")
    if not isinstance(table_inputs, list) or not table_inputs:
        table_inputs = [row.get("table")]
    tables: list[CanonicalTable] = []
    for index, item in enumerate(table_inputs):
        table_id = f"t{index}"
        region_id = f"region{index}"
        table_value = item
        table_caption = caption
        table_metadata = dict(metadata)
        if isinstance(item, dict) and any(key in item for key in ("table", "data", "rows")):
            table_id = str(item.get("table_id") or item.get("id") or table_id)
            region_id = str(item.get("region_id") or region_id)
            table_value = item.get("table", item.get("data", item.get("rows")))
            table_caption = str(item.get("caption") or item.get("title") or caption or "")
            table_metadata.update(dict(item.get("metadata", {}) or {}))
        canonical = table_from_rows(
            table_id,
            table_value,
            caption=table_caption,
            metadata=table_metadata,
            region_id=region_id,
        )
        if isinstance(item, dict):
            header_paths = item.get("header_paths")
            if isinstance(header_paths, list) and len(header_paths) == len(canonical.headers):
                canonical.header_paths = [
                    [str(part) for part in path if str(part).strip()]
                    for path in header_paths
                ]
                canonical.headers = [
                    " / ".join(path) or f"col_{index}"
                    for index, path in enumerate(canonical.header_paths)
                ]
            row_header_paths = item.get("row_header_paths")
            if isinstance(row_header_paths, list) and len(row_header_paths) == len(canonical.rows):
                canonical.row_header_paths = [
                    [str(part) for part in path if str(part).strip()]
                    for path in row_header_paths
                ]
        tables.append(canonical)
    paragraphs = []
    for idx, paragraph in enumerate(row.get("paragraphs", []) or []):
        obj = dict(paragraph) if isinstance(paragraph, dict) else {"text": str(paragraph)}
        obj.setdefault("paragraph_id", f"p{idx}")
        paragraphs.append(obj)
    return TableArtifact(artifact_id=qid, tables=tables, paragraphs=paragraphs, metadata=metadata)
