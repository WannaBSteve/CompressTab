"""Answerability signatures, deterministic compilation, and table lenses."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from compresstab.table.artifact import TableArtifact
from compresstab.table.store import TableStore, approx_token_count, parse_number


ResolutionMode = Literal["exact", "identity", "equality", "order", "threshold"]
DerivedOperator = Literal[
    "difference", "ratio", "percent_change",
    "sum", "mean", "min", "max", "range", "magnitude",
]
DerivedMode = Literal["exact", "order", "threshold"]
ThresholdOperator = Literal["lt", "le", "gt", "ge"]
OrderKind = Literal["value", "source"]
PredicateLogic = Literal["all", "any"]
PredicateOperator = Literal[
    "eq", "ne", "lt", "le", "gt", "ge", "contains", "starts_with",
    "ends_with", "in", "between", "under",
]


class EvidencePredicate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    col: str
    op: PredicateOperator = "eq"
    value: Any = None


class EvidenceResolution(BaseModel):
    """One observable distinction that an evidence representation must preserve."""

    model_config = ConfigDict(extra="forbid")

    column: str
    mode: ResolutionMode = "identity"
    boundary: float | None = None
    operator: ThresholdOperator | None = None
    anchors: list[str] = Field(default_factory=list)
    order_kind: OrderKind | None = None


class DerivedOperand(BaseModel):
    """One source or expression node used by a row-bound expression."""

    model_config = ConfigDict(extra="forbid")

    column: str
    role: Literal[
        "value", "input", "left", "right", "numerator", "denominator",
        "earlier", "later", "group", "rank",
    ]
    value_mode: Literal["signed", "magnitude"] = "signed"


class DerivedObservable(BaseModel):
    """A deterministic virtual field whose required distinction can be coded."""

    model_config = ConfigDict(extra="forbid")

    observable_id: str
    operator: DerivedOperator
    operands: list[DerivedOperand]
    mode: DerivedMode = "order"
    direction: Literal["ascending", "descending"] = "descending"
    boundary: float | None = None
    threshold_operator: ThresholdOperator | None = None
    unit: str = "number"
    scale: float = 1.0
    required: bool = True


def topological_observables(
    observables: list[DerivedObservable],
    source_node_ids: set[str],
) -> list[DerivedObservable]:
    """Validate and order a derived-expression DAG from dependencies to users."""

    by_id = {item.observable_id: item for item in observables}
    if len(by_id) != len(observables):
        raise ValueError("Derived observable ids must be unique within one table plan")

    unknown = sorted({
        operand.column
        for observable in observables
        for operand in observable.operands
        if operand.column not in source_node_ids and operand.column not in by_id
    })
    if unknown:
        raise ValueError(
            "Derived expressions reference unknown evidence nodes: "
            + ", ".join(unknown)
        )

    state: dict[str, int] = {}
    ordered: list[DerivedObservable] = []
    path: list[str] = []

    def visit(node_id: str) -> None:
        status = state.get(node_id, 0)
        if status == 2:
            return
        if status == 1:
            start = path.index(node_id) if node_id in path else 0
            cycle = [*path[start:], node_id]
            raise ValueError(
                "Derived-expression cycle: " + " -> ".join(cycle)
            )
        state[node_id] = 1
        path.append(node_id)
        for operand in by_id[node_id].operands:
            if operand.column in by_id:
                visit(operand.column)
        path.pop()
        state[node_id] = 2
        ordered.append(by_id[node_id])

    for observable in observables:
        visit(observable.observable_id)
    return ordered


class EvidenceSlice(BaseModel):
    """A source scope, its fact links, and required value resolutions."""

    model_config = ConfigDict(extra="forbid")

    source: str = "t0"
    columns: list[str] = Field(default_factory=list)
    where: list[EvidencePredicate] = Field(default_factory=list)
    where_logic: PredicateLogic = "all"
    rows: list[str] = Field(default_factory=list)
    links: list[list[str]] = Field(default_factory=list)
    resolution: list[EvidenceResolution] = Field(default_factory=list)
    derived: list[DerivedObservable] = Field(default_factory=list)


class AnswerabilitySignature(BaseModel):
    """The complete Scope + Links + Resolution artifact for one question."""

    model_config = ConfigDict(extra="forbid")

    slices: list[EvidenceSlice] = Field(default_factory=list)


def matches_predicate(raw: Any, op: str, target: Any) -> bool:
    """Evaluate one generic predicate without inferring query semantics."""

    op = str(op or "eq").lower()
    left_num = parse_number(raw)
    if op == "contains":
        # Word separators in semi-structured sources are often typographic
        # variants (hyphen, slash, Unicode dash) rather than semantic content.
        normalize = lambda value: " ".join(
            re.findall(r"\w+", str(value).casefold(), flags=re.UNICODE)
        )
        return normalize(target) in normalize(raw)
    if op == "starts_with":
        return str(raw).strip().lower().startswith(str(target).strip().lower())
    if op == "ends_with":
        return str(raw).strip().lower().endswith(str(target).strip().lower())
    if op == "in":
        values = target if isinstance(target, (list, tuple, set)) else [target]
        return any(matches_predicate(raw, "eq", value) for value in values)
    if op == "between":
        values = target if isinstance(target, (list, tuple)) else []
        if len(values) != 2 or left_num is None:
            return False
        low, high = parse_number(values[0]), parse_number(values[1])
        return low is not None and high is not None and min(low, high) <= left_num <= max(low, high)
    if op in {"eq", "ne"}:
        right_num = parse_number(target)
        equal = (
            abs(left_num - right_num) <= 1e-9
            if left_num is not None and right_num is not None
            else str(raw).strip().lower() == str(target).strip().lower()
        )
        return equal if op == "eq" else not equal
    if op not in {"lt", "le", "gt", "ge"}:
        return False

    right_num = parse_number(target)
    # Mixed numeric/non-numeric values are incomparable.  Falling back to
    # lexical order here would, for example, make "NA" satisfy age >= 44.71.
    if (left_num is None) != (right_num is None):
        return False
    if left_num is None:
        left, right = str(raw).strip().lower(), str(target).strip().lower()
    else:
        left, right = left_num, right_num
    return {
        "lt": left < right,
        "le": left <= right,
        "gt": left > right,
        "ge": left >= right,
    }[op]


def matches_table_predicate(
    table: Any,
    row_id: str,
    predicate: EvidencePredicate,
) -> bool:
    """Evaluate a predicate whose semantics may depend on table structure."""

    if not (row_id.startswith("r") and row_id[1:].isdigit()):
        return False
    row_idx = int(row_id[1:])
    col_idx = (
        int(predicate.col[1:])
        if predicate.col.startswith("c") and predicate.col[1:].isdigit()
        else -1
    )
    if not (0 <= row_idx < len(table.rows) and 0 <= col_idx < len(table.headers)):
        return False
    if predicate.op != "under":
        return matches_predicate(
            table.rows[row_idx][col_idx], predicate.op, predicate.value
        )

    # `under` is an axis relation, not a string predicate over the displayed
    # cell. Strict ancestors exclude the named parent while retaining all
    # deeper descendants in their original source order.
    if len(table.row_header_paths) != len(table.rows):
        return False
    normalize = lambda value: " ".join(
        re.findall(r"\w+", str(value).casefold(), flags=re.UNICODE)
    )
    target = normalize(predicate.value)
    ancestors = [normalize(value) for value in table.row_header_paths[row_idx][:-1]]
    return bool(target) and target in ancestors


class EvidenceTablePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str
    row_ids: list[str]
    col_ids: list[str]
    context_ids: list[str] = Field(default_factory=list)
    layouts: list[dict[str, Any]] = Field(default_factory=list)
    dependencies: list[dict[str, Any]] = Field(default_factory=list)
    derived_observables: list[DerivedObservable] = Field(default_factory=list)


class EvidenceCompilation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signature: AnswerabilitySignature
    tables: list[EvidenceTablePlan]
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def has_blocking_diagnostics(self) -> bool:
        return any(bool(item.get("blocking", True)) for item in self.diagnostics)


def merge_signatures(
    base: AnswerabilitySignature,
    residual: AnswerabilitySignature,
) -> AnswerabilitySignature:
    """Return a monotone, de-duplicated union without changing existing slices."""

    slices: list[EvidenceSlice] = []
    seen: set[str] = set()
    for evidence_slice in [*base.slices, *residual.slices]:
        key = evidence_slice.model_dump_json(exclude_none=True)
        if key not in seen:
            seen.add(key)
            slices.append(evidence_slice)
    return AnswerabilitySignature(slices=slices)


class EvidenceCompiler:
    """Compile Scope + Links + Resolution into codec obligations."""

    def __init__(self, artifact: TableArtifact):
        self.artifact = artifact
        self._tables = {table.table_id: table for table in artifact.tables}

    def compile(
        self,
        signature: AnswerabilitySignature | dict[str, Any],
    ) -> EvidenceCompilation:
        parsed = (
            signature
            if isinstance(signature, AnswerabilitySignature)
            else AnswerabilitySignature.model_validate(signature)
        )
        diagnostics: list[dict[str, Any]] = []
        grouped: dict[str, list[EvidenceSlice]] = defaultdict(list)
        for evidence_slice in parsed.slices:
            if evidence_slice.source not in self._tables:
                diagnostics.append(self._diagnostic(
                    "UNKNOWN_SOURCE",
                    f"Unknown source {evidence_slice.source}.",
                    True,
                ))
                continue
            grouped[evidence_slice.source].append(evidence_slice)

        if not grouped and self.artifact.tables:
            # A malformed Agent response degrades visibly to a lossless full-table
            # request rather than inventing query semantics or returning no evidence.
            table = self.artifact.tables[0]
            fallback = EvidenceSlice(
                source=table.table_id,
                columns=[f"c{i}" for i in range(len(table.headers))],
                links=[[f"c{i}" for i in range(len(table.headers))]],
                resolution=[
                    EvidenceResolution(column=f"c{i}", mode="exact")
                    for i in range(len(table.headers))
                ],
            )
            parsed = AnswerabilitySignature(slices=[fallback])
            grouped[table.table_id].append(fallback)
            diagnostics.append(self._diagnostic(
                "EMPTY_SIGNATURE",
                "No valid slice was provided; retained the full first source exactly.",
                False,
            ))

        plans = [
            self._compile_table(table_id, needs, diagnostics)
            for table_id, needs in grouped.items()
        ]
        return EvidenceCompilation(signature=parsed, tables=plans, diagnostics=diagnostics)

    def _compile_table(
        self,
        table_id: str,
        slices: list[EvidenceSlice],
        diagnostics: list[dict[str, Any]],
    ) -> EvidenceTablePlan:
        table = self._tables[table_id]
        all_rows = [f"r{i}" for i in range(len(table.rows))]
        all_cols = [f"c{i}" for i in range(len(table.headers))]
        selected_rows: list[str] = []
        selected_cols: list[str] = []
        dependencies: list[dict[str, Any]] = []
        derived_observables: list[DerivedObservable] = []
        derived_scopes: dict[str, set[str]] = defaultdict(set)
        context_ids: list[str] = []
        layouts: list[dict[str, Any]] = []

        def add_dependency(
            kind: str,
            col_ids: list[str],
            row_ids: list[str],
            reason: str,
            **properties: Any,
        ) -> None:
            candidate = {
                "kind": kind,
                "col_ids": list(dict.fromkeys(col_ids)),
                "row_ids": list(dict.fromkeys(row_ids)),
                "mandatory": True,
                "reason": reason,
                **properties,
            }
            signature = (
                kind,
                tuple(candidate["col_ids"]),
                tuple(candidate["row_ids"]),
                tuple(sorted((key, str(value)) for key, value in properties.items())),
            )
            if not any(
                (
                    item["kind"],
                    tuple(item["col_ids"]),
                    tuple(item["row_ids"]),
                    tuple(sorted(
                        (key, str(value))
                        for key, value in item.items()
                        if key not in {"kind", "col_ids", "row_ids", "mandatory", "reason"}
                    )),
                ) == signature
                for item in dependencies
            ):
                dependencies.append(candidate)

        for evidence_slice in slices:
            cols = (
                list(all_cols)
                if "*" in evidence_slice.columns
                else [col_id for col_id in evidence_slice.columns if col_id in all_cols]
            )
            linked_cols = [
                col_id
                for link in evidence_slice.links
                for col_id in (all_cols if "*" in link else link)
                if col_id in all_cols
            ]
            predicate_cols = [
                item.col for item in evidence_slice.where if item.col in all_cols
            ]
            for col_id in [*cols, *linked_cols, *predicate_cols]:
                if col_id not in selected_cols:
                    selected_cols.append(col_id)

            invalid_cols = [
                col_id
                for col_id in [
                    *evidence_slice.columns,
                    *(col_id for link in evidence_slice.links for col_id in link),
                ]
                if col_id != "*" and col_id not in all_cols
            ]
            if invalid_cols:
                diagnostics.append(self._diagnostic(
                    "INVALID_COLUMN",
                    f"Evidence slice referenced invalid columns: {', '.join(invalid_cols)}.",
                    True,
                ))

            explicit_rows = [row_id for row_id in evidence_slice.rows if row_id in all_rows]
            if explicit_rows:
                rows = explicit_rows
            elif evidence_slice.where:
                combine = any if evidence_slice.where_logic == "any" else all
                rows = [
                    row_id for row_id in all_rows
                    if combine(
                        self._matches(table, row_id, predicate)
                        for predicate in evidence_slice.where
                    )
                ]
            else:
                rows = all_rows
            for row_id in rows:
                if row_id not in selected_rows:
                    selected_rows.append(row_id)

            layout = {
                "axis": "rows",
                "grain": "record",
                "rows": rows,
                "columns": cols,
                "where": [
                    predicate.model_dump(mode="json")
                    for predicate in evidence_slice.where
                ],
                "where_logic": evidence_slice.where_logic,
                "links": [
                    (
                        list(all_cols)
                        if "*" in link
                        else [col_id for col_id in link if col_id in all_cols]
                    )
                    for link in evidence_slice.links
                    if "*" in link or len([col_id for col_id in link if col_id in all_cols]) >= 2
                ],
                "align_by": "row",
            }
            if layout not in layouts:
                layouts.append(layout)

            if not cols:
                diagnostics.append(self._diagnostic(
                    "EMPTY_COLUMNS",
                    "Evidence slice selected no valid evidence column.",
                    True,
                ))
                continue

            derived_operand_cols = {
                operand.column
                for observable in evidence_slice.derived
                for operand in observable.operands
                if operand.column in all_cols
            }
            for observable in evidence_slice.derived:
                invalid_operands = [
                    operand.column
                    for operand in observable.operands
                    if operand.column not in all_cols
                    and not re.fullmatch(r"v\d+", operand.column)
                ]
                if invalid_operands:
                    diagnostics.append(self._diagnostic(
                        "INVALID_DERIVED_OPERAND",
                        (
                            f"Derived observable {observable.observable_id} references "
                            f"invalid columns: {', '.join(invalid_operands)}."
                        ),
                        True,
                    ))
                    continue
                if observable.observable_id in all_cols:
                    diagnostics.append(self._diagnostic(
                        "DERIVED_ID_COLLISION",
                        (
                            f"Derived observable {observable.observable_id} collides "
                            "with a source column id."
                        ),
                        True,
                    ))
                    continue
                if any(
                    existing.observable_id == observable.observable_id
                    and existing != observable
                    for existing in derived_observables
                ):
                    diagnostics.append(self._diagnostic(
                        "CONFLICTING_DERIVED_OBSERVABLE",
                        f"Derived observable {observable.observable_id} has conflicting definitions.",
                        True,
                    ))
                    continue
                if observable not in derived_observables:
                    derived_observables.append(observable)
                derived_scopes[observable.observable_id].update(rows)
                for operand in observable.operands:
                    if operand.column in all_cols and operand.column not in selected_cols:
                        selected_cols.append(operand.column)

            add_dependency(
                "complete_domain",
                cols,
                rows,
                "Every fact in the selected scope must remain visible",
            )

            resolutions_by_col: dict[str, list[EvidenceResolution]] = defaultdict(list)
            for item in evidence_slice.resolution:
                if item.column in cols:
                    resolutions_by_col[item.column].append(item)
            for col_id in cols:
                resolutions = resolutions_by_col.get(col_id, [])
                if not resolutions and col_id not in derived_operand_cols:
                    col_idx = int(col_id[1:])
                    values = [table.rows[int(row_id[1:])][col_idx] for row_id in rows]
                    nonempty = [value for value in values if str(value).strip()]
                    numeric = nonempty and all(parse_number(value) is not None for value in nonempty)
                    resolutions = [EvidenceResolution(
                        column=col_id,
                        mode="exact" if numeric else "identity",
                    )]
                for resolution in resolutions:
                    if resolution.mode == "exact":
                        add_dependency("exact_value", [col_id], rows, "Exact source values must survive")
                        add_dependency(
                            "direct_value",
                            [col_id],
                            rows,
                            "Answer-bearing source values must remain directly visible",
                        )
                    elif resolution.mode == "identity":
                        add_dependency(
                            "identity_recoverable",
                            [col_id],
                            rows,
                            "Original labels must be recoverable",
                        )
                        add_dependency(
                            "direct_value",
                            [col_id],
                            rows,
                            "Entity labels must remain directly addressable in their source tuples",
                        )
                    elif resolution.mode == "equality":
                        source_values = [
                            str(table.rows[int(row_id[1:])][int(col_id[1:])])
                            for row_id in rows
                        ]
                        canonical_by_key: dict[str, str] = {}
                        for value in source_values:
                            canonical_by_key.setdefault(value.strip().casefold(), value)
                        requested_anchors = list(dict.fromkeys(
                            str(value).strip()
                            for value in resolution.anchors
                            if str(value).strip()
                        ))
                        resolved_anchors = [
                            canonical_by_key[value.casefold()]
                            for value in requested_anchors
                            if value.casefold() in canonical_by_key
                        ]
                        if requested_anchors and len(resolved_anchors) == len(requested_anchors):
                            add_dependency(
                                "anchor_membership",
                                [col_id],
                                rows,
                                "Membership in each named equality class must survive",
                                anchors=resolved_anchors,
                            )
                        else:
                            add_dependency(
                                "equality",
                                [col_id],
                                rows,
                                "Equality classes must survive",
                            )
                            add_dependency(
                                "identity_recoverable",
                                [col_id],
                                rows,
                                "Ungrounded equality retains recoverable source labels",
                            )
                            diagnostics.append(self._diagnostic(
                                "UNGROUNDED_EQUALITY",
                                (
                                    f"Equality resolution for {col_id} has no fully resolved "
                                    "source anchor; retained recoverable labels."
                                ),
                                False,
                            ))
                    elif resolution.mode == "order":
                        if resolution.order_kind == "source":
                            add_dependency(
                                "source_order",
                                [col_id],
                                rows,
                                "Source-row sequence must survive",
                                order_kind="source",
                            )
                        else:
                            add_dependency(
                                "order",
                                [col_id],
                                rows,
                                "Relative numeric value order must survive",
                                order_kind="value",
                            )
                    elif resolution.mode == "threshold":
                        if resolution.boundary is None:
                            diagnostics.append(self._diagnostic(
                                "MISSING_THRESHOLD",
                                f"Threshold resolution for {col_id} has no boundary; exact values retained.",
                                False,
                            ))
                            add_dependency("exact_value", [col_id], rows, "Missing threshold falls back to exact values")
                        else:
                            predicate_operators = {
                                item.op for item in evidence_slice.where
                                if item.col == col_id and item.op in {"lt", "le", "gt", "ge"}
                            }
                            operator = resolution.operator
                            if operator is None and len(predicate_operators) == 1:
                                operator = next(iter(predicate_operators))
                            add_dependency(
                                "threshold_membership",
                                [col_id],
                                rows,
                                "Each value must preserve the requested threshold relation",
                                boundary=float(resolution.boundary),
                                operator=operator or "ge",
                            )

            for link in evidence_slice.links:
                expanded_link = cols if "*" in link else link
                aligned = list(dict.fromkeys(col_id for col_id in expanded_link if col_id in cols))
                if len(aligned) < 2:
                    continue
                add_dependency(
                    "alignment",
                    aligned,
                    rows,
                    "Linked values must remain one row-level fact",
                    axis="rows",
                    grain="record",
                    align_by="row",
                )

            if any(len(table.header_path(int(col_id[1:]))) > 1 for col_id in cols):
                add_dependency(
                    "header_hierarchy",
                    cols,
                    rows,
                    "Selected fields depend on a multi-level header path",
                )

            context_ids.extend(
                item.context_id for item in table.contexts
                if item.context_id not in context_ids
            )
            if table.contexts:
                add_dependency("context", [], [], "Table-scoped context remains visible")

        if not selected_cols:
            selected_cols = all_cols
        # An empty predicate result is a valid complete domain (and can prove
        # absence). Only an unfiltered empty row declaration means all rows.
        if not selected_rows and not any(item.where for item in slices):
            selected_rows = all_rows
        try:
            derived_observables = topological_observables(
                derived_observables, set(all_cols)
            )
        except ValueError as exc:
            diagnostics.append(self._diagnostic(
                "INVALID_DERIVED_GRAPH",
                str(exc),
                True,
            ))

        for observable in derived_observables:
            if derived_scopes[observable.observable_id] != set(selected_rows):
                diagnostics.append(self._diagnostic(
                    "DERIVED_SCOPE_CONFLICT",
                    (
                        f"Derived observable {observable.observable_id} does not cover "
                        "the complete physical row scope for its table."
                    ),
                    True,
                ))
        add_dependency("provenance", selected_cols, selected_rows, "Every visible fact must remain source-addressable")

        for col_id in selected_cols:
            col_idx = int(col_id[1:])
            dirty_rows = [
                row_id for row_id in selected_rows
                if table.cell_status(int(row_id[1:]), col_idx) in {"ambiguous", "missing", "reversible"}
            ]
            if dirty_rows:
                add_dependency("dirty_raw", [col_id], dirty_rows, "Dirty values retain immutable raw text")

        return EvidenceTablePlan(
            table_id=table_id,
            row_ids=selected_rows,
            col_ids=selected_cols,
            context_ids=context_ids,
            layouts=layouts,
            dependencies=dependencies,
            derived_observables=derived_observables,
        )

    def _matches(self, table: Any, row_id: str, predicate: EvidencePredicate) -> bool:
        return matches_table_predicate(table, row_id, predicate)

    def _diagnostic(self, code: str, message: str, blocking: bool) -> dict[str, Any]:
        return {"code": code, "message": message, "candidate_ids": [], "blocking": blocking}


class TableLens:
    """A bounded, generic interface for inspecting evidence omitted by a witness."""

    def __init__(self, store: TableStore):
        self.store = store

    def overview(self, compilation: EvidenceCompilation) -> dict[str, Any]:
        plans = {plan.table_id: plan for plan in compilation.tables}
        sources = []
        for table in self.store.artifact.tables:
            plan = plans.get(table.table_id)
            sources.append({
                "id": table.table_id,
                "shape": [len(table.rows), len(table.headers)],
                "headers": [
                    {"id": f"c{i}", "path": table.header_path(i)}
                    for i in range(len(table.headers))
                ],
                "visible_rows": len(plan.row_ids) if plan else 0,
                "visible_columns": list(plan.col_ids) if plan else [],
                "omitted_rows": max(0, len(table.rows) - (len(plan.row_ids) if plan else 0)),
                "contexts": [
                    {"id": item.context_id, "kind": item.kind, "value": item.value, "scope": item.scope}
                    for item in table.contexts
                ],
            })
        return {"sources": sources}

    def execute(
        self,
        actions: list[dict[str, Any]],
        *,
        visible_ref_ids: set[str],
        token_budget: int,
    ) -> dict[str, Any]:
        observations: list[dict[str, Any]] = []
        valid_refs: set[str] = set()
        for action in actions[:3]:
            tool = action.get("tool")
            if tool == "search":
                observation = self._search(action, visible_ref_ids, valid_refs)
            elif tool == "slice":
                observation = self._slice(action, visible_ref_ids, valid_refs)
            elif tool == "filter":
                observation = self._filter(action, visible_ref_ids, valid_refs)
            elif tool == "profile":
                observation = self._profile(action, visible_ref_ids, valid_refs)
            else:
                continue
            observations.append(observation)

        result = {"observations": observations, "valid_ref_ids": sorted(valid_refs)}
        budget = max(256, int(token_budget))
        while approx_token_count(result) > budget and observations:
            last = observations[-1]
            collection = last.get("matches") or last.get("rows") or last.get("profiles")
            if isinstance(collection, list) and len(collection) > 1:
                collection.pop()
            else:
                observations.pop()
            valid_refs = set(self._collect_refs(observations))
            result = {"observations": observations, "valid_ref_ids": sorted(valid_refs)}
        return result

    def _search(self, action: dict[str, Any], visible: set[str], valid: set[str]) -> dict[str, Any]:
        table_id = str(action.get("source", "t0"))
        table = self.store.table(table_id)
        columns = self._valid_cols(action.get("columns", []), len(table.headers))
        if not columns:
            columns = [f"c{i}" for i in range(min(8, len(table.headers)))]
        limit = max(1, min(20, int(action.get("limit", 8) or 8)))
        matches = []
        seen_rows: set[str] = set()
        for hit in self.store.search(str(action.get("query", "")), top_k=limit * 2, table_id=table_id):
            for row_id in hit.ref.row_ids:
                if row_id in seen_rows:
                    continue
                item = self._row_item(table_id, row_id, columns, visible, valid)
                if item:
                    seen_rows.add(row_id)
                    item["score"] = round(hit.score, 4)
                    matches.append(item)
            if len(matches) >= limit:
                break
        return {"tool": "search", "query": action.get("query", ""), "matches": matches}

    def _slice(self, action: dict[str, Any], visible: set[str], valid: set[str]) -> dict[str, Any]:
        table_id = str(action.get("source", "t0"))
        table = self.store.table(table_id)
        rows = self._valid_rows(action.get("rows", []), len(table.rows))[:20]
        columns = self._valid_cols(action.get("columns", []), len(table.headers))
        if not columns:
            columns = [f"c{i}" for i in range(len(table.headers))]
        items = [self._row_item(table_id, row_id, columns, visible, valid) for row_id in rows]
        return {"tool": "slice", "rows": [item for item in items if item]}

    def _filter(self, action: dict[str, Any], visible: set[str], valid: set[str]) -> dict[str, Any]:
        """Evaluate an Agent-proposed predicate over the complete source table."""

        table_id = str(action.get("source", "t0"))
        table = self.store.table(table_id)
        predicates = []
        for raw in action.get("where", []) or []:
            try:
                predicate = EvidencePredicate.model_validate(raw)
            except Exception:
                continue
            if predicate.col in self._valid_cols([predicate.col], len(table.headers)):
                predicates.append(predicate)
        columns = self._valid_cols(action.get("columns", []), len(table.headers))
        if not columns:
            columns = [f"c{i}" for i in range(len(table.headers))]
        limit = max(1, min(20, int(action.get("limit", 20) or 20)))
        matching_rows = [
            f"r{row_index}"
            for row_index in range(len(table.rows))
            if predicates and all(
                matches_table_predicate(table, f"r{row_index}", predicate)
                for predicate in predicates
            )
        ]
        rows = [
            self._row_item(table_id, row_id, columns, visible, valid)
            for row_id in matching_rows[:limit]
        ]
        return {
            "tool": "filter",
            "source": table_id,
            "complete": True,
            "scanned_rows": len(table.rows),
            "where": [item.model_dump(mode="json") for item in predicates],
            "match_count": len(matching_rows),
            "matches_truncated": len(matching_rows) > limit,
            "rows": [item for item in rows if item],
        }

    def _profile(self, action: dict[str, Any], visible: set[str], valid: set[str]) -> dict[str, Any]:
        table_id = str(action.get("source", "t0"))
        table = self.store.table(table_id)
        columns = self._valid_cols(action.get("columns", []), len(table.headers))[:8]
        profiles = []
        for col_id in columns:
            col_idx = int(col_id[1:])
            values = [row[col_idx] for row in table.rows]
            numeric = [(index, parse_number(value)) for index, value in enumerate(values)]
            numeric = [(index, value) for index, value in numeric if value is not None]
            profile: dict[str, Any] = {
                "source": table_id,
                "column": col_id,
                "header": table.header_path(col_idx),
                "count": len(values),
                "distinct": len(set(str(value) for value in values)),
                "missing": sum(not str(value).strip() for value in values),
            }
            if numeric:
                for name, pair in (("min", min(numeric, key=lambda item: item[1])), ("max", max(numeric, key=lambda item: item[1]))):
                    row_idx, number = pair
                    ref_id = f"{table_id}:r{row_idx}:{col_id}"
                    if ref_id not in visible:
                        profile[name] = {"value": number, "row": f"r{row_idx}", "ref": ref_id}
                        valid.add(ref_id)
            profiles.append(profile)
        return {"tool": "profile", "profiles": profiles}

    def _row_item(
        self,
        table_id: str,
        row_id: str,
        columns: list[str],
        visible: set[str],
        valid: set[str],
    ) -> dict[str, Any] | None:
        table = self.store.table(table_id)
        row_idx = int(row_id[1:])
        values: dict[str, str] = {}
        refs: list[str] = []
        for col_id in columns:
            col_idx = int(col_id[1:])
            ref_id = f"{table_id}:{row_id}:{col_id}"
            if ref_id in visible:
                continue
            values[col_id] = table.rows[row_idx][col_idx]
            refs.append(ref_id)
            valid.add(ref_id)
        if not refs:
            return None
        return {"source": table_id, "row": row_id, "values": values, "refs": refs}

    def _collect_refs(self, value: Any) -> list[str]:
        refs: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "ref" and isinstance(item, str):
                    refs.append(item)
                elif key == "refs" and isinstance(item, list):
                    refs.extend(str(ref) for ref in item)
                else:
                    refs.extend(self._collect_refs(item))
        elif isinstance(value, list):
            for item in value:
                refs.extend(self._collect_refs(item))
        return refs

    def _valid_rows(self, values: Any, length: int) -> list[str]:
        return self._valid_ids(values, "r", length)

    def _valid_cols(self, values: Any, length: int) -> list[str]:
        return self._valid_ids(values, "c", length)

    def _valid_ids(self, values: Any, prefix: str, length: int) -> list[str]:
        result = []
        for raw in values or []:
            text = str(raw)
            if not text.startswith(prefix):
                text = f"{prefix}{text}"
            if text[len(prefix):].isdigit() and int(text[len(prefix):]) < length and text not in result:
                result.append(text)
        return result
