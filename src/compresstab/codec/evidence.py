"""Distinction-preserving evidence coding for CompressTab."""

from __future__ import annotations

import re
import math
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from itertools import combinations
from typing import Any, Callable, Iterable

from compresstab.evidence import DerivedObservable, topological_observables
from compresstab.profile import ColumnProfile, ProfileStore
from compresstab.table.store import approx_token_count, format_number, parse_number, tokens


STRUCTURAL_REQUIREMENTS = {
    "complete_domain",
    "provenance",
    "alignment",
    "header_hierarchy",
    "context",
    "source_order",
}
SECTION_HEADINGS = {
    "codes": "Coded columns (rows follow grid order):",
    "decoder": "Encoding guide:",
    "patches": "Exact patches:",
}
# Readable line and cell boundaries; costs are measured on rendered text.
CANONICAL_LINE_SEPARATOR = "\t\n"
CANONICAL_CELL_SEPARATOR = " | "
GRID_SECTION = "__grid__"
CANONICAL_DECIMAL = re.compile(r"^[+-]?(?:0|[1-9]\d*)(?:\.\d+)?$")
SEMANTIC_DECIMAL = re.compile(
    r"(?:[$\u00a3\u20ac\u00a5]\s*)?"
    r"(?P<number>[-+]?(?:\d{1,3}(?:[,\s\u00a0\u202f]\d{3})+|\d+)"
    r"(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*%?"
)


@dataclass
class EncodingCandidate:
    """One certified Reader-facing realization of an evidence-graph node."""

    col_id: str
    source_header: str
    display_header: str
    operator: str
    values: dict[str, str]
    token_cost: int
    lexical_cost: int
    byte_cost: int
    append_token_cost: int = 0
    grid_visible: bool = True
    shared_sections: frozenset[str] = frozenset()
    code_lines: list[str] = field(default_factory=list)
    decoder_lines: list[str] = field(default_factory=list)
    patch_lines: list[str] = field(default_factory=list)
    residual_cells: list[dict[str, Any]] = field(default_factory=list)
    covered_requirements: list[str] = field(default_factory=list)
    certification_checks: dict[str, bool] = field(default_factory=dict)
    certified: bool = False
    source_reversible: bool = False
    raw_equivalent_tokens: int = 0
    source_col_ids: tuple[str, ...] = ()
    render_headers: tuple[str, ...] = ()
    render_values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    derived_expression: str = ""
    lineage_by_row: dict[str, tuple[str, ...]] = field(default_factory=dict)
    required_properties: frozenset[str] = frozenset()
    provided_properties: frozenset[str] = frozenset()
    dependencies: tuple[tuple[str, str], ...] = ()
    node_kind: str = "source"
    covered_node_ids: tuple[str, ...] = ()
    required_properties_by_node: tuple[tuple[str, tuple[str, ...]], ...] = ()


class EvidencePlanOptimizer:
    """Select the minimum-token certified evidence plan for one table view."""

    def __init__(
        self,
        store: Any,
        *,
        profile_cache_dir: str | None = None,
        encoding_mode: str = "optimized",
        optimization_strategy: str = "global_dp",
        token_counter: Callable[[Any], int] | None = None,
        token_counter_is_exact: bool | None = None,
        joint_realizations: bool = True,
        factor_row_addresses: bool = True,
    ) -> None:
        if encoding_mode not in {"optimized", "raw"}:
            raise ValueError(f"Unknown evidence coding mode: {encoding_mode}")
        if optimization_strategy not in {"global_dp", "local_greedy"}:
            raise ValueError(
                f"Unknown evidence-plan optimization strategy: {optimization_strategy}"
            )
        self.store = store
        self.profiles = ProfileStore(profile_cache_dir)
        self.encoding_mode = encoding_mode
        self.optimization_strategy = optimization_strategy
        self.has_exact_token_counter = (
            token_counter is not None
            if token_counter_is_exact is None
            else bool(token_counter_is_exact)
        )
        self.count_tokens = token_counter or approx_token_count
        self._line_token_cost_cache: dict[str, int] = {}
        self._grid_layout_cache: dict[
            tuple[str, bool, tuple[str, ...]], dict[str, Any]
        ] = {}
        self._raw_equivalent_cost_cache: dict[
            tuple[str, tuple[tuple[str, str], ...]], int
        ] = {}
        self.joint_realizations = bool(joint_realizations)
        self.factor_row_addresses = bool(factor_row_addresses)

    def materialize(
        self,
        *,
        slice_view: dict[str, Any],
        dependencies: list[dict[str, Any]] | None = None,
        target_tokens: int = 0,
        fixed_rendering: str = "",
        fixed_cost: tuple[int, int] | None = None,
        derived_observables: list[DerivedObservable | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._rendering_frontiers = {}
        table = self.store.table(slice_view.get("table_id"))
        row_ids = [str(row["row_id"]) for row in slice_view.get("rows", [])]
        col_ids = [str(header["col_id"]) for header in slice_view.get("headers", [])]
        requirements = self._requirement_map(dependencies or [], col_ids)
        profiles = {col_id: self.profiles.profile(table, col_id) for col_id in col_ids}
        groups = self._evidence_groups(col_ids, dependencies or [])

        observables = [
            item if isinstance(item, DerivedObservable) else DerivedObservable.model_validate(item)
            for item in (derived_observables or [])
        ]
        source_order = list(dict.fromkeys(
            col_id for group in groups for col_id in group
        ))
        observables = topological_observables(observables, set(source_order))
        observable_ids = [observable.observable_id for observable in observables]
        node_values, node_headers, node_lineage = self._materialize_evidence_graph(
            table=table,
            row_ids=row_ids,
            source_order=source_order,
            observables=observables,
        )

        derived_candidates: dict[str, list[EncodingCandidate]] = {}
        for observable in observables:
            generated = [self._price_candidate(candidate) for candidate in self._generate_derived_candidates(
                table=table,
                row_ids=row_ids,
                observable=observable,
                node_values=node_values,
                node_headers=node_headers,
                node_lineage=node_lineage,
                source_node_ids=set(source_order),
            )]
            certified = [candidate for candidate in generated if candidate.certified]
            if not certified:
                raise RuntimeError(
                    f"No distinction-preserving representation for "
                    f"{table.table_id}:{observable.observable_id}"
                )
            derived_candidates[observable.observable_id] = self._deduplicate_candidates(
                certified
            )

        order = [*source_order, *observable_ids]
        render_order = [*reversed(observable_ids), *source_order]
        self._search_render_context = (table, render_order, row_ids)
        fixed_lexical, fixed_bytes = (
            fixed_cost if fixed_cost is not None else self._cost_vector(fixed_rendering)
        )
        optimizer_args = dict(
            table=table,
            row_ids=row_ids,
            source_order=source_order,
            profiles=profiles,
            source_requirements=requirements,
            observables=observables,
            derived_candidates=derived_candidates,
            source_groups=groups,
            fixed_cost=(fixed_lexical, fixed_bytes),
        )
        optimizer = (
            self._minimum_cost_dependency_plan
            if self.optimization_strategy == "global_dp"
            else self._local_greedy_dependency_plan
        )
        selected, plan_trace, plan_frontier, candidate_catalog = optimizer(
            **optimizer_args,
        )
        plan_frontier = [
            (active, self._rendered_body_token_cost(self._render_selected(
                table, render_order, row_ids, alternative)), lexical, byte, order_, alternative)
            for active, _estimated, lexical, byte, order_, alternative in plan_frontier
        ]
        if self.optimization_strategy == "global_dp":
            best = min(plan_frontier, key=lambda item: (item[1], item[4]))
            selected = best[5]
        plan_trace["selected_value_tokens"] = self._rendered_body_token_cost(
            self._render_selected(table, render_order, row_ids, selected)
        )
        selected_by_unit = {
            unit_id: candidate
            for candidate in selected
            for unit_id in (candidate.covered_node_ids or (candidate.col_id,))
        }
        selected_tokens = int(plan_trace["selected_value_tokens"])
        candidate_pool: list[EncodingCandidate] = []
        for candidates in candidate_catalog.values():
            for candidate in candidates:
                if candidate not in candidate_pool:
                    candidate_pool.append(candidate)
        layer_widths = [
            int(layer.get("states", 0) or 0)
            for layer in plan_trace.get("dp_layers", [])
        ]

        trace = {
            "objective": "minimize complete rendered tokens subject to distinction preservation",
            "encoding_mode": self.encoding_mode,
            "optimization_strategy": self.optimization_strategy,
            "joint_realizations": self.joint_realizations,
            "factor_row_addresses": self.factor_row_addresses,
            "target_evidence_tokens": int(target_tokens or 0),
            "minimum_required_tokens": selected_tokens,
            "within_target": (
                not target_tokens or selected_tokens <= int(target_tokens)
            ),
            "evidence_groups": groups,
            "evidence_graph": self._evidence_graph_trace(
                source_order=source_order,
                source_requirements=requirements,
                observables=observables,
            ),
            "planned_nodes": [
                unit_id for unit_id in order if unit_id in selected_by_unit
            ],
            "candidate_count": len(candidate_pool),
            "visited_states": sum(layer_widths),
            "widest_layer_states": max(layer_widths or [0]),
            **plan_trace,
            "selected": {
                candidate.col_id: self._candidate_trace(candidate)
                for candidate in selected
            },
            "candidates": {
                unit_id: [
                    self._candidate_trace(candidate)
                    for candidate in candidate_catalog[unit_id]
                ]
                for unit_id in order
                if unit_id in candidate_catalog
            },
            "candidate_pool": [
                self._candidate_trace(candidate) for candidate in candidate_pool
            ],
        }
        rendered = self._render_selected(table, render_order, row_ids, selected)
        rendered_cost = self._rendered_body_token_cost(rendered)
        if self.has_exact_token_counter and rendered_cost != selected_tokens:
            raise RuntimeError(
                "Canonical evidence cost mismatch: "
                f"table={table.table_id}, planned={selected_tokens}, "
                f"rendered={rendered_cost}, "
                f"operators={[candidate.operator for candidate in selected]}"
            )
        trace["canonical_cost_verified"] = (
            self.has_exact_token_counter and rendered_cost == selected_tokens
        )
        plan_alternatives = []
        for active, token_cost, lexical_cost, byte_cost, alternative_order, alternative in plan_frontier:
            alternative_rendered = self._render_selected(
                table, render_order, row_ids, alternative
            )
            alternative_cost = self._rendered_body_token_cost(alternative_rendered)
            if self.has_exact_token_counter and alternative_cost != token_cost:
                raise RuntimeError(
                    "Canonical alternative cost mismatch: "
                    f"table={table.table_id}, planned={token_cost}, "
                    f"rendered={alternative_cost}, "
                    f"operators={[candidate.operator for candidate in alternative]}, "
                    f"lines={self._rendered_body_lines(alternative_rendered)!r}"
                )
            plan_alternatives.append({
                **alternative_rendered,
                "lexical_cost": lexical_cost,
                "byte_cost": byte_cost,
                "token_cost": token_cost,
                "shared_sections": sorted(active - {GRID_SECTION}),
                "plan_order": list(alternative_order),
                "selected_traces": {
                    candidate.col_id: self._candidate_trace(candidate)
                    for candidate in alternative
                },
            })
        return {
            **rendered,
            "selection_trace": trace,
            "plan_alternatives": plan_alternatives,
            "profiles": {col_id: asdict(profile) for col_id, profile in profiles.items()},
        }

    def _minimum_cost_dependency_plan(
        self,
        *,
        table: Any,
        row_ids: list[str],
        source_order: list[str],
        profiles: dict[str, ColumnProfile],
        source_requirements: dict[str, dict[str, Any]],
        observables: list[DerivedObservable],
        derived_candidates: dict[str, list[EncodingCandidate]],
        source_groups: list[list[str]],
        fixed_cost: tuple[int, int],
    ) -> tuple[
        list[EncodingCandidate],
        dict[str, Any],
        list[tuple[frozenset[str], int, int, int, tuple[str, ...], list[EncodingCandidate]]],
        dict[str, list[EncodingCandidate]],
    ]:
        """Find the minimum-token complete plan over grounded realizations."""

        State = tuple[int, int, int, tuple[str, ...], list[EncodingCandidate]]
        ObligationKey = tuple[tuple[str, tuple[str, ...]], ...]
        states: dict[tuple[frozenset[str], ObligationKey], list[State]] = {
            (frozenset(), ()): [(0, 0, 0, (), [])]
        }
        layers: list[dict[str, Any]] = []
        candidate_catalog: dict[str, list[EncodingCandidate]] = {
            unit_id: list(candidates)
            for unit_id, candidates in derived_candidates.items()
        }

        # Traverse expression nodes from consumers to dependencies so that a
        # derive choice can add obligations before its operand node is planned.
        for observable in reversed(observables):
            base_requirements = (
                {f"derived_{observable.mode}"} if observable.required else set()
            )
            layer_requirements = set(base_requirements)
            for _active, obligation_key in states:
                for node_id, properties in obligation_key:
                    if node_id == observable.observable_id:
                        layer_requirements.update(properties)
            layer: dict[tuple[frozenset[str], ObligationKey], list[State]] = {}
            for (active, obligation_key), frontier in states.items():
                obligations = {
                    node_id: set(properties)
                    for node_id, properties in obligation_key
                }
                required = base_requirements | obligations.pop(
                    observable.observable_id, set()
                )
                for token_cost, lexical_cost, byte_cost, plan_order, selected in frontier:
                    if not required:
                        next_key: ObligationKey = tuple(sorted(
                            (node_id, tuple(sorted(properties)))
                            for node_id, properties in obligations.items()
                        ))
                        self._insert_best_state(
                            layer.setdefault((active, next_key), []),
                            (
                                token_cost,
                                lexical_cost,
                                byte_cost,
                                plan_order,
                                list(selected),
                            ),
                        )
                        continue
                    for candidate in derived_candidates[observable.observable_id]:
                        if not required <= set(candidate.provided_properties):
                            continue
                        next_obligations = {
                            node_id: set(properties)
                            for node_id, properties in obligations.items()
                        }
                        for node_id, property_name in candidate.dependencies:
                            next_obligations.setdefault(node_id, set()).add(property_name)
                        next_key: ObligationKey = tuple(sorted(
                            (node_id, tuple(sorted(properties)))
                            for node_id, properties in next_obligations.items()
                        ))
                        candidate_sections = self._candidate_sections(candidate)
                        activated = candidate_sections - active
                        heading_vectors = [
                            self._shared_cost_vector(
                                section, table=table, row_ids=row_ids
                            )
                            for section in activated
                        ]
                        heading_tokens = sum(
                            self._shared_token_cost(
                                section, table=table, row_ids=row_ids
                            )
                            for section in activated
                        )
                        next_active = active | candidate_sections
                        next_state: State = (
                            token_cost
                            + self._candidate_transition_token_cost(
                                candidate,
                                active,
                                table=table,
                                row_ids=row_ids,
                            )
                            + heading_tokens,
                            lexical_cost + candidate.lexical_cost
                            + sum(item[0] for item in heading_vectors),
                            byte_cost + candidate.byte_cost
                            + sum(item[1] for item in heading_vectors),
                            (*plan_order, candidate.operator),
                            [*selected, candidate],
                        )
                        self._insert_best_state(
                            layer.setdefault((next_active, next_key), []), next_state
                        )
            states = layer
            layers.append({
                "node": observable.observable_id,
                "kind": "expression",
                "required_properties": sorted(layer_requirements),
                "states": sum(len(frontier) for frontier in states.values()),
            })

        unresolved_expression_obligations = sorted({
            node_id
            for (_active, obligation_key) in states
            for node_id, _properties in obligation_key
            if node_id not in source_order
        })
        if unresolved_expression_obligations:
            raise RuntimeError(
                "Unresolved expression obligations after DAG planning: "
                + ", ".join(unresolved_expression_obligations)
            )

        operand_columns = {
            operand.column
            for observable in observables
            for operand in observable.operands
            if operand.column in source_order
        }
        final_states: dict[frozenset[str], list[State]] = {}
        for (initial_active, obligation_key), frontier in states.items():
            obligations = {
                node_id: set(properties)
                for node_id, properties in obligation_key
            }
            required_order, source_candidates = self._source_candidate_space(
                table=table,
                row_ids=row_ids,
                source_order=source_order,
                profiles=profiles,
                source_requirements=source_requirements,
                source_groups=source_groups,
                obligations=obligations,
                operand_columns=operand_columns,
            )
            for candidate in source_candidates:
                self._append_unique_candidate(candidate_catalog, candidate)
            coverers = {
                node_id: [
                    candidate for candidate in source_candidates
                    if node_id in (candidate.covered_node_ids or (candidate.col_id,))
                ]
                for node_id in required_order
            }
            branch: dict[tuple[frozenset[str], frozenset[str]], list[State]] = {
                (initial_active, frozenset()): list(frontier)
            }
            complete = frozenset(required_order)
            while branch and any(covered != complete for _active, covered in branch):
                next_branch: dict[
                    tuple[frozenset[str], frozenset[str]], list[State]
                ] = {}
                for (active, covered), active_frontier in branch.items():
                    if covered == complete:
                        next_branch.setdefault((active, covered), []).extend(active_frontier)
                        continue
                    node_id = next(item for item in required_order if item not in covered)
                    for candidate in coverers[node_id]:
                        candidate_nodes = frozenset(
                            candidate.covered_node_ids or (candidate.col_id,)
                        )
                        if candidate_nodes & covered or not candidate_nodes <= complete:
                            continue
                        candidate_sections = self._candidate_sections(candidate)
                        activated = candidate_sections - active
                        heading_tokens = sum(
                            self._shared_token_cost(
                                section, table=table, row_ids=row_ids
                            )
                            for section in activated
                        )
                        heading_vectors = [
                            self._shared_cost_vector(
                                section, table=table, row_ids=row_ids
                            )
                            for section in activated
                        ]
                        next_active = active | candidate_sections
                        next_covered = covered | candidate_nodes
                        for token_cost, lexical_cost, byte_cost, plan_order, selected in active_frontier:
                            self._insert_best_state(
                                next_branch.setdefault((next_active, next_covered), []),
                                (
                                    token_cost
                                    + self._candidate_transition_token_cost(
                                        candidate,
                                        active,
                                        table=table,
                                        row_ids=row_ids,
                                    )
                                    + heading_tokens,
                                    lexical_cost + candidate.lexical_cost
                                    + sum(item[0] for item in heading_vectors),
                                    byte_cost + candidate.byte_cost
                                    + sum(item[1] for item in heading_vectors),
                                    (*plan_order, candidate.operator),
                                    [*selected, candidate],
                                ),
                            )
                branch = next_branch
            layers.append({
                "node": "+".join(required_order),
                "kind": "source_exact_cover",
                "required_nodes": list(required_order),
                "joint_candidates": sum(
                    len(candidate.covered_node_ids) > 1
                    for candidate in source_candidates
                ),
                "states": sum(len(items) for items in branch.values()),
            })

            for (active, covered), active_frontier in branch.items():
                if covered != complete:
                    continue
                for state in active_frontier:
                    self._insert_best_state(
                        final_states.setdefault(active, []), state
                    )

        plan_frontier = [
            (active, *state)
            for active, frontier in final_states.items()
            for state in frontier
        ]
        if not plan_frontier:
            raise RuntimeError(f"No complete evidence plan for {table.table_id}")
        active, token_cost, lexical_cost, byte_cost, plan_order, selected = min(
            plan_frontier,
            key=lambda item: (
                item[1],
                *self._dominant_cost_order(
                    fixed_cost[0] + item[2], fixed_cost[1] + item[3]
                ),
                item[4],
                sorted(item[0]),
            ),
        )
        raw_tokens = sum(candidate.raw_equivalent_tokens for candidate in selected)
        trace = {
            "selection_reason": "MINIMUM_COST_CERTIFIED_EVIDENCE_PLAN",
            "selected_value_tokens": token_cost,
            "selected_lexical_cost": lexical_cost,
            "selected_utf8_bytes": byte_cost,
            "fixed_lexical_cost": fixed_cost[0],
            "fixed_utf8_bytes": fixed_cost[1],
            "raw_equivalent_tokens": raw_tokens,
            "reader_token_savings": raw_tokens - token_cost,
            "reader_value_compression_ratio": round(raw_tokens / max(1, token_cost), 6),
            "shared_sections": sorted(active - {GRID_SECTION}),
            "shared_section_tokens": sum(
                self._shared_token_cost(
                    section, table=table, row_ids=row_ids
                )
                for section in active
            ),
            "dp_layers": layers,
            "plan_order": list(plan_order),
            "dependency_obligations": {
                node_id: sorted({
                    property_name
                    for candidate in selected
                    for dependency_node, property_name in candidate.dependencies
                    if dependency_node == node_id
                })
                for node_id in sorted({
                    dependency_node
                    for candidate in selected
                    for dependency_node, _property_name in candidate.dependencies
                })
            },
        }
        trace["source_obligations"] = {
            node_id: properties
            for node_id, properties in trace["dependency_obligations"].items()
            if node_id in source_order
        }
        return selected, trace, plan_frontier, candidate_catalog

    def _local_greedy_dependency_plan(
        self,
        *,
        table: Any,
        row_ids: list[str],
        source_order: list[str],
        profiles: dict[str, ColumnProfile],
        source_requirements: dict[str, dict[str, Any]],
        observables: list[DerivedObservable],
        derived_candidates: dict[str, list[EncodingCandidate]],
        source_groups: list[list[str]],
        fixed_cost: tuple[int, int],
    ) -> tuple[
        list[EncodingCandidate],
        dict[str, Any],
        list[tuple[frozenset[str], int, int, int, tuple[str, ...], list[EncodingCandidate]]],
        dict[str, list[EncodingCandidate]],
    ]:
        """Select each legal realization by immediate cost only.

        This deliberately myopic policy is an experimental control for the
        global dependency DP. It sees the same certified candidates and shared
        section costs, but it does not price obligations that a choice pushes
        onto later source or expression nodes.
        """

        active: frozenset[str] = frozenset()
        obligations: dict[str, set[str]] = defaultdict(set)
        selected: list[EncodingCandidate] = []
        plan_order: list[str] = []
        token_cost = lexical_cost = byte_cost = 0
        layers: list[dict[str, Any]] = []
        candidate_catalog: dict[str, list[EncodingCandidate]] = {
            unit_id: list(candidates)
            for unit_id, candidates in derived_candidates.items()
        }

        def incremental_cost(candidate: EncodingCandidate) -> tuple[int, int, int]:
            activated = self._candidate_sections(candidate) - active
            return (
                self._candidate_transition_token_cost(
                    candidate,
                    active,
                    table=table,
                    row_ids=row_ids,
                ) + sum(
                    self._shared_token_cost(
                        section, table=table, row_ids=row_ids
                    )
                    for section in activated
                ),
                candidate.lexical_cost + sum(
                    self._shared_cost_vector(
                        section, table=table, row_ids=row_ids
                    )[0]
                    for section in activated
                ),
                candidate.byte_cost + sum(
                    self._shared_cost_vector(
                        section, table=table, row_ids=row_ids
                    )[1]
                    for section in activated
                ),
            )

        def choose(candidates: list[EncodingCandidate]) -> EncodingCandidate:
            return min(
                candidates,
                key=lambda candidate: (
                    *incremental_cost(candidate),
                    candidate.operator,
                ),
            )

        def accept(candidate: EncodingCandidate) -> None:
            nonlocal active, token_cost, lexical_cost, byte_cost
            delta_tokens, delta_lexical, delta_bytes = incremental_cost(candidate)
            token_cost += delta_tokens
            lexical_cost += delta_lexical
            byte_cost += delta_bytes
            active = active | self._candidate_sections(candidate)
            selected.append(candidate)
            plan_order.append(candidate.operator)
            for node_id, property_name in candidate.dependencies:
                obligations[node_id].add(property_name)

        # Consumers are visited before dependencies, matching the global DP.
        for observable in reversed(observables):
            required = (
                {f"derived_{observable.mode}"} if observable.required else set()
            ) | obligations.pop(observable.observable_id, set())
            legal = [
                candidate
                for candidate in derived_candidates[observable.observable_id]
                if required <= set(candidate.provided_properties)
            ]
            if required:
                if not legal:
                    raise RuntimeError(
                        f"No locally legal realization for {observable.observable_id}"
                    )
                accept(choose(legal))
            layers.append({
                "node": observable.observable_id,
                "kind": "expression",
                "required_properties": sorted(required),
                "legal_candidates": len(legal),
            })

        unresolved = sorted(node_id for node_id in obligations if node_id not in source_order)
        if unresolved:
            raise RuntimeError(
                "Unresolved expression obligations after local planning: "
                + ", ".join(unresolved)
            )

        operand_columns = {
            operand.column
            for observable in observables
            for operand in observable.operands
            if operand.column in source_order
        }
        required_order, source_candidates = self._source_candidate_space(
            table=table,
            row_ids=row_ids,
            source_order=source_order,
            profiles=profiles,
            source_requirements=source_requirements,
            source_groups=source_groups,
            obligations=obligations,
            operand_columns=operand_columns,
        )
        for candidate in source_candidates:
            self._append_unique_candidate(candidate_catalog, candidate)
        covered: set[str] = set()
        for node_id in required_order:
            if node_id in covered:
                continue
            legal = [
                candidate for candidate in source_candidates
                if node_id in (candidate.covered_node_ids or (candidate.col_id,))
                and not covered.intersection(
                    candidate.covered_node_ids or (candidate.col_id,)
                )
            ]
            if not legal:
                raise RuntimeError(
                    f"No locally legal source realization for {table.table_id}:{node_id}"
                )
            selected_candidate = choose(legal)
            accept(selected_candidate)
            covered.update(
                selected_candidate.covered_node_ids or (selected_candidate.col_id,)
            )
            layers.append({
                "node": node_id,
                "kind": "source_greedy_cover",
                "covered_nodes": list(
                    selected_candidate.covered_node_ids
                    or (selected_candidate.col_id,)
                ),
                "legal_candidates": len(legal),
            })

        raw_tokens = sum(candidate.raw_equivalent_tokens for candidate in selected)
        dependency_obligations = {
            node_id: sorted({
                property_name
                for candidate in selected
                for dependency_node, property_name in candidate.dependencies
                if dependency_node == node_id
            })
            for node_id in sorted({
                dependency_node
                for candidate in selected
                for dependency_node, _property_name in candidate.dependencies
            })
        }
        trace = {
            "selection_reason": "LOCAL_GREEDY_CERTIFIED_PLAN",
            "selected_value_tokens": token_cost,
            "selected_lexical_cost": lexical_cost,
            "selected_utf8_bytes": byte_cost,
            "fixed_lexical_cost": fixed_cost[0],
            "fixed_utf8_bytes": fixed_cost[1],
            "raw_equivalent_tokens": raw_tokens,
            "reader_token_savings": raw_tokens - token_cost,
            "reader_value_compression_ratio": round(raw_tokens / max(1, token_cost), 6),
            "shared_sections": sorted(active - {GRID_SECTION}),
            "shared_section_tokens": sum(
                self._shared_token_cost(
                    section, table=table, row_ids=row_ids
                )
                for section in active
            ),
            "dp_layers": layers,
            "plan_order": plan_order,
            "dependency_obligations": dependency_obligations,
            "source_obligations": {
                node_id: properties
                for node_id, properties in dependency_obligations.items()
                if node_id in source_order
            },
        }
        frontier = [(
            active,
            token_cost,
            lexical_cost,
            byte_cost,
            tuple(plan_order),
            selected,
        )]
        return selected, trace, frontier, candidate_catalog

    def _source_candidate_space(
        self,
        *,
        table: Any,
        row_ids: list[str],
        source_order: list[str],
        profiles: dict[str, ColumnProfile],
        source_requirements: dict[str, dict[str, Any]],
        source_groups: list[list[str]],
        obligations: dict[str, set[str]],
        operand_columns: set[str],
    ) -> tuple[list[str], list[EncodingCandidate]]:
        """Build the certified exact-cover space for required source nodes."""

        required_order: list[str] = []
        effective_by_node: dict[str, dict[str, Any]] = {}
        values_by_node: dict[str, dict[str, str]] = {}
        candidates: list[EncodingCandidate] = []
        raw_cost_by_node: dict[str, int] = {}
        for node_id in source_order:
            effective = self._requirements_with_obligations(
                source_requirements[node_id],
                obligations.get(node_id, set()),
                row_ids=row_ids,
            )
            semantic = set(effective["mandatory"]) - STRUCTURAL_REQUIREMENTS
            has_dependency = bool(obligations.get(node_id))
            independently_required = bool(
                effective["mandatory"] and node_id not in operand_columns
            )
            if not (semantic or has_dependency or independently_required):
                continue
            required_order.append(node_id)
            effective_by_node[node_id] = effective
            values = {
                row_id: str(
                    table.rows[self._row_idx(row_id)][profiles[node_id].col_idx]
                )
                for row_id in row_ids
            }
            values_by_node[node_id] = values
            generated = [
                self._price_candidate(candidate)
                for candidate in self._generate_candidates(
                table=table,
                row_ids=row_ids,
                profile=profiles[node_id],
                selected=values,
                requirements=effective,
                )
            ]
            certified = self._deduplicate_candidates([
                candidate for candidate in generated if candidate.certified
            ])
            if not certified:
                failed = ", ".join(candidate.operator for candidate in generated) or "none"
                raise RuntimeError(
                    f"No distinction-preserving representation for "
                    f"{table.table_id}:{node_id}; candidates={failed}"
                )
            candidates.extend(certified)
            raw = next(candidate for candidate in generated if candidate.operator == "raw")
            raw_cost_by_node[node_id] = raw.token_cost

        if self.encoding_mode == "raw" or not self.joint_realizations:
            return required_order, candidates

        required = set(required_order)
        joint_node_sets: set[tuple[str, ...]] = set()
        for source_group in source_groups:
            group = [node_id for node_id in source_group if node_id in required]
            if len(group) < 2:
                continue
            if len(group) <= 8:
                for size in range(2, min(4, len(group)) + 1):
                    joint_node_sets.update(combinations(group, size))
            else:
                for size in (2, 3, 4):
                    for start in range(0, len(group) - size + 1):
                        joint_node_sets.add(tuple(group[start:start + size]))
                for start in range(0, len(group), 4):
                    chunk = tuple(group[start:start + 4])
                    if len(chunk) >= 2:
                        joint_node_sets.add(chunk)

        for node_ids in sorted(joint_node_sets, key=lambda item: (len(item), item)):
            raw_cost = sum(raw_cost_by_node[node_id] for node_id in node_ids)
            for candidate in (
                self._price_candidate(item)
                for item in self._generate_joint_source_candidates(
                table=table,
                row_ids=row_ids,
                node_ids=node_ids,
                profiles=profiles,
                values_by_node=values_by_node,
                requirements_by_node=effective_by_node,
                raw_equivalent_tokens=raw_cost,
                )
            ):
                # Keep every certified joint realization until global planning;
                # its advantage can differ when it opens or appends to a grid.
                if candidate.certified:
                    candidates.append(candidate)
        return required_order, candidates

    def _generate_joint_source_candidates(
        self,
        *,
        table: Any,
        row_ids: list[str],
        node_ids: tuple[str, ...],
        profiles: dict[str, ColumnProfile],
        values_by_node: dict[str, dict[str, str]],
        requirements_by_node: dict[str, dict[str, Any]],
        raw_equivalent_tokens: int,
    ) -> list[EncodingCandidate]:
        """Create exact multi-node realizations with executable reconstruction."""

        if not row_ids:
            return []
        headers = tuple(self._header(profiles[node_id]) for node_id in node_ids)
        tuples_by_row = {
            row_id: tuple(values_by_node[node_id][row_id] for node_id in node_ids)
            for row_id in row_ids
        }
        ordered_tuples = list(dict.fromkeys(tuples_by_row.values()))
        required_by_node = tuple(
            (node_id, tuple(sorted(requirements_by_node[node_id]["mandatory"])))
            for node_id in node_ids
        )
        all_requirements = {
            property_name
            for _node_id, properties in required_by_node
            for property_name in properties
        }

        def make(operator: str, code_line: str) -> EncodingCandidate:
            reconstructed = {
                node_id: {
                    row_id: tuples_by_row[row_id][index]
                    for row_id in row_ids
                }
                for index, node_id in enumerate(node_ids)
            }
            exact = all(
                reconstructed[node_id] == values_by_node[node_id]
                for node_id in node_ids
            )
            checks = {
                "source_reconstruction": exact,
                "row_binding": set(tuples_by_row) == set(row_ids),
                "joint_binding": exact,
                "rendering_complete": bool(code_line),
                **{property_name: exact for property_name in all_requirements},
            }
            lexical_cost, byte_cost = self._cost_vector(code_line)
            return EncodingCandidate(
                col_id="joint[" + "+".join(node_ids) + "]",
                source_header=" + ".join(headers),
                display_header="(" + ", ".join(headers) + ")",
                operator=operator,
                values={
                    row_id: json.dumps(
                        list(values), ensure_ascii=False, separators=(",", ":")
                    )
                    for row_id, values in tuples_by_row.items()
                },
                token_cost=(
                    0 if self.has_exact_token_counter
                    else max(1, self.count_tokens(code_line))
                ),
                lexical_cost=lexical_cost,
                byte_cost=byte_cost,
                grid_visible=False,
                shared_sections=frozenset({"codes"}),
                code_lines=[code_line],
                covered_requirements=[
                    f"{node_id}:{property_name}"
                    for node_id, properties in required_by_node
                    for property_name in properties
                ],
                certification_checks=checks,
                certified=all(checks.values()),
                source_reversible=exact,
                raw_equivalent_tokens=raw_equivalent_tokens,
                source_col_ids=node_ids,
                required_properties=frozenset(all_requirements),
                provided_properties=frozenset(
                    property_name for property_name, passed in checks.items() if passed
                ),
                node_kind="joint_source",
                covered_node_ids=node_ids,
                required_properties_by_node=required_by_node,
            )

        candidates: list[EncodingCandidate] = []
        tuple_groups = [
            (
                values,
                [row_id for row_id, row_values in tuples_by_row.items() if row_values == values],
            )
            for values in ordered_tuples
        ]
        tuple_body = "; ".join(
            json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
            + "@" + self._compact_row_set(members)
            for values, members in tuple_groups
        )
        candidates.append(make(
            "tuple_postings",
            "(" + ",".join(headers) + ") := " + tuple_body,
        ))

        modal_values, modal_rows = max(
            tuple_groups,
            key=lambda item: (len(item[1]), -ordered_tuples.index(item[0])),
        )
        exceptions = [
            (values, members) for values, members in tuple_groups
            if values != modal_values
        ]
        default_line = (
            "(" + ",".join(headers) + ") := "
            + json.dumps(list(modal_values), ensure_ascii=False, separators=(",", ":"))
            + " otherwise"
        )
        if exceptions:
            default_line += "; " + "; ".join(
                json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
                + "@" + self._compact_row_set(members)
                for values, members in exceptions
            )
        candidates.append(make("default_tuple_exceptions", default_line))

        posting_parts = []
        for node_id, header in zip(node_ids, headers):
            values = values_by_node[node_id]
            distinct = list(dict.fromkeys(values.values()))
            posting_parts.append(
                header + ":" + ",".join(
                    json.dumps(value, ensure_ascii=False)
                    + "@" + self._compact_row_set([
                        row_id for row_id, raw in values.items() if raw == value
                    ])
                    for value in distinct
                )
            )
        candidates.append(make(
            "shared_postings",
            "{" + "; ".join(posting_parts) + "}",
        ))
        return candidates

    @staticmethod
    def _requirements_with_obligations(
        base: dict[str, Any],
        obligations: set[str],
        *,
        row_ids: list[str],
    ) -> dict[str, Any]:
        result = {
            "mandatory": set(base["mandatory"]),
            "optional": set(base["optional"]),
            "rows_by_kind": {
                kind: set(rows) for kind, rows in base["rows_by_kind"].items()
            },
            "properties_by_kind": {
                kind: list(properties)
                for kind, properties in base["properties_by_kind"].items()
            },
        }
        for property_name in obligations:
            result["mandatory"].add(property_name)
            result["rows_by_kind"].setdefault(property_name, set()).update(row_ids)
        return result

    @staticmethod
    def _append_unique_candidate(
        catalog: dict[str, list[EncodingCandidate]],
        candidate: EncodingCandidate,
    ) -> None:
        signature = (
            candidate.operator,
            candidate.covered_node_ids,
            candidate.required_properties,
            candidate.dependencies,
            candidate.render_headers,
            tuple(sorted(candidate.shared_sections)),
        )
        existing = catalog.setdefault(candidate.col_id, [])
        if not any((
            item.operator,
            item.covered_node_ids,
            item.required_properties,
            item.dependencies,
            item.render_headers,
            tuple(sorted(item.shared_sections)),
        ) == signature for item in existing):
            existing.append(candidate)

    @staticmethod
    def _evidence_graph_trace(
        *,
        source_order: list[str],
        source_requirements: dict[str, dict[str, Any]],
        observables: list[DerivedObservable],
    ) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": col_id,
                    "kind": "source",
                    "required_properties": sorted(
                        set(source_requirements[col_id]["mandatory"])
                        - STRUCTURAL_REQUIREMENTS
                    ),
                }
                for col_id in source_order
            ] + [
                {
                    "id": observable.observable_id,
                    "kind": "expression",
                    "operator": observable.operator,
                    "required_properties": (
                        [f"derived_{observable.mode}"]
                        if observable.required else []
                    ),
                }
                for observable in observables
            ],
            "edges": [
                {
                    "from": operand.column,
                    "to": observable.observable_id,
                    "role": operand.role,
                }
                for observable in observables
                for operand in observable.operands
            ],
        }

    @staticmethod
    def _render_cell(value: Any) -> str:
        """Escape values that would be ambiguous in the pipe-delimited grid."""

        text = str(value)
        if not text:
            return ""
        if (
            text != text.strip()
            or any(character in text for character in ("|", "\t", "\n", "\r"))
        ):
            return json.dumps(text, ensure_ascii=False)
        return text

    @staticmethod
    def _render_line(value: Any) -> str:
        """Normalize prose lines; grid rows keep empty cells through final pipes."""

        return str(value).rstrip(" \r\n")

    @classmethod
    def _render_grid_line(cls, values: Iterable[Any]) -> str:
        """Render one readable row with an explicit final cell boundary."""

        return CANONICAL_CELL_SEPARATOR.join(
            cls._render_cell(value) for value in values
        ) + " |"

    def _line_increment(self, line: Any) -> int:
        text = self._render_line(line)
        cached = self._line_token_cost_cache.get(text)
        if cached is not None:
            return cached
        anchor = "@"
        cost = self.count_tokens(
            anchor + CANONICAL_LINE_SEPARATOR + text
        ) - self.count_tokens(anchor)
        self._line_token_cost_cache[text] = cost
        return cost

    def _candidate_sections(self, candidate: EncodingCandidate) -> frozenset[str]:
        sections = set(candidate.shared_sections)
        if candidate.grid_visible:
            sections.add(GRID_SECTION)
        return frozenset(sections)

    def _grid_layout(self, table: Any, row_ids: list[str]) -> dict[str, Any]:
        hierarchical = bool(getattr(table, "metadata", {}).get("hierarchical"))
        cache_key = (str(table.table_id), hierarchical, tuple(row_ids))
        cached = self._grid_layout_cache.get(cache_key)
        if cached is not None:
            return cached
        explicit_base = [
            self._render_grid_line(["source_row"]),
            *(self._render_grid_line([row_id]) for row_id in row_ids),
        ]
        compact_axis = "Grid row order := " + self._compact_row_set(row_ids)
        compact_base: list[str] = []
        explicit_cost = sum(self._line_increment(line) for line in explicit_base)
        compact_cost = self._line_increment(compact_axis)
        if (
            not hierarchical
            and self.encoding_mode != "raw"
            and self.factor_row_addresses
            and row_ids
            and compact_cost < explicit_cost
        ):
            result = {
                "mode": "ordered_range",
                "axis_lines": [compact_axis],
                "base_lines": compact_base,
                "base_cost": compact_cost,
            }
        else:
            result = {
                "mode": "explicit",
                "axis_lines": [],
                "base_lines": explicit_base,
                "base_cost": explicit_cost,
            }
        self._grid_layout_cache[cache_key] = result
        return result

    def _shared_token_cost(
        self,
        section: str,
        *,
        table: Any,
        row_ids: list[str],
    ) -> int:
        if section == GRID_SECTION:
            return int(self._grid_layout(table, row_ids)["base_cost"])
        return self._line_increment(SECTION_HEADINGS[section])

    def _shared_cost_vector(
        self,
        section: str,
        *,
        table: Any,
        row_ids: list[str],
    ) -> tuple[int, int]:
        if section == GRID_SECTION:
            layout = self._grid_layout(table, row_ids)
            text = CANONICAL_LINE_SEPARATOR.join([
                *layout["axis_lines"], *layout["base_lines"]
            ])
        else:
            text = SECTION_HEADINGS[section]
        return self._cost_vector(text)

    def _candidate_private_token_cost(self, candidate: EncodingCandidate) -> int:
        private_lines = [
            rendered
            for line in (
                *candidate.decoder_lines,
                *candidate.code_lines,
                *candidate.patch_lines,
            )
            if (rendered := self._render_line(line))
        ]
        anchor = "@"
        fixed_cost = 0
        if private_lines:
            text = (
                anchor
                + CANONICAL_LINE_SEPARATOR
                + CANONICAL_LINE_SEPARATOR.join(
                    self._render_line(line) for line in private_lines
                )
            )
            fixed_cost += self.count_tokens(text) - self.count_tokens(anchor)
        if candidate.grid_visible:
            headers = candidate.render_headers or (candidate.display_header,)
            first_lines = [
                self._render_grid_line(headers),
                *(
                    self._render_grid_line(values)
                    for values in candidate.render_values.values()
                ),
            ]
            append_lines = [
                "@ | " + self._render_grid_line(headers),
                *(
                    "@ | " + self._render_grid_line(values)
                    for values in candidate.render_values.values()
                ),
            ]
            base_lines = ["@ |"] * len(append_lines)
            candidate_text = CANONICAL_LINE_SEPARATOR.join(append_lines)
            base_text = CANONICAL_LINE_SEPARATOR.join(base_lines)
            candidate.append_token_cost = fixed_cost + (
                self.count_tokens(candidate_text) - self.count_tokens(base_text)
            )
            return fixed_cost + sum(
                self._line_increment(line) for line in first_lines
            )
        candidate.append_token_cost = fixed_cost
        return fixed_cost

    def _price_candidate(self, candidate: EncodingCandidate) -> EncodingCandidate:
        candidate.token_cost = self._candidate_private_token_cost(candidate)
        return candidate

    def _candidate_transition_token_cost(
        self,
        candidate: EncodingCandidate,
        active: frozenset[str],
        *,
        table: Any,
        row_ids: list[str],
    ) -> int:
        if not candidate.grid_visible:
            return candidate.token_cost
        if GRID_SECTION in active:
            return candidate.append_token_cost
        if self._grid_layout(table, row_ids)["mode"] == "explicit":
            return candidate.append_token_cost
        return candidate.token_cost

    def _rendered_body_lines(self, rendered: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        if rendered["decoder_lines"]:
            lines.extend([SECTION_HEADINGS["decoder"], *rendered["decoder_lines"]])
        if rendered["code_lines"]:
            lines.extend([SECTION_HEADINGS["codes"], *rendered["code_lines"]])
        lines.extend(rendered.get("row_axis_lines", []))
        if rendered["grid_header"]:
            lines.append(self._render_grid_line(rendered["grid_header"]))
            lines.extend(
                self._render_grid_line(row)
                for row in rendered["grid_rows"]
            )
        if rendered["patch_lines"]:
            lines.extend([SECTION_HEADINGS["patches"], *rendered["patch_lines"]])
        return lines

    def _rendered_body_token_cost(self, rendered: dict[str, Any]) -> int:
        lines = self._rendered_body_lines(rendered)
        if not lines:
            return 0
        return self.count_tokens(CANONICAL_LINE_SEPARATOR.join(lines))

    def _render_selected(
        self,
        table: Any,
        unit_ids: list[str],
        row_ids: list[str],
        selected: list[EncodingCandidate],
    ) -> dict[str, Any]:
        order_index = {unit_id: index for index, unit_id in enumerate(unit_ids)}
        ordered_selected = sorted(
            selected,
            key=lambda candidate: (
                min(
                    order_index.get(unit_id, len(order_index))
                    for unit_id in (
                        candidate.covered_node_ids or (candidate.col_id,)
                    )
                ),
                candidate.col_id,
            ),
        )
        grid_candidates = [
            candidate for candidate in ordered_selected if candidate.grid_visible
        ]
        grid_column_ids = [
            col_id
            for candidate in grid_candidates
            for col_id in (
                candidate.source_col_ids
                if len(candidate.render_headers) > 1
                else (candidate.col_id,)
            )
        ]
        value_headers = [
            str(header)
            for candidate in grid_candidates
            for header in (candidate.render_headers or (candidate.display_header,))
        ]
        value_rows = [
            [
                str(value)
                for candidate in grid_candidates
                for value in (
                    candidate.render_values.get(row_id, ())
                    or (candidate.values.get(row_id, ""),)
                )
            ]
            for row_id in row_ids
        ] if grid_candidates else []
        explicit_header = ["source_row", *value_headers] if grid_candidates else []
        explicit_rows = [
            [row_id, *values] for row_id, values in zip(row_ids, value_rows)
        ]
        layout = self._grid_layout(table, row_ids)
        grid_header = explicit_header
        grid_rows = explicit_rows
        row_axis_lines: list[str] = []
        row_axis_mode = "explicit"
        row_axis_checks = {
            "row_order_complete": True,
            "row_addresses_recoverable": True,
        }
        if grid_candidates and layout["mode"] == "ordered_range":
            grid_header = value_headers
            grid_rows = value_rows
            row_axis_lines = list(layout["axis_lines"])
            row_axis_mode = "ordered_range"
            row_axis_checks = {
                "row_order_complete": len(value_rows) == len(row_ids),
                "row_addresses_recoverable": bool(self._compact_row_set(row_ids)),
            }
        selected_ops = {
            unit_id: candidate.operator
            for candidate in ordered_selected
            for unit_id in (candidate.covered_node_ids or (candidate.col_id,))
        }
        return {
            "code_lines": [
                rendered
                for item in ordered_selected
                for line in item.code_lines
                if (rendered := self._render_line(line))
            ],
            "decoder_lines": [
                rendered
                for item in ordered_selected
                for line in item.decoder_lines
                if (rendered := self._render_line(line))
            ],
            "patch_lines": [
                rendered
                for item in ordered_selected
                for line in item.patch_lines
                if (rendered := self._render_line(line))
            ],
            "grid_header": grid_header,
            "grid_rows": grid_rows,
            "explicit_grid_header": explicit_header,
            "explicit_grid_rows": explicit_rows,
            "row_axis_lines": row_axis_lines,
            "row_axis_mode": row_axis_mode,
            "row_axis_certification": row_axis_checks,
            "selected_ops": selected_ops,
            "grid_column_ids": grid_column_ids,
        }

    def _requirement_map(
        self,
        dependencies: list[dict[str, Any]],
        col_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        result = {
            col_id: {
                "mandatory": set(),
                "optional": set(),
                "rows_by_kind": {},
                "properties_by_kind": {},
            }
            for col_id in col_ids
        }
        for dependency in dependencies:
            kind = str(dependency.get("kind", "") or "")
            bucket = "mandatory" if dependency.get("mandatory", True) else "optional"
            for col_id in dependency.get("col_ids", []) or []:
                if col_id not in result or not kind:
                    continue
                result[col_id][bucket].add(kind)
                rows = {str(row_id) for row_id in dependency.get("row_ids", []) or []}
                result[col_id]["rows_by_kind"].setdefault(kind, set()).update(rows)
                properties = {
                    key: value
                    for key, value in dependency.items()
                    if key not in {"kind", "col_ids", "row_ids", "mandatory", "reason"}
                }
                if properties:
                    result[col_id]["properties_by_kind"].setdefault(kind, []).append(properties)
        return result

    def _evidence_groups(
        self,
        col_ids: list[str],
        dependencies: list[dict[str, Any]],
    ) -> list[list[str]]:
        """Union columns whose tuple binding must be preserved."""

        parent = {col_id: col_id for col_id in col_ids}

        def find(value: str) -> str:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for dependency in dependencies:
            if dependency.get("kind") != "alignment":
                continue
            linked = [col_id for col_id in dependency.get("col_ids", []) if col_id in parent]
            for left, right in zip(linked, linked[1:]):
                union(left, right)

        grouped: dict[str, list[str]] = {}
        for col_id in col_ids:
            grouped.setdefault(find(col_id), []).append(col_id)
        return list(grouped.values())

    def _generate_candidates(
        self,
        *,
        table: Any,
        row_ids: list[str],
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
    ) -> list[EncodingCandidate]:
        candidates = [
            self._make_candidate(
                table=table,
                profile=profile,
                selected=selected,
                requirements=requirements,
                operator="raw",
                display_header=self._header(profile),
                values=dict(selected),
                reconstructed_values=dict(selected),
            )
        ]
        if self.encoding_mode == "raw":
            return candidates

        dictionary = self._dictionary_candidate(table, profile, selected, requirements)
        if dictionary is not None:
            candidates.append(dictionary)
        default_factorization = self._default_factorization_candidate(
            table, profile, selected, requirements,
        )
        if default_factorization is not None:
            candidates.append(default_factorization)
        grouped_dictionary = self._grouped_dictionary_candidate(
            table, profile, selected, requirements,
        )
        if grouped_dictionary is not None:
            candidates.append(grouped_dictionary)

        structured_template = self._structured_template_candidate(
            table, profile, selected, requirements,
        )
        if structured_template is not None:
            candidates.append(structured_template)

        candidates.extend(self._affix_candidates(table, profile, selected, requirements))

        exact_delta = self._exact_delta_candidate(table, profile, selected, requirements)
        if exact_delta is not None:
            candidates.append(exact_delta)

        anchored = self._anchored_candidate(table, profile, selected, requirements)
        if anchored is not None:
            candidates.append(anchored)
        sparse_anchored = self._sparse_anchored_candidate(
            table, profile, selected, requirements,
        )
        if sparse_anchored is not None:
            candidates.append(sparse_anchored)

        rank = self._rank_candidate(table, profile, selected, requirements)
        if rank is not None:
            candidates.append(rank)
        grouped_rank = self._grouped_rank_candidate(
            table, profile, selected, requirements,
        )
        if grouped_rank is not None:
            candidates.append(grouped_rank)

        threshold = self._threshold_candidate(table, profile, selected, requirements)
        if threshold is not None:
            candidates.append(threshold)
        sparse_threshold = self._sparse_threshold_candidate(
            table, profile, selected, requirements,
        )
        if sparse_threshold is not None:
            candidates.append(sparse_threshold)
        return candidates

    def _default_factorization_candidate(
        self,
        table: Any,
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
    ) -> EncodingCandidate | None:
        """Store the modal value once and materialize only exact exceptions."""

        distinct = self._ordered_distinct(selected.values())
        if not distinct or len(selected) < 2 or len(distinct) >= len(selected):
            return None
        default = distinct[0]
        exceptions = []
        for raw in distinct[1:]:
            members = [
                row_id for row_id, value in selected.items() if value == raw
            ]
            exceptions.append(
                f"{json.dumps(raw, ensure_ascii=False)}@{self._compact_row_set(members)}"
            )
        line = f"{self._header(profile)} := {json.dumps(default, ensure_ascii=False)}"
        line += (
            " otherwise; " + "; ".join(exceptions)
            if exceptions else
            " for all source rows"
        )
        return self._make_candidate(
            table=table,
            profile=profile,
            selected=selected,
            requirements=requirements,
            operator="default_factorization",
            display_header=self._header(profile),
            values=dict(selected),
            code_lines=[line],
            grid_visible=False,
            reconstructed_values=dict(selected),
        )

    def _grouped_dictionary_candidate(
        self,
        table: Any,
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
    ) -> EncodingCandidate | None:
        """Invert repeated values into row groups while retaining tuple bindings."""

        distinct = self._ordered_distinct(selected.values())
        if not distinct or len(distinct) >= len(selected):
            return None
        groups = []
        for raw in distinct:
            members = [
                row_id for row_id, value in selected.items() if value == raw
            ]
            groups.append(
                f"{json.dumps(raw, ensure_ascii=False)}@{self._compact_row_set(members)}"
            )
        return self._make_candidate(
            table=table,
            profile=profile,
            selected=selected,
            requirements=requirements,
            operator="grouped_dictionary",
            display_header=self._header(profile),
            values=dict(selected),
            code_lines=[f"{self._header(profile)} := " + "; ".join(groups)],
            grid_visible=False,
            reconstructed_values=dict(selected),
        )

    def _dictionary_candidate(
        self,
        table: Any,
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
    ) -> EncodingCandidate | None:
        distinct = self._ordered_distinct(selected.values())
        if not distinct or len(distinct) >= len(selected):
            return None
        noun = "id" if profile.type_hint == "identifier" else "category"
        mapping = {raw: f"{noun} {index + 1}" for index, raw in enumerate(distinct)}
        values = {row_id: mapping[raw] for row_id, raw in selected.items()}
        inverse = {alias: raw for raw, alias in mapping.items()}
        reconstructed = {row_id: inverse[alias] for row_id, alias in values.items()}
        decoder = [
            f"- {self._header(profile)}: {alias} = {raw}"
            for raw, alias in mapping.items()
        ]
        return self._make_candidate(
            table=table,
            profile=profile,
            selected=selected,
            requirements=requirements,
            operator="local_dictionary",
            display_header=self._header(profile),
            values=values,
            decoder_lines=decoder,
            reconstructed_values=reconstructed,
        )

    def _structured_template_candidate(
        self,
        table: Any,
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
    ) -> EncodingCandidate | None:
        """Factor repeated structure from exact identifiers and timestamps.

        A template such as ``TR{1}_{2}_{3}`` is stored once, while each row
        retains only its ordered variable slots.  The candidate is admitted
        only when every source string can be reconstructed byte-for-byte.
        """

        if len(selected) < 4:
            return None
        raw_values = list(selected.values())
        if any(
            not re.fullmatch(r"[A-Za-z0-9_./:-]+", raw)
            for raw in raw_values
        ):
            return None

        chunked = [
            re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]+", raw)
            for raw in raw_values
        ]
        if not chunked or any(len(parts) != len(chunked[0]) for parts in chunked):
            return None

        slots: list[tuple[str, str, str]] = []
        variable_count = 0
        factored_characters = 0
        for index in range(len(chunked[0])):
            parts = [row[index] for row in chunked]
            if all(not part.isalnum() for part in parts):
                if len(set(parts)) != 1:
                    return None
                slots.append(("literal", parts[0], ""))
                factored_characters += len(parts[0])
                continue
            if not all(part.isalnum() for part in parts):
                return None

            prefix = self._common_prefix(parts)
            remainders = [part[len(prefix):] for part in parts]
            suffix = self._common_suffix(remainders)
            variable_parts = [
                remainder[:-len(suffix)] if suffix else remainder
                for remainder in remainders
            ]
            if all(not part for part in variable_parts):
                literal = prefix + suffix
                slots.append(("literal", literal, ""))
                factored_characters += len(literal)
                continue
            slots.append(("variable", prefix, suffix))
            variable_count += 1
            factored_characters += len(prefix) + len(suffix)

        if (
            variable_count == 0
            or factored_characters < 2
            # Leave ordinary one-sided factoring to the simpler prefix/suffix
            # operators. A single-slot template is useful only for compact
            # identifiers such as TR{1}, where the alphanumeric boundary makes
            # implicit concatenation ambiguous.
            or (variable_count == 1 and len(slots) > 1)
        ):
            return None

        encoded: dict[str, str] = {}
        reconstructed: dict[str, str] = {}
        for (row_id, raw), parts in zip(selected.items(), chunked):
            variables: list[str] = []
            rebuilt: list[str] = []
            for part, (kind, prefix, suffix) in zip(parts, slots):
                if kind == "literal":
                    rebuilt.append(prefix)
                    continue
                end = len(part) - len(suffix) if suffix else len(part)
                variable = part[len(prefix):end]
                variables.append(variable)
                rebuilt.append(prefix + variable + suffix)
            encoded[row_id] = "|".join(variables)
            reconstructed[row_id] = "".join(rebuilt)

        template_parts: list[str] = []
        variable_index = 1
        for kind, prefix, suffix in slots:
            if kind == "literal":
                template_parts.append(prefix)
                continue
            template_parts.extend([prefix, f"{{{variable_index}}}", suffix])
            variable_index += 1
        template = "".join(template_parts)
        slot_word = "var" if variable_count == 1 else "vars"
        display_header = (
            f"{self._header(profile)} [template={template}; "
            f"{slot_word}={'|'.join(str(i) for i in range(1, variable_count + 1))}]"
        )
        return self._make_candidate(
            table=table,
            profile=profile,
            selected=selected,
            requirements=requirements,
            operator="structured_template",
            display_header=display_header,
            values=encoded,
            reconstructed_values=reconstructed,
        )

    def _affix_candidates(
        self,
        table: Any,
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
    ) -> list[EncodingCandidate]:
        """Factor a shared string prefix or suffix without losing source values."""

        if len(selected) < 2:
            return []
        raw_values = list(selected.values())
        result: list[EncodingCandidate] = []
        prefix = self._common_prefix(raw_values)
        prefix_remainders = [raw[len(prefix):] for raw in raw_values] if prefix else []
        prefix_boundary_safe = bool(prefix) and all(
            not (prefix[-1].isalnum() and remainder[:1].isalnum())
            for remainder in prefix_remainders
        )
        if prefix_boundary_safe and any(prefix_remainders):
            suffixes = {
                row_id: json.dumps(raw[len(prefix):], ensure_ascii=False)
                for row_id, raw in selected.items()
            }
            reconstructed = {
                row_id: prefix + str(json.loads(suffix))
                for row_id, suffix in suffixes.items()
            }
            result.append(self._make_candidate(
                table=table,
                profile=profile,
                selected=selected,
                requirements=requirements,
                operator="prefix_factorization",
                display_header=(
                    f"{self._header(profile)} = "
                    f"{json.dumps(prefix, ensure_ascii=False)} + suffix"
                ),
                values=suffixes,
                reconstructed_values=reconstructed,
            ))

        suffix = self._common_suffix(raw_values)
        suffix_stems = [raw[:-len(suffix)] for raw in raw_values] if suffix else []
        suffix_boundary_safe = bool(suffix) and all(
            not (stem[-1:].isalnum() and suffix[0].isalnum())
            for stem in suffix_stems
        )
        if (
            suffix_boundary_safe
            and suffix != prefix
            and any(suffix_stems)
        ):
            prefixes = {
                row_id: json.dumps(raw[:-len(suffix)], ensure_ascii=False)
                for row_id, raw in selected.items()
            }
            reconstructed = {
                row_id: str(json.loads(prefix_value)) + suffix
                for row_id, prefix_value in prefixes.items()
            }
            result.append(self._make_candidate(
                table=table,
                profile=profile,
                selected=selected,
                requirements=requirements,
                operator="suffix_factorization",
                display_header=(
                    f"{self._header(profile)} = prefix + "
                    f"{json.dumps(suffix, ensure_ascii=False)}"
                ),
                values=prefixes,
                reconstructed_values=reconstructed,
            ))
        return result

    def _exact_delta_candidate(
        self,
        table: Any,
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
    ) -> EncodingCandidate | None:
        parsed = {row_id: self._decimal(raw) for row_id, raw in selected.items()}
        numeric = {row_id: value for row_id, value in parsed.items() if value is not None}
        if len(numeric) < 2:
            return None
        nonnumeric = [row_id for row_id, value in parsed.items() if value is None]
        if nonnumeric and len(nonnumeric) > max(4, len(selected) // 5):
            return None

        base = min(numeric.values())
        base_text = self._format_decimal(base)
        values: dict[str, str] = {}
        reconstructed: dict[str, str] = {}
        residual_cells: list[dict[str, Any]] = []
        patch_lines: list[str] = []
        for row_id, raw in selected.items():
            number = parsed[row_id]
            if number is not None:
                delta = number - base
                delta_text = self._format_decimal(delta)
                values[row_id] = delta_text if delta < 0 else f"+{delta_text}"
                reconstructed[row_id] = self._format_decimal(base + Decimal(delta_text))
                continue
            values[row_id] = "see exact patch"
            residual = self._residual_cell(table, row_id, profile.col_id, raw, ["exact_value"])
            residual_cells.append(residual)
            patch_lines.append(self._patch_line(residual))
            reconstructed[row_id] = raw

        operator = "exact_delta+exact_patch" if patch_lines else "exact_delta"
        return self._make_candidate(
            table=table,
            profile=profile,
            selected=selected,
            requirements=requirements,
            operator=operator,
            display_header=f"{self._header(profile)} = {base_text} + delta",
            values=values,
            patch_lines=patch_lines,
            residual_cells=residual_cells,
            reconstructed_values=reconstructed,
        )

    def _anchored_candidate(
        self,
        table: Any,
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
    ) -> EncodingCandidate | None:
        mandatory = requirements["mandatory"]
        if "anchor_membership" not in mandatory:
            return None
        if mandatory - (STRUCTURAL_REQUIREMENTS | {"anchor_membership"}):
            return None
        anchors = self._anchors(requirements)
        present = set(selected.values())
        if not anchors or not set(anchors) <= present:
            return None
        if len(anchors) == 1:
            anchor = anchors[0]
            values = {
                row_id: "yes" if raw == anchor else "no"
                for row_id, raw in selected.items()
            }
            matches = sum(value == "yes" for value in values.values())
            others = len(values) - matches
            header = (
                f"{self._header(profile)} == {anchor}? "
                f"(yes={matches}, no={others})"
            )
            operator = "equality_predicate"
        else:
            values = {
                row_id: raw if raw in anchors else "other"
                for row_id, raw in selected.items()
            }
            header = f"{self._header(profile)} (requested identity)"
            operator = "anchored_classes"
        return self._make_candidate(
            table=table,
            profile=profile,
            selected=selected,
            requirements=requirements,
            operator=operator,
            display_header=header,
            values=values,
        )

    def _rank_candidate(
        self,
        table: Any,
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
    ) -> EncodingCandidate | None:
        mandatory = requirements["mandatory"]
        if "order" not in mandatory or mandatory - (STRUCTURAL_REQUIREMENTS | {"order"}):
            return None
        numbers = {row_id: parse_number(raw) for row_id, raw in selected.items()}
        if any(value is None for value in numbers.values()):
            return None
        distinct = sorted({float(value) for value in numbers.values()}, reverse=True)
        ranks = {value: index + 1 for index, value in enumerate(distinct)}
        values = {row_id: str(ranks[float(value)]) for row_id, value in numbers.items()}
        return self._make_candidate(
            table=table,
            profile=profile,
            selected=selected,
            requirements=requirements,
            operator="dense_rank",
            display_header=f"{self._header(profile)} rank (1 = highest)",
            values=values,
        )

    def _sparse_anchored_candidate(
        self,
        table: Any,
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
    ) -> EncodingCandidate | None:
        mandatory = requirements["mandatory"]
        if "anchor_membership" not in mandatory:
            return None
        if mandatory - (STRUCTURAL_REQUIREMENTS | {"anchor_membership"}):
            return None
        anchors = self._anchors(requirements)
        if not anchors or not set(anchors) <= set(selected.values()):
            return None
        if len(anchors) == 1:
            anchor = anchors[0]
            values = {
                row_id: "yes" if raw == anchor else "no"
                for row_id, raw in selected.items()
            }
            matches = [row_id for row_id, value in values.items() if value == "yes"]
            line = (
                f"{self._header(profile)} == {json.dumps(anchor, ensure_ascii=False)}"
                f" @ {self._row_set(matches)}; others=no"
            )
            operator = "sparse_equality"
        else:
            values = {
                row_id: raw if raw in anchors else "other"
                for row_id, raw in selected.items()
            }
            groups = []
            for anchor in anchors:
                members = [
                    row_id for row_id, value in selected.items() if value == anchor
                ]
                groups.append(
                    f"{json.dumps(anchor, ensure_ascii=False)}@{self._row_set(members)}"
                )
            line = f"{self._header(profile)} := " + "; ".join([*groups, "others=other"])
            operator = "sparse_anchored_classes"
        return self._make_candidate(
            table=table,
            profile=profile,
            selected=selected,
            requirements=requirements,
            operator=operator,
            display_header=self._header(profile),
            values=values,
            code_lines=[line],
            grid_visible=False,
        )

    def _grouped_rank_candidate(
        self,
        table: Any,
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
    ) -> EncodingCandidate | None:
        mandatory = requirements["mandatory"]
        if "order" not in mandatory or mandatory - (STRUCTURAL_REQUIREMENTS | {"order"}):
            return None
        numbers = {row_id: parse_number(raw) for row_id, raw in selected.items()}
        if any(value is None for value in numbers.values()):
            return None
        distinct = sorted({float(value) for value in numbers.values()}, reverse=True)
        ranks = {value: index + 1 for index, value in enumerate(distinct)}
        values = {row_id: str(ranks[float(value)]) for row_id, value in numbers.items()}
        groups = []
        for rank in range(1, len(distinct) + 1):
            members = [
                row_id for row_id, value in values.items() if int(value) == rank
            ]
            groups.append(f"{rank}@{self._row_set(members)}")
        return self._make_candidate(
            table=table,
            profile=profile,
            selected=selected,
            requirements=requirements,
            operator="grouped_rank",
            display_header=self._header(profile),
            values=values,
            code_lines=[
                f"{self._header(profile)} rank (1=highest) := " + "; ".join(groups)
            ],
            grid_visible=False,
        )

    def _threshold_candidate(
        self,
        table: Any,
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
    ) -> EncodingCandidate | None:
        mandatory = requirements["mandatory"]
        legal = STRUCTURAL_REQUIREMENTS | {
            "threshold_membership",
            "exact_value",
            "dirty_raw",
            "direct_value",
        }
        if "threshold_membership" not in mandatory or mandatory - legal:
            return None
        threshold_specs = {
            (float(item["boundary"]), str(item.get("operator", "ge") or "ge"))
            for item in requirements["properties_by_kind"].get("threshold_membership", [])
            if item.get("boundary") is not None
        }
        if len(threshold_specs) != 1:
            return None
        boundary, comparison = next(iter(threshold_specs))
        if comparison not in {"lt", "le", "gt", "ge"}:
            return None
        numbers = {row_id: parse_number(raw) for row_id, raw in selected.items()}
        if any(value is None for value in numbers.values()):
            return None
        compare = {
            "lt": lambda value: value < boundary,
            "le": lambda value: value <= boundary,
            "gt": lambda value: value > boundary,
            "ge": lambda value: value >= boundary,
        }[comparison]
        display_operator = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}[comparison]
        values = {
            row_id: "yes" if compare(float(number)) else "no"
            for row_id, number in numbers.items()
        }
        required_patches = self._required_patch_rows(requirements) & set(selected)
        residual_cells = [
            self._residual_cell(
                table,
                row_id,
                profile.col_id,
                selected[row_id],
                [
                    kind
                    for kind in ("exact_value", "dirty_raw")
                    if row_id in requirements["rows_by_kind"].get(kind, set())
                ],
            )
            for row_id in selected
            if row_id in required_patches
        ]
        patch_lines = [self._patch_line(item) for item in residual_cells]
        operator = "threshold_predicate+exact_patch" if patch_lines else "threshold_predicate"
        return self._make_candidate(
            table=table,
            profile=profile,
            selected=selected,
            requirements=requirements,
            operator=operator,
            display_header=f"{self._header(profile)} {display_operator} {format_number(boundary)}",
            values=values,
            patch_lines=patch_lines,
            residual_cells=residual_cells,
        )

    def _sparse_threshold_candidate(
        self,
        table: Any,
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
    ) -> EncodingCandidate | None:
        mandatory = requirements["mandatory"]
        legal = STRUCTURAL_REQUIREMENTS | {
            "threshold_membership", "exact_value", "dirty_raw", "direct_value",
        }
        if "threshold_membership" not in mandatory or mandatory - legal:
            return None
        threshold_specs = {
            (float(item["boundary"]), str(item.get("operator", "ge") or "ge"))
            for item in requirements["properties_by_kind"].get("threshold_membership", [])
            if item.get("boundary") is not None
        }
        if len(threshold_specs) != 1:
            return None
        boundary, comparison = next(iter(threshold_specs))
        compare = {
            "lt": lambda value: value < boundary,
            "le": lambda value: value <= boundary,
            "gt": lambda value: value > boundary,
            "ge": lambda value: value >= boundary,
        }.get(comparison)
        if compare is None:
            return None
        numbers = {row_id: parse_number(raw) for row_id, raw in selected.items()}
        if any(value is None for value in numbers.values()):
            return None
        values = {
            row_id: "yes" if compare(float(number)) else "no"
            for row_id, number in numbers.items()
        }
        required_patches = self._required_patch_rows(requirements) & set(selected)
        residual_cells = [
            self._residual_cell(
                table,
                row_id,
                profile.col_id,
                selected[row_id],
                [
                    kind for kind in ("exact_value", "dirty_raw")
                    if row_id in requirements["rows_by_kind"].get(kind, set())
                ],
            )
            for row_id in selected
            if row_id in required_patches
        ]
        patch_lines = [self._patch_line(item) for item in residual_cells]
        display_operator = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}[comparison]
        matches = [row_id for row_id, value in values.items() if value == "yes"]
        operator = "sparse_threshold+exact_patch" if patch_lines else "sparse_threshold"
        return self._make_candidate(
            table=table,
            profile=profile,
            selected=selected,
            requirements=requirements,
            operator=operator,
            display_header=self._header(profile),
            values=values,
            code_lines=[
                f"{self._header(profile)} {display_operator} {format_number(boundary)}"
                f" @ {self._row_set(matches)}; others=no"
            ],
            patch_lines=patch_lines,
            residual_cells=residual_cells,
            grid_visible=False,
        )

    def _materialize_evidence_graph(
        self,
        *,
        table: Any,
        row_ids: list[str],
        source_order: list[str],
        observables: list[DerivedObservable],
    ) -> tuple[
        dict[str, dict[str, str]],
        dict[str, str],
        dict[str, dict[str, tuple[str, ...]]],
    ]:
        """Evaluate a validated expression DAG while retaining source lineage."""

        node_values: dict[str, dict[str, str]] = {}
        node_headers: dict[str, str] = {}
        node_lineage: dict[str, dict[str, tuple[str, ...]]] = {}
        for col_id in source_order:
            col_idx = int(col_id[1:])
            node_values[col_id] = {
                row_id: str(table.rows[self._row_idx(row_id)][col_idx])
                for row_id in row_ids
            }
            node_headers[col_id] = (
                " / ".join(table.header_path(col_idx)) or table.headers[col_idx]
            )
            node_lineage[col_id] = {
                row_id: (f"{table.table_id}:{row_id}:{col_id}",)
                for row_id in row_ids
            }

        for observable in observables:
            raw_by_row = {
                row_id: {
                    operand.column: node_values[operand.column][row_id]
                    for operand in observable.operands
                }
                for row_id in row_ids
            }
            node_values[observable.observable_id] = self._evaluate_derived_observable(
                observable=observable,
                raw_by_row=raw_by_row,
            )
            node_headers[observable.observable_id] = self._derived_expression_text(
                observable, node_headers
            )
            node_lineage[observable.observable_id] = {
                row_id: tuple(dict.fromkeys(
                    source_ref
                    for operand in observable.operands
                    for source_ref in node_lineage[operand.column][row_id]
                ))
                for row_id in row_ids
            }
        return node_values, node_headers, node_lineage

    def _generate_derived_candidates(
        self,
        *,
        table: Any,
        row_ids: list[str],
        observable: DerivedObservable,
        node_values: dict[str, dict[str, str]],
        node_headers: dict[str, str],
        node_lineage: dict[str, dict[str, tuple[str, ...]]],
        source_node_ids: set[str],
    ) -> list[EncodingCandidate]:
        """Materialize one row-bound expression and its legal physical forms."""

        operand_nodes = list(dict.fromkeys(
            operand.column for operand in observable.operands
        ))
        raw_by_row = {
            row_id: {
                node_id: node_values[node_id][row_id]
                for node_id in operand_nodes
            }
            for row_id in row_ids
        }
        values = node_values[observable.observable_id]
        expression = node_headers[observable.observable_id]
        lineage = node_lineage[observable.observable_id]
        raw_headers = tuple(
            f"{node_id}:{node_headers[node_id]}" for node_id in operand_nodes
        )
        raw_render_values = {
            row_id: tuple(raw_by_row[row_id][node_id] for node_id in operand_nodes)
            for row_id in row_ids
        }
        raw_fragment = "\n".join([
            *raw_headers,
            *(value for row_id in row_ids for value in raw_render_values[row_id]),
            expression,
        ])
        raw_tokens = max(1, self.count_tokens(raw_fragment))
        requirement = f"derived_{observable.mode}"
        source_columns = tuple(dict.fromkeys(
            source_ref.rsplit(":", 1)[-1]
            for row_id in row_ids
            for source_ref in lineage[row_id]
        ))

        def make(
            *,
            operator: str,
            display_header: str,
            encoded: dict[str, str],
            render_headers: tuple[str, ...] | None = None,
            render_values: dict[str, tuple[str, ...]] | None = None,
            decoder_lines: list[str] | None = None,
            source_reversible: bool = False,
            dependencies: tuple[tuple[str, str], ...] = (),
            grid_visible: bool = True,
        ) -> EncodingCandidate:
            checks = self._certify_derived(
                observable=observable,
                expected=values,
                encoded=encoded,
                operator=operator,
                source_reversible=source_reversible,
                dependency_reconstructable=bool(dependencies),
            )
            decoder_lines = list(decoder_lines or [])
            shown_headers = (
                render_headers
                if render_headers is not None
                else ((display_header,) if grid_visible else ())
            )
            shown_values = (
                render_values
                if render_values is not None
                else (
                    {row_id: (encoded[row_id],) for row_id in row_ids}
                    if grid_visible else {}
                )
            )
            fragment = "\n".join([
                *shown_headers,
                *(value for row_id in row_ids for value in shown_values[row_id]),
                *decoder_lines,
            ]) if grid_visible else "\n".join(decoder_lines)
            lexical_cost, byte_cost = self._cost_vector(fragment)
            intended_property = {
                "derive_from_operands": "derived_exact",
                "derived_exact": "derived_exact",
                "derived_dense_rank": "derived_order",
                "derived_threshold": "derived_threshold",
            }[operator]
            certified = (
                bool(checks.get(intended_property))
                and checks.get("row_binding", False)
                and checks.get("derived_expression", False)
            )
            return EncodingCandidate(
                col_id=observable.observable_id,
                source_header=expression,
                display_header=display_header,
                operator=operator,
                values=encoded,
                token_cost=(
                    0 if self.has_exact_token_counter
                    else max(1, self.count_tokens(fragment))
                ),
                lexical_cost=lexical_cost,
                byte_cost=byte_cost,
                grid_visible=grid_visible,
                shared_sections=frozenset({"decoder"}) if decoder_lines else frozenset(),
                decoder_lines=decoder_lines,
                covered_requirements=sorted(
                    property_name
                    for property_name in (
                        "derived_exact", "derived_order", "derived_threshold"
                    )
                    if checks.get(property_name)
                ),
                certification_checks=checks,
                certified=certified,
                source_reversible=source_reversible,
                raw_equivalent_tokens=0 if dependencies else raw_tokens,
                source_col_ids=source_columns,
                render_headers=shown_headers,
                render_values=shown_values,
                derived_expression=expression,
                lineage_by_row=lineage,
                required_properties=frozenset({requirement}),
                provided_properties=frozenset(
                    kind for kind, passed in checks.items() if passed
                ),
                dependencies=dependencies,
                node_kind="expression",
            )

        operand_dependencies = tuple(
            (
                node_id,
                "source_reconstruction"
                if node_id in source_node_ids
                else "derived_exact",
            )
            for node_id in operand_nodes
        )
        candidates = [make(
            operator="derive_from_operands",
            display_header=f"{observable.observable_id}:{expression}",
            encoded=values,
            decoder_lines=[
                f"{observable.observable_id} = {expression} (aligned by source_row)"
            ],
            dependencies=operand_dependencies,
            grid_visible=False,
        )]
        if self.encoding_mode == "raw":
            return candidates

        candidates.append(make(
            operator="derived_exact",
            display_header=f"{observable.observable_id}:{expression}",
            encoded=values,
        ))
        if observable.mode == "order":
            numeric = {row_id: Decimal(value) for row_id, value in values.items()}
            reverse = observable.direction == "descending"
            distinct = sorted(set(numeric.values()), reverse=reverse)
            ranks = {value: index + 1 for index, value in enumerate(distinct)}
            encoded = {row_id: str(ranks[value]) for row_id, value in numeric.items()}
            direction = "highest" if reverse else "lowest"
            candidates.append(make(
                operator="derived_dense_rank",
                display_header=(
                    f"{observable.observable_id}:{expression} rank "
                    f"(1={direction}; ties share rank)"
                ),
                encoded=encoded,
            ))
        elif observable.mode == "threshold" and observable.boundary is not None:
            boundary = Decimal(str(observable.boundary))
            comparison = observable.threshold_operator or "ge"
            compare = {
                "lt": lambda value: value < boundary,
                "le": lambda value: value <= boundary,
                "gt": lambda value: value > boundary,
                "ge": lambda value: value >= boundary,
            }[comparison]
            encoded = {
                row_id: "yes" if compare(Decimal(value)) else "no"
                for row_id, value in values.items()
            }
            candidates.append(make(
                operator="derived_threshold",
                display_header=(
                    f"{observable.observable_id}:{expression} {comparison} "
                    f"{self._format_decimal(boundary)}"
                ),
                encoded=encoded,
            ))
        return candidates

    def _evaluate_derived_observable(
        self,
        *,
        observable: DerivedObservable,
        raw_by_row: dict[str, dict[str, str]],
    ) -> dict[str, str]:
        grouped: dict[str, list[Any]] = {}
        for operand in observable.operands:
            grouped.setdefault(operand.role, []).append(operand)

        def value(row_id: str, role: str) -> Decimal:
            operands = grouped.get(role, [])
            if not operands:
                raise ValueError(
                    f"Derived observable {observable.observable_id} has no {role} operand"
                )
            result = Decimal(0)
            for operand in operands:
                number = self._semantic_decimal(raw_by_row[row_id][operand.column])
                if number is None:
                    raise ValueError(
                        f"Non-numeric operand for {observable.observable_id} at "
                        f"{row_id}:{operand.column}"
                    )
                result += abs(number) if operand.value_mode == "magnitude" else number
            return result

        def all_values(row_id: str) -> list[Decimal]:
            result = []
            for operand in observable.operands:
                number = self._semantic_decimal(raw_by_row[row_id][operand.column])
                if number is None:
                    raise ValueError(
                        f"Non-numeric operand for {observable.observable_id} at "
                        f"{row_id}:{operand.column}"
                    )
                result.append(
                    abs(number) if operand.value_mode == "magnitude" else number
                )
            if not result:
                raise ValueError(
                    f"Derived observable {observable.observable_id} has no operands"
                )
            return result

        output: dict[str, str] = {}
        for row_id in raw_by_row:
            if observable.operator == "difference":
                left_role = "later" if grouped.get("later") else "left"
                right_role = "earlier" if grouped.get("earlier") else "right"
                derived = value(row_id, left_role) - value(row_id, right_role)
            elif observable.operator == "ratio":
                denominator = value(row_id, "denominator")
                if denominator == 0:
                    raise ValueError(
                        f"Zero denominator for {observable.observable_id} at {row_id}"
                    )
                derived = value(row_id, "numerator") / denominator
            elif observable.operator == "percent_change":
                earlier = value(row_id, "earlier")
                if earlier == 0:
                    raise ValueError(
                        f"Zero earlier value for {observable.observable_id} at {row_id}"
                    )
                derived = (value(row_id, "later") - earlier) / earlier
            else:
                values = all_values(row_id)
                if observable.operator == "sum":
                    derived = sum(values, Decimal(0))
                elif observable.operator == "mean":
                    derived = sum(values, Decimal(0)) / Decimal(len(values))
                elif observable.operator == "min":
                    derived = min(values)
                elif observable.operator == "max":
                    derived = max(values)
                elif observable.operator == "range":
                    derived = max(values) - min(values)
                elif observable.operator == "magnitude":
                    derived = abs(sum(values, Decimal(0)))
                else:
                    raise ValueError(
                        f"Unsupported derived operator {observable.operator!r}"
                    )
            derived *= Decimal(str(observable.scale))
            output[row_id] = self._format_decimal(derived)
        return output

    def _derived_expression_text(
        self,
        observable: DerivedObservable,
        headers: dict[str, str],
    ) -> str:
        by_role: dict[str, list[str]] = {}
        for operand in observable.operands:
            label = headers[operand.column]
            if operand.value_mode == "magnitude":
                label = f"abs({label})"
            by_role.setdefault(operand.role, []).append(label)

        def group(*roles: str) -> str:
            values = [item for role in roles for item in by_role.get(role, [])]
            return " + ".join(values)

        if observable.operator == "difference":
            return f"{group('later', 'left')} - {group('earlier', 'right')}"
        if observable.operator == "ratio":
            return f"({group('numerator')}) / ({group('denominator')})"
        if observable.operator == "percent_change":
            return (
                f"(({group('later')}) - ({group('earlier')})) / "
                f"({group('earlier')}) * "
                f"{self._format_decimal(Decimal(str(observable.scale)))}"
            )
        values = group("value", "input", "group", "rank", "left", "right")
        if observable.operator == "sum":
            return values
        if observable.operator == "mean":
            return f"mean({values})"
        if observable.operator in {"min", "max", "range"}:
            return f"{observable.operator}({values})"
        if observable.operator == "magnitude":
            return f"abs({values})"
        raise ValueError(f"Unsupported derived operator {observable.operator!r}")

    def _certify_derived(
        self,
        *,
        observable: DerivedObservable,
        expected: dict[str, str],
        encoded: dict[str, str],
        operator: str,
        source_reversible: bool,
        dependency_reconstructable: bool = False,
    ) -> dict[str, bool]:
        expected_numbers = {row_id: Decimal(value) for row_id, value in expected.items()}
        checks = {
            "row_binding": set(encoded) == set(expected),
            "derived_expression": bool(observable.operands),
            "source_reconstruction": source_reversible,
            "dependency_reconstruction": dependency_reconstructable,
        }
        reconstructable = (
            (operator == "derived_exact" and encoded == expected)
            or source_reversible
            or dependency_reconstructable
        )
        checks["derived_exact"] = reconstructable
        if operator == "derived_dense_rank":
            ranks = {row_id: int(value) for row_id, value in encoded.items()}
            direction = -1 if observable.direction == "descending" else 1
            checks["derived_order"] = all(
                self._relation(expected_numbers[left], expected_numbers[right])
                == direction * self._relation(ranks[left], ranks[right])
                for left in expected
                for right in expected
            )
        else:
            checks["derived_order"] = reconstructable

        if observable.boundary is not None:
            boundary = Decimal(str(observable.boundary))
            comparison = observable.threshold_operator or "ge"
            compare = {
                "lt": lambda value: value < boundary,
                "le": lambda value: value <= boundary,
                "gt": lambda value: value > boundary,
                "ge": lambda value: value >= boundary,
            }[comparison]
            checks["derived_threshold"] = all(
                encoded[row_id] == ("yes" if compare(value) else "no")
                for row_id, value in expected_numbers.items()
            ) if operator == "derived_threshold" else (
                reconstructable
            )
        return checks

    def _make_candidate(
        self,
        *,
        table: Any,
        profile: ColumnProfile,
        selected: dict[str, str],
        requirements: dict[str, Any],
        operator: str,
        display_header: str,
        values: dict[str, str],
        code_lines: list[str] | None = None,
        grid_visible: bool = True,
        decoder_lines: list[str] | None = None,
        patch_lines: list[str] | None = None,
        residual_cells: list[dict[str, Any]] | None = None,
        reconstructed_values: dict[str, str] | None = None,
    ) -> EncodingCandidate:
        code_lines = list(code_lines or [])
        decoder_lines = list(decoder_lines or [])
        patch_lines = list(patch_lines or [])
        residual_cells = list(residual_cells or [])
        reconstructed_values = dict(reconstructed_values or {})
        source_reversible = reconstructed_values == selected
        shared_sections = frozenset(
            section
            for section, lines in (
                ("codes", code_lines),
                ("decoder", decoder_lines),
                ("patches", patch_lines),
            )
            if lines
        )
        checks = self._certify(
            operator=operator,
            selected=selected,
            encoded=values,
            requirements=requirements,
            residual_cells=residual_cells,
            decoder_lines=decoder_lines,
            source_reversible=source_reversible,
        )
        mandatory = requirements["mandatory"]
        checks["rendering_complete"] = bool(grid_visible or code_lines)
        certified = (
            mandatory <= set(checks)
            and all(checks[kind] for kind in mandatory)
            and checks["rendering_complete"]
        )
        visible_fragment = [display_header, *values.values()] if grid_visible else code_lines
        fragment = "\n".join([
            *visible_fragment,
            *decoder_lines,
            *patch_lines,
        ])
        raw_key = (
            f"{table.table_id}:{profile.col_id}:{self._header(profile)}",
            tuple(selected.items()),
        )
        raw_equivalent_tokens = self._raw_equivalent_cost_cache.get(raw_key)
        if raw_equivalent_tokens is None:
            raw_fragment = "\n".join([self._header(profile), *selected.values()])
            raw_equivalent_tokens = max(1, self.count_tokens(raw_fragment))
            self._raw_equivalent_cost_cache[raw_key] = raw_equivalent_tokens
        lexical_cost, byte_cost = self._cost_vector(fragment)
        return EncodingCandidate(
            col_id=profile.col_id,
            source_header=self._header(profile),
            display_header=display_header,
            operator=operator,
            values=values,
            token_cost=(
                0 if self.has_exact_token_counter
                else max(1, self.count_tokens(fragment))
            ),
            lexical_cost=lexical_cost,
            byte_cost=byte_cost,
            grid_visible=grid_visible,
            shared_sections=shared_sections,
            code_lines=code_lines,
            decoder_lines=decoder_lines,
            patch_lines=patch_lines,
            residual_cells=residual_cells,
            covered_requirements=sorted(mandatory) if certified else [
                kind for kind in sorted(mandatory) if checks.get(kind)
            ],
            certification_checks=checks,
            certified=certified,
            source_reversible=source_reversible,
            raw_equivalent_tokens=raw_equivalent_tokens,
            source_col_ids=(profile.col_id,),
            render_headers=(display_header,),
            render_values={row_id: (value,) for row_id, value in values.items()},
            required_properties=frozenset(mandatory),
            provided_properties=frozenset(
                kind for kind, passed in checks.items() if passed
            ),
            node_kind="source",
        )

    def _certify(
        self,
        *,
        operator: str,
        selected: dict[str, str],
        encoded: dict[str, str],
        requirements: dict[str, Any],
        residual_cells: list[dict[str, Any]],
        decoder_lines: list[str],
        source_reversible: bool,
    ) -> dict[str, bool]:
        mandatory = requirements["mandatory"]
        checks = {kind: True for kind in mandatory & STRUCTURAL_REQUIREMENTS}
        checks["source_reconstruction"] = source_reversible
        residual_rows = {str(item["row_id"]) for item in residual_cells}

        if "exact_value" in mandatory:
            exact_rows = requirements["rows_by_kind"].get("exact_value", set())
            checks["exact_value"] = source_reversible or exact_rows <= residual_rows
        if "dirty_raw" in mandatory:
            dirty_rows = requirements["rows_by_kind"].get("dirty_raw", set())
            checks["dirty_raw"] = source_reversible or dirty_rows <= residual_rows
        if "direct_value" in mandatory:
            direct_rows = requirements["rows_by_kind"].get("direct_value", set())
            patched_rows = {
                str(cell.get("row_id", ""))
                for cell in residual_cells
                if str(cell.get("raw_value", ""))
                == str(selected.get(str(cell.get("row_id", "")), ""))
            }
            checks["direct_value"] = all(
                encoded.get(row_id) == selected.get(row_id) or row_id in patched_rows
                for row_id in direct_rows
            )
        if "identity_recoverable" in mandatory:
            checks["identity_recoverable"] = source_reversible
        if "equality" in mandatory:
            raw_to_encoded: dict[str, set[str]] = {}
            encoded_to_raw: dict[str, set[str]] = {}
            for row_id, raw in selected.items():
                coded = encoded[row_id]
                raw_to_encoded.setdefault(raw, set()).add(coded)
                encoded_to_raw.setdefault(coded, set()).add(raw)
            checks["equality"] = source_reversible or (
                all(len(values) == 1 for values in raw_to_encoded.values())
                and all(len(values) == 1 for values in encoded_to_raw.values())
            )
        if "anchor_membership" in mandatory:
            anchors = self._anchors(requirements)
            if operator in {"equality_predicate", "sparse_equality"} and len(anchors) == 1:
                anchor = anchors[0]
                checks["anchor_membership"] = all(
                    (encoded[row_id] == "yes") == (raw == anchor)
                    for row_id, raw in selected.items()
                )
            elif operator in {"anchored_classes", "sparse_anchored_classes"} and anchors:
                checks["anchor_membership"] = all(
                    encoded[row_id] == (raw if raw in anchors else "other")
                    for row_id, raw in selected.items()
                )
            else:
                checks["anchor_membership"] = source_reversible and bool(anchors)
        if "order" in mandatory:
            if source_reversible:
                checks["order"] = True
            elif operator in {"dense_rank", "grouped_rank"}:
                parsed = {
                    row_id: parse_number(raw) for row_id, raw in selected.items()
                }
                distinct = sorted(
                    {float(value) for value in parsed.values() if value is not None},
                    reverse=True,
                )
                expected = {value: index + 1 for index, value in enumerate(distinct)}
                checks["order"] = (
                    len(distinct) > 0
                    and all(value is not None for value in parsed.values())
                    and all(
                        int(encoded[row_id]) == expected[float(value)]
                        for row_id, value in parsed.items()
                        if value is not None
                    )
                )
            else:
                checks["order"] = False
        if "threshold_membership" in mandatory:
            threshold_specs = {
                (float(item["boundary"]), str(item.get("operator", "ge") or "ge"))
                for item in requirements["properties_by_kind"].get("threshold_membership", [])
                if item.get("boundary") is not None
            }
            if source_reversible:
                checks["threshold_membership"] = len(threshold_specs) == 1
            elif (
                operator.startswith("threshold_predicate")
                or operator.startswith("sparse_threshold")
            ) and len(threshold_specs) == 1:
                boundary, comparison = next(iter(threshold_specs))
                compare = {
                    "lt": lambda value: value < boundary,
                    "le": lambda value: value <= boundary,
                    "gt": lambda value: value > boundary,
                    "ge": lambda value: value >= boundary,
                }.get(comparison)
                checks["threshold_membership"] = all(
                    parse_number(raw) is not None
                    and compare is not None
                    and encoded[row_id] == ("yes" if compare(float(parse_number(raw))) else "no")
                    for row_id, raw in selected.items()
                )
            else:
                checks["threshold_membership"] = False

        if operator == "local_dictionary":
            checks["decoder_complete"] = bool(decoder_lines) and len(set(encoded.values())) == len(
                set(selected.values())
            )
            if not checks["decoder_complete"]:
                for kind in mandatory - STRUCTURAL_REQUIREMENTS:
                    checks[kind] = False
        return checks

    def _deduplicate_candidates(
        self,
        candidates: list[EncodingCandidate],
    ) -> list[EncodingCandidate]:
        """Remove exact duplicates without pruning distinct renderings.

        A candidate that is shorter in isolation need not yield the shortest
        complete witness after shared sections and canonical rendering are
        applied.  Distinct certified candidates therefore remain available
        until complete plans are rendered with the configured tokenizer.
        """

        unique: list[EncodingCandidate] = []
        for candidate in candidates:
            if candidate in unique:
                continue
            unique.append(candidate)
        return sorted(
            unique,
            key=lambda item: (
                len(item.shared_sections),
                sorted(item.shared_sections),
                item.operator,
                item.col_id,
            ),
        )

    def _candidate_trace(
        self,
        candidate: EncodingCandidate,
        requirements: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mandatory = (
            set(requirements["mandatory"])
            if requirements is not None
            else set(candidate.required_properties)
        )
        return {
            "operator": candidate.operator,
            "realization_family": self._realization_family(candidate),
            "display_header": candidate.display_header,
            "token_cost": candidate.token_cost,
            "append_token_cost": candidate.append_token_cost,
            "lexical_cost": candidate.lexical_cost,
            "utf8_bytes": candidate.byte_cost,
            "reader_token_savings": candidate.raw_equivalent_tokens - candidate.token_cost,
            "shared_sections": sorted(candidate.shared_sections),
            "grid_visible": candidate.grid_visible,
            "code_lines": candidate.code_lines,
            "decoder_lines": candidate.decoder_lines,
            "patch_lines": candidate.patch_lines,
            "covered_requirements": candidate.covered_requirements,
            "mandatory_requirements": sorted(mandatory),
            "provided_properties": sorted(candidate.provided_properties),
            "dependencies": [
                {"node": node_id, "property": property_name}
                for node_id, property_name in candidate.dependencies
            ],
            "node_kind": candidate.node_kind,
            "covered_node_ids": list(
                candidate.covered_node_ids or (candidate.col_id,)
            ),
            "required_properties_by_node": {
                node_id: list(properties)
                for node_id, properties in candidate.required_properties_by_node
            },
            "residual_cells": candidate.residual_cells,
            "certified": candidate.certified,
            "source_reversible": candidate.source_reversible,
            "raw_equivalent_tokens": candidate.raw_equivalent_tokens,
            "certification_checks": candidate.certification_checks,
            "source_col_ids": list(candidate.source_col_ids),
            "render_headers": list(candidate.render_headers),
            "derived_expression": candidate.derived_expression,
            "lineage_by_row": {
                row_id: list(lineage)
                for row_id, lineage in candidate.lineage_by_row.items()
            },
            "visible_values": {
                row_id: list(values)
                for row_id, values in candidate.render_values.items()
            },
        }

    @staticmethod
    def _realization_family(candidate: EncodingCandidate) -> str:
        """Expose the three semantic realization families used by the planner."""

        if candidate.operator == "derive_from_operands":
            return "derive"
        if candidate.source_reversible or candidate.operator == "derived_exact":
            return "value"
        return "property"

    def _required_patch_rows(self, requirements: dict[str, Any]) -> set[str]:
        result: set[str] = set()
        for kind in ("exact_value", "dirty_raw"):
            result.update(requirements["rows_by_kind"].get(kind, set()))
        return result

    def _anchors(self, requirements: dict[str, Any]) -> list[str]:
        return list(dict.fromkeys(
            str(anchor)
            for spec in requirements["properties_by_kind"].get("anchor_membership", [])
            for anchor in spec.get("anchors", [])
        ))

    def _residual_cell(
        self,
        table: Any,
        row_id: str,
        col_id: str,
        raw_value: str,
        requirements: list[str],
    ) -> dict[str, Any]:
        return {
            "table_id": table.table_id,
            "row_id": row_id,
            "col_id": col_id,
            "raw_value": raw_value,
            "source_id": f"{table.table_id}:{row_id}:{col_id}",
            "requirements": requirements,
        }

    def _patch_line(self, item: dict[str, Any]) -> str:
        return (
            f"- {item['row_id']}, {item['col_id']} = {item['raw_value']} "
            f"[source={item['source_id']}]"
        )

    def _header(self, profile: ColumnProfile) -> str:
        return " / ".join(profile.header_path) or profile.header

    def _ordered_distinct(self, values: Iterable[str]) -> list[str]:
        counts = Counter(str(value) for value in values)
        return [
            value
            for value, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]

    @staticmethod
    def _row_set(row_ids: Iterable[str]) -> str:
        """Render ordered row IDs and collapse consecutive runs."""

        ids = [str(row_id) for row_id in row_ids]
        if not ids:
            return "[]"
        parsed: list[int] = []
        for row_id in ids:
            match = re.fullmatch(r"r(\d+)", row_id)
            if match is None:
                return "[" + ",".join(ids) + "]"
            parsed.append(int(match.group(1)))
        chunks: list[str] = []
        start = previous = parsed[0]
        for value in parsed[1:]:
            if value == previous + 1:
                previous = value
                continue
            chunks.append(f"r{start}-r{previous}" if start != previous else f"r{start}")
            start = previous = value
        chunks.append(f"r{start}-r{previous}" if start != previous else f"r{start}")
        return "[" + ",".join(chunks) + "]"

    @staticmethod
    def _compact_row_set(row_ids: Iterable[str]) -> str:
        """Render source rows once under an explicit r-index namespace."""

        ids = [str(row_id) for row_id in row_ids]
        if not ids:
            return "r[]"
        parsed: list[int] = []
        for row_id in ids:
            match = re.fullmatch(r"r(\d+)", row_id)
            if match is None:
                return "[" + ",".join(ids) + "]"
            parsed.append(int(match.group(1)))
        chunks: list[str] = []
        start = previous = parsed[0]
        for value in parsed[1:]:
            if value == previous + 1:
                previous = value
                continue
            chunks.append(f"{start}-{previous}" if start != previous else str(start))
            start = previous = value
        chunks.append(f"{start}-{previous}" if start != previous else str(start))
        return "r[" + ",".join(chunks) + "]"

    @staticmethod
    def _common_prefix(values: list[str]) -> str:
        if not values:
            return ""
        first, last = min(values), max(values)
        index = 0
        while index < min(len(first), len(last)) and first[index] == last[index]:
            index += 1
        return first[:index]

    @staticmethod
    def _common_suffix(values: list[str]) -> str:
        reversed_prefix = EvidencePlanOptimizer._common_prefix([
            value[::-1] for value in values
        ])
        return reversed_prefix[::-1]

    def _decimal(self, raw: str) -> Decimal | None:
        text = str(raw).strip()
        if not CANONICAL_DECIMAL.fullmatch(text):
            return None
        try:
            value = Decimal(text)
        except InvalidOperation:
            return None
        # Formatting is part of semi-structured table meaning. Delta coding is
        # admitted only when its canonical reconstruction exactly matches the
        # visible source scalar; formatted values remain Raw or exact-patched.
        return value if self._format_decimal(value) == text else None

    @staticmethod
    def _semantic_decimal(raw: Any) -> Decimal | None:
        """Parse a complete table scalar into Decimal without a float round-trip."""

        text = str(raw if raw is not None else "").strip()
        if not text:
            return None
        accounting_negative = text.startswith("(") and text.endswith(")")
        if accounting_negative:
            text = text[1:-1].strip()
        match = SEMANTIC_DECIMAL.fullmatch(text)
        if match is None:
            return None
        normalized = re.sub(r"[,\s\u00a0\u202f]", "", match.group("number"))
        try:
            value = Decimal(normalized)
        except InvalidOperation:
            return None
        if not value.is_finite():
            return None
        return -abs(value) if accounting_negative else value

    def _format_decimal(self, value: Decimal) -> str:
        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    def _relation(self, left: Any, right: Any) -> int:
        if left is None or right is None:
            return 99
        return (left > right) - (left < right)

    def _row_idx(self, row_id: str) -> int:
        return int(row_id[1:] if str(row_id).startswith("r") else row_id)

    @staticmethod
    def _cost_vector(text: str) -> tuple[int, int]:
        return len(tokens(text)), len(str(text).encode("utf-8"))

    @staticmethod
    def _estimated_tokens(lexical_cost: int, byte_cost: int) -> int:
        return max(int(lexical_cost), math.ceil(int(byte_cost) / 4))

    @staticmethod
    def _dominant_cost_order(lexical_cost: int, byte_cost: int) -> tuple[int, int]:
        byte_tokens = math.ceil(int(byte_cost) / 4)
        if byte_tokens >= int(lexical_cost):
            return int(byte_cost), int(lexical_cost)
        return int(lexical_cost), int(byte_cost)

    def _insert_best_state(self, frontier, candidate) -> None:
        """Merge only identical logical and rendering continuation states.

        The surrounding map supplies obligations and active sections. This
        key retains ordered candidate content, including provenance and layout,
        so adding any legal suffix renders identically for merged prefixes.
        Distinct text is never pruned on an additive token estimate.
        """
        table, order, rows = self._search_render_context
        selected = candidate[4]
        rendering_key = json.dumps(
            [asdict(item) for item in selected], sort_keys=True,
            ensure_ascii=False, default=lambda value: sorted(value),
        )
        bucket = self._rendering_frontiers.setdefault(id(frontier), (frontier, {}))[1]
        cost = self._rendered_body_token_cost(
            self._render_selected(table, order, rows, selected)
        )
        candidate = (cost, *candidate[1:])
        index = bucket.get(rendering_key)
        if index is None:
            bucket[rendering_key] = len(frontier)
            frontier.append(candidate)
        elif (candidate[0], candidate[3]) < (frontier[index][0], frontier[index][3]):
            frontier[index] = candidate
