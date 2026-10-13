from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from compresstab.compress import CompressTab
from compresstab.codec.evidence import (
    CANONICAL_CELL_SEPARATOR,
    CANONICAL_LINE_SEPARATOR,
    EvidencePlanOptimizer,
)
from compresstab.artifacts import prepared_evidence_identity
from compresstab.evidence import AnswerabilitySignature
import compresstab.profile as profile_module
from compresstab.profile import ProfileStore, table_fingerprint
from compresstab.table.artifact import artifact_from_dataset_row
from compresstab.table.store import TableStore


def make_compresstab(
    rows,
    *,
    profile_cache_dir=None,
    evidence_coding="optimized",
    optimization_strategy="global_dp",
    token_counter=None,
    joint_realizations=True,
):
    artifact = artifact_from_dataset_row({"question_id": "core-test", "table": rows})
    return CompressTab(
        TableStore(artifact),
        profile_cache_dir=profile_cache_dir,
        evidence_coding=evidence_coding,
        optimization_strategy=optimization_strategy,
        token_counter=token_counter,
        joint_realizations=joint_realizations,
    )


def make_signature(columns, resolution, *, rows=None, links=None):
    return AnswerabilitySignature.model_validate({"slices": [{
        "source": "t0",
        "columns": columns,
        "rows": list(rows or []),
        "links": links if links is not None else [columns] if len(columns) > 1 else [],
        "resolution": resolution,
    }]})


def encode(rows, resolution, budget=4096, *, evidence_coding="optimized"):
    columns = [f"c{index}" for index in range(len(rows[0]))]
    compresstab = make_compresstab(rows, evidence_coding=evidence_coding)
    signature = make_signature(columns, resolution)
    return compresstab.encode_signature(
        question_id="core-test",
        signature=signature,
        token_budget=budget,
    )


def long_exact_table(size=80):
    return [["entity", "measure"], *[
        [f"e{index}", f"{1000000000000 + index}.123456789"]
        for index in range(size)
    ]]


def selected_trace(compressed, col_id):
    return compressed.encoding_trace["tables"]["t0"]["selected"][col_id]


def test_links_compile_to_explicit_tuple_binding():
    rows = [
        ["metric", "2013", "2014", "2015"],
        ["applications", "10", "20", "30"],
        ["offer rate", "5", "8", "13"],
    ]
    compresstab = make_compresstab(rows)
    signature = make_signature(
        ["c1", "c2", "c3"],
        [
            {"column": "c1", "mode": "exact"},
            {"column": "c2", "mode": "exact"},
            {"column": "c3", "mode": "exact"},
        ],
        rows=["r0", "r1"],
        links=[["c1", "c2", "c3"]],
    )
    plan = compresstab.compile_signature(signature).tables[0]
    alignment = next(item for item in plan.dependencies if item["kind"] == "alignment")
    assert alignment["axis"] == "rows"
    assert alignment["grain"] == "record"
    assert alignment["align_by"] == "row"
    assert plan.layouts[0]["links"] == [["c1", "c2", "c3"]]


def test_single_anchor_uses_readable_sparse_equality_predicate():
    anchor = "a_repeated_very_long_category_name"
    rows = [["group"], *[
        [anchor if index % 3 == 0 else f"unneeded_long_category_{index}"]
        for index in range(180)
    ]]
    compressed = encode(rows, [{
        "column": "c0", "mode": "equality", "anchors": [anchor],
    }], budget=900)

    assert compressed.selected_encodings["c0"] == "sparse_equality"
    assert f'group == "{anchor}"' in compressed.text
    assert "others=no" in compressed.text
    assert "r0" in compressed.text and "r177" in compressed.text
    assert selected_trace(compressed, "c0")["certification_checks"]["anchor_membership"]
    assert "t0:r1:c0" not in compressed.recoverable_source_ref_ids


def test_multiple_anchors_preserve_named_classes_without_opaque_codes():
    anchors = [
        "first_repeated_very_long_category_name",
        "second_repeated_very_long_category_name",
    ]
    rows = [["group"], *[
        [anchors[(index // 4) % 2] if index % 4 == 0 else f"unneeded_long_category_{index}"]
        for index in range(180)
    ]]
    compressed = encode(rows, [{
        "column": "c0", "mode": "equality", "anchors": anchors,
    }], budget=900)

    assert compressed.selected_encodings["c0"] == "sparse_anchored_classes"
    assert all(anchor in compressed.text for anchor in anchors)
    assert "other" in compressed.text
    assert "unneeded_long_category_1" not in compressed.text
    assert "M0" not in compressed.text and "O=" not in compressed.text


def test_repeated_equality_classes_use_reversible_default_factorization():
    values = [
        "alpha_repeated_category_name_with_many_tokens_X",
        "beta_repeated_category_name_with_many_tokens_Y",
    ]
    rows = [["entity", "group"], *[
        [f"entity_{index}", values[index % 2]] for index in range(120)
    ]]
    compressed = encode(rows, [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "equality"},
    ], budget=1200)

    assert compressed.selected_encodings["c1"] == "default_factorization"
    assert compressed.text.count("Coded columns (rows follow grid order):") == 1
    assert "otherwise" in compressed.text and "@r[" in compressed.text
    assert all(value in compressed.text for value in values)
    assert selected_trace(compressed, "c1")["source_reversible"]


def test_identity_values_remain_recoverable_from_default_factorization():
    values = [
        "alpha_repeated_category_name_with_many_tokens_X",
        "beta_repeated_category_name_with_many_tokens_Y",
    ]
    rows = [["group"], *[[values[index % 2]] for index in range(40)]]
    compressed = encode(rows, [{"column": "c0", "mode": "identity"}])

    assert compressed.selected_encodings["c0"] == "default_factorization"
    assert selected_trace(compressed, "c0")["certification_checks"]["direct_value"]
    assert all(value in compressed.text for value in values)


def test_constant_column_is_stored_once_without_losing_row_binding():
    rows = [["entity", "country"], *[
        [f"entity_{index}", "Czech Republic"] for index in range(80)
    ]]
    compressed = encode(rows, [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "exact"},
    ])

    assert compressed.selected_encodings["c1"] in {
        "default_factorization", "grouped_dictionary",
    }
    assert 'country := "Czech Republic"' in compressed.text
    assert selected_trace(compressed, "c1")["source_reversible"]


@pytest.mark.parametrize(
    ("operator", "values", "header_fragment"),
    [
        (
            "prefix_factorization",
            [f"shared_namespace/very/long/path/item_{index}" for index in range(100)],
            'value = "shared_namespace/very/long/path/item_" + suffix',
        ),
        (
            "suffix_factorization",
            [f"{index}_shared_repeated_very_long_suffix" for index in range(100)],
            'value = prefix + "_shared_repeated_very_long_suffix"',
        ),
    ],
)
def test_reversible_affix_factorization_preserves_equality_classes(
    operator, values, header_fragment,
):
    rows = [["value"], *[[value] for value in values]]
    compressed = encode(rows, [{"column": "c0", "mode": "equality"}])

    assert compressed.selected_encodings["c0"] == operator
    assert header_fragment in compressed.text
    trace = selected_trace(compressed, "c0")
    assert trace["source_reversible"]
    assert trace["certification_checks"]["equality"]
    assert len(compressed.recoverable_source_ref_ids) == len(values)


def test_affix_factorization_rejects_empty_remainders():
    value = "American Pale Ale (APA)"
    compressed = encode(
        [["style"], [value], [value]],
        [{"column": "c0", "mode": "identity"}],
    )

    assert compressed.selected_encodings["c0"] not in {
        "prefix_factorization",
        "suffix_factorization",
    }
    assert "suffix\tr" not in compressed.text
    assert value in compressed.text


def test_structured_template_factorization_is_exact_and_token_efficient():
    rows = [["bond_id"], *[
        [
            f"TR{index:03d}_{index % 17}_{(index * 3) % 29}"
            "_CORPORATE-BOND-LONG-SUFFIX"
        ]
        for index in range(120)
    ]]
    signature = make_signature(
        ["c0"], [{"column": "c0", "mode": "equality"}],
    )
    compressed = make_compresstab(rows).encode_signature(
        question_id="structured-template", signature=signature, token_budget=10000,
    )

    assert compressed.selected_encodings["c0"] == "structured_template"
    assert "template=TR{1}_{2}_{3}" in compressed.text
    trace = selected_trace(compressed, "c0")
    assert trace["source_reversible"]
    assert trace["certification_checks"]["equality"]
    assert trace["token_cost"] < trace["raw_equivalent_tokens"]


def test_structured_template_factorization_rejects_inconsistent_shapes():
    rows = [
        ["identifier"],
        ["TR001_1"],
        ["TR002-2"],
        ["TR003_3"],
        ["TR004-4"],
    ]
    compressed = encode(rows, [{"column": "c0", "mode": "identity"}])

    assert compressed.selected_encodings["c0"] != "structured_template"


def test_compiled_filter_is_complete_and_visible_for_empty_matches():
    compresstab = make_compresstab([
        ["app", "rating"],
        ["Alpha", "4.0"],
        ["Beta", "4.5"],
    ])
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "source": "t0",
        "columns": ["c0", "c1"],
        "where": [{"col": "c0", "op": "eq", "value": "Missing"}],
        "links": [["c0", "c1"]],
        "resolution": [
            {"column": "c0", "mode": "identity"},
            {"column": "c1", "mode": "exact"},
        ],
    }]})
    compilation = compresstab.compile_signature(signature)
    compressed = compresstab.encode_signature(
        question_id="empty-filter",
        signature=signature,
        token_budget=512,
    )

    assert compilation.tables[0].row_ids == []
    assert "Scope: 0/2 rows" in compressed.text
    assert 'Complete source filter: c0:app eq "Missing"' in compressed.text


def test_long_numeric_order_uses_dense_rank_and_preserves_pairwise_order():
    rows = [["entity", "measure"], *[
        [f"e{index}", f"{1000000000000 + index}.1234567890123456789"]
        for index in range(80)
    ]]
    compressed = encode(rows, [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "order"},
    ], budget=1000)

    assert compressed.selected_encodings["c1"] == "dense_rank"
    trace = selected_trace(compressed, "c1")
    assert trace["certification_checks"]["order"]
    assert trace["reader_token_savings"] > 0
    assert "rank (1 = highest)" in compressed.text


def test_small_order_column_keeps_raw_when_rank_overhead_does_not_pay():
    rows = [["entity", "score"], ["alpha", "1"], ["beta", "2"]]
    compressed = encode(rows, [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "order"},
    ])
    assert compressed.selected_encodings == {"c0": "raw", "c1": "raw"}
    assert compressed.encoding_trace["tables"]["t0"]["selection_reason"] == (
        "GLOBAL_MINIMUM_COST_CERTIFIED_PLAN"
    )


def test_source_sequence_order_never_uses_numeric_dense_rank():
    rows = [["caller", "year"], *[
        [f"caller_{index}", str(2000 + index)] for index in range(80)
    ]]
    compresstab = make_compresstab(rows)
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "source": "t0",
        "columns": ["c0", "c1"],
        "links": [["c0", "c1"]],
        "resolution": [
            {"column": "c0", "mode": "identity"},
            {"column": "c1", "mode": "order", "order_kind": "source"},
        ],
    }]})
    compressed = compresstab.encode_signature(
        question_id="source-order", signature=signature, token_budget=2000,
    )

    assert compressed.selected_encodings["c1"] != "dense_rank"
    dependency = next(
        item for item in compresstab.compile_signature(signature).tables[0].dependencies
        if item["kind"] == "source_order"
    )
    assert dependency["order_kind"] == "source"


def test_complete_cost_tie_uses_the_dominant_byte_component():
    rows = [["forest", "area"], *[
        [f"forest_{index}", f"{index * 123.4:.1f} ha"] for index in range(15)
    ]]
    compressed = encode(rows, [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "exact"},
    ])

    # Quoting every prefix while factoring " ha" adds bytes to this small view.
    # Raw therefore wins the complete rendered-token tie as an ordinary plan.
    assert compressed.selected_encodings["c1"] == "raw"


def test_threshold_requirement_uses_semantic_predicate():
    rows = [["entity", "measure"], *[
        [
            f"e{index}",
            str(10**30 + index * 9876543210123456789) + ".123456789",
        ]
        for index in range(100)
    ]]
    compressed = encode(rows, [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "threshold", "boundary": 1000000000050},
    ], budget=900)

    assert compressed.selected_encodings["c1"] == "sparse_threshold"
    assert "measure >= 1000000000050" in compressed.text
    assert selected_trace(compressed, "c1")["certification_checks"]["threshold_membership"]


def test_sparse_exact_requirement_adds_source_cited_patch():
    rows = long_exact_table(100)
    compresstab = make_compresstab(rows)
    signature = AnswerabilitySignature.model_validate({"slices": [
        {
            "columns": ["c0", "c1"],
            "links": [["c0", "c1"]],
            "resolution": [
                {"column": "c0", "mode": "identity"},
                {"column": "c1", "mode": "threshold", "boundary": 1e30},
            ],
        },
        {
            "columns": ["c1"],
            "rows": ["r0"],
            "resolution": [{"column": "c1", "mode": "exact"}],
        },
    ]})
    compressed = compresstab.encode_signature(
        question_id="core-test", signature=signature, token_budget=1000,
    )

    assert compressed.selected_encodings["c1"] == "sparse_threshold+exact_patch"
    assert compressed.text.count("Exact patches:") == 1
    assert "[source=t0:r0:c1]" in compressed.text
    trace = selected_trace(compressed, "c1")
    assert trace["certification_checks"]["exact_value"]
    assert "t0:r0:c1" in compressed.recoverable_source_ref_ids
    assert "t0:r1:c1" not in compressed.recoverable_source_ref_ids


@pytest.mark.parametrize("start", ["-1000000000040.125", "1000000000000.125"])
def test_exact_values_remain_directly_observable(start):
    base = float(start)
    rows = [["entity", "value"], *[
        [f"e{index}", f"{base + index:.3f}"] for index in range(40)
    ]]
    compressed = encode(rows, [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "exact"},
    ], budget=1000)

    assert compressed.selected_encodings["c1"] == "raw"
    trace = selected_trace(compressed, "c1")
    assert trace["source_reversible"]
    assert trace["certification_checks"]["exact_value"]
    assert trace["certification_checks"]["direct_value"]
    assert all(f"t0:r{index}:c1" in compressed.recoverable_source_ref_ids for index in range(40))


def test_formatted_numeric_strings_are_not_silently_delta_normalized():
    rows = [["entity", "value"], *[
        [f"e{index}", f"{index:03d}"] for index in range(20)
    ]]
    compressed = encode(rows, [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "exact"},
    ])
    candidates = compressed.encoding_trace["tables"]["t0"]["candidates"]["c1"]

    assert compressed.selected_encodings["c1"] == "raw"
    assert {item["operator"] for item in candidates} == {"raw"}
    assert "000" in compressed.text and "019" in compressed.text


def test_exact_requirements_cannot_be_certified_by_rank_or_threshold_alone():
    compressed = encode(long_exact_table(60), [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "exact"},
    ], budget=1000)
    operators = {
        item["operator"]
        for item in compressed.encoding_trace["tables"]["t0"]["candidates"]["c1"]
    }
    assert "dense_rank" not in operators
    assert "threshold_predicate" not in operators
    assert "exact_delta" not in operators
    assert compressed.selected_encodings["c1"] == "raw"
    assert selected_trace(compressed, "c1")["certification_checks"]["direct_value"]


def test_shared_code_heading_is_paid_once_for_two_dictionary_columns():
    rows = [["entity", "left group", "right group"], *[
        [
            f"e{index}",
            "alpha left repeated category with a long name X" if index % 2 else
            "beta left repeated category with a long name Y",
            "gamma right repeated category with a long name U" if index % 3 else
            "delta right repeated category with a long name V",
        ]
        for index in range(120)
    ]]
    compressed = encode(rows, [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "equality"},
        {"column": "c2", "mode": "equality"},
    ], budget=1800)
    trace = compressed.encoding_trace["tables"]["t0"]

    assert compressed.selected_encodings["c1"] == "default_factorization"
    assert compressed.selected_encodings["c2"] == "default_factorization"
    assert compressed.text.count("Coded columns (rows follow grid order):") == 1
    assert trace["shared_sections"] == ["codes"]
    assert trace["shared_section_tokens"] > 0


def test_tuple_bound_columns_remain_row_aligned_after_coding():
    rows = [["entity", "value"], *[
        [f"entity_{index}", f"{1000000000000 + index}.25"] for index in range(50)
    ]]
    compressed = encode(rows, [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "exact"},
    ])
    trace = compressed.encoding_trace["tables"]["t0"]

    assert "Grid row order := r[0-49]" in compressed.text
    assert trace["axis_addressing"]["row_axis_mode"] == "ordered_range"
    assert all(trace["axis_addressing"]["row_axis_certification"].values())
    assert trace["evidence_groups"] == [["c0", "c1"]]


def test_raw_ablation_keeps_signature_scope_but_not_optimized_representation():
    rows = long_exact_table(80)
    resolution = [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "order"},
    ]
    optimized = encode(rows, resolution, evidence_coding="optimized")
    raw = encode(rows, resolution, evidence_coding="raw")

    assert optimized.selected_row_ids == raw.selected_row_ids
    assert optimized.selected_col_ids == raw.selected_col_ids
    assert optimized.answerability_signature == raw.answerability_signature
    assert optimized.selected_encodings["c1"] == "dense_rank"
    assert set(raw.selected_encodings.values()) == {"raw"}
    assert optimized.token_count < raw.token_count


def test_expression_scoped_coding_preserves_order_ties_and_reduces_full_witness():
    rows = [["club", "attendance 2022", "attendance 2023"]]
    expected_deltas = {}
    for index in range(120):
        base = 10**25 + index * 1000000000003
        delta = 999999999999 if index in {37, 91} else (index * 7919) % 9999999
        expected_deltas[f"r{index}"] = delta
        rows.append([
            f"Club_{index:03d}_Athletic_Association",
            str(base),
            str(base + delta),
        ])
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "source": "t0",
        "columns": ["c0", "c1", "c2"],
        "links": [["c0", "c1", "c2"]],
        "resolution": [{"column": "c0", "mode": "identity"}],
        "derived": [{
            "observable_id": "v0",
            "operator": "difference",
            "operands": [
                {"column": "c1", "role": "earlier"},
                {"column": "c2", "role": "later"},
            ],
            "mode": "order",
            "direction": "descending",
        }],
    }]})
    optimized_compresstab = make_compresstab(rows, evidence_coding="optimized")
    raw_compresstab = make_compresstab(rows, evidence_coding="raw")
    optimized = optimized_compresstab.encode_signature(
        question_id="derived-order", signature=signature, token_budget=10000,
    )
    raw = raw_compresstab.encode_signature(
        question_id="derived-order", signature=signature, token_budget=10000,
    )

    assert optimized.answerability_signature == raw.answerability_signature
    assert optimized.selected_row_ids == raw.selected_row_ids
    assert optimized.selected_encodings["v0"] == "derived_dense_rank"
    assert raw.selected_encodings["v0"] == "derive_from_operands"
    assert raw.selected_encodings["c1"] == "raw"
    assert raw.selected_encodings["c2"] == "raw"
    trace = selected_trace(optimized, "v0")
    ranks = {
        row_id: int(values[0])
        for row_id, values in trace["visible_values"].items()
    }
    assert trace["certification_checks"]["derived_order"]
    assert ranks["r37"] == ranks["r91"] == 1
    assert all(
        ((expected_deltas[left] > expected_deltas[right]) == (ranks[left] < ranks[right]))
        and ((expected_deltas[left] == expected_deltas[right]) == (ranks[left] == ranks[right]))
        for left in expected_deltas
        for right in expected_deltas
    )
    assert optimized.token_count < raw.token_count * 0.6
    assert "Links: club <-> v0:" in optimized.text
    assert "derived v0=attendance 2023 - attendance 2022" in optimized.text

    derived_refs = [
        ref for ref in optimized.evidence_refs if ref.source_type == "derived"
    ]
    assert len(derived_refs) == 120
    ref = next(item for item in derived_refs if item.ref_id == "t0:r37:v0")
    assert ref.lineage_ref_ids == ["t0:r37:c1", "t0:r37:c2"]
    assert ref.source_cell_ids


def test_exact_derived_observable_never_uses_an_order_only_code():
    rows = [["entity", "earlier", "later"], *[
        [f"e{index}", str(10**18 + index), str(10**18 + index * 2)]
        for index in range(80)
    ]]
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "columns": ["c0", "c1", "c2"],
        "links": [["c0", "c1", "c2"]],
        "resolution": [{"column": "c0", "mode": "identity"}],
        "derived": [{
            "observable_id": "v0",
            "operator": "difference",
            "operands": [
                {"column": "c1", "role": "earlier"},
                {"column": "c2", "role": "later"},
            ],
            "mode": "exact",
        }],
    }]})
    compressed = make_compresstab(rows).encode_signature(
        question_id="derived-exact", signature=signature, token_budget=10000,
    )
    operators = {
        item["operator"]
        for item in compressed.encoding_trace["tables"]["t0"]["candidates"]["v0"]
    }

    assert "derived_dense_rank" not in operators
    assert compressed.selected_encodings["v0"] in {
        "derived_exact", "derive_from_operands",
    }


def test_independently_required_operand_composes_with_derived_realization():
    rows = [["entity", "earlier", "later"], *[
        [f"e{index}", str(10**18 + index), str(10**18 + index * 2)]
        for index in range(80)
    ]]
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "columns": ["c0", "c1", "c2"],
        "links": [["c0", "c1", "c2"]],
        "resolution": [
            {"column": "c0", "mode": "identity"},
            {"column": "c1", "mode": "exact"},
        ],
        "derived": [{
            "observable_id": "v0",
            "operator": "difference",
            "operands": [
                {"column": "c1", "role": "earlier"},
                {"column": "c2", "role": "later"},
            ],
            "mode": "order",
        }],
    }]})
    compressed = make_compresstab(rows).encode_signature(
        question_id="source-and-derived", signature=signature, token_budget=10000,
    )

    assert "c1" in compressed.selected_encodings
    assert compressed.selected_encodings["v0"] in {
        "derived_exact", "derived_dense_rank", "derive_from_operands",
    }
    source_trace = selected_trace(compressed, "c1")
    assert source_trace["source_reversible"]
    assert "exact_value" in source_trace["mandatory_requirements"]


def test_operand_bundle_can_be_compressed_and_shared_with_expression_plan():
    rows = [["entity", "earlier", "later"], *[
        [f"e{index}", str(10**30), str(10**30 + index * 2)]
        for index in range(80)
    ]]
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "columns": ["c0", "c1", "c2"],
        "links": [["c0", "c1", "c2"]],
        "resolution": [
            {"column": "c0", "mode": "identity"},
            {"column": "c1", "mode": "exact"},
            {"column": "c2", "mode": "exact"},
        ],
        "derived": [{
            "observable_id": "v0",
            "operator": "difference",
            "operands": [
                {"column": "c1", "role": "earlier"},
                {"column": "c2", "role": "later"},
            ],
            "mode": "order",
        }],
    }]})
    compressed = make_compresstab(rows).encode_signature(
        question_id="compressed-operands", signature=signature, token_budget=10000,
    )

    assert compressed.selected_encodings["v0"] == "derive_from_operands"
    assert selected_trace(compressed, "v0")["dependencies"] == [
        {"node": "c1", "property": "source_reconstruction"},
        {"node": "c2", "property": "source_reconstruction"},
    ]
    assert selected_trace(compressed, "c1")["source_reversible"]
    assert selected_trace(compressed, "c2")["source_reversible"]
    assert any(
        compressed.selected_encodings[col_id] != "raw"
        for col_id in ("c1", "c2")
    )


def test_derived_ratio_with_zero_denominator_fails_explicitly():
    rows = [["entity", "part", "total"], ["a", "10", "0"]]
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "columns": ["c0", "c1", "c2"],
        "links": [["c0", "c1", "c2"]],
        "resolution": [{"column": "c0", "mode": "identity"}],
        "derived": [{
            "observable_id": "v0",
            "operator": "ratio",
            "operands": [
                {"column": "c1", "role": "numerator"},
                {"column": "c2", "role": "denominator"},
            ],
            "mode": "order",
        }],
    }]})

    with pytest.raises(ValueError, match="Zero denominator.*r0"):
        make_compresstab(rows).encode_signature(
            question_id="zero-denominator", signature=signature, token_budget=1000,
        )


def test_rowwise_sum_observable_is_ranked_and_certified_generically():
    rows = [
        ["entity", "home", "away"],
        ["Tigers", "1200", "900"],
        ["Falcons", "800", "1400"],
        ["Owls", "1000", "1100"],
    ]
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "columns": ["c0", "c1", "c2"],
        "links": [["c0", "c1", "c2"]],
        "resolution": [{"column": "c0", "mode": "identity"}],
        "derived": [{
            "observable_id": "v0",
            "operator": "sum",
            "operands": [
                {"column": "c1", "role": "value"},
                {"column": "c2", "role": "value"},
            ],
            "mode": "order",
            "direction": "descending",
        }],
    }]})
    compressed = make_compresstab(rows).encode_signature(
        question_id="rowwise-sum", signature=signature, token_budget=1000,
    )

    trace = selected_trace(compressed, "v0")
    assert trace["certified"]
    assert trace["certification_checks"]["derived_order"]
    if compressed.selected_encodings["v0"] == "derived_dense_rank":
        ranks = {
            row_id: int(values[0])
            for row_id, values in trace["visible_values"].items()
        }
        assert ranks == {"r0": 2, "r1": 1, "r2": 2}


def test_recoverable_source_refs_are_isolated_by_table():
    long_rows = [["entity", "earlier", "later"], *[
        [f"e{index}", str(10**22 + index * 100003), str(10**22 + index * 100003 + index)]
        for index in range(60)
    ]]
    artifact = artifact_from_dataset_row({
        "question_id": "multi-derived",
        "tables": [
            long_rows,
            [["entity", "earlier", "later"], ["other", "4", "9"]],
        ],
    })
    compresstab = CompressTab(TableStore(artifact), evidence_coding="optimized")
    signature = AnswerabilitySignature.model_validate({"slices": [
        {
            "source": "t0",
            "columns": ["c0", "c1", "c2"],
            "links": [["c0", "c1", "c2"]],
            "resolution": [{"column": "c0", "mode": "identity"}],
            "derived": [{
                "observable_id": "v0",
                "operator": "difference",
                "operands": [
                    {"column": "c1", "role": "earlier"},
                    {"column": "c2", "role": "later"},
                ],
                "mode": "order",
            }],
        },
        {
            "source": "t1",
            "columns": ["c0", "c1", "c2"],
            "links": [["c0", "c1", "c2"]],
            "resolution": [
                {"column": "c0", "mode": "identity"},
                {"column": "c1", "mode": "exact"},
                {"column": "c2", "mode": "exact"},
            ],
        },
    ]})
    compressed = compresstab.encode_signature(
        question_id="multi-derived", signature=signature, token_budget=10000,
    )

    assert compressed.selected_encodings["t0:v0"] in {
        "derived_exact", "derived_dense_rank"
    }
    assert "t0:r0:c2" not in compressed.recoverable_source_ref_ids
    assert "t1:r0:c2" in compressed.recoverable_source_ref_ids


def test_prepared_evidence_identity_ignores_new_empty_derived_fields():
    old = {
        "schema_version": "compresstab.evidence.v2",
        "dataset": "wtq",
        "question_id": "q0",
        "answerability_signature": {"slices": [{"source": "t0"}]},
        "compilation": {"tables": [{"table_id": "t0"}]},
        "witness": {},
        "preparation": {},
    }
    new = json.loads(json.dumps(old))
    new["answerability_signature"]["slices"][0]["derived"] = []
    new["compilation"]["tables"][0]["derived_observables"] = []

    assert prepared_evidence_identity(old) == prepared_evidence_identity(new)


def test_overlapping_derived_observables_share_source_nodes():
    rows = [["entity", "a", "b", "c"], ["x", "1", "2", "3"]]
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "columns": ["c0", "c1", "c2", "c3"],
        "resolution": [{"column": "c0", "mode": "identity"}],
        "derived": [
            {
                "observable_id": "v0",
                "operator": "difference",
                "operands": [
                    {"column": "c1", "role": "earlier"},
                    {"column": "c2", "role": "later"},
                ],
            },
            {
                "observable_id": "v1",
                "operator": "ratio",
                "operands": [
                    {"column": "c2", "role": "numerator"},
                    {"column": "c3", "role": "denominator"},
                ],
            },
        ],
    }]})
    compresstab = make_compresstab(rows)
    compilation = compresstab.compile_signature(signature)

    assert not compilation.has_blocking_diagnostics
    raw = make_compresstab(rows, evidence_coding="raw").encode_signature(
        question_id="overlap", signature=signature, token_budget=1000,
    )
    assert raw.selected_encodings["v0"] == "derive_from_operands"
    assert raw.selected_encodings["v1"] == "derive_from_operands"
    assert raw.selected_encodings["c2"] == "raw"
    header = next(
        line for line in raw.text.splitlines()
        if line.startswith("source_row | ")
    )
    assert header.count(" | b |") == 1


def test_exact_dp_trace_uses_configured_token_cost():
    compressed = encode(long_exact_table(80), [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "exact"},
    ])
    trace = compressed.encoding_trace["tables"]["t0"]
    from compresstab.table.store import approx_token_count
    assert compressed.token_count == approx_token_count(compressed.text)
    assert compressed.encoding_trace["optimized_complete_witness_tokens"] == compressed.token_count
    assert trace["dp_layers"]
    assert all(item["certified"] for item in trace["selected"].values())


def test_global_dp_and_local_greedy_use_the_same_certified_candidate_space():
    rows = [["entity", "earlier", "later"], *[
        [f"entity_{index:03d}", str(10**18 + index), str(10**18 + index * 3)]
        for index in range(80)
    ]]
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "source": "t0",
        "columns": ["c0", "c1", "c2"],
        "links": [["c0", "c1", "c2"]],
        "resolution": [{"column": "c0", "mode": "identity"}],
        "derived": [{
            "observable_id": "v0",
            "operator": "difference",
            "operands": [
                {"column": "c1", "role": "earlier"},
                {"column": "c2", "role": "later"},
            ],
            "mode": "order",
        }],
    }]})
    global_plan = make_compresstab(
        rows, optimization_strategy="global_dp",
    ).encode_signature(question_id="global", signature=signature, token_budget=10000)
    greedy_plan = make_compresstab(
        rows, optimization_strategy="local_greedy",
    ).encode_signature(question_id="greedy", signature=signature, token_budget=10000)

    assert global_plan.answerability_signature == greedy_plan.answerability_signature
    assert global_plan.selected_row_ids == greedy_plan.selected_row_ids
    assert global_plan.selected_col_ids == greedy_plan.selected_col_ids
    assert global_plan.token_count <= greedy_plan.token_count
    assert global_plan.encoding_trace["tables"]["t0"]["selection_reason"] == (
        "GLOBAL_MINIMUM_COST_CERTIFIED_PLAN"
    )
    assert greedy_plan.encoding_trace["tables"]["t0"]["selection_reason"] == (
        "LOCAL_GREEDY_CERTIFIED_PLAN"
    )
    assert all(
        item["certified"]
        for item in global_plan.encoding_trace["tables"]["t0"]["selected"].values()
    )
    assert all(
        item["certified"]
        for item in greedy_plan.encoding_trace["tables"]["t0"]["selected"].values()
    )


def test_joint_tuple_realization_covers_linked_source_nodes_once():
    regions = ["Northern Europe", "Southern Europe", "Western Europe"]
    statuses = ["Currently Active", "Temporarily Closed", "Pending Review"]
    rows = [["entity", "regional classification", "operational status"], *[
        [f"entity_{index:03d}", regions[index % 3], statuses[index % 3]]
        for index in range(120)
    ]]
    signature = make_signature(
        ["c0", "c1", "c2"],
        [
            {"column": "c0", "mode": "identity"},
            {"column": "c1", "mode": "exact"},
            {"column": "c2", "mode": "exact"},
        ],
        links=[["c0", "c1", "c2"]],
    )
    without_joint = make_compresstab(
        rows, token_counter=lambda text: len(str(text)), joint_realizations=False,
    )
    local = without_joint.encode_signature(
        question_id="node-local", signature=signature, token_budget=10000,
    )
    joint = make_compresstab(
        rows, token_counter=lambda text: len(str(text)),
    ).encode_signature(
        question_id="joint", signature=signature, token_budget=10000,
    )

    assert joint.token_count < local.token_count
    assert joint.selected_encodings["c1"] == "default_tuple_exceptions"
    assert joint.selected_encodings["c2"] == "default_tuple_exceptions"
    traces = joint.encoding_trace["tables"]["t0"]["selected"]
    bundle = next(item for item in traces.values() if item["node_kind"] == "joint_source")
    assert bundle["covered_node_ids"] == ["c1", "c2"]
    assert bundle["certification_checks"]["source_reconstruction"]
    assert bundle["certification_checks"]["joint_binding"]


def test_fully_coded_joint_plan_does_not_render_an_empty_row_grid():
    rows = [["region", "status"], *[
        ["North", "Active"] if index % 2 == 0 else ["South", "Closed"]
        for index in range(100)
    ]]
    compressed = encode(
        rows,
        [
            {"column": "c0", "mode": "exact"},
            {"column": "c1", "mode": "exact"},
        ],
        budget=10000,
    )

    assert compressed.selected_encodings["c0"] in {
        "tuple_postings", "default_tuple_exceptions", "shared_postings",
    }
    assert "source_row\nr0\nr1" not in compressed.text
    assert "@r[" in compressed.text


def test_rectangular_grid_keeps_empty_cells_visibly_empty():
    assert EvidencePlanOptimizer._render_cell("") == ""
    assert EvidencePlanOptimizer._render_cell(" value ") == '" value "'

    rows = [["entity", "value"], *[
        ["", ""] if index == 10 else [f"e{index}", f"v{index}"]
        for index in range(30)
    ]]
    compressed = encode(
        rows,
        [
            {"column": "c0", "mode": "exact"},
            {"column": "c1", "mode": "exact"},
        ],
        budget=10000,
    )
    lines = compressed.text.split(CANONICAL_LINE_SEPARATOR)
    preceding = lines.index(f"e9{CANONICAL_CELL_SEPARATOR}v9 |")
    assert lines[preceding + 1] == f"{CANONICAL_CELL_SEPARATOR} |"
    assert lines[preceding + 2] == f"e11{CANONICAL_CELL_SEPARATOR}v11 |"


def test_cross_table_distinct_plans_preserve_tokenizer_shorter_plan():
    tokenizer_shorter = (10, 30, 300, ("raw",), {"t0": {"kind": "raw"}})
    lexical_shorter = (
        14,
        20,
        200,
        ("grouped_dictionary",),
        {"t0": {"kind": "grouped_dictionary"}},
    )

    frontier = CompressTab._distinct_evidence_plans([
        tokenizer_shorter,
        lexical_shorter,
    ])

    assert tokenizer_shorter in frontier
    assert lexical_shorter in frontier


def test_complete_witness_over_target_is_returned_intact():
    rows = [["entity", "score"], ["alpha", "1"], ["beta", "2"]]
    resolution = [
        {"column": "c0", "mode": "identity"},
        {"column": "c1", "mode": "exact"},
    ]
    full = encode(rows, resolution, budget=4096)
    variable_minimum = full.encoding_trace["tables"]["t0"]["minimum_required_tokens"]
    budget = min(full.token_count - 1, variable_minimum + 4)

    compressed = encode(rows, resolution, budget=budget)
    assert compressed.text == full.text
    assert compressed.token_count == full.token_count
    assert compressed.encoding_trace["target_evidence_tokens"] == budget
    assert compressed.encoding_trace["actual_evidence_tokens"] == full.token_count
    assert compressed.encoding_trace["within_target"] is False
    assert compressed.encoding_trace["tokens_over_target"] == full.token_count - budget


def test_complete_witness_target_uses_injected_tokenizer_counter():
    rows = [["entity", "score"], ["alpha", "1"], ["beta", "2"]]
    signature = make_signature(
        ["c0", "c1"],
        [
            {"column": "c0", "mode": "identity"},
            {"column": "c1", "mode": "exact"},
        ],
    )
    compresstab = make_compresstab(rows, token_counter=lambda _text: 101)

    compressed = compresstab.encode_signature(
        question_id="core-test",
        signature=signature,
        token_budget=100,
    )

    assert compressed.token_count == 101
    assert compressed.encoding_trace["actual_evidence_tokens"] == 101
    assert compressed.encoding_trace["within_target"] is False


def test_small_target_never_drops_required_domain():
    rows = [["label", "value"], *[
        [f"very_long_entity_name_{index:03d}", f"{index}.123456789"]
        for index in range(120)
    ]]
    resolution = [
        {"column": "c0", "mode": "exact"},
        {"column": "c1", "mode": "exact"},
    ]
    compressed = encode(rows, resolution, budget=256)

    assert compressed.token_count > compressed.token_budget
    assert compressed.encoding_trace["actual_evidence_tokens"] == compressed.token_count
    assert compressed.encoding_trace["within_target"] is False
    assert len(compressed.selected_row_ids) == 120
    assert set(compressed.selected_col_ids) == {"c0", "c1"}


def test_threshold_resolution_compiles_with_its_explicit_boundary():
    rows = [["name", "score"], ["a", "1"], ["b", "2"]]
    compresstab = make_compresstab(rows)
    signature = make_signature(
        ["c1"],
        [{"column": "c1", "mode": "threshold", "boundary": 1.5}],
    )
    dependency = next(
        item for item in compresstab.compile_signature(signature).tables[0].dependencies
        if item["kind"] == "threshold_membership"
    )
    assert dependency["boundary"] == 1.5


def test_offline_profiles_are_persisted_by_table_fingerprint(tmp_path):
    rows = [["id", "value"], ["x", "1"], ["y", "2"]]
    first = make_compresstab(rows, profile_cache_dir=str(tmp_path))
    first_payload = first.offline_profile()["t0"]
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    persisted = json.loads(files[0].read_text(encoding="utf-8"))
    assert persisted["fingerprint"] == first_payload["fingerprint"]

    table = first.store.table("t0")
    second_payload = ProfileStore(tmp_path).describe(table)
    assert second_payload == first_payload
    assert files[0].stem == table_fingerprint(table)


def test_profile_store_fingerprints_each_table_object_once(monkeypatch):
    compresstab = make_compresstab([
        ["id", "value"],
        ["x", "1"],
        ["y", "2"],
    ])
    table = compresstab.store.table("t0")
    original = profile_module.table_fingerprint
    calls = []

    def counted(candidate):
        calls.append(id(candidate))
        return original(candidate)

    monkeypatch.setattr(profile_module, "table_fingerprint", counted)
    profiles = ProfileStore()
    profiles.describe(table)
    profiles.profile(table, "c0")
    profiles.profile(table, "c1")

    assert calls == [id(table)]


def test_nested_expression_dag_propagates_exact_obligations():
    rows = [
        ["entity", "revenue", "cost", "assets"],
        ["alpha", "120", "20", "50"],
        ["beta", "150", "30", "40"],
        ["gamma", "90", "10", "80"],
    ]
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "source": "t0",
        "columns": ["c0", "c1", "c2", "c3"],
        "resolution": [{"column": "c0", "mode": "identity"}],
        "derived": [
            {
                "observable_id": "v0",
                "operator": "difference",
                "operands": [
                    {"column": "c1", "role": "left"},
                    {"column": "c2", "role": "right"},
                ],
                "mode": "exact",
                "required": False,
            },
            {
                "observable_id": "v1",
                "operator": "ratio",
                "operands": [
                    {"column": "v0", "role": "numerator"},
                    {"column": "c3", "role": "denominator"},
                ],
                "mode": "order",
            },
        ],
    }]})
    compressed = make_compresstab(rows, evidence_coding="raw").encode_signature(
        question_id="nested-expression",
        signature=signature,
        token_budget=4096,
    )

    assert compressed.selected_encodings["v0"] == "derive_from_operands"
    assert compressed.selected_encodings["v1"] == "derive_from_operands"
    trace = compressed.encoding_trace["tables"]["t0"]
    assert trace["dependency_obligations"]["v0"] == ["derived_exact"]
    assert trace["evidence_graph"]["edges"] == [
        {"from": "c1", "to": "v0", "role": "left"},
        {"from": "c2", "to": "v0", "role": "right"},
        {"from": "v0", "to": "v1", "role": "numerator"},
        {"from": "c3", "to": "v1", "role": "denominator"},
    ]
    assert trace["selected"]["v1"]["certification_checks"]["derived_order"]
    assert set(trace["selected"]["v1"]["source_col_ids"]) == {"c1", "c2", "c3"}

    optimized = make_compresstab(rows).encode_signature(
        question_id="nested-expression",
        signature=signature,
        token_budget=4096,
    )
    assert optimized.selected_encodings["v1"] == "derived_exact"
    assert "v0" not in optimized.selected_encodings


def test_expression_cycle_is_rejected_before_physical_planning():
    rows = [["entity", "value"], ["alpha", "1"], ["beta", "2"]]
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "source": "t0",
        "columns": ["c0", "c1"],
        "resolution": [{"column": "c0", "mode": "identity"}],
        "derived": [
            {
                "observable_id": "v0",
                "operator": "sum",
                "operands": [{"column": "v1", "role": "value"}],
                "mode": "exact",
            },
            {
                "observable_id": "v1",
                "operator": "sum",
                "operands": [{"column": "v0", "role": "value"}],
                "mode": "exact",
            },
        ],
    }]})
    compilation = make_compresstab(rows).compile_signature(signature)

    assert compilation.has_blocking_diagnostics
    diagnostic = next(
        item for item in compilation.diagnostics
        if item["code"] == "INVALID_DERIVED_GRAPH"
    )
    assert "cycle" in diagnostic["message"].lower()


def test_percent_change_expression_uses_declared_scale():
    rows = [
        ["entity", "earlier", "later"],
        ["alpha", "100", "125"],
        ["beta", "80", "100"],
    ]
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "source": "t0",
        "columns": ["c0", "c1", "c2"],
        "resolution": [{"column": "c0", "mode": "identity"}],
        "derived": [{
            "observable_id": "v0",
            "operator": "percent_change",
            "operands": [
                {"column": "c1", "role": "earlier"},
                {"column": "c2", "role": "later"},
            ],
            "mode": "exact",
            "scale": 100,
            "unit": "percent",
        }],
    }]})
    compressed = make_compresstab(rows).encode_signature(
        question_id="percent-change",
        signature=signature,
        token_budget=4096,
    )

    trace = selected_trace(compressed, "v0")
    assert "* 100" in trace["derived_expression"]
    assert trace["visible_values"] == {"r0": ["25"], "r1": ["25"]}
