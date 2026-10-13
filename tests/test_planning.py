from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from compresstab.agents import (  # noqa: E402
    BackwardPlannerAgent,
    AnswerAgent,
    clean_signature,
    lower_answer_requirements,
    normalize_structured_answer,
)
from compresstab.compress import CompressTab  # noqa: E402
from compresstab.evidence import (  # noqa: E402
    EvidencePredicate,
    matches_predicate,
    matches_table_predicate,
)
from compresstab.table.artifact import artifact_from_dataset_row  # noqa: E402
from compresstab.table.store import TableStore, approx_token_count  # noqa: E402


def make_compresstab(headers, rows, question="find target"):
    source = {
        "question_id": "q",
        "question": question,
        "table": [headers, *rows],
    }
    return CompressTab(TableStore(artifact_from_dataset_row(source)))


def test_signature_cleaner_preserves_expression_dependencies():
    sketch = {"sources": [{
        "id": "t0",
        "shape": [2, 4],
        "headers": [{"id": f"c{index}"} for index in range(4)],
        "evidence": [{"row": "r0"}, {"row": "r1"}],
    }]}
    signature = clean_signature({"slices": [{
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
    }]}, table_sketch=sketch)

    assert [item.observable_id for item in signature.slices[0].derived] == ["v0", "v1"]
    assert signature.slices[0].derived[1].operands[0].column == "v0"


def test_contains_treats_typographic_word_separators_as_equivalent():
    assert matches_predicate("On a cost-plus type contract", "contains", "cost plus")
    assert matches_predicate("time-and-material type", "contains", "time and material")


def test_under_predicate_selects_only_strict_hierarchical_descendants():
    compresstab = make_compresstab(
        ["item", "value"],
        [
            ["Current", ""],
            ["Trade payables", "24.6"],
            ["Non-current", ""],
            ["Other payables", "0.8"],
            ["Government grants", "0.2"],
            ["Equity", ""],
            ["Share capital", "8.0"],
        ],
    )
    table = compresstab.store.artifact.tables[0]
    table.row_header_paths = [
        ["Current"],
        ["Current", "Trade payables"],
        ["Non-current"],
        ["Non-current", "Other payables"],
        ["Non-current", "Government grants"],
        ["Equity"],
        ["Equity", "Share capital"],
    ]
    predicate = EvidencePredicate(col="c0", op="under", value="Non-current")

    matches = [
        f"r{index}"
        for index in range(len(table.rows))
        if matches_table_predicate(table, f"r{index}", predicate)
    ]

    assert matches == ["r3", "r4"]
    assert not matches_table_predicate(table, "r2", predicate)
    assert not matches_table_predicate(table, "r6", predicate)


def test_under_predicate_requires_a_complete_row_hierarchy():
    compresstab = make_compresstab(
        ["item", "value"],
        [["Parent", ""], ["Child", "1"]],
    )
    table = compresstab.store.artifact.tables[0]
    table.row_header_paths = []

    assert not matches_table_predicate(
        table,
        "r1",
        EvidencePredicate(col="c0", op="under", value="Parent"),
    )


def test_under_requirement_lowers_and_compiles_as_complete_scope():
    compresstab = make_compresstab(
        ["item", "value"],
        [["Parent", ""], ["Child A", "1"], ["Child B", "2"]],
        "What are the components under Parent?",
    )
    table = compresstab.store.artifact.tables[0]
    table.row_header_paths = [
        ["Parent"], ["Parent", "Child A"], ["Parent", "Child B"]
    ]
    sketch = compresstab.describe("What are the components under Parent?", 800)

    contract, signature, violations = lower_answer_requirements(
        {
            "answer": {"shape": "list", "coverage": "all"},
            "requirements": [{
                "source": "t0",
                "scope": {
                    "where": [{"column": "c0", "op": "under", "value": "Parent"}]
                },
                "fields": [
                    {"column": "c0", "use": "filter"},
                    {"column": "c0", "use": "return"},
                ],
            }],
        },
        table_sketch=sketch,
    )

    assert violations == []
    assert contract["coverage"] == "all"
    assert signature.slices[0].where[0].op == "under"
    assert compresstab.compile_signature(signature).tables[0].row_ids == ["r1", "r2"]


def test_reader_does_not_guess_scale_when_model_omits_it():
    class FakeLLM:
        def chat_json(self, **_kwargs):
            payload = {"answer": 10, "scale": "", "status": "answered", "verification": "verified"}
            return payload, "{}"

    reader = AnswerAgent(FakeLLM(), "fake")
    result = reader.answer(
        question="What was the revenue?",
        compressed_table="COMPRESSTAB WITNESS\n\nTABLE t0\nContext:\n- scale: millions (scope=table)\nr0\t10",
        source_ref_ids=[],
        answer_contract={"expression": {"op": "lookup", "unit": "source"}},
    )

    assert result["scale"] == ""


def test_small_table_sketch_exposes_complete_domain_after_linked_rows():
    compresstab = make_compresstab(
        ["item", "value"],
        [["alpha", "10"], ["target", "20"], ["omega", "30"]],
        "What is the value for target?",
    )

    source = compresstab.describe("What is the value for target?", 3200)["sources"][0]

    assert source["evidence_complete"] is True
    assert source["evidence_omitted"] == 0
    assert {item["row"] for item in source["evidence"]} == {"r0", "r1", "r2"}
    assert source["evidence"][0]["row"] == "r1"


def test_full_row_requirement_stays_symbolic_and_compiles_to_every_column():
    compresstab = make_compresstab(
        ["name", "team", "score"],
        [["Ada", "A", "10"], ["Bob", "B", "20"]],
        "Return Ada's complete row",
    )
    sketch = compresstab.describe("Return Ada's complete row", 800)
    contract, signature, violations = lower_answer_requirements(
        {
            "answer": {"shape": "row", "coverage": "one", "history": []},
            "requirements": [{
                "source": "t0",
                "scope": {"where": [{"column": "c0", "op": "eq", "value": "Ada"}]},
                "fields": [{"column": "c0", "use": "filter"}],
            }],
        },
        table_sketch=sketch,
    )

    assert contract["shape"] == "row"
    assert violations == []
    assert signature.slices[0].columns == ["*"]
    assert signature.slices[0].links == [["*"]]
    plan = compresstab.compile_signature(signature).tables[0]
    assert plan.row_ids == ["r0"]
    assert plan.col_ids == ["c0", "c1", "c2"]
    assert any(item["kind"] == "alignment" for item in plan.dependencies)


def test_complete_population_rejects_sampled_rows():
    compresstab = make_compresstab(["id", "kind"], [["1", "x"], ["2", "x"]])
    sketch = compresstab.describe("List all ids", 800)
    _contract, _signature, violations = lower_answer_requirements(
        {
            "answer": {"shape": "list", "coverage": "all"},
            "requirements": [{
                "source": "t0",
                "scope": {"rows": ["r0"]},
                "fields": [{"column": "c0", "use": "return"}],
            }],
        },
        table_sketch=sketch,
    )

    assert {item["code"] for item in violations} == {"SAMPLED_COMPLETE_DOMAIN"}


def test_answer_contract_preserves_explicit_list_multiplicity():
    compresstab = make_compresstab(["item"], [["x"], ["x"]])
    sketch = compresstab.describe("Which distinct items appear?", 800)
    contract, _signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "list",
                "coverage": "all",
                "multiplicity": "distinct",
            },
            "requirements": [{
                "source": "t0",
                "scope": {"where": []},
                "fields": [{"column": "c0", "use": "return"}],
            }],
        },
        table_sketch=sketch,
    )

    assert violations == []
    assert contract["multiplicity"] == "distinct"


def test_numeric_expression_cannot_return_an_entity_label():
    compresstab = make_compresstab(
        ["item", "value"], [["a", "10"], ["b", "20"]], "ratio of a to b"
    )
    sketch = compresstab.describe("ratio of a to b", 800)
    _contract, _signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "value",
                "coverage": "one",
                "expression": {
                    "op": "ratio",
                    "args": [
                        {"source": "t0", "column": "c1", "row": "r0", "role": "numerator"},
                        {"source": "t0", "column": "c1", "row": "r1", "role": "denominator"},
                    ],
                    "return": {"source": "t0", "column": "c0", "kind": "entity"},
                    "unit": "ratio",
                    "scale": 1,
                },
            },
            "requirements": [{
                "source": "t0",
                "scope": {"rows": ["r0", "r1"]},
                "fields": [{"column": "c1", "use": "compute"}],
            }],
        },
        table_sketch=sketch,
    )

    assert "INVALID_NUMERIC_RETURN" in {item["code"] for item in violations}


def test_disjunctive_scope_preserves_or_semantics_during_compilation():
    compresstab = make_compresstab(
        ["octal", "decimal", "name"],
        [["60", "48", "a"], ["70", "45", "b"], ["70", "55", "c"]],
    )
    sketch = compresstab.describe("octal in the 60s or decimal in the 40s", 800)
    _contract, signature, violations = lower_answer_requirements(
        {
            "answer": {"shape": "value", "coverage": "all"},
            "requirements": [{
                "source": "t0",
                "scope": {
                    "where_logic": "any",
                    "where": [
                        {"column": "c0", "op": "between", "value": [60, 69]},
                        {"column": "c1", "op": "between", "value": [40, 49]},
                    ],
                },
                "fields": [
                    {"column": "c0", "use": "filter"},
                    {"column": "c1", "use": "filter"},
                    {"column": "c2", "use": "return"},
                ],
            }],
        },
        table_sketch=sketch,
    )

    assert violations == []
    assert signature.slices[0].where_logic == "any"
    plan = compresstab.compile_signature(signature).tables[0]
    assert plan.row_ids == ["r0", "r1"]
    assert plan.layouts[0]["where_logic"] == "any"


def test_one_field_can_preserve_filter_and_return_distinctions():
    compresstab = make_compresstab(["year"], [["2004"], ["2006"], ["2008"]])
    sketch = compresstab.describe("which years were after 2005", 800)
    _contract, signature, violations = lower_answer_requirements(
        {
            "answer": {"shape": "list", "coverage": "all"},
            "requirements": [{
                "source": "t0",
                "scope": {"where": [{"column": "c0", "op": "gt", "value": 2005}]},
                "fields": [
                    {"column": "c0", "use": "filter"},
                    {"column": "c0", "use": "return"},
                ],
            }],
        },
        table_sketch=sketch,
    )

    assert violations == []
    resolutions = signature.slices[0].resolution
    assert {(item.mode, item.operator) for item in resolutions} == {
        ("threshold", "gt"),
        ("exact", None),
    }
    dependencies = compresstab.compile_signature(signature).tables[0].dependencies
    assert any(
        item["kind"] == "threshold_membership" and item["operator"] == "gt"
        for item in dependencies
    )
    assert any(item["kind"] == "direct_value" for item in dependencies)


def test_compare_only_requirement_is_rejected_without_answer_target():
    compresstab = make_compresstab(
        ["race", "viewers"],
        [["2007", "10"], ["2008", "20"]],
    )
    sketch = compresstab.describe("which race had the most viewers", 800)
    _contract, _signature, violations = lower_answer_requirements(
        {
            "answer": {"shape": "value", "coverage": "all"},
            "requirements": [{
                "source": "t0",
                "scope": {"where": []},
                "fields": [{"column": "c1", "use": "compare"}],
            }],
        },
        table_sketch=sketch,
    )

    assert {item["code"] for item in violations} == {"MISSING_ANSWER_TARGET"}


class QueueLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat_json(self, **_kwargs):
        self.calls += 1
        return self.responses.pop(0), "{}"


def test_planner_audits_a_source_without_query_linked_evidence():
    sketch = {
        "sources": [
            {
                "id": "empty_relation",
                "shape": [1, 1],
                "caption": "attendance",
                "relevance": 0,
                "headers": [{"id": "c0", "path": ["event_id"], "type": "identifier"}],
                "evidence": [{"row": "r0", "values": {"c0": "1"}}],
            },
            {
                "id": "members",
                "shape": [2, 2],
                "caption": "members",
                "relevance": 4,
                "headers": [
                    {"id": "c0", "path": ["name"], "type": "text"},
                    {"id": "c1", "path": ["zip_type"], "type": "text"},
                ],
                "evidence": [{"row": "r0", "values": {"c0": "Christof", "c1": "home"}}],
            },
        ]
    }
    first = {
        "answer": {"shape": "row", "coverage": "one"},
        "requirements": [{
            "source": "empty_relation",
            "scope": {"rows": ["r0"]},
            "fields": [{"column": "c0", "use": "filter"}],
        }],
    }
    repaired = {
        "answer": {"shape": "row", "coverage": "one"},
        "requirements": [{
            "source": "members",
            "scope": {"rows": ["r0"]},
            "fields": [{"column": "c0", "use": "filter"}],
        }],
    }
    llm = QueueLLM([first, repaired])

    planner = BackwardPlannerAgent(llm, "fake")
    signature = planner.plan(question="Return Christof's zip row", table_sketch=sketch)

    assert llm.calls == 2
    assert planner.last_trace["audited"] is True
    assert planner.last_trace["final_violations"] == []
    assert signature.slices[0].source == "members"
    assert signature.slices[0].columns == ["*"]


def test_planner_independently_reviews_hierarchical_semantics():
    sketch = {
        "sources": [{
            "id": "t0",
            "shape": [2, 2],
            "relevance": 2,
            "headers": [
                {"id": "c0", "path": ["group"], "type": "text"},
                {"id": "c1", "path": ["measure", "value"], "type": "numeric"},
            ],
            "evidence": [{"row": "r0", "values": {"c0": "A", "c1": "1"}}],
            "axis_index": {
                "row_max_depth": 2,
                "question_linked_row_paths": [{"row": "r0", "path": ["parent", "A"]}],
            },
        }],
    }
    proposal = {
        "answer": {
            "shape": "value",
            "coverage": "one",
            "expression": {
                "op": "lookup",
                "args": [{"source": "t0", "column": "c1", "row": "r0", "role": "value"}],
                "unit": "number",
                "scale": 1,
                "order": "question",
            },
        },
        "requirements": [{
            "source": "t0",
            "scope": {"rows": ["r0"]},
            "fields": [{"column": "c1", "use": "return"}],
        }],
    }
    llm = QueueLLM([proposal, proposal])

    planner = BackwardPlannerAgent(llm, "fake")
    planner.plan(question="What is the value for parent A?", table_sketch=sketch)

    assert llm.calls == 2
    assert planner.last_trace["audited"] is True
    assert planner.last_trace["final_violations"] == []


def test_mschema_samples_and_candidate_cells_share_a_hard_budget():
    headers = [f"field_{index}" for index in range(120)]
    rows = [[f"value_{row}_{col}" for col in range(120)] for row in range(8)]
    rows[5][97] = "needle-person"
    compresstab = make_compresstab(headers, rows, "Find needle-person")

    sketch = compresstab.describe("Find needle-person", 800)

    assert approx_token_count(sketch) <= 800
    source = sketch["sources"][0]
    assert "c97" in source["candidate_columns"]
    assert any(
        "samples" in header
        for header in source["headers"]
    ) or source.get("headers_omitted", 0) > 0
    assert any(
        "c97" in row["values"]
        for row in source["evidence"]
    )


def test_sketch_budget_uses_configured_token_counter():
    headers = [f"identifier_{index}" for index in range(24)]
    rows = [[f"long-value-{row}-{col}" for col in range(24)] for row in range(12)]
    observed = []

    def count_rendered(value):
        assert isinstance(value, str)
        observed.append(value)
        return approx_token_count(value)

    source = {
        "question_id": "q",
        "question": "identifier 23",
        "table": [headers, *rows],
    }
    compresstab = CompressTab(
        TableStore(artifact_from_dataset_row(source)),
        token_counter=count_rendered,
    )

    sketch = compresstab.describe("identifier 23", 800)

    assert observed
    assert approx_token_count(observed[-1]) <= 800
    assert approx_token_count(sketch) <= 800



def test_positional_source_alias_and_serialized_in_list_are_lowered():
    sketch = {
        "sources": [{
            "id": "district",
            "shape": [2, 2],
            "headers": [
                {"id": "c0", "path": ["district_id"], "type": "numeric"},
                {"id": "c1", "path": ["name"], "type": "text"},
            ],
            "evidence": [],
        }],
    }

    _contract, signature, violations = lower_answer_requirements(
        {
            "answer": {"shape": "list", "coverage": "all"},
            "requirements": [{
                "source": "t0",
                "scope": {"where": [{"column": "c0", "op": "in", "value": "[29, 34]"}]},
                "fields": [{"column": "c1", "use": "return"}],
            }],
        },
        table_sketch=sketch,
    )

    assert violations == []
    assert signature.slices[0].source == "district"
    assert signature.slices[0].where[0].value == [29, 34]


def test_expression_closure_reuses_a_positional_source_requirement():
    sketch = {
        "sources": [{
            "id": "district",
            "shape": [2, 2],
            "headers": [
                {"id": "c0", "path": ["name"], "type": "text"},
                {"id": "c1", "path": ["value"], "type": "numeric"},
            ],
            "evidence": [],
        }],
    }
    _contract, signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "value",
                "coverage": "one",
                "expression": {
                    "op": "lookup",
                    "args": [{"source": "t0", "column": "c1", "row": "r0"}],
                },
            },
            "requirements": [{
                "source": "t0",
                "scope": {"rows": ["r0"]},
                "fields": [{"column": "c0", "use": "link"}],
            }],
        },
        table_sketch=sketch,
    )

    assert violations == []
    assert len(signature.slices) == 1
    assert signature.slices[0].source == "district"
    assert signature.slices[0].columns == ["c0", "c1"]


def test_prefix_and_suffix_predicates_preserve_string_position():
    assert matches_predicate("TR407_4_7", "starts_with", "TR4")
    assert not matches_predicate("XTR407_4_7", "starts_with", "TR4")
    assert matches_predicate("TR043_8_14", "ends_with", "14")
    assert not matches_predicate("TR043_14_8", "ends_with", "14")


def test_structured_answer_unwrap_is_recursive_but_bounded():
    assert normalize_structured_answer('"[632, 10807]"') == [632, 10807]
    assert normalize_structured_answer('"ordinary text"') == '"ordinary text"'
    assert normalize_structured_answer("[malformed") == "[malformed"


def test_reader_prompt_separates_provenance_address_from_row_payload():
    assert "keep them in `refs`" in AnswerAgent.PROMPT
    assert "never insert them into `answer`" in AnswerAgent.PROMPT


def test_typed_ratio_expression_preserves_ordered_operands_and_rows():
    compresstab = make_compresstab(
        ["group", "amount"],
        [["earlier", "20"], ["later", "50"]],
    )
    sketch = compresstab.describe("What is later divided by earlier?", 800)
    contract, signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "value",
                "coverage": "one",
                "expression": {
                    "op": "ratio",
                    "args": [
                        {"source": "t0", "column": "c1", "row": "r1", "role": "numerator"},
                        {"source": "t0", "column": "c1", "row": "r0", "role": "denominator"},
                    ],
                    "unit": "ratio",
                    "scale": 1,
                    "order": "question",
                },
            },
            "requirements": [{"source": "t0", "scope": {}, "fields": []}],
        },
        table_sketch=sketch,
    )

    assert violations == []
    assert contract["expression"]["args"][0]["role"] == "numerator"
    assert contract["expression"]["args"][1]["role"] == "denominator"
    assert signature.slices[0].rows == ["r1", "r0"]
    assert signature.slices[0].columns == ["c1"]
    assert signature.slices[0].resolution[0].mode == "exact"


def test_typed_argmax_expression_closes_rank_and_return_fields():
    compresstab = make_compresstab(
        ["team", "score"],
        [["A", "20"], ["B", "50"]],
    )
    sketch = compresstab.describe("Which team has the highest score?", 800)
    contract, signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "value",
                "coverage": "all",
                "expression": {
                    "op": "argmax",
                    "args": [{"source": "t0", "column": "c1", "role": "rank"}],
                    "return": {"source": "t0", "column": "c0", "kind": "entity"},
                    "unit": "text",
                    "scale": 1,
                    "order": "descending",
                },
            },
            "requirements": [{"source": "t0", "scope": {}, "fields": []}],
        },
        table_sketch=sketch,
    )

    assert violations == []
    assert contract["expression"]["return"]["kind"] == "entity"
    assert signature.slices[0].rows == []
    assert signature.slices[0].columns == ["c1", "c0"]
    assert {item.mode for item in signature.slices[0].resolution} == {"order", "identity"}


def test_transformed_argmax_lowers_to_row_bound_derived_observable():
    compresstab = make_compresstab(
        ["club", "founded", "attendance 2022", "attendance 2023"],
        [
            ["Tigers", "1942", "1200", "1212"],
            ["Falcons", "1938", "800", "818"],
            ["Owls", "1948", "620", "638"],
        ],
    )
    sketch = compresstab.describe(
        "Which club had the largest attendance increase from 2022 to 2023?",
        800,
    )
    contract, signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "list",
                "coverage": "all",
                "expression": {
                    "op": "argmax",
                    "transform": "difference",
                    "args": [
                        {"source": "t0", "column": "c2", "role": "earlier"},
                        {"source": "t0", "column": "c3", "role": "later"},
                    ],
                    "return": {"source": "t0", "column": "c0", "kind": "entity"},
                    "unit": "number",
                    "scale": 1,
                    "order": "descending",
                },
            },
            "requirements": [{"source": "t0", "scope": {}, "fields": []}],
        },
        table_sketch=sketch,
    )

    assert violations == []
    assert contract["expression"]["transform"] == "difference"
    evidence_slice = signature.slices[0]
    assert evidence_slice.rows == []
    assert evidence_slice.columns == ["c2", "c3", "c0"]
    assert {(item.column, item.mode) for item in evidence_slice.resolution} == {
        ("c0", "identity")
    }
    observable = evidence_slice.derived[0]
    assert observable.observable_id == "v0"
    assert observable.operator == "difference"
    assert observable.mode == "order"
    assert observable.direction == "descending"
    assert {(item.column, item.role) for item in observable.operands} == {
        ("c2", "earlier"),
        ("c3", "later"),
    }


def test_rowwise_sum_extremum_uses_the_same_derived_observable_interface():
    compresstab = make_compresstab(
        ["club", "home attendance", "away attendance"],
        [["Tigers", "1200", "900"], ["Falcons", "800", "1400"]],
    )
    sketch = compresstab.describe("Which club had the largest combined attendance?", 800)
    contract, signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "value",
                "coverage": "one",
                "expression": {
                    "op": "argmax",
                    "transform": "sum",
                    "args": [
                        {"source": "t0", "column": "c1", "role": "input"},
                        {"source": "t0", "column": "c2", "role": "input"},
                    ],
                    "return": {"source": "t0", "column": "c0", "kind": "entity"},
                    "unit": "number",
                    "scale": 1,
                },
            },
            "requirements": [{"source": "t0", "scope": {}, "fields": []}],
        },
        table_sketch=sketch,
    )

    assert violations == []
    assert contract["expression"]["transform"] == "sum"
    observable = signature.slices[0].derived[0]
    assert observable.operator == "sum"
    assert observable.mode == "order"
    assert {(item.column, item.role) for item in observable.operands} == {
        ("c1", "value"), ("c2", "value"),
    }


def test_compound_rowwise_ratio_lowers_to_optional_expression_dag():
    compresstab = make_compresstab(
        ["club", "home", "away", "capacity"],
        [["Tigers", "1200", "900", "3000"], ["Falcons", "800", "1400", "2800"]],
    )
    sketch = compresstab.describe("Which club has the largest attendance ratio?", 800)
    _contract, signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "value",
                "coverage": "all",
                "expression": {
                    "op": "argmax",
                    "transform": "ratio",
                    "args": [
                        {"source": "t0", "column": "c1", "role": "numerator"},
                        {"source": "t0", "column": "c2", "role": "numerator"},
                        {"source": "t0", "column": "c3", "role": "denominator"},
                    ],
                    "return": {"source": "t0", "column": "c0", "kind": "entity"},
                    "unit": "ratio",
                    "scale": 1,
                    "order": "descending",
                },
            },
            "requirements": [{"source": "t0", "scope": {}, "fields": []}],
        },
        table_sketch=sketch,
    )

    assert violations == []
    helper, target = signature.slices[0].derived
    assert helper.operator == "sum"
    assert helper.mode == "exact"
    assert helper.required is False
    assert {(item.column, item.role) for item in helper.operands} == {
        ("c1", "value"), ("c2", "value"),
    }
    assert target.operator == "ratio"
    assert target.required is True
    assert {(item.column, item.role) for item in target.operands} == {
        (helper.observable_id, "numerator"), ("c3", "denominator"),
    }


def test_invalid_transformed_extremum_does_not_materialize_partial_expression():
    compresstab = make_compresstab(
        ["club", "attendance 2022", "attendance 2023"],
        [["Tigers", "1200", "1212"]],
    )
    sketch = compresstab.describe("largest increase", 800)
    _contract, signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "value",
                "coverage": "all",
                "expression": {
                    "op": "argmax",
                    "transform": "difference",
                    "args": [
                        {"source": "t0", "column": "c2", "role": "later"},
                    ],
                    "return": {"source": "t0", "column": "c0", "kind": "entity"},
                },
            },
            "requirements": [{"source": "t0", "scope": {}, "fields": []}],
        },
        table_sketch=sketch,
    )

    assert "INCOMPLETE_ANSWER_ARGUMENTS" in {item["code"] for item in violations}
    assert signature.slices[0].derived == []


def test_cross_column_argmax_uses_column_entity_return_axis():
    compresstab = make_compresstab(
        ["measure", "2019", "2018"],
        [["Total deferred tax assets", "115.0", "120.7"]],
    )
    sketch = compresstab.describe("In which year was the value larger?", 800)
    contract, signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "value",
                "coverage": "one",
                "expression": {
                    "op": "argmax",
                    "args": [
                        {"source": "t0", "column": "c1", "row": "r0", "role": "rank"},
                        {"source": "t0", "column": "c2", "row": "r0", "role": "rank"},
                    ],
                    "return": {"source": "t0", "kind": "column_entity"},
                    "unit": "text",
                    "scale": 1,
                    "order": "question",
                },
            },
            "requirements": [{"source": "t0", "scope": {}, "fields": []}],
        },
        table_sketch=sketch,
    )

    assert violations == []
    assert contract["expression"]["return"] == {"source": "t0", "kind": "column_entity"}
    assert signature.slices[0].rows == ["r0"]
    assert signature.slices[0].columns == ["c1", "c2"]


def test_cross_column_argmax_rejects_row_entity_return_axis():
    compresstab = make_compresstab(
        ["measure", "2019", "2018"],
        [["Total deferred tax assets", "115.0", "120.7"]],
    )
    sketch = compresstab.describe("In which year was the value larger?", 800)
    _contract, _signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "value",
                "coverage": "one",
                "expression": {
                    "op": "argmax",
                    "args": [
                        {"source": "t0", "column": "c1", "row": "r0", "role": "rank"},
                        {"source": "t0", "column": "c2", "row": "r0", "role": "rank"},
                    ],
                    "return": {"source": "t0", "column": "c0", "kind": "entity"},
                    "unit": "text",
                    "scale": 1,
                },
            },
            "requirements": [{"source": "t0", "scope": {}, "fields": []}],
        },
        table_sketch=sketch,
    )

    assert "INVALID_RETURN_AXIS" in {item["code"] for item in violations}


def test_incomplete_typed_expression_is_an_executable_plan_violation():
    compresstab = make_compresstab(["name", "value"], [["A", "10"]])
    sketch = compresstab.describe("ratio", 800)
    _contract, _signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "value",
                "coverage": "one",
                "expression": {
                    "op": "ratio",
                    "args": [{"source": "t0", "column": "c1", "row": "r0", "role": "numerator"}],
                },
            },
            "requirements": [{"source": "t0", "scope": {}, "fields": []}],
        },
        table_sketch=sketch,
    )

    assert "INCOMPLETE_ANSWER_ARGUMENTS" in {item["code"] for item in violations}


def test_scalar_numeric_operator_rejects_list_shaped_contract():
    compresstab = make_compresstab(
        ["item", "2019", "2018"], [["revenue", "108", "114"]]
    )
    sketch = compresstab.describe("What was the total revenue over both years?", 800)
    _contract, _signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "list",
                "coverage": "one",
                "expression": {
                    "op": "sum",
                    "args": [
                        {"source": "t0", "column": "c1", "row": "r0", "role": "value"},
                        {"source": "t0", "column": "c2", "row": "r0", "role": "value"},
                    ],
                    "unit": "source",
                    "scale": 1,
                },
            },
            "requirements": [{"source": "t0", "scope": {}, "fields": []}],
        },
        table_sketch=sketch,
    )

    assert "INVALID_NUMERIC_SHAPE" in {item["code"] for item in violations}


def test_operator_specific_roles_are_normalized_before_type_checking():
    compresstab = make_compresstab(
        ["item", "2019", "2018"], [["rate", "7.0%", "6.4%"]]
    )
    sketch = compresstab.describe("What is the change in the rate?", 800)
    contract, _signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "value",
                "coverage": "one",
                "expression": {
                    "op": "difference",
                    "args": [
                        {"source": "t0", "column": "c1", "row": "r0", "role": "later"},
                        {"source": "t0", "column": "c2", "row": "r0", "role": "earlier"},
                    ],
                    "unit": "percent",
                    "scale": 1,
                },
            },
            "requirements": [{"source": "t0", "scope": {}, "fields": []}],
        },
        table_sketch=sketch,
    )

    assert violations == []
    assert [item["role"] for item in contract["expression"]["args"]] == ["left", "right"]
    assert contract["expression"]["scale"] == 1


def test_scalar_arithmetic_ignores_redundant_value_return_target():
    compresstab = make_compresstab(
        ["item", "2019", "2018"], [["revenue", "45", "51"]]
    )
    sketch = compresstab.describe("What is the total revenue?", 800)
    contract, _signature, violations = lower_answer_requirements(
        {
            "answer": {
                "shape": "value",
                "coverage": "one",
                "expression": {
                    "op": "sum",
                    "args": [
                        {"source": "t0", "column": "c1", "row": "r0", "role": "input"},
                        {"source": "t0", "column": "c2", "row": "r0", "role": "input"},
                    ],
                    "return": {"source": "t0", "column": "c0", "kind": "value"},
                    "unit": "number",
                    "scale": 1,
                },
            },
            "requirements": [{"source": "t0", "scope": {}, "fields": []}],
        },
        table_sketch=sketch,
    )

    assert violations == []
    assert "return" not in contract["expression"]
    assert [item["role"] for item in contract["expression"]["args"]] == ["value", "value"]
