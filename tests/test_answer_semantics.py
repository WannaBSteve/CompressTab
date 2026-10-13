"""Answer and scope checks on synthetic tables, without model or dataset access."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from compresstab.agents import AnswerAgent, EvidenceAuditorAgent, refinement_obligations
from compresstab.evidence import AnswerabilitySignature
from compresstab.workflow import materialization_certificate, read_with_answer_contract


class Client:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def chat_json(self, **kwargs):
        self.requests.append(kwargs)
        return deepcopy(self.response), ""


def evidence():
    return SimpleNamespace(text="synthetic source", evidence_ref_ids=[], evidence_refs=[])


@pytest.mark.parametrize("question,answer,shape", [
    ("Which years correspond to the two observations, respectively?", ["2022", "2022"], "list"),
    ("Which fiscal years are shown?", ["FY 2022", "FY 2023"], "list"),
    ("Which years are shown?", ["1890", "2101"], "list"),
    ("Which station?", "Station (2)", "value"),
    ("Which stations?", ["Station (2)", "Station (3)"], "list"),
    ("What is the stored label?", "[1, 2]", "value"),
    ("What are the two values?", [0, 0], "list"),
    ("What is the stored flag?", False, "value"),
])
def test_output_checks_preserve_valid_payloads(question, answer, shape):
    client = Client({"status": "answered", "verification": "verified", "answer": answer})
    result, _ = read_with_answer_contract(
        AnswerAgent(client, "mock"), question=question, compressed=evidence(),
        answer_contract={"shape": shape, "multiplicity": "positional"},
    )
    assert result["answer"] == answer
    assert type(result["answer"]) is type(answer)
    assert len(client.requests) == 1


def test_textual_extremum_uses_reasoning_not_extractive_routing():
    client = Client({"status": "answered", "verification": "verified", "answer": "B"})
    contract = {"shape": "value", "expression": {
        "op": "argmax", "unit": "text", "return": {"kind": "entity"},
    }}
    AnswerAgent(client, "mock").answer(
        question="Which item is largest?", compressed_table="A: 1; B: 2",
        source_ref_ids=[], answer_contract=contract,
    )
    assert client.requests[0]["system_prompt"] == AnswerAgent.PROMPT
    assert "extractive_span" not in refinement_obligations(contract)["certificate_modes"]


@pytest.mark.parametrize("response", [
    {}, {"answer": "A"},
    {"status": "answered", "verification": "unknown", "answer": "A"},
    {"status": "blocked", "verification": "verified", "answer": "A"},
    {"status": "answered", "verification": "verified", "answer": None},
])
def test_invalid_reader_status_never_verifies_answer(response):
    result = AnswerAgent(Client(response), "mock").answer(
        question="Which item?", compressed_table="A", source_ref_ids=[],
    )
    assert result["verification"] == "MISSING"
    assert result["answer"] is None


@pytest.mark.parametrize("response", [
    {}, {"status": "unknown"}, {"status": "inconclusive"},
    {"status": "counterexample", "refs": ["not-visible"]},
    {"status": "counterexample", "refs": ["t0:r0:c0"]},
    {"status": "counterexample", "refs": ["t0:r0:c0"], "certificate": {"mode": "unsupported"}},
    {"status": "counterexample", "refs": ["t0:r0:c0"], "certificate": {"mode": "list_order"}},
])
def test_invalid_audit_is_inconclusive_not_pass(response):
    result = EvidenceAuditorAgent(Client(response), "mock").audit(
        question="What value?", signature=AnswerabilitySignature(), witness="5",
        candidate_answer={"answer": 5}, visible_ref_ids=["t0:r0:c0"],
        answer_contract={"shape": "value"},
    )
    assert result["status"] == "INCONCLUSIVE"


@pytest.mark.parametrize("response,expected", [
    ({"status": "pass"}, "PASS"),
    ({"status": "missing", "missing": "The unit is absent"}, "MISSING"),
    ({"status": "counterexample", "refs": ["t0:r0:c0"],
      "certificate": {"mode": "source_binding", "claim": "The cell differs"}}, "COUNTEREXAMPLE"),
])
def test_valid_audit_decisions_are_retained(response, expected):
    result = EvidenceAuditorAgent(Client(response), "mock").audit(
        question="What value?", signature=AnswerabilitySignature(), witness="5",
        candidate_answer={"answer": 6}, visible_ref_ids=["t0:r0:c0"],
    )
    assert result["status"] == expected


def certificate(question, context, selected_rows, where, *, shape="list", op="lookup"):
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "source": "t0", "columns": ["c0", "c1"], "where": where,
    }]})
    table = SimpleNamespace(table_id="t0", row_ids=selected_rows,
                            col_ids=["c0", "c1"], dependencies=[])
    return materialization_certificate(
        question=question,
        table_sketch={"sources": [{
            "id": "t0", "shape": [2, 2],
            "headers": [{"id": "c0", "path": ["Company"]},
                        {"id": "c1", "path": ["Sales"]}],
            "contexts": [{"value": context}],
        }]},
        answer_contract={"shape": shape, "coverage": "all", "expression": {"op": op}},
        requirements=[], signature=signature,
        compilation=SimpleNamespace(diagnostics=[], tables=[table]),
    )


@pytest.mark.parametrize("year,threshold", [(2010, 10), (2020, 20), (2030, 30)])
def test_context_number_overlap_does_not_override_predicate(year, threshold):
    report = certificate(
        f"Which companies had sales above {threshold}?",
        f"The table lists sales for {year}.", ["r1"],
        [{"col": "c1", "op": "gt", "value": threshold}],
    )
    assert report["conflicts"] == []
    assert report["sources"][0]["complete_predicate"][0]["value"] == threshold


def test_exclusion_is_not_rejected_based_on_absence_of_label_in_question():
    report = certificate("Which active companies are listed?", "", ["r1"],
                         [{"col": "c0", "op": "ne", "value": "Inactive company"}])
    assert report["conflicts"] == []


def test_zero_match_count_is_a_valid_population():
    report = certificate("How many sales exceed 100?", "", [],
                         [{"col": "c1", "op": "gt", "value": 100}],
                         shape="value", op="count")
    assert report["conflicts"] == []


def test_numeric_scale_metadata_does_not_change_answer():
    from compresstab.workflow import normalize_typed_answer_metadata

    candidate = {"verification": "VERIFIED", "answer": "12%", "scale": ""}
    normalized, correction = normalize_typed_answer_metadata(candidate, {}, evidence())
    assert normalized["answer"] == candidate["answer"]
    assert normalized["scale"] == "percent"
    assert correction["answer_unchanged"]


def test_distinct_labels_are_not_merged_by_case():
    from compresstab.workflow import answer_surface_contract_violation

    assert answer_surface_contract_violation(
        question="Which labels?",
        candidate={"verification": "VERIFIED", "answer": ["a", "A"]},
        answer_contract={"shape": "list", "multiplicity": "distinct"},
        compressed=evidence(),
    ) is None
