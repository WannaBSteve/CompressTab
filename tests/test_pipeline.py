from types import SimpleNamespace

import compresstab.pipeline as module
from compresstab.evidence import AnswerabilitySignature


def setup(monkeypatch, reports, answers):
    signature = AnswerabilitySignature.model_validate({"slices": [{
        "source": "t0", "columns": ["c0"], "rows": ["r0"],
        "resolution": [{"column": "c0", "mode": "exact"}],
    }]})
    evidence = SimpleNamespace(text="value |\n5 |", evidence_ref_ids=["t0:r0:c0"],
                               recoverable_source_ref_ids=["t0:r0:c0"])
    seen = []

    class Planner:
        last_trace = {}
        def __init__(self, *_): pass
        def plan(self, **kwargs):
            seen.append(kwargs)
            return signature

    class Auditor:
        def __init__(self, *_): pass
        def audit(self, **kwargs):
            seen.append(kwargs)
            return reports.pop(0)

    monkeypatch.setattr(module, "BackwardPlannerAgent", Planner)
    monkeypatch.setattr(module, "EvidenceAuditorAgent", Auditor)
    monkeypatch.setattr(module, "omitted_candidate_preview", lambda *_: {})
    monkeypatch.setattr(module, "compile_and_encode_with_repair",
                        lambda **_: (signature, None, None, evidence, []))
    monkeypatch.setattr(module, "read_with_answer_contract",
                        lambda *_, **kwargs: ({"answer": answers.pop(0)}, None))
    counter = SimpleNamespace(count=len, name="character-test-double")
    client = SimpleNamespace(token_totals=lambda: {"input": 0, "output": 0, "calls": 0})
    pipeline = module.CompressTabPipeline(client, counter, module.PipelineConfig(model="mock", max_corrections=2))
    return pipeline, seen


def test_initial_accept_and_no_gold_access(monkeypatch):
    pipeline, seen = setup(monkeypatch, [{"status": "PASS"}], [5])
    result = pipeline.run("What value?", {"table": [["value"], [5]], "answer": "SECRET_GOLD"})
    assert result.answer == 5 and result.status == "ACCEPTED"
    assert "SECRET_GOLD" not in str(seen)


def test_rejected_revision_keeps_incumbent_and_retries(monkeypatch):
    pipeline, _ = setup(monkeypatch, [
        {"status": "COUNTEREXAMPLE"}, {"status": "COUNTEREXAMPLE"},
        {"status": "COUNTEREXAMPLE"}, {"status": "PASS"},
    ], [3, 4, 5])
    result = pipeline.run("What value?", {"table": [["value"], [5]]})
    assert result.answer == 5
    assert result.status == "ACCEPTED"
    assert [event["round"] for event in result.events] == [0, 0, 1, 1]


def test_correction_limit_returns_incumbent_not_rejected_proposal(monkeypatch):
    pipeline, _ = setup(monkeypatch, [{"status": "COUNTEREXAMPLE"}] * 4, [3, 4, 6])
    result = pipeline.run("What value?", {"table": [["value"], [5]]})
    assert result.answer == 3 and result.status == "UNRESOLVED"


def test_zero_correction_limit(monkeypatch):
    pipeline, _ = setup(monkeypatch, [], [5])
    pipeline.config = module.PipelineConfig(model="mock", max_corrections=0)
    assert pipeline.answer("What value?", {"table": [["value"], [5]]}) == 5


def test_inconclusive_audits_neither_accept_nor_rewrite_answer(monkeypatch):
    pipeline, _ = setup(monkeypatch, [{"status": "INCONCLUSIVE"}] * 2, [5])
    result = pipeline.run("What value?", {"table": [["value"], [5]]})
    assert result.answer == 5 and result.status == "UNRESOLVED"
    assert all(event["action"] == "audit" for event in result.events)


def test_inconclusive_revision_never_replaces_incumbent(monkeypatch):
    pipeline, _ = setup(monkeypatch, [
        {"status": "COUNTEREXAMPLE"}, {"status": "INCONCLUSIVE"},
        {"status": "COUNTEREXAMPLE"}, {"status": "INCONCLUSIVE"},
    ], [3, 4, 6])
    result = pipeline.run("What value?", {"table": [["value"], [5]]})
    assert result.answer == 3 and result.status == "UNRESOLVED"


def test_missing_evidence_is_grounded_recompiled_and_rechecked(monkeypatch):
    pipeline, seen = setup(monkeypatch, [{"status": "MISSING"}, {"status": "PASS"}], [3, 5])
    extension = AnswerabilitySignature.model_validate({"slices": [{
        "source": "t0", "columns": ["c0"], "rows": ["r1"],
        "resolution": [{"column": "c0", "mode": "exact"}],
    }]})
    class Lens:
        def __init__(self, *_): pass
        def overview(self, *_): return {"rows": 2}
        def execute(self, actions, **kwargs):
            assert kwargs["token_budget"] == pipeline.config.inspection_tokens
            return [{"source_ref": "t0:r1:c0", "value": "5"}]
    monkeypatch.setattr(module, "TableLens", Lens)
    monkeypatch.setattr(module.EvidenceAuditorAgent, "plan_inspection",
                        lambda *_, **__: {"status": "PROBE", "actions": [{}]}, raising=False)
    def findings(*_, **kwargs):
        assert kwargs["observations"][0]["value"] == "5"
        return {"status": "COUNTEREXAMPLE", "signature": extension, "why": "Missing row"}
    monkeypatch.setattr(module.EvidenceAuditorAgent, "inspect_findings", findings, raising=False)
    result = pipeline.run("Which is larger?", {"table": [["value"], [3], [5]]})
    assert result.answer == 5 and result.status == "ACCEPTED"
    assert "r1" in str(result.signature)
    assert "5" in result.evidence
    assert any(event["action"] == "extend_evidence" for event in result.events)
    assert seen[-1]["witness"] == result.evidence
