"""Run the actual three-stage pipeline with deterministic model responses."""

import json
from copy import deepcopy
from types import SimpleNamespace

from compresstab.agents import AnswerAgent, BackwardPlannerAgent, EvidenceAuditorAgent, SignatureAuditorAgent
from compresstab.pipeline import CompressTabPipeline, PipelineConfig


class Model:
    def __init__(self):
        self.requests = []

    def token_totals(self):
        return {"calls": len(self.requests)}

    def chat_json(self, *, system_prompt, **kwargs):
        self.requests.append({"system_prompt": system_prompt, **kwargs})
        plan = {
            "answer": {"shape": "value", "coverage": "one", "expression": {
                "op": "lookup", "args": [{"source": "t0", "column": "c1", "row": "r0"}],
                "unit": "number",
            }},
            "requirements": [{"source": "t0", "scope": {"rows": ["r0"]}, "fields": [
                {"column": "c0", "use": "link"}, {"column": "c1", "use": "return"},
            ]}],
        }
        if system_prompt in {BackwardPlannerAgent.PROMPT, SignatureAuditorAgent.PROMPT}:
            return deepcopy(plan), ""
        if system_prompt == AnswerAgent.PROMPT:
            assert "5" in json.loads(kwargs["user_prompt"])["compressed_table"]
            return {"status": "answered", "verification": "verified", "answer": 5,
                    "refs": ["t0:r0:c1"]}, ""
        if system_prompt == EvidenceAuditorAgent.PROMPT:
            return {"status": "pass", "why": "The cited value agrees"}, ""
        raise AssertionError("Unexpected model request")


def test_full_pipeline_is_independent_of_question_ids_and_reference_annotations():
    outputs = []
    requests = []
    for index in range(2):
        model = Model()
        pipeline = CompressTabPipeline(
            model, SimpleNamespace(count=len, name="character-test-double"),
            PipelineConfig(model="mock", max_corrections=1),
        )
        source = {
            "question_id": f"arbitrary-{index}", "table": [["Item", "Value"], ["A", 5]],
            "answer": f"REFERENCE-{index}", "gold_answer": f"GOLD-{index}",
            "metadata": {"gold": f"ANNOTATION-{index}"},
            "task": f"CATEGORY-{index}",
        }
        result = pipeline.run("What is A's value?", source)
        assert result.answer == 5 and result.status == "ACCEPTED"
        assert result.usage["calls"] == len(model.requests)
        serialized = json.dumps(model.requests)
        for prefix in ("REFERENCE-", "GOLD-", "ANNOTATION-", "CATEGORY-", "arbitrary-"):
            assert prefix not in serialized
        outputs.append((result.answer, result.signature, result.evidence))
        requests.append(model.requests)
    assert outputs[0] == outputs[1]
    assert requests[0] == requests[1]
