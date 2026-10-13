import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location("evaluation_script", Path(__file__).parents[1] / "scripts/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_duplicate_predictions_are_rejected(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('\n'.join([json.dumps({"question_id": "a", "answer": 1})] * 2))
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.records(path)


def test_unknown_predictions_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        evaluation.evaluate("wtq", {"a": {}}, {"b": {}}, Path("unused"))


def test_missing_answers_are_not_dropped(monkeypatch):
    scorer = SimpleNamespace(to_value_list=lambda values: values,
                             check_denotation=lambda gold, pred: gold == pred)
    monkeypatch.setattr(evaluation, "wtq_evaluator", lambda _: scorer)
    result = evaluation.evaluate("wtq", {"a": {"answer": ["5"]}, "b": {"answer": ["6"]}},
                                 {"a": {"answer": "5"}}, Path("unused"))
    assert result["accuracy"] == 50
    assert result["missing"] == 1


def test_wtq_uses_official_canonical_targets(monkeypatch, tmp_path):
    tagged = tmp_path / "tagged/data/pristine-unseen-tables.tagged"
    tagged.parent.mkdir(parents=True)
    tagged.write_text("id\ttargetValue\ttargetCanon\na\tfive\t5\n")
    scorer = SimpleNamespace(
        to_value_list=lambda values, canonical=None: canonical if canonical is not None else values,
        tsv_unescape_list=lambda value: value.split("|"),
        check_denotation=lambda gold, pred: gold == pred,
    )
    monkeypatch.setattr(evaluation, "wtq_evaluator", lambda _: scorer)
    result = evaluation.evaluate("wtq", {"a": {"answer": ["five"]}},
                                 {"a": {"answer": ["5"]}}, tmp_path)
    assert result["accuracy"] == 100


def test_conversation_and_task_aggregation_are_distinct(monkeypatch):
    import compresstab.datasets.longtablebench_metrics as metrics
    monkeypatch.setattr(metrics, "score_official", lambda gold, pred, **_: (0, 0, float(gold == pred), ""))
    refs = {
        "a1": {"answer": 1, "conversation_id": "a", "task": "X"},
        "a2": {"answer": 2, "conversation_id": "a", "task": "X"},
        "b1": {"answer": 3, "conversation_id": "b", "task": "X"},
        "c1": {"answer": 4, "conversation_id": "c", "task": "Y"},
    }
    result = evaluation.evaluate("longtablebench", refs, {"a1": {"answer": 1}}, Path("unused"))
    assert result["round_macro_f1"] == 25
    assert result["conversation_macro_f1"] == pytest.approx(100 / 6)
    assert result["task_macro_f1"] == 12.5


def test_tatqa_prediction_adapter_accepts_no_reference_arguments():
    from compresstab.datasets.tatqa_metrics import align_tatqa_prediction

    assert align_tatqa_prediction(0, "percentage") == ("0", "percent")
    assert align_tatqa_prediction([0, 2], "million") == (["0", "2"], "million")
    with pytest.raises(TypeError):
        align_tatqa_prediction(1, "", "reference answer", "reference scale")
