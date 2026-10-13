"""Thin wrapper around HiTab's official denotation evaluator."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from compresstab.datasets.hitab_data import DEFAULT_HITAB_ROOT


@lru_cache(maxsize=1)
def _official_hitab_utils():
    root = DEFAULT_HITAB_ROOT
    metric_path = root / "qa" / "table" / "utils.py"
    if not metric_path.is_file():
        raise RuntimeError(f"Official HiTab evaluator not found: {metric_path}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("hitab_official_table_utils", metric_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official HiTab evaluator from {metric_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_hitab_prediction(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return [normalize_hitab_prediction(item) for item in value]
    if isinstance(value, list):
        return [normalize_hitab_prediction(item) for item in value]
    if not isinstance(value, str):
        return value
    text = value.strip().strip("` ")
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return normalize_hitab_prediction(parsed)
        except json.JSONDecodeError:
            pass
    parts = [part.strip() for part in re.split(r"\s*\|\s*", text) if part.strip()]
    return parts if len(parts) > 1 else text


def score_hitab_row(row: dict[str, Any]) -> tuple[float, float]:
    pred = normalize_hitab_prediction(row.get("pred_answer", ""))
    gold = row.get("answer", [])
    if pred is None or pred == "" or gold is None or gold == "" or gold == []:
        return 0.0, 0.0
    try:
        score = float(_official_hitab_utils().hmt_score(pred, gold))
    except (IndexError, TypeError, ValueError):
        score = 0.0
    return score, score


def summarize_hitab(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    scores = [score_hitab_row(record)[0] for record in records]
    accuracy = sum(scores) / max(1, count) * 100.0
    return {
        "count": count,
        "accuracy": round(accuracy, 4),
        "em": round(accuracy, 4),
        "correct": int(sum(scores)),
        "evaluator": "official HiTab qa/table/utils.py::hmt_score",
    }


def write_hitab_cases(path: str | Path, records: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            if score_hitab_row(record)[0] >= 1.0:
                continue
            handle.write(json.dumps({
                "question_id": record.get("question_id"),
                "question": record.get("question"),
                "answer": record.get("answer"),
                "prediction": record.get("pred_answer"),
                "trace_path": record.get("trace_path"),
            }, ensure_ascii=False, default=str) + "\n")
