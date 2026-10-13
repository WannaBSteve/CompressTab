"""Thin wrapper around the official LongTableBench structured-answer scorer."""

from __future__ import annotations

import importlib.util
import json
import statistics
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


@lru_cache(maxsize=2)
def _official_metrics(official_root_text: str):
    path = Path(official_root_text) / "eval" / "metrics.py"
    if not path.exists():
        raise FileNotFoundError(f"Official LongTableBench metric not found: {path}")
    spec = importlib.util.spec_from_file_location("compresstab_longtablebench_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official LongTableBench metric: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction_to_official_text(value: Any) -> str:
    """Serialize only the Answer Agent value in the benchmark's expected shape."""

    if isinstance(value, dict):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return repr(list(value))
    if value is None:
        return "[]"
    return repr([str(value)])


def score_official(
    gold: Any,
    prediction: Any,
    *,
    official_root: str | Path,
) -> tuple[float, float, float, str]:
    prediction_text = prediction_to_official_text(prediction)
    module = _official_metrics(str(Path(official_root).resolve()))
    precision, recall, f1 = module.cal_score3(gold, prediction_text)
    return float(precision), float(recall), float(f1), prediction_text


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def _percent(value: float) -> float:
    return round(float(value) * 100.0, 4)


def _group_scores(
    conversations: list[dict[str, Any]],
    key: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in conversations:
        grouped[str(item.get(key, "unknown"))].append(float(item["f1"]))
    return {name: _percent(_mean(values)) for name, values in sorted(grouped.items())}


def _numeric_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "sum": 0.0}
    ordered = sorted(float(value) for value in values)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))
    return {
        "mean": round(_mean(ordered), 4),
        "median": round(statistics.median(ordered), 4),
        "p95": round(ordered[p95_index], 4),
        "sum": round(sum(ordered), 4),
    }


def summarize_release(
    records: list[dict[str, Any]],
    expected_rounds: list[dict[str, Any]],
    *,
    pipeline: str,
    model: str,
    elapsed_sec: float,
) -> dict[str, Any]:
    """Aggregate round scores into official conversation-level macro F1.

    LongTableBench counts a multi-turn conversation as one seed question.  We
    therefore average its rounds first and then aggregate conversation scores.
    Missing rounds remain explicit zeros.
    """

    by_round_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: Counter[str] = Counter()
    for record in records:
        round_key = str(record.get("question_id", ""))
        if round_key in by_round_id:
            duplicate_ids[round_key] += 1
        by_round_id[round_key] = record

    expected_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for expected in expected_rounds:
        expected_by_conversation[str(expected["conversation_id"])].append(expected)

    conversations = []
    round_scores = []
    completed_rounds = 0
    for conversation_key, expected_items in expected_by_conversation.items():
        expected_items.sort(key=lambda item: int(item.get("round", 1)))
        scores = []
        completed = 0
        for expected in expected_items:
            record = by_round_id.get(str(expected["question_id"]))
            if record is None:
                score = 0.0
            else:
                score = float(record.get("official_f1", 0.0) or 0.0)
                completed += 1
                completed_rounds += 1
            scores.append(score)
            round_scores.append(score)
        first = expected_items[0]
        conversations.append({
            "conversation_id": conversation_key,
            "seed_id": first.get("seed_id"),
            "tier": first.get("tier"),
            "turn": first.get("turn"),
            "table_format": first.get("table_format"),
            "task": first.get("task"),
            "is_multi_table": bool(first.get("is_multi_table")),
            "round_count": len(expected_items),
            "completed_rounds": completed,
            "complete": completed == len(expected_items),
            "f1": _mean(scores),
        })

    by_task = _group_scores(conversations, "task")
    by_tier = _group_scores(conversations, "tier")
    by_format = _group_scores(conversations, "table_format")
    by_turn = _group_scores(conversations, "turn")
    table_scope = {
        "multi": _percent(_mean(item["f1"] for item in conversations if item["is_multi_table"])),
        "single": _percent(_mean(item["f1"] for item in conversations if not item["is_multi_table"])),
    }
    task_macro = _mean(score / 100.0 for score in by_task.values())
    conversation_macro = _mean(item["f1"] for item in conversations)
    format_values = list(by_format.values())

    token_fields = (
        "raw_context_tokens",
        "compressed_table_tokens",
        "llm_input_tokens",
        "llm_output_tokens",
        "llm_total_tokens",
        "llm_calls",
    )
    token_stats = {}
    for field in token_fields:
        values = [
            float((record.get("token_stats") or {}).get(field, 0.0) or 0.0)
            for record in by_round_id.values()
        ]
        token_stats[field] = _numeric_stats(values)
    latencies = [float(record.get("latency_sec", 0.0) or 0.0) for record in by_round_id.values()]

    # One canonical CompressTab execution may serve several format-equivalent
    # benchmark records. Quality remains format-expanded, while online cost is
    # counted once per actual execution and amortized over all expected rounds.
    by_execution: dict[str, dict[str, Any]] = {}
    for round_key, record in by_round_id.items():
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        execution_key = str(metadata.get("execution_source_question_id") or round_key)
        current = by_execution.get(execution_key)
        if current is None or str(record.get("question_id")) == execution_key:
            by_execution[execution_key] = record

    def execution_total(field: str) -> float:
        return sum(
            float((record.get("token_stats") or {}).get(field, 0.0) or 0.0)
            for record in by_execution.values()
        )

    expected_count = len(expected_rounds)
    execution_latencies = [
        float(record.get("latency_sec", 0.0) or 0.0)
        for record in by_execution.values()
    ]
    online_cost = {
        "executed_rounds": len(by_execution),
        "format_reused_rounds": max(0, completed_rounds - len(by_execution)),
        "execution_reuse_factor": round(completed_rounds / max(1, len(by_execution)), 6),
        "llm_input_tokens_total": round(execution_total("llm_input_tokens"), 4),
        "llm_output_tokens_total": round(execution_total("llm_output_tokens"), 4),
        "llm_total_tokens_total": round(execution_total("llm_total_tokens"), 4),
        "llm_calls_total": round(execution_total("llm_calls"), 4),
        "provider_measured_calls_total": round(execution_total("provider_measured_calls"), 4),
        "llm_input_tokens_per_expected_round": round(
            execution_total("llm_input_tokens") / max(1, expected_count), 4
        ),
        "llm_output_tokens_per_expected_round": round(
            execution_total("llm_output_tokens") / max(1, expected_count), 4
        ),
        "llm_total_tokens_per_expected_round": round(
            execution_total("llm_total_tokens") / max(1, expected_count), 4
        ),
        "llm_calls_per_expected_round": round(
            execution_total("llm_calls") / max(1, expected_count), 6
        ),
        "latency_sec_per_execution": _numeric_stats(execution_latencies),
        "latency_sec_amortized_per_expected_round": round(
            sum(execution_latencies) / max(1, expected_count), 4
        ),
    }

    completed_conversations = sum(bool(item["complete"]) for item in conversations)
    return {
        "dataset": "longtablebench",
        "protocol": "official-release953",
        "pipeline": pipeline,
        "model": model,
        "expected_conversations": len(conversations),
        "completed_conversations": completed_conversations,
        "conversation_coverage": round(completed_conversations / max(1, len(conversations)), 6),
        "expected_rounds": expected_count,
        "completed_rounds": completed_rounds,
        "round_coverage": round(completed_rounds / max(1, expected_count), 6),
        "failed_rounds": expected_count - completed_rounds,
        "conversation_macro_f1": _percent(conversation_macro),
        "round_macro_f1": _percent(_mean(round_scores)),
        "task_macro_f1": _percent(task_macro),
        "by_task_f1": by_task,
        "by_format_f1": by_format,
        "by_length_f1": {
            "8K": by_tier.get("8k", 0.0),
            "32K": by_tier.get("32k", 0.0),
            "128K": by_tier.get("inf", 0.0),
        },
        "by_turn_f1": by_turn,
        "by_table_scope_f1": table_scope,
        "format_range": round(max(format_values) - min(format_values), 4) if format_values else 0.0,
        "token_stats": token_stats,
        "online_cost": online_cost,
        "latency_sec": _numeric_stats(latencies),
        "wall_time_sec": round(float(elapsed_sec), 3),
        "duplicate_stream_round_ids": dict(sorted(duplicate_ids.items())),
        "aggregation": {
            "round_to_conversation": "mean F1 over all expected rounds; missing rounds score zero",
            "conversation_macro": "mean conversation F1 over release records and formats",
            "task_macro": "unweighted mean of the six task F1 values, matching paper Table 1 Avg",
            "release_note": (
                "Official commit contains 953 conversation seeds; the paper's unreleased 850-seed "
                "snapshot cannot be reconstructed without an author manifest."
            ),
        },
    }


def badcase_records(
    records: list[dict[str, Any]],
    expected_rounds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest = {str(record.get("question_id")): record for record in records}
    badcases = []
    for expected in expected_rounds:
        round_key = str(expected["question_id"])
        record = latest.get(round_key)
        if record is None:
            badcases.append({**expected, "failure": "missing_prediction", "official_f1": 0.0})
        elif float(record.get("official_f1", 0.0) or 0.0) < 1.0:
            badcases.append(record)
    return badcases


def dump_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
