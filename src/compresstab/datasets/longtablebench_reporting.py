"""Validated LongTableBench summary export for the ACL paper tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compresstab.datasets.longtablebench_data import RELEASE_COMMIT


EXPECTED_CONVERSATIONS = 6671
EXPECTED_ROUNDS = 12397
EXPECTED_TASKS = {"BCF", "EKF", "EM", "FCM", "FR", "INI"}
EXPECTED_FORMATS = {"csv", "html", "json", "latex", "markdown", "sql", "xml"}


def load_full_summary(path: str | Path, *, pipeline: str) -> dict[str, Any]:
    path = Path(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    checks = {
        "dataset": (summary.get("dataset"), "longtablebench"),
        "protocol": (summary.get("protocol"), "official-release953"),
        "pipeline": (summary.get("pipeline"), pipeline),
        "official_commit": (summary.get("official_commit"), RELEASE_COMMIT),
        "expected_conversations": (summary.get("expected_conversations"), EXPECTED_CONVERSATIONS),
        "completed_conversations": (summary.get("completed_conversations"), EXPECTED_CONVERSATIONS),
        "expected_rounds": (summary.get("expected_rounds"), EXPECTED_ROUNDS),
        "completed_rounds": (summary.get("completed_rounds"), EXPECTED_ROUNDS),
        "failed_rounds": (summary.get("failed_rounds"), 0),
        "conversation_coverage": (summary.get("conversation_coverage"), 1.0),
        "round_coverage": (summary.get("round_coverage"), 1.0),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{name}={actual!r}, expected {expected!r}")
    tasks = set((summary.get("by_task_f1") or {}).keys())
    if tasks != EXPECTED_TASKS:
        errors.append(f"by_task_f1 keys={sorted(tasks)!r}, expected {sorted(EXPECTED_TASKS)!r}")
    formats = set((summary.get("by_format_f1") or {}).keys())
    if formats != EXPECTED_FORMATS:
        errors.append(
            f"by_format_f1 keys={sorted(formats)!r}, expected {sorted(EXPECTED_FORMATS)!r}"
        )
    if not isinstance(summary.get("online_cost"), dict):
        errors.append("online_cost is missing; rerun with execution-deduplicated cost accounting")
    if errors:
        raise ValueError(f"Refusing to export non-final summary {path}: " + "; ".join(errors))
    return summary


def _number(value: Any, *, digits: int = 2) -> str:
    return f"{float(value):,.{digits}f}"


def _macro(name: str, value: str) -> str:
    return f"\\providecommand{{\\{name}}}{{{value}}}"


def _summary_values(prefix: str, summary: dict[str, Any]) -> dict[str, str]:
    lengths = summary["by_length_f1"]
    cost = summary["online_cost"]
    tasks = summary["by_task_f1"]
    formats = summary["by_format_f1"]
    format_names = {
        "markdown": "Markdown",
        "html": "HTML",
        "json": "JSON",
        "sql": "SQL",
        "xml": "XML",
        "latex": "Latex",
        "csv": "CSV",
    }
    input_tokens = float(cost["llm_input_tokens_per_expected_round"])
    output_tokens = float(cost.get("llm_output_tokens_per_expected_round", 0.0))
    total_tokens = float(cost.get(
        "llm_total_tokens_per_expected_round",
        input_tokens + output_tokens,
    ))
    values = {
        f"{prefix}AvgFOne": _number(summary["task_macro_f1"]),
        f"{prefix}EightKFOne": _number(lengths["8K"]),
        f"{prefix}ThirtyTwoKFOne": _number(lengths["32K"]),
        f"{prefix}OneTwentyEightKFOne": _number(lengths["128K"]),
        f"{prefix}LLMInputTokens": _number(input_tokens),
        f"{prefix}LLMOutputTokens": _number(output_tokens),
        f"{prefix}LLMTotalTokens": _number(total_tokens),
        f"{prefix}Calls": _number(cost["llm_calls_per_expected_round"]),
        f"{prefix}Latency": _number(cost["latency_sec_per_execution"]["mean"]),
        f"{prefix}FormatAvgFOne": _number(sum(formats.values()) / len(formats)),
        f"{prefix}FormatRange": _number(max(formats.values()) - min(formats.values())),
    }
    values.update({f"{prefix}{task}FOne": _number(value) for task, value in tasks.items()})
    values.update({
        f"{prefix}{format_names[name]}FOne": _number(value)
        for name, value in formats.items()
    })
    return values


def _placeholder_values(prefix: str) -> dict[str, str]:
    names = {
        "AvgFOne", "EightKFOne", "ThirtyTwoKFOne", "OneTwentyEightKFOne",
        "LLMInputTokens", "LLMOutputTokens", "LLMTotalTokens", "Calls", "Latency",
        "FormatAvgFOne", "FormatRange", "TokenReduction",
        "EvidenceTokens", "Compression",
        *(f"{task}FOne" for task in EXPECTED_TASKS),
        "MarkdownFOne", "HTMLFOne", "JSONFOne", "SQLFOne", "XMLFOne",
        "LatexFOne", "CSVFOne",
    }
    return {f"{prefix}{name}": "\\texttt{xx}" for name in names}


def _paper_values(raw: dict[str, Any], comp: dict[str, Any] | None) -> dict[str, str]:
    if comp is not None and raw.get("model") != comp.get("model"):
        raise ValueError(
            f"Matched export requires one model, found {raw.get('model')!r} and {comp.get('model')!r}"
        )
    raw_tokens = float(raw["token_stats"]["raw_context_tokens"]["mean"])
    if raw_tokens <= 0:
        raise ValueError(f"Raw evidence-token mean must be positive, found {raw_tokens}")
    values = _placeholder_values("LTBComp")
    values.update(_summary_values("LTBRaw", raw))
    values.update({
        "LTBRawEvidenceTokens": _number(raw_tokens),
        "LTBRawCompression": "1.00$\\times$",
        "LTBRawTokenReduction": "0.00\\%",
        "LTBResultModel": str(raw["model"]),
    })
    if comp is None:
        return values

    comp_tokens = float(comp["token_stats"]["compressed_table_tokens"]["mean"])
    if comp_tokens <= 0:
        raise ValueError(
            f"CompressTab evidence-token mean must be positive, found {comp_tokens}"
        )
    values.update(_summary_values("LTBComp", comp))
    raw_cost = raw["online_cost"]
    comp_cost = comp["online_cost"]
    raw_total = float(raw_cost.get(
        "llm_total_tokens_per_expected_round",
        float(raw_cost["llm_input_tokens_per_expected_round"])
        + float(raw_cost.get("llm_output_tokens_per_expected_round", 0.0)),
    ))
    comp_total = float(comp_cost.get(
        "llm_total_tokens_per_expected_round",
        float(comp_cost["llm_input_tokens_per_expected_round"])
        + float(comp_cost.get("llm_output_tokens_per_expected_round", 0.0)),
    ))
    values.update({
        "LTBCompEvidenceTokens": _number(comp_tokens),
        "LTBCompCompression": f"{raw_tokens / comp_tokens:.2f}$\\times$",
        "LTBCompTokenReduction": f"{(1.0 - comp_total / raw_total) * 100.0:.2f}\\%",
    })
    return values


def render_paper_macros(raw: dict[str, Any], comp: dict[str, Any] | None = None) -> str:
    values = _paper_values(raw, comp)
    lines = [
        "% Auto-generated by scripts/export_longtablebench_paper.py.",
        "% Every populated result passed full official-release953 coverage validation.",
    ]
    lines.extend(_macro(name, value) for name, value in sorted(values.items()))
    return "\n".join(lines) + "\n"


def export_paper_macros(
    raw_summary_path: str | Path,
    compresstab_summary_path: str | Path | None,
    output_path: str | Path,
) -> dict[str, Any]:
    raw = load_full_summary(raw_summary_path, pipeline="raw-reader")
    comp = (
        load_full_summary(compresstab_summary_path, pipeline="compresstab")
        if compresstab_summary_path
        else None
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_paper_macros(raw, comp), encoding="utf-8")
    return {
        "raw_summary": str(Path(raw_summary_path).resolve()),
        "compresstab_summary": (
            str(Path(compresstab_summary_path).resolve()) if compresstab_summary_path else None
        ),
        "output": str(output_path.resolve()),
        "model": raw["model"],
        "protocol": raw["protocol"],
    }
