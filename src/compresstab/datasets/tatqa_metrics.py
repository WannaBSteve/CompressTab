import importlib.util
import json
import os
from pathlib import Path
import re
import sys


def align_tatqa_prediction(pred_answer, pred_scale):
    """Convert model output to the official evaluator's input format."""
    scale_alias = {
        "%": "percent",
        "percentage": "percent",
        "percent": "percent",
        "million": "million",
        "millions": "million",
        "billion": "billion",
        "billions": "billion",
        "thousand": "thousand",
        "thousands": "thousand",
        "thousand usd": "thousand",
        "thousands usd": "thousand",
        "thousand of usd": "thousand",
        "thousands of usd": "thousand",
        "usd thousand": "thousand",
        "usd thousands": "thousand",
        "usd 000": "thousand",
        "usd '000": "thousand",
        "usd ‘000": "thousand",
        "us$ 000": "thousand",
        "$ 000": "thousand",
        "$'000": "thousand",
        "$’000": "thousand",
    }
    # The official evaluator accepts scalar/list predictions but attempts to
    # order mappings, which raises on a malformed row-shaped model answer.
    # Preserve that output as one canonical string so it scores as incorrect
    # instead of aborting the entire evaluation.
    if isinstance(pred_answer, dict):
        pred_answer = json.dumps(pred_answer, ensure_ascii=False, sort_keys=True)
    elif isinstance(pred_answer, (list, tuple)):
        # TAT-QA sorts multi-span predictions before normalization. Model
        # outputs such as [None, 79] or mixed mappings/scalars are not
        # orderable in Python 3. Keep the malformed prediction intact as a
        # deterministic string so it scores as wrong without aborting the run.
        malformed = any(
            item is None or isinstance(item, (dict, list, tuple, set))
            for item in pred_answer
        )
        try:
            if not malformed:
                sorted(pred_answer)
        except (TypeError, ValueError):
            malformed = True
        if malformed:
            pred_answer = json.dumps(pred_answer, ensure_ascii=False, sort_keys=True)
        else:
            # The official evaluator's public interface is string/list based.
            # Numeric 0 is otherwise treated as missing, and numeric lists are
            # sorted differently from gold string lists before exact matching.
            pred_answer = [
                str(item)
                if isinstance(item, (int, float)) and not isinstance(item, bool)
                else item
                for item in pred_answer
            ]
    elif isinstance(pred_answer, (int, float)) and not isinstance(pred_answer, bool):
        pred_answer = str(pred_answer)
    pred_scale_norm = re.sub(r"[\s,]+", " ", str(pred_scale or "").strip().lower())
    pred_scale_norm = pred_scale_norm.replace("`", "'").replace("’", "'")
    return pred_answer, scale_alias.get(pred_scale_norm, pred_scale_norm)

def _resolve_tatqa_root():
    explicit = os.environ.get("TATQA_ROOT", "").strip()
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        Path("data/raw/tatqa"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "tatqa_metric.py").is_file():
            return candidate.resolve()
    checked = ", ".join(str(path) for path in candidates if path is not None)
    raise RuntimeError(f"Official TAT-QA repository not found; checked: {checked}")


def _load_tatqa_metric():
    tatqa_root = _resolve_tatqa_root()
    metric_path = tatqa_root / "tatqa_metric.py"
    if not metric_path.exists():
        raise RuntimeError(f"Official TAT-QA metric not found: {metric_path}")
    if str(tatqa_root) not in sys.path:
        sys.path.insert(0, str(tatqa_root))
    spec = importlib.util.spec_from_file_location("tatqa_metric_official", metric_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official TAT-QA metric from {metric_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def score_tatqa_row(row):
    metric_module = _load_tatqa_metric()
    metric = metric_module.TaTQAEmAndF1()
    gt = {
        "uid": row.get("question_id", ""),
        "answer_type": row.get("answer_type", ""),
        "scale": row.get("scale", ""),
        "answer": row.get("answer", ""),
        "answer_from": row.get("answer_from", "table"),
    }
    prediction, pred_scale = align_tatqa_prediction(
        row.get("pred_answer", ""), row.get("pred_scale", "")
    )
    metric(ground_truth=gt, prediction=prediction, pred_scale=pred_scale)
    raw = metric.get_raw()
    if not raw:
        return 0.0, 0.0
    return float(raw[0].get("em", 0.0)), float(raw[0].get("f1", 0.0))


def summarize_tatqa(records):
    metric_module = _load_tatqa_metric()
    metric = metric_module.TaTQAEmAndF1()
    for row in records:
        gt = {
            "uid": row.get("question_id", ""),
            "answer_type": row.get("answer_type", ""),
            "scale": row.get("scale", ""),
            "answer": row.get("answer", ""),
            "answer_from": row.get("answer_from", "table"),
        }
        prediction, pred_scale = align_tatqa_prediction(
            row.get("pred_answer", ""), row.get("pred_scale", "")
        )
        metric(ground_truth=gt, prediction=prediction, pred_scale=pred_scale)
    em, f1, scale, op = metric.get_overall_metric()
    return {
        "count": len(records),
        "em": float(em) * 100.0,
        "f1": float(f1) * 100.0,
        "scale": float(scale) * 100.0,
        "op": float(op) * 100.0,
    }


def write_tatqa_cases(path, records):
    lines = []
    for row in records:
        if float(row.get("em", 0.0)) >= 1.0:
            continue
        lines.append(f"[CASE #{row['idx']}]")
        lines.append(f"Question: {row['question']}")
        lines.append(f"Family: {row.get('compiled', {}).get('family', '')}")
        lines.append(f"Plan kind: {row.get('plan', {}).get('plan_kind', '')}")
        lines.append(f"Gold: {row.get('answer')}")
        lines.append(f"Pred: {row.get('pred_answer')}")
        lines.append(f"Scale: gold={row.get('scale','')} pred={row.get('pred_scale','')}")
        plan = row.get("plan", {})
        if plan:
            lines.append(f"Access path: {plan.get('access_path', {})}")
            lines.append(f"Cost/Risk: cost={plan.get('estimated_token_cost', '')} risk={plan.get('estimated_risk', '')}")
        if row.get("audit"):
            lines.append(f"Audit: {row.get('audit')}")
        lines.append("")
        if row.get("witness", {}).get("operator_result"):
            lines.append(f"Operator result: {row.get('witness', {}).get('operator_result', {})}")
        if row.get("witness", {}).get("evidence_summary"):
            lines.append(f"Witness summary: {row.get('witness', {}).get('evidence_summary', {})}")
        lines.append(str(row.get("rendered_witness", "")))
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
