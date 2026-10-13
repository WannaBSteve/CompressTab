"""Evaluate predictions by ID with the pinned benchmark's official scorer."""

import argparse
from collections import defaultdict
import csv
import importlib.util
import json
from pathlib import Path
import sys
import types


def records(path):
    indexed = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["question_id"])
            if key in indexed:
                raise ValueError(f"Duplicate question_id: {key}")
            indexed[key] = row
    return indexed


def load_official(path, name):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wtq_evaluator(root):
    # The official scorer is Python 2. Translate syntax, not scoring rules.
    try:
        from fissix.refactor import RefactoringTool, get_fixers_from_package
        package = "fissix.fixes"
    except ImportError:
        from lib2to3.refactor import RefactoringTool, get_fixers_from_package
        package = "lib2to3.fixes"
    path = root / "evaluator.py"
    translated = str(RefactoringTool(get_fixers_from_package(package)).refactor_string(
        path.read_text(encoding="utf-8") + "\n", str(path)))
    module = types.ModuleType("official_wtq_evaluator")
    exec(compile(translated, str(path), "exec"), module.__dict__)
    return module


def evaluate(dataset, references, predictions, root):
    if not references:
        raise ValueError("Reference set is empty")
    extras = set(predictions) - set(references)
    if extras:
        raise ValueError(f"Predictions contain {len(extras)} IDs outside the reference set")
    if dataset == "wtq":
        scorer = wtq_evaluator(root)
        tagged_path = root / "tagged/data/pristine-unseen-tables.tagged"
        tagged_targets = {}
        if tagged_path.exists():
            with tagged_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    tagged_targets[row["id"]] = scorer.to_value_list(
                        scorer.tsv_unescape_list(row["targetValue"]),
                        scorer.tsv_unescape_list(row["targetCanon"]))
    elif dataset == "hitab":
        sys.path.insert(0, str(root))
        scorer = load_official(root / "qa/table/utils.py", "official_hitab")
    elif dataset == "tatqa":
        scorer = load_official(root / "tatqa_metric.py", "official_tatqa")
    scores, conversations, tasks = [], defaultdict(list), {}
    for key, gold in references.items():
        prediction = predictions.get(key, {})
        answer = prediction.get("answer")
        if key not in predictions or answer is None:
            em = f1 = 0.0
        elif dataset == "wtq":
            values = answer if isinstance(answer, list) else str(answer).split("|")
            targets = gold["answer"] if isinstance(gold["answer"], list) else [gold["answer"]]
            # Official annotations preserve numeric and date denotations.
            target_values = (
                tagged_targets[key] if key in tagged_targets else
                scorer.to_value_list([str(v) for v in targets]))
            em = f1 = float(scorer.check_denotation(
                target_values,
                scorer.to_value_list([str(v) for v in values])))
        elif dataset == "hitab":
            from compresstab.datasets.hitab_metrics import normalize_hitab_prediction
            em = f1 = float(scorer.hmt_score(normalize_hitab_prediction(answer), gold["answer"]))
        elif dataset == "tatqa":
            from compresstab.datasets.tatqa_metrics import align_tatqa_prediction
            if not gold.get("answer_type"):
                raise ValueError("TAT-QA references must include answer_type")
            metric = scorer.TaTQAEmAndF1()
            answer, scale = align_tatqa_prediction(answer, prediction.get("scale", ""))
            metric(ground_truth={**gold, "uid": key}, prediction=answer, pred_scale=scale)
            raw = metric.get_raw()[0]
            em, f1 = float(raw["em"]), float(raw["f1"])
        else:
            from compresstab.datasets.longtablebench_metrics import score_official
            _, _, f1, _ = score_official(gold["answer"], answer, official_root=root)
            em = None
        scores.append((em, f1))
        if dataset == "longtablebench":
            group = str(gold["conversation_id"])
            conversations[group].append(f1)
            tasks[group] = gold["task"]
    report = {"dataset": dataset, "expected": len(references), "predicted": len(predictions),
              "missing": len(references) - len(predictions),
              "missing_prediction_score": 0,
              "f1": 100 * sum(item[1] for item in scores) / len(scores)}
    if dataset != "longtablebench":
        report["accuracy" if dataset in {"wtq", "hitab"} else "em"] = (
            100 * sum(item[0] for item in scores) / len(scores))
        if dataset in {"wtq", "hitab"}:
            del report["f1"]
    else:
        report["round_macro_f1"] = report.pop("f1")
        means = {key: sum(values) / len(values) for key, values in conversations.items()}
        by_task = defaultdict(list)
        for key, score in means.items():
            by_task[tasks[key]].append(score)
        report["conversation_macro_f1"] = 100 * sum(means.values()) / len(means)
        report["by_task_f1"] = {task: 100 * sum(values) / len(values) for task, values in sorted(by_task.items())}
        report["task_macro_f1"] = sum(report["by_task_f1"].values()) / len(by_task)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["wtq", "hitab", "tatqa", "longtablebench"])
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.dataset, records(args.references), records(args.predictions), args.official_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
