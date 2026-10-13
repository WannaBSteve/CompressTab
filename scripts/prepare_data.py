"""Prepare model inputs and reference answers as separate JSONL files."""

import argparse
import hashlib
import json
from pathlib import Path

from compresstab.datasets.hitab_data import load_hitab_rows
from compresstab.datasets.tatqa_data import load_tatqa_rows
from compresstab.datasets.tabular_adapter import load_wtq_rows
from compresstab.datasets.longtablebench_data import (
    TABLE_FORMATS, canonical_tables, load_official_conversations, render_raw_context,
)


def serializable(value):
    if hasattr(value, "headers") and hasattr(value, "rows"):
        return [value.headers, *value.rows]
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serializable(v) for v in value]
    return value


def prepare(dataset, path, split, formats, manifest):
    if dataset == "longtablebench":
        ids = set(json.loads(manifest.read_text())["ids"]) if manifest else None
        conversations = load_official_conversations(path)
        if ids is not None and ids - {item.seed_id for item in conversations}:
            raise ValueError("Sample manifest contains IDs absent from the source release")
        for conversation in conversations:
            if ids is not None and conversation.seed_id not in ids:
                continue
            for fmt in formats:
                tables = canonical_tables(path, conversation.record, fmt)
                conversation_id = f"{conversation.seed_id}:{fmt}"
                for index, qa in enumerate(conversation.rounds, 1):
                    qid = f"{conversation_id}:r{index}"
                    yield ({"question_id": qid, "question": qa["question"],
                            "tables": tables, "conversation_id": conversation_id,
                            "round": index, "tier": conversation.tier,
                            "task": conversation.task,
                            "external_context": conversation.record.get("evidence", ""),
                            "raw_context": render_raw_context(path, conversation.record, fmt)},
                           {"question_id": qid, "answer": qa.get("answer"),
                            "conversation_id": conversation_id, "task": conversation.task})
        return
    if dataset == "wtq":
        rows = load_wtq_rows(path)
    elif dataset == "hitab":
        rows = load_hitab_rows(path, split=split)
    else:
        rows = load_tatqa_rows(path)
    for row in rows:
        inputs = {key: serializable(row[key]) for key in (
            "question_id", "question", "table", "tables", "paragraphs",
        ) if key in row}
        # Preserve source captions without copying answer annotations.
        inputs["metadata"] = {key: row.get("metadata", {})[key] for key in (
            "caption", "title",
        ) if key in row.get("metadata", {})}
        reference = {"question_id": row["question_id"], "answer": row.get("answer"),
                     "scale": row.get("scale", "")}
        if dataset == "tatqa":
            reference.update({key: row.get(key, "") for key in ("answer_type", "answer_from")})
        yield inputs, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["wtq", "hitab", "tatqa", "longtablebench"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--formats", nargs="+", default=["markdown"], choices=TABLE_FORMATS)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = args.output_dir / "inputs.jsonl"
    references_path = args.output_dir / "references.jsonl"
    if any(path.exists() for path in (inputs_path, references_path, args.output_dir / "manifest.json")):
        raise FileExistsError("Output directory already contains prepared data")
    count = 0
    with inputs_path.open("x") as inputs, references_path.open("x") as references:
        for item, reference in prepare(args.dataset, args.source, args.split, args.formats, args.manifest):
            inputs.write(json.dumps(item, ensure_ascii=False) + "\n")
            references.write(json.dumps(reference, ensure_ascii=False) + "\n")
            count += 1
    manifest = {"dataset": args.dataset, "split": args.split, "rounds": count,
                "formats": args.formats if args.dataset == "longtablebench" else [],
                "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (inputs_path, references_path)}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
