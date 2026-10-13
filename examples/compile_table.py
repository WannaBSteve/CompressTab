"""Compile the example's supplied signature without a model call."""

import argparse
import json
from pathlib import Path

from compresstab import CompressTab, ExactTokenCounter
from compresstab.table.artifact import artifact_from_dataset_row
from compresstab.table.store import TableStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    row = json.loads(Path(__file__).with_name("compile.jsonl").read_text(encoding="utf-8").strip())
    counter = ExactTokenCounter(args.tokenizer)
    compiler = CompressTab(
        TableStore(artifact_from_dataset_row(row)),
        token_counter=counter.count,
        token_counter_name=counter.name,
    )
    evidence = compiler.encode_signature(
        question_id=row["question_id"],
        signature=row["answerability_signature"],
        token_budget=4096,
    )
    print(evidence.text)
    print(f"\nEvidence tokens: {evidence.token_count}")


if __name__ == "__main__":
    main()
