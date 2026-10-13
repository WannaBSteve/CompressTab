"""Export the actual runtime prompt constants without model calls."""

import argparse
import json
from pathlib import Path

from compresstab import agents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = {
        "Backward Planner": [agents.BackwardPlannerAgent],
        "Signature Auditor": [agents.SignatureAuditorAgent, agents._ContextCoverageCheck],
        "Answer Agent": [agents.AnswerAgent],
        "Evidence Auditor": [agents.EvidenceAuditorAgent, agents._SourceInspector],
    }
    prompts = {
        group: {f"{cls.__name__.lstrip('_')}.{name}": value
                for cls in classes for name, value in vars(cls).items()
                if "PROMPT" in name and isinstance(value, str)}
        for group, classes in groups.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(prompts, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
