"""Portable JSONL inference and compilation entry point."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
import json
from pathlib import Path

from compresstab.compress import CompressTab
from compresstab.pipeline import CompressTabPipeline, PipelineConfig
from compresstab.llm.client import JsonLLMClient
from compresstab.table.artifact import artifact_from_dataset_row
from compresstab.table.store import TableStore
from compresstab.tokenization import ExactTokenCounter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["compile", "answer", "raw"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json or its directory")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    args = parser.parse_args()
    counter = ExactTokenCounter(args.tokenizer)
    tokenizer_sha256 = hashlib.sha256(counter.path.read_bytes()).hexdigest()
    settings = json.loads(args.config.read_text()) if args.config else {}
    if args.model:
        settings["model"] = args.model
    llm = pipeline = None
    if args.command != "compile":
        config = PipelineConfig(**settings)
        llm = JsonLLMClient(base_url=args.base_url,
                            max_completion_tokens=args.max_completion_tokens)
        pipeline = CompressTabPipeline(llm, counter, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    histories = {}
    with args.input.open(encoding="utf-8") as source, args.output.open("x", encoding="utf-8") as out:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row["question"])
            conversation_id = row.get("conversation_id")
            if conversation_id:
                previous = histories.setdefault(conversation_id, [])
                if row.get("round") != len(previous) + 1:
                    raise ValueError("Conversation rounds must be supplied once in source order")
                if previous:
                    question = "Conversation history:\n" + "\n".join(
                        json.dumps(item, ensure_ascii=False) for item in previous
                    ) + "\nCurrent question:\n" + question
            if row.get("external_context"):
                question = str(row["external_context"]) + "\n\n" + question
            if args.command == "answer":
                result = asdict(pipeline.run(question, row))
            elif args.command == "raw":
                compiler = CompressTab(TableStore(artifact_from_dataset_row(row)),
                                       token_counter=counter.count, token_counter_name=counter.name)
                before = llm.token_totals()
                response, _ = llm.chat_json(
                    model=settings["model"], temperature=0,
                    system_prompt='Answer from the supplied table evidence. Return JSON: {"answer": ..., "scale": ""}. Use scale for an explicit unit such as percent, thousand, million, or billion.',
                    user_prompt=question + "\n\n" + str(row.get("raw_context") or compiler.raw_context()))
                after = llm.token_totals()
                result = {"answer": response.get("answer"),
                          "scale": response.get("scale", ""),
                          "usage": {key: after[key] - before[key] for key in before}}
            else:
                compiler = CompressTab(TableStore(artifact_from_dataset_row(row)),
                                       token_counter=counter.count, token_counter_name=counter.name)
                evidence = compiler.encode_signature(
                    question_id=str(row.get("question_id", "example")),
                    signature=row["answerability_signature"],
                    token_budget=int(settings.get("evidence_target", 4096)))
                result = {"evidence": evidence.text, "tokens": evidence.token_count,
                          "signature": evidence.answerability_signature,
                          "trace": evidence.encoding_trace}
            result["question_id"] = str(row.get("question_id", "example"))
            result["tokenizer_sha256"] = tokenizer_sha256
            if conversation_id:
                histories[conversation_id].append({"question": row["question"],
                                                   "answer": result.get("answer")})
            out.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            out.flush()
