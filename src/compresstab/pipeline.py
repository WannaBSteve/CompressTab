"""Backward planning, evidence compilation, and counterexample-guided refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compresstab.agents import AnswerAgent, BackwardPlannerAgent, EvidenceAuditorAgent
from compresstab.compress import CompressTab
from compresstab.evidence import TableLens, merge_signatures
from compresstab.table.artifact import artifact_from_dataset_row
from compresstab.table.store import TableStore
from compresstab.workflow import (
    compile_and_encode_with_repair,
    omitted_candidate_preview,
    read_with_answer_contract,
)


@dataclass(frozen=True)
class PipelineConfig:
    model: str
    planner_model: str = ""
    auditor_model: str = ""
    evidence_target: int = 4096
    inspection_tokens: int = 2048
    max_corrections: int = 5
    signature_checks: int = 1

    def __post_init__(self):
        if not self.model or self.evidence_target <= 0 or self.inspection_tokens <= 0:
            raise ValueError("A model and positive token limits are required")
        if self.max_corrections < 0 or self.signature_checks < 0:
            raise ValueError("Correction limits must be non-negative")


@dataclass
class InferenceResult:
    answer: Any
    scale: str
    status: str
    evidence: str
    signature: dict
    events: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class CompressTabPipeline:
    def __init__(self, llm, token_counter, config: PipelineConfig):
        self.llm = llm
        self.token_counter = token_counter
        self.config = config

    def answer(self, question: str, source: dict) -> Any:
        """Return the final answer; diagnostic artifacts are available via run()."""
        return self.run(question, source).answer

    def run(self, question: str, source: dict) -> InferenceResult:
        cfg = self.config
        # Labels and benchmark answer annotations never enter inference.
        row = {key: source[key] for key in (
            "question_id", "table", "tables", "paragraphs", "title", "caption",
            "context", "table_id",
        ) if key in source}
        row["question"] = question
        row["metadata"] = {key: source.get("metadata", {})[key] for key in (
            "caption", "title",
        ) if key in source.get("metadata", {})}
        question_id = str(row.get("question_id", "question"))
        store = TableStore(artifact_from_dataset_row(row))
        compiler = CompressTab(store, token_counter=self.token_counter.count,
                               token_counter_name=str(self.token_counter.name))
        sketch = compiler.describe(question, cfg.evidence_target)
        planner_model = cfg.planner_model or cfg.model
        planner = BackwardPlannerAgent(self.llm, planner_model)
        reader = AnswerAgent(self.llm, cfg.model)
        auditor = EvidenceAuditorAgent(self.llm, cfg.auditor_model or cfg.model)
        usage_before = self.llm.token_totals()

        signature = planner.plan(question=question, table_sketch=sketch)
        signature, contract, compilation, evidence, events = compile_and_encode_with_repair(
            question=question, question_id=question_id, table_sketch=sketch,
            initial_signature=signature, compresstab=compiler,
            token_budget=cfg.evidence_target, llm=self.llm, repair_model=planner_model,
            answer_contract=planner.last_trace.get("contract"),
            planner_requirements=planner.last_trace.get("final_requirements", []),
            max_plan_verifications=cfg.signature_checks,
        )
        incumbent, _ = read_with_answer_contract(
            reader, question=question, compressed=evidence, answer_contract=contract)
        status = "UNRESOLVED"
        history = []
        for round_index in range(cfg.max_corrections):
            report = auditor.audit(
                question=question, signature=signature, witness=evidence.text,
                candidate_answer=incumbent, visible_ref_ids=evidence.evidence_ref_ids,
                answer_contract=contract,
                omitted_candidates=omitted_candidate_preview(sketch, evidence),
                history=history,
            )
            events.append({"stage": "refinement", "round": round_index,
                           "action": "audit", "report": report})
            if report["status"] == "INCONCLUSIVE":
                history.append({"audit": report})
                continue
            if report["status"] == "PASS" and incumbent.get("answer") is not None:
                status = "ACCEPTED"
                break

            proposed_signature, proposed_evidence = signature, evidence
            if report["status"] == "MISSING" or incumbent.get("answer") is None:
                lens = TableLens(store)
                inspection = auditor.plan_inspection(
                    question=question, candidate_answer=incumbent,
                    missing_report=report, signature=signature,
                    visible_witness=evidence.text, table_lens=lens.overview(compilation),
                    answer_contract=contract,
                )
                if inspection["status"] != "PROBE":
                    history.append({"audit": report, "inspection": inspection})
                    continue
                observations = lens.execute(
                    inspection["actions"],
                    visible_ref_ids=set(evidence.recoverable_source_ref_ids),
                    token_budget=cfg.inspection_tokens,
                )
                finding = auditor.inspect_findings(
                    question=question, candidate_answer=incumbent,
                    observations=observations, table_sketch=sketch, answer_contract=contract)
                if finding["status"] != "COUNTEREXAMPLE":
                    history.append({"audit": report, "finding": finding["why"]})
                    continue
                report = {key: value for key, value in finding.items() if key != "signature"}
                proposed_signature = merge_signatures(signature, finding["signature"])
                proposed_evidence = compiler.encode_signature(
                    question_id=question_id, signature=proposed_signature,
                    token_budget=cfg.evidence_target)
                events.append({"stage": "refinement", "round": round_index,
                               "action": "extend_evidence", "report": report})

            proposal, _ = read_with_answer_contract(
                reader, question=question, compressed=proposed_evidence,
                answer_contract=contract, critique=report)
            verification = auditor.audit(
                question=question, signature=proposed_signature,
                witness=proposed_evidence.text, candidate_answer=proposal,
                visible_ref_ids=proposed_evidence.evidence_ref_ids,
                answer_contract=contract, history=history)
            history.append({"audit": report, "verification": verification})
            events.append({"stage": "refinement", "round": round_index,
                           "action": "verify_revision", "report": verification})
            if verification["status"] == "PASS" and proposal.get("answer") is not None:
                signature, evidence, incumbent = proposed_signature, proposed_evidence, proposal
                status = "ACCEPTED"
                break

        usage_after = self.llm.token_totals()
        return InferenceResult(
            answer=incumbent.get("answer"), scale=str(incumbent.get("scale", "")),
            status=status, evidence=evidence.text, signature=signature.model_dump(mode="json"),
            events=events,
            usage={key: usage_after.get(key, 0) - usage_before.get(key, 0)
                   for key in ("input", "output", "total", "calls", "provider_measured_calls")},
        )
