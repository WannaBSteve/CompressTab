"""Versioned artifacts shared by evidence preparation and answering."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from compresstab.compress import CompressedTable
from compresstab.evidence import AnswerabilitySignature, EvidenceCompilation
from compresstab.table.artifact import TableArtifact, artifact_from_dataset_row


EVIDENCE_SCHEMA_VERSION = "compresstab.evidence.v2"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_table_payload(artifact: TableArtifact) -> dict[str, Any]:
    """Return the question-independent data plane used by CompressTab.

    Dataset bookkeeping such as question type and source-line number is kept in
    prediction records, not duplicated in a content-addressed table artifact.
    Captions, contexts, structure candidates, source cells, and paragraphs are
    already represented inside the canonical tables and remain intact.
    """

    tables = []
    for table in artifact.tables:
        payload = table.model_dump(mode="json")
        payload["metadata"] = {}
        tables.append(payload)
    return {
        "tables": tables,
        "paragraphs": artifact.paragraphs,
        "metadata": {},
    }


def canonical_table_sha256(artifact: TableArtifact) -> str:
    return sha256_bytes(_canonical_bytes(canonical_table_payload(artifact)))


def write_canonical_table(output_dir: Path, artifact: TableArtifact) -> tuple[str, str]:
    """Atomically write one content-addressed canonical table artifact."""

    payload = canonical_table_payload(artifact)
    data = _canonical_bytes(payload)
    digest = sha256_bytes(data)
    canonical_dir = output_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    path = canonical_dir / f"{digest}.json"
    if path.exists():
        if sha256_bytes(path.read_bytes()) != digest:
            raise ValueError(f"Canonical artifact hash mismatch at {path}")
    else:
        fd, temporary = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=canonical_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return path.relative_to(output_dir).as_posix(), digest


def load_canonical_table(
    *,
    base_dir: Path,
    canonical_path: str,
    expected_sha256: str,
    artifact_id: str,
) -> TableArtifact:
    relative = Path(canonical_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("canonical_path must stay inside the Stage A artifact directory")
    base = base_dir.resolve()
    path = (base / relative).resolve()
    if path.parent != (base / "canonical").resolve():
        raise ValueError("canonical_path must reference the canonical directory")
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical artifact: {path}")
    data = path.read_bytes()
    actual = sha256_bytes(data)
    if actual != expected_sha256:
        raise ValueError(
            f"Canonical artifact hash mismatch: expected {expected_sha256}, got {actual}"
        )
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("Canonical artifact must be a JSON object")
    return TableArtifact.model_validate({"artifact_id": artifact_id, **payload})


class CompilationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plans: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)

    def to_compilation(self, signature: AnswerabilitySignature) -> EvidenceCompilation:
        return EvidenceCompilation.model_validate({
            "signature": signature.model_dump(mode="json"),
            "tables": self.plans,
            "diagnostics": self.diagnostics,
        })


class PreparationStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planner_model: str
    signature_source: Literal["BackwardPlannerAgent", "replay_plan", "replay", "full_table"]
    evidence_coding: Literal["optimized", "raw"]
    token_budget: int = Field(ge=1)
    prompt_hash: str
    raw_table_tokens: int = Field(ge=0)
    raw_context_tokens: int = Field(ge=0)
    llm_input_tokens: int = Field(ge=0)
    llm_output_tokens: int = Field(ge=0)
    llm_total_tokens: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    provider_measured_calls: int = Field(ge=0)
    latency_sec: float = Field(ge=0.0)
    plan_repair_model: str = ""
    max_plan_repairs: int = Field(default=0, ge=0)
    plan_repairs: int = Field(default=0, ge=0)
    max_plan_verifications: int = Field(default=0, ge=0)
    plan_verifications: int = Field(default=0, ge=0)


class PreparedEvidenceArtifact(BaseModel):
    """The complete frozen input to Stage B."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["compresstab.evidence.v2"] = EVIDENCE_SCHEMA_VERSION
    evidence_id: str
    dataset: str
    dataset_position: int = Field(ge=0)
    question_id: str
    question: str
    canonical_path: str
    canonical_sha256: str
    table_sketch: dict[str, Any]
    answer_contract: dict[str, Any] | None = None
    answerability_signature: AnswerabilitySignature
    compilation: CompilationArtifact
    witness: dict[str, Any]
    preparation: PreparationStats
    plan_repair_events: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("evidence_id", "canonical_sha256")
    @classmethod
    def valid_digest(cls, value: str) -> str:
        value = str(value or "").strip().lower()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("expected a lowercase SHA-256 digest")
        return value

    @field_validator("dataset", "question_id")
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("field must be non-empty")
        return value

    @model_validator(mode="after")
    def consistent_witness(self):
        compressed = CompressedTable.from_dict(self.witness)
        if compressed.question_id != self.question_id:
            raise ValueError("Witness question_id does not match the artifact")
        if compressed.answerability_signature != self.answerability_signature.model_dump(mode="json"):
            raise ValueError("Witness Signature does not match the artifact Signature")
        # Parsing the compilation here rejects malformed saved plans before an
        # Answer Agent can be called.
        self.compilation.to_compilation(self.answerability_signature)
        payload = self.model_dump(mode="json", exclude={"evidence_id"})
        if self.evidence_id != prepared_evidence_identity(payload):
            raise ValueError("evidence_id does not match the frozen evidence content")
        return self


def evidence_identity(payload: dict[str, Any]) -> str:
    return sha256_bytes(_canonical_bytes(payload))


def prepared_evidence_identity(payload: dict[str, Any]) -> str:
    """Hash semantic evidence content, excluding nondeterministic runtime cost."""

    def compatibility_normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: compatibility_normalize(item)
                for key, item in value.items()
                if not (
                    (key == "order_kind" and item is None)
                    or (key in {"derived", "derived_observables"} and item == [])
                )
            }
        if isinstance(value, list):
            return [compatibility_normalize(item) for item in value]
        return value

    preparation = payload.get("preparation", {})
    semantic_payload = {
        "schema_version": payload.get("schema_version"),
        "dataset": payload.get("dataset"),
        "dataset_position": payload.get("dataset_position"),
        "question_id": payload.get("question_id"),
        "question": payload.get("question"),
        "canonical_path": payload.get("canonical_path"),
        "canonical_sha256": payload.get("canonical_sha256"),
        "table_sketch": payload.get("table_sketch"),
        "answerability_signature": payload.get("answerability_signature"),
        "compilation": payload.get("compilation"),
        "witness": payload.get("witness"),
        "preparation_config": {
            "planner_model": preparation.get("planner_model"),
            "signature_source": preparation.get("signature_source"),
            "evidence_coding": preparation.get("evidence_coding"),
            "token_budget": preparation.get("token_budget"),
            "prompt_hash": preparation.get("prompt_hash"),
        },
    }
    # Keep frozen v2 artifacts created before answer-contract propagation
    # loadable while hashing the contract for every new artifact that has one.
    if payload.get("answer_contract") is not None:
        semantic_payload["answer_contract"] = payload.get("answer_contract")
    return evidence_identity(compatibility_normalize(semantic_payload))


def load_evidence_jsonl(path: str | Path) -> dict[str, PreparedEvidenceArtifact]:
    """Strictly load Stage A outputs and reject silent skips or duplicate IDs."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Evidence input not found: {source}")
    result: dict[str, PreparedEvidenceArtifact] = {}
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                artifact = PreparedEvidenceArtifact.model_validate(raw)
            except Exception as exc:
                raise ValueError(f"Invalid evidence artifact at {source}:{line_number}: {exc}") from exc
            if artifact.question_id in result:
                raise ValueError(
                    f"Duplicate question_id {artifact.question_id!r} in {source}"
                )
            result[artifact.question_id] = artifact
    if not result:
        raise ValueError(f"No evidence artifacts found in {source}")
    return result


def load_plan_replays(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the frozen semantic proposal needed for a matched verifier run."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Plan input not found: {source}")
    result: dict[str, dict[str, Any]] = {}
    with source.open(encoding="utf-8") as handle:
        raw_artifacts = [json.loads(line) for line in handle if line.strip()]
    for raw in raw_artifacts:
        question_id = str(raw.get("question_id", "") or "")
        if not question_id or question_id in result:
            raise ValueError(f"Invalid or duplicate question_id {question_id!r} in {source}")
        signature = AnswerabilitySignature.model_validate(raw.get("answerability_signature", {}))
        planning_event = next(
            (
                event for event in (raw.get("plan_repair_events") or [])
                if event.get("step") in {
                    "close_answer_requirements",
                    "load_frozen_answerability_plan",
                }
            ),
            None,
        )
        direct_requirements = raw.get("requirements")
        if planning_event is None and not isinstance(direct_requirements, list):
            raise ValueError(
                f"Evidence artifact {question_id!r} has no frozen answer-requirement proposal"
            )
        requirements = (
            direct_requirements
            if isinstance(direct_requirements, list)
            else planning_event.get("final_requirements")
        )
        if not isinstance(requirements, list):
            raise ValueError(
                f"Evidence artifact {question_id!r} has no valid final_requirements"
            )
        result[question_id] = {
            "question": str(raw.get("question", "")),
            "answer_contract": raw.get("answer_contract") or (
                planning_event.get("contract") if planning_event is not None else None
            ),
            "requirements": requirements,
            "answerability_signature": signature.model_dump(mode="json"),
        }
    if not result:
        raise ValueError(f"No replayable semantic plans found in {source}")
    return result


def validate_evidence_for_row(
    prepared: PreparedEvidenceArtifact | dict[str, Any],
    *,
    row: dict[str, Any],
    dataset_name: str,
    question_id: str,
    evidence_base_dir: Path,
) -> tuple[PreparedEvidenceArtifact, TableArtifact]:
    parsed = (
        prepared
        if isinstance(prepared, PreparedEvidenceArtifact)
        else PreparedEvidenceArtifact.model_validate(prepared)
    )
    question = str(row.get("question", ""))
    if parsed.dataset != dataset_name:
        raise ValueError(
            f"Evidence dataset mismatch: expected {dataset_name}, got {parsed.dataset}"
        )
    if parsed.question_id != question_id:
        raise ValueError(
            f"Evidence question_id mismatch: expected {question_id}, got {parsed.question_id}"
        )
    if parsed.question != question:
        raise ValueError(f"Evidence question text mismatch for {question_id}")

    stored = load_canonical_table(
        base_dir=evidence_base_dir,
        canonical_path=parsed.canonical_path,
        expected_sha256=parsed.canonical_sha256,
        artifact_id=question_id,
    )
    fresh = artifact_from_dataset_row(row)
    fresh_digest = canonical_table_sha256(fresh)
    if fresh_digest != parsed.canonical_sha256:
        raise ValueError(
            f"Dataset table differs from frozen Stage A table for {question_id}: "
            f"expected {parsed.canonical_sha256}, got {fresh_digest}"
        )
    return parsed, stored
