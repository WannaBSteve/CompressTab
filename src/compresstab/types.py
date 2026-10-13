"""Small shared data types for evidence provenance."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SourceType = Literal["cell", "row", "column", "section", "paragraph", "aggregate", "derived"]


class EvidenceRef(BaseModel):
    """A readable value's source location in the original table."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ref_id: str
    table_id: str
    source_type: SourceType
    row_ids: list[str] = Field(default_factory=list)
    col_ids: list[str] = Field(default_factory=list)
    header_path: list[str] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    unit: str | None = None
    scale: str | None = None
    value_preview: str | None = None
    lineage_ref_ids: list[str] = Field(default_factory=list)
    source_cell_ids: list[str] = Field(default_factory=list)
    interpretation_status: str = "clean"
    source_hash: str

    @field_validator("ref_id", "table_id", "source_hash")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("field must be non-empty")
        return value

    @field_validator("lineage_ref_ids")
    @classmethod
    def calculations_have_sources(cls, value: list[str], info):
        if info.data.get("source_type") in {"aggregate", "derived"} and not value:
            raise ValueError("aggregate and derived evidence must identify its source values")
        return value
