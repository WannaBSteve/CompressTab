"""Dataset adapter protocol for TabCodec-QA."""

from __future__ import annotations

from typing import Any, Protocol


class DatasetAdapter(Protocol):
    def load_rows(self, dataset_path: str, limit: int = 0) -> list[dict[str, Any]]: ...

    def normalize_prediction(self, row: dict[str, Any]) -> dict[str, Any]: ...

    def score_row(self, row: dict[str, Any]) -> tuple[float, float]: ...

    def summarize(self, records: list[dict[str, Any]]) -> dict[str, Any]: ...
