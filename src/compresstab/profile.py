"""Question-independent table profiles persisted for reuse across queries."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from compresstab.table.store import approx_token_count, norm_text, parse_number


PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ColumnProfile:
    col_id: str
    col_idx: int
    header: str
    header_path: list[str]
    type_hint: str
    non_empty: int
    distinct_count: int
    numeric_count: int
    ambiguous_count: int
    reversible_count: int
    missing_count: int
    avg_fragmentation: float
    cardinality_ratio: float
    entropy_bits: float
    numeric_min: float | None
    numeric_max: float | None
    numeric_variance: float | None
    missing_rate: float
    rare_value_mass: float

    @property
    def has_dirty_values(self) -> bool:
        return bool(self.ambiguous_count or self.reversible_count or self.missing_count)


def table_fingerprint(table: Any) -> str:
    payload = {
        "headers": list(table.headers),
        "header_paths": list(table.header_paths),
        "rows": list(table.rows),
        "source_rows": list(table.source_row_indices),
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ProfileStore:
    """Build each table profile once and optionally persist it by content hash."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._memory: dict[str, dict[str, ColumnProfile]] = {}
        # One CompressTab instance owns stable table objects. Avoid serializing
        # the same large table again for every per-column profile lookup.
        self._fingerprint_by_object: dict[int, str] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _fingerprint(self, table: Any) -> str:
        object_id = id(table)
        fingerprint = self._fingerprint_by_object.get(object_id)
        if fingerprint is None:
            fingerprint = table_fingerprint(table)
            self._fingerprint_by_object[object_id] = fingerprint
        return fingerprint

    def table_profiles(self, table: Any) -> dict[str, ColumnProfile]:
        fingerprint = self._fingerprint(table)
        if fingerprint in self._memory:
            return self._memory[fingerprint]
        profiles = self._load(fingerprint)
        if profiles is None:
            profiles = {
                f"c{index}": self._build_column(table, index)
                for index in range(len(table.headers))
            }
            self._save(fingerprint, table, profiles)
        self._memory[fingerprint] = profiles
        return profiles

    def profile(self, table: Any, col_id: str) -> ColumnProfile:
        profiles = self.table_profiles(table)
        if col_id not in profiles:
            raise KeyError(f"Unknown profile column {table.table_id}:{col_id}")
        return profiles[col_id]

    def describe(self, table: Any) -> dict[str, Any]:
        fingerprint = self._fingerprint(table)
        profiles = self.table_profiles(table)
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "table_id": table.table_id,
            "fingerprint": fingerprint,
            "shape": [len(table.rows), len(table.headers)],
            "columns": {key: asdict(value) for key, value in profiles.items()},
        }

    def _cache_path(self, fingerprint: str) -> Path | None:
        return self.cache_dir / f"{fingerprint}.json" if self.cache_dir else None

    def _load(self, fingerprint: str) -> dict[str, ColumnProfile] | None:
        path = self._cache_path(fingerprint)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema_version") != PROFILE_SCHEMA_VERSION:
                return None
            if data.get("fingerprint") != fingerprint:
                return None
            columns = data.get("columns", {})
            return {
                str(col_id): ColumnProfile(**payload)
                for col_id, payload in columns.items()
                if isinstance(payload, dict)
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _save(
        self,
        fingerprint: str,
        table: Any,
        profiles: dict[str, ColumnProfile],
    ) -> None:
        path = self._cache_path(fingerprint)
        if path is None:
            return
        payload = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "table_id": table.table_id,
            "shape": [len(table.rows), len(table.headers)],
            "columns": {key: asdict(value) for key, value in profiles.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{fingerprint}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    def _build_column(self, table: Any, col_idx: int) -> ColumnProfile:
        col_id = f"c{col_idx}"
        values = [row[col_idx] for row in table.rows if col_idx < len(row)]
        non_empty = [str(value).strip() for value in values if str(value).strip()]
        numeric_values = [number for value in non_empty if (number := parse_number(value)) is not None]
        statuses = [table.cell_status(row_idx, col_idx) for row_idx in range(len(table.rows))]
        frequencies = Counter(non_empty)
        distinct_count = len(frequencies)
        cardinality = distinct_count / max(1, len(non_empty))
        avg_fragmentation = (
            sum(max(1, approx_token_count(value)) for value in non_empty) / max(1, len(non_empty))
        )
        entropy = -sum(
            (count / max(1, len(non_empty))) * math.log2(count / max(1, len(non_empty)))
            for count in frequencies.values()
        )
        numeric_mean = sum(numeric_values) / max(1, len(numeric_values))
        numeric_variance = (
            sum((value - numeric_mean) ** 2 for value in numeric_values) / len(numeric_values)
            if numeric_values else None
        )
        header_low = norm_text(table.headers[col_idx])
        if len(numeric_values) >= max(2, int(0.6 * max(1, len(non_empty)))):
            type_hint = "mixed_numeric" if len(numeric_values) < len(non_empty) else "numeric"
        elif any(key in header_low for key in ("id", "uuid", "url", "code", "key", "hash")) or (
            cardinality > 0.75 and avg_fragmentation >= 1.8
        ):
            type_hint = "identifier"
        elif distinct_count <= max(3, int(0.45 * max(1, len(non_empty)))):
            type_hint = "categorical"
        else:
            type_hint = "text"
        return ColumnProfile(
            col_id=col_id,
            col_idx=col_idx,
            header=table.headers[col_idx],
            header_path=table.header_path(col_idx),
            type_hint=type_hint,
            non_empty=len(non_empty),
            distinct_count=distinct_count,
            numeric_count=len(numeric_values),
            ambiguous_count=statuses.count("ambiguous"),
            reversible_count=statuses.count("reversible"),
            missing_count=statuses.count("missing"),
            avg_fragmentation=avg_fragmentation,
            cardinality_ratio=cardinality,
            entropy_bits=entropy,
            numeric_min=min(numeric_values) if numeric_values else None,
            numeric_max=max(numeric_values) if numeric_values else None,
            numeric_variance=numeric_variance,
            missing_rate=statuses.count("missing") / max(1, len(statuses)),
            rare_value_mass=sum(count for count in frequencies.values() if count == 1) / max(1, len(non_empty)),
        )
