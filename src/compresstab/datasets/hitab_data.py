"""Official HiTab split loader with hierarchy-preserving table conversion."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from compresstab.datasets.tabular_adapter import SimpleTable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HITAB_ROOT = Path(os.environ.get("HITAB_ROOT", "data/raw/hitab"))
OFFICIAL_COMMIT = "d179602662b490249baf068a76fbe4137029126e"


def resolve_hitab_root(dataset_path: str | Path | None = None) -> tuple[Path, Path | None]:
    """Resolve an official repository root and an optional explicit split file."""

    path = Path(dataset_path) if dataset_path else DEFAULT_HITAB_ROOT
    split_path = path if path.is_file() else None
    root = path.parent if split_path is not None else path
    if root.name == "data":
        root = root.parent
    elif not (root / "data").is_dir() and (root.parent / "data").is_dir():
        root = root.parent
    data_dir = root / "data"
    required = [
        data_dir / "train_samples.jsonl",
        data_dir / "dev_samples.jsonl",
        data_dir / "test_samples.jsonl",
        data_dir / "tables" / "raw",
        data_dir / "tables" / "hmt",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(
            "Official HiTab data is incomplete; missing: " + ", ".join(missing)
        )
    commit_file = root / "OFFICIAL_COMMIT"
    if commit_file.is_file():
        actual = commit_file.read_text(encoding="utf-8").strip()
        if actual != OFFICIAL_COMMIT:
            raise ValueError(
                f"HiTab commit mismatch: expected {OFFICIAL_COMMIT}, got {actual}"
            )
    return root, split_path


def _tree_paths(root: dict[str, Any], expected_size: int) -> list[list[str]]:
    """Recover one explicit root-to-node path for every official line index."""

    paths: dict[int, list[str]] = {}

    def visit(node: dict[str, Any], prefix: list[str]) -> None:
        name = str(node.get("name", node.get("value", "")) or "").strip()
        current = list(prefix)
        if name and name not in {"<TOP>", "<LEFT>"}:
            current.append(name)
        line_idx = node.get("line_idx")
        if line_idx is not None:
            index = int(line_idx)
            if index in paths and paths[index] != current:
                # A merged row header can occupy multiple left-header columns.
                # HiTab represents those cells as sibling nodes with the same
                # line index (for example, "career" and "179"). Preserve every
                # label in source order as one explicit row-header path.
                paths[index] = [
                    *paths[index],
                    *(part for part in current if part not in paths[index]),
                ]
            else:
                paths[index] = current
        for child in node.get("children_dict", []) or []:
            if isinstance(child, dict):
                visit(child, current)

    visit(root, [])
    missing = [index for index in range(expected_size) if index not in paths]
    if missing:
        raise ValueError(f"HiTab hierarchy is missing line indices: {missing[:10]}")
    return [paths[index] for index in range(expected_size)]


def _hmt_data_region(data: list[list[dict[str, Any]]], width: int) -> list[list[str]]:
    """Read values from the matrix indexed by the HMT row and column trees.

    Raw HiTab tables contain merged hierarchy-label rows that are not a fixed
    prefix of the data region. Taking the last N raw rows therefore shifts
    values against ``line_idx`` on tables with nested groups. The HMT matrix is
    the official representation whose rows and columns share those indices.
    """

    result: list[list[str]] = []
    for row in data:
        if len(row or []) != width:
            raise ValueError("HiTab HMT data matrix is not rectangular")
        values = []
        for cell in row:
            value = cell.get("value", "") if isinstance(cell, dict) else cell
            if value is None:
                value = ""
            elif isinstance(value, float) and value.is_integer():
                value = int(value)
            values.append(str(value))
        result.append(values)
    return result


def canonicalize_hitab_table(
    *,
    table_id: str,
    raw: dict[str, Any],
    hmt: dict[str, Any],
) -> dict[str, Any]:
    """Return a canonical matrix plus lossless row/column hierarchy paths."""

    data = hmt.get("data", []) or []
    width = max((len(row or []) for row in data), default=0)
    if not data or width <= 0 or any(len(row or []) != width for row in data):
        raise ValueError(f"Invalid HiTab HMT data matrix for table {table_id}")
    column_paths = _tree_paths(hmt.get("top_root", {}), width)
    row_paths = _tree_paths(hmt.get("left_root", {}), len(data))
    raw_values = _hmt_data_region(data, width)

    row_header = ["row hierarchy"]
    header_paths = [row_header, *column_paths]
    matrix = [
        [" / ".join(path) or f"column {index}" for index, path in enumerate(header_paths)]
    ]
    for path, values in zip(row_paths, raw_values):
        matrix.append([" / ".join(path), *values])

    title = str(raw.get("title") or hmt.get("title") or "").strip()
    return {
        "table_id": "t0",
        "table": SimpleTable(matrix),
        "caption": title,
        "header_paths": header_paths,
        "row_header_paths": row_paths,
        "metadata": {
            "benchmark": "hitab",
            "official_table_id": table_id,
            "hierarchical": True,
            "raw_shape": [len(raw.get("texts", []) or []), max(
                (len(row or []) for row in raw.get("texts", []) or []), default=0
            )],
            "data_shape": [len(data), width],
        },
    }


def load_hitab_rows(
    dataset_path: str | Path | None = None,
    *,
    split: str = "test",
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Load one official split without exposing annotation links or formulas."""

    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown HiTab split: {split}")
    root, explicit_split = resolve_hitab_root(dataset_path)
    split_path = explicit_split or root / "data" / f"{split}_samples.jsonl"
    if not split_path.is_file():
        raise FileNotFoundError(f"Official HiTab split not found: {split_path}")
    inferred_split = split_path.name.removesuffix("_samples.jsonl")
    if inferred_split in {"train", "dev", "test"}:
        split = inferred_split

    raw_dir = root / "data" / "tables" / "raw"
    hmt_dir = root / "data" / "tables" / "hmt"
    rows: list[dict[str, Any]] = []
    table_cache: dict[str, dict[str, Any]] = {}
    with split_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            table_id = str(sample.get("table_id", "")).strip()
            question_id = str(sample.get("id", "")).strip()
            if not table_id or not question_id:
                raise ValueError(f"Invalid HiTab sample at {split_path}:{line_number}")
            if table_id not in table_cache:
                raw_path = raw_dir / f"{table_id}.json"
                hmt_path = hmt_dir / f"{table_id}.json"
                if not raw_path.is_file() or not hmt_path.is_file():
                    raise FileNotFoundError(f"Missing official HiTab table {table_id}")
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                hmt = json.loads(hmt_path.read_text(encoding="utf-8"))
                table_cache[table_id] = canonicalize_hitab_table(
                    table_id=table_id, raw=raw, hmt=hmt
                )
            table = table_cache[table_id]
            rows.append({
                "idx": len(rows) + 1,
                "question_id": question_id,
                "question": str(sample.get("question", "")).strip(),
                "answer_type": "list" if isinstance(sample.get("answer"), list) else "scalar",
                "answer": sample.get("answer", []),
                "scale": "",
                "answer_from": "table",
                "tables": [table],
                "paragraphs": [],
                "metadata": {
                    "benchmark": "hitab",
                    "split": split,
                    "table_id": table_id,
                    "table_source": str(sample.get("table_source", "")),
                    "title": table.get("caption", ""),
                    "official_commit": OFFICIAL_COMMIT,
                    "source_line": line_number,
                },
            })
            if limit > 0 and len(rows) >= limit:
                break
    return rows
