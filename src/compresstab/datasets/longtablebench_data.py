"""Official LongTableBench loading, rendering, and canonicalization.

The benchmark repository owns the question files, table files, and seven
serialization functions.  This module keeps those assets external and builds a
small, stable interface for the shared CompressTab runner.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import itertools
import json
import math
import re
from xml.etree import ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from compresstab.table.store import approx_token_count


TABLE_FORMATS = ("markdown", "html", "json", "latex", "sql", "xml", "csv")
LENGTH_TIERS = ("8k", "32k", "inf")
TURN_TYPES = ("single", "multi")
RELEASE_COMMIT = "8db45ac102632cba3bcc1393e1023f0689da7d1c"


@dataclass(frozen=True)
class OfficialConversation:
    """One record from an official question file before format expansion."""

    release_position: int
    tier_position: int
    tier: str
    turn: str
    source_path: str
    record: dict[str, Any]

    @property
    def seed_id(self) -> str:
        return f"{self.tier}:{self.turn}:{self.record['question_id']}"

    @property
    def task(self) -> str:
        return str(self.record.get("question_type", ""))

    @property
    def rounds(self) -> list[dict[str, Any]]:
        if self.turn == "multi":
            return [dict(item) for item in self.record.get("QA", [])]
        return [{
            "round": 1,
            "question": self.record.get("question", ""),
            "answer": self.record.get("answer", ""),
        }]


def _selection(value: str | Iterable[str], allowed: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        values = allowed if value.strip().lower() == "all" else (value.strip().lower(),)
    else:
        values = tuple(str(item).strip().lower() for item in value)
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown selection {unknown}; allowed values are {allowed}")
    return tuple(item for item in allowed if item in values)


def load_official_conversations(
    official_root: str | Path,
    *,
    tiers: str | Iterable[str] = "all",
    turns: str | Iterable[str] = "all",
) -> list[OfficialConversation]:
    root = Path(official_root)
    selected_tiers = _selection(tiers, LENGTH_TIERS)
    selected_turns = _selection(turns, TURN_TYPES)
    conversations: list[OfficialConversation] = []
    release_position = 0
    for tier in selected_tiers:
        for turn in selected_turns:
            path = root / "datasets" / "questions" / tier / f"longtablebench_{turn}.json"
            if not path.exists():
                raise FileNotFoundError(f"Official question file not found: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"Official question file is not a list: {path}")
            for tier_position, record in enumerate(payload, start=1):
                if not isinstance(record, dict):
                    raise ValueError(f"Invalid record {tier_position} in {path}")
                release_position += 1
                conversations.append(OfficialConversation(
                    release_position=release_position,
                    tier_position=tier_position,
                    tier=tier,
                    turn=turn,
                    source_path=str(path),
                    record=record,
                ))
    return conversations


def selected_formats(value: str | Iterable[str]) -> tuple[str, ...]:
    return _selection(value, TABLE_FORMATS)


def table_directory(official_root: str | Path, record: dict[str, Any]) -> Path:
    return Path(official_root) / "datasets" / "tables" / str(record["db_path"]) / str(record["db_id"])


def _case_insensitive_csv(directory: Path, table_name: str) -> Path:
    target = f"{table_name}.csv".lower()
    matches = sorted(path for path in directory.iterdir() if path.is_file() and path.name.lower() == target)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one case-insensitive match for {table_name}.csv in {directory}; "
            f"found {[path.name for path in matches]}"
        )
    return matches[0]


def table_paths(official_root: str | Path, record: dict[str, Any]) -> tuple[Path, ...]:
    directory = table_directory(official_root, record)
    if not directory.is_dir():
        raise FileNotFoundError(f"Official database directory not found: {directory}")
    if bool(record.get("is_multi_table")):
        paths = tuple(sorted(path for path in directory.iterdir() if path.is_file() and path.suffix == ".csv"))
        if not paths:
            raise FileNotFoundError(f"No CSV tables found in official database: {directory}")
        return paths
    highlighted = record.get("highlighted_table", [])
    if not highlighted:
        raise ValueError(f"Single-table record has no highlighted_table: {record.get('question_id')}")
    return (_case_insensitive_csv(directory, str(highlighted[0])),)


@lru_cache(maxsize=2048)
def _read_csv(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text, low_memory=False)


def source_tables(official_root: str | Path, record: dict[str, Any]) -> tuple[tuple[str, pd.DataFrame], ...]:
    return tuple((path.stem, _read_csv(str(path))) for path in table_paths(official_root, record))


@lru_cache(maxsize=2)
def _official_reformat_module(official_root_text: str):
    path = Path(official_root_text) / "eval" / "reformat.py"
    if not path.exists():
        raise FileNotFoundError(f"Official formatter not found: {path}")
    spec = importlib.util.spec_from_file_location("compresstab_longtablebench_reformat", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official formatter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The release pins pandas 2.2, while the shared experiment environment uses
    # pandas 3 where applymap was renamed to map.  This is an API alias only;
    # the official conversion lambda and all resulting values stay unchanged.
    if not hasattr(pd.DataFrame, "applymap") and hasattr(pd.DataFrame, "map"):
        pd.DataFrame.applymap = pd.DataFrame.map  # type: ignore[attr-defined]
    official_df2json = module.df2json

    def compatible_df2json(frame: pd.DataFrame) -> str:
        # orient='table' serializes a synthetic row index. Pandas names an
        # unnamed index "index" and raises when the source already has that
        # column. Give only the synthetic field a collision-free visible name.
        synthetic_name = str(frame.index.name or "index")
        source_columns = {str(column) for column in frame.columns}
        if synthetic_name in source_columns:
            candidate = "__row_index__"
            while candidate in source_columns:
                candidate = f"_{candidate}"
            frame = frame.copy()
            frame.index.name = candidate
        return official_df2json(frame)

    module.df2json = compatible_df2json
    return module


def render_raw_context(
    official_root: str | Path,
    record: dict[str, Any],
    table_format: str,
) -> str:
    fmt = table_format.strip().lower()
    if fmt not in TABLE_FORMATS:
        raise ValueError(f"Unsupported LongTableBench table format: {table_format}")
    module = _official_reformat_module(str(Path(official_root).resolve()))
    return "".join(module.trans_tables(name, frame, fmt) for name, frame in source_tables(official_root, record))


def _serialized_table(module, table_name: str, frame: pd.DataFrame, table_format: str) -> str:
    if table_format == "markdown":
        text = module.df2mk(frame)
    elif table_format == "html":
        text = module.df2html(frame)
    elif table_format == "json":
        text = module.df2json(frame)
    elif table_format == "latex":
        text = module.df2latex(frame)
    elif table_format == "sql":
        text = module.df2sql(frame, table_name)
    elif table_format == "xml":
        text = module.df2xml(frame)
    elif table_format == "csv":
        text = module.df2csv(frame)
    else:
        raise ValueError(table_format)
    # The official prompt renderer performs this normalization after every
    # serialization, so the canonical front end must parse what the model would
    # actually receive rather than a cleaner hidden representation.
    return re.sub(r" +", " ", text)


def _parse_serialized_table(module, text: str, table_format: str) -> pd.DataFrame:
    if table_format == "markdown":
        return _parse_markdown_serialization(text)
    if table_format == "html":
        return module.html2df(text)
    if table_format == "json":
        return module.json2df(text)
    if table_format == "latex":
        return _parse_latex_serialization(text)
    if table_format == "sql":
        return _parse_sql_serialization(text)
    if table_format == "xml":
        return _parse_xml_serialization(text)
    if table_format == "csv":
        return pd.read_csv(io.StringIO(text), low_memory=False)
    raise ValueError(table_format)


def _parse_markdown_serialization(text: str) -> pd.DataFrame:
    """Parse visible pipe-table syntax without inventing ambiguous boundaries.

    Pandas ``to_markdown`` leaves literal pipes inside source cells unescaped,
    whereas LongTableBench's reverse formatter splits on every pipe.  Numeric
    columns learned from already well-formed visible rows can make some of
    those boundaries uniquely recoverable.  Rows with more than one valid
    alignment remain verbatim instead of being guessed or restored from CSV.
    """

    def split_line(line: str) -> list[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [value.strip() for value in stripped.split("|")]

    lines = [line for line in text.splitlines() if line.strip()]
    separator_index = None
    separator_cells: list[str] = []
    for index, line in enumerate(lines):
        cells = split_line(line)
        if cells and all(re.fullmatch(r":?-+:?", value) for value in cells):
            separator_index = index
            separator_cells = cells
            break
    if separator_index is None or separator_index == 0:
        raise ValueError("Markdown serialization has no visible header separator")

    width = len(separator_cells)
    columns = split_line(lines[separator_index - 1])
    if len(columns) != width:
        raise ValueError(
            "Markdown header has ambiguous pipe boundaries: "
            f"{len(columns)} visible fragments for {width} columns"
        )

    raw_rows = [split_line(line) for line in lines[separator_index + 1 :]]
    exact_rows = [row for row in raw_rows if len(row) == width]
    nulls = {"", "nan", "null", "none"}

    def numeric(value: str) -> bool:
        if value.strip().lower() in nulls:
            return True
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    numeric_columns = set()
    for index in range(width):
        values = [row[index] for row in exact_rows if row[index].strip().lower() not in nulls]
        if values and all(numeric(value) for value in values):
            numeric_columns.add(index)

    def partitions(parts: list[str]) -> list[list[str]]:
        if len(parts) < width:
            return []
        if len(parts) == width:
            return [parts]
        if math.comb(len(parts) - 1, width - 1) > 10000:
            return []
        valid = []
        for boundaries in itertools.combinations(range(1, len(parts)), width - 1):
            starts = (0, *boundaries)
            ends = (*boundaries, len(parts))
            groups = [parts[start:end] for start, end in zip(starts, ends)]
            if any(
                len(group) > 1 and any(value.strip().lower() in nulls for value in group)
                for group in groups
            ):
                continue
            candidate = [" | ".join(group) for group in groups]
            if all(numeric(candidate[index]) for index in numeric_columns):
                valid.append(candidate)
                if len(valid) > 1:
                    break
        return valid

    parsed_rows: list[list[str] | None] = []
    unresolved = False
    for raw_row in raw_rows:
        candidates = partitions(raw_row)
        if len(candidates) == 1:
            parsed_rows.append(candidates[0])
        else:
            unresolved = True
            parsed_rows.append(None)

    if not unresolved:
        return pd.DataFrame(parsed_rows, columns=columns)

    marker = "__ambiguous_markdown_row__"
    output_rows = []
    for parsed, source_line in zip(parsed_rows, lines[separator_index + 1 :]):
        output_rows.append(
            [*parsed, ""] if parsed is not None else [*["" for _ in columns], source_line.strip()]
        )
    return pd.DataFrame(output_rows, columns=[*columns, marker])


def _parse_sql_serialization(text: str) -> pd.DataFrame:
    """Parse the exact SQL emitted by LongTableBench's ``df2sql``.

    The official reverse converter splits rows on every comma, including
    commas inside quoted cells. The forward format is unambiguous for values:
    every cell is single-quoted and embedded quotes are doubled. This scanner
    consumes that visible syntax directly and never consults the source CSV.
    """

    create_match = re.search(r"\bCREATE\s+TABLE\b", text, flags=re.I)
    insert_match = re.search(r"\bINSERT\s+INTO\b", text, flags=re.I)
    values_match = re.search(r"\bVALUES\b", text, flags=re.I)
    if create_match is None or insert_match is None or values_match is None:
        raise ValueError("SQL serialization must contain CREATE TABLE and VALUES")
    header_open = text.find("(", create_match.end())
    # Column names are emitted as bare identifiers and may themselves contain
    # parentheses. The final ')' before INSERT INTO is the visible outer CREATE
    # boundary; choosing the first ')' truncates schemas such as "Count (K-12)".
    header_close = text.rfind(")", header_open + 1, insert_match.start())
    if header_open < 0 or header_close < 0 or header_close > insert_match.start():
        raise ValueError("Could not locate SQL CREATE TABLE column list")
    columns = [value.strip() for value in text[header_open + 1 : header_close].split(",")]
    if not columns or any(not value for value in columns):
        raise ValueError(f"Invalid SQL column list: {columns!r}")

    rows: list[list[str]] = []
    position = values_match.end()
    limit = len(text)

    def skip_space_and_separators(index: int) -> int:
        while index < limit and (text[index].isspace() or text[index] in ",;"):
            index += 1
        return index

    position = skip_space_and_separators(position)
    while position < limit:
        if text[position] != "(":
            raise ValueError(f"Expected SQL row at character {position}")
        position += 1
        row: list[str] = []
        while True:
            while position < limit and text[position].isspace():
                position += 1
            if position >= limit or text[position] != "'":
                raise ValueError(f"Expected quoted SQL cell at character {position}")
            position += 1
            cell: list[str] = []
            while position < limit:
                char = text[position]
                if char != "'":
                    cell.append(char)
                    position += 1
                    continue
                if position + 1 < limit and text[position + 1] == "'":
                    cell.append("'")
                    position += 2
                    continue
                position += 1
                break
            else:
                raise ValueError("Unterminated quoted SQL cell")
            row.append("".join(cell))
            while position < limit and text[position].isspace():
                position += 1
            if position < limit and text[position] == ",":
                position += 1
                continue
            if position < limit and text[position] == ")":
                position += 1
                break
            raise ValueError(f"Expected comma or row terminator at character {position}")
        if len(row) != len(columns):
            raise ValueError(
                f"SQL row has {len(row)} visible values but CREATE TABLE declares {len(columns)} columns"
            )
        rows.append(row)
        position = skip_space_and_separators(position)

    return pd.DataFrame(rows, columns=columns)


def _parse_xml_serialization(text: str) -> pd.DataFrame:
    """Parse visible XML while preserving empty elements as empty cells."""

    root = ET.fromstring(text)
    rows: list[dict[str, str]] = []
    columns: list[str] = []
    for row_element in root.findall("row"):
        row: dict[str, str] = {}
        for cell in row_element:
            # Match the official visible-tag decoding without consulting the
            # source schema. Empty tags such as <Description/> carry "".
            column = str(cell.tag).replace("pppersent", "%")
            if column not in columns:
                columns.append(column)
            row[column] = cell.text or ""
        rows.append(row)
    if not rows:
        # <root/> contains no recoverable header names.
        return pd.DataFrame()
    return pd.DataFrame(
        [[row.get(column, "") for column in columns] for row in rows],
        columns=columns,
    )


def _parse_latex_serialization(text: str) -> pd.DataFrame:
    """Parse visible LongTableBench LaTeX and expose unresolved cell boundaries."""

    row_texts: list[str] = []
    pending: list[str] = []
    structural = {r"\centering", r"\hline", r"\end{table}"}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line in structural or line.startswith(r"\begin{") or line.startswith(r"\end{tabular}"):
            continue
        pending.append(line)
        if line.endswith(r"\\"):
            pending[-1] = pending[-1][:-2].rstrip()
            row_texts.append("\n".join(pending).strip())
            pending = []
    if pending:
        raise ValueError("LaTeX serialization ended inside an unterminated row")
    if not row_texts:
        raise ValueError("LaTeX serialization contains no visible header row")

    def split_cells(row: str) -> list[str]:
        return [value.strip() for value in re.split(r"\s+&\s+", row)]

    columns = split_cells(row_texts[0])
    width = len(columns)
    raw_rows = [split_cells(row) for row in row_texts[1:]]
    exact_rows = [row for row in raw_rows if len(row) == width]

    nulls = {"", "nan", "null", "none"}

    def numeric(value: str) -> bool:
        if value.strip().lower() in nulls:
            return True
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    numeric_columns = set()
    for index in range(width):
        values = [row[index] for row in exact_rows if row[index].strip().lower() not in nulls]
        if values and all(numeric(value) for value in values):
            numeric_columns.add(index)

    def partitions(parts: list[str]) -> list[list[str]]:
        if len(parts) < width:
            return []
        if len(parts) == width:
            return [parts]
        # Avoid an exponential guess when the serialized row is highly
        # ambiguous. Such rows are retained verbatim below.
        if math.comb(len(parts) - 1, width - 1) > 10000:
            return []
        valid = []
        for boundaries in itertools.combinations(range(1, len(parts)), width - 1):
            starts = (0, *boundaries)
            ends = (*boundaries, len(parts))
            groups = [parts[start:end] for start, end in zip(starts, ends)]
            # Missing-value sentinels are complete serialized cells. Merging
            # ``nan`` with adjacent text cannot be a valid recovery.
            if any(
                len(group) > 1 and any(value.strip().lower() in nulls for value in group)
                for group in groups
            ):
                continue
            candidate = [" & ".join(group) for group in groups]
            if all(numeric(candidate[index]) for index in numeric_columns):
                valid.append(candidate)
                if len(valid) > 1:
                    # A unique visible alignment is required; no ranking
                    # heuristic is allowed to choose among tied parses.
                    break
        return valid

    parsed_rows: list[list[str] | None] = []
    unresolved = False
    for raw_row, source_text in zip(raw_rows, row_texts[1:]):
        candidates = partitions(raw_row)
        if len(candidates) == 1:
            parsed_rows.append(candidates[0])
        else:
            unresolved = True
            parsed_rows.append(None)

    if not unresolved:
        return pd.DataFrame(parsed_rows, columns=columns)

    marker = "__ambiguous_latex_row__"
    output_rows = []
    for parsed, source_text in zip(parsed_rows, row_texts[1:]):
        output_rows.append(
            [*parsed, ""] if parsed is not None else [*["" for _ in columns], source_text]
        )
    return pd.DataFrame(output_rows, columns=[*columns, marker])


def _cell_text(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def dataframe_payload(table_name: str, frame: pd.DataFrame) -> dict[str, Any]:
    headers = [_cell_text(value) for value in frame.columns.tolist()]
    rows = [[_cell_text(value) for value in row] for row in frame.values.tolist()]
    return {
        "table_id": table_name,
        "region_id": f"region:{table_name}",
        "data": [headers, *rows],
        "metadata": {"source_table": table_name},
    }


def canonical_tables(
    official_root: str | Path,
    record: dict[str, Any],
    table_format: str,
) -> list[dict[str, Any]]:
    """Parse the exact official serialization into CompressTab artifacts.

    A parsing error is a method failure.  The function deliberately has no
    source-DataFrame fallback because that would grant the method hidden cells
    that were not recoverable from the selected official serialization.
    """

    fmt = table_format.strip().lower()
    module = _official_reformat_module(str(Path(official_root).resolve()))
    payloads = []
    for table_name, frame in source_tables(official_root, record):
        serialized = _serialized_table(module, table_name, frame, fmt)
        parsed = _parse_serialized_table(module, serialized, fmt)
        if not isinstance(parsed, pd.DataFrame):
            raise ValueError(f"Official {fmt} parser did not produce a DataFrame for {table_name}")
        # XML represents a source table with zero rows as <root/> and therefore
        # carries no recoverable headers. Preserve that genuinely empty serialized
        # view as a zero-column artifact instead of consulting hidden CSV schema.
        payloads.append(dataframe_payload(table_name, parsed))
    return payloads


def canonical_hash(tables: list[dict[str, Any]]) -> str:
    payload = json.dumps(tables, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def conversation_id(conversation: OfficialConversation, table_format: str) -> str:
    return f"{conversation.seed_id}:{table_format}"


def round_id(conversation: OfficialConversation, table_format: str, round_index: int) -> str:
    return f"{conversation_id(conversation, table_format)}:r{round_index}"


def question_with_history(
    conversation: OfficialConversation,
    round_index: int,
    prior_predictions: list[Any],
) -> str:
    """Build a non-oracle current question from model-generated history."""

    rounds = conversation.rounds
    if not 1 <= round_index <= len(rounds):
        raise IndexError(round_index)
    if len(prior_predictions) != round_index - 1:
        raise ValueError(
            f"Round {round_index} requires {round_index - 1} prior predictions; "
            f"received {len(prior_predictions)}"
        )
    lines = []
    evidence = str(conversation.record.get("evidence", "") or "").strip()
    if evidence:
        lines.extend(["External context supplied by the benchmark:", evidence, ""])
    if prior_predictions:
        lines.append("Conversation so far (answers are this model's own previous outputs):")
        for index, prediction in enumerate(prior_predictions, start=1):
            lines.append(f"Question {index}: {rounds[index - 1].get('question', '')}")
            lines.append(
                "Answer " + str(index) + ": "
                + json.dumps(prediction, ensure_ascii=False, default=str)
            )
        lines.append("")
    lines.append("Current question:")
    lines.append(str(rounds[round_index - 1].get("question", "")))
    return "\n".join(lines).strip()


def prepared_conversation_record(
    conversation: OfficialConversation,
    *,
    table_format: str,
    raw_context_path: str,
    raw_context: str,
    tables: list[dict[str, Any]],
    canonical_path: str,
) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id(conversation, table_format),
        "seed_id": conversation.seed_id,
        "release_position": conversation.release_position,
        "tier_position": conversation.tier_position,
        "tier": conversation.tier,
        "turn": conversation.turn,
        "table_format": table_format,
        "task": conversation.task,
        "question_id": conversation.record.get("question_id"),
        "db_path": conversation.record.get("db_path"),
        "db_id": conversation.record.get("db_id"),
        "is_multi_table": bool(conversation.record.get("is_multi_table")),
        "highlighted_table": conversation.record.get("highlighted_table", []),
        "round_count": len(conversation.rounds),
        "source_question_path": conversation.source_path,
        "raw_context_path": raw_context_path,
        "raw_context_sha256": text_sha256(raw_context),
        "raw_context_tokens_est": approx_token_count(raw_context),
        "canonical_path": canonical_path,
        "canonical_sha256": canonical_hash(tables),
    }


def release_statistics(conversations: list[OfficialConversation]) -> dict[str, Any]:
    task_counts: dict[str, int] = {}
    tier_counts: dict[str, int] = {}
    turn_counts: dict[str, int] = {}
    round_count = 0
    for conversation in conversations:
        task_counts[conversation.task] = task_counts.get(conversation.task, 0) + 1
        tier_counts[conversation.tier] = tier_counts.get(conversation.tier, 0) + 1
        turn_counts[conversation.turn] = turn_counts.get(conversation.turn, 0) + 1
        round_count += len(conversation.rounds)
    return {
        "conversation_seeds": len(conversations),
        "answer_rounds": round_count,
        "task_counts": dict(sorted(task_counts.items())),
        "tier_counts": {tier: tier_counts.get(tier, 0) for tier in LENGTH_TIERS},
        "turn_counts": {turn: turn_counts.get(turn, 0) for turn in TURN_TYPES},
    }
