"""TableStore and low-cost table access tools."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from nltk.stem import PorterStemmer

from compresstab.table.artifact import CanonicalTable, TableArtifact
from compresstab.table.provenance import ProvenanceIndex, cell_ref, column_ref, row_ref
from compresstab.types import EvidenceRef


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "with", "all", "combined", "table", "data", "number", "value",
}
STEMMER = PorterStemmer()


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"[^a-z0-9%$/().,\-+ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(value: Any) -> list[str]:
    result = []
    for token in re.findall(r"[a-z0-9%$]+", norm_text(value)):
        if not token or token in STOPWORDS:
            continue
        result.append(STEMMER.stem(token) if token.isalpha() and len(token) > 3 else token)
    return result


def approx_token_count(value: Any) -> int:
    if isinstance(value, (dict, list)):
        value = str(value)
    text = str(value if value is not None else "")
    # Word counting severely underestimates long numbers, identifiers, and
    # punctuation-heavy table cells. Byte length is a conservative,
    # tokenizer-independent fallback until exact provider tokenizers are used.
    lexical = len(tokens(text))
    byte_estimate = math.ceil(len(text.encode("utf-8")) / 4)
    return max(lexical, byte_estimate)


def parse_number(value: Any) -> float | None:
    """Parse one scalar numeric cell without extracting numbers from text.

    TableQA inputs often contain ranges, dates, footnote-bearing values, and
    identifiers.  Taking the first numeric substring from those values silently
    changes their meaning, so this parser accepts only a complete, reversible
    scalar representation.
    """

    text = str(value if value is not None else "").strip()
    if not text:
        return None
    neg = text.startswith("(") and text.endswith(")")
    if neg:
        text = text[1:-1].strip()
    # Spaces, non-breaking spaces, and narrow spaces are common thousands
    # separators in semi-structured tables (for example, "9 000").
    match = re.fullmatch(
        r"(?:[$\u00a3\u20ac\u00a5]\s*)?"
        r"([-+]?(?:\d{1,3}(?:[,\s\u00a0\u202f]\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
        r"\s*%?",
        text,
    )
    if not match:
        return None
    try:
        num = float(re.sub(r"[,\s\u00a0\u202f]", "", match.group(1)))
    except Exception:
        return None
    if not math.isfinite(num):
        return None
    return -abs(num) if neg else num


def format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if abs(float(value) - round(float(value))) < 1e-9:
        return str(int(round(float(value))))
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class SearchHit:
    ref: EvidenceRef
    score: float
    text: str


class TableStore:
    """A deterministic data plane for the codec loop.

    It exposes only bounded views and provenance-bearing refs.  LLM agents should
    receive rendered tool outputs, never direct access to raw Python objects.
    """

    def __init__(self, artifact: TableArtifact):
        self.artifact = artifact
        self.provenance = ProvenanceIndex()
        for table in artifact.tables:
            for col_idx in range(len(table.headers)):
                self.provenance.add(column_ref(table, col_idx))
            for row_idx in range(len(table.rows)):
                self.provenance.add(row_ref(table, row_idx))
                for col_idx in range(len(table.headers)):
                    self.provenance.add(cell_ref(table, row_idx, col_idx))

    def table(self, table_id: str | None = None) -> CanonicalTable:
        if table_id is None:
            return self.artifact.tables[0]
        for table in self.artifact.tables:
            if table.table_id == table_id:
                return table
        raise KeyError(table_id)

    def raw_table_tokens(self, table_id: str | None = None) -> int:
        tables = [self.table(table_id)] if table_id is not None else self.artifact.tables
        lines: list[str] = []
        for table in tables:
            if len(tables) > 1:
                lines.append(f"TABLE {table.table_id}")
            lines.append("\t".join(table.headers))
            lines.extend("\t".join(row) for row in table.rows)
        return approx_token_count("\n".join(lines))

    def overview(self, *, max_rows: int = 6) -> dict[str, Any]:
        tables = []
        for table in self.artifact.tables:
            numeric_cols = []
            categorical_cols = []
            for col_idx, header in enumerate(table.headers):
                values = [row[col_idx] for row in table.rows if col_idx < len(row)]
                nums = [parse_number(v) for v in values]
                num_values = [x for x in nums if x is not None]
                distinct = len(set(norm_text(v) for v in values if norm_text(v)))
                if len(num_values) >= max(2, len(values) // 2):
                    numeric_cols.append({
                        "col_id": f"c{col_idx}",
                        "header": header,
                        "count": len(num_values),
                        "min": format_number(min(num_values)),
                        "max": format_number(max(num_values)),
                    })
                else:
                    categorical_cols.append({
                        "col_id": f"c{col_idx}",
                        "header": header,
                        "distinct": distinct,
                        "examples": values[:3],
                    })
            tables.append({
                "table_id": table.table_id,
                "region_id": table.region_id,
                "source_level": table.source_level,
                "caption": table.caption,
                "n_rows": len(table.rows),
                "n_cols": len(table.headers),
                "headers": [
                    {"col_id": f"c{i}", "header_path": table.header_path(i), "display": h}
                    for i, h in enumerate(table.headers)
                ],
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "sample_rows": [dict(row_id=f"r{i}", values=table.rows[i]) for i in range(min(max_rows, len(table.rows)))],
                "contexts": [item.model_dump() for item in table.contexts],
                "structure_candidates": [item.model_dump() for item in table.structure_candidates],
            })
        return {
            "artifact_id": self.artifact.artifact_id,
            "table_count": len(self.artifact.tables),
            "tables": tables,
            "paragraph_count": len(self.artifact.paragraphs),
            "metadata": self.artifact.metadata,
        }

    def search(
        self,
        query: str,
        *,
        novelty_against: set[str] | None = None,
        top_k: int = 20,
        table_id: str | None = None,
    ) -> list[SearchHit]:
        q_tokens = set(tokens(query))
        if not q_tokens:
            q_tokens = set(norm_text(query).split())
        novelty_against = novelty_against or set()
        hits: list[SearchHit] = []
        tables = [self.table(table_id)] if table_id is not None else self.artifact.tables
        for table in tables:
            for col_idx, header in enumerate(table.headers):
                ref = self.provenance.get(f"{table.table_id}:c{col_idx}")
                if ref is None:
                    continue
                score = self._text_score(q_tokens, header)
                if ref.ref_id not in novelty_against and score > 0:
                    hits.append(SearchHit(ref=ref, score=score + 0.2, text=f"column {header}"))
            for row_idx, row in enumerate(table.rows):
                row_text = " | ".join(row)
                score = self._text_score(q_tokens, row_text)
                if score <= 0:
                    continue
                ref = self.provenance.get(f"{table.table_id}:r{row_idx}")
                if ref is None:
                    continue
                novelty_bonus = 0.4 if ref.ref_id not in novelty_against else -0.2
                hits.append(SearchHit(ref=ref, score=score + novelty_bonus, text=row_text))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def _text_score(self, q_tokens: set[str], text: str) -> float:
        hay = set(tokens(text))
        if not hay:
            return 0.0
        overlap = q_tokens & hay
        score = float(len(overlap))
        low_text = norm_text(text)
        for tok in q_tokens:
            if len(tok) > 3 and tok in low_text:
                score += 0.5
        return score

    def slice_table(
        self,
        *,
        table_id: str | None = None,
        row_ids: list[str] | None = None,
        col_ids: list[str] | None = None,
        max_rows: int | None = None,
    ) -> dict[str, Any]:
        table = self.table(table_id)
        row_indices = self._parse_indices(row_ids, prefix="r", max_len=len(table.rows)) if row_ids is not None else list(range(len(table.rows)))
        col_indices = self._parse_indices(col_ids, prefix="c", max_len=len(table.headers)) if col_ids is not None else list(range(len(table.headers)))
        if max_rows is not None:
            row_indices = row_indices[:max_rows]
        refs = []
        rows = []
        for row_idx in row_indices:
            values = []
            for col_idx in col_indices:
                values.append(table.rows[row_idx][col_idx])
                ref = self.provenance.get(f"{table.table_id}:r{row_idx}:c{col_idx}")
                if ref:
                    refs.append(ref)
            rows.append({"row_id": f"r{row_idx}", "values": values})
        return {
            "table_id": table.table_id,
            "headers": [
                {"col_id": f"c{i}", "header_path": table.header_path(i), "display": table.headers[i]}
                for i in col_indices
            ],
            "rows": rows,
            "evidence_refs": refs,
        }

    def candidate_rows_and_cols(
        self,
        question: str,
        *,
        max_rows: int = 16,
        max_cols: int = 8,
        table_id: str | None = None,
        hits: list[SearchHit] | None = None,
    ) -> tuple[list[str], list[str]]:
        table = self.table(table_id)
        hits = hits or self.search(
            question,
            top_k=max_rows + max_cols,
            table_id=table.table_id,
        )
        rows: list[str] = []
        cols: list[str] = []
        q_tokens = set(tokens(question))
        for hit in hits:
            if hit.ref.source_type == "row":
                rows.extend(hit.ref.row_ids)
                # A row-level lexical hit must expose the cells that caused it;
                # otherwise a wide-table sketch shows the right tuple with only
                # unrelated leading columns.
                for row_id in hit.ref.row_ids:
                    if not row_id.startswith("r") or not row_id[1:].isdigit():
                        continue
                    row_idx = int(row_id[1:])
                    if not 0 <= row_idx < len(table.rows):
                        continue
                    for col_idx, value in enumerate(table.rows[row_idx]):
                        if self._text_score(q_tokens, str(value)) > 0:
                            col_id = f"c{col_idx}"
                            if col_id not in cols:
                                cols.append(col_id)
            elif hit.ref.source_type == "column":
                cols.extend(hit.ref.col_ids)
        scored_cols = []
        for idx, header in enumerate(table.headers):
            scored_cols.append((self._text_score(q_tokens, header), f"c{idx}"))
        scored_cols.sort(reverse=True)
        for score, col_id in scored_cols:
            if score > 0 and col_id not in cols:
                cols.append(col_id)
        if not rows:
            rows = [f"r{i}" for i in range(min(max_rows, len(table.rows)))]
        if not cols:
            cols = [f"c{i}" for i in range(min(max_cols, len(table.headers)))]
        return rows[:max_rows], cols[:max_cols]

    def _parse_indices(self, ids: list[str], *, prefix: str, max_len: int) -> list[int]:
        out = []
        for raw in ids:
            text = str(raw)
            if text.startswith(prefix):
                text = text[len(prefix):]
            try:
                idx = int(text)
            except Exception:
                continue
            if 0 <= idx < max_len and idx not in out:
                out.append(idx)
        return out

    def complement_row_ids(self, included_ref_ids: set[str], table_id: str | None = None) -> list[str]:
        table = self.table(table_id)
        retained = set()
        for ref_id in included_ref_ids:
            ref = self.provenance.get(ref_id)
            if ref:
                retained.update(ref.row_ids)
        return [f"r{i}" for i in range(len(table.rows)) if f"r{i}" not in retained]
