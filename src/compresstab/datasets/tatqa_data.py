import json
from pathlib import Path
import re

from compresstab.datasets.tabular_common import norm_text


class SimpleTable:
    def __init__(self, rows, *, headers=None):
        rows = rows or []
        if headers is None:
            self.headers = [norm_text(x) for x in (rows[0] if rows else [])]
            body = rows[1:] if len(rows) > 1 else []
        else:
            self.headers = [norm_text(x) for x in headers]
            body = rows
        self.rows = [[norm_text(x) for x in row] for row in body]


def _fill_header_level(values, *, right_aligned=False):
    """Expand merged header cells using only their spatial structure."""

    values = [norm_text(value) for value in values]
    anchors = [(index, value) for index, value in enumerate(values) if value]
    if not anchors:
        return values
    if len(anchors) == 1:
        return [anchors[0][1]] * len(values)

    # Flattened financial tables use both left- and right-aligned labels for
    # merged cells. A leading blank before the first of several anchors means
    # that labels sit at the right edge of their groups, e.g. two year labels
    # over two pairs of child columns.
    if right_aligned and anchors[0][0] > 0:
        expanded = [""] * len(values)
        group_start = 0
        for anchor_index, value in anchors:
            for index in range(group_start, anchor_index + 1):
                expanded[index] = value
            group_start = anchor_index + 1
        for index in range(group_start, len(values)):
            expanded[index] = anchors[-1][1]
        return expanded

    expanded = []
    current = ""
    for value in values:
        if value:
            current = value
        expanded.append(current)
    return expanded


def _is_right_aligned_merged_level(values):
    """Detect equal-width groups whose labels occupy their right edge."""

    values = [norm_text(value) for value in values]
    anchors = [index for index, value in enumerate(values) if value]
    if len(anchors) < 2 or anchors[0] == 0 or anchors[-1] != len(values) - 1:
        return False
    group_width = anchors[0] + 1
    return all(
        current - previous == group_width
        for previous, current in zip(anchors, anchors[1:])
    )


def _explicit_scale_context(matrix, paragraphs):
    candidates = []
    for item in paragraphs or []:
        if not isinstance(item, dict):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", str(item.get("text", ""))):
            # Parenthetical scale declarations and "as follows" clauses are
            # document structure, unlike incidental amounts in prose.
            if (
                re.search(r"\([^)]*\bin\s+(?:thousands?|millions?|billions?)\b", sentence, re.I)
                or re.search(r"\b(?:table|figures?|amounts?|as\s+follows|following)\b", sentence, re.I)
                and re.search(r"\bin\s+(?:thousands?|millions?|billions?)\b", sentence, re.I)
            ):
                candidates.append(sentence)
    text = " ".join(candidates)
    matches = {
        match.group(1).casefold().rstrip("s")
        for match in re.finditer(
            r"\b(?:in|amounts?\s+in)\s+(thousands?|millions?|billions?)\b",
            text,
            flags=re.IGNORECASE,
        )
    }
    # Financial tables often dedicate an early unit row to every value column.
    early_rows = list(matrix or [])[:3]
    for row_index, row in enumerate(early_rows):
        # A unit embedded in one ordinary header names that column, not the
        # whole table. Only a dedicated unit row (all populated value cells are
        # unit declarations) is promoted to table scope.
        populated_units = [str(value).strip() for value in row[1:] if str(value).strip()]
        if populated_units and all(re.search(
            r"(?:\bin\s+)?(?:[$€£¥]|rmb|usd|eur|gbp)?\s*[‘’']?"
            r"(?:000s?|thousands?|m|mn|millions?|b|bn|billions?)\b",
            value,
            re.I,
        ) for value in populated_units):
            unit_cells = " ".join(populated_units)
            if re.search(r"(?:000s?|thousands?)\b", unit_cells, re.I):
                matches.add("thousand")
            if re.search(r"(?:\bm\b|\bmn\b|millions?)\b", unit_cells, re.I):
                matches.add("million")
            if re.search(r"(?:\bb\b|\bbn\b|billions?)\b", unit_cells, re.I):
                matches.add("billion")
        # A unit declaration often occupies the row-key header itself, for
        # example "£m", "€ million", or "All figures in USD '000".
        # Restrict this to the first source row and a unit-only phrase so that
        # measure labels such as "Operating profit (£m)" do not become a
        # table-wide scale.
        if row:
            first_cell = str(row[0]).strip()
            currency = r"(?:[$€£¥]|[A-Z]{2,3}\$|rmb|usd|eur|gbp)?"
            if re.fullmatch(
                rf"\(?\s*(?:all\s+figures\s+in\s+)?{currency}\s*[‘’']?000s?\s*\)?",
                first_cell,
                flags=re.I,
            ):
                matches.add("thousand")
            if re.fullmatch(
                rf"\(?\s*(?:all\s+figures\s+in\s+)?{currency}\s*(?:m|mn|millions?)\s*\)?",
                first_cell,
                flags=re.I,
            ):
                matches.add("million")
            if re.fullmatch(
                rf"\(?\s*(?:all\s+figures\s+in\s+)?{currency}\s*(?:b|bn|billions?)\s*\)?",
                first_cell,
                flags=re.I,
            ):
                matches.add("billion")
    return next(iter(matches)) if len(matches) == 1 else ""


def _row_label_footnote_bases(data_rows, paragraphs):
    """Recover display labels when flattened superscripts are source-verifiable.

    TAT-QA's plain-text tables concatenate superscript note markers to row
    labels. The raw cell remains untouched; this map is used only by the
    structural row-axis view when a matching numbered note and repeated source
    pattern make the interpretation explicit.
    """

    note_numbers = {
        match.group(1)
        for item in paragraphs or []
        if isinstance(item, dict)
        for match in [re.match(r"\s*(\d{1,2})[.)]\s+", str(item.get("text", "")))]
        if match
    }
    labels = {
        str(row[0]).strip().casefold()
        for row in data_rows
        if row and str(row[0]).strip()
    }
    candidates = {}
    by_marker = {}
    for row in data_rows:
        label = str(row[0]).strip() if row else ""
        # A whitespace-separated number is usually semantic (for example,
        # "Phase 2"); flattened superscripts are attached to the label.
        match = re.fullmatch(r"(.+?\D)(\d{1,2})", label)
        if not match or match.group(2) not in note_numbers:
            continue
        base = match.group(1).rstrip()
        if not re.search(r"[A-Za-z]", base):
            continue
        by_marker.setdefault(match.group(2), []).append((label, base))

    for _marker, items in by_marker.items():
        repeated_marker = len(items) >= 2
        for label, base in items:
            if repeated_marker or base.casefold() in labels:
                candidates[label] = base
    return candidates


def _table_semantic_contexts(paragraphs):
    """Recover source-grounded headings and sentences that introduce a table."""

    contexts = []
    source_row = 0
    for item in paragraphs or []:
        if not isinstance(item, dict):
            item = {"text": str(item)}
        value = norm_text(item.get("text", ""))
        if not value:
            continue
        words = value.split()
        short_heading = (
            len(words) <= 12
            and len(value) <= 100
            and not re.search(r"[.!?]\s*$", value)
        )
        table_introduction = bool(
            len(value) <= 700
            and (
                re.search(r"\b(?:following|preceding|below|above)\s+table\b", value, re.I)
                or re.search(
                    r"\btable\s+(?:shows?|lists?|presents?|sets\s+forth|summari[sz]es?|reconciles?)\b",
                    value,
                    re.I,
                )
                or re.search(r"\bshown\s+in\s+the\s+(?:following\s+)?table\b", value, re.I)
                or re.search(r"\bas\s+follows\s*:?\s*$", value, re.I)
            )
        )
        if short_heading or table_introduction:
            contexts.append({
                "kind": "note" if table_introduction else "section",
                "value": value,
                "source_cell_ids": [f"t1:source:r{source_row}:c2"],
            })
        source_row += 1
    return contexts[:4]


def _row_semantic_signals(data_rows):
    """Expose explicit source cues without deciding queried membership."""

    signals = []
    for row_index, row in enumerate(data_rows):
        label = str(row[0]).strip() if row else ""
        row_signals = []
        if re.match(r"^(?:thereof|of\s+which|including)\b", label, re.I):
            row_signals.append("explicit_subcomponent_cue")
        if re.search(r"\b(?:grand\s+total|subtotal|total)\b", label, re.I):
            row_signals.append("explicit_aggregate_cue")
        if label and not any(str(value).strip() for value in row[1:]):
            row_signals.append("blank_valued_section_cue")
        if not label and any(str(value).strip() for value in row[1:]):
            row_signals.append("unlabeled_summary_cue")
        if row_signals:
            signals.append({
                "row": f"r{row_index}",
                "label": label,
                "signals": row_signals,
            })
    return signals


def _structured_table_source(matrix, *, paragraphs=None):
    """Recover leading merged header rows as full column paths."""

    matrix = [[norm_text(value) for value in row] for row in (matrix or [])]
    width = max((len(row) for row in matrix), default=0)
    if width == 0:
        return {
            "table_id": "t0",
            "caption": "document table",
            "table": SimpleTable([]),
        }
    matrix = [(row + [""] * width)[:width] for row in matrix]

    # Separate leading table titles, then recover every consecutive header
    # level. A financial unit declaration may occupy the row-key cell while
    # the remaining cells contain the actual period labels.
    title_count = 0
    title_parts = []
    while title_count < len(matrix):
        row = matrix[title_count]
        if not row[0] or any(row[1:]):
            break
        title_parts.append(row[0])
        title_count += 1

    header_start = title_count
    header_end = min(len(matrix), header_start + 1)
    while header_end < len(matrix):
        row = matrix[header_end]
        non_empty_value_cells = [str(value).strip() for value in row[1:] if str(value).strip()]
        previous_value_cells = [
            str(value).strip() for value in matrix[header_end - 1][1:]
            if str(value).strip()
        ]
        unit_stub = bool(
            row[0]
            and any(row[1:])
            and re.search(
                r"\b(?:in|except|amounts?)\b.*\b(?:thousands?|millions?|billions?|per\s+share)\b",
                row[0],
                re.I,
            )
        )
        temporal_stub = bool(
            row[0]
            and any(row[1:])
            and re.search(
                r"\b(?:years?|quarters?|periods?)\s+(?:ended|ending)\b|\bas\s+of\b",
                row[0],
                re.I,
            )
        )
        temporal_value_level = bool(
            non_empty_value_cells
            and all(re.fullmatch(
                r"(?:FY\s*)?(?:19|20)\d{2}|Q[1-4]|(?:19|20)\d{2}\s*Q[1-4]",
                value,
                flags=re.I,
            ) for value in non_empty_value_cells)
        )
        repeated_temporal_parent = bool(
            len(previous_value_cells) >= 2
            and len({value.casefold() for value in previous_value_cells})
            < len(previous_value_cells)
            and any(
                re.search(r"\b(?:year|quarter|period)s?\b|\bended\b", value, re.I)
                for value in previous_value_cells
            )
        )
        temporal_child_level = bool(
            repeated_temporal_parent
            and non_empty_value_cells
            and all(
                re.search(
                    r"\b(?:19|20)\d{2}\b|\bQ[1-4]\b|"
                    r"\b(?:january|february|march|april|may|june|july|august|"
                    r"september|october|november|december)\b",
                    value,
                    re.I,
                )
                for value in non_empty_value_cells
            )
        )
        scale_cells = sum(
            bool(re.fullmatch(
                r"\(?\s*(?:[$€£¥]|[A-Z]{2,3}\$|rmb|usd|eur|gbp)?\s*"
                r"(?:[‘’']?000s?|m|mn|millions?|b|bn|billions?)\s*\)?",
                value,
                flags=re.I,
            ))
            for value in non_empty_value_cells
        )
        unit_level = bool(
            non_empty_value_cells
            and scale_cells >= max(1, (len(non_empty_value_cells) + 1) // 2)
        )
        if (
            (not row[0] and any(row[1:]))
            or unit_stub
            or temporal_stub
            or temporal_value_level
            or temporal_child_level
            or unit_level
        ):
            header_end += 1
            continue
        break

    header_rows = matrix[header_start:header_end]
    data_rows = matrix[header_end:]
    row_label_footnotes = _row_label_footnote_bases(data_rows, paragraphs)

    # A blank-valued row in a financial table is often a section header rather
    # than a missing record.  Preserve its scope as a row-axis path so a query
    # for an aggregate such as "current assets" can retrieve the descendant
    # measures instead of treating the section label as an empty-valued tuple.
    section_rows = [
        index for index, row in enumerate(data_rows)
        if row and row[0] and not any(str(value).strip() for value in row[1:])
    ]
    row_header_paths = []
    if section_rows:
        initial_section = str(header_rows[-1][0]).strip() if header_rows else ""
        current_section = initial_section
        for index, row in enumerate(data_rows):
            label = str(row[0]).strip() if row else ""
            axis_label = row_label_footnotes.get(label, label)
            if index in section_rows:
                current_section = axis_label
                row_header_paths.append([axis_label] if axis_label else [])
            elif current_section and axis_label:
                row_header_paths.append([current_section, axis_label])
            elif current_section:
                # Keep an unlabeled continuation row addressable without
                # inventing a source label.
                row_header_paths.append([current_section])
            elif axis_label:
                row_header_paths.append([axis_label])
            else:
                row_header_paths.append([])

    header_paths = [["row label"]]
    for col_idx in range(1, width):
        path = []
        for row in header_rows:
            values = row[1:]
            expanded = _fill_header_level(
                values,
                right_aligned=_is_right_aligned_merged_level(values),
            )
            value = expanded[col_idx - 1] if col_idx - 1 < len(expanded) else ""
            if value and (not path or value.casefold() != path[-1].casefold()):
                path.append(value)
        header_paths.append(path or [f"value {col_idx}"])

    headers = [" / ".join(path) for path in header_paths]
    metadata = {
        "source_kind": "table",
        "header_rows": len(header_rows),
        "title_rows": title_count,
        "semantic_contexts": _table_semantic_contexts(paragraphs),
        "row_semantic_signals": _row_semantic_signals(data_rows),
    }
    if row_label_footnotes:
        metadata["row_label_footnotes"] = row_label_footnotes
    if scale := _explicit_scale_context(matrix, paragraphs):
        metadata["scale"] = scale
    result = {
        "table_id": "t0",
        "caption": " / ".join(title_parts) or "document table",
        "table": SimpleTable(data_rows, headers=headers),
        "header_paths": header_paths,
        "metadata": metadata,
    }
    if row_header_paths:
        result["row_header_paths"] = row_header_paths
    return result


def _paragraph_source(paragraphs):
    rows = []
    for index, paragraph in enumerate(paragraphs or []):
        if not isinstance(paragraph, dict):
            paragraph = {"text": str(paragraph)}
        text = str(paragraph.get("text", "")).strip()
        if not text:
            continue
        rows.append([
            str(paragraph.get("uid") or f"p{index}"),
            str(paragraph.get("order") or index + 1),
            text,
        ])
    return {
        "table_id": "t1",
        "caption": "document paragraphs",
        "table": SimpleTable(rows, headers=["paragraph id", "source order", "text"]),
        "metadata": {"source_kind": "text"},
    }


def load_tatqa_rows(dataset_path, answer_from="", limit=0):
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    rows = []
    for document_index, item in enumerate(data):
        paragraph_items = item.get("paragraphs", []) or []
        paragraphs = [
            str(x.get("text", "")).strip()
            for x in paragraph_items
            if isinstance(x, dict) and str(x.get("text", "")).strip()
        ]
        table_source = _structured_table_source(
            item.get("table", {}).get("table", []),
            paragraphs=paragraph_items,
        )
        text_source = _paragraph_source(paragraph_items)
        table = table_source["table"]
        for qa in item.get("questions", []):
            af = str(qa.get("answer_from", "") or "")
            if answer_from and af != answer_from:
                continue
            rows.append(
                {
                    "idx": len(rows) + 1,
                    "question_id": str(qa.get("uid") or qa.get("id") or ""),
                    "question": str(qa.get("question") or ""),
                    "answer_type": str(qa.get("answer_type") or ""),
                    "answer": qa.get("answer"),
                    "scale": str(qa.get("scale") or ""),
                    "answer_from": af,
                    "table": table,
                    "tables": [table_source, text_source],
                    "paragraphs": paragraphs,
                    "metadata": {
                        "benchmark": "tatqa",
                        "source_path": str(Path(dataset_path).resolve()),
                        "document_index": document_index,
                        "answer_from": af,
                    },
                }
            )
            if limit > 0 and len(rows) >= limit:
                return rows
    return rows
