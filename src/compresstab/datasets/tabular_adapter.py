import csv
import ast
import json
import re
import string
import unicodedata
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from compresstab.datasets.tabular_common import norm_key, norm_text, tokenize


class SimpleTable:
    def __init__(self, rows):
        rows = rows or []
        if isinstance(rows, pd.DataFrame):
            self.headers = [norm_text(x) for x in rows.columns.tolist()]
            self.rows = [[norm_text(x) for x in row] for row in rows.values.tolist()]
        else:
            self.headers = [norm_text(x) for x in (rows[0] if rows else [])]
            self.rows = [[norm_text(x) for x in row] for row in (rows[1:] if len(rows) > 1 else [])]


def _dataframe_to_simple_table(df: pd.DataFrame) -> SimpleTable:
    if df is None:
        return SimpleTable([])
    return SimpleTable(df)


def _normalize_counter_values(values):
    out = set()
    for value in values or []:
        text = norm_key(value)
        if text in {"1", "true"}:
            text = "yes"
        elif text in {"0", "false"}:
            text = "no"
        elif text in {"united states", "u s", "u s a", "us"}:
            text = "usa"
        if text:
            out.add(text)
    return Counter(out)


def _normalize_wtq_denotation_text(value: Any) -> str:
    """Mirror the official WTQ string normalization used for denotation match."""
    if value is None:
        return ""
    text = str(value)
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn")
    text = re.sub(r"[‘’´`]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    while True:
        old = text
        text = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", text.strip())
        text = re.sub(r"(?<!^)( \([^)]*\))*$", "", text.strip())
        text = re.sub(r'^"([^"]*)"$', r"\1", text.strip())
        if text == old:
            break
    if text.endswith("."):
        text = text[:-1]
    return re.sub(r"\s+", " ", text).lower().strip()


def _wtq_number_value(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value) if float(value).is_integer() else float(value)
        except Exception:
            return None
    text = str(value).strip().replace(",", "")
    unit_match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s+([A-Za-z][A-Za-z\s-]{0,40})", text)
    if unit_match:
        tail_tokens = [tok for tok in re.split(r"[\s-]+", unit_match.group(2).lower()) if tok]
        if 1 <= len(tail_tokens) <= 3:
            text = unit_match.group(1)
    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        num = float(text)
        return int(num) if num.is_integer() else num
    except Exception:
        return None


def _wtq_date_value(value: Any) -> tuple[int, int, int] | None:
    text = str(value).strip().lower()
    match = re.fullmatch(r"(\d{4}|xx|xxxx)-(\d{1,2}|xx)-(\d{1,2}|xx)", text)
    if not match:
        return None
    year = -1 if match.group(1) in {"xx", "xxxx"} else int(match.group(1))
    month = -1 if match.group(2) == "xx" else int(match.group(2))
    day = -1 if match.group(3) == "xx" else int(match.group(3))
    if year == month == day == -1:
        return None
    if month != -1 and not (1 <= month <= 12):
        return None
    if day != -1 and not (1 <= day <= 31):
        return None
    return year, month, day


def _wtq_denotation_value(value: Any) -> tuple:
    num = _wtq_number_value(value)
    if num is not None:
        return ("number", float(num), _normalize_wtq_denotation_text(value))
    date = _wtq_date_value(value)
    if date is not None:
        if date[1] == date[2] == -1:
            return ("number", float(date[0]), _normalize_wtq_denotation_text(value))
        return ("date", *date, _normalize_wtq_denotation_text(value))
    return ("string", _normalize_wtq_denotation_text(value))


def _wtq_values_match(target: tuple, pred: tuple) -> bool:
    if target[-1] and target[-1] == pred[-1]:
        return True
    if target[0] == "number" and pred[0] == "number":
        return abs(float(target[1]) - float(pred[1])) <= 1e-6
    if target[0] == "date" and pred[0] == "date":
        return target[1:4] == pred[1:4]
    return False


def _wtq_split_prediction(pred: Any) -> list[Any]:
    if isinstance(pred, list):
        return pred
    if isinstance(pred, str):
        parts = [p.strip() for p in re.split(r"\s*\|\s*|\n+|\s*;\s*", pred) if p.strip()]
        if len(parts) > 1:
            return parts
    return [pred]


def _wtq_denotation_equal(gold: Any, pred: Any) -> bool:
    gold_items = gold if isinstance(gold, list) else [gold]
    pred_items = _wtq_split_prediction(pred)
    gold_values = list({_wtq_denotation_value(x) for x in gold_items if str(x).strip()})
    pred_values = list({_wtq_denotation_value(x) for x in pred_items if str(x).strip()})
    if not gold_values or len(gold_values) != len(pred_values):
        return False
    return all(any(_wtq_values_match(g, p) for p in pred_values) for g in gold_values)


def _normalize_wtq_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        try:
            if isinstance(v, float) and v.is_integer():
                return int(v)
        except Exception:
            pass
        return v
    s = str(v).strip().strip("\"'`")
    if not s:
        return None
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    s = s.replace("\\p", "|").replace("\\n", " ")
    s = " ".join(s.split())
    try:
        num = float(s.replace(",", ""))
        if num.is_integer():
            return int(num)
        return num
    except Exception:
        return s.lower()


def _normalize_tabfact_label(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        try:
            return 1 if int(v) != 0 else 0
        except Exception:
            return None
    s = norm_key(v)
    if s in {"1", "true", "yes", "entailed", "support", "supported"}:
        return 1
    if s in {"0", "false", "no", "refuted", "refute", "contradiction"}:
        return 0
    try:
        num = float(s)
        return 1 if int(num) != 0 else 0
    except Exception:
        return None


def _extract_final_answer_text(value: Any) -> str:
    text = norm_text(value)
    if not text:
        return ""
    low = text.lower()
    for marker in ["[final answer]:", "final answer:"]:
        pos = low.rfind(marker)
        if pos >= 0:
            text = text[pos + len(marker):].strip()
            break
    if "\n" in text:
        text = text.splitlines()[0].strip()
    return text.strip().strip("\"'`").strip()


def _normalize_tablebench_answer(value: Any) -> str:
    text = _extract_final_answer_text(value)
    text = text.replace("[", "").replace("]", "")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = text.strip(" .。:;!?\t\r\n")
    if text in {"no answer", "cannot answer", "unable to answer", "refuse", "refused"}:
        return "refuse to answer"
    if text in {"wrong", "incorrect", "false"}:
        return "wrong"
    if text in {"correct", "right", "true"}:
        return "correct"
    return text


def _tablebench_official_normalize_answer(value: Any) -> str:
    """Mirror TableBench's official QAMetric.normalize_answer."""

    text = _normalize_tablebench_answer(value).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _tablebench_split_official_answers(value: Any) -> list[str]:
    # Official TableBench QAMetric normalizes text before calling custom_em_metric.
    # This removes punctuation, including commas, decimal points, and percent signs.
    text = _tablebench_official_normalize_answer(value)
    if not text:
        return []
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return parts or [text]


def _tablebench_text_units(value: Any) -> list[str]:
    text = _normalize_tablebench_answer(value)
    text = re.sub(r"\band\b", ",", text)
    parts = [p.strip(" .;:") for p in re.split(r"\s*,\s*|\s*;\s*|\s*\|\s*|\s*/\s*", text) if p.strip(" .;:")]
    return parts if len(parts) > 1 else ([text] if text else [])


def _tablebench_number_close(gold: float, pred: float) -> bool:
    # Keep the local runner scorer conservative: enough for formatting/rounding,
    # not enough to hide genuinely wrong numerical reasoning.
    return abs(gold - pred) <= max(1e-3, abs(gold) * 0.005)


def _tablebench_custom_normalize_number(value: str) -> Decimal:
    if value.endswith("%"):
        value = value.strip("%")
        decimal_value = Decimal(value) / Decimal("100")
        return decimal_value.quantize(Decimal("1.0000"), rounding=ROUND_HALF_UP)
    return Decimal(value)


def _tablebench_custom_is_number(value: str) -> bool:
    return bool(re.match(r"^-?\d+(\.\d+)?%?$", str(value).strip()))


def _tablebench_custom_decimal_precision(values: list[str]) -> int:
    precisions = []
    for value in values:
        if value.endswith("%"):
            continue
        if "." in value:
            precisions.append(len(value.split(".")[-1]))
        else:
            precisions.append(0)
    return min(precisions) if precisions else 0


def _tablebench_custom_round_decimal(value: Decimal, precision: int) -> str:
    rounding_format = f'1.{"0" * precision}'
    return str(value.quantize(Decimal(rounding_format), rounding=ROUND_HALF_UP))


def _tablebench_compute_em(gold: Any, pred: Any) -> float:
    """Mirror official metrics/custom_em_metric.compute_em for one example."""

    gold_parts = _tablebench_split_official_answers(gold)
    pred_parts = _tablebench_split_official_answers(pred)
    if not gold_parts or not pred_parts:
        return 0.0
    match_score = 0.0
    weight = 1.0 / len(gold_parts)
    for idx, ref_value in enumerate(gold_parts):
        if idx >= len(pred_parts):
            continue
        pred_value = pred_parts[idx]
        if _tablebench_custom_is_number(ref_value):
            try:
                if ref_value.endswith("%"):
                    norm_ref = _tablebench_custom_normalize_number(ref_value)
                    norm_pred = _tablebench_custom_normalize_number(pred_value)
                    if norm_ref == norm_pred:
                        match_score += weight
                else:
                    ref_vals = [x for x in gold_parts if _tablebench_custom_is_number(x) and not x.endswith("%")]
                    precision = _tablebench_custom_decimal_precision(ref_vals)
                    norm_ref = _tablebench_custom_round_decimal(_tablebench_custom_normalize_number(ref_value), precision)
                    norm_pred = _tablebench_custom_round_decimal(_tablebench_custom_normalize_number(pred_value), precision)
                    if norm_ref == norm_pred:
                        match_score += weight
            except Exception:
                continue
        elif ref_value == pred_value:
            match_score += weight
        elif _tablebench_official_normalize_answer(ref_value) == _tablebench_official_normalize_answer(pred_value):
            match_score += weight
    return match_score


def _tablebench_compute_em_with_tolerance(gold: Any, pred: Any, error_range: float = 10.0) -> float:
    """Mirror official compute_em_with_tolerance for one example."""

    gold_parts = _tablebench_split_official_answers(gold)
    pred_parts = _tablebench_split_official_answers(pred)
    if not gold_parts or not pred_parts:
        return 0.0
    match_score = 0.0
    weight = 1.0 / len(gold_parts)
    for idx, ref_value in enumerate(gold_parts):
        if idx >= len(pred_parts):
            continue
        pred_value = pred_parts[idx]
        if _tablebench_custom_is_number(ref_value):
            try:
                ref_num = _tablebench_custom_normalize_number(ref_value)
                pred_num = _tablebench_custom_normalize_number(pred_value)
                if ref_num == Decimal("0"):
                    if pred_num == ref_num:
                        match_score += weight
                else:
                    error = abs(ref_num - pred_num) / abs(ref_num)
                    if error <= Decimal(str(error_range)) / Decimal("100"):
                        match_score += weight
            except Exception:
                continue
        elif ref_value == pred_value:
            match_score += weight
        elif _tablebench_official_normalize_answer(ref_value) == _tablebench_official_normalize_answer(pred_value):
            match_score += weight
    return match_score


def _tablebench_rouge_tokens(value: Any) -> list[str]:
    text = _tablebench_official_normalize_answer(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [tok for tok in text.split() if tok]


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for token in left:
        cur = [0]
        for j, other in enumerate(right, start=1):
            if token == other:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _tablebench_rouge_l(gold: Any, pred: Any) -> float:
    gold_tokens = _tablebench_rouge_tokens(gold)
    pred_tokens = _tablebench_rouge_tokens(pred)
    if not gold_tokens or not pred_tokens:
        return 0.0
    lcs = _lcs_length(gold_tokens, pred_tokens)
    if lcs <= 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _tablebench_row_type(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
    qtype = str(metadata.get("qtype", row.get("qtype", "")) or "")
    qsubtype = str(metadata.get("qsubtype", row.get("qsubtype", "")) or "")
    return qtype.strip(), qsubtype.strip()


def _tablebench_y_reference_match(gold: Any, pred: Any) -> float:
    gold_matrix = _parse_tablebench_y_references(gold)
    pred_matrix = _parse_tablebench_y_references(pred)
    if gold_matrix is not None and pred_matrix is not None:
        return 1.0 if _tablebench_y_reference_matrices_equal(gold_matrix, pred_matrix) else 0.0
    return _tablebench_compute_em(gold, pred)


def _parse_tablebench_y_references(value: Any) -> list[list[Any]] | None:
    text = _extract_final_answer_text(value)
    if not re.search(r"\by_?references\b", text, flags=re.I):
        return None
    rhs = re.sub(r"^.*?\by_?references\b\s*=\s*", "", text, flags=re.I).strip()
    if not rhs:
        return None
    try:
        parsed = ast.literal_eval(re.sub(r"\bnp\.nan\b|\bnan\b", "None", rhs, flags=re.I))
    except Exception:
        nums = re.findall(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+)|(?:\.\d+))(?:\.\d+)?", rhs)
        if not nums:
            return None
        return [[_normalize_tablebench_yref_value(x) for x in nums]]
    if isinstance(parsed, tuple):
        parsed = list(parsed)
    if not isinstance(parsed, list):
        return None
    if parsed and all(isinstance(item, (list, tuple)) for item in parsed):
        return [[_normalize_tablebench_yref_value(cell) for cell in series] for series in parsed]
    return [[_normalize_tablebench_yref_value(cell) for cell in parsed]]


def _normalize_tablebench_yref_value(value: Any) -> Any:
    if value is None:
        return "nan"
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    text = norm_text(value).strip().strip("\"'")
    cleaned = re.sub(r"[$£€¥,]", "", text)
    cleaned = cleaned.replace("%", "")
    try:
        return float(cleaned)
    except Exception:
        return _tablebench_official_normalize_answer(text)


def _tablebench_yref_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= max(1e-2, abs(float(left)) * 1e-6)
    return str(left) == str(right)


def _tablebench_y_reference_matrices_equal(left: list[list[Any]], right: list[list[Any]]) -> bool:
    if len(left) != len(right):
        left_flat = [value for series in left for value in series]
        right_flat = [value for series in right for value in series]
        if len(left_flat) != len(right_flat):
            return False
        return all(_tablebench_yref_values_equal(a, b) for a, b in zip(left_flat, right_flat))
    for left_series, right_series in zip(left, right):
        if len(left_series) != len(right_series):
            return False
        if not all(_tablebench_yref_values_equal(a, b) for a, b in zip(left_series, right_series)):
            return False
    return True


def _tablebench_official_local_score(row: dict[str, Any]) -> float:
    """Dependency-free local vendor of the official TableBench metric choices.

    This follows the official TableBench metric choices where possible: EM, EM
    with numeric tolerance, ROUGE-L for open DataAnalysis subtypes, and Pass@1
    for Visualization. Visualization remains a local y-reference proxy; the
    active runner separately invokes the official chart-code parser.
    """

    qtype, qsubtype = _tablebench_row_type(row)
    gold = row.get("answer", "")
    pred = row.get("pred_answer", "")
    if qtype == "DataAnalysis":
        if qsubtype in {"CorrelationAnalysis", "TrendForecasting", "StatisticalAnalysis"}:
            return _tablebench_compute_em_with_tolerance(gold, pred, error_range=10.0)
        if qsubtype == "ImpactAnalysis":
            return _tablebench_compute_em(gold, pred)
        return _tablebench_rouge_l(gold, pred)
    if qtype == "Visualization":
        return _tablebench_y_reference_match(gold, pred)
    return _tablebench_compute_em(gold, pred)


def _read_table_dataframe(path: Path) -> pd.DataFrame:
    def _detect_sep(p: Path) -> str:
        if p.suffix.lower() == ".tsv":
            return "\t"
        try:
            sample = p.read_text(encoding="utf-8", errors="replace")[:4096]
        except Exception:
            return ","
        first = next((line for line in sample.splitlines() if line.strip()), "")
        if not first:
            return ","
        candidates = [",", "\t", "#", ";", "|"]
        return max(candidates, key=lambda sep: first.count(sep))

    def _csv_module_read(p: Path, sep: str, *, escapechar: str | None) -> pd.DataFrame:
        try:
            with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
                rows = list(csv.reader(f, delimiter=sep, escapechar=escapechar))
        except Exception:
            return pd.DataFrame()
        if not rows:
            return pd.DataFrame()
        header = [str(x) for x in rows[0]]
        width = len(header)
        data = []
        for row in rows[1:]:
            cells = [str(x) for x in row[:width]]
            if len(cells) < width:
                cells.extend([""] * (width - len(cells)))
            data.append(cells)
        return pd.DataFrame(data, columns=header)

    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = _detect_sep(path)
        sample = path.read_text(encoding="utf-8", errors="replace")[:16384]
        # Many web tables use backslash-escaped quotes inside quoted, multiline
        # cells. Recognizing the source dialect prevents commas and newlines in
        # one entity from becoming synthetic rows or columns.
        escapechar = "\\" if '\\"' in sample else None
        try:
            return pd.read_csv(path, sep=sep, escapechar=escapechar)
        except Exception:
            csv_df = _csv_module_read(path, sep, escapechar=escapechar)
            if not csv_df.empty:
                return csv_df
            return pd.read_csv(
                path,
                sep=sep,
                escapechar=escapechar,
                engine="python",
                on_bad_lines="skip",
            )
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_raw_matrix_dataframe(path: Path) -> pd.DataFrame:
    """Read semi-structured CSV/TSV as a raw cell matrix without header inference."""
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            rows = [[str(cell) for cell in row] for row in csv.reader(f, delimiter=sep)]
    except Exception:
        return pd.DataFrame()
    rows = [row for row in rows if row]
    if not rows:
        return pd.DataFrame()
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(padded, columns=[f"col_{i}" for i in range(width)])


def _resolve_local_path(path_text: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if base_dir is not None:
        return Path(base_dir) / path
    return path


def _load_table_from_path(path_text: str | Path, base_dir: str | Path | None = None) -> pd.DataFrame:
    path = _resolve_local_path(path_text, base_dir=base_dir)
    if not path.exists():
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return _read_table_dataframe(path)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return pd.DataFrame()
        return _load_table_from_record_table(payload)
    if suffix == ".jsonl":
        rows = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    rows.append(json.loads(s))
        except Exception:
            return pd.DataFrame()
        return _load_table_from_record_table(rows)
    return _read_table_dataframe(path)


def _load_table_from_record_table(record_table: Any) -> pd.DataFrame:
    if isinstance(record_table, str):
        try:
            record_table = json.loads(record_table)
        except Exception:
            return pd.DataFrame()
    if isinstance(record_table, dict):
        path_text = (
            record_table.get("path")
            or record_table.get("table_path")
            or record_table.get("table_file")
            or record_table.get("csv_path")
            or record_table.get("parquet_path")
        )
        if path_text:
            return _load_table_from_path(path_text)
        if "columns" in record_table and "data" in record_table:
            cols = [str(x) for x in record_table.get("columns", [])]
            data = record_table.get("data", []) or []
            return pd.DataFrame(data, columns=cols)
        if "headers" in record_table and "rows" in record_table:
            cols = [str(x) for x in record_table.get("headers", [])]
            data = record_table.get("rows", []) or []
            return pd.DataFrame(data, columns=cols)
    if isinstance(record_table, list):
        if not record_table:
            return pd.DataFrame()
        if isinstance(record_table[0], list):
            cols = [f"col_{i}" for i in range(len(record_table[0]))]
            return pd.DataFrame(record_table, columns=cols)
    return pd.DataFrame()


def _resolve_wtq_root(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    if path.is_file():
        return path.parent.parent if path.parent.name == "data" else path.parent
    if (path / "data").exists():
        return path
    if (path / "misc").exists() and (path / "csv").exists():
        return path
    return path


def _resolve_tabfact_root(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    if path.is_file():
        return path.parent.parent if path.parent.name == "tokenized_data" else path.parent
    return path


def _resolve_tablebench_files(dataset_path: str | Path) -> list[Path]:
    path = Path(dataset_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".jsonl", ".json"}])
        if files:
            return files
        files = sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".jsonl", ".json"}])
        if files:
            return files
    return []


def _iter_json_records(files: list[Path]):
    for file_path in files:
        if file_path.suffix.lower() == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    s = line.strip()
                    if not s:
                        continue
                    yield file_path, line_no, json.loads(s)
        else:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = [payload]
            if not isinstance(payload, list):
                continue
            for line_no, record in enumerate(payload, start=1):
                yield file_path, line_no, record


def _load_table_dataframe_from_record(record: dict[str, Any], base_dir: Path) -> pd.DataFrame:
    table_df = _load_table_from_record_table(record.get("table"))
    if not table_df.empty:
        return table_df
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    is_tablebench = str(metadata.get("benchmark", record.get("benchmark", ""))).strip().lower() == "tablebench"
    path_text = (
        record.get("table_path")
        or record.get("table_file")
        or record.get("csv_path")
        or record.get("parquet_path")
    )
    if path_text and is_tablebench:
        table_df = _read_raw_matrix_dataframe(_resolve_local_path(path_text, base_dir=base_dir))
        if not table_df.empty:
            return table_df
    if not path_text and record.get("FileName"):
        filename = str(record.get("FileName", "") or "").strip()
        if filename:
            path_text = str(Path("csv") / f"{filename}.csv")
            table_df = _read_raw_matrix_dataframe(_resolve_local_path(path_text, base_dir=base_dir))
            if not table_df.empty:
                return table_df
    if not path_text and metadata:
        path_text = metadata.get("table_path") or metadata.get("table_file")
    if not path_text:
        return pd.DataFrame()
    return _load_table_from_path(path_text, base_dir=base_dir)


def _build_question_record(
    record: dict[str, Any],
    *,
    idx: int,
    source_path: Path,
    line_no: int,
    benchmark: str,
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    table_df = _load_table_dataframe_from_record(record, base_dir=source_path.parent)
    if table_df.empty:
        return None
    question = str(record.get("question", record.get("instruction", record.get("Question", "")))).strip()
    answer = record.get(
        "answer",
        record.get("response", record.get("ProcessedAnswer", record.get("FinalAnswer", record.get("Answer", "")))),
    )
    answer_type = str(record.get("answer_type", record.get("qtype", record.get("QuestionType", ""))) or "")
    metadata = dict(record.get("metadata", {}) or {})
    metadata.setdefault("benchmark", benchmark)
    metadata.setdefault("source_path", str(source_path))
    metadata.setdefault("source_line", line_no)
    if "table_path" in record:
        metadata.setdefault("table_path", str(record.get("table_path")))
    if "table_file" in record:
        metadata.setdefault("table_file", str(record.get("table_file")))
    if record.get("FileName"):
        metadata.setdefault("table_file", str(record.get("FileName")))
        metadata.setdefault("table_path", str(source_path.parent / "csv" / f"{record.get('FileName')}.csv"))
    field_aliases = {
        "qtype": ("qtype", "QuestionType"),
        "qsubtype": ("qsubtype", "SubQType"),
        "instruction_type": ("instruction_type",),
        "db_id": ("db_id",),
        "table_name": ("table_name", "FileName"),
        "caption": ("caption", "Source"),
        "table_id": ("table_id", "FileName"),
        "sql": ("sql",),
        "evidence": ("evidence", "COT"),
        "difficulty": ("difficulty",),
        "structure_type": ("structure_type", "CompStrucCata"),
        "error_type": ("error_type", "ErrorType"),
        "error_location": ("error_location", "ErrorLocation"),
        "chart_type": ("chart_type",),
    }
    for out_key, aliases in field_aliases.items():
        if out_key in metadata:
            continue
        for key in aliases:
            if key in record:
                metadata[out_key] = record.get(key)
                break
    for key in ("qtype", "qsubtype", "instruction_type", "db_id", "table_name", "caption", "table_id", "sql", "evidence", "difficulty"):
        if key in record and key not in metadata:
            metadata[key] = record.get(key)
    table_rows = [list(table_df.columns)] + table_df.astype(str).fillna("").values.tolist()
    return {
        "idx": idx,
        "question_id": str(record.get("id") or record.get("question_id") or f"{source_path.stem}-{line_no}"),
        "question": question,
        "answer_type": answer_type,
        "answer": answer,
        "scale": str(record.get("scale", "") or ""),
        "answer_from": str(record.get("answer_from", "table")),
        "table": SimpleTable(table_rows),
        "paragraphs": record.get("paragraphs", []),
        "metadata": metadata,
    }


def load_record_rows(dataset_path, limit=0, benchmark="generic"):
    files = _resolve_tablebench_files(dataset_path)
    if not files:
        raise FileNotFoundError(f"Dataset file(s) not found under {dataset_path}")
    rows = []
    for source_path, line_no, record in _iter_json_records(files):
        row = _build_question_record(
            record,
            idx=len(rows) + 1,
            source_path=source_path,
            line_no=line_no,
            benchmark=benchmark,
        )
        if row is None:
            continue
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            return rows
    return rows


def load_wtq_rows(dataset_path, limit=0):
    root = _resolve_wtq_root(dataset_path)
    path = Path(dataset_path)
    if path.is_file() and path.suffix.lower() == ".tsv":
        split_path = path
    else:
        candidates = [
            root / "data" / "pristine-unseen-tables.tsv",
            root / "data" / "test.tsv",
            root / "data" / "dev.tsv",
            root / "data" / "training.tsv",
        ]
        split_path = next((p for p in candidates if p.exists()), None)
        if split_path is None:
            tsvs = sorted((root / "data").glob("*.tsv")) if (root / "data").exists() else []
            split_path = tsvs[0] if tsvs else None
    if split_path is None or not split_path.exists():
        raise FileNotFoundError(f"WTQ split not found under {root}")
    rows = []
    table_meta = {}
    meta_path = root / "misc" / "table-metadata.tsv"
    if meta_path.exists():
        try:
            meta_df = pd.read_csv(meta_path, sep="\t")
            if "contextId" in meta_df.columns:
                meta_df = meta_df.set_index("contextId")
                table_meta = meta_df.to_dict(orient="index")
        except Exception:
            table_meta = {}
    with open(split_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, rec in enumerate(reader, start=1):
            context = str(rec.get("context", "")).strip()
            table_path = root / context
            if not table_path.exists():
                alt = table_path.with_suffix(".tsv")
                table_path = alt if alt.exists() else table_path
            df = _read_table_dataframe(table_path)
            if df.empty:
                continue
            title = ""
            caption = ""
            meta = table_meta.get(context, {}) if isinstance(table_meta, dict) else {}
            if isinstance(meta, dict):
                raw_title = meta.get("title", "")
                raw_caption = meta.get("caption", "")
                title = "" if pd.isna(raw_title) else str(raw_title or "")
                caption = "" if pd.isna(raw_caption) else str(raw_caption or "")
            table_rows = [list(df.columns)] + df.astype(str).fillna("").values.tolist()
            rows.append(
                {
                    "idx": idx,
                    "question_id": str(rec.get("id", f"wtq-{idx}")),
                    "question": str(rec.get("utterance", "")).strip(),
                    "answer_type": "span",
                    "answer": [_normalize_wtq_value(x) for x in str(rec.get("targetValue", "")).split("|") if str(x).strip()],
                    "scale": "",
                    "answer_from": "table",
                    "table": SimpleTable(table_rows),
                    "paragraphs": [],
                    "metadata": {
                        "benchmark": "wtq",
                        "split": split_path.name,
                        "context": context,
                        "title": title,
                        "caption": caption,
                        "table_path": str(table_path),
                    },
                }
            )
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_tabfact_rows(dataset_path, limit=0):
    root = _resolve_tabfact_root(dataset_path)
    path = Path(dataset_path)
    if path.is_file() and path.suffix.lower() == ".json":
        split_path = path
    else:
        candidates = [
            root / "tokenized_data" / "val_examples.json",
            root / "tokenized_data" / "train_examples.json",
            root / "tokenized_data" / "test_examples.json",
        ]
        split_path = next((p for p in candidates if p.exists()), None)
        if split_path is None:
            jsons = sorted((root / "tokenized_data").glob("*.json")) if (root / "tokenized_data").exists() else []
            split_path = jsons[0] if jsons else None
    if split_path is None or not split_path.exists():
        raise FileNotFoundError(f"TabFact split not found under {root}")
    blob = json.loads(split_path.read_text(encoding="utf-8"))
    rows = []
    for table_id, payload in blob.items():
        if not isinstance(payload, list) or len(payload) < 2:
            continue
        statements = payload[0] if isinstance(payload[0], list) else []
        labels = payload[1] if isinstance(payload[1], list) else []
        caption = payload[2] if len(payload) > 2 else ""
        table_path = root / "data" / "all_csv" / str(table_id)
        df = _read_table_dataframe(table_path)
        if df.empty:
            continue
        table_rows = [list(df.columns)] + df.astype(str).fillna("").values.tolist()
        for i, stmt in enumerate(statements):
            if i >= len(labels):
                break
            rows.append(
                {
                    "idx": len(rows) + 1,
                    "question_id": f"{table_id}::{i}",
                    "question": "Fact verification over a table. Decide whether the statement is entailed by the table. Return only 1 or 0.\nStatement: " + str(stmt).strip(),
                    "answer_type": "binary",
                    "answer": _normalize_tabfact_label(labels[i]),
                    "scale": "",
                    "answer_from": "table",
                    "table": SimpleTable(table_rows),
                    "paragraphs": [],
                    "metadata": {
                        "benchmark": "tabfact",
                        "split": split_path.name,
                        "table_id": str(table_id),
                        "caption": str(caption or ""),
                        "table_path": str(table_path),
                        "statement": str(stmt).strip(),
                    },
                }
            )
            if limit > 0 and len(rows) >= limit:
                return rows
    return rows


def load_tablebench_rows(dataset_path, limit=0):
    rows = load_record_rows(dataset_path, limit=limit, benchmark="tablebench")
    for row in rows:
        meta = row.get("metadata", {})
        source_path = Path(str(meta.get("source_path", ""))) if meta.get("source_path") else None
        meta.setdefault("split", source_path.stem if source_path else "")
        meta.setdefault("qtype", str(meta.get("qtype", "")))
        meta.setdefault("qsubtype", str(meta.get("qsubtype", "")))
        meta.setdefault("instruction_type", str(meta.get("instruction_type", "")))
    return rows


class TabularDatasetAdapter:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name or "generic"

    def is_supported(self):
        return True

    def normalize_question(self, question):
        return norm_key(question)

    def answer_from(self, row):
        return row.get("answer_from", "table")

    def question_id(self, row, idx):
        return str(row.get("question_id") or row.get("uid") or f"{self.dataset_name}_{idx}")

    def answer_shape_hint(self, row):
        ans = row.get("answer")
        if isinstance(ans, list):
            return "list"
        return "scalar"

    def scale_hint(self, row):
        return str(row.get("scale", "") or "")

    def load_rows(self, dataset_path, answer_from=None, limit=0):
        ds = str(self.dataset_name or "generic").strip().lower()
        if ds == "tatqa":
            from compresstab.datasets.tatqa_data import load_tatqa_rows

            source_filter = "" if answer_from is None else answer_from
            return load_tatqa_rows(dataset_path, answer_from=source_filter, limit=limit)
        if ds in {"wtq", "wikitq"}:
            return load_wtq_rows(dataset_path, limit=limit)
        if ds == "tabfact":
            return load_tabfact_rows(dataset_path, limit=limit)
        if ds == "tablebench":
            return load_tablebench_rows(dataset_path, limit=limit)
        if ds == "hitab":
            from compresstab.datasets.hitab_data import load_hitab_rows

            return load_hitab_rows(dataset_path, limit=limit)
        if ds == "birdqa":
            return load_record_rows(dataset_path, limit=limit, benchmark="birdqa")
        if ds in {"syntabfact", "synthetic_tabfact", "tabfact_large"}:
            return load_record_rows(dataset_path, limit=limit, benchmark="tabfact")
        answer_from = "table" if answer_from is None else answer_from
        data = Path(dataset_path).read_text(encoding="utf-8")
        items = json.loads(data)
        if isinstance(items, dict):
            items = items.get("data", items.get("examples", items.get("rows", [])))
        if isinstance(items, dict):
            items = [items]
        rows = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            if answer_from and str(item.get("answer_from", "table")) != answer_from:
                continue
            table = item.get("table", item.get("table_data", item.get("rows", [])))
            if isinstance(table, dict):
                table = table.get("table", table.get("rows", []))
            if isinstance(table, dict):
                table = table.get("data", table.get("cells", []))
            if isinstance(table, pd.DataFrame):
                table_obj = SimpleTable(table)
            else:
                table_obj = SimpleTable(table or [])
            rows.append(
                {
                    "idx": idx,
                    "question_id": self.question_id(item, idx),
                    "question": str(item.get("question", "")),
                    "answer_type": str(item.get("answer_type", "")),
                    "answer": item.get("answer", ""),
                    "scale": self.scale_hint(item),
                    "answer_from": str(item.get("answer_from", answer_from)),
                    "table": table_obj,
                    "paragraphs": item.get("paragraphs", []),
                }
            )
            if limit > 0 and len(rows) >= limit:
                break
        return rows

    def normalize_prediction(self, row):
        ds = str(self.dataset_name or "generic").strip().lower()
        if ds == "tatqa":
            from compresstab.datasets.tatqa_metrics import align_tatqa_prediction

            pred_answer, pred_scale = align_tatqa_prediction(
                row.get("pred_answer", ""),
                row.get("pred_scale", ""),
            )
            row["pred_answer"] = pred_answer
            row["pred_scale"] = pred_scale
        elif ds in {"wtq", "wikitq"}:
            pred = row.get("pred_answer", "")
            if isinstance(pred, str):
                parts = [p.strip() for p in re.split(r"[|]", pred) if p.strip()]
                if len(parts) == 1:
                    row["pred_answer"] = _normalize_wtq_value(parts[0])
                elif len(parts) > 1:
                    row["pred_answer"] = [_normalize_wtq_value(x) for x in parts]
        elif ds == "birdqa":
            pred = row.get("pred_answer", "")
            if isinstance(pred, str):
                parts = [p.strip() for p in re.split(r"[|]", pred) if p.strip()]
                if len(parts) == 1:
                    row["pred_answer"] = _normalize_wtq_value(parts[0])
                elif len(parts) > 1:
                    row["pred_answer"] = [_normalize_wtq_value(x) for x in parts]
        elif ds == "tabfact":
            pred = _normalize_tabfact_label(row.get("pred_answer", None))
            if pred is not None:
                row["pred_answer"] = pred
        elif ds in {"syntabfact", "synthetic_tabfact", "tabfact_large"}:
            pred = _normalize_tabfact_label(row.get("pred_answer", None))
            if pred is not None:
                row["pred_answer"] = pred
        elif ds == "tablebench":
            qtype, _qsubtype = _tablebench_row_type(row)
            pred = row.get("pred_answer", "")
            if qtype == "Visualization" and isinstance(pred, str) and re.search(r"\by_?references\b", pred, flags=re.I):
                row["pred_answer"] = _extract_final_answer_text(pred)
            else:
                row["pred_answer"] = _normalize_tablebench_answer(pred)
        elif ds == "hitab":
            from compresstab.datasets.hitab_metrics import normalize_hitab_prediction

            row["pred_answer"] = normalize_hitab_prediction(row.get("pred_answer", ""))
        return row

    def score_row(self, row):
        ds = str(self.dataset_name or "generic").strip().lower()
        if ds == "tatqa":
            from compresstab.datasets.tatqa_metrics import score_tatqa_row

            return score_tatqa_row(row)
        if ds == "hitab":
            from compresstab.datasets.hitab_metrics import score_hitab_row

            return score_hitab_row(row)
        if ds in {"wtq", "wikitq"}:
            pred = row.get("pred_answer", "")
            gold = row.get("answer", "")
            em = 1.0 if _wtq_denotation_equal(gold, pred) else 0.0
            return em, em
        if ds == "birdqa":
            pred = row.get("pred_answer", "")
            gold = row.get("answer", "")
            pred_vals = pred if isinstance(pred, list) else [pred]
            gold_vals = gold if isinstance(gold, list) else [gold]
            pred_norm = _normalize_counter_values(pred_vals)
            gold_norm = _normalize_counter_values(gold_vals)
            em = 1.0 if pred_norm == gold_norm and bool(pred_norm) else 0.0
            return em, em
        if ds == "tabfact":
            pred = _normalize_tabfact_label(row.get("pred_answer", None))
            gold = _normalize_tabfact_label(row.get("answer", None))
            if pred is None or gold is None:
                return 0.0, 0.0
            em = 1.0 if pred == gold else 0.0
            return em, em
        if ds in {"syntabfact", "synthetic_tabfact", "tabfact_large"}:
            pred = _normalize_tabfact_label(row.get("pred_answer", None))
            gold = _normalize_tabfact_label(row.get("answer", None))
            if pred is None or gold is None:
                return 0.0, 0.0
            em = 1.0 if pred == gold else 0.0
            return em, em
        if ds == "tablebench":
            score = _tablebench_official_local_score(row)
            return score, score
        pred = row.get("pred_answer", "")
        gold = row.get("answer", "")
        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)
        return em, f1

    def summarize(self, records):
        ds = str(self.dataset_name or "generic").strip().lower()
        if ds == "tatqa":
            from compresstab.datasets.tatqa_metrics import summarize_tatqa

            return summarize_tatqa(records)
        if ds == "hitab":
            from compresstab.datasets.hitab_metrics import summarize_hitab

            return summarize_hitab(records)
        if not records:
            return {"count": 0, "em": 0.0, "f1": 0.0, "scale": 0.0, "op": 0.0}
        em = sum(float(r.get("em", 0.0)) for r in records) / len(records) * 100.0
        f1 = sum(float(r.get("f1", 0.0)) for r in records) / len(records) * 100.0
        return {
            "count": len(records),
            "em": em,
            "f1": f1,
            "scale": 0.0,
            "op": 0.0,
            "family": dict(Counter(r.get("compiled", {}).get("family", "missing") for r in records)),
        }

    def write_cases(self, path, records):
        ds = str(self.dataset_name or "generic").strip().lower()
        if ds == "tatqa":
            from compresstab.datasets.tatqa_metrics import write_tatqa_cases

            return write_tatqa_cases(path, records)
        if ds == "hitab":
            from compresstab.datasets.hitab_metrics import write_hitab_cases

            return write_hitab_cases(path, records)
        lines = []
        for row in records:
            if float(row.get("em", 0.0)) >= 1.0:
                continue
            lines.append(f"[CASE #{row.get('idx')}]")
            lines.append(f"Dataset: {ds}")
            lines.append(f"Question: {row.get('question')}")
            lines.append(f"Gold: {row.get('answer')}")
            lines.append(f"Pred: {row.get('pred_answer')}")
            lines.append(str(row.get("rendered_witness", "")))
            lines.append("")
        from pathlib import Path

        Path(path).write_text("\n".join(lines), encoding="utf-8")


def normalize_answer(x):
    if isinstance(x, list):
        return "|".join(normalize_answer(v) for v in x)
    return norm_key(str(x))


def _normalize_for_score(x):
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    if x is None:
        return []
    s = norm_key(str(x))
    if not s:
        return []
    return s.split()


def exact_match(pred, gold):
    return 1.0 if norm_key(str(pred)) == norm_key(str(gold)) and norm_key(str(pred)) else 0.0


def token_f1(pred, gold):
    pred_toks = _normalize_for_score(pred)
    gold_toks = _normalize_for_score(gold)
    if not pred_toks or not gold_toks:
        return 1.0 if not pred_toks and not gold_toks else 0.0
    pred_counter = Counter(pred_toks)
    gold_counter = Counter(gold_toks)
    overlap = sum(min(pred_counter[t], gold_counter[t]) for t in pred_counter)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)
