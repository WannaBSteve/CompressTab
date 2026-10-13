import json
import math
import re
from collections import defaultdict
from pathlib import Path


MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

MONTH_ALIASES = {
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "sept": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}

SCALE_MULTIPLIERS = {
    "percent": 0.01,
    "%": 0.01,
    "thousand": 1_000.0,
    "thousands": 1_000.0,
    "k": 1_000.0,
    "million": 1_000_000.0,
    "millions": 1_000_000.0,
    "m": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "billions": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
}


def norm_text(x):
    if x is None:
        return ""
    return str(x).replace("\n", " ").strip()


def norm_key(x):
    s = norm_text(x).lower()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    s = re.sub(r"[^a-z0-9%$€£/().,\-+ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(x):
    return [t for t in re.findall(r"[a-z0-9%$€£]+", norm_key(x)) if t]


def approx_token_count(x):
    if isinstance(x, (dict, list)):
        text = json.dumps(x, ensure_ascii=False, default=str)
    else:
        text = norm_text(x)
    if not text:
        return 0
    return len(tokenize(text))


def table_artifact_text(table_artifact):
    headers = [norm_text(x) for x in table_artifact.get("headers", []) if norm_text(x)]
    rows = table_artifact.get("rows", []) or []
    parts = []
    if headers:
        parts.append(" | ".join(headers))
    for row in rows:
        row_text = [norm_text(x) for x in row if norm_text(x)]
        if row_text:
            parts.append(" | ".join(row_text))
    return "\n".join(parts)


def table_artifact_token_count(table_artifact):
    return approx_token_count(table_artifact_text(table_artifact))


def get_years(x):
    s = norm_text(x)
    years = re.findall(r"(19\d{2}|20\d{2}|21\d{2})", s)
    if years:
        return years
    compact = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    return re.findall(r"(19\d{2}|20\d{2}|21\d{2})", compact)


def get_months(x):
    s = norm_key(x)
    out = []
    for token in re.findall(r"[a-z]+", s):
        month = token if token in MONTHS else MONTH_ALIASES.get(token)
        if month and month not in out:
            out.append(month)
    return out


def get_year_intervals(x):
    """Return normalized year/period intervals without using question answers.

    Examples:
    - 1940/41 -> {"start": "1940", "end": "1941", "label": "1940/41"}
    - 2018-2019 -> {"start": "2018", "end": "2019", "label": "2018-2019"}
    """
    s = norm_text(x)
    out = []
    for match in re.finditer(r"\b((?:19|20|21)\d{2})\s*[/\-]\s*((?:19|20|21)?\d{2})\b", s):
        start = match.group(1)
        end_raw = match.group(2)
        if len(end_raw) == 2:
            end = start[:2] + end_raw
        else:
            end = end_raw
        try:
            if int(end) < int(start) and len(end_raw) == 2:
                end = str(int(start[:2]) * 100 + int(end_raw) + 100)
        except Exception:
            pass
        item = {"label": match.group(0), "start": start, "end": end, "width": max(1, int(end) - int(start) + 1)}
        if item not in out:
            out.append(item)
    return out


def infer_unit_scale(x):
    s = norm_key(x)
    units = []
    for token in ["percent", "%", "thousand", "million", "billion", "bn"]:
        if token == "%" and "%" not in s:
            continue
        if token != "%" and re.search(rf"\b{re.escape(token)}s?\b", s):
            units.append("billion" if token == "bn" else token.rstrip("s"))
        elif token == "%":
            units.append("percent")
    currency = []
    raw = norm_text(x)
    if "$" in raw:
        currency.append("usd_or_dollar")
    if "£" in raw:
        currency.append("gbp_or_pound")
    if "€" in raw:
        currency.append("eur_or_euro")
    return {
        "scale_terms": list(dict.fromkeys(units)),
        "currency_terms": currency,
    }


def parse_number_with_unit(x):
    """Parse a numeric literal plus inline unit into a normalized value.

    This is for profiling/predicate synopses. It does not replace raw cell text
    in the witness because scale declarations can be table-scoped.
    """
    s = norm_text(x)
    base = parse_number(s)
    scale = infer_unit_scale(s)
    if base is None:
        cleaned = s.replace(",", "")
        match = re.search(r"[-+]?\(?\d+(?:\.\d+)?\)?", cleaned)
        if match:
            base = parse_number(match.group(0))
    if base is None:
        return None
    multiplier = 1.0
    for term in scale.get("scale_terms", []):
        if term in {"percent"}:
            continue
        multiplier = max(multiplier, SCALE_MULTIPLIERS.get(term, 1.0))
    return float(base) * multiplier


def canonical_cell_semantics(x):
    text = norm_text(x)
    number = parse_number(text)
    number_with_unit = parse_number_with_unit(text)
    years = get_years(text)
    intervals = get_year_intervals(text)
    months = get_months(text)
    unit_scale = infer_unit_scale(text)
    if intervals:
        semantic_type = "year_interval"
    elif months and years:
        semantic_type = "date"
    elif years and not number:
        semantic_type = "year_like"
    elif number is not None:
        semantic_type = "numeric"
    elif number_with_unit is not None:
        semantic_type = "numeric_with_unit"
    elif months:
        semantic_type = "month_like"
    elif text:
        semantic_type = "text"
    else:
        semantic_type = "missing"
    return {
        "raw": text,
        "norm": norm_key(text),
        "number": format_number(number) if number is not None else "",
        "number_with_unit": format_number(number_with_unit) if number_with_unit is not None else "",
        "years": [str(y) for y in years],
        "year_intervals": intervals,
        "months": months,
        "unit_scale": unit_scale,
        "semantic_type": semantic_type,
    }


def parse_number(x):
    s = norm_text(x)
    if not s:
        return None
    s = s.replace("$", "").replace("€", "").replace("£", "").strip()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace(",", "").replace("%", "").strip()
    s = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", s)
    if s in {"", "-", "--", "nm", "n/m", "na", "n/a"}:
        return None
    try:
        val = float(s)
        if not math.isfinite(val):
            return None
        return -val if neg else val
    except Exception:
        return None


def format_number(x):
    if x is None:
        return ""
    if not math.isfinite(float(x)):
        return ""
    if abs(float(x) - round(float(x))) < 1e-9:
        return str(int(round(float(x))))
    out = f"{float(x):.2f}"
    return out.rstrip("0").rstrip(".")


def score_overlap(a, b):
    ak = norm_key(a)
    bk = norm_key(b)
    at = set(tokenize(a))
    bt = set(tokenize(b))
    score = 0.0
    score += sum(2.0 if len(t) > 4 else 1.0 for t in at & bt)
    if bk and bk in ak:
        score += 6.0
    if ak and ak in bk:
        score += 2.0
    score += 2.0 * len(set(get_years(a)) & set(get_years(b)))
    return score


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, obj):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_div(a, b):
    if b is None or abs(float(b)) < 1e-9:
        return None
    return float(a) / float(b)


def avg(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else None


def row_blob(row):
    return " | ".join(norm_text(x) for x in row if norm_text(x))


def flatten_topk_unique(xs, k):
    out = []
    seen = set()
    for x in xs:
        nk = norm_key(x)
        if nk and nk not in seen:
            seen.add(nk)
            out.append(x)
        if len(out) >= k:
            break
    return out


def build_text_inverted_index(strings):
    inv = defaultdict(list)
    for key, text in strings:
        for tok in set(tokenize(text)):
            inv[tok].append(key)
    return {k: v for k, v in inv.items()}
