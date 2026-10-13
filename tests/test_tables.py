from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from compresstab.datasets.tabular_adapter import _read_table_dataframe


def test_backslash_escaped_quotes_commas_and_multiline_cells_stay_atomic(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text(
        '"Title","Year","Value"\n'
        '"\\"Simple\\"","2008","8.5"\n'
        '"\\"Grannies, Guns, Love Mints\\"","2009","9.47"\n'
        '"\\"Song\\"\n(featuring Artist)","2010","10"\n',
        encoding="utf-8",
    )

    frame = _read_table_dataframe(path)

    assert frame.shape == (3, 3)
    assert frame.iloc[1]["Title"] == '"Grannies, Guns, Love Mints"'
    assert frame.iloc[2]["Title"] == '"Song"\n(featuring Artist)'
    assert str(frame.iloc[1]["Year"]) == "2009"


def test_duplicate_headers_remain_separately_addressable(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text(
        '"Winner","Score","Runner Up","Score"\n'
        '"A","20","B","10"\n',
        encoding="utf-8",
    )

    frame = _read_table_dataframe(path)

    assert list(frame.columns) == ["Winner", "Score", "Runner Up", "Score.1"]
    assert frame.iloc[0]["Score"] == 20
    assert frame.iloc[0]["Score.1"] == 10
