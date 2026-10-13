import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "release_check", Path(__file__).parents[1] / "scripts/check_release.py")
release_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_check)


def scan(monkeypatch, tmp_path, relative, content):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    monkeypatch.setattr(release_check, "__file__", str(tmp_path / "scripts/check_release.py"))
    monkeypatch.setattr(release_check.subprocess, "check_output", lambda _: (relative + "\0").encode())
    release_check.main()


def test_prediction_email_is_answer_data(monkeypatch, tmp_path):
    address = "office" + "@" + "example.org"
    scan(monkeypatch, tmp_path, "results/longtablebench/main/example.jsonl",
         '{"question_id":"q1","answer":"' + address + '"}\n')


def test_non_prediction_email_is_flagged(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="email address"):
        scan(monkeypatch, tmp_path, "README.md", "person" + "@" + "example.org")


def test_prediction_id_email_is_still_flagged(monkeypatch, tmp_path):
    address = "person" + "@" + "example.org"
    with pytest.raises(RuntimeError, match="email address"):
        scan(monkeypatch, tmp_path, "results/wtq/main/example.jsonl",
             '{"question_id":"' + address + '","answer":"x"}\n')


@pytest.mark.parametrize("content,reason", [
    ('{"question_id":"q1","answer":"x","trace":"local"}\n', "invalid prediction fields"),
    ('{"question_id":"q1","answer":"x"}\n' * 2, "duplicate question ID"),
    ('', "empty prediction file"),
    ('not-json\n', "Expecting value"),
])
def test_invalid_results_are_flagged(monkeypatch, tmp_path, content, reason):
    with pytest.raises(RuntimeError, match=reason):
        scan(monkeypatch, tmp_path, "results/wtq/main/example.jsonl", content)


def test_prediction_credential_is_still_flagged(monkeypatch, tmp_path):
    credential = "sk-" + "x" * 24
    with pytest.raises(RuntimeError, match="API credential"):
        scan(monkeypatch, tmp_path, "results/wtq/main/example.jsonl",
             '{"question_id":"q1","answer":"' + credential + '"}\n')
