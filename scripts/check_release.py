"""Check source release files for private artifacts before packaging."""

import re
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output([
        "git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z",
    ]).decode().split("\0")
    paths = [Path(name) for name in tracked if name]
    if not paths:
        raise RuntimeError("Stage the intended release files before checking")
    forbidden_parts = {"result", "outputs", "cache", "__pycache__", ".venv", "models"}
    figures = {
        Path("assets/framework.pdf"): b"%PDF-",
        Path("assets/framework.png"): b"\x89PNG\r\n\x1a\n",
    }
    patterns = {
        "private filesystem path": re.compile(r"/(?:home|Users|data\d*)/[A-Za-z0-9_.-]+/"),
        "API credential": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
        "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "email address": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    }
    failures = []
    for relative in paths:
        path = root / relative
        if path.is_symlink():
            failures.append(f"{relative}: symbolic link")
            continue
        if forbidden_parts.intersection(relative.parts):
            failures.append(f"{relative}: runtime artifact directory")
        if relative.name.startswith(".env") and relative.name != ".env.example":
            failures.append(f"{relative}: environment file")
        if relative in figures:
            if not path.read_bytes().startswith(figures[relative]):
                failures.append(f"{relative}: invalid figure format")
            continue
        if relative.suffix in {".pdf", ".png", ".pt", ".bin", ".log", ".pyc", ".pkl"}:
            failures.append(f"{relative}: binary, output, or cache file")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"{relative}: unexpected binary file")
            continue
        email_scan_text = text
        if relative.parts[0] == "results":
            if relative.suffix == ".jsonl":
                ids = set()
                metadata = []
                required = {"question_id", "answer"}
                if relative.parts[1] == "tatqa":
                    required.add("scale")
                for number, line in enumerate(text.splitlines(), 1):
                    try:
                        row = json.loads(line)
                        if set(row) != required or not isinstance(row["question_id"], str):
                            raise ValueError("invalid prediction fields")
                        if row["question_id"] in ids:
                            raise ValueError("duplicate question ID")
                        ids.add(row["question_id"])
                        metadata.append(json.dumps({key: value for key, value in row.items()
                                                    if key != "answer"}))
                    except (ValueError, TypeError, KeyError) as error:
                        failures.append(f"{relative}:{number}: {error}")
                if not ids:
                    failures.append(f"{relative}: empty prediction file")
                email_scan_text = "\n".join(metadata)
            elif relative != Path("results/README.md"):
                failures.append(f"{relative}: unexpected result file")
        for label, pattern in patterns.items():
            # Email-valued answers are data; other fields still undergo scanning.
            scanned_text = email_scan_text if label == "email address" else text
            if pattern.search(scanned_text):
                failures.append(f"{relative}: {label}")
    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"Release scan passed for {len(paths)} source files")


if __name__ == "__main__":
    main()
