"""Fetch pinned official benchmark repositories into an excluded directory."""

import argparse
from pathlib import Path
import subprocess

SOURCES = {
    "wtq": ("https://github.com/ppasupat/WikiTableQuestions.git", "7d455a5a707b96341ef72aff9428749d443d8aa9"),
    "hitab": ("https://github.com/microsoft/HiTab.git", "d179602662b490249baf068a76fbe4137029126e"),
    "tatqa": ("https://github.com/NExTplusplus/TAT-QA.git", "870accc41953dcde885aabeb963d94aabdc0fbc3"),
    "longtablebench": ("https://github.com/liyaooi/LongTableBench.git", "8db45ac102632cba3bcc1393e1023f0689da7d1c"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(SOURCES))
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    url, commit = SOURCES[args.dataset]
    target = args.root / args.dataset
    if target.exists():
        raise FileExistsError(f"Refusing to replace {target}")
    target.mkdir(parents=True)
    for command in (["git", "init", "-q", str(target)],
                    ["git", "-C", str(target), "remote", "add", "origin", url],
                    ["git", "-C", str(target), "fetch", "--depth", "1", "origin", commit],
                    ["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"]):
        subprocess.run(command, check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise RuntimeError("Downloaded revision does not match the requested revision")


if __name__ == "__main__":
    main()
