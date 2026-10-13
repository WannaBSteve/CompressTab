"""Reproducible tokenizer-backed evidence accounting."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_TOKENIZER_PATH = Path(os.environ.get("COMPRESSTAB_TOKENIZER", "models/tokenizer.json"))


class ExactTokenCounter:
    """Count serialized evidence with a frozen local tokenizer vocabulary."""

    def __init__(self, path: str | Path = DEFAULT_TOKENIZER_PATH):
        tokenizer_path = Path(path).expanduser().resolve()
        if tokenizer_path.is_dir():
            tokenizer_path = tokenizer_path / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Exact token accounting requires the tokenizers package; install the tokenizers package."
            ) from exc
        self.path = tokenizer_path
        self.name = f"tokenizers:{tokenizer_path}"
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def count(self, value: Any) -> int:
        text = str(value if value is not None else "")
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
