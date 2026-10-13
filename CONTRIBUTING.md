# Contributing

Useful contributions include evidence operators, dataset adapters, reader
integrations, and compact examples of difficult table reasoning.

## Local Checks

```bash
python -m pip install -e '.[test,evaluation]'
pytest -q
COMPRESSTAB_TEST_TOKENIZER=models/tokenizer.json pytest -q
python scripts/check_release.py
```

Use a local tokenizer for the tokenizer-backed tests. Tests and examples should
run without paid API access. Keep API credentials in environment variables.

## Changes

- Describe the behavior your change enables or fixes.
- Add a focused test. Prefer small, independently constructed tables.
- For an encoding, test its preserved distinctions and source bindings.
- For a search change, compare against exhaustive small-space oracles.
- Keep public examples, configuration, and interface documentation consistent.

Do not commit model weights, downloaded benchmarks, API responses, or local
environment files. Report credential exposure privately through GitHub's
security reporting when available rather than including the secret in an issue.
