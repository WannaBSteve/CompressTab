# Third-Party Assets

Benchmark data and model files are downloaded separately and remain subject to
their upstream terms. Attribution is retained through the official links and
revisions in `docs/data.md` and `scripts/fetch_data.py`.

No third-party dataset, model checkpoint, or official scoring implementation is
redistributed here. Evaluation loads the scorer from the downloaded benchmark.
For WikiTableQuestions, the Python 2 scorer is syntax-translated in memory for
Python 3 without changing its scoring rules.

This source package does not assign a license to the project's original code.
A redistribution license must be selected by the rights holders before a
licensed open-source release.
