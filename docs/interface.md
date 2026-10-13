# Interface

## JSONL Inputs

Each line contains `question_id`, `question`, and either `table` or `tables`.
A simple `table` is an array whose first row is the header. Structured table
objects can additionally contain row and column hierarchy paths and metadata
describing source units. `paragraphs` supplies associated source text.

`compile` additionally requires `answerability_signature`. It bypasses planning
and answering and makes no model calls. See `examples/compile.jsonl`.

The signature schema is defined by `AnswerabilitySignature` in `evidence.py`:

| Paper concept | JSON fields |
| --- | --- |
| Scope | `slices[].source`, `columns`, `rows`, `where`, `where_logic` |
| Bindings | `slices[].links`, hierarchy paths in the canonical table |
| Distinctions | `slices[].resolution` with `exact`, `identity`, `equality`, `order`, or `threshold` |
| Derived quantities | `slices[].derived` with typed operators, operands, and required modes |

Empty explicit row selection means resolve the declared scope, not take an
empty table. Source references identify the canonical table, row, and column.
The planner also supplies a structured answer expression used to retain the
question's requested operation, answer shape, units, and multiplicity.

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | Required | Answer Agent model ID |
| `planner_model` | `model` | Backward Planner and Signature Auditor model ID |
| `auditor_model` | `model` | Evidence Auditor model ID |
| `evidence_target` | 4096 | Requested evidence-token target, not a hard truncation bound |
| `inspection_tokens` | 2048 | Source-inspection observation budget |
| `max_corrections` | 5 | Maximum correction rounds |
| `signature_checks` | 1 | Source-grounding signature-check limit |

The runner's `--max-completion-tokens` defaults to 2048 per model response.
Each stage may make multiple requests; a correction round is not one API call.
Invalid JSON may require a format retry. Network failures are reported rather
than recorded as successful answers. `--base-url` overrides the explicit
environment endpoint. Use environment variables for secrets, never JSON files.

## Outputs

`answer` writes `answer`, `scale`, `status`, `evidence`, `signature`, `events`,
`usage`, `question_id`, and `tokenizer_sha256`. `raw` writes its answer, scale,
usage, question ID, and tokenizer hash. `compile` writes evidence, tokens,
signature, and the encoding trace.

The library API is:

```python
from compresstab.pipeline import CompressTabPipeline, PipelineConfig
from compresstab.llm.client import JsonLLMClient
from compresstab.tokenization import ExactTokenCounter

pipeline = CompressTabPipeline(
    JsonLLMClient(),
    ExactTokenCounter("models/tokenizer.json"),
    PipelineConfig(model="Qwen3-8B"),
)
answer = pipeline.answer("Which value is largest?", {
    "table": [["item", "value"], ["A", "12"], ["B", "7"]]
})
```

All model inference uses source inputs, not reference answers. The separate
evaluation command reads `references.jsonl` after predictions are written.
Runtime outputs can contain dataset text and deployment metadata and are
excluded from the source release.
