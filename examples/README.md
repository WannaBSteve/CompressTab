# Examples

## Local Compilation

`compile.jsonl` contains a small table and a supplied Answerability Signature.
The compiler resolves the requirements and selects a compact representation.
No API key or model call is needed.

```bash
python examples/compile_table.py --tokenizer models/tokenizer.json
```

## Question Answering

`questions.jsonl` supplies the source table and a question, without a signature
or reference answer. Use the complete pipeline with a configured model endpoint:

```bash
compresstab answer --input examples/questions.jsonl \
  --output outputs/answers.jsonl --config configs/qwen.json \
  --tokenizer models/tokenizer.json
```

`OPENAI_BASE_URL` and `OPENAI_API_KEY` configure the endpoint. Set the model in
`configs/qwen.json` to the ID served by that endpoint. Inputs may contain a
single `table`, multiple `tables`, or associated `paragraphs`; see the
[interface guide](../docs/interface.md).
