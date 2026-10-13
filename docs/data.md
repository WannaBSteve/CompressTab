# Data and Official Evaluation

Run commands from the repository root after installation. Downloads go to
`data/raw/`; prepared inputs and labels go to `data/prepared/`. Both directories
are excluded from the source package. The downloader pins the revision and
refuses to overwrite an existing checkout.

## Sources

| Dataset | Official repository | Pinned revision | Upstream terms |
| --- | --- | --- | --- |
| WikiTableQuestions | https://github.com/ppasupat/WikiTableQuestions | `7d455a5a707b96341ef72aff9428749d443d8aa9` | CC BY-SA 4.0 |
| HiTab | https://github.com/microsoft/HiTab | `d179602662b490249baf068a76fbe4137029126e` | Computational Use of Data Agreement |
| TAT-QA | https://github.com/NExTplusplus/TAT-QA | `870accc41953dcde885aabeb963d94aabdc0fbc3` | Upstream MIT repository; respect source-document terms |
| LongTableBench | https://github.com/liyaooi/LongTableBench | `8db45ac102632cba3bcc1393e1023f0689da7d1c` | Upstream MIT repository |

Read each upstream license before redistribution. This package distributes
download instructions and a sample-ID manifest, not the benchmark datasets.
Allow several hundred MB for the table datasets and their supporting files.

```bash
python scripts/fetch_data.py wtq
python scripts/fetch_data.py hitab
python scripts/fetch_data.py tatqa
python scripts/fetch_data.py longtablebench
```

## Prepare Evaluation Inputs

WikiTableQuestions uses the official unseen-table test file. HiTab preserves
the row and column hierarchy from its official tables. TAT-QA uses the public
gold file and retains associated paragraphs. Explicit WTQ and TAT-QA filenames
select the split; `--split` records that choice, rather than inferring another
file from its name.

```bash
python scripts/prepare_data.py wtq \
  --source data/raw/wtq/data/pristine-unseen-tables.tsv \
  --output-dir data/prepared/wtq

python scripts/prepare_data.py hitab \
  --source data/raw/hitab --split test \
  --output-dir data/prepared/hitab

python scripts/prepare_data.py tatqa \
  --source data/raw/tatqa/dataset_raw/tatqa_dataset_test_gold.json \
  --output-dir data/prepared/tatqa

python scripts/prepare_data.py longtablebench \
  --source data/raw/longtablebench --formats markdown \
  --manifest data/manifests/longtablebench_sample.json \
  --output-dir data/prepared/longtablebench
```

The checked-in manifest selects 300 conversation seeds, expanding to 545
question rounds in Markdown. It records the random seed, stratum allocations,
and exact selected IDs. A seed conversation and a question round are different
units. The pinned upstream release contains 953 seeds; preparation uses that
release rather than assuming the size of an earlier benchmark snapshot.

Omit `--manifest` to use all released conversations. Supported serialization
names are `markdown html json latex sql xml csv`. Supply the names explicitly
to `--formats`; each format has its own conversation history. No answer reuse
across formats is performed by the runner.

Preparation creates:

- `inputs.jsonl`: source tables, question IDs, questions, and source context.
- `references.jsonl`: reference answers and evaluation-only annotations.
- `manifest.json`: split, sample count, formats, and SHA-256 hashes of both files.

The inference commands read only `inputs.jsonl`.

## Score Predictions

Use the corresponding official checkout, with the same prepared reference file
for every method:

```bash
python scripts/evaluate.py hitab \
  --references data/prepared/hitab/references.jsonl \
  --predictions outputs/hitab.jsonl \
  --official-root data/raw/hitab \
  --output outputs/hitab-metrics.json
```

Replace `hitab` and its paths with `wtq`, `tatqa`, or `longtablebench` as needed.

| Dataset | Evaluation |
| --- | --- |
| WikiTableQuestions | Official denotation accuracy (`evaluator.py`) |
| HiTab | Official `hmt_score` accuracy |
| TAT-QA | Official `TaTQAEmAndF1`, including predicted answer scale |
| LongTableBench | Official `cal_score3` structured-answer F1 |

LongTableBench reports round-macro F1, conversation-macro F1, task F1, and
task-macro F1 under separate names. Conversation F1 averages its expected
rounds; task F1 averages conversation scores in that task; task-macro F1 averages
the represented tasks. Missing predictions score zero in all relevant groups.
These averages are not interchangeable.

## Additional Release Materials

Suitable source-package additions include independently constructed examples,
schema documentation, operator definitions, and reproducible sample manifests.
Benchmark text or prepared data requires checking its source license. Model
weights are linked rather than bundled. The [prediction index](../results/README.md)
lists the answer files for the paper's evaluation settings. Runtime traces
and deployment credentials are not part of the repository.
