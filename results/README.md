# CompressTab Predictions

Final predictions for the evaluation settings reported in the paper.

Each JSONL row contains `question_id` and the predicted `answer`. TAT-QA also
includes the predicted `scale`. Reference answers are supplied separately by
the benchmark datasets. A missing prediction is represented by `null` and
receives zero credit.

## Files

| Predictions | Questions | Metric (%) |
| --- | ---: | --- |
| [wtq/main/compresstab_deepseek_v3_2.jsonl](wtq/main/compresstab_deepseek_v3_2.jsonl) | 4344 | accuracy: 85.15 |
| [hitab/main/compresstab_deepseek_v3_2.jsonl](hitab/main/compresstab_deepseek_v3_2.jsonl) | 1584 | accuracy: 74.49 |
| [wtq/main/vanilla_deepseek_v3_2.jsonl](wtq/main/vanilla_deepseek_v3_2.jsonl) | 4344 | accuracy: 78.41 |
| [hitab/main/vanilla_deepseek_v3_2.jsonl](hitab/main/vanilla_deepseek_v3_2.jsonl) | 1584 | accuracy: 72.16 |
| [hitab/main/dater_deepseek_v3_2.jsonl](hitab/main/dater_deepseek_v3_2.jsonl) | 1584 | accuracy: 72.41 |
| [tatqa/main/vanilla_deepseek_v3_2.jsonl](tatqa/main/vanilla_deepseek_v3_2.jsonl) | 1663 | em: 58.27, f1: 65.94 |
| [tatqa/main/tablerag_deepseek_v3_2.jsonl](tatqa/main/tablerag_deepseek_v3_2.jsonl) | 1663 | em: 54.96, f1: 63.69 |
| [hitab/main/vanilla_qwen3_8b.jsonl](hitab/main/vanilla_qwen3_8b.jsonl) | 1584 | accuracy: 50.19 |
| [tatqa/main/vanilla_qwen3_8b.jsonl](tatqa/main/vanilla_qwen3_8b.jsonl) | 1663 | em: 38.73, f1: 47.43 |
| [tatqa/main/compresstab_deepseek_v3_2.jsonl](tatqa/main/compresstab_deepseek_v3_2.jsonl) | 1663 | em: 84.55, f1: 89.58 |
| [wtq/main/chain_of_table_deepseek_v3_2.jsonl](wtq/main/chain_of_table_deepseek_v3_2.jsonl) | 4344 | accuracy: 70.63 |
| [wtq/main/dater_deepseek_v3_2.jsonl](wtq/main/dater_deepseek_v3_2.jsonl) | 4344 | accuracy: 73.80 |
| [wtq/main/table_critic_deepseek_v3_2.jsonl](wtq/main/table_critic_deepseek_v3_2.jsonl) | 4344 | accuracy: 78.13 |
| [wtq/transfer/direct_answer_qwen3_8b.jsonl](wtq/transfer/direct_answer_qwen3_8b.jsonl) | 4344 | accuracy: 49.24 |
| [wtq/transfer/direct_answer_compresstab_qwen3_8b.jsonl](wtq/transfer/direct_answer_compresstab_qwen3_8b.jsonl) | 4344 | accuracy: 50.85 |
| [wtq/transfer/dater_qwen3_8b.jsonl](wtq/transfer/dater_qwen3_8b.jsonl) | 4344 | accuracy: 49.68 |
| [wtq/transfer/dater_compresstab_qwen3_8b.jsonl](wtq/transfer/dater_compresstab_qwen3_8b.jsonl) | 4344 | accuracy: 51.91 |
| [wtq/transfer/chain_of_table_qwen3_8b.jsonl](wtq/transfer/chain_of_table_qwen3_8b.jsonl) | 4344 | accuracy: 50.18 |
| [wtq/transfer/chain_of_table_compresstab_qwen3_8b.jsonl](wtq/transfer/chain_of_table_compresstab_qwen3_8b.jsonl) | 4344 | accuracy: 53.41 |
| [wtq/transfer/table_critic_qwen3_8b.jsonl](wtq/transfer/table_critic_qwen3_8b.jsonl) | 4344 | accuracy: 57.18 |
| [wtq/transfer/table_critic_compresstab_qwen3_8b.jsonl](wtq/transfer/table_critic_compresstab_qwen3_8b.jsonl) | 4344 | accuracy: 58.13 |
| [wtq/transfer/tablezoomer_qwen3_8b.jsonl](wtq/transfer/tablezoomer_qwen3_8b.jsonl) | 4344 | accuracy: 47.38 |
| [wtq/transfer/tablezoomer_compresstab_qwen3_8b.jsonl](wtq/transfer/tablezoomer_compresstab_qwen3_8b.jsonl) | 4344 | accuracy: 49.24 |
| [wtq/ablation/without_planning_deepseek_v3_2.jsonl](wtq/ablation/without_planning_deepseek_v3_2.jsonl) | 4344 | accuracy: 80.82 |
| [wtq/ablation/without_coding_deepseek_v3_2.jsonl](wtq/ablation/without_coding_deepseek_v3_2.jsonl) | 4344 | accuracy: 80.73 |
| [wtq/ablation/without_refinement_deepseek_v3_2.jsonl](wtq/ablation/without_refinement_deepseek_v3_2.jsonl) | 4344 | accuracy: 83.93 |
| [longtablebench/main/vanilla_deepseek_v3_2.jsonl](longtablebench/main/vanilla_deepseek_v3_2.jsonl) | 12397 | task_macro_f1: 64.81 |
| [longtablebench/main/compresstab_deepseek_v3_2.jsonl](longtablebench/main/compresstab_deepseek_v3_2.jsonl) | 12397 | task_macro_f1: 71.32 |
| [longtablebench/evidence/raw_qwen3_8b.jsonl](longtablebench/evidence/raw_qwen3_8b.jsonl) | 545 | task_macro_f1: 30.38 |
| [longtablebench/evidence/schema_samples_qwen3_8b.jsonl](longtablebench/evidence/schema_samples_qwen3_8b.jsonl) | 545 | task_macro_f1: 19.10 |
| [longtablebench/evidence/bm25_qwen3_8b.jsonl](longtablebench/evidence/bm25_qwen3_8b.jsonl) | 545 | task_macro_f1: 36.86 |
| [longtablebench/evidence/column_pruning_qwen3_8b.jsonl](longtablebench/evidence/column_pruning_qwen3_8b.jsonl) | 545 | task_macro_f1: 23.97 |
| [longtablebench/evidence/tableprovider_qwen3_8b.jsonl](longtablebench/evidence/tableprovider_qwen3_8b.jsonl) | 545 | task_macro_f1: 38.63 |
| [longtablebench/evidence/tablerag_qwen3_8b.jsonl](longtablebench/evidence/tablerag_qwen3_8b.jsonl) | 545 | task_macro_f1: 27.59 |
| [longtablebench/evidence/dater_qwen3_8b.jsonl](longtablebench/evidence/dater_qwen3_8b.jsonl) | 545 | task_macro_f1: 30.31 |
| [longtablebench/evidence/compresstab_qwen3_8b.jsonl](longtablebench/evidence/compresstab_qwen3_8b.jsonl) | 545 | task_macro_f1: 41.52 |
| [longtablebench/ablation/raw_signature_qwen3_8b.jsonl](longtablebench/ablation/raw_signature_qwen3_8b.jsonl) | 545 | task_macro_f1: 38.11 |
| [longtablebench/ablation/node_local_qwen3_8b.jsonl](longtablebench/ablation/node_local_qwen3_8b.jsonl) | 545 | task_macro_f1: 39.17 |
| [longtablebench/ablation/row_factorization_qwen3_8b.jsonl](longtablebench/ablation/row_factorization_qwen3_8b.jsonl) | 545 | task_macro_f1: 40.67 |
| [longtablebench/ablation/joint_realizations_qwen3_8b.jsonl](longtablebench/ablation/joint_realizations_qwen3_8b.jsonl) | 545 | task_macro_f1: 40.83 |
| [longtablebench/plugin/bm25_coding_qwen3_8b.jsonl](longtablebench/plugin/bm25_coding_qwen3_8b.jsonl) | 545 | task_macro_f1: 38.31 |
| [longtablebench/plugin/column_pruning_coding_qwen3_8b.jsonl](longtablebench/plugin/column_pruning_coding_qwen3_8b.jsonl) | 545 | task_macro_f1: 35.45 |
| [longtablebench/plugin/tableprovider_coding_qwen3_8b.jsonl](longtablebench/plugin/tableprovider_coding_qwen3_8b.jsonl) | 545 | task_macro_f1: 41.10 |
| [longtablebench/plugin/tablerag_coding_qwen3_8b.jsonl](longtablebench/plugin/tablerag_coding_qwen3_8b.jsonl) | 545 | task_macro_f1: 27.59 |
| [longtablebench/plugin/dater_coding_qwen3_8b.jsonl](longtablebench/plugin/dater_coding_qwen3_8b.jsonl) | 545 | task_macro_f1: 27.92 |
| [longtablebench/optimizer/bm25_greedy_qwen3_8b.jsonl](longtablebench/optimizer/bm25_greedy_qwen3_8b.jsonl) | 545 | task_macro_f1: 38.09 |
| [longtablebench/optimizer/column_pruning_greedy_qwen3_8b.jsonl](longtablebench/optimizer/column_pruning_greedy_qwen3_8b.jsonl) | 545 | task_macro_f1: 34.81 |
| [longtablebench/optimizer/tableprovider_greedy_qwen3_8b.jsonl](longtablebench/optimizer/tableprovider_greedy_qwen3_8b.jsonl) | 545 | task_macro_f1: 39.61 |
| [longtablebench/optimizer/tablerag_greedy_qwen3_8b.jsonl](longtablebench/optimizer/tablerag_greedy_qwen3_8b.jsonl) | 545 | task_macro_f1: 27.84 |
| [longtablebench/optimizer/dater_greedy_qwen3_8b.jsonl](longtablebench/optimizer/dater_greedy_qwen3_8b.jsonl) | 545 | task_macro_f1: 29.55 |
| [longtablebench/optimizer/bm25_dp_qwen3_8b.jsonl](longtablebench/optimizer/bm25_dp_qwen3_8b.jsonl) | 545 | task_macro_f1: 38.09 |
| [longtablebench/optimizer/column_pruning_dp_qwen3_8b.jsonl](longtablebench/optimizer/column_pruning_dp_qwen3_8b.jsonl) | 545 | task_macro_f1: 35.62 |
| [longtablebench/optimizer/tableprovider_dp_qwen3_8b.jsonl](longtablebench/optimizer/tableprovider_dp_qwen3_8b.jsonl) | 545 | task_macro_f1: 39.65 |
| [longtablebench/optimizer/tablerag_dp_qwen3_8b.jsonl](longtablebench/optimizer/tablerag_dp_qwen3_8b.jsonl) | 545 | task_macro_f1: 27.84 |
| [longtablebench/optimizer/dater_dp_qwen3_8b.jsonl](longtablebench/optimizer/dater_dp_qwen3_8b.jsonl) | 545 | task_macro_f1: 29.55 |

## Evaluation

Use `scripts/evaluate.py` from the CompressTab library with the official
benchmark references. WikiTQ and HiTab use accuracy; TAT-QA uses EM and F1.
LongTableBench uses task-macro F1, averaging rounds within each conversation
and conversations within each task before averaging tasks.

LongTableBench `main/` covers the full released evaluation across seven
table formats. Its other folders use the 545-round Markdown evaluation
specified by `data/manifests/longtablebench_sample.json` in the library.

The WikiTQ direct-answer Qwen3-8B file in `transfer/` also supplies the
49.24 baseline in the main comparison. Full-method ablation rows use the
corresponding main or evidence-comparison prediction file.

For optimizer comparisons, average the five method-level task-macro F1
scores for each strategy. These files contain answer predictions only;
evidence size and inference cost require their respective measurements.
