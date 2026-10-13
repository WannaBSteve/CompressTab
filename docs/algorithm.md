# Evidence Compilation

## Objects

The Answerability Signature specifies source scope, structural links, and value
distinctions. Grounding resolves these requirements to table rows, columns,
hierarchy paths, and derived quantities. A derived quantity retains the rows
and operands from which it was computed.

A candidate is one representation of one or more requirements. Candidates may
keep exact values, share repeated text, represent an ordering including ties,
materialize a derived quantity, or expose an expression together with its
operands. They are not precomputed complete answers.

| Family | Representations | Preserved information |
| --- | --- | --- |
| Exact | Raw values, dictionary, repeated prefix/suffix, default plus exceptions, exact numeric differences | Recoverable source values |
| Question-specific | Equality classes, anchored values, dense ranks, threshold membership | The distinction required by the signature |
| Derived | Exact derived values, derived ranks, derived membership, expression with operands | Required property of a row-bound expression |
| Shared | Joint source representations and factored row addresses | Required values and their source bindings |

Each candidate is checked against the properties it claims to supply. These
checks establish preservation of the given requirements. Signature inference
and final answer correctness are handled by the planning and auditing stages.

## Layered Search

The search first visits derived quantities from consumers to operands. Choosing
an expression adds requirements on its operands; choosing its materialized
values may remove the need to render those operands. It then covers outstanding
source requirements, allowing a joint representation to cover multiple nodes.

A search state records:

- The current layer and unresolved requirements, or covered source nodes.
- The shared rendering structures already in use.
- The ordered candidate content needed to determine subsequent rendering.

The last component is important for actual tokenizer costs. Two different
prefixes can have the same requirements and headings but tokenize differently
after another column is added. The implementation therefore merges only
identical continuation states, not different texts with similar lengths.

For a partial plan `P` and a legal candidate `c`, the transition is `P -> P+c`.
Its cost is measured by rendering the new partial plan. No dominance rule based
on summed isolated token counts deletes a distinct rendering. Complete
alternatives from every table are assembled with their context, provenance,
and decoding instructions. The selected plan minimizes the tokenizer count of
that complete text.

## Why the Search Is Exact

Fix the grounded signature, generated candidates, and deterministic renderer.
Every transition either resolves a derived node or covers an uncovered source
node. Dependency order is acyclic, so the process terminates. Source transitions
do not overlap already covered nodes. Selecting a candidate that covers the
first uncovered node enumerates every legal exact cover without relying on a
local cost ordering.

Identical continuation states admit the same legal suffixes and produce the
same final rendering for any such suffix. Merging them is safe. All other
prefixes remain available. Since every complete alternative is tokenized,
selecting the minimum gives the shortest complete rendering in this candidate
space. Token increments need not be additive or nonnegative.

This is layered dynamic programming with conservative state equivalence. It
does not imply a polynomial-time bound: retaining rendering context can leave
exponentially many states. If `S_i` states are retained in layer `i`, at most `K`
transitions leave a state, and rendering plus tokenization costs `R`, transition
work is bounded by `O(K R sum_i S_i)`, before duplicate-key and candidate-generation
costs. Storage includes the retained candidate sequences. Composing table-local
alternatives can additionally require their Cartesian product.

The implementation records `dp_layers`, `candidate_count`, `visited_states`,
`widest_layer_states`, and `global_plan_frontier_size`. The source-cover entry
in `dp_layers` reports its completed frontier rather than every intermediate
coverage expansion; consequently `visited_states` is a recorded-layer summary,
not a count of all transition attempts. Use these diagnostics with measured
runtime when describing search scale.

## Token Target

The current objective is minimum complete evidence, not arbitrary truncation to
a hard capacity. `evidence_target` records a requested operating target, helps
bound the planning sketch, and is reported alongside the realized evidence
length. When the shortest generated complete representation exceeds the target,
the compiler keeps that evidence and reports `tokens_over_target`; it does not
return empty evidence. A strict downstream context limit must be provisioned
separately. This release does not claim that every question is answerable under
every token budget.

## Refinement

The Evidence Auditor checks scope, bindings, value interpretation, computation,
and answer form. A reasoning correction keeps the evidence. A missing-evidence
correction inspects the source, extends the signature, and recompiles. The same
Evidence Auditor checks the proposed answer before it is committed. Rejected
proposals do not replace the incumbent. The loop retains its previous checks
and stops after the configured number of correction rounds.

Source-shape and operand checks are deterministic. Lexical overlap supplies
inspection hints, not a proof that an answer or predicate must change. Answer
format checks preserve source labels and positional multiplicity. The reader
handles semantic corrections from the evidence.

`ACCEPTED` means the auditor accepted the answer. At the limit, `UNRESOLVED`
retains the incumbent answer and evidence for ordinary downstream scoring.
Neither status is a ground-truth correctness label.
An invalid or unsupported audit is `INCONCLUSIVE`: it neither accepts a proposal
nor changes the incumbent answer.
