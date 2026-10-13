"""Independent small-space oracles for dependency-aware plan selection."""

from itertools import combinations
import os

import pytest

from compresstab.codec.evidence import EncodingCandidate, EvidencePlanOptimizer
from compresstab.table.artifact import artifact_from_dataset_row
from compresstab.table.store import TableStore
from compresstab.evidence import DerivedObservable
from compresstab.tokenization import ExactTokenCounter


def candidate(node, operator, text, *, covers=(), dependencies=(), provides=()):
    return EncodingCandidate(
        col_id=node, source_header=node, display_header=node, operator=operator,
        values={"r0": text}, token_cost=0, lexical_cost=0, byte_cost=0,
        certified=True, covered_node_ids=covers or (node,),
        dependencies=dependencies, provided_properties=frozenset(provides),
    )


def optimizer(counter):
    store = TableStore(artifact_from_dataset_row({"table": [["a", "b"], ["1", "2"]]}))
    opt = EvidencePlanOptimizer(store, token_counter=counter, factor_row_addresses=False)
    table = store.table("t0")
    opt._search_render_context = (table, ["d0", "c0", "c1"], ["r0"])
    opt._rendering_frontiers = {}
    return opt, table


def price(opt, table, choices):
    return opt._rendered_body_token_cost(
        opt._render_selected(table, ["d0", "c0", "c1"], ["r0"], list(choices)))


def search(opt, table, **kwargs):
    return opt._minimum_cost_dependency_plan(
        table=table, row_ids=["r0"], source_order=["c0", "c1"], profiles={},
        source_requirements={}, source_groups=[["c0", "c1"]], fixed_cost=(0, 0),
        observables=kwargs.get("observables", []),
        derived_candidates=kwargs.get("derived_candidates", {}),
    )


def test_nonadditive_cost_does_not_prune_longer_prefix(monkeypatch):
    # Deliberately nonadditive to test the optimizer independently of one vocabulary.
    def count(text):
        if "BBBB" in text and "SUFFIX" in text:
            return 1
        return len(text)

    opt, table = optimizer(count)
    a = candidate("c0", "short", "A")
    b = candidate("c0", "long", "BBBB")
    suffix = candidate("c1", "suffix", "SUFFIX")
    assert price(opt, table, [a]) < price(opt, table, [b])
    monkeypatch.setattr(opt, "_source_candidate_space", lambda **_: (["c0", "c1"], [a, b, suffix]))
    selected, _, alternatives, _ = search(opt, table)
    assert selected == [b, suffix]
    assert len(alternatives) == 2


@pytest.mark.parametrize("seed", range(12))
def test_joint_cover_matches_independent_subset_enumeration(monkeypatch, seed):
    def count(text):
        return len(text.replace("X | Y", "Z")) + (seed * len(text) % 7)

    opt, table = optimizer(count)
    pool = [candidate("c0", "a", "X"), candidate("c0", "b", "A" * (seed + 1)),
            candidate("c1", "c", "Y"), candidate("c1", "d", "B" * (13 - seed)),
            candidate("c0", "joint", "JOINT" * (1 + seed % 3), covers=("c0", "c1"))]
    monkeypatch.setattr(opt, "_source_candidate_space", lambda **_: (["c0", "c1"], pool))
    valid = []
    for size in range(1, len(pool) + 1):
        for subset in combinations(pool, size):
            covered = [node for item in subset for node in item.covered_node_ids]
            if sorted(covered) == ["c0", "c1"]:
                valid.append(price(opt, table, subset))
    selected, _, alternatives, _ = search(opt, table)
    assert len(alternatives) == len(valid)
    assert price(opt, table, selected) == min(valid)


def test_derived_choice_propagates_operand_requirements(monkeypatch):
    opt, table = optimizer(len)
    materialized = candidate("d0", "materialize", "M" * 200, provides=("derived_exact",))
    derive = candidate("d0", "derive", "a+b", provides=("derived_exact",),
                       dependencies=(("c0", "exact"), ("c1", "exact")))
    a, b = candidate("c0", "raw", "1"), candidate("c1", "raw", "2")
    seen = []
    def source_space(**kwargs):
        seen.append(kwargs["obligations"])
        return (["c0", "c1"], [a, b]) if kwargs["obligations"] else ([], [])
    monkeypatch.setattr(opt, "_source_candidate_space", source_space)
    observable = DerivedObservable.model_validate({"observable_id": "d0", "operator": "sum",
        "mode": "exact", "operands": [{"column": "c0", "role": "value"}, {"column": "c1", "role": "value"}]})
    selected, _, _, _ = search(opt, table, observables=[observable],
                              derived_candidates={"d0": [materialized, derive]})
    assert {item.operator for item in selected} == {"derive", "raw"}
    assert {"c0": {"exact"}, "c1": {"exact"}} in seen
    assert price(opt, table, selected) == min(price(opt, table, [materialized]),
                                            price(opt, table, [derive, a, b]))


@pytest.mark.skipif(not os.environ.get("COMPRESSTAB_TEST_TOKENIZER"), reason="Set a local tokenizer path")
@pytest.mark.parametrize("text", ['a|b', '"quoted"', '', '\u4e2d\u6587', 'a\nb'])
def test_real_tokenizer_complete_plan_oracle(monkeypatch, text):
    counter = ExactTokenCounter(os.environ["COMPRESSTAB_TEST_TOKENIZER"])
    opt, table = optimizer(counter.count)
    a = candidate("c0", "raw", text)
    b = candidate("c1", "raw", "15.25%")
    joint = candidate("c0", "joint", text + " 15.25%", covers=("c0", "c1"))
    monkeypatch.setattr(opt, "_source_candidate_space", lambda **_: (["c0", "c1"], [a, b, joint]))
    selected, _, _, _ = search(opt, table)
    assert price(opt, table, selected) == min(price(opt, table, [a, b]), price(opt, table, [joint]))
