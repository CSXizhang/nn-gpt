# Split provenance and structure-signature confirmation

## Split provenance

The later canonical split-name commit `f9485dc4e7` is not an ancestor of the twelve CIFAR-10 runs, but this is only a history/naming distinction: all archived runtime configs used the same corrected `trainvaltest` behavior, with CIFAR-10 train[45k], reward-eval train[5k], and held-out test[10k].

The six-seed source commit `c91714db` already contains the semantic split fix `6186ebe2c1` and the smoke-weight fix `85c3b6ed10`.

## Two distinct structure metrics

Faraz's reading is correct. The two identifiers are different and must not be called by the same unqualified “graph” label.

| Current use | Identifier | Meaning | Recommended label |
|---|---|---|---|
| manuscript Graph Eff./Graph Top-1 tables | `actual_structure_signature` | family-plus-block composite signature | Family+Block Eff./Top-1 |
| granularity sweep exact_graph rows | `graph_hash` | AST-canonicalized forward-graph hash | Exact Forward-Graph Eff./Top-1 |

They are intentionally not interchangeable. The seed777 discrepancy (74.11 effective exact forward graphs versus 24.51 effective family-plus-block signatures) is therefore expected rather than a recomputation error.
