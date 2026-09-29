# Threshold sensitivity field check

> Updated 2026-09-09 with recovered session-history generator evidence and an independent raw-archive rerun.

Status: **VERIFIED** for the field, algorithm, and reported threshold counts.

## Historical generator

Session `019f2703-a6dc-76b2-ada2-dccb85045e27`, timestamp `2026-07-04T10:35:07.102Z`, JSONL line 3274 (ordinal 3273), contains the corrected inline Python generator. The relevant source is preserved in `session_evidence/historical_generator_excerpt.md`; its paired output is line 3275.

The exact metric mapping is:

```python
metrics=[('family','family_hash'),('descriptor','descriptor_key'),('graph','graph_hash')]
```

For graph, `key_for()` first reads `graph_hash`; only if missing does it try `signature` and then `actual_structure_signature`. The independent coverage audit found `graph_hash` on all 8,030 formal-success rows, with zero uses of either fallback. Therefore this result is directly an **Exact Forward-Graph (`graph_hash`)** statistic, not Family+Block.

## Exact method and rerun

The analysis reads 12 runs × 800 attempted rows and partitions each run into eight **non-overlapping** 100-attempt windows. In each window it filters `formal_success_candidate`, drops missing values per metric, requires at least 20 valid signatures for that metric-window, computes dominant-count/valid-signature-N, and stores the first window end reaching each threshold.

Independent rerun over the delivered ZIP processed 9,600 rows and 8,030 formal-success rows. `session_evidence/threshold_sensitivity_recomputed.csv` matches the historical condition/seed/role/row-count/onset table; `threshold_field_coverage.json` records field coverage.

| Threshold | Runs whose Exact Forward-Graph top-1 first reached threshold |
|---:|---:|
| 70% | 4/12 |
| 80% | 3/12 |
| 90% | 2/12 |

This is an onset-in-any-eligible-window result. It does **not** mean continuous or persistent collapse. Four-pattern seed114 has only four formal-success rows, so it has zero eligible graph windows; it remains included in the historical 12-run denominator. Thus the safe detail is: 11 runs had at least one eligible window, while the historical cohort-level counts remain 4/12, 3/12, and 2/12.

## Terminology

- `actual_structure_signature`: **Family+Block Eff./Top-1**
- `graph_hash`: **Exact Forward-Graph Eff./Top-1**

Safe statement: **“Exact forward-graph concentration reached the 70%, 80%, and 90% top-1 thresholds in at least one eligible non-overlapping 100-attempt window in 4/12, 3/12, and 2/12 runs, respectively.”**

## Provenance limitation

The historical source is an inline Python tool call, not a recovered standalone script file. The earlier local/Julia2 filename/content search did not find that session source and is retained in `remote_evidence/threshold/` as historical search evidence. The recovered source plus independent raw rerun resolves the field/count question without inventing a standalone file path.
