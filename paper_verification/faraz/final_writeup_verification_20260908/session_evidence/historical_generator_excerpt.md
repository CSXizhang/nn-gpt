# Historical generator evidence

Source session: `019f2703-a6dc-76b2-ada2-dccb85045e27`

Source JSONL: `/Users/zhangxi/.codex/sessions/2026/07/03/rollout-2026-07-03T16-06-18-019f2703-a6dc-76b2-ada2-dccb85045e27.jsonl`

The source below is reconstructed verbatim from the relevant parts of the `exec_command` tool call at JSONL line 3274, ordinal 3273, timestamp `2026-07-04T10:35:07.102Z`. Paths and unrelated report-writing code are omitted. This is the corrected second computation, not the superseded first pass at line 3258.

```python
thresholds=[0.70,0.80,0.90]
metrics=[('family','family_hash'),('descriptor','descriptor_key'),('graph','graph_hash')]

def api(row):
    return row.get('api_result') if isinstance(row.get('api_result'),dict) else {}

def formal_ok(a):
    return bool(a.get('formal_success_candidate'))

def key_for(a, field):
    v=a.get(field)
    if field=='graph_hash' and not v:
        v=a.get('signature') or a.get('actual_structure_signature')
    if field=='descriptor_key' and not v:
        v=a.get('actual_structure_signature')
    if field=='family_hash' and not v:
        v=a.get('family_id') or a.get('family_expr')
    if v is None: return ''
    return str(v)

for start in range(0, len(rows), 100):
    end=min(start+100, len(rows))
    chunk=rows[start:end]
    success=[api(r) for r in chunk if formal_ok(api(r))]
    for metric_name, field in metrics:
        vals=[key_for(a, field) for a in success]
        vals=[v for v in vals if v]
        share, dom, denom=top1_share(vals)
        if denom >= 20 and share is not None:
            for thr in thresholds:
                if per_metric_onsets[(metric_name,thr)]=='never' and share >= thr:
                    per_metric_onsets[(metric_name,thr)]=str(end)
```

The paired tool output is at JSONL line 3275, ordinal 3274, timestamp `2026-07-04T10:35:08.722Z`. It reports 12 runs and these aggregate graph-threshold results:

```text
graph_top1 70%: 4/12
graph_top1 80%: 3/12
graph_top1 90%: 2/12
```

The first pass at JSONL line 3258 read an existing `window_metrics.csv` and did not enforce the minimum valid-signature denominator. It was immediately replaced by the direct-JSONL computation above, which filters `formal_success_candidate` and requires a per-metric signature denominator of at least 20. The first pass is therefore provenance for the correction, not the final method.
