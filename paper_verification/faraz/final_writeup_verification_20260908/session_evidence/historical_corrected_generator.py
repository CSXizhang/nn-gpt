import csv, json
from pathlib import Path
from collections import Counter, defaultdict
root=Path('/Users/zhangxi/Desktop/SFT_RL/paper_used_data_check_20260704')
index=root/'01_processed_results/handover_processed_20260703/six_seed_robustness_current.csv'
outcsv=root/'01_processed_results/article_diagnostics_20260627/threshold_sensitivity_70_80_90.csv'
outmd=root/'01_processed_results/article_diagnostics_20260627/threshold_sensitivity_70_80_90.md'
thresholds=[0.70,0.80,0.90]
metrics=[('family','family_hash'),('descriptor','descriptor_key'),('graph','graph_hash')]
run_rows=[]
with index.open(newline='') as f:
    for r in csv.DictReader(f):
        run_rows.append(r)

def local_jsonl(run_root, source_file):
    run_id=Path(run_root).name
    return root/'02_raw_data/cluster_core/parallel_runs'/run_id/source_file

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

def top1_share(items):
    if not items: return None, '', 0
    c=Counter(items)
    k,n=c.most_common(1)[0]
    return n/len(items), k, len(items)

records=[]
window_records=[]
for rr in run_rows:
    p=local_jsonl(rr['raw_run_root'], rr['raw_source_file'])
    rows=[]
    if p.exists():
        with p.open() as f:
            for line in f:
                try: rows.append(json.loads(line))
                except Exception: pass
    rec={'condition':rr['condition'], 'seed':rr['seed'], 'role':rr['role'], 'run_id':Path(rr['raw_run_root']).name, 'rows':len(rows)}
    per_metric_onsets={}
    for metric_name, field in metrics:
        for thr in thresholds:
            per_metric_onsets[(metric_name,thr)]='never'
    for start in range(0, len(rows), 100):
        end=min(start+100, len(rows))
        chunk=rows[start:end]
        if not chunk: continue
        base={'condition':rr['condition'], 'seed':rr['seed'], 'role':rr['role'], 'run_id':Path(rr['raw_run_root']).name, 'window_start':start+1, 'window_end':end, 'n':len(chunk)}
        success=[api(r) for r in chunk if formal_ok(api(r))]
        base['formal_success_n']=len(success)
        for metric_name, field in metrics:
            vals=[key_for(a, field) for a in success]
            vals=[v for v in vals if v]
            share, dom, denom=top1_share(vals)
            base[f'{metric_name}_top1_share']=share if share is not None else ''
            base[f'{metric_name}_dominant']=dom
            base[f'{metric_name}_denom']=denom
            if denom >= 20 and share is not None:
                for thr in thresholds:
                    if per_metric_onsets[(metric_name,thr)]=='never' and share >= thr:
                        per_metric_onsets[(metric_name,thr)]=str(end)
        window_records.append(base)
    for metric_name,_ in metrics:
        for thr in thresholds:
            rec[f'{metric_name}_top1_ge_{int(thr*100)}_onset']=per_metric_onsets[(metric_name,thr)]
    records.append(rec)
fieldnames=['condition','seed','role','run_id','rows']+[f'{m}_top1_ge_{int(t*100)}_onset' for m,_ in metrics for t in thresholds]
with outcsv.open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(records)
lines=[]
lines.append('# Threshold sensitivity 70/80/90\n\n')
lines.append('Input: `six_seed_robustness_current.csv` and each listed raw `generation_samples.jsonl`. Onset is earliest 100-sample `window_end` where top-1 share among formal-success samples reaches the threshold. Windows with fewer than 20 formal-success samples are ignored for onset.\n\n')
for metric_name,_ in metrics:
    lines.append(f'## {metric_name}_top1\n\n')
    lines.append('| threshold | reached / runs | onset values |\n')
    lines.append('|---:|---:|---|\n')
    for thr in thresholds:
        col=f'{metric_name}_top1_ge_{int(thr*100)}_onset'
        vals=[r[col] for r in records]
        reached=[v for v in vals if v!='never']
        counts=Counter(vals)
        def sortkey(x): return (x=='never', int(x) if x!='never' else 10**9)
        onset_values=', '.join(f'{k}:{counts[k]}' for k in sorted(counts, key=sortkey))
        lines.append(f'| {int(thr*100)}% | {len(reached)}/{len(vals)} | {onset_values} |\n')
    lines.append('\n')
lines.append('## Per-run table\n\n')
lines.append('| condition | seed | role | rows | family 70 | family 80 | family 90 | descriptor 70 | descriptor 80 | descriptor 90 | graph 70 | graph 80 | graph 90 |\n')
lines.append('|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
for r in records:
    lines.append('| {condition} | {seed} | {role} | {rows} | {family_top1_ge_70_onset} | {family_top1_ge_80_onset} | {family_top1_ge_90_onset} | {descriptor_top1_ge_70_onset} | {descriptor_top1_ge_80_onset} | {descriptor_top1_ge_90_onset} | {graph_top1_ge_70_onset} | {graph_top1_ge_80_onset} | {graph_top1_ge_90_onset} |\n'.format(**r))
outmd.write_text(''.join(lines))
print('wrote', outcsv)
print('wrote', outmd)
print('runs', len(records))
