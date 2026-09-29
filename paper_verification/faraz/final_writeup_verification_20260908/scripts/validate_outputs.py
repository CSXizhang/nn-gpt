"""Final artifact acceptance; does not change experiment inputs."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
required = ['AUDIT_REPORT.md', 'FINAL_REPLY_FACTS.md', 'sft_family_pre_post.csv',
            'sft_family_recompute.json', 'reward_components_rebuild.csv',
            'reward_components_rebuild.md', 'gpu_hours_recomputed.json',
            'threshold_field_check.md', 'code_availability_check.md']
for name in required:
    assert (root / name).stat().st_size > 0, name
report = (root / 'AUDIT_REPORT.md').read_text()
for heading in ['Claim being checked', 'Evidence used', 'Recompute method',
                'Exact result', 'Difference from Faraz / previous Xi summary',
                'Safe statement for the paper/email', 'Remaining uncertainty']:
    assert report.count('### ' + heading) == 8, heading
sft = json.loads((root / 'sft_family_recompute.json').read_text())
with (root / 'sft_family_pre_post.csv').open() as f:
    rows = list(csv.DictReader(f))
assert len(rows) == len(sft['runs']) == 12
for row, run in zip(rows, sft['runs']):
    assert row['Condition'] == run['condition']
    assert int(row['Seed']) == run['seed']
    assert int(row['Formal-success N']) == run['formal_success_n']
    assert float(row['Family top-1']) == run['metrics']['family_hash']['top1_share']
    assert float(row['Family eff.']) == run['metrics']['family_hash']['effective_number']
    assert float(row['Family+Block eff.']) == run['metrics']['actual_structure_signature']['effective_number']
    source = run['source']
    if '!' in source:
        archive, member = source.split('!', 1)
        with zipfile.ZipFile(archive) as z:
            content = z.read(member)
    else:
        content = Path(source).read_bytes()
    assert hashlib.sha256(content).hexdigest() == run['source_sha256'], source
with (root / 'reward_components_rebuild.csv').open() as f:
    reward_rows = list(csv.DictReader(f))
assert len(reward_rows) == 12
evidence = json.loads((root / 'reward_audit_evidence.json').read_text())
assert evidence['commit_type'] == 'commit'
assert all(h['formula_matches'] for h in evidence['raw_log_hits'] if h['field'] == 'r_dense')
gpu = json.loads((root / 'gpu_hours_recomputed.json').read_text())
assert gpu['unique_grand_total']['rounded_gpu_hours_2dp'] == '1885.91'
assert gpu['extended']['proxy_wide38_missing_task_ids'] == [0]
secondary = json.loads((root / 'secondary_consistency_recomputed.json').read_text())
for key in ['1pattern', '4pattern', 'cifar100_four_pattern_completed_package']:
    for run in secondary[key]['seeds']:
        assert run['fallback_used_n'] == 0
        assert run['test_acc_numeric_n'] == run['formal_success_n']
print('PASS: required artifacts, 8-item report contract, CSV/JSON agreement, raw input hashes, reward evidence, GPU scope, and accuracy-field coverage.')
