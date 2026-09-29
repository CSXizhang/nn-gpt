"""Offline acceptance of recovered remote evidence; never contacts the server."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'remote_evidence_manifest.json').read_text())
for entry in manifest['files']:
    path = root / entry['local_path']
    content = path.read_bytes()
    assert len(content) == entry['size'], path
    assert hashlib.sha256(content).hexdigest() == entry['sha256'], path
states = list((root / 'remote_evidence/provenance').rglob('run_state.json'))
assert len(states) == 6
for path in states:
    data = json.loads(path.read_text())
    assert data['commit_hash'] == 'c91714dbe7dad1d02a9080243945bbf8e8ec9300', path
raw = root / 'remote_evidence/imagenette/generation_samples.jsonl'
with raw.open() as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
assert len(rows) == 1000
assert hashlib.sha256(raw.read_bytes()).hexdigest() == '9e43736e9a4b1a69c3047d5a620a7ade743a7c9800520ee4fab9127c4dbbe2fc'
sacct = (root / 'remote_evidence/imagenette_and_nndataset_raw.txt').read_text()
assert '2666209|2666209|tunerl-dscoder_imagenette_h100|COMPLETED|' in sacct
gpu = json.loads((root / 'remote_evidence/wide38/gpu_task0_recomputed.json').read_text())
assert gpu['task0']['job_id'] == '2968490_0'
assert gpu['extended_unique_total']['rounded_2dp'] == '1886.45'
print(f'PASS: {len(manifest["files"])} recovered-file hashes; six base-commit records; 1000-row Imagenette evidence; completed Slurm job; corrected GPU accounting.')
