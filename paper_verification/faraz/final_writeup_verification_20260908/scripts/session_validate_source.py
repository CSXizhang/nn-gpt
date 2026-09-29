"""Verify recovered generator/output against the original session events."""
import hashlib
import itertools
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'session_evidence_manifest.json').read_text())
session = Path(manifest['session_path'])
with session.open() as stream:
    events = [json.loads(line) for line in itertools.islice(stream, 3273, 3275)]
call, output = [event['payload'] for event in events]
assert call['call_id'] == output['call_id']
arguments = json.loads(call['arguments'])
source = (root / 'session_evidence/historical_corrected_generator.py').read_text().strip()
assert source in arguments['cmd'], 'Recovered source differs from original call'
with (root / 'session_evidence/historical_paired_tool_output.txt').open(newline='') as stream:
    saved_output = stream.read().strip()
assert saved_output == output['output'].strip(), 'Recovered output differs from original event'
with (root / 'session_evidence/historical_threshold_output.csv').open(newline='') as stream:
    historical_csv = stream.read().strip()
# CSV extraction normalizes CRLF to LF; the paired output above preserves it.
assert historical_csv.replace('\r\n', '\n') in output['output'].replace('\r\n', '\n'), 'Historical CSV differs beyond line endings'
digest = hashlib.sha256()
with session.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
assert digest.hexdigest() == manifest['session_sha256']
print('PASS: original session hash, paired call IDs, verbatim generator, and verbatim tool output.')
