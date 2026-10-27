"""Recompute all recorded statistics from numerical observations, without an API or GPU."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics import _summary, compare


def main():
    result_dir = ROOT / 'results/final'
    paired = {'baseline': {}, 'candidate': {}}
    with (result_dir / 'observations.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            observations = paired[record['backend']]
            if record['id'] in observations:
                raise ValueError('Duplicate observation')
            observations[record['id']] = record['observation']
    if list(paired['baseline']) != list(paired['candidate']):
        raise ValueError('Pair identifiers or their order differ')
    rows = {name: list(values.values()) for name, values in paired.items()}
    recomputed = compare(rows['baseline'], rows['candidate'], evidence_kind='heldout')
    recomputed['by_type'] = {
        kind: {name: _summary([row for row in values if row['kind'] == kind])
               for name, values in rows.items()}
        for kind in sorted({row['kind'] for row in rows['baseline']})
    }
    expected = json.loads((result_dir / 'summary.json').read_text())
    if recomputed != expected:
        raise ValueError('Recomputed statistics differ from the published summary')
    print(json.dumps({'statistics_match': True, 'pairs': len(rows['baseline']),
                      'recorded_verdict': expected['verdict']}))


if __name__ == '__main__':
    main()
