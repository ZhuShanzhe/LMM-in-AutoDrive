"""Audit physical speed; action-label switches are not acceptance criteria."""

import argparse
import json
from pathlib import Path

import numpy as np


def assess_speed(speeds, reference_kmh, *, eligible, tolerance_kmh=3., required_fraction=.95):
    values = np.asarray(speeds, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite speed observations')
    if not np.isfinite(reference_kmh):
        raise ValueError('Expected a finite independently specified reference')
    fraction = float(np.mean(np.abs(values-reference_kmh) <= tolerance_kmh + 1e-9))
    return dict(reference_speed_kmh=reference_kmh, tolerance_kmh=tolerance_kmh,
                required_fraction=required_fraction, within_band_fraction=fraction,
                median_kmh=float(np.median(values)), p5_p95_kmh=np.percentile(values,[5,95]).tolist(),
                status=('pass' if fraction >= required_fraction else 'fail') if eligible else 'not_applicable')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.run/'report.json').read_text())
    cases = []
    for case in original['cases']:
        rows = [json.loads(x) for x in (args.run/(case['case']+'_truth.jsonl')).read_text().splitlines()]
        if len(rows) != 300:
            raise ValueError('This audit expects the existing 300-step, 0.05-second test recipe')
        speeds = [r['speed_kmh'] for r in rows[200:300]]
        cruise = case['mode'] == 'clear_road'
        result = assess_speed(speeds, 30., eligible=cruise)
        if not cruise:
            result['reference_speed_kmh'] = None
            result['within_band_fraction'] = None
        result.update(case=case['case'], evaluation_window_s=[10.,15.],
                      criterion_scope='unobstructed cruise' if cruise else 'transient braking, stopping or restarting; no annotated steady following interval',
                      action_switches_diagnostic_only=case['action_switches'],
                      overall_following_acceptance='not_assessed')
        cases.append(result)
    report = dict(schema_version='physical_speed_stability_audit/1.0',
                  criterion='Within independently specified reference speed +/-3 km/h in at least 95% of eligible frames',
                  notes=['User-selected engineering criterion, not an official competition threshold.',
                         'Action switches do not fail this speed criterion.',
                         'Speed stability does not override collision, unsafe following, or failure to follow a moving lead.',
                         'The final five-second window is fixed before inspecting results, not selected for a favorable score.',
                         'These short adverse tests do not provide an annotated sustained constant-speed following interval.'],
                  cases=cases)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
