"""Compile an existing calibrated BC; estimates are not board measurements."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bc', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--march', default='nash-p')
    parser.add_argument('--opt', type=int, choices=(0, 1, 2), default=2)
    parser.add_argument('--balance', type=int, choices=range(101), default=2)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if not args.bc.is_file() or args.jobs < 1:
        parser.error('An existing BC and positive job count are required')
    args.output.mkdir(parents=True, exist_ok=False)
    from hbdk4.compiler import compile, hbm_perf, load
    source = args.bc.resolve()
    options = dict(march=args.march, opt=args.opt, jobs=args.jobs,
                   balance=args.balance, max_l2m_size=None, debug=True, progress_bar=True)
    report = dict(schema_version='horizon_compile_candidate/1.0',
        scope='Compiler schedule/memory estimates only; not actual board latency or utilization',
        source_bc=str(source), source_bc_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        options=options, closed_loop_accepted=False, board_tested=False)
    manifest = args.output / 'compile_manifest.json'
    manifest.write_text(json.dumps(report, indent=2)+'\n')
    hbm = args.output / 'policy.hbm'
    compile(load(str(source)), str(hbm), **options)
    hbm_perf(str(hbm), output_dir=str(args.output))
    report.update(hbm_sha256=hashlib.sha256(hbm.read_bytes()).hexdigest(),
                  hbm_bytes=hbm.stat().st_size, compilation_completed=True)
    manifest.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
