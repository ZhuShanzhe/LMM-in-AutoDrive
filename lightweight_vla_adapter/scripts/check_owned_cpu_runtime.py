"""Offline checks for owned language/decision models, not driving or J6P acceptance."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(model_root, output):
    import torch
    from lightweight_vla_adapter.scripts.prepare_challenge_runtime import verify_assets, resolve_config
    from lightweight_vla_adapter.scripts.smoke_sequence_runtime import smoke
    from lightweight_vla_adapter.scripts.run_offline_inference import build_model
    from lightweight_vla_adapter.scripts.benchmark_latency import configured_inputs
    from lightweight_vla_adapter.src.inference_optimization import fuse_inference_conv_bn
    from structured_command_parser.src.modernbert_service import ModernBertCommandService
    from structured_command_parser.src.schema_tools import validate_document

    torch.set_num_threads(2)
    torch.manual_seed(925)
    manifest = json.loads((ROOT/'lightweight_vla_adapter/configs/challenge_assets.json').read_text())
    verified = verify_assets(model_root, manifest)
    for town in ['Town01', 'Town04', 'Town05']:
        config = resolve_config(ROOT, model_root, ROOT/f'lightweight_vla_adapter/configs/maps/{town}_traffic_heads.json')
        (output/f'{town}.json').write_text(json.dumps(config, indent=2)+'\n')
    checkpoint = model_root/'challenge/phase_recovery_v1.pt'
    sequence = smoke(checkpoint, 'cpu')
    service = ModernBertCommandService(str(model_root/'modernbert-drive-command-compositional'), device='cpu')
    documents = []
    for index, text in enumerate([
        'Maintain 40 km/h.',
        'Slow down and stop before the red truck.',
        'Do not turn left. Continue straight.',
    ]):
        doc = service.handle_message(dict(text=text, language='en-US', request_id=f'cpu-{index}'))
        validate_document(doc)
        documents.append(doc)
    speed_values = [step.get('parameters', {}).get('target_speed_mps')
                    for step in documents[0]['intent']['steps']]
    if not any(isinstance(v, (int, float)) and abs(v-40/3.6)<1e-3 for v in speed_values):
        raise ValueError('40 km/h target speed was lost at the parser boundary')
    rejected = []
    for message in [dict(text='Turn left.', language='zh-CN'), dict(text='', language='en-US')]:
        try:
            service.handle_message(message)
        except ValueError:
            rejected.append(message)
        else:
            raise ValueError('Invalid language/empty input was not rejected')
    language_parameters = sum(p.numel() for p in service.parser.model.parameters())
    del service
    artifact = torch.load(checkpoint, map_location='cpu', weights_only=True)
    baseline = build_model(artifact['config']).eval()
    baseline.load_state_dict(artifact['base'], strict=True)
    optimized, pairs = fuse_inference_conv_bn(baseline)
    inputs = configured_inputs(artifact['config'], torch.device('cpu'), torch.float32)
    with torch.inference_mode():
        expected, actual = baseline(**inputs), optimized(**inputs)
    differences = {}
    for field in fields(type(expected)):
        before, after = getattr(expected, field.name), getattr(actual, field.name)
        if isinstance(before, torch.Tensor):
            torch.testing.assert_close(after, before, rtol=3e-4, atol=3e-5)
            differences[field.name] = float((after-before).abs().max())
        elif before != after:
            raise ValueError(f'Non-tensor output changed: {field.name}')
    profiles = {}
    for name, model in [('baseline', baseline), ('conv_bn_folded', optimized)]:
        with torch.inference_mode(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
        ) as profiler:
            model(**inputs)
        counted = {event.key: event.flops for event in profiler.key_averages() if event.flops}
        profiles[name] = dict(parameters=sum(p.numel() for p in model.parameters()),
                              profiler_counted_flops=sum(counted.values()), counted_ops=counted)
    return dict(status='passed', verified_asset_count=len(verified), device='cpu',
                scope='Integrity, actual-weight CPU loading, schema/target-speed guards, synthetic sequence and folding parity only',
                not_validated=['driving accuracy', 'full-chain integration', 'ASR', 'J6P latency/power/memory/utilization'],
                language_parameters=language_parameters, parser_documents=documents,
                rejected_inputs=rejected, sequence=sequence,
                folding=dict(fused_pairs=pairs, max_abs_differences=differences, profiles=profiles,
                             default_changed=False,
                             flops_scope='Only operators counted by PyTorch CPU profiler for the configured adapter; excludes uncounted ops, language and memory; not official full-chain FLOPs'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        report = run(args.model_root.resolve(), args.output)
    except Exception as error:
        (args.output/'report.json').write_text(json.dumps(dict(status='failed', error=str(error)), indent=2)+'\n')
        raise
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['parser_documents', 'sequence']}, indent=2))


if __name__ == '__main__':
    main()
