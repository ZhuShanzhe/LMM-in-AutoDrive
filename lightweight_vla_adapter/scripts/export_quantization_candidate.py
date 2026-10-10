"""Export frozen policy and prepare an isolated mixed-precision PTQ config."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def int8_nodes(graph, scope):
    paths = ('/ffn/', '/intent_projection/', '/ego_projection/',
             '/environment_projection/', '/raw_camera_encoder/projection/')
    return {node.name: {'qtype': 'int8'} for node in graph.graph.node
            if (node.op_type in {'MatMul', 'Gemm'} and any(p in node.name for p in paths))
            or (scope == 'bev_and_projections' and node.op_type == 'Conv'
                and '/bev_encoder/' in node.name)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--compile-template', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scope', choices=('projections', 'bev_and_projections'),
                        default='bev_and_projections')
    args = parser.parse_args()
    for path in (args.checkpoint, args.config, args.input, args.compile_template):
        if not path.is_file():
            parser.error('Required file missing: '+str(path))
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import yaml
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from lightweight_vla_adapter.scripts.run_offline_inference import build_model
    from lightweight_vla_adapter.src.deployment_export import (
        FixedShapePolicy, INPUT_NAMES, OUTPUT_NAMES, prepare_fixed_shape_export)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    model = build_model(json.loads(args.config.read_text())).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    with np.load(args.input, allow_pickle=False) as sample:
        arrays = {name: sample[name].copy() for name in INPUT_NAMES}
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite calibration input')
    inputs = tuple(torch.from_numpy(arrays[name]) for name in INPUT_NAMES)
    prepared = FixedShapePolicy(prepare_fixed_shape_export(model)).eval()
    output = args.output.resolve()
    onnx_path = output / 'policy.onnx'
    with torch.inference_mode():
        reference = FixedShapePolicy(model)(*inputs)
        lowered = prepared(*inputs)
        errors = [float((a-b).abs().max()) for a, b in zip(reference, lowered)]
        for a, b in zip(reference, lowered):
            torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
        torch.onnx.export(prepared, inputs, str(onnx_path), input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES), opset_version=18, dynamo=False, do_constant_folding=True)
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    outputs = session.run(None, {node.name: arrays[node.name] for node in session.get_inputs()})
    for a, b in zip(lowered, outputs):
        np.testing.assert_allclose(a.numpy(), b, rtol=3e-4, atol=3e-4)
    config = yaml.safe_load(args.compile_template.read_text())
    if not isinstance(config, dict):
        raise ValueError('Invalid SDK compile template')
    nodes = int8_nodes(graph, args.scope)
    if not nodes:
        raise ValueError('No intended quantization nodes found')
    config['model_parameters'].update(onnx_model=str(onnx_path), working_dir=str(output/'compiled'),
                                     output_model_file_prefix='policy_mixed_int8')
    config['calibration_parameters']['quant_config'] = {
        'model_config': {'all_node_type': 'float32', 'model_output_type': 'float32'}, 'node_config': nodes}
    config['compiler_parameters'].update(jobs=1, optimize_level='O0')
    (output/'compile.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    report = dict(schema_version='frozen_policy_ptq_candidate/1.0',
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        input_shapes={k: list(v.shape) for k, v in arrays.items()},
        active_inputs=[n.name for n in session.get_inputs()], lowering_max_abs_errors=errors,
        int8_nodes=list(nodes), calibration_template=str(args.compile_template.resolve()),
        protected='Raw image convolutions, attention score products, normalization and output heads remain float',
        scope='Fixed-shape export parity and PTQ configuration; not quantized accuracy or closed-loop acceptance',
        compile_template_paths_unchanged=True, quantization_scope=args.scope,
        torch_version=torch.__version__, onnx_version=onnx.__version__)
    (output/'export_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
