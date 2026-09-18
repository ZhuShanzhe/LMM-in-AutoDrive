"""Data-free layered runtime checks; synthetic input is not driving evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.src.event_observation import OBSERVATION_VERSION
from lightweight_vla_adapter.src.behavior_memory import SUMMARY_DIM, SUMMARY_SLOTS
from lightweight_vla_adapter.src.layered_context import (
    LayeredSequenceHead, LAYERED_SCHEMA, DIRECT_LAYERED_SCHEMA, CONTEXT_VERSION,
)
from lightweight_vla_adapter.src.sequence_policy import SequenceMemoryRuntime, SEQUENCE_SCHEMA


def smoke(checkpoint=None, device='cpu'):
    torch.manual_seed(911)
    torch.set_num_threads(2)
    config = json.loads((ROOT/'lightweight_vla_adapter/configs/challenge_signal_generalization.json').read_text())
    artifact = None
    digest = None
    direct = True
    if checkpoint:
        with open(checkpoint, 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        artifact = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (artifact.get('schema_version') not in (LAYERED_SCHEMA, DIRECT_LAYERED_SCHEMA)
                or artifact.get('stage') != 'carla'
                or artifact.get('sequence_schema') != SEQUENCE_SCHEMA
                or artifact.get('observation_version') != OBSERVATION_VERSION
                or artifact.get('layered_context_version') != CONTEXT_VERSION
                or artifact.get('smoke_test', False)):
            raise ValueError('Checkpoint contract mismatch')
        direct = artifact['schema_version'] == DIRECT_LAYERED_SCHEMA
        config = artifact['config']
    head = LayeredSequenceHead(direct_sequence=direct).to(device).eval()
    base = build_model(config).to(device).eval()
    if artifact is not None:
        head.load_state_dict(artifact['head'], strict=True)
        base.load_state_dict(artifact['base'], strict=True)
    zero = lambda *shape: torch.zeros(*shape, device=device)
    inputs = dict(camera_bev=zero(1,8,64,64), lidar_bev=zero(1,4,64,64),
                  ego_features=zero(1,8), environment_features=zero(1,14),
                  candidate_features=zero(1,32,12), candidate_mask=zero(1,32).bool(),
                  intent_tokens=zero(1,32,768), intent_mask=torch.ones(1,32,dtype=torch.bool,device=device),
                  camera_images=zero(1,4,3,224,224),
                  camera_view_mask=torch.ones(1,4,dtype=torch.bool,device=device))
    memory = zero(1,80,52)
    memory[:,-1,40] = 5./40.
    memory[:,-1,42] = .3
    behavior_valid = zero(1,SUMMARY_SLOTS).bool()
    behavior_valid[:,-1] = True
    batch = dict(event_memory=memory, event_memory_valid=torch.ones(1,80,dtype=torch.bool,device=device),
                 behavior_memory=zero(1,SUMMARY_SLOTS,SUMMARY_DIM), behavior_memory_valid=behavior_valid,
                 recursive_state=zero(1,6,32), layered_context=zero(1,48))
    runtime = SequenceMemoryRuntime(head)
    with torch.inference_mode():
        original = base(**inputs)
        result = runtime(original, batch, longitudinal_authorized=torch.tensor([True],device=device))
        seq = runtime.diagnostics['longitudinal_sequence']
        unchanged = runtime(original, batch, longitudinal_authorized=torch.tensor([False],device=device))
    assert unchanged is original
    assert len(seq['speed_mps']) == len(seq['acceleration_mps2']) == 30
    speed = torch.tensor(seq['speed_mps'])
    acceleration = torch.tensor(seq['acceleration_mps2'])
    assert torch.isfinite(speed).all() and torch.isfinite(acceleration).all()
    torch.testing.assert_close(speed, torch.cat((torch.tensor([5.]),speed[:-1])) + .1*acceleration)
    assert abs(float(result.target_speed_kmh.reshape(-1)[0]) - float(speed[0])*3.6) < 1e-4
    return dict(status='pass', checkpoint_sha256=digest, device=device,
                schema_version=head.schema_version,
                weight_mode='trained' if checkpoint else 'random contract fixture',
                scope='Synthetic shape, trained asset loading, authorization and kinematic consistency only',
                sequence=seq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    args = parser.parse_args()
    print(json.dumps(smoke(args.checkpoint,args.device),indent=2))
