"""Preserve RPC spawn-point ordering alongside OpenDRIVE for offline route audits."""
import hashlib
import json
from pathlib import Path

from .catalog import ConfigError


def save_map_snapshot(world_map, directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    payload=world_map.to_opendrive().encode('utf-8')
    points=[]
    for transform in world_map.get_spawn_points():
        points.append(dict(location={key:float(getattr(transform.location,key)) for key in ('x','y','z')},
                           rotation={key:float(getattr(transform.rotation,key)) for key in ('pitch','yaw','roll')}))
    (directory/'map.xodr').write_bytes(payload)
    manifest=dict(schema_version='carla_map_snapshot/1.0',map_name=world_map.name,
                  opendrive_sha256=hashlib.sha256(payload).hexdigest(),spawn_points=points,
                  spawn_points_sha256=hashlib.sha256(json.dumps(points,sort_keys=True,allow_nan=False).encode()).hexdigest(),
                  scope='road geometry and ordered spawn transforms; no mesh, traffic light actors or navigation mesh')
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    return manifest


def load_map_snapshot(directory, carla, expected_map):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version')!='carla_map_snapshot/1.0':
        raise ConfigError('unsupported map snapshot')
    if manifest['map_name'].split('/')[-1]!=expected_map:
        raise ConfigError('map snapshot belongs to another map')
    payload=(directory/'map.xodr').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=manifest['opendrive_sha256']:
        raise ConfigError('map snapshot OpenDRIVE hash mismatch')
    if not manifest.get('spawn_points'):
        raise ConfigError('map snapshot missing ordered spawn points')
    digest=hashlib.sha256(json.dumps(manifest['spawn_points'],sort_keys=True,allow_nan=False).encode()).hexdigest()
    if digest!=manifest.get('spawn_points_sha256'):
        raise ConfigError('map snapshot spawn-point hash mismatch')
    transforms=[carla.Transform(carla.Location(**point['location']),carla.Rotation(**point['rotation']))
                for point in manifest['spawn_points']]
    native=carla.Map(expected_map,payload.decode('utf-8'))
    class SnapshotMap:
        def __getattr__(self, name):
            return getattr(native,name)
        def get_spawn_points(self):
            return list(transforms)
        def to_opendrive(self):
            return payload.decode('utf-8')
    return SnapshotMap()
