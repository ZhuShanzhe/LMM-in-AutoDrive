import json
from types import SimpleNamespace as NS
import pytest
from benchmark.map_snapshot import save_map_snapshot,load_map_snapshot
from benchmark.catalog import ConfigError


def test_spawn_order_and_map_identity_round_trip(tmp_path):
    transforms=[NS(location=NS(x=x,y=0,z=1),rotation=NS(pitch=0,yaw=90,roll=0)) for x in (20,10)]
    world_map=NS(name='Carla/Maps/Town05_Opt',to_opendrive=lambda:'<OpenDRIVE/>',get_spawn_points=lambda:transforms)
    save_map_snapshot(world_map,tmp_path/'snapshot')
    api=NS(Location=lambda **kw:NS(**kw),Rotation=lambda **kw:NS(**kw),
           Transform=lambda location,rotation:NS(location=location,rotation=rotation),
           Map=lambda name,xml:NS(name=name,to_opendrive=lambda:'changed'))
    restored=load_map_snapshot(tmp_path/'snapshot',api,'Town05_Opt')
    assert [p.location.x for p in restored.get_spawn_points()]==[20,10]
    assert restored.to_opendrive()=='<OpenDRIVE/>'
    path=tmp_path/'snapshot/manifest.json'
    manifest=json.loads(path.read_text())
    manifest['spawn_points'].reverse()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ConfigError,match='spawn-point hash'):
        load_map_snapshot(tmp_path/'snapshot',api,'Town05_Opt')
