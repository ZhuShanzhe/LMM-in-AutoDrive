"""Re-run scene2 geometry checks from bundled OpenDRIVE; no CARLA server."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT),str(ROOT/'experiment/CARLA')]
import carla
from benchmark.catalog import ConfigError,load_catalog
from benchmark.task_oracle import load_profile
from benchmark.turn_fixture import bind_junction_sequence

world_map=carla.Map('Town05_Opt_handoff',(HERE/'_common/scene2/map.xodr').read_text())
route=json.loads((HERE/'_common/scene2/route.json').read_text())
catalog=load_catalog('scene_2')
results=[]
for index,task in enumerate(catalog.tasks):
    profile=load_profile(catalog,task)
    if profile is None or not any(s['kind'] in ('turn','straight_junction') for s in profile['steps']):
        results.append(dict(task=task.task_id,status='NO_JUNCTION_REQUIREMENT'))
        continue
    end=profile.get('end_route_s_m',catalog.tasks[index+1].activate_m if index+1<len(catalog.tasks) else catalog.route_length_m)
    try:
        binding=bind_junction_sequence(route,world_map,carla.Location,profile['steps'],task.activate_m,end)
        results.append(dict(task=task.task_id,status='BOUND',evidence=binding))
    except ConfigError as exc:
        results.append(dict(task=task.task_id,status='MISMATCH',reason=str(exc)))
print(json.dumps(dict(scope='static geometry check, not driving acceptance',rows=results),ensure_ascii=False,indent=2))
