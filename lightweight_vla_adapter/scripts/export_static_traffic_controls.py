"""Export static HD-map stop lines, never dynamic signal colors or other actors."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    import carla
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=2300)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    client=carla.Client('127.0.0.1',args.port);client.set_timeout(20.)
    world=client.get_world();world_map=world.get_map();signals={};heads={};groups={}
    for actor in world.get_actors().filter('traffic.traffic_light*'):
        entries=[]
        for waypoint in actor.get_stop_waypoints():
            transform=waypoint.transform;loc=transform.location;heading=transform.get_forward_vector()
            entries.append(dict(road_id=waypoint.road_id,section_id=waypoint.section_id,lane_id=waypoint.lane_id,
                s=waypoint.s,position=[loc.x,loc.y,loc.z],forward=[heading.x,heading.y,heading.z]))
        signals[str(actor.get_opendrive_id())]=entries
        groups[str(actor.get_opendrive_id())]=sorted(str(a.get_opendrive_id()) for a in actor.get_group_traffic_lights())
        heads[str(actor.get_opendrive_id())]=[
            [[v.x,v.y,v.z] for v in box.get_world_vertices(carla.Transform())]
            for box in actor.get_light_boxes()]
    if not signals:raise RuntimeError('No traffic-light actors found; static map was not exported')
    document=dict(schema_version='static_traffic_controls/1.1',town=world_map.name.rsplit('/',1)[-1],
        opendrive_sha256=hashlib.sha256(world_map.to_opendrive().encode()).hexdigest(),
        source='CARLA static stop-waypoint and lamp geometry export',dynamic_signal_state_exported=False,signals=signals,heads=heads,groups=groups)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(document,indent=2))
    print(json.dumps(dict(signals=len(signals),stop_lines=sum(map(len,signals.values())),output=str(args.output))))


if __name__=='__main__':main()
