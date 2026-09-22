from types import SimpleNamespace as NS
from benchmark.traffic_observation import observe_traffic, TrafficDensitySummary


def actor(x,y,yaw=0):
    return NS(get_transform=lambda:NS(location=NS(x=x,y=y,z=0),rotation=NS(yaw=yaw)))


def test_front_cone_counts_distinguish_directions_and_ego_lane():
    actors={1:actor(0,0),2:actor(20,0),3:actor(30,4),4:actor(40,-4,180),5:actor(-20,0)}
    snapshot=NS(frame=10,find=actors.get)
    world_map=NS(get_waypoint=lambda p:NS(road_id=1,section_id=0,lane_id=-1 if p.y==0 else -2))
    row=observe_traffic(snapshot,1,[2,3,4,5,99],world_map)
    assert row['counts']['front_cone']==3
    assert row['counts']['same_direction']==2
    assert row['counts']['opposite_direction']==1
    assert row['counts']['ego_lane']==1
    assert row['actors_missing_from_snapshot']==1
    summary=TrafficDensitySummary()
    summary.update(row)
    assert summary.result()['below_three_front_actor_fraction']==0
