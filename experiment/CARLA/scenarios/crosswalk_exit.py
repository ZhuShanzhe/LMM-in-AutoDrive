"""Static-map pedestrian exit planning for deterministic simulation fixtures."""
import math


def sidewalk_exit_targets(road_map, target, tangent, *, max_distance_m=16., sample_step_m=2.):
    """Leave junction turning lanes via a short curb leg and a sidewalk leg."""
    import carla
    length=math.hypot(float(tangent.x),float(tangent.y))
    if length<1e-6 or not math.isfinite(length):raise ValueError('Finite road tangent required')
    if not math.isfinite(max_distance_m) or max_distance_m<4:raise ValueError('Invalid exit search distance')
    if not math.isfinite(sample_step_m) or sample_step_m<=0:raise ValueError('Invalid exit sample distance')
    ux,uy=tangent.x/length,tangent.y/length
    candidates=[]
    for sign in (-1.,1.):
        for distance in (4.,8.,12.,16.):
            if distance>max_distance_m:continue
            curb=carla.Location(x=target.x+sign*ux*distance,y=target.y+sign*uy*distance,z=target.z)
            sidewalk=road_map.get_waypoint(curb,project_to_road=True,lane_type=carla.LaneType.Sidewalk)
            if sidewalk is None or sidewalk.is_junction:continue
            final=sidewalk.transform.location
            if abs(final.z-target.z)>1.5 or math.hypot(final.x-curb.x,final.y-curb.y)>8.:continue
            # A final resting point must actually be outside every driving lane.
            if road_map.get_waypoint(final,project_to_road=False,lane_type=carla.LaneType.Driving) is not None:continue
            samples=[]
            points=[target,curb,final]
            for a,b in zip(points,points[1:]):
                count=max(1,int(math.ceil(math.hypot(b.x-a.x,b.y-a.y)/sample_step_m)))
                for index in range(1,count+1):
                    f=index/count
                    at=carla.Location(x=a.x+f*(b.x-a.x),y=a.y+f*(b.y-a.y),z=a.z+f*(b.z-a.z))
                    samples.append(road_map.get_waypoint(at,project_to_road=False,lane_type=carla.LaneType.Driving) is not None)
            # Once outside the road, never cross back through another lane.
            outside=False
            valid=True
            for driving in samples:
                if not driving:outside=True
                elif outside:valid=False;break
            if not valid:continue
            score=(sum(samples),distance+math.hypot(final.x-curb.x,final.y-curb.y))
            candidates.append((score,[curb,carla.Location(x=final.x,y=final.y,z=target.z)]))
    if not candidates:raise ValueError('No lane-free nonjunction sidewalk exit within search bound')
    return min(candidates,key=lambda value:value[0])[1]
