"""Scene-side crossing start gate, never a source of ego-policy permission."""
import math


def crossing_start_observation(start, target, vehicles, *, margin_m=.5):
    values=[float(getattr(p,k)) for p in (start,target) for k in ('x','y','z')]
    if not all(math.isfinite(v) for v in values) or not math.isfinite(margin_m) or margin_m<0:
        raise ValueError('Crossing geometry and margin must be finite')
    if math.hypot(target.x-start.x,target.y-start.y)<.1:
        raise ValueError('Degenerate crossing path')
    blocked=[]
    for actor in vehicles:
        transform=actor.get_transform();yaw=math.radians(float(transform.rotation.yaw))
        c,s=math.cos(yaw),math.sin(yaw)
        vertices=actor.bounding_box.get_world_vertices(transform)
        if len(vertices)!=8:raise ValueError('Vehicle body must supply eight vertices')
        if any(not all(math.isfinite(float(getattr(p,k))) for k in ('x','y','z')) for p in vertices):
            raise ValueError('Nonfinite vehicle body')
        # CARLA supplies transformed body vertices, including box offsets and slopes.
        # Projection makes a conservative yaw-aligned rectangle on the road plane.
        if abs(sum(p.z for p in vertices)/8.-(start.z+target.z)*.5)>3.:continue
        def xy(p):return c*p.x+s*p.y,-s*p.x+c*p.y
        corners=[xy(p) for p in vertices];a=xy(start);b=xy(target)
        bounds=[(min(p[i] for p in corners)-margin_m,max(p[i] for p in corners)+margin_m) for i in (0,1)]
        lower,upper=0.,1.
        for axis,(lo,hi) in enumerate(bounds):
            delta=b[axis]-a[axis]
            if abs(delta)<1e-9:
                if not lo<=a[axis]<=hi:lower,upper=1.,0.;break
            else:
                t0,t1=(lo-a[axis])/delta,(hi-a[axis])/delta
                lower=max(lower,min(t0,t1));upper=min(upper,max(t0,t1))
        if lower<=upper:blocked.append(int(actor.id))
    return dict(clear=not blocked,blocked_actor_ids=blocked,
        source='scene_fixture_measured_vehicle_bodies',fixture_truth_used=True,policy_input=False,
        margin_m=float(margin_m))
