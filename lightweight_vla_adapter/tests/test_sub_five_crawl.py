from lightweight_vla_adapter.src.risk_behavior import RiskBehaviorPolicy,cautious_permission,stopping_clearance
from lightweight_vla_adapter.tests.test_risk_behavior import risk,clearance


def test_measured_short_horizon_has_no_five_kmh_dead_band():
    policy=RiskBehaviorPolicy()
    final=None
    for i in range(16):
        t=i*.1
        r=risk('f'+str(i),t)
        r['risk_evidence']['sensor_frame']=i
        r['physical_forward_radar']['sensor_frame']=i
        lidar=clearance()
        lidar.update(sensor_frame=i,observed_at_s=t,clear_distance_m=4.17,nearest_obstacle_m=5.17)
        final=policy.apply(r,lidar,frame_id=r['frame_id'],timestamp_s=t,
            ego_speed_kmh=0.,parsed_intent='KEEP_LANE')
    permission=cautious_permission(final,frame_id=final['frame_id'])
    assert permission is not None
    assert 1. <= permission['max_speed_kmh'] < 5.
    assert stopping_clearance(permission['max_speed_kmh'])<=4.17


def test_unmeasured_or_close_obstacle_never_gets_crawl_authority():
    for measured,obstacle in [(0.,None),(4.17,.5),(4.17,3.)]:
        policy=RiskBehaviorPolicy()
        for i in range(16):
            t=i*.1;r=risk('f'+str(i),t);lidar=clearance()
            r['risk_evidence']['sensor_frame']=i
            r['physical_forward_radar']['sensor_frame']=i
            lidar.update(sensor_frame=i,observed_at_s=t,clear_distance_m=measured,nearest_obstacle_m=obstacle)
            final=policy.apply(r,lidar,frame_id=r['frame_id'],timestamp_s=t,
                ego_speed_kmh=0.,parsed_intent='KEEP_LANE')
        assert cautious_permission(final,frame_id=final['frame_id']) is None
