from benchmark.route_audit import audit_route


def point(x,y,s,yaw,road,junction=False):
    return dict(x=x,y=y,z=0,distance_m=s,yaw=yaw,road_id=road,section_id=0,lane_id=-1,is_junction=junction)


def test_distinct_road_turn_and_shallow_curve_are_different():
    route=[point(0,0,0,0,1),point(5,0,5,20,10,True),point(8,4,10,60,10,True),point(8,9,15,90,2)]
    report=audit_route(route)
    assert report['distinct_road_turn_candidates'][0]['direction']=='RIGHT'
    assert not report['legal_turns_verified']
    for p in route:
        p['yaw']/=10
    assert not audit_route(route)['distinct_road_turn_candidates']


def test_incomplete_junction_cannot_prove_turn():
    report=audit_route([point(0,0,0,0,1,True),point(5,0,5,90,2)])
    assert not report['junction_visits'][0]['complete']
    assert not report['distinct_road_turn_candidates']


def test_discontinuous_geometry_is_reported():
    report=audit_route([point(0,0,0,0,1),point(100,0,5,0,1)])
    assert len(report['suspicious_gaps'])==1
