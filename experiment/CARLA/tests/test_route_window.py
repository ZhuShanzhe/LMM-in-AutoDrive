from benchmark.route_window import RouteWindow


def test_window_advances_without_returning_to_route_start():
    route=[{'distance_m':i,'x':i} for i in range(0,1001,5)]
    window=RouteWindow(route)
    points=window.points(600)
    assert points[0]['distance_m']==605
    assert points[-1]['distance_m']==850
    assert not window.at_end(990)
    assert window.at_end(997)
    assert window.points(1000)==[]
