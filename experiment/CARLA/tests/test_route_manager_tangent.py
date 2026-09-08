from continuous.route_manager import RouteManager


def test_target_yaw_follows_polyline_tangent_not_jumping_waypoint_rotation():
    manager = RouteManager.__new__(RouteManager)
    manager.route = [
        {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "distance_m": 0.0},
        {"x": 10.0, "y": 2.5, "z": 0.0, "yaw": 65.0, "distance_m": 10.5},
    ]
    manager.progress_m = 0.0
    manager.current_index = 0
    target = manager.target_point(10.0)
    assert 13.0 < target["yaw"] < 15.0
