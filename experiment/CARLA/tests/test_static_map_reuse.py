from types import SimpleNamespace
from unittest.mock import Mock

from evaluation.ground_truth import FrameGroundTruthRecorder


def test_recorder_reuses_static_map_but_reads_current_pose_and_new_world(tmp_path):
    world, other_world, ego = Mock(), Mock(), Mock()
    waypoint = SimpleNamespace(road_id=1, section_id=0, lane_id=-1, is_junction=False)
    world.get_map.return_value.get_waypoint.return_value = waypoint
    other_world.get_map.return_value.get_waypoint.return_value = SimpleNamespace(
        road_id=2, section_id=0, lane_id=-1, is_junction=False)
    location = SimpleNamespace(x=1.0, y=0.0, z=0.0)
    ego.id = 1
    ego.get_transform.return_value = SimpleNamespace(location=location,
        rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0))
    ego.get_velocity.return_value = SimpleNamespace(x=2.0, y=0.0, z=0.0)
    recorder = FrameGroundTruthRecorder(tmp_path / "truth.jsonl", scene_id="scene_2", events=[])
    try:
        def record(active_world, frame):
            return recorder.record(world=active_world, ego=ego, simulation_frame=frame,
                timestamp_s=frame * .05, route_s_m=frame, event_states={}, actor_bindings={})
        first = record(world, 1)
        location.x = 7.0
        second = record(world, 2)
        assert first["ego"]["transform"]["location"]["x"] == 1.0
        assert second["ego"]["transform"]["location"]["x"] == 7.0
        world.get_map.assert_called_once()
        assert world.get_map.return_value.get_waypoint.call_count == 2
        third = record(other_world, 3)
        other_world.get_map.assert_called_once()
        assert third["ego"]["lane"]["road_id"] == 2
    finally:
        recorder.close()
def test_background_traffic_static_map_cache_invalidates_on_world_switch():
    from types import SimpleNamespace
    from scenarios.complex.town05_scene2 import TownTrafficFlow

    calls=[]
    road=object()
    flow=TownTrafficFlow.__new__(TownTrafficFlow)
    flow.world=SimpleNamespace(get_map=lambda:calls.append('first') or road)
    assert flow._static_road_map() is road
    assert flow._static_road_map() is road
    assert calls==['first']
    other=object()
    flow.world=SimpleNamespace(get_map=lambda:calls.append('second') or other)
    assert flow._static_road_map() is other
    assert calls==['first','second']

