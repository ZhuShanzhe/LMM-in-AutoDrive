from scene_understanding.realtime_perception.evaluate_scene3_capture import (
    front_relevant_truth,
    nearest_truth,
)


def test_nearest_truth_reports_auditable_frame_delta():
    row = {"simulation_frame": 120}
    found, delta = nearest_truth({120: row}, 111, 12)
    assert found is row
    assert delta == 9


def test_front_relevant_truth_filters_rear_and_far_actors():
    def actor(type_id, longitudinal, lateral, distance):
        return {
            "type_id": type_id,
            "relation_to_ego": {
                "longitudinal_m": longitudinal,
                "lateral_m": lateral,
                "euclidean_distance_m": distance,
            },
        }

    truth = {
        "actors": {
            "front_vehicle": actor("vehicle.audi.tt", 30, 1, 30),
            "front_worker": actor("walker.pedestrian.0001", 20, 2, 20),
            "rear_vehicle": actor("vehicle.tesla.model3", -5, 0, 5),
            "far_vehicle": actor("vehicle.lincoln.mkz", 140, 0, 140),
        }
    }
    assert front_relevant_truth(truth) == {"vehicle": 1, "pedestrian": 1}
