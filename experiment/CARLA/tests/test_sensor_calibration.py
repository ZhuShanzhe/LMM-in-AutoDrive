from types import SimpleNamespace
import pytest
from evaluation.sensor_calibration import sensor_record, calibration_document


def test_camera_intrinsics_and_mounting():
    matrix = [[1, 0, 0, 1.5], [0, 1, 0, 0], [0, 0, 1, 2.4], [0, 0, 0, 1]]
    actor = SimpleNamespace(type_id="sensor.camera.rgb", attributes={
        "image_size_x": "960", "image_size_y": "540", "fov": "90", "gamma": "2.2"})
    result = sensor_record(actor, SimpleNamespace(get_matrix=lambda: matrix))
    assert result["intrinsic_matrix"][0] == pytest.approx([480, 0, 480])
    assert result["intrinsic_matrix"][1] == pytest.approx([0, 480, 270])
    assert result["sensor_to_ego"] == matrix
    assert result["attributes"]["gamma"] == "2.2"
    assert calibration_document({"front_rgb": result})["length_unit"] == "metre"
