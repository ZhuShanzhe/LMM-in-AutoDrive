import pytest
from universal_vla_controller import UniversalVLAController


@pytest.mark.parametrize('direction',['front','left','right'])
def test_unavailable_camera_cannot_authorize_a_lane_change(direction):
    controller=UniversalVLAController.__new__(UniversalVLAController)
    controller.available_cameras=[]
    result=controller.predict_target_lane_risk(None,direction)
    assert result['risk_level']=='high'
    assert result['reason_codes']==['requested_camera_unavailable']
    assert not result['lane_change']['left']['is_safe']
    assert not result['lane_change']['right']['is_safe']
