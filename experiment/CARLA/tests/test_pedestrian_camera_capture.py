from types import SimpleNamespace as NS
import pytest
from control.pedestrian_camera_capture import pedestrian_capture

def test_normal_exposure_and_its_pose_fov_are_used_for_the_requested_frame():
    rgb=object();pose=object();requests=[]
    def view(frame,name):
        requests.append((frame,name));return rgb,pose,95.
    rig=NS(view_capture=view,signal_capture=lambda frame:pytest.fail('Lamp exposure must not be used'))
    assert pedestrian_capture(rig,123)==(rgb,pose,95.,'synchronized_normal_front_capture')
    assert requests==[(123,'front')]

@pytest.mark.parametrize('missing',['rgb','pose','fov','nonfinite_fov'])
def test_missing_normal_capture_fails_closed_without_using_lamp_pixels(missing):
    values=[object(),object(),100.]
    values[{'rgb':0,'pose':1,'fov':2,'nonfinite_fov':2}[missing]]=float('nan') if missing=='nonfinite_fov' else None
    rig=NS(view_capture=lambda frame,name:values,signal_capture=lambda frame:pytest.fail('No dark-image fallback'))
    assert pedestrian_capture(rig,123)==(None,None,None,'synchronized_normal_front_unavailable')
