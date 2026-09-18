from collections import OrderedDict
from threading import Condition
from types import SimpleNamespace
import numpy as np
from carla_multiview_sensor import SynchronizedMultiviewCameraRig


def test_high_resolution_front_preserves_exact_frame_and_model_shape():
    rig=SynchronizedMultiviewCameraRig.__new__(SynchronizedMultiviewCameraRig)
    rig.width=rig.height=2;rig.front_capture_size=8;rig._front_images=OrderedDict()
    rig._condition=Condition();rig._frames=OrderedDict();rig.available_cameras=('front',)
    rig._retained_frames=2;rig.enable_lidar=False;rig._latest_complete_frame=-1
    receive=rig._callback('front')
    for frame in (10,11,12):
        bgra=np.full((8,8,4),frame,dtype=np.uint8)
        pose=np.eye(4);pose[0,3]=frame
        receive(SimpleNamespace(raw_data=bgra.tobytes(),height=8,width=8,frame=frame,
            transform=SimpleNamespace(get_inverse_matrix=lambda p=pose:p)))
    assert tuple(rig._frames[12]['front'].shape)==(3,2,2)
    assert rig.front_capture(11).shape==(8,8,3)
    assert rig.front_capture(11)[0,0,0]==11
    assert rig.front_capture(10) is None and rig.front_capture(13) is None
    assert rig.front_capture_inverse(11)[0,3]==11
    assert rig.front_capture_inverse(10) is None and rig.front_capture_inverse(13) is None
    copy=rig.front_capture_inverse(11);copy[0,3]=999
    assert rig.front_capture_inverse(11)[0,3]==11
