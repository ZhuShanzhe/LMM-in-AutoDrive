from collections import OrderedDict
from threading import Condition
import numpy as np
from carla_multiview_sensor import SynchronizedMultiviewCameraRig


def rig():
    r=SynchronizedMultiviewCameraRig.__new__(SynchronizedMultiviewCameraRig)
    r.signal_camera=object();r._condition=Condition()
    r._signal_images=OrderedDict();r._signal_poses=OrderedDict()
    return r


def test_bracket_does_not_reuse_future_or_other_exposure_pose():
    r=rig();r._signal_images[12]=np.ones((2,2,3));r._signal_poses[12]=np.eye(4)
    assert r.signal_capture(11,timeout_s=0.)==(None,None)
    image,pose=r.signal_capture(12,timeout_s=0.)
    assert image is r._signal_images[12]
    pose[0,0]=99.
    assert r._signal_poses[12][0,0]==1.


def test_legacy_rig_fallback_remains_exact_frame():
    r=rig();r.signal_camera=None
    r._front_images={10:np.ones((2,2,3))};r._front_poses={10:np.eye(4)}
    assert r.signal_capture(9)==(None,None)
    image,pose=r.signal_capture(10)
    assert image is r._front_images[10] and np.array_equal(pose,np.eye(4))
