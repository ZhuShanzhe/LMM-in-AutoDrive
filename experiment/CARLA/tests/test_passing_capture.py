import threading
from collections import OrderedDict
import numpy as np
import pytest
import carla
from carla_multiview_sensor import SynchronizedMultiviewCameraRig,CAMERA_TRANSFORMS


def test_view_capture_never_substitutes_another_sensor_frame():
    rig=SynchronizedMultiviewCameraRig.__new__(SynchronizedMultiviewCameraRig)
    rig._condition=threading.Condition();rig._camera_fov=100.
    rgb=np.zeros((12,12,3),dtype=np.uint8);pose=np.eye(4)
    rig._view_images={'rear':OrderedDict({5:rgb})}
    rig._view_poses={'rear':OrderedDict({5:pose})}
    image,inverse,fov=rig.view_capture(5,'rear')
    assert image is rgb and fov==100.
    inverse[0,3]=3.
    assert rig._view_poses['rear'][5][0,3]==0.
    assert rig.view_capture(6,'rear')[:2]==(None,None)
    assert rig.view_capture(5,'front')[:2]==(None,None)
    with pytest.raises(ValueError):rig.view_capture(5,'unavailable')


def test_exposure_time_ego_pose_recovery_preserves_mounting_geometry():
    ego=carla.Transform(carla.Location(x=12.,y=8.,z=2.),carla.Rotation(pitch=3.,yaw=72.))
    x,y,z,pitch,yaw=CAMERA_TRANSFORMS['front']
    mounting=carla.Transform(carla.Location(x=x,y=y,z=z),carla.Rotation(pitch=pitch,yaw=yaw))
    camera=np.asarray(ego.get_matrix())@np.asarray(mounting.get_matrix())
    recovered=np.linalg.inv(np.linalg.inv(camera))@np.asarray(mounting.get_inverse_matrix())
    np.testing.assert_allclose(recovered,np.asarray(ego.get_matrix()),atol=1e-6)
