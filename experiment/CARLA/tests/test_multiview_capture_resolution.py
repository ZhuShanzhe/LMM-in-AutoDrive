from collections import OrderedDict
import threading
from types import SimpleNamespace as NS
import numpy as np
import pytest
from carla_multiview_sensor import SynchronizedMultiviewCameraRig


@pytest.mark.parametrize('view,size',[('front',1280),('left',640),('right',640),('rear',640)])
def test_source_resolution_retained_but_neural_tensor_stays_fixed(view,size):
    rig=SynchronizedMultiviewCameraRig.__new__(SynchronizedMultiviewCameraRig)
    rig.width=rig.height=224;rig.front_capture_size=1280;rig.side_capture_size=640
    rig._capture=None;rig._condition=threading.Condition();rig._retained_frames=2
    rig._view_images={view:OrderedDict()};rig._view_poses={view:OrderedDict()}
    rig._front_images=OrderedDict();rig._front_poses=OrderedDict();rig._frames=OrderedDict()
    rig.available_cameras=[view];rig.enable_lidar=False
    rig._latest_complete_frame=-1
    image=np.zeros((size,size,4),dtype=np.uint8);image[:,:,2]=127
    rig._callback(view)(NS(frame=12,height=size,width=size,raw_data=image.tobytes(),
        transform=NS(get_inverse_matrix=lambda:np.eye(4))))
    assert rig._view_images[view][12].shape==(size,size,3)
    assert rig._frames[12][view].shape==(3,224,224)
    assert np.all(rig._frames[12][view][0].numpy()==127)
    assert rig._latest_complete_frame==12
