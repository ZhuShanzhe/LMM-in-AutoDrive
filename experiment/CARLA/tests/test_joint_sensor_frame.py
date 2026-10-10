from collections import OrderedDict
import threading
import time
import pytest
import torch
from carla_multiview_sensor import SynchronizedMultiviewCameraRig,CAMERA_ORDER


def rig():
    result=SynchronizedMultiviewCameraRig.__new__(SynchronizedMultiviewCameraRig)
    result.enable_lidar=True;result.enable_radar=True
    result.width=result.height=2;result.available_cameras=CAMERA_ORDER
    result._condition=threading.Condition();result._latest_multisensor_frame=11
    result._frames=OrderedDict((f,{**{name:torch.full((3,2,2),f,dtype=torch.uint8)
        for name in CAMERA_ORDER},'lidar_bev':torch.full((4,64,64),float(f))}) for f in (10,11))
    result._radar_observations={'front':OrderedDict({10:{'sensor_frame':10}}),
        'rear':OrderedDict({10:{'sensor_frame':10},11:{'sensor_frame':11}})}
    return result


def test_latest_camera_frame_cannot_mix_with_an_older_radar():
    instance=rig()
    frame,images,mask,bev,_=instance.latest_multisensor(minimum_frame=11,timeout_s=0.)
    assert frame==10 and bool(mask.all())
    assert bool((images==10).all()) and bool((bev==10).all())


def test_wait_wakes_when_the_matching_radar_callback_arrives():
    instance=rig()
    def publish():
        time.sleep(.01)
        with instance._condition:
            instance._radar_observations['front'][11]={'sensor_frame':11}
            instance._condition.notify_all()
    worker=threading.Thread(target=publish);worker.start()
    try:
        frame,images,_,bev,_=instance.latest_multisensor(minimum_frame=11,timeout_s=.2)
    finally:worker.join()
    assert frame==11 and bool((images==11).all()) and bool((bev==11).all())


def test_no_common_frame_is_not_a_zero_filled_clear_radar():
    instance=rig();instance._radar_observations['rear'].clear()
    with pytest.raises(RuntimeError,match='no synchronized'):
        instance.latest_multisensor(minimum_frame=11,timeout_s=0.)


def test_late_older_callback_cannot_replace_the_newest_capture(monkeypatch):
    import radar_ground_filter
    monkeypatch.setattr(radar_ground_filter,'filter_ground',lambda observation,*args,**kwargs:dict(
        observation,ground_filter={}))
    instance=rig()
    instance._radar_observations['front']=OrderedDict((f,dict(sensor_frame=f)) for f in (11,10))
    assert instance.latest_radar('front',maximum_frame=11)['sensor_frame']==11
    assert instance.latest_radar('front',maximum_frame=10)['sensor_frame']==10
