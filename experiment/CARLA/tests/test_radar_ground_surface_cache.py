from copy import deepcopy
import threading
from unittest.mock import patch
import numpy as np
from carla_multiview_sensor import SynchronizedMultiviewCameraRig
import radar_ground_filter as module


def fixture():
    x,y=np.meshgrid(np.linspace(-20,20,161),np.linspace(-20,20,161))
    cloud=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    matrix=np.eye(4);matrix[2,3]=1.6
    item=dict(distance_m=10.,azimuth_deg=0.,altitude_deg=-9.2,
        relative_velocity_mps=-2.,closing_speed_mps=2.,relative_height_m=-1.6)
    observation=dict(schema_version='physical_front_radar/1.0',sensor_frame=42,
        measurement_timestamp_s=1.,_sensor_matrix=matrix,_ground_candidates=[item])
    return cloud,matrix,observation


def test_prepared_surface_has_identical_filter_output():
    cloud,matrix,observation=fixture()
    expected=module.filter_ground(observation,cloud,matrix)
    surface=module.prepare_ground_surface(cloud)
    assert module.filter_ground(observation,cloud,matrix,surface=surface)==expected


def test_front_rear_reuse_one_frame_bound_surface_index():
    cloud,matrix,observation=fixture()
    rig=SynchronizedMultiviewCameraRig.__new__(SynchronizedMultiviewCameraRig)
    rig.enable_radar=True;rig._condition=threading.Condition()
    rig._frames={42:dict(ground_points_world=cloud,ground_timestamp_s=1.)}
    rig._radar_observations={name:{42:deepcopy(observation)} for name in ('front','rear')}
    with patch.object(module,'cKDTree',wraps=module.cKDTree) as build:
        front=rig.latest_radar('front',maximum_frame=42)
        rear=rig.latest_radar('rear',maximum_frame=42)
        assert build.call_count==1
        assert front==rear
        next_observation=deepcopy(observation)
        next_observation.update(sensor_frame=43,measurement_timestamp_s=1.05)
        rig._radar_observations['front'][43]=next_observation
        rig._frames[43]=dict(ground_points_world=cloud,ground_timestamp_s=1.05)
        rig.latest_radar('front',maximum_frame=43)
        assert build.call_count==2


def test_invalid_surface_never_authorizes_ground_rejection():
    _,matrix,observation=fixture()
    for cloud in (None,[],np.zeros((7,3)),np.full((20,3),np.nan),np.zeros((20,2))):
        assert module.prepare_ground_surface(cloud) is None
        assert module.filter_ground(observation,cloud,matrix)['ground_filter']['status']=='unavailable'


def test_ground_filter_does_not_hold_sensor_callback_lock():
    cloud,matrix,observation=fixture()
    rig=SynchronizedMultiviewCameraRig.__new__(SynchronizedMultiviewCameraRig)
    rig.enable_radar=True;rig._condition=threading.Condition()
    rig._frames={42:dict(ground_points_world=cloud,ground_timestamp_s=1.)}
    rig._radar_observations={'front':{42:deepcopy(observation)}}
    original=module.filter_ground
    def checked_filter(*args,**kwargs):
        acquired=threading.Event()
        def callback():
            with rig._condition:acquired.set()
        worker=threading.Thread(target=callback,daemon=True)
        worker.start()
        assert acquired.wait(1.),'Ground calculation blocked sensor callback'
        worker.join(timeout=1.)
        return original(*args,**kwargs)
    with patch.object(module,'filter_ground',side_effect=checked_filter):
        result=rig.latest_radar('front',maximum_frame=42)
    assert result['ground_filter']['status']=='same_frame_local_plane'
