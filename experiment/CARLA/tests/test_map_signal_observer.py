import numpy as np
from control.map_signal_observer import classify_signal_crop,observe_tracked_lamp


def track():
    return dict(signal_id='a',state='GREEN',bbox=[4.,4.,7.,7.],x=5.,y=5.,depth=10.,timestamp_s=0.)


def test_tracked_lamp_color_always_comes_from_current_pixels():
    image=np.zeros((12,12,3),dtype=np.uint8)
    for color,state in [([255,0,0],'RED'),([0,255,0],'GREEN'),([0,0,0],'UNKNOWN')]:
        image[4:8,4:8]=color
        result=observe_tracked_lamp(image,track(),signal_id='a',x=5.,y=5.,depth=10.,timestamp_s=.1)
        assert result['state']==state


def test_expired_or_out_of_view_signal_does_not_provide_a_color():
    image=np.full((12,12,3),[0,255,0],dtype=np.uint8)
    assert observe_tracked_lamp(image,track(),signal_id='a',x=-50.,y=5.,depth=10.,timestamp_s=1.) is None
    assert observe_tracked_lamp(image,track(),signal_id='a',x=5.,y=5.,depth=10.,timestamp_s=31.) is None
    assert observe_tracked_lamp(image,track(),signal_id='b',x=5.,y=5.,depth=10.,timestamp_s=1.) is None


def test_conflicting_emitter_colors_are_unknown():
    image=np.zeros((6,6,3),dtype=np.uint8)
    image[:3]=[255,0,0];image[3:]=[0,255,0]
    assert classify_signal_crop(image)[0]=='UNKNOWN'


def test_stop_line_uses_mapped_line_not_signal_pole_reference():
    from types import SimpleNamespace as Obj
    from control.map_signal_observer import mapped_stop_distance
    entries=[dict(road_id=1,section_id=0,lane_id=-1,position=[20.,0.,0.],forward=[1.,0.,0.])]
    assert mapped_stop_distance(entries,Obj(road_id=1,section_id=0,lane_id=-1),[2.,0.,0.])==18.
    assert mapped_stop_distance(entries,Obj(road_id=1,section_id=0,lane_id=-1),[21.,0.,0.])==-1.
    assert mapped_stop_distance(entries,Obj(road_id=2,section_id=0,lane_id=-1),[2.,0.,0.]) is None


def test_tracking_cannot_introduce_green_without_recent_detector_confirmation():
    image=np.full((12,12,3),[0,255,0],dtype=np.uint8)
    last=track();last['state']='RED'
    assert observe_tracked_lamp(image,last,signal_id='a',x=5.,y=5.,depth=10.,timestamp_s=.1)['state']=='UNKNOWN'
    assert observe_tracked_lamp(image,track(),signal_id='a',x=5.,y=5.,depth=10.,timestamp_s=.6)['state']=='UNKNOWN'


def test_stop_line_on_next_road_is_associated_through_route_landmark():
    from types import SimpleNamespace as Obj
    from control.map_signal_observer import MapSignalObserver
    effective=Obj(road_id=2,section_id=0,lane_id=-1)
    landmark=Obj(waypoint=effective,distance=28.,id='a',get_lane_validities=lambda:[(-1,-1)],
        transform=Obj(location=Obj(x=20.,y=0.,z=0.)))
    current=Obj(road_id=1,section_id=0,lane_id=-1,get_landmarks_of_type=lambda *args:[landmark])
    observer=MapSignalObserver.__new__(MapSignalObserver)
    observer.world_map=Obj(get_waypoint=lambda _:current)
    observer.camera=Obj(attributes={'fov':'100'},get_transform=lambda:Obj(get_inverse_matrix=lambda:np.eye(4)))
    observer.detector=Obj(predict=lambda **kwargs:[Obj(boxes=[])])
    observer.classes=[9];observer.device='cpu';observer.last_frame=None;observer.last_observation=None
    observer.tracked_signal=None;observer.debug_directory=None;observer.last_debug_bucket=None
    observer.stop_lines={'a':[dict(road_id=2,section_id=0,lane_id=-1,position=[20.,0.,0.],forward=[1.,0.,0.])]}
    transform=Obj(location=Obj(x=0.,y=0.,z=0.),get_forward_vector=lambda:Obj(x=1.,y=0.,z=0.))
    ego=Obj(get_location=lambda:transform.location,get_transform=lambda:transform,bounding_box=Obj(extent=Obj(x=2.)))
    observation=observer.observe(ego,np.zeros((16,16,3),dtype=np.uint8),frame=1,timestamp_s=0.)
    assert observation['stop_line_source']=='static_hdmap_stop_line'
    assert observation['stop_distance_m']==18.
    current.get_landmarks_of_type=lambda *args:[]
    observation=observer.observe(ego,np.zeros((16,16,3),dtype=np.uint8),frame=2,timestamp_s=.1)
    assert observation['applicable'] and observation['stop_distance_m']==18.
