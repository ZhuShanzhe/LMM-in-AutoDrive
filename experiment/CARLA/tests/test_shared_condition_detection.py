from types import SimpleNamespace as NS
import numpy as np
from control.passing_conditions import PassingConditionObserver


def observer():
    calls=[]
    value=PassingConditionObserver.__new__(PassingConditionObserver)
    value.detector=NS(detect=lambda image:(calls.append('regular') or ['regular'],{}))
    value.front_detector=NS(detect=lambda image:(calls.append('front') or ['front'],{}))
    return value,calls


def test_same_image_source_can_share_exact_detections():
    value,calls=observer();rgb=np.zeros((8,8,3),dtype=np.uint8)
    packet=dict(source='synchronized_normal_front_capture',image=rgb,detections=('same',))
    result=value.detect_views([('front',rgb,None,100),('rear',rgb,None,100)],packet)
    assert result==[['same'],['regular']]
    assert calls==['regular']


def test_different_image_buffer_is_not_shared_even_with_same_pixels():
    value,calls=observer();rgb=np.zeros((8,8,3),dtype=np.uint8)
    packet=dict(source='synchronized_normal_front_capture',image=rgb.copy(),detections=('wrong',))
    assert value.detect_views([('front',rgb,None,100)],packet)==[['front']]
    assert calls==['front']


def test_exposure_bracket_source_is_not_shared():
    value,calls=observer();rgb=np.zeros((8,8,3),dtype=np.uint8)
    packet=dict(source='signal_front',image=rgb,detections=('wrong',))
    assert value.detect_views([('front',rgb,None,100)],packet)==[['front']]
    assert calls==['front']
