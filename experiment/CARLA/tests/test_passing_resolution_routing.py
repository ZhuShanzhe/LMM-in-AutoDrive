import numpy as np
from control.passing_conditions import PassingConditionObserver

class Detector:
    def __init__(self):self.calls=[]
    def detect(self,image):
        self.calls.append(('one',int(image[0,0,0])))
        return [int(image[0,0,0])],0.
    def detect_many(self,images):
        self.calls.append(('batch',len(images)))
        return [[int(image[0,0,0])] for image in images],0.

def observer(front=True):
    o=PassingConditionObserver.__new__(PassingConditionObserver)
    o.detector=Detector();o.front_detector=Detector() if front else None
    return o

def views(names):
    return [(name,np.full((4,4,3),index,dtype=np.uint8),np.eye(4),100.) for index,name in enumerate(names)]

def test_high_resolution_front_preserves_output_view_order():
    o=observer();assert o.detect_views(views(['left','front','rear','right']))==[[0],[1],[2],[3]]
    assert o.front_detector.calls==[('one',1)] and o.detector.calls==[('batch',3)]

def test_no_front_view_does_not_synthesize_an_image():
    o=observer();assert o.detect_views(views(['left','right']))==[[0],[1]]
    assert not o.front_detector.calls

def test_detector_without_front_override_keeps_batch_path():
    o=observer(False);assert o.detect_views(views(['front','rear']))==[[0],[1]]
    assert o.detector.calls==[('batch',2)]

def test_empty_views_do_not_invoke_detector():
    o=observer();assert o.detect_views([])==[] and not o.detector.calls and not o.front_detector.calls
