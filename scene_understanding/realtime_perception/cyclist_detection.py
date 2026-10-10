"""Generic bicycle-class augmentation; independent of routes and instructions."""
import numpy as np
from .detector import Detection


def merge_cyclist_expert(detections, expert_detections, *, authoritative=False):
    cyclists=[Detection(d.bbox_xyxy,d.confidence,'cyclist',d.subtype,1)
        for d in expert_detections if d.category=='cyclist' and d.confidence>=.25]
    def duplicate_rider(detection):
        if detection.category not in {'motorcycle','pedestrian','cyclist'}:return False
        a=np.asarray(detection.bbox_xyxy,dtype=float)
        for cyclist in cyclists:
            b=np.asarray(cyclist.bbox_xyxy,dtype=float)
            intersection=np.maximum(0.,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            area_a=np.maximum(0.,a[2:]-a[:2]).prod()
            area_b=np.maximum(0.,b[2:]-b[:2]).prod()
            union=area_a+area_b-intersection
            if union>0 and intersection/union>=.5:return True
            # COCO's bicycle-only box can sit inside the expert's rider+bike box.
            if (detection.category=='cyclist' and area_a>0 and area_b>0
                    and intersection/area_a>=.85 and .15<=area_a/area_b<=1.):return True
        return False
    return [d for d in detections if not (authoritative and d.category=='cyclist')
        and not duplicate_rider(d)]+cyclists


class CyclistAugmentedDetector:
    """Preserve base categories while supplying explicitly detected bicycles."""
    def __init__(self,base,expert):
        self.base,self.expert=base,expert

    def detect(self,image):
        existing,first_ms=self.base.detect(image)
        extra,second_ms=self.expert.detect(image)
        return merge_cyclist_expert(existing,extra,authoritative=True),first_ms+second_ms

    def detect_many(self,images):
        existing,first_ms=self.base.detect_many(images)
        extra,second_ms=self.expert.detect_many(images)
        if len(existing)!=len(images) or len(extra)!=len(images):
            raise RuntimeError('Cyclist detector batch must preserve every input view')
        return [merge_cyclist_expert(a,b,authoritative=True) for a,b in zip(existing,extra)],first_ms+second_ms
