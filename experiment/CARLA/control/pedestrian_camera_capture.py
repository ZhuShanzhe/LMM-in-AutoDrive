"""Normal-exposure pixels for people; lamp exposure is a separate sensor task."""
import math

def pedestrian_capture(rig,sensor_frame):
    rgb,pose,fov=rig.view_capture(sensor_frame,'front')
    if rgb is None or pose is None or not isinstance(fov,(int,float)) or not math.isfinite(fov):
        return None,None,None,'synchronized_normal_front_unavailable'
    return rgb,pose,float(fov),'synchronized_normal_front_capture'
