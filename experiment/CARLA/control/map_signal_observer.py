"""Prototype RGB traffic-light observer with static OpenDRIVE association.

Only map landmarks, camera pixels and ego pose are consumed. Dynamic CARLA
traffic-light actor states are forbidden in this module.
"""

import math
import numpy as np


def mapped_stop_distance(entries,waypoint,front_position):
    candidates=[item for item in entries if (item['road_id'],item['section_id'],item['lane_id'])==
        (waypoint.road_id,waypoint.section_id,waypoint.lane_id)]
    if not candidates:return None
    distances=[sum((item['position'][i]-front_position[i])*item['forward'][i] for i in range(3)) for item in candidates]
    ahead=[distance for distance in distances if distance>=-1.]
    return min(ahead) if ahead else max(distances)


def classify_signal_crop(rgb):
    import cv2
    if rgb.size==0:return 'UNKNOWN',0.
    hsv=cv2.cvtColor(rgb.astype(np.uint8),cv2.COLOR_RGB2HSV)
    h,s,v=(hsv[:,:,i] for i in range(3))
    bright=(s>90)&(v>150)
    counts=dict(RED=int((bright&((h<12)|(h>168))).sum()),
        YELLOW=int((bright&(h>=15)&(h<=40)).sum()),
        GREEN=int((bright&(h>=42)&(h<=95)).sum()))
    state=max(counts,key=counts.get);count=counts[state]
    total=sum(counts.values())
    confidence=count/max(1,total)
    return (state,float(confidence)) if count>=3 and confidence>=.75 else ('UNKNOWN',0.)


def observe_tracked_lamp(rgb,tracked,*,signal_id,x,y,depth,timestamp_s):
    if tracked['signal_id']!=signal_id or not 0.<=timestamp_s-tracked['timestamp_s']<=30. or depth<=.1:return None
    ratio=tracked['depth']/depth;old=tracked['bbox']
    x1,x2=[x+(value-tracked['x'])*ratio for value in (old[0],old[2])]
    y1,y2=[y+(value-tracked['y'])*ratio for value in (old[1],old[3])]
    height,width=rgb.shape[:2]
    xa=max(0,min(width,int(x1)));xb=max(0,min(width,int(x2)+1))
    ya=max(0,min(height,int(y1)));yb=max(0,min(height,int(y2)+1))
    if xb<=xa or yb<=ya:return None
    state,confidence=classify_signal_crop(rgb[ya:yb,xa:xb])
    if state=='GREEN' and (tracked.get('state')!='GREEN' or timestamp_s-tracked['timestamp_s']>.5):
        state,confidence='UNKNOWN',0.
    return dict(bbox=[x1,y1,x2,y2],state=state,color_confidence=confidence,
        source='projected_detected_lamp_current_rgb',position_age_s=timestamp_s-tracked['timestamp_s'])


class MapSignalObserver:
    def __init__(self,world_map,camera_sensor,weights,device='cuda:0',debug_directory=None,static_map_path=None,state_checkpoint=None):
        if static_map_path is None:raise ValueError('A verified static stop-line map is required for traffic control')
        self.world_map=world_map
        self.camera=camera_sensor
        self.device=device
        self.detector=None;self.classes=[]
        if state_checkpoint is None:
            from ultralytics import YOLO
            self.detector=YOLO(str(weights))
            self.classes=[i for i,name in self.detector.names.items() if name.replace('_',' ')=='traffic light']
            if not self.classes:raise ValueError('Detector has no traffic-light class')
        self.last_observation=None
        self.last_frame=None
        self.debug_directory=debug_directory
        self.tracked_signal=None
        self.last_debug_bucket=None
        self.active_landmark=None
        self.stop_lines=None
        self.static_heads={}
        self.state_classifier=None
        if static_map_path is not None:
            import json,hashlib
            from pathlib import Path
            document=json.loads(Path(static_map_path).read_text())
            if document['opendrive_sha256']!=hashlib.sha256(world_map.to_opendrive().encode()).hexdigest():
                raise ValueError('Static traffic-control map does not match the loaded OpenDRIVE map')
            self.stop_lines=document['signals']
            self.static_heads=document.get('heads',{})
        if state_checkpoint is not None:
            from control.signal_lamp_state import LampStateClassifier
            if not self.static_heads:
                raise ValueError('Lamp-state model requires a static map with lamp-head geometry')
            self.state_classifier=LampStateClassifier(state_checkpoint,device)

    def _observe_static_heads(self,signal_id,rgb,inverse_camera):
        from control.signal_lamp_state import project_head,head_crop
        height,width=rgb.shape[:2]
        projected=[];crops=[]
        for index,vertices in enumerate(self.static_heads.get(signal_id,[])):
            box=project_head(vertices,inverse_camera,width,height,float(self.camera.attributes['fov']))
            if box is None:continue
            crop=head_crop(rgb,box)
            if crop is None:continue
            projected.append(dict(head_index=index,bbox=box));crops.append(crop)
        observations=[dict(**location,**state) for location,state in zip(projected,self.state_classifier.predict(crops))]
        valid=[item for item in observations if item['state']!='UNKNOWN']
        if not valid or len({item['state'] for item in valid})!=1:
            return 'UNKNOWN',0.,observations
        # Multiple heads may be occluded, but any conflicting visible head rejects release.
        state=valid[0]['state'];confidence=min(item['color_confidence'] for item in valid)
        return state,confidence,observations

    def observe(self,ego,rgb,*,frame,timestamp_s,camera_inverse=None):
        if frame==self.last_frame and self.last_observation is not None:
            return dict(self.last_observation)
        self.last_frame=frame
        transform=ego.get_transform();loc=transform.location;forward=transform.get_forward_vector()
        front=[getattr(loc,axis)+getattr(forward,axis)*ego.bounding_box.extent.x for axis in ('x','y','z')]
        waypoint=self.world_map.get_waypoint(ego.get_location())
        landmarks=waypoint.get_landmarks_of_type(60.,'1000001',False) if waypoint else []
        applicable=[]
        for landmark in landmarks:
            effective=landmark.waypoint
            if effective is None:continue
            validities=landmark.get_lane_validities()
            spans=[(int(v[0]),int(v[1])) if isinstance(v,(tuple,list)) else (int(v.from_lane),int(v.to_lane)) for v in validities]
            if spans and not any(min(a,b)<=effective.lane_id<=max(a,b) for a,b in spans):continue
            if landmark.distance>=0.:applicable.append(landmark)
        prior=getattr(self,'active_landmark',None)
        if prior is not None and self.stop_lines is not None:
            entries=[item for item in self.stop_lines.get(str(prior.id),[]) if
                (item['road_id'],item['section_id'],item['lane_id'])==
                (prior.waypoint.road_id,prior.waypoint.section_id,prior.waypoint.lane_id)]
            on_approach=any(
                abs((front[0]-item['position'][0])*item['forward'][1]-(front[1]-item['position'][1])*item['forward'][0])<2.5
                and forward.x*item['forward'][0]+forward.y*item['forward'][1]>.7 for item in entries)
            remaining=mapped_stop_distance(entries,prior.waypoint,front)
            if on_approach and remaining is not None and remaining>=-1.:
                applicable=[prior]
            else:self.active_landmark=None
        if not applicable:
            result=dict(applicable=False,state='UNKNOWN',timestamp_s=timestamp_s,source='rgb_static_opendrive')
            self.last_observation=result
            return result
        landmark=min(applicable,key=lambda item:item.distance)
        stop_distance=float(landmark.distance)-float(ego.bounding_box.extent.x)
        stop_source='unverified_signal_reference_point'
        if self.stop_lines is not None:
            mapped=mapped_stop_distance(self.stop_lines.get(str(landmark.id),[]),waypoint,front)
            if mapped is None and landmark.waypoint is not None:
                mapped=mapped_stop_distance(self.stop_lines.get(str(landmark.id),[]),landmark.waypoint,front)
            stop_distance=mapped if mapped is not None else 0.
            stop_source='static_hdmap_stop_line' if mapped is not None else 'stop_line_association_missing'
            if mapped is not None and mapped>=-1.:self.active_landmark=landmark
            if mapped is not None and mapped < -1.:
                self.last_observation=dict(applicable=False,state='UNKNOWN',timestamp_s=timestamp_s,source='passed_mapped_stop_line')
                return dict(self.last_observation)
        if hasattr(rgb,'detach'):rgb=rgb.detach().cpu().numpy()
        if rgb.ndim==3 and rgb.shape[0]==3:rgb=rgb.transpose(1,2,0)
        if rgb.max()<=1.:rgb=rgb*255.
        rgb=np.ascontiguousarray(rgb.astype(np.uint8))
        height,width=rgb.shape[:2]
        pose_verified=camera_inverse is not None
        inv=np.asarray(camera_inverse if pose_verified else self.camera.get_transform().get_inverse_matrix())
        pose_verified=pose_verified and inv.shape==(4,4) and np.isfinite(inv).all()
        if inv.shape!=(4,4) or not np.isfinite(inv).all():inv=np.eye(4)
        loc=landmark.transform.location
        point=inv@np.asarray([loc.x,loc.y,loc.z,1.])
        focal=width/(2.*math.tan(math.radians(float(self.camera.attributes['fov']))/2.))
        expected_x=width/2.+focal*point[1]/point[0] if point[0]>.1 else None
        expected_y=height/2.-focal*point[2]/point[0] if point[0]>.1 else None
        state='UNKNOWN';confidence=0.;boxes=[]
        roi_bounds=None
        if getattr(self,'state_classifier',None) is not None:
            if pose_verified:state,confidence,boxes=self._observe_static_heads(str(landmark.id),rgb,inv)
        elif expected_x is not None and -width*.2<=expected_x<=width*1.2:
            # OpenDRIVE landmarks locate a signal assembly, not each small lamp.
            radius=max(24.,focal*6./point[0])
            xa=max(0,int(expected_x-radius));xb=min(width,int(expected_x+radius)+1)
            ya=max(0,int(expected_y-focal*14./point[0]));yb=min(height,int(expected_y+focal*4./point[0])+1)
            if xb<=xa or yb<=ya:
                self.last_observation=dict(applicable=True,signal_id=str(landmark.id),state='UNKNOWN',confidence=0.,
                    stop_distance_m=stop_distance,timestamp_s=timestamp_s,
                    source='rgb_static_opendrive_out_of_view',truth_state_access=False)
                return dict(self.last_observation)
            roi_bounds=[xa,ya,xb,yb]
            output=self.detector.predict(source=np.ascontiguousarray(rgb[ya:yb,xa:xb,::-1]),imgsz=640,
                conf=.15,classes=self.classes,device=self.device,verbose=False)[0]
            for box in output.boxes:
                x1,y1,x2,y2=box.xyxy[0].detach().cpu().tolist()
                x1+=xa;x2+=xa;y1+=ya;y2+=ya
                crop=rgb[max(0,int(y1)):min(height,int(y2)+1),max(0,int(x1)):min(width,int(x2)+1)]
                color,score=classify_signal_crop(crop)
                boxes.append(dict(bbox=[x1,y1,x2,y2],state=color,color_confidence=score,detector_confidence=float(box.conf[0])))
            valid=[b for b in boxes if b['state']!='UNKNOWN']
            if valid and len({b['state'] for b in valid})==1:
                best=max(valid,key=lambda b:b['color_confidence']);state=best['state'];confidence=best['color_confidence']
                self.tracked_signal=dict(signal_id=str(landmark.id),state=state,bbox=best['bbox'],x=expected_x,y=expected_y,
                    depth=float(point[0]),timestamp_s=timestamp_s)
            elif not boxes and self.tracked_signal is not None:
                tracked=observe_tracked_lamp(rgb,self.tracked_signal,signal_id=str(landmark.id),
                    x=expected_x,y=expected_y,depth=float(point[0]),timestamp_s=timestamp_s)
                if tracked is not None:
                    state=tracked['state'];confidence=tracked['color_confidence'];boxes.append(tracked)
        if stop_source=='stop_line_association_missing':state,confidence='UNKNOWN',0.
        result=dict(applicable=True,signal_id=str(landmark.id),state=state,confidence=confidence,
            stop_distance_m=stop_distance,stop_line_source=stop_source,timestamp_s=timestamp_s,
            source='rgb_lamp_net_static_head' if getattr(self,'state_classifier',None) is not None else 'rgb_detector_color_static_opendrive',truth_state_access=False,
            camera_pose_source='image_exposure' if pose_verified else 'unverified_current_pose',
            expected_pixel_x=expected_x,expected_pixel_y=expected_y,roi_bounds=roi_bounds,
            landmark_depth_m=float(point[0]),detections=boxes)
        self.last_observation=result
        if self.debug_directory and frame//10!=self.last_debug_bucket:
            self.last_debug_bucket=frame//10
            from pathlib import Path
            import json
            from PIL import Image
            root=Path(self.debug_directory);root.mkdir(parents=True,exist_ok=True)
            Image.fromarray(rgb).save(root/f'{frame}.png')
            (root/f'{frame}.json').write_text(json.dumps(result,indent=2))
        return result
