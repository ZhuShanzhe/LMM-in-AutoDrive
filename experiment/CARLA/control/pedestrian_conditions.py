"""Identity-bound pedestrian clearance from RGB tracks and static road geometry."""
import math
import numpy as np


def corridor_position(point, route):
    best = None
    offset = 0.
    for first, last in zip(route[:-1], route[1:]):
        delta = np.asarray(last[:2])-first[:2]
        length = float(np.linalg.norm(delta))
        if length < .01:
            continue
        fraction = float(np.clip(np.dot(np.asarray(point[:2])-first[:2], delta)/length**2, 0., 1.))
        distance = float(np.linalg.norm(np.asarray(point[:2])-first[:2]-fraction*delta))
        value = (distance, offset+fraction*length)
        if best is None or value[0] < best[0]:
            best = value
        offset += length
    return best


def ground_footpoint(box, inverse_camera, image_size, fov, road):
    """Intersect a tracked footpoint ray with the locally supported road plane."""
    if len(road) < 2 or inverse_camera is None:
        return None
    width, height = image_size
    x1, _, x2, y2 = box
    if not 0 <= x1 < x2 <= width or not 0 < y2 < height-2:
        return None
    matrix = np.asarray(inverse_camera, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    camera = np.linalg.inv(matrix)
    focal = width/(2*math.tan(math.radians(fov)/2))
    ray = camera[:3, :3] @ np.array([1., ((x1+x2)/2-width/2)/focal, -(y2-height/2)/focal])
    origin = camera[:3, 3]
    surface = np.asarray(road, dtype=float)
    lengths = np.linalg.norm(np.diff(surface[:, :2], axis=0), axis=1)
    if (not np.isfinite(surface).all() or (lengths < .01).any()
            or (np.abs(np.diff(surface[:, 2]))/lengths > .3).any() or ray[2] >= -.03):
        return None
    starts = surface[:-1]
    delta = surface[1:]-starts
    squared = lengths**2
    height_at_foot = float(surface[0, 2])
    for _ in range(8):
        scale = (height_at_foot-origin[2])/ray[2]
        point = origin+scale*ray
        if not .5 <= scale <= 40. or not np.isfinite(point).all():
            return None
        relative = point[:2]-starts[:, :2]
        fractions = np.clip(np.sum(relative*delta[:, :2],axis=1)/squared, 0., 1.)
        distances = np.linalg.norm(relative-fractions[:, None]*delta[:, :2],axis=1)
        heights = starts[:, 2]+fractions*delta[:, 2]
        # Preserve Python min(tuple)'s secondary height ordering at exact ties.
        index = np.lexsort((heights, distances))[0]
        distance, new_height = float(distances[index]), float(heights[index])
        if abs(distance-8.) <= 1e-8 or abs(abs(new_height-height_at_foot)-.02) <= 1e-8:
            candidates = []
            for first, last, length in zip(surface[:-1], surface[1:], lengths):
                fraction = float(np.clip(np.dot(point[:2]-first[:2], last[:2]-first[:2])/length**2, 0., 1.))
                d = float(np.linalg.norm(point[:2]-first[:2]-fraction*(last[:2]-first[:2])))
                candidates.append((d,first[2]+fraction*(last[2]-first[2])))
            distance, new_height = min(candidates)
        if distance > 8.:
            return None
        if abs(new_height-height_at_foot) < .02:
            return point.tolist()
        height_at_foot = new_height
    return None


class PedestrianClearanceMemory:
    def __init__(self):
        self.request_id = None
        self.targets = {}
        self.last_time = None
        self.last_frame = None
        self.completed_events = {}
        self.event_counter = 0

    def update(self, request_id, tracks, route, *, half_width, timestamp_s, sensor_frame):
        if request_id != self.request_id:
            self.request_id = request_id
            self.targets = {}
            self.last_time = self.last_frame = None
            self.completed_events = {}
            self.event_counter = 0
        if sensor_frame == self.last_frame:
            return self.result(timestamp_s)
        continuous = self.last_time is not None and 0 < timestamp_s-self.last_time <= .25
        self.last_time, self.last_frame = timestamp_s, sensor_frame
        observed = set()
        for track in tracks:
            position = corridor_position(track['position'], route)
            if position is None:
                continue
            lateral, forward = position
            identity = track['track_id']
            observed.add(identity)
            inside = lateral <= half_width+.5 and forward <= 30.
            if inside:
                old=self.targets.get(identity) or {}
                event_id=old.get('event_id')
                if not event_id or identity in self.completed_events:
                    self.event_counter+=1
                    event_id=f'corridor-crossing-{self.event_counter}'
                self.completed_events.pop(identity,None)
                self.targets[identity] = dict(last_seen_s=timestamp_s, outside_since_s=None,
                    status='IN_CORRIDOR', position=track['position'],verified_exit_at_s=None,
                    event_id=event_id)
            elif identity in self.targets:
                target = self.targets[identity]
                uninterrupted = continuous and timestamp_s-target['last_seen_s'] <= .25
                since = target['outside_since_s'] if uninterrupted else None
                since = timestamp_s if since is None else since
                target.update(last_seen_s=timestamp_s, outside_since_s=since,
                    status='OBSERVED_OUTSIDE' if timestamp_s-since >= .5 else 'EXIT_CONFIRMING',
                    position=track['position'])
                if target['status']=='OBSERVED_OUTSIDE':
                    target['verified_exit_at_s']=timestamp_s
                    target.pop('position',None)
                    self.completed_events.setdefault(identity,dict(event_id=target['event_id'],
                        identity=identity, fact='OBSERVED_CORRIDOR_EXIT', completed_at_s=timestamp_s,
                        sensor_frame=sensor_frame, source='continuous_identity_exit_observation',
                        event_status='COMPLETED', semantic_scope='observed_driving_corridor_crossing'))
        for identity, target in list(self.targets.items()):
            if identity not in observed:
                target['outside_since_s'] = None
                target['status'] = 'OCCLUDED'
            if timestamp_s-target['last_seen_s'] > 30.:
                target.pop('position', None)
                target['status'] = 'EXPIRED_UNKNOWN'
                if identity in self.completed_events:
                    del self.targets[identity]
        while len(self.completed_events)>128:
            self.completed_events.pop(next(iter(self.completed_events)))
        return self.result(timestamp_s)

    def result(self, timestamp_s, *, identities=None, corridor_clear=False):
        selected=set(self.targets)|set(self.completed_events) if identities is None else set(identities)
        targets={key:self.targets[key] for key in selected if key in self.targets}
        facts={key:self.completed_events[key] for key in selected if key in self.completed_events}
        def verified(target):
            if target['status']=='IN_CORRIDOR':
                return False
            if target['status']=='OBSERVED_OUTSIDE' and timestamp_s-target['last_seen_s'] <= .25:
                return True
            exit_at=target.get('verified_exit_at_s')
            return corridor_clear and isinstance(exit_at,(int,float)) and 0<=timestamp_s-exit_at
        proven=all(key in facts or key in targets and verified(targets[key]) for key in selected)
        # Historical event completion never authorizes motion into a new obstacle.
        clear = bool(selected) and proven and (corridor_clear or
            bool(targets) and all(verified(t) for t in targets.values()))
        return dict(satisfied=clear, can_proceed=clear, status='CLEAR' if clear else 'UNKNOWN',
            reason='tracked_pedestrians_observed_outside_corridor' if clear
                else 'current_corridor_clearance_required' if selected and proven
                else 'pedestrian_identity_not_observed' if not selected else 'pedestrian_exit_not_confirmed',
            event_completion_verified=bool(selected and proven),
            target_evidence={key: dict(value) for key, value in targets.items()},
            completed_event_facts={key:dict(value) for key,value in facts.items()})


class PedestrianConditionObserver:
    def __init__(self, weights, device):
        from pathlib import Path
        from scene_understanding.realtime_perception.ultralytics_backend import UltralyticsTrafficDetector
        from scene_understanding.realtime_perception.tracker import ByteTrackAdapter
        self.weights, self.device = Path(weights), device
        self.detector = UltralyticsTrafficDetector(self.weights, device=device, image_size=1280)
        self._vehicle_detector = None
        self.tracker = ByteTrackAdapter(frame_rate=20)
        self.memory = PedestrianClearanceMemory()
        self.request_id = None
        self.last_sensor_frame = None
        self.observations = []
        self.pedestrian_boxes = []
        self.condition_targets = {}
        self._front_detection_cache = None

    def front_detection_cache(self, sensor_frame):
        value=self._front_detection_cache
        return value if value is not None and value['sensor_frame']==sensor_frame else None

    def vehicle_detector(self):
        if self._vehicle_detector is None:
            from scene_understanding.realtime_perception.ultralytics_backend import UltralyticsTrafficDetector
            self._vehicle_detector = UltralyticsTrafficDetector(self.weights,
                device=self.device, image_size=640)
        return self._vehicle_detector

    def observe(self, document, rgb, inverse_camera, road, *, fov, frame_id,
                sensor_frame, observed_at_s, timestamp_s, half_width, corridor_evidence=None,
                crosswalk_polygons=None, cloud=None, views=None):
        conditions = [s for s in document['intent']['steps'] if s.get('action') in {'WAIT', 'CHECK', 'CONFIRM', 'YIELD'}
            and (s.get('parameters') or {}).get('condition') in {'PEDESTRIAN_CLEAR', 'PASSENGERS_CLEAR', 'CROSSWALK_CLEAR'}]
        fresh = (isinstance(observed_at_s, (int, float)) and math.isfinite(observed_at_s)
                 and 0 <= timestamp_s-observed_at_s <= .25 and rgb is not None and inverse_camera is not None)
        proof = dict(satisfied=False, status='UNKNOWN', reason='pedestrian_observation_unavailable')
        if fresh:
            if sensor_frame != self.last_sensor_frame:
                captures = {'front': (rgb, inverse_camera, fov)}
                for name, capture in (views or {}).items():
                    if name != 'front' and capture is not None and len(capture) == 3:
                        image, pose, view_fov = capture
                        if image is not None and pose is not None and isinstance(view_fov, (float, int)):
                            captures[name] = capture
                names = list(captures)
                images = [np.ascontiguousarray(captures[name][0][:, :, ::-1]) for name in names]
                if len(images) > 1:
                    batches, _ = self.detector.detect_many(images)
                    if len(batches) != len(names):
                        raise RuntimeError('Pedestrian detector batch does not match camera views')
                else:
                    batches = [self.detector.detect(images[0])[0]]
                detections = batches[0]
                self._front_detection_cache=dict(sensor_frame=sensor_frame,image=rgb,
                    detections=tuple(detections),source='synchronized_normal_front_capture')
                self.pedestrian_boxes=[d.bbox_xyxy for d in detections if d.category=='pedestrian']
                self.observations = []
                projection_diagnostics = []
                from .pedestrian_multiview import (clothing_descriptor, measured_pedestrian_point,
                    merge_cross_view_observations)
                if not hasattr(self, '_view_trackers'):
                    self._view_trackers = {'front': self.tracker}
                for name, view_detections in zip(names, batches):
                    image, pose, view_fov = captures[name]
                    if name not in self._view_trackers:
                        from scene_understanding.realtime_perception.tracker import ByteTrackAdapter
                        self._view_trackers[name] = ByteTrackAdapter(frame_rate=20)
                    tracks = self._view_trackers[name].update(
                        [d for d in view_detections if d.category == 'pedestrian'])
                    for track in tracks:
                        size = (image.shape[1], image.shape[0])
                        point = measured_pedestrian_point(cloud, pose, size, view_fov,
                            track['bbox_xyxy'], road)
                        ray_point = ground_footpoint(track['bbox_xyxy'], pose, size, view_fov, road)
                        source = 'same_frame_rgb_lidar_body'
                        if point is None:
                            point = ray_point
                            source = 'static_road_footpoint'
                        elif ray_point is not None and np.linalg.norm(
                                np.asarray(point)[:2]-np.asarray(ray_point)[:2]) > 2.:
                            point = None
                            source = 'inconsistent_rgb_lidar_depth'
                        if point is not None:
                            self.observations.append(dict(track_id=f"{name}:{track['track_id']}",
                                view=name, position=point, age_frames=track['age_frames'],
                                appearance=clothing_descriptor(image, track['bbox_xyxy']), source=source))
                        projection_diagnostics.append(dict(track_id=track['track_id'], view=name,
                            bbox_xyxy=list(track['bbox_xyxy']), age_frames=track['age_frames'],
                            footpoint=point, measurement_source=source))
                self.observations = merge_cross_view_observations(self.observations)
                from .pedestrian_identity import PedestrianIdentityAssociation
                if not hasattr(self, '_identity_association'):
                    self._identity_association = PedestrianIdentityAssociation()
                self.observations = self._identity_association.update(self.observations, observed_at_s)
                self._save_diagnostic(rgb, inverse_camera, road, fov, sensor_frame,
                    observed_at_s, projection_diagnostics)
                self.last_sensor_frame = sensor_frame
            proof = self.memory.update('rgb-corridor-session', self.observations, road,
                half_width=half_width, timestamp_s=observed_at_s, sensor_frame=sensor_frame)
        result = {}
        for step in conditions:
            key=(document['request_id'],step['step_id'])
            if not self.condition_targets.get(key):
                self.condition_targets[key]=[identity for identity,target in self.memory.targets.items()
                    if 0<=timestamp_s-target['last_seen_s']<=30.
                    and (step.get('action')!='YIELD' or target['status']=='IN_CORRIDOR')]
                if step.get('action')!='YIELD':
                    self.condition_targets[key].extend(identity for identity in self.memory.completed_events
                        if identity not in self.condition_targets[key])
            for identity,target in self.memory.targets.items():
                if target['status']=='IN_CORRIDOR' and identity not in self.condition_targets[key]:
                    self.condition_targets[key].append(identity)
            while len(self.condition_targets)>128:
                self.condition_targets.pop(next(iter(self.condition_targets)))
            if fresh:
                proof=self.memory.result(timestamp_s,identities=self.condition_targets[key],
                    corridor_clear=bool(corridor_evidence and corridor_evidence.get('satisfied') is True
                        and corridor_evidence.get('valid') is True
                        and corridor_evidence.get('frame_id')==frame_id))
            parameters=step.get('parameters') or {}
            explicit_target = any(value is not None for value in (
                step.get('target'),step.get('target_ref'),parameters.get('target'),parameters.get('target_ref')))
            condition=step['parameters']['condition']
            supported=condition=='PEDESTRIAN_CLEAR'
            if condition=='CROSSWALK_CLEAR':
                proof=dict(satisfied=False,can_proceed=False,status='UNKNOWN',reason='crosswalk_region_not_observed')
                if fresh and crosswalk_polygons is not None:
                    from .crosswalk_conditions import observed_crosswalk_clearance
                    proof=observed_crosswalk_clearance(crosswalk_polygons,road,cloud,inverse_camera,
                        (rgb.shape[1],rgb.shape[0]),fov,getattr(self,'pedestrian_boxes',[]))
                supported=proof.get('status') in {'CLEAR','OCCUPIED'}
                corridor_confirmed=bool(corridor_evidence and corridor_evidence.get('valid') is True
                    and corridor_evidence.get('satisfied') is True and corridor_evidence.get('frame_id')==frame_id)
                proof['satisfied']=bool(proof['satisfied'] and corridor_confirmed)
                proof['can_proceed']=proof['satisfied']
                artifact_dir=getattr(self,'artifact_dir',None)
                saved=getattr(self,'saved_crosswalks',set())
                if proof['satisfied'] and not explicit_target and artifact_dir and key not in saved and len(saved)<128:
                    from pathlib import Path
                    from PIL import Image
                    import json
                    folder=Path(artifact_dir);folder.mkdir(parents=True,exist_ok=True)
                    stem=f'crosswalk_{sensor_frame}_{len(saved)}'
                    Image.fromarray(rgb).save(folder/f'{stem}.png')
                    np.savez_compressed(folder/f'{stem}.npz',world_cloud=cloud,inverse_camera=inverse_camera,
                        polygon_xyz=proof['region_polygon_xyz'])
                    (folder/f'{stem}.json').write_text(json.dumps(dict(proof,request_id=key[0],step_id=key[1],
                        sensor_frame=sensor_frame,observed_at_s=observed_at_s,image_size=[rgb.shape[1],rgb.shape[0]],fov=fov),indent=2))
                    proof['evidence_artifact_stem']=str(folder/stem)
                    saved.add(key);self.saved_crosswalks=saved
            valid = fresh and not explicit_target and supported
            if (not supported and condition!='CROSSWALK_CLEAR') or explicit_target:
                proof=dict(satisfied=False,can_proceed=False,status='UNKNOWN',
                    reason='role_or_region_binding_required',target_evidence={},completed_event_facts={},
                    event_completion_verified=False)
            result[step['step_id']] = dict(proof, valid=valid,
                satisfied=bool(valid and proof['satisfied']),can_proceed=bool(valid and proof['satisfied']), request_id=document['request_id'],
                step_id=step['step_id'], condition=step['parameters']['condition'], frame_id=frame_id,
                observed_at_s=observed_at_s, valid_until_s=observed_at_s+.25 if fresh else None,
                sensor_frame=sensor_frame, source='rgb_lidar_static_crosswalk_region' if condition=='CROSSWALK_CLEAR'
                    else 'rgb_bytetrack_static_road_identity_observer',
                semantic_scope='fully_projected_crosswalk_with_current_rgb_and_lidar_coverage' if condition=='CROSSWALK_CLEAR'
                    else 'previously_observed_pedestrians_in_driving_corridor',
                corridor_evidence=corridor_evidence,
                policy_truth_access=False)
        return result

    def _save_diagnostic(self, rgb, pose, road, fov, frame, timestamp, tracks):
        import os
        folder = os.environ.get('PEDESTRIAN_DIAGNOSTIC_DIR')
        if not folder:
            return
        from pathlib import Path
        import json
        from PIL import Image
        count = getattr(self, '_diagnostic_count', 0)
        if count >= 1600:
            return
        self._diagnostic_count = count + 1
        root = Path(folder)
        root.mkdir(parents=True, exist_ok=True)
        stem = f'pedestrian_{frame}'
        image = f'{stem}.jpg' if count % 5 == 0 else None
        if image:
            Image.fromarray(rgb).save(root / image, quality=90)
        with (root / 'observations.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(sensor_frame=frame, timestamp_s=timestamp,
                camera_inverse=np.asarray(pose).tolist(), road=np.asarray(road).tolist(),
                image_size=[rgb.shape[1], rgb.shape[0]], fov=fov,
                pedestrian_boxes=[list(box) for box in self.pedestrian_boxes],
                tracks=tracks, observations=self.observations, image=image))+'\n')
