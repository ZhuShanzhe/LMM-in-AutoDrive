"""Causal physical-radar tracks in world coordinates; no actor truth inputs."""

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from scene_understanding.core.target_history import TargetHistory

SCHEMA = 'tracked_route_observation/1.0'


def capture_ego_state(actor):
    """Read only ego telemetry; shared by collection and online inference."""
    transform=actor.get_transform();velocity=actor.get_velocity()
    acceleration=actor.get_acceleration();angular=actor.get_angular_velocity()
    return dict(x=float(transform.location.x),y=float(transform.location.y),
        vx_mps=float(velocity.x),vy_mps=float(velocity.y),
        ax_mps2=float(acceleration.x),ay_mps2=float(acceleration.y),
        yaw_deg=float(transform.rotation.yaw),yaw_rate_dps=float(angular.z),
        half_length_m=float(actor.bounding_box.extent.x),
        corridor_half_width_m=float(actor.bounding_box.extent.y)+.35)


class RouteGeometry:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 2:
            raise ValueError('Route needs at least two XY points')
        if not np.isfinite(self.points).all():
            raise ValueError('Nonfinite route')
        delta = np.diff(self.points, axis=0)
        length = np.linalg.norm(delta, axis=1)
        keep = np.r_[True, length > 1e-4]
        self.points = self.points[keep]
        self.delta = np.diff(self.points, axis=0)
        self.length = np.linalg.norm(self.delta, axis=1)
        if not len(self.length):
            raise ValueError('Degenerate route')
        self.prefix = np.r_[0., np.cumsum(self.length)]

    def project(self, point):
        p = np.asarray(point, dtype=float)
        fractions = np.clip(np.sum((p-self.points[:-1])*self.delta, axis=1)/self.length**2, 0., 1.)
        residual = p-(self.points[:-1]+fractions[:,None]*self.delta)
        index = int(np.argmin(np.linalg.norm(residual, axis=1)))
        tangent = self.delta[index]/self.length[index]
        return dict(s=float(self.prefix[index]+fractions[index]*self.length[index]),
            lateral=float(tangent[0]*residual[index,1]-tangent[1]*residual[index,0]),
            distance=float(np.linalg.norm(residual[index])), tangent=tangent)


@dataclass
class Track:
    identity: str
    state: np.ndarray
    covariance: np.ndarray
    time: float
    measured_at: float
    hits: int = 1
    acceleration_mps2: float = 0.


class TrackedObservation:
    """Track radar surfaces, not semantically classified vehicle instances."""
    def __init__(self, max_age_s=.25, coast_s=.5, max_tracks=64):
        self.max_age_s, self.coast_s, self.max_tracks = max_age_s, coast_s, max_tracks
        self.reset()

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.last_frame = -1
        self.last_time = None
        self.selected = None
        self.target_history = TargetHistory(retention_s=30.)

    @staticmethod
    def _clusters(radar):
        pose = radar['measurement_pose']
        yaw = math.radians(pose['yaw_deg'])
        points = []
        for item in radar['tracking_points']:
            try:
                d, a, h, v = (float(item[k]) for k in ('distance_m','azimuth_deg','altitude_deg','relative_velocity_mps'))
            except (KeyError,TypeError,ValueError):
                continue
            if not all(math.isfinite(x) for x in (d,a,h,v)) or not .5 <= d <= 80:
                continue
            angle = yaw+math.radians(a)
            distance = d*math.cos(math.radians(h))
            points.append([pose['x']+distance*math.cos(angle), pose['y']+distance*math.sin(angle), v,
                           math.cos(angle), math.sin(angle)])
        if not points:
            return []
        values = np.asarray(points)
        # Connected components join nearby returns with compatible Doppler velocity.
        adjacency = (np.linalg.norm(values[:,None,:2]-values[None,:,:2],axis=-1)<1.5)
        adjacency &= abs(values[:,None,2]-values[None,:,2])<2.
        remaining = set(range(len(values))); clusters = []
        while remaining:
            stack = [remaining.pop()]; group = []
            while stack:
                i = stack.pop(); group.append(i)
                neighbors = remaining.intersection(np.flatnonzero(adjacency[i]).tolist())
                remaining.difference_update(neighbors); stack.extend(neighbors)
            if len(group) < 2:
                continue
            block = values[group]
            if np.ptp(block[:,:2],axis=0).max() > 6.:
                continue  # Extended surfaces are not compact tracked entities.
            median = np.median(block,axis=0)
            los = median[3:5]; los /= max(np.linalg.norm(los),1e-6)
            clusters.append((median[:2], float(median[2]), los))
        return clusters

    @staticmethod
    def _predict(track, timestamp):
        dt = max(0.,timestamp-track.time)
        transition = np.eye(4); transition[0,2] = transition[1,3] = dt
        noise = np.array([[dt**2/2,0],[0,dt**2/2],[dt,0],[0,dt]])
        track.state = transition@track.state
        track.covariance = transition@track.covariance@transition.T + 4.*noise@noise.T + np.eye(4)*1e-5
        track.time = timestamp

    @staticmethod
    def _correct(track, position, radial, los, ego_velocity, timestamp):
        old_v = track.state[2:].copy()
        h = np.array([[1,0,0,0],[0,1,0,0],[0,0,los[0],los[1]]])
        z = np.r_[position, radial+np.dot(ego_velocity,los)]
        r = np.diag([.5**2,.5**2,.35**2])
        innovation = h@track.covariance@h.T+r
        gain = np.linalg.solve(innovation,h@track.covariance).T
        track.state += gain@(z-h@track.state)
        residual = np.eye(4)-gain@h
        track.covariance = residual@track.covariance@residual.T+gain@r@gain.T
        dt = timestamp-track.measured_at
        if dt > .001:
            acceleration = float(np.dot(track.state[2:]-old_v,los)/dt)
            track.acceleration_mps2 = .7*track.acceleration_mps2+.3*np.clip(acceleration,-10.,10.)
        track.measured_at = timestamp
        track.hits += 1

    def update(self, radar, *, ego, route_points, now_s, frame, coverage_clear=False, requested_track_id=None):
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError('Nonfinite observation time')
        if self.last_time is not None and now < self.last_time:
            self.reset()
        elif self.last_time is not None and now-self.last_time>1.:
            self.tracks=[];self.selected=None;self.last_frame=-1
        self.last_time = now
        try:
            route = RouteGeometry(route_points)
        except (TypeError,ValueError):
            route = None
        fresh = False; reason = 'missing_or_stale_radar'
        stamp = radar.get('measurement_timestamp_s')
        sensor_frame = radar.get('sensor_frame',-1)
        pose = radar.get('measurement_pose',{})
        try:
            fresh = (math.isfinite(float(stamp)) and 0 <= now-float(stamp) <= self.max_age_s
                and 0 <= sensor_frame <= frame and isinstance(radar.get('tracking_points'),list)
                and all(math.isfinite(float(pose[k])) for k in ('x','y','yaw_deg')))
            if self.tracks and fresh and float(stamp)<max(t.time for t in self.tracks)-1e-6:
                fresh = False
        except (ValueError,TypeError,KeyError):
            fresh = False
        velocity = np.array([ego['vx_mps'],ego['vy_mps']],float)
        if not np.isfinite([ego['x'],ego['y'],*velocity]).all():
            raise ValueError('Invalid ego pose or velocity')
        self.tracks = [t for t in self.tracks if now-t.measured_at<=self.coast_s]
        new_measurement = fresh and sensor_frame > self.last_frame
        if new_measurement:
            stamp = float(stamp)
            clusters = self._clusters(radar)
            for track in self.tracks:
                self._predict(track,stamp)
            cost = np.full((len(self.tracks),len(clusters)),1e6)
            if clusters:
                positions = np.stack([c[0] for c in clusters])
                radial = np.asarray([c[1] for c in clusters])
                los = np.stack([c[2] for c in clusters])
                for i,track in enumerate(self.tracks):
                    offset = positions-track.state[:2]
                    solved = np.linalg.solve(track.covariance[:2,:2]+np.eye(2)*.25,offset.T).T
                    mahal = np.sum(offset*solved,axis=1)
                    doppler = abs(los@(track.state[2:]-velocity)-radial)
                    usable = (np.linalg.norm(offset,axis=1)<4.) & (mahal<16.) & (doppler<4.)
                    cost[i,usable] = (mahal+doppler**2)[usable]
            assigned = set()
            if cost.size:
                rows,cols = linear_sum_assignment(cost)
                for i,j in zip(rows,cols):
                    if cost[i,j]>=1e6:
                        continue
                    self._correct(self.tracks[i],*clusters[j],velocity,stamp)
                    assigned.add(j)
            for j,(position,radial,los) in enumerate(clusters):
                if j in assigned or len(self.tracks)>=self.max_tracks:
                    continue
                initial_velocity = velocity+radial*los
                self.tracks.append(Track(f'radar-{self.next_id}',np.r_[position,initial_velocity],
                    np.diag([.5,.5,9.,9.]),stamp,stamp))
                self.next_id += 1
            self.last_frame = sensor_frame
        entities = []
        ego_projection = route.project([ego['x'],ego['y']]) if route else None
        for track in self.tracks:
            dt = max(0.,now-track.time)
            position = track.state[:2]+track.state[2:]*dt
            projected = route.project(position) if route else None
            gap = projected['s']-ego_projection['s']-ego.get('half_length_m',2.4) if projected else None
            corridor = bool(projected and ego_projection['distance']<3.
                and projected['distance']<=ego.get('corridor_half_width_m',1.5) and gap>0.)
            age = now-track.measured_at
            directly_measured = bool(fresh and sensor_frame==frame and abs(track.measured_at-float(stamp))<1e-6)
            status = 'TRACKED' if track.hits>=3 and directly_measured else 'PREDICTED' if track.hits>=3 else 'TENTATIVE'
            closing = float(np.dot(velocity,ego_projection['tangent'])-np.dot(track.state[2:],projected['tangent'])) if projected else None
            entities.append(dict(track_id=track.identity,kind='UNCLASSIFIED_RADAR_SURFACE',
                source='physical_radar',status=status,measured_at_s=track.measured_at,age_s=age,
                position_xy_m=position.tolist(),velocity_xy_mps=track.state[2:].tolist(),
                radial_acceleration_mps2=float(track.acceleration_mps2),
                position_std_m=float(np.sqrt(np.max(np.diag(track.covariance)[:2])+dt**2*np.max(np.diag(track.covariance)[2:]))),
                route_gap_m=float(gap) if gap is not None else None,
                route_lateral_m=projected['lateral'] if projected else None,
                in_route_corridor=corridor,closing_speed_mps=closing,
                closing_ttc_s=gap/closing if corridor and closing>.5 else None,
                distance_reference='observed_surface_to_ego_front_along_route'))
        eligible = [x for x in entities if x['in_route_corridor'] and x['status']!='TENTATIVE']
        nearest = min(eligible,key=lambda x:x['route_gap_m'],default=None)
        previous = next((x for x in eligible if x['track_id']==self.selected),None)
        # Continuity reference and closest obstacle have different responsibilities.
        # A new obstacle must be reported immediately, not overwrite an active target.
        selected = previous or nearest
        if requested_track_id is not None:
            selected = next((x for x in eligible if x['track_id']==requested_track_id),None)
        self.selected = selected['track_id'] if selected else None
        status = selected['status'] if selected else 'CLEAR' if fresh and coverage_clear and route else 'UNKNOWN'
        if route is None:
            status = 'UNKNOWN';reason = 'invalid_route_geometry'
        elif selected:
            reason = 'confirmed_surface_track' if status=='TRACKED' else 'short_occlusion_prediction'
        elif fresh:
            reason = 'coverage_confirmed_clear' if status=='CLEAR' else 'no_confirmed_target_not_proof_of_clear_road'
        if new_measurement:
            # Cache the measurement-time estimate, never the extrapolation to decision time.
            measured = [dict(track_id=t.identity,kind='UNCLASSIFIED_RADAR_SURFACE',
                source='physical_radar',status='TRACKED',measured_at_s=t.measured_at,
                position_xy_m=t.state[:2].tolist(),velocity_xy_mps=t.state[2:].tolist())
                for t in self.tracks if t.hits>=3 and abs(t.measured_at-stamp)<1e-6]
            self.target_history.update('front_radar',measured,stamp)
        history=self.target_history.snapshot('front_radar',now)
        return dict(schema_version=SCHEMA,frame=frame,timestamp_s=now,status=status,reason=reason,
            radar_fresh=fresh,selected_track_id=self.selected,selected=selected,entities=entities,
            nearest_obstacle=nearest,selected_role='CONTINUITY_REFERENCE_NOT_CLEARANCE_AUTHORITY',
            requested_track_resolved=requested_track_id is None or self.selected==requested_track_id,
            target_history=history,
            ego=dict(ego),capabilities=dict(semantic_classification=False,front_radar_tracking=True,
                all_around_coverage=False),invalid_route=route is None)
