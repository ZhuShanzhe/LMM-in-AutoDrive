"""Separate command submission evidence from subsequent physical response."""
import json
import math


def control_values(control):
    result = {key: float(getattr(control, key)) for key in ('throttle','brake','steer')}
    if any(not math.isfinite(v) for v in result.values()):
        raise ValueError('non-finite vehicle control')
    result.update({key: getattr(control, key) for key in ('hand_brake','reverse','manual_gear_shift','gear')})
    return result


class ControlExecutionJournal:
    def __init__(self, path):
        self.stream = path.open('x', encoding='utf-8')
        self.pending = None
        self.sequence = 0

    def _write(self, **record):
        self.stream.write(json.dumps(dict(schema_version='control_execution/1.0', **record),allow_nan=False)+'\n')
        self.stream.flush()

    def observe(self, world, ego):
        if self.pending is None:
            return
        snapshot=world.get_snapshot()
        if snapshot.frame <= self.pending['submission_frame']:
            return
        actor=snapshot.find(ego.id)
        if actor is None:
            self._write(status='OBSERVATION_UNAVAILABLE', **self.pending, observed_frame=snapshot.frame)
        else:
            velocity=actor.get_velocity()
            transform=actor.get_transform()
            self._write(status='PHYSICS_OBSERVED', **self.pending, observed_frame=snapshot.frame,
                observed_time_s=snapshot.timestamp.elapsed_seconds,
                speed_kmh=3.6*math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2),
                location={key:float(getattr(transform.location,key)) for key in ('x','y','z')},
                yaw_deg=float(transform.rotation.yaw),
                client_reported_control=control_values(ego.get_control()),
                interpretation='later physical state; not proof of task success or exact actuation latency')
        self.pending=None

    def apply(self, world, ego, control, decision_frame):
        self.observe(world,ego)
        if self.pending is not None:
            self._write(status='SUPERSEDED_BEFORE_PHYSICS_OBSERVATION',**self.pending)
            self.pending=None
        snapshot=world.get_snapshot()
        self.sequence+=1
        record=dict(control_id=self.sequence,decision_frame=decision_frame,
                    submission_frame=int(snapshot.frame),submission_time_s=float(snapshot.timestamp.elapsed_seconds),
                    submitted_control=control_values(control))
        self._write(status='SUBMISSION_REQUESTED',**record)
        try:
            ego.apply_control(control)
        except Exception as error:
            self._write(status='SUBMISSION_FAILED',**record,error_type=type(error).__name__,error=str(error))
            raise
        self._write(status='SUBMITTED',**record)
        self.pending=record

    def close(self):
        if self.stream.closed:
            return
        if self.pending is not None:
            self._write(status='NO_LATER_PHYSICS_OBSERVATION',**self.pending)
            self.pending=None
        self.stream.close()
