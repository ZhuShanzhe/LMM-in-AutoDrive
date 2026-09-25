"""Frame-indexed event accounting with explicit bounded-drain provenance."""
import threading
import time

from .catalog import ConfigError


class SafetyLedger:
    def __init__(self):
        self.lock = threading.Lock()
        self.events = []
        self.late_events = []
        self.last_received = time.monotonic()
        self.sealed = False

    def record(self, kind, frame, **details):
        if kind not in {'collision','lane_invasion'} or type(frame) is not int or frame < 0:
            raise ConfigError('invalid safety event')
        with self.lock:
            event = dict(kind=kind,frame=frame,**details)
            self.last_received = time.monotonic()
            (self.late_events if self.sealed else self.events).append(event)

    def seal_after_quiet(self, quiet_s=.5, timeout_s=5):
        """Call only after stopping sensors; silence is a bounded assumption, not an ACK."""
        if quiet_s <= 0 or timeout_s < quiet_s:
            raise ConfigError('invalid event drain bounds')
        started = time.monotonic()
        while time.monotonic()-started <= timeout_s:
            with self.lock:
                if time.monotonic()-max(started,self.last_received) >= quiet_s:
                    self.sealed = True
                    return
            time.sleep(.01)
        raise ConfigError('safety callbacks did not quiesce')

    def packet(self, frame):
        with self.lock:
            if not self.sealed or self.late_events:
                raise ConfigError('safety ledger unsealed or received late events')
            events = [e for e in self.events if e['frame'] <= frame]
            collisions = {(e['frame'],e.get('other_actor_id')) for e in events if e['kind']=='collision'}
            violations = {e['frame'] for e in events if e['kind']=='lane_invasion' and e.get('illegal',False)}
        return dict(frame=frame,complete=True,collisions=len(collisions),violations=len(violations),
                    completion_policy='stopped_sensors_bounded_quiet_drain',
                    coverage=['collision','solid_lane_marking'],transport_acknowledged=False)

    def export(self):
        with self.lock:
            return dict(events=list(self.events),late_events=list(self.late_events),sealed=self.sealed,
                        coverage=['collision','solid_lane_marking'],
                        limitations=['no_transport_ack','red_light_and_other_violations_not_measured'])

    def episode_result(self):
        """Do not let a task's early success hide later recorded safety events."""
        data=self.export()
        if not data['sealed'] or data['late_events']:
            status='UNKNOWN'
        elif any(e['kind']=='collision' or e.get('illegal',False) for e in data['events']):
            status='FAILURE'
        else:
            status='NO_RECORDED_VIOLATION'
        return dict(status=status,coverage=data['coverage'],limitations=data['limitations'])
