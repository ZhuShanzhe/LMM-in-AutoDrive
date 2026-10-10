"""Conservative short-gap pedestrian association in measured world coordinates."""
import math
import numpy as np


class PedestrianIdentityAssociation:
    def __init__(self, max_gap_s=1., max_speed_mps=3., position_noise_m=1.5):
        self.max_gap_s = max_gap_s
        self.max_speed_mps = max_speed_mps
        self.position_noise_m = position_noise_m
        self.states = {}
        self.aliases = {}
        self.counter = 0

    def update(self, observations, timestamp_s):
        if not isinstance(timestamp_s, (int, float)) or not math.isfinite(timestamp_s):
            return []
        current = []
        for item in observations:
            position = np.asarray(item['position'], dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                continue
            current.append((item, position))
        proposals = {}
        for index, (item, position) in enumerate(current):
            scores = []
            for identity, state in self.states.items():
                dt = timestamp_s - state['timestamp_s']
                if not 0 < dt <= 5.:
                    continue
                distance = float(np.linalg.norm(position[:2]-state['position'][:2]))
                if abs(position[2]-state['position'][2]) > .5:
                    continue
                if dt <= self.max_gap_s and distance <= self.position_noise_m+self.max_speed_mps*dt:
                    scores.append((distance, identity))
                elif dt > self.max_gap_s:
                    appearance = item.get('appearance')
                    descriptors = state.get('appearances', [])
                    history = state.get('history', [])
                    if appearance is None or not descriptors or len(history) < 8:
                        continue
                    interval = history[-1][0]-history[0][0]
                    if interval < .5:
                        continue
                    velocity = (history[-1][1][:2]-history[0][1][:2])/interval
                    if np.linalg.norm(velocity) > self.max_speed_mps:
                        continue
                    similarity = max(float(np.asarray(old) @ appearance) for old in descriptors)
                    predicted = state['position'][:2]+velocity*dt
                    error = float(np.linalg.norm(position[:2]-predicted))
                    bound = min(3., 1.+.4*dt)
                    if similarity >= .97 and error <= bound:
                        scores.append((error+4.*(1.-similarity), identity))
            scores.sort()
            # Nearby competing people must not transfer an exit fact to one another.
            existing = self.aliases.get(item['track_id'])
            if existing is not None and any(identity == existing for _, identity in scores):
                proposals[index] = existing
            elif scores and (len(scores) == 1 or scores[1][0]-scores[0][0] >= .5):
                proposals[index] = scores[0][1]
        claimed = {}
        for index, identity in proposals.items():
            claimed.setdefault(identity, []).append(index)
        output = []
        for index, (item, position) in enumerate(current):
            identity = proposals.get(index)
            if identity is not None and len(claimed[identity]) != 1:
                identity = None
            if identity is None:
                if item.get('age_frames', 2) < 2:
                    continue
                self.counter += 1
                identity = f'world-pedestrian-{self.counter}'
            for alias in item.get('aliases', [item['track_id']]):
                self.aliases[alias] = identity
            old_state = self.states.get(identity, {})
            history = [value for value in old_state.get('history', [])
                if 0 <= timestamp_s-value[0] <= 1.]
            history.append((timestamp_s, position.copy()))
            descriptors = list(old_state.get('appearances', []))
            if item.get('appearance') is not None:
                descriptors.append(np.asarray(item['appearance']).copy())
            self.states[identity] = dict(position=position, timestamp_s=timestamp_s,
                history=history[-24:], appearances=descriptors[-12:])
            output.append(dict(track_id=identity, position=position.tolist(),
                image_track_id=item['track_id'], view=item.get('view', 'front'),
                measurement_source=item.get('source', 'static_road_footpoint')))
        self.states = {key: value for key, value in self.states.items()
            if 0 <= timestamp_s-value['timestamp_s'] <= 30.}
        live = set(self.states)
        self.aliases = {key: value for key, value in self.aliases.items() if value in live}
        return output
