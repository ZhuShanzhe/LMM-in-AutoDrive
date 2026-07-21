"""CARLA sensors that turn collisions and lane invasions into log events."""

import time


class EventMonitor:
    def __init__(self, world, ego_vehicle):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.collision_events = []
        self.lane_invasion_events = []
        self._sensors = []
        self._reported_collision_count = 0
        self._reported_lane_invasion_count = 0

    def start(self):
        blueprints = self.world.get_blueprint_library()
        collision_bp = blueprints.find("sensor.other.collision")
        lane_bp = blueprints.find("sensor.other.lane_invasion")
        collision_sensor = self.world.spawn_actor(collision_bp, self._identity_transform(), attach_to=self.ego_vehicle)
        lane_sensor = self.world.spawn_actor(lane_bp, self._identity_transform(), attach_to=self.ego_vehicle)
        collision_sensor.listen(self._on_collision)
        lane_sensor.listen(self._on_lane_invasion)
        self._sensors = [collision_sensor, lane_sensor]

    @staticmethod
    def _identity_transform():
        import carla
        return carla.Transform()

    def _on_collision(self, event):
        other_actor = getattr(event, "other_actor", None)
        self.collision_events.append({
            "frame": int(event.frame),
            "time_s": time.time(),
            "other_actor_id": getattr(other_actor, "id", None),
            "other_actor_type": getattr(other_actor, "type_id", None),
        })

    def _on_lane_invasion(self, event):
        markings = [str(marking.type) for marking in event.crossed_lane_markings]
        self.lane_invasion_events.append({
            "frame": int(event.frame),
            "time_s": time.time(),
            "markings": markings,
        })

    def snapshot(self, current_frame):
        del current_frame
        new_collisions = self.collision_events[self._reported_collision_count:]
        new_lane_invasions = self.lane_invasion_events[self._reported_lane_invasion_count:]
        self._reported_collision_count = len(self.collision_events)
        self._reported_lane_invasion_count = len(self.lane_invasion_events)
        return {
            "collision": bool(new_collisions),
            "lane_invasion": bool(new_lane_invasions),
            "new_collision_events": new_collisions,
            "new_lane_invasion_events": new_lane_invasions,
            "collision_count": len(self.collision_events),
            "lane_invasion_count": len(self.lane_invasion_events),
        }

    def destroy(self):
        for sensor in self._sensors:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._sensors = []
