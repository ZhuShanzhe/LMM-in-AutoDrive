"""Optional adapters for CARLA BasicAgent and BehaviorAgent.

These agents should be selected when the decision layer has a concrete target
location, especially at a junction.  The default experiment runner uses the
PID controller so it remains deterministic for the three starter scenarios.
"""

import carla

from control.protocol import normalize_intent


class CarlaAgentController:
    def __init__(self, vehicle, mode="behavior"):
        if mode == "basic":
            from agents.navigation.basic_agent import BasicAgent
            self.agent = BasicAgent(vehicle)
        elif mode == "behavior":
            from agents.navigation.behavior_agent import BehaviorAgent
            self.agent = BehaviorAgent(vehicle, behavior="normal")
        else:
            raise ValueError("mode must be 'basic' or 'behavior'")
        self.vehicle = vehicle

    def run_step(self, intent, dt):
        del dt
        intent = normalize_intent(intent)
        if intent["emergency"] or intent["action"] == "emergency_brake":
            return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0), intent
        if intent["target_location"] is not None:
            target = intent["target_location"]
            destination = carla.Location(x=target["x"], y=target["y"], z=target["z"])
            self.agent.set_destination(destination)
        self.agent.set_target_speed(intent["target_speed_kmh"])
        return self.agent.run_step(), intent
