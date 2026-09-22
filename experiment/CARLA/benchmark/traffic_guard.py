"""Truth-based longitudinal guard for simulator fixtures, not a model adapter."""
import math


def follow_target(requested_kmh, ego_mps, lead_mps, gap_m):
    values=(requested_kmh,ego_mps,lead_mps,gap_m)
    if any(not math.isfinite(v) for v in values) or min(values[:3])<0:
        raise ValueError('invalid following kinematics')
    closing=max(0.,ego_mps-lead_mps)
    braking_gap=4.+max(0.,ego_mps**2-lead_mps**2)/(2.*4.)
    emergency=gap_m<=4. or (closing>0 and gap_m<braking_gap)
    following=max(0.,lead_mps+(gap_m-6.-1.5*ego_mps)/2.)
    stopping=math.sqrt(max(0.,lead_mps**2+2.*3.*(gap_m-6.)))
    return min(requested_kmh,3.6*following,3.6*stopping), emergency


class BenchmarkTrafficGuard:
    def __init__(self, vehicle):
        from agents.navigation.basic_agent import BasicAgent
        self.vehicle=vehicle
        self.world=vehicle.get_world()
        self.map=self.world.get_map()
        self.agent=BasicAgent(vehicle)

    def apply(self, intent):
        from agents.navigation.local_planner import RoadOption
        result=dict(intent)
        ego=self.vehicle
        velocity=ego.get_velocity()
        speed=math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
        horizon=max(30.,speed*3.+speed**2/6.)
        waypoint=self.map.get_waypoint(ego.get_location())
        plan=[]
        for _ in range(int(horizon/5)+2):
            if waypoint is None:
                break
            plan.append((waypoint,RoadOption.LANEFOLLOW))
            options=waypoint.next(5.)
            if len(options)!=1:
                break
            waypoint=options[0]
        self.agent.set_global_plan(plan,stop_waypoint_creation=True,clean_queue=True)
        evidence=dict(scope='simulator_truth_baseline_only',reason='clear',
                      requested_target_kmh=intent['target_speed_kmh'],
                      detection_horizon_m=horizon,plan_points=len(plan))
        blocked, light=self.agent._affected_by_traffic_light(max_distance=horizon)
        if blocked:
            result.update(target_speed_kmh=0.,emergency=True)
            evidence.update(reason='official_red_light_detection',actor_id=light.id)
        else:
            actors=sorted(self.world.get_actors().filter('vehicle.*'),
                          key=lambda a:a.get_location().distance(ego.get_location()))
            blocked,lead,_=self.agent._vehicle_obstacle_detected(actors,max_distance=horizon)
            if blocked:
                lead_velocity=lead.get_velocity()
                lead_speed=math.sqrt(lead_velocity.x**2+lead_velocity.y**2+lead_velocity.z**2)
                gap=(lead.get_location().distance(ego.get_location())-
                     ego.bounding_box.extent.x-lead.bounding_box.extent.x)
                target,emergency=follow_target(float(intent['target_speed_kmh']),speed,lead_speed,gap)
                result['target_speed_kmh']=target
                result['emergency']=bool(result.get('emergency')) or emergency
                evidence.update(reason='official_lead_detection',actor_id=lead.id,gap_m=gap,
                                lead_speed_kmh=lead_speed*3.6,emergency=emergency)
        evidence['applied_target_kmh']=result['target_speed_kmh']
        evidence['limitations']=['not_full_collision_avoidance',
                                 'junction_branch_and_lane_change_rear_gap_not_covered']
        return result,evidence
