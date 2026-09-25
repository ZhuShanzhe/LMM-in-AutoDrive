"""Build an explicit traffic extension of the existing basic test route."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'configs/basic_voice_control_5km.json'
    config = json.loads(source.read_text(encoding='utf-8'))
    config['scenario_id'] = 'basic_voice_traffic_preview'
    config['scenario_name'] = 'Basic route with same-direction background traffic'
    config['environment']['dynamic_traffic'] = True
    types = ['vehicle.audi.tt', 'vehicle.tesla.model3', 'vehicle.lincoln.mkz_2020',
             'vehicle.nissan.patrol', 'vehicle.mercedes.coupe', 'vehicle.volkswagen.t2']
    vehicles = []
    for row in range(24):
        for lane in range(1, 5):
            i = row * 4 + lane - 1
            vehicles.append(dict(id=f'flow_{i:03d}', route_distance_m=35 + row * 50 + lane * 6,
                                 lane_from_right=lane, vehicle_type=types[i % len(types)],
                                 speed_kmh=42 + i % 9, auto_lane_change=False))
    config['traffic'] = dict(enabled=True, traffic_manager_port=8000, seed=20260919,
                             following_distance_m=8, ignore_traffic_lights=False,
                             ignore_traffic_signs=False, traffic_manager_set_path=True,
                             traffic_manager_path_mode='local', recycle_enabled=True,
                             maintenance_mode='route_density', density_check_ticks=10,
                             density_spacing_m=50, density_ahead_start_m=400,
                             density_ahead_end_m=800, lifecycle_protected_radius_m=350,
                             density_max_actors=128,
                             density_lanes=[1, 2, 3, 4], recycle_min_speed_kmh=42,
                             recycle_max_speed_kmh=50, absolute_cruise_speed=True,
                             vehicles=vehicles)
    destination = root / 'configs/basic_voice_traffic_preview.json'
    destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
