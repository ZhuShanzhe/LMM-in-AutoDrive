"""Camera-only full-route traffic inspection; not a driving evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import math
import queue
import subprocess
from carla_bootstrap import setup_carla_api
setup_carla_api()
import carla
from scenarios.basic.urban_voice_5km import UrbanVoice5KmScenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--speed-kmh', type=float, default=50.0)
    args = parser.parse_args()
    if not 0 < args.speed_kmh <= 130:
        parser.error('--speed-kmh must be in (0, 130]')
    args.output.mkdir(parents=True, exist_ok=True)
    client = carla.Client('localhost', 2000)
    client.set_timeout(60)
    world = client.load_world('Town04_Opt')
    saved = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = .05
    world.apply_settings(settings)
    scene = UrbanVoice5KmScenario(world, config_path='configs/basic_voice_traffic_preview.json')
    scene.client = client
    camera = encoder = None
    try:
        scene.setup()
        scene.ego_vehicle.destroy()
        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', '1920')
        blueprint.set_attribute('image_size_y', '1080')
        blueprint.set_attribute('fov', '90')
        blueprint.set_attribute('sensor_tick', '0.0')
        camera = world.spawn_actor(blueprint, carla.Transform())
        frames = queue.Queue()
        camera.listen(frames.put)
        encoder = subprocess.Popen([args.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'bgra', '-s', '1920x1080', '-r', '20', '-i', '-', '-an',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
            str(args.output / 'route_traffic.mp4')], stdin=subprocess.PIPE)
        route = scene.route_manager.route
        (args.output / 'route.json').write_text(json.dumps(route), encoding='utf-8')
        (args.output / 'config.json').write_text(json.dumps(scene.config, ensure_ascii=False, indent=2), encoding='utf-8')
        index = 0
        total = float(route[-1]['distance_m'])
        counts = []
        with (args.output / 'traffic.jsonl').open('w', encoding='utf-8') as log:
            stride = args.speed_kmh / 3.6 * .05
            for tick in range(math.ceil(total / stride) + 1):
                progress = min(total, tick * stride)
                while index + 1 < len(route) and route[index + 1]['distance_m'] <= progress:
                    index += 1
                point = route[index]
                following = route[min(index + 1, len(route) - 1)]
                fraction = min(1.0, (progress - point['distance_m']) /
                               max(.001, following['distance_m'] - point['distance_m']))
                xyz = [point[k] + fraction * (following[k] - point[k]) for k in ('x', 'y', 'z')]
                yaw_delta = (following['yaw'] - point['yaw'] + 180) % 360 - 180
                camera.set_transform(carla.Transform(carla.Location(x=xyz[0], y=xyz[1], z=xyz[2] + 7),
                    carla.Rotation(pitch=-16, yaw=point['yaw'] + fraction * yaw_delta)))
                scene.traffic.tick(camera, progress)
                frame_id = world.tick()
                while True:
                    frame = frames.get(timeout=15)
                    if frame.frame >= frame_id:
                        break
                if frame.frame != frame_id:
                    raise RuntimeError('camera frame mismatch')
                encoder.stdin.write(frame.raw_data)
                front_actors = scene.traffic._visible_actors(camera)
                count = len(front_actors)
                forward = camera.get_transform().get_forward_vector()
                same_direction = sum(
                    a.get_transform().get_forward_vector().dot(forward) > .5
                    for a in front_actors)
                counts.append(count)
                log.write(json.dumps(dict(frame=frame_id, progress_m=progress, front_count=count,
                    front_same_direction_count=same_direction,
                    traffic=scene.traffic.snapshot())) + '\n')
                if tick % 400 == 0:
                    print(f'progress={progress:.0f}m front={count} recycled={scene.traffic._recycle_count}', flush=True)
        encoder.stdin.close()
        if encoder.wait(timeout=60):
            raise RuntimeError('video encoder failed')
        encoder = None
        summary = dict(route_m=total, frames=len(counts), front_min=min(counts),
                       front_mean=sum(counts)/len(counts), zero_frames=counts.count(0),
                       recycled=scene.traffic._recycle_count, camera_only=True,
                       observer_speed_kmh=args.speed_kmh, fps=20)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(summary)
    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()
        scene.traffic.destroy() if scene.traffic else None
        if scene.ego_vehicle is not None and scene.ego_vehicle.is_alive:
            scene.ego_vehicle.destroy()
        if encoder is not None:
            encoder.stdin.close()
            encoder.wait(timeout=60)
        world.apply_settings(saved)


if __name__ == '__main__':
    main()
