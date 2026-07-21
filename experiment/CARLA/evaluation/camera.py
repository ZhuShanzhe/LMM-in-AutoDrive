"""Optional front-camera recorder for experiment evidence and demo videos."""

import os
import queue
import time

from evaluation.video import FfmpegVideoWriter


class ExperimentCamera:
    def __init__(self, world, ego_vehicle, output_dir, every_n_frames=1, width=1920, height=1080,
                 save_images=True, video_output=None, video_fps=30.0, ffmpeg_path=None):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.output_dir = output_dir
        self.every_n_frames = max(1, int(every_n_frames))
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.save_images = bool(save_images)
        self.video_output = video_output
        self.video_fps = float(video_fps)
        self.ffmpeg_path = ffmpeg_path
        self.sensor = None
        self.video_writer = None
        self.saved_frames = 0
        self._images = queue.Queue()
        self._pending_images = {}

    def start(self):
        import carla

        if self.save_images:
            os.makedirs(self.output_dir, exist_ok=True)
        blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(self.width))
        blueprint.set_attribute("image_size_y", str(self.height))
        blueprint.set_attribute("fov", "90")
        if self.video_output is not None:
            blueprint.set_attribute("sensor_tick", str(1.0 / self.video_fps))
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensor = self.world.spawn_actor(blueprint, transform, attach_to=self.ego_vehicle)
        if self.video_output is not None:
            self.video_writer = FfmpegVideoWriter(
                self.video_output,
                self.width,
                self.height,
                self.video_fps,
                self.ffmpeg_path,
            )
            self.video_writer.start()
        self.sensor.listen(self._on_image)

    def _on_image(self, image):
        if self.video_writer is not None:
            self.video_writer.write(image)
        if self.save_images:
            self._images.put(image)

    def save_frame(self, frame, timeout_s=1.0):
        """Save the camera image belonging to a just-completed world tick."""
        if not self.save_images:
            return False
        frame = int(frame)
        if frame % self.every_n_frames != 0:
            return False
        image = self._pending_images.pop(frame, None)
        deadline = time.time() + timeout_s
        while image is None and time.time() < deadline:
            try:
                candidate = self._images.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                break
            candidate_frame = int(candidate.frame)
            if candidate_frame < frame:
                continue
            if candidate_frame > frame:
                self._pending_images[candidate_frame] = candidate
                break
            image = candidate
        if image is None:
            return False
        path = os.path.join(self.output_dir, "{0:08d}.png".format(frame))
        image.save_to_disk(path)
        self.saved_frames += 1
        return True

    def destroy(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
        self.sensor = None
        if self.video_writer is not None:
            self.video_writer.close()
            print(
                "[Camera] Direct video frames: {0}, dropped: {1}".format(
                    self.video_writer.frame_count,
                    self.video_writer.dropped_frames,
                )
            )
            self.video_writer = None
