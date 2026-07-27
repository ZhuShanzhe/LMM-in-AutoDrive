"""Optional front-camera recorder for experiment evidence and demo videos."""

from collections import deque
import math
import os
import queue
import time

from evaluation.video import FfmpegVideoWriter


class ExperimentCamera:
    def __init__(self, world, ego_vehicle, output_dir, every_n_frames=1, width=1920, height=1080,
                 save_images=True, video_output=None, video_fps=30.0, ffmpeg_path=None,
                 video_overlay=False, sensor_tick=None, camera_view="hood",
                 video_profile="quality"):
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
        self.video_overlay = bool(video_overlay)
        self.sensor_tick = None if sensor_tick is None else max(0.0, float(sensor_tick))
        self.camera_view = str(camera_view).lower()
        self.video_profile = str(video_profile).lower()
        self.sensor = None
        self.video_writer = None
        self.saved_frames = 0
        self._images = queue.Queue()
        self._pending_images = {}
        self._speed_history = deque()
        self._font_cache = {}
        self._last_source_frame = None
        self._video_frame_remainder = 0.0

    def start(self):
        import carla

        if self.save_images:
            os.makedirs(self.output_dir, exist_ok=True)
        blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(self.width))
        blueprint.set_attribute("image_size_y", str(self.height))
        blueprint.set_attribute("fov", "100" if self.camera_view == "chase" else "90")
        # Use a fixed exposure for repeatable evidence video on the unshaded
        # generated road. Histogram exposure overreacts to the bright sky and
        # lane markings, washing out the ego vehicle during lane changes.
        self._set_camera_attribute(blueprint, "enable_postprocess_effects", "true")
        self._set_camera_attribute(blueprint, "exposure_mode", "manual")
        self._set_camera_attribute(blueprint, "iso", "50")
        self._set_camera_attribute(blueprint, "shutter_speed", "200")
        self._set_camera_attribute(blueprint, "fstop", "2.8")
        self._set_camera_attribute(blueprint, "exposure_compensation", "0.0")
        self._set_camera_attribute(blueprint, "bloom_intensity", "0.0")
        self._set_camera_attribute(blueprint, "motion_blur_intensity", "0.12")
        self._set_camera_attribute(blueprint, "lens_flare_intensity", "0.0")
        self._set_camera_attribute(blueprint, "gamma", "2.2")
        if self.sensor_tick is not None:
            blueprint.set_attribute("sensor_tick", str(self.sensor_tick))
        if self.camera_view == "chase":
            transform = carla.Transform(
                carla.Location(x=-10.0, z=3.0), carla.Rotation(pitch=-8.0)
            )
        else:
            transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        attachment_type = None
        if self.camera_view == "chase":
            attachment_type = carla.AttachmentType.SpringArmGhost
        spawn_kwargs = {"attach_to": self.ego_vehicle}
        if attachment_type is not None:
            spawn_kwargs["attachment_type"] = attachment_type
        self.sensor = self.world.spawn_actor(blueprint, transform, **spawn_kwargs)
        if self.video_output is not None:
            self.video_writer = FfmpegVideoWriter(
                self.video_output,
                self.width,
                self.height,
                self.video_fps,
                self.ffmpeg_path,
                self.video_profile,
            )
            self.video_writer.start()
        self.sensor.listen(self._on_image)

    @staticmethod
    def _set_camera_attribute(blueprint, name, value):
        if blueprint.has_attribute(name):
            blueprint.set_attribute(name, str(value))

    def _on_image(self, image):
        if self.save_images or self.video_writer is not None:
            self._images.put(image)

    def save_frame(self, frame, overlay=None, timeout_s=1.0, sample_period_s=None):
        """Persist the latest camera image available for a completed world tick."""
        if not self.save_images and self.video_writer is None:
            return False
        frame = int(frame)
        if frame % self.every_n_frames != 0:
            return False
        image = self._latest_pending_image(frame)
        deadline = time.time() + timeout_s
        while image is None and time.time() < deadline:
            try:
                candidate = self._images.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                break
            candidate_frame = int(candidate.frame)
            if candidate_frame <= frame:
                image = candidate
                break
            if candidate_frame > frame:
                self._pending_images[candidate_frame] = candidate
                break
        if image is None:
            return False
        source_frame = bytes(image.raw_data)
        self._last_source_frame = source_frame
        raw_frame = source_frame
        if self.video_overlay and overlay:
            raw_frame = self._render_overlay(raw_frame, overlay)
        if self.save_images:
            path = os.path.join(self.output_dir, "{0:08d}.png".format(frame))
            if raw_frame is source_frame:
                image.save_to_disk(path)
            else:
                from PIL import Image
                rendered = Image.frombuffer(
                    "RGBA", (self.width, self.height), raw_frame, "raw", "BGRA", 0, 1
                ).copy()
                rendered.save(path)
            self.saved_frames += 1
        if self.video_writer is not None:
            for _ in range(self._video_repeat_count(sample_period_s)):
                self.video_writer.write_raw(raw_frame)
        return True

    def _video_repeat_count(self, sample_period_s=None):
        """Resample fixed-step camera output to the requested video frame rate."""
        sample_period = sample_period_s
        if sample_period is None:
            sample_period = self.sensor_tick
        if sample_period is None:
            sample_period = 1.0 / self.video_fps
        self._video_frame_remainder += sample_period * self.video_fps
        frame_count = int(self._video_frame_remainder)
        self._video_frame_remainder -= frame_count
        return frame_count

    def _latest_pending_image(self, frame):
        eligible = [candidate_frame for candidate_frame in self._pending_images if candidate_frame <= frame]
        if not eligible:
            return None
        newest = max(eligible)
        image = self._pending_images.pop(newest)
        for candidate_frame in eligible:
            if candidate_frame != newest:
                self._pending_images.pop(candidate_frame, None)
        return image

    def hold_last_video_frame(self, duration_s):
        if self.video_writer is None:
            return 0
        return self.video_writer.append_last_frame(round(max(0.0, float(duration_s)) * self.video_fps))

    def append_terminal_overlay(self, overlay, duration_s):
        """Append a terminal status using the last unmodified camera frame."""
        if self.video_writer is None or self._last_source_frame is None:
            return 0
        raw_frame = self._last_source_frame
        if self.video_overlay and overlay:
            raw_frame = self._render_overlay(raw_frame, overlay)
        self.video_writer.write_raw(raw_frame)
        hold_frames = max(0, round(max(0.0, float(duration_s)) * self.video_fps) - 1)
        return 1 + self.video_writer.append_last_frame(hold_frames)

    def _render_overlay(self, raw_frame, overlay):
        from PIL import Image, ImageDraw, ImageFont

        image = Image.frombuffer(
            "RGBA", (self.width, self.height), raw_frame, "raw", "BGRA", 0, 1
        ).copy()
        draw = ImageDraw.Draw(image, "RGBA")

        def font(size, bold=False):
            key = (int(size), bool(bold))
            cached = self._font_cache.get(key)
            if cached is not None:
                return cached
            name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
            bundled_font = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "assets", "fonts", name
            )
            try:
                loaded = ImageFont.truetype(bundled_font, size)
            except OSError:
                loaded = ImageFont.load_default()
            self._font_cache[key] = loaded
            return loaded

        status = str(overlay.get("status", "RUNNING")).upper()
        text_color = (255, 255, 255, 255)
        muted_color = (218, 222, 228, 255)
        accent_color = (255, 255, 255, 255)
        speed_kmh = float(overlay.get("speed_kmh", 0.0))
        target_speed_kmh = float(overlay.get("target_speed_kmh", 0.0))
        progress_m = overlay.get("route_progress_m")
        route_length_m = overlay.get("route_length_m")
        progress_ratio = 0.0
        if progress_m is not None and route_length_m:
            progress_ratio = max(0.0, min(1.0, float(progress_m) / float(route_length_m)))

        panel_height = max(120, min(200, round(self.height * 0.185)))
        panel = (0, 0, self.width, panel_height)
        draw.rectangle(panel, fill=(40, 44, 50, 158))
        margin = 54
        layout_scale = panel_height / 200.0

        def y(value):
            return round(value * layout_scale)

        def sized(value):
            return max(10, round(value * layout_scale))

        line_y = y(28)
        draw.line((margin, line_y, self.width - margin, line_y), fill=(255, 255, 255, 105), width=2)
        progress_x = margin + progress_ratio * (self.width - 2 * margin)
        draw.line((margin, line_y, progress_x, line_y), fill=accent_color, width=5)
        draw.ellipse((progress_x - 6, line_y - 6, progress_x + 6, line_y + 6), fill=accent_color)
        draw.text(
            (margin, y(38)),
            "ROUTE  {0:.2f} / {1:.2f} km".format(
                float(progress_m or 0.0) / 1000.0,
                float(route_length_m or 5000.0) / 1000.0,
            ),
            font=font(sized(19), bold=True),
            fill=text_color,
        )
        status_box = draw.textbbox((0, 0), status, font=font(sized(20), bold=True))
        draw.text(
            (self.width - margin - (status_box[2] - status_box[0]), y(38)),
            status,
            font=font(sized(20), bold=True),
            fill=text_color,
        )

        columns = [margin, int(self.width * 0.29), int(self.width * 0.53), int(self.width * 0.72)]
        labels = [
            ("COMMAND", str(overlay.get("asr_text", ""))[:32]),
            ("DRIVING INTENT", "{0}  {1}".format(
                str(overlay.get("source_step_action") or overlay.get("action", "")).upper(),
                str(overlay.get("active_step_id") or overlay.get("parse_status") or "PENDING"),
            )),
            ("SCENE / RISK", "{0}  TRAFFIC {1}  PED {2}".format(
                str(overlay.get("risk_level", "LOW")),
                int(overlay.get("traffic_count", 0)),
                int(overlay.get("pedestrian_count", 0)),
            )),
            ("CONTROL  {0}".format(str(overlay.get("policy_state", "RUNNING"))),
             "{0:.0f} / {1:.0f} km/h".format(speed_kmh, target_speed_kmh)),
        ]
        for x, (label, value) in zip(columns, labels):
            draw.text((x, y(77)), label, font=font(sized(15), bold=True), fill=muted_color)
            draw.text((x, y(101)), value, font=font(sized(19), bold=True), fill=text_color)

        def ms(value):
            return "n/a" if value is None else "{0:.1f}ms".format(float(value))

        detail = "LATENCY  PARSE {0}  PERCEPTION {1}  SCENE {2}  E2E {3}".format(
            ms(overlay.get("parse_latency_ms")),
            ms(overlay.get("perception_latency_ms")),
            ms(overlay.get("scene_decision_latency_ms")),
            ms(overlay.get("end_to_end_ms")),
        )
        draw.text((margin, y(137)), detail, font=font(sized(16), bold=True), fill=text_color)
        qwen_status = overlay.get("qwen_status")
        qwen_worker = overlay.get("qwen_worker") or {}
        qwen_detail = "QWEN OFF"
        if qwen_status or qwen_worker:
            qwen_detail = "QWEN {0}".format(str(qwen_status or "WAITING").upper())
            if overlay.get("qwen_latency_s") is not None:
                qwen_detail += " {0:.1f}s".format(float(overlay["qwen_latency_s"]))
            if qwen_worker:
                qwen_detail += "  DONE {0}  DROP {1}".format(
                    int(qwen_worker.get("completed", 0)),
                    int(qwen_worker.get("dropped", 0)),
                )
        draw.text(
            (margin, y(161)),
            qwen_detail,
            font=font(sized(14)),
            fill=muted_color,
        )
        draw.text(
            (margin, y(181)),
            "SIM {0:.1f}s   COLL {1}   LANE {2}".format(
                float(overlay.get("sim_time_s", 0.0)),
                int(overlay.get("collisions", 0)),
                int(overlay.get("lane_events", 0)),
            ),
            font=font(sized(14)),
            fill=muted_color,
        )
        self._draw_speed_chart(
            draw,
            overlay,
            font,
            muted_color,
            (int(self.width * 0.58), y(143), self.width - margin, y(190)),
        )
        return image.tobytes("raw", "BGRA")

    def _draw_speed_chart(self, draw, overlay, font, muted_color, plot):
        sim_time = float(overlay.get("sim_time_s", 0.0))
        self._speed_history.append((sim_time, float(overlay.get("speed_kmh", 0.0)), float(overlay.get("target_speed_kmh", 0.0))))
        while self._speed_history and self._speed_history[0][0] < sim_time - 12.0:
            self._speed_history.popleft()
        draw.text((plot[0], plot[1] - 28), "SPD", font=font(18, bold=True), fill=muted_color)
        values = [max(speed, target) for _, speed, target in self._speed_history]
        max_speed = max(30.0, max(values) if values else 30.0)
        max_speed = max(10.0, (int(max_speed / 10.0) + 1) * 10.0)
        for level in range(3):
            y = plot[3] - (plot[3] - plot[1]) * level / 2.0
            draw.line((plot[0], y, plot[2], y), fill=(70, 80, 92, 150), width=1)
        if len(self._speed_history) < 2:
            return
        first_time = self._speed_history[0][0]
        span = max(12.0, sim_time - first_time)

        def points(index):
            result = []
            for time_s, speed, target in self._speed_history:
                value = speed if index == 1 else target
                x = plot[0] + (time_s - first_time) / span * (plot[2] - plot[0])
                y = plot[3] - min(value, max_speed) / max_speed * (plot[3] - plot[1])
                result.append((x, y))
            return result

        draw.line(points(1), fill=(91, 211, 255, 255), width=3, joint="curve")
        draw.line(points(2), fill=(255, 211, 54, 255), width=3, joint="curve")

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
