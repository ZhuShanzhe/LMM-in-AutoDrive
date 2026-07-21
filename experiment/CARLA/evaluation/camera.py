"""Optional front-camera recorder for experiment evidence and demo videos."""

from collections import deque
import os
import queue
import time

from evaluation.video import FfmpegVideoWriter


class ExperimentCamera:
    def __init__(self, world, ego_vehicle, output_dir, every_n_frames=1, width=1920, height=1080,
                 save_images=True, video_output=None, video_fps=30.0, ffmpeg_path=None,
                 video_overlay=False):
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
        self.sensor = None
        self.video_writer = None
        self.saved_frames = 0
        self._images = queue.Queue()
        self._pending_images = {}
        self._speed_history = deque()

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
        if self.save_images or self.video_writer is not None:
            self._images.put(image)

    def save_frame(self, frame, overlay=None, timeout_s=1.0):
        """Persist the camera image belonging to a just-completed world tick."""
        if not self.save_images and self.video_writer is None:
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
        if self.save_images:
            path = os.path.join(self.output_dir, "{0:08d}.png".format(frame))
            image.save_to_disk(path)
            self.saved_frames += 1
        if self.video_writer is not None:
            raw_frame = bytes(image.raw_data)
            if self.video_overlay and overlay:
                raw_frame = self._render_overlay(raw_frame, overlay)
            self.video_writer.write_raw(raw_frame)
        return True

    def _render_overlay(self, raw_frame, overlay):
        import math

        from PIL import Image, ImageDraw, ImageFont

        image = Image.frombuffer(
            "RGBA", (self.width, self.height), raw_frame, "raw", "BGRA", 0, 1
        ).copy()
        draw = ImageDraw.Draw(image, "RGBA")

        def font(size, bold=False):
            name = "consolab.ttf" if bold else "consola.ttf"
            try:
                return ImageFont.truetype(os.path.join("C:\\Windows\\Fonts", name), size)
            except OSError:
                return ImageFont.load_default()

        status = str(overlay.get("status", "RUNNING")).upper()
        status_color = {
            "RUNNING": (255, 211, 54, 255),
            "SUCCESS": (72, 208, 113, 255),
            "FAILURE": (244, 80, 80, 255),
            "INCOMPLETE": (255, 153, 51, 255),
        }.get(status, (230, 230, 230, 255))
        text_color = (244, 247, 250, 255)
        muted_color = (182, 192, 204, 255)

        panel = (28, 28, min(self.width - 28, 1190), 428)
        draw.rounded_rectangle(panel, radius=12, fill=(8, 12, 18, 220), outline=(79, 94, 110, 210), width=2)
        speed_kmh = float(overlay.get("speed_kmh", 0.0))
        target_speed_kmh = float(overlay.get("target_speed_kmh", 0.0))
        gauge_center = (178, 214)
        gauge_radius = 116
        gauge_box = (
            gauge_center[0] - gauge_radius,
            gauge_center[1] - gauge_radius,
            gauge_center[0] + gauge_radius,
            gauge_center[1] + gauge_radius,
        )
        gauge_max = max(40.0, (int(max(speed_kmh, target_speed_kmh, 1.0) / 10.0) + 1) * 10.0)
        draw.arc(gauge_box, 135, 405, fill=(58, 69, 82, 255), width=18)
        for tick in range(7):
            angle = math.radians(135 + 270 * tick / 6.0)
            outer = (
                gauge_center[0] + math.cos(angle) * (gauge_radius - 2),
                gauge_center[1] + math.sin(angle) * (gauge_radius - 2),
            )
            inner = (
                gauge_center[0] + math.cos(angle) * (gauge_radius - 20),
                gauge_center[1] + math.sin(angle) * (gauge_radius - 20),
            )
            draw.line((outer, inner), fill=muted_color, width=3)
        needle_angle = math.radians(135 + 270 * min(speed_kmh, gauge_max) / gauge_max)
        needle_end = (
            gauge_center[0] + math.cos(needle_angle) * (gauge_radius - 32),
            gauge_center[1] + math.sin(needle_angle) * (gauge_radius - 32),
        )
        draw.line((gauge_center, needle_end), fill=(91, 211, 255, 255), width=8)
        draw.ellipse((gauge_center[0] - 11, gauge_center[1] - 11, gauge_center[0] + 11, gauge_center[1] + 11), fill=(91, 211, 255, 255))
        draw.text((111, 167), "{0:.0f}".format(speed_kmh), font=font(47, bold=True), fill=text_color)
        draw.text((123, 222), "km/h", font=font(22), fill=muted_color)

        draw.text((338, 52), "CARLA RUN TELEMETRY", font=font(34, bold=True), fill=text_color)
        draw.text(
            (338, 105),
            "scenario={0}".format(overlay.get("scenario", "")),
            font=font(27),
            fill=muted_color,
        )
        draw.text(
            (338, 143),
            "frame={0}   sim={1:.2f}s".format(
                overlay.get("frame", 0), overlay.get("sim_time_s", 0.0)
            ),
            font=font(25),
            fill=muted_color,
        )
        draw.text(
            (338, 184),
            "{0}  ->  target {1:.1f} km/h".format(
                str(overlay.get("action", "")).upper(), overlay.get("target_speed_kmh", 0.0)
            ),
            font=font(31, bold=True),
            fill=status_color if overlay.get("emergency", False) else text_color,
        )
        draw.text(
            (338, 229),
            "reason={0}  collisions={1}  lane_events={2}".format(
                str(overlay.get("reason", ""))[:42], overlay.get("collisions", 0), overlay.get("lane_events", 0)
            ),
            font=font(24),
            fill=muted_color,
        )

        def draw_bar(label, value, y, color, centered=False):
            x = 338
            width = 570
            height = 26
            draw.text((x, y - 4), label, font=font(23, bold=True), fill=text_color)
            left = x + 70
            right = left + width
            draw.rounded_rectangle((left, y, right, y + height), radius=8, fill=(45, 55, 67, 245))
            if centered:
                center = (left + right) / 2.0
                draw.line((center, y - 4, center, y + height + 4), fill=muted_color, width=2)
                end = center + max(-1.0, min(1.0, value)) * width / 2.0
                draw.rectangle((min(center, end), y + 4, max(center, end), y + height - 4), fill=color)
            else:
                end = left + max(0.0, min(1.0, value)) * width
                draw.rounded_rectangle((left, y, end, y + height), radius=8, fill=color)

        draw_bar("THR", float(overlay.get("throttle", 0.0)), 276, (72, 208, 113, 255))
        draw_bar("BRK", float(overlay.get("brake", 0.0)), 320, (244, 80, 80, 255))
        draw_bar("STR", float(overlay.get("steer", 0.0)), 364, (255, 211, 54, 255), centered=True)

        badge = (panel[2] - 275, 55, panel[2] - 30, 122)
        draw.rounded_rectangle(badge, radius=10, fill=(20, 25, 32, 245), outline=status_color, width=4)
        badge_font = font(31, bold=True)
        box = draw.textbbox((0, 0), status, font=badge_font)
        text_x = badge[0] + (badge[2] - badge[0] - (box[2] - box[0])) / 2
        draw.text((text_x, 70), status, font=badge_font, fill=status_color)

        self._draw_speed_chart(draw, overlay, font, text_color, muted_color)
        return image.tobytes("raw", "BGRA")

    def _draw_speed_chart(self, draw, overlay, font, text_color, muted_color):
        sim_time = float(overlay.get("sim_time_s", 0.0))
        self._speed_history.append((sim_time, float(overlay.get("speed_kmh", 0.0)), float(overlay.get("target_speed_kmh", 0.0))))
        while self._speed_history and self._speed_history[0][0] < sim_time - 12.0:
            self._speed_history.popleft()

        chart = (28, self.height - 262, min(self.width - 28, 1190), self.height - 28)
        draw.rounded_rectangle(chart, radius=12, fill=(8, 12, 18, 220), outline=(79, 94, 110, 210), width=2)
        draw.text((54, chart[1] + 20), "SPEED HISTORY (last 12s)", font=font(25, bold=True), fill=text_color)
        draw.text((chart[2] - 284, chart[1] + 24), "actual", font=font(21), fill=(91, 211, 255, 255))
        draw.line((chart[2] - 354, chart[1] + 39, chart[2] - 298, chart[1] + 39), fill=(91, 211, 255, 255), width=4)
        draw.text((chart[2] - 120, chart[1] + 24), "target", font=font(21), fill=(255, 211, 54, 255))
        draw.line((chart[2] - 190, chart[1] + 39, chart[2] - 134, chart[1] + 39), fill=(255, 211, 54, 255), width=4)

        plot = (86, chart[1] + 74, chart[2] - 32, chart[3] - 34)
        values = [max(speed, target) for _, speed, target in self._speed_history]
        max_speed = max(30.0, max(values) if values else 30.0)
        max_speed = max(10.0, (int(max_speed / 10.0) + 1) * 10.0)
        for level in range(5):
            y = plot[3] - (plot[3] - plot[1]) * level / 4.0
            value = max_speed * level / 4.0
            draw.line((plot[0], y, plot[2], y), fill=(70, 80, 92, 150), width=1)
            draw.text((34, y - 12), "{0:.0f}".format(value), font=font(18), fill=muted_color)

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

        draw.line(points(1), fill=(91, 211, 255, 255), width=5, joint="curve")
        draw.line(points(2), fill=(255, 211, 54, 255), width=4, joint="curve")

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
