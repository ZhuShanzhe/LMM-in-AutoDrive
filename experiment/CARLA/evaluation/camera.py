"""Front RGB camera with direct H.264 recording and pre-encode HUD."""

from collections import deque
import math
import os
import queue
import time

from evaluation.video import FfmpegVideoWriter


class ExperimentCamera:
    """Synchronize an attached RGB sensor with simulation ticks."""

    def __init__(
        self,
        world,
        ego_vehicle,
        output_dir,
        every_n_frames=1,
        width=1920,
        height=1080,
        save_images=True,
        video_output=None,
        video_fps=30.0,
        ffmpeg_path=None,
        video_overlay=False,
        camera_attributes=None,
        camera_pose=None,
    ):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.output_dir = output_dir
        self.every_n_frames = max(
            1,
            int(every_n_frames),
        )
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.save_images = bool(save_images)
        self.video_output = video_output
        self.video_fps = float(video_fps)
        self.ffmpeg_path = ffmpeg_path
        self.video_overlay = bool(video_overlay)
        self.camera_attributes = dict(
            camera_attributes or {}
        )
        self.camera_pose = tuple(
            camera_pose
            or (
                1.5,
                0.0,
                2.4,
                0.0,
                0.0,
            )
        )
        if len(self.camera_pose) != 5:
            raise ValueError(
                "camera_pose must contain "
                "x, y, z, pitch, yaw"
            )
        self.sensor = None
        self.video_writer = None
        self.saved_frames = 0
        self.written_video_frames = 0
        self.dropped_video_frames = 0
        self._images = queue.Queue()
        self._pending_images = {}
        self._speed_history = deque()

    def start(self):
        import carla

        if self.save_images:
            os.makedirs(
                self.output_dir,
                exist_ok=True,
            )
        blueprint = (
            self.world.get_blueprint_library()
            .find("sensor.camera.rgb")
        )
        blueprint.set_attribute(
            "image_size_x",
            str(self.width),
        )
        blueprint.set_attribute(
            "image_size_y",
            str(self.height),
        )
        blueprint.set_attribute("fov", "90")
        for name, value in (
            self.camera_attributes.items()
        ):
            if blueprint.has_attribute(name):
                blueprint.set_attribute(
                    name,
                    str(value),
                )
        if self.video_output is not None:
            blueprint.set_attribute(
                "sensor_tick",
                str(1.0 / self.video_fps),
            )
        x, y, z, pitch, yaw = self.camera_pose
        transform = carla.Transform(
            carla.Location(
                x=x,
                y=y,
                z=z,
            ),
            carla.Rotation(
                pitch=pitch,
                yaw=yaw,
                roll=0.0,
            ),
        )
        self.sensor = self.world.spawn_actor(
            blueprint,
            transform,
            attach_to=self.ego_vehicle,
        )
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
        if (
            self.save_images
            or self.video_writer is not None
        ):
            self._images.put(image)

    def save_frame(
        self,
        frame,
        overlay=None,
        timeout_s=1.0,
    ):
        """Persist the front image belonging to a completed world tick."""

        if (
            not self.save_images
            and self.video_writer is None
        ):
            return False
        frame = int(frame)
        save_image = (
            self.save_images
            and frame % self.every_n_frames == 0
        )
        write_video = self.video_writer is not None
        # Image sampling and direct video have independent cadences.  A
        # sparse --record-every-n value must not turn a 20 FPS recording
        # into a handful of still frames.
        if not save_image and not write_video:
            return False
        image = self._pending_images.pop(
            frame,
            None,
        )
        deadline = time.time() + timeout_s
        while (
            image is None
            and time.time() < deadline
        ):
            try:
                candidate = self._images.get(
                    timeout=max(
                        0.01,
                        deadline - time.time(),
                    )
                )
            except queue.Empty:
                break
            candidate_frame = int(candidate.frame)
            if candidate_frame < frame:
                continue
            if candidate_frame > frame:
                self._pending_images[
                    candidate_frame
                ] = candidate
                break
            image = candidate
        if image is None:
            return False
        if save_image:
            path = os.path.join(
                self.output_dir,
                "{0:08d}.png".format(frame),
            )
            image.save_to_disk(path)
            self.saved_frames += 1
        if write_video:
            raw_frame = bytes(image.raw_data)
            if self.video_overlay and overlay:
                raw_frame = self._render_overlay(
                    raw_frame,
                    overlay,
                )
            self.video_writer.write_raw(raw_frame)
        return True

    def hold_last_video_frame(self, duration_s):
        if self.video_writer is None:
            return 0
        return self.video_writer.append_last_frame(
            round(
                max(0.0, float(duration_s))
                * self.video_fps
            )
        )

    def _render_overlay(self, raw_frame, overlay):
        from PIL import (
            Image,
            ImageDraw,
            ImageFont,
        )

        image = Image.frombuffer(
            "RGBA",
            (self.width, self.height),
            raw_frame,
            "raw",
            "BGRA",
            0,
            1,
        ).copy()
        draw = ImageDraw.Draw(image, "RGBA")

        def font(size, bold=False):
            candidates = (
                [
                    "/usr/share/fonts/opentype/noto/"
                    "NotoSansCJK-Bold.ttc",
                    "/usr/share/fonts/truetype/"
                    "dejavu/DejaVuSans-Bold.ttf",
                ]
                if bold
                else [
                    "/usr/share/fonts/opentype/noto/"
                    "NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/"
                    "dejavu/DejaVuSans.ttf",
                ]
            )
            for path in candidates:
                try:
                    return ImageFont.truetype(
                        path,
                        size,
                    )
                except OSError:
                    continue
            return ImageFont.load_default()

        status = str(
            overlay.get("status", "RUNNING")
        ).upper()
        status_color = {
            "RUNNING": (255, 211, 54, 255),
            "SUCCESS": (72, 208, 113, 255),
            "FAILURE": (244, 80, 80, 255),
            "INCOMPLETE": (255, 153, 51, 255),
        }.get(
            status,
            (230, 230, 230, 255),
        )
        text_color = (244, 247, 250, 255)
        muted_color = (182, 192, 204, 255)
        speed_kmh = float(
            overlay.get("speed_kmh", 0.0)
        )
        target_speed_kmh = float(
            overlay.get(
                "target_speed_kmh",
                0.0,
            )
        )

        left_panel = (24, 24, 322, 264)
        right_panel = (
            self.width - 720,
            24,
            self.width - 24,
            296,
        )
        for panel in (left_panel, right_panel):
            draw.rounded_rectangle(
                panel,
                radius=10,
                fill=(62, 68, 78, 170),
                outline=(183, 190, 200, 140),
                width=2,
            )

        badge = (57, 40, 290, 99)
        draw.rounded_rectangle(
            badge,
            radius=9,
            fill=(70, 76, 86, 205),
            outline=status_color,
            width=3,
        )
        badge_font = font(29, bold=True)
        box = draw.textbbox(
            (0, 0),
            status,
            font=badge_font,
        )
        badge_x = (
            badge[0]
            + (
                badge[2]
                - badge[0]
                - (box[2] - box[0])
            )
            / 2
        )
        draw.text(
            (badge_x, 54),
            status,
            font=badge_font,
            fill=status_color,
        )

        gauge_center = (173, 180)
        gauge_radius = 76
        gauge_box = (
            gauge_center[0] - gauge_radius,
            gauge_center[1] - gauge_radius,
            gauge_center[0] + gauge_radius,
            gauge_center[1] + gauge_radius,
        )
        gauge_max = max(
            40.0,
            (
                int(
                    max(
                        speed_kmh,
                        target_speed_kmh,
                        1.0,
                    )
                    / 10.0
                )
                + 1
            )
            * 10.0,
        )
        draw.arc(
            gauge_box,
            135,
            405,
            fill=(58, 69, 82, 255),
            width=13,
        )
        for tick in range(6):
            angle = math.radians(
                135 + 270 * tick / 5.0
            )
            outer = (
                gauge_center[0]
                + math.cos(angle)
                * (gauge_radius - 1),
                gauge_center[1]
                + math.sin(angle)
                * (gauge_radius - 1),
            )
            inner = (
                gauge_center[0]
                + math.cos(angle)
                * (gauge_radius - 14),
                gauge_center[1]
                + math.sin(angle)
                * (gauge_radius - 14),
            )
            draw.line(
                (outer, inner),
                fill=muted_color,
                width=2,
            )
        needle_angle = math.radians(
            135
            + 270
            * min(speed_kmh, gauge_max)
            / gauge_max
        )
        needle_end = (
            gauge_center[0]
            + math.cos(needle_angle)
            * (gauge_radius - 20),
            gauge_center[1]
            + math.sin(needle_angle)
            * (gauge_radius - 20),
        )
        draw.line(
            (gauge_center, needle_end),
            fill=(91, 211, 255, 255),
            width=6,
        )
        draw.ellipse(
            (
                gauge_center[0] - 8,
                gauge_center[1] - 8,
                gauge_center[0] + 8,
                gauge_center[1] + 8,
            ),
            fill=(91, 211, 255, 255),
        )
        draw.text(
            (125, 142),
            "{0:.0f}".format(speed_kmh),
            font=font(38, bold=True),
            fill=text_color,
        )
        draw.text(
            (128, 190),
            "km/h",
            font=font(19),
            fill=muted_color,
        )

        rx = right_panel[0] + 20
        heading = "ASR TEXT (DEMO BASELINE)"
        if "route_s_m" in overlay:
            heading = (
                "SCENE 3 | ROUTE "
                "{0:.2f}/6.00 KM"
            ).format(
                float(overlay["route_s_m"])
                / 1000.0
            )
        draw.text(
            (rx, 42),
            heading,
            font=font(16, bold=True),
            fill=muted_color,
        )
        draw.text(
            (rx, 62),
            str(
                overlay.get("asr_text", "")
            )[:42],
            font=font(21, bold=True),
            fill=text_color,
        )
        draw.text(
            (rx, 95),
            "STRUCTURED INTENT (RULE)",
            font=font(16, bold=True),
            fill=muted_color,
        )
        draw.text(
            (rx, 115),
            (
                "action={0}  target={1:.0f} km/h  "
                "emergency={2}"
            ).format(
                str(
                    overlay.get("action", "")
                ).upper(),
                target_speed_kmh,
                overlay.get(
                    "emergency",
                    False,
                ),
            ),
            font=font(20),
            fill=(
                status_color
                if overlay.get(
                    "emergency",
                    False,
                )
                else text_color
            ),
        )

        def chip(
            label,
            value,
            x,
            y,
            color,
        ):
            draw.text(
                (x, y),
                label,
                font=font(15, bold=True),
                fill=muted_color,
            )
            width = (
                210
                if label == "POLICY STATE"
                else 170
            )
            box = (
                x,
                y + 20,
                x + width,
                y + 50,
            )
            draw.rounded_rectangle(
                box,
                radius=6,
                fill=(70, 76, 86, 205),
                outline=color,
                width=2,
            )
            draw.text(
                (x + 10, y + 24),
                value,
                font=font(17, bold=True),
                fill=color,
            )

        risk_level = str(
            overlay.get("risk_level", "LOW")
        )
        risk_color = {
            "HIGH": (244, 80, 80, 255),
            "MEDIUM": (255, 153, 51, 255),
            "LOW": (72, 208, 113, 255),
        }.get(risk_level, muted_color)
        chip(
            "RISK",
            risk_level,
            rx,
            150,
            risk_color,
        )
        chip(
            "POLICY STATE",
            str(
                overlay.get(
                    "policy_state",
                    "",
                )
            ),
            rx + 188,
            150,
            status_color,
        )

        def bar(
            label,
            value,
            y,
            color,
            centered=False,
        ):
            left = rx + 48
            right = rx + 250
            draw.text(
                (rx, y - 2),
                label,
                font=font(18, bold=True),
                fill=text_color,
            )
            draw.rounded_rectangle(
                (left, y, right, y + 18),
                radius=5,
                fill=(45, 55, 67, 245),
            )
            if centered:
                middle = (left + right) / 2.0
                draw.line(
                    (
                        middle,
                        y - 3,
                        middle,
                        y + 21,
                    ),
                    fill=muted_color,
                    width=1,
                )
                end = (
                    middle
                    + max(
                        -1.0,
                        min(1.0, value),
                    )
                    * (right - left)
                    / 2.0
                )
                draw.rectangle(
                    (
                        min(middle, end),
                        y + 3,
                        max(middle, end),
                        y + 15,
                    ),
                    fill=color,
                )
            else:
                end = (
                    left
                    + max(
                        0.0,
                        min(1.0, value),
                    )
                    * (right - left)
                )
                draw.rounded_rectangle(
                    (
                        left,
                        y,
                        end,
                        y + 18,
                    ),
                    radius=5,
                    fill=color,
                )

        bar(
            "T",
            float(
                overlay.get("throttle", 0.0)
            ),
            224,
            (72, 208, 113, 255),
        )
        bar(
            "B",
            float(
                overlay.get("brake", 0.0)
            ),
            248,
            (244, 80, 80, 255),
        )
        bar(
            "S",
            float(overlay.get("steer", 0.0)),
            272,
            (255, 211, 54, 255),
            centered=True,
        )
        self._draw_speed_chart(
            draw,
            overlay,
            font,
            muted_color,
            (
                right_panel[0] + 420,
                160,
                right_panel[2] - 18,
                242,
            ),
        )
        return image.tobytes("raw", "BGRA")

    def _draw_speed_chart(
        self,
        draw,
        overlay,
        font,
        muted_color,
        plot,
    ):
        sim_time = float(
            overlay.get("sim_time_s", 0.0)
        )
        self._speed_history.append(
            (
                sim_time,
                float(
                    overlay.get(
                        "speed_kmh",
                        0.0,
                    )
                ),
                float(
                    overlay.get(
                        "target_speed_kmh",
                        0.0,
                    )
                ),
            )
        )
        while (
            self._speed_history
            and self._speed_history[0][0]
            < sim_time - 12.0
        ):
            self._speed_history.popleft()
        draw.text(
            (plot[0], plot[1] - 28),
            "SPD",
            font=font(18, bold=True),
            fill=muted_color,
        )
        values = [
            max(speed, target)
            for _, speed, target
            in self._speed_history
        ]
        max_speed = max(
            30.0,
            max(values) if values else 30.0,
        )
        max_speed = max(
            10.0,
            (
                int(max_speed / 10.0)
                + 1
            )
            * 10.0,
        )
        for level in range(3):
            y = (
                plot[3]
                - (plot[3] - plot[1])
                * level
                / 2.0
            )
            draw.line(
                (plot[0], y, plot[2], y),
                fill=(70, 80, 92, 150),
                width=1,
            )
        if len(self._speed_history) < 2:
            return
        first_time = self._speed_history[0][0]
        span = max(
            12.0,
            sim_time - first_time,
        )

        def points(index):
            result = []
            for (
                time_s,
                speed,
                target,
            ) in self._speed_history:
                value = (
                    speed
                    if index == 1
                    else target
                )
                x = (
                    plot[0]
                    + (time_s - first_time)
                    / span
                    * (plot[2] - plot[0])
                )
                y = (
                    plot[3]
                    - min(value, max_speed)
                    / max_speed
                    * (plot[3] - plot[1])
                )
                result.append((x, y))
            return result

        draw.line(
            points(1),
            fill=(91, 211, 255, 255),
            width=3,
            joint="curve",
        )
        draw.line(
            points(2),
            fill=(255, 211, 54, 255),
            width=3,
            joint="curve",
        )

    def destroy(self):
        if (
            self.sensor is not None
            and self.sensor.is_alive
        ):
            self.sensor.stop()
            self.sensor.destroy()
        self.sensor = None
        if self.video_writer is not None:
            self.video_writer.close()
            self.written_video_frames = (
                self.video_writer.frame_count
            )
            self.dropped_video_frames = (
                self.video_writer.dropped_frames
            )
            print(
                "[Camera] Direct video frames: "
                "{0}, dropped: {1}".format(
                    self.written_video_frames,
                    self.dropped_video_frames,
                )
            )
            self.video_writer = None
