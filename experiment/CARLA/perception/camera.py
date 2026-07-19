import carla
import os
import numpy as np
import cv2



class CameraSensor:


    def __init__(
        self,
        world,
        vehicle
    ):


        self.world = world

        self.vehicle = vehicle


        self.camera = None


        self.save_dir = (
            "outputs/camera_images"
        )


        os.makedirs(
            self.save_dir,
            exist_ok=True
        )


        self.frame_id = 0


        # 当前最新图像
        self.latest_image = None




    def setup(self):


        blueprint_library = (
            self.world
            .get_blueprint_library()
        )


        camera_bp = blueprint_library.find(
            "sensor.camera.rgb"
        )


        camera_bp.set_attribute(
            "image_size_x",
            "800"
        )


        camera_bp.set_attribute(
            "image_size_y",
            "600"
        )


        camera_bp.set_attribute(
            "fov",
            "90"
        )


        transform = carla.Transform(

            carla.Location(
                x=1.5,
                z=2.4
            )

        )


        self.camera = (
            self.world.spawn_actor(
                camera_bp,
                transform,
                attach_to=self.vehicle
            )
        )


        self.camera.listen(
            self.save_image
        )





    def save_image(
        self,
        image
    ):


        array = np.frombuffer(
            image.raw_data,
            dtype=np.uint8
        )


        array = array.reshape(

            (
                image.height,
                image.width,
                4
            )

        )


        # CARLA输出 BGRA
        rgb = array[:, :, :3]


        # 保存最新帧
        self.latest_image = rgb



        filename = os.path.join(

            self.save_dir,

            f"{self.frame_id:06d}.png"

        )


        cv2.imwrite(

            filename,

            rgb

        )


        self.frame_id += 1





    def get_image(self):


        return self.latest_image





    def destroy(self):


        if self.camera:


            self.camera.stop()


            self.camera.destroy()


            self.camera = None