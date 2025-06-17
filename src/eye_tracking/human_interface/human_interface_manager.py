import numpy as np

from src.eye_tracking.human_interface.zaber_human_interface import ZaberHumanInterface
from src.eye_tracking.human_interface.base_zaber_human_interface import BaseZaberHumanInterface
from src.eye_tracking.human_interface.zaber_human_interface_dummy import ZaberHumanInterfaceDummy
from src.eye_tracking.pupil_detection_laser_focus import PupilDetection
from src.eye_tracking.human_interface.dummy_allied_vision import DummyMakoCamera
from src.eye_tracking.human_interface.allied_vision import AlliedVisionCamera

import cv2
from PyQt5.QtGui import QPixmap, QDoubleValidator, QImage


class HumanInterfaceManager:
    def __init__(self, use_dummy=True):
        if use_dummy:
            from src.eye_tracking.human_interface.zaber_human_interface_dummy import ZaberHumanInterfaceDummy as Zaber
        else:
            from src.eye_tracking.human_interface.zaber_human_interface import ZaberHumanInterface as Zaber

        self.motor_interface = Zaber()
        self.pupil_detector = PupilDetection()
        self.last_image = None
        self.pupil_center = (np.nan, np.nan)
        self.pupil_radius = float('nan')
        self.ellipse = None

        self.camera = DummyMakoCamera()
        self.latest_frame = None

        self.camera.set_roi(0, 0, 640, 480)
        self.camera.set_exposure(10000)
        self.camera.set_gain(10.0)
        self.camera.set_acquisition_mode("Continuous")
        self.camera.start_stream(self._on_new_frame)

    # def update_frame(self, image: np.ndarray, radius_guess: float = 50.0):
    #     self.last_image = image
    #     result = self.pupil_detector.DetectPupil(image, radius_guess)
    #     if result:
    #         drawing, center, ellipse, radius = result
    #         self.pupil_center = center
    #         self.ellipse = ellipse
    #         self.pupil_radius = radius
    #         return drawing
    #     return image
    #
    # def get_overlay_image(self):
    #     if self.last_image is None:
    #         return None
    #     image = self.update_frame(self.last_image)
    #     from PyQt5.QtGui import QImage, QPixmap
    #     height, width, channel = image.shape
    #     bytes_per_line = 3 * width
    #     qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    #     return QPixmap.fromImage(qimage)

#updated other version
    def _on_new_frame(self, frame):
        self.latest_frame = frame

    def get_overlay_image(self):
        if self.latest_frame is None:
            return None

        rgb_image = cv2.cvtColor(self.latest_frame, cv2.COLOR_GRAY2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def get_pupil_coords(self):
        return self.pupil_center

    def get_laser_coords(self):
        # TODO: Replace with actual laser detection logic
        return (100, 100)

    def get_motor_position(self, axis: str = None):
        pos = self.motor_interface.get_position()
        if axis:
            return pos.get(axis.lower(), float('nan'))
        return pos

    def move_motor_absolute(self, axis: str, pos_um: float):
        method = {
            'x': self.motor_interface.leftright_abs,
            'y': self.motor_interface.updown_abs,
            'z': self.motor_interface.forwardbackwards_abs
        }.get(axis)
        if method:
            method(pos_um)

    def move_motor_relative(self, axis: str, delta_um: float):
        method = {
            'x': self.motor_interface.leftright_rel,
            'y': self.motor_interface.updown_rel,
            'z': self.motor_interface.forwardbackwards_rel
        }.get(axis)
        if method:
            method(delta_um)

    def get_status_summary(self):
        return {
            'pupil_center': self.pupil_center,
            'pupil_radius': self.pupil_radius,
            'motor_position': self.get_motor_position()
        }
