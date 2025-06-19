import numpy as np

from src.eye_tracking.human_interface.zaber_human_interface import ZaberHumanInterface
from src.eye_tracking.human_interface.base_zaber_human_interface import BaseZaberHumanInterface
from src.eye_tracking.human_interface.zaber_human_interface_dummy import ZaberHumanInterfaceDummy
from src.eye_tracking.pupil_detection_laser_focus import PupilDetection
from src.eye_tracking.human_interface.dummy_allied_vision import DummyMakoCamera
#from src.eye_tracking.human_interface.allied_vision import AlliedVisionCamera

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

        self.laser_distance = None  # <-- New attribute

    def set_laser_distance(self, distance_um: float):
        self.laser_distance = distance_um

    def _on_new_frame(self, frame):
        self.latest_frame = frame

    def get_overlay_image(self):
        if self.latest_frame is None:
            return None

        result = self.pupil_detector.DetectPupil(
            self.latest_frame,
            radiusGuess=50,  # You may want to expose this as a setting
            laser_distance=self.laser_distance  # Now passed from GUI
        )

        if result:
            drawing, center, ellipse, radius = result
            self.pupil_center = center
            self.ellipse = ellipse
            self.pupil_radius = radius

            rgb_image = cv2.cvtColor(drawing, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(q_img)
        return None

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

    def set_exposure(self, exposure_us: float):
        self.camera.set_exposure(exposure_us)

    def set_gain(self, gain: float):
        self.camera.set_gain(gain)

    def set_frame_rate(self, fps: float):
        self.camera.set_frame_rate(fps)

    def set_roi(self, x: int, y: int, width: int, height: int):
        self.camera.set_roi(x, y, width, height)

    def get_camera_settings(self):
        return {
            "exposure": self.camera.get_exposure(),
            "gain": self.camera.get_gain(),
            "frame_rate": self.camera.get_frame_rate(),
            "roi": self.camera.get_roi(),  # should return (x, y, width, height)
        }

    def start_camera_stream(self):
        self.camera.start_stream(self._on_new_frame)

    def stop_camera_stream(self):
        self.camera.stop_stream()

    def snap_camera_image(self):
        frame = self.camera.snap()
        self.latest_frame = frame