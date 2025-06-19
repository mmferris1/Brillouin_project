from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QPixmap
from src.eye_tracking.human_interface.human_interface_manager import HumanInterfaceManager



class HumanInterfaceSignaller(QObject):
    image_ready = pyqtSignal(QPixmap)
    pupil_laser_coords_ready = pyqtSignal(tuple, tuple)
    motor_position_updated = pyqtSignal(str, float)

    def __init__(self, manager: HumanInterfaceManager):
        super().__init__()
        self.manager = manager

    def update_image(self):
        pixmap = self.manager.get_overlay_image()
        if pixmap:
            self.image_ready.emit(pixmap)
            self.fetch_coords()

    def fetch_coords(self):
        pupil = self.manager.get_pupil_coords()
        laser = self.manager.get_laser_coords()
        self.pupil_laser_coords_ready.emit(pupil, laser)

    def move_motor_relative(self, axis: str, delta: float):
        self.manager.move_motor_relative(axis, delta)
        position = self.manager.get_motor_position(axis)
        self.motor_position_updated.emit(axis, position)

    def move_motor_absolute(self, axis: str, position: float):
        self.manager.move_motor_absolute(axis, position)
        actual_position = self.manager.get_motor_position(axis)
        self.motor_position_updated.emit(axis, actual_position)

    def start_camera_stream(self):
        self.manager.start_camera_stream()

    def stop_camera_stream(self):
        self.manager.stop_camera_stream()

    def snap_camera_image(self):
        self.manager.snap_camera_image()
        self.update_image()
        self.fetch_coords()
