from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import cv2

from src.eye_position.gui.stereo_interface_manager import StereoInterfaceManager


class StereoInterfaceSignaller(QObject):
    left_image_ready = pyqtSignal(QPixmap)
    right_image_ready = pyqtSignal(QPixmap)
    pupil3d_ready = pyqtSignal(tuple)
    motor_position_updated = pyqtSignal(str, float)

    def __init__(self, manager: StereoInterfaceManager):
        super().__init__()
        self.manager = manager

    def update_images(self):
        result = self.manager.process_latest_frames()
        if result is None:
            return

        imgL, imgR, point_3d, _ = result

        if imgL is not None:
            self.left_image_ready.emit(self.numpy_to_qpixmap(imgL))
        if imgR is not None:
            self.right_image_ready.emit(self.numpy_to_qpixmap(imgR))
        self.pupil3d_ready.emit(point_3d)

    def move_motor_relative(self, axis: str, delta: float):
        self.manager.move_motor_relative(axis, delta)
        position = self.manager.get_motor_position(axis)
        self.motor_position_updated.emit(axis, position)

    def move_motor_absolute(self, axis: str, position: float):
        self.manager.move_motor_absolute(axis, position)
        actual_position = self.manager.get_motor_position(axis)
        self.motor_position_updated.emit(axis, actual_position)

    def set_laser_distance(self, distance: float):
        self.manager.set_laser_distance(distance)

    def set_exposure(self, exposure: float):
        self.manager.set_exposure(exposure)

    def set_gain(self, gain: float):
        self.manager.set_gain(gain)

    def set_roi(self, x: int, y: int, w: int, h: int):
        self.manager.set_roi(x, y, w, h)

    @staticmethod
    def numpy_to_qpixmap(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)
