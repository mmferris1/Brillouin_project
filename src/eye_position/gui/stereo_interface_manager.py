import numpy as np
import cv2
from PyQt5.QtCore import QMutex
from src.eye_tracking.pupil_detection_laser_focus import PupilDetection
from src.eye_tracking.laser_focus_visualizer import LaserFocusVisualizer
from src.eye_position.triangulation_v3 import StereoCalibrator
from src.eye_tracking.devices.connor.dual_allied_vision_cameras import DualAlliedVisionCameras
from src.eye_tracking.human_interface.zaber_human_interface import ZaberHumanInterface
from src.eye_tracking.human_interface.zaber_human_interface_dummy import ZaberHumanInterfaceDummy


class StereoInterfaceManager:
    def __init__(self, use_dummy_motor=False):
        self.cameras = DualAlliedVisionCameras()
        self.pupil_detector = PupilDetection()
        self.laser_visualizer = LaserFocusVisualizer()
        self.laser_distance = None
        self.frame_mutex = QMutex()

        self.last_left = None
        self.last_right = None
        self.last_3d_point = (np.nan, np.nan, np.nan)

        self.motor_interface = ZaberHumanInterfaceDummy() if use_dummy_motor else ZaberHumanInterface()

        # Load stereo calibration
        calib = np.load("stereo_calibration.npz")
        self.calibrator = StereoCalibrator(None, None)
        self.calibrator.cameraMatrix1 = calib["K1"]
        self.calibrator.distCoeffs1 = calib["D1"]
        self.calibrator.cameraMatrix2 = calib["K2"]
        self.calibrator.distCoeffs2 = calib["D2"]
        self.calibrator.R = calib["R"]
        self.calibrator.T = calib["T"]
        self.calibrator.compute_rectification(image_size=(2048, 2048))

        self.latest_raw_left = None
        self.latest_raw_right = None
        self._setup_camera_stream()

    def _setup_camera_stream(self):
        def left_handler(cam, frame):
            if frame.get_status() == 0:
                self.frame_mutex.lock()
                self.latest_raw_left = frame.as_numpy_ndarray()
                self.frame_mutex.unlock()

        def right_handler(cam, frame):
            if frame.get_status() == 0:
                self.frame_mutex.lock()
                self.latest_raw_right = frame.as_numpy_ndarray()
                self.frame_mutex.unlock()

        self.cameras.cam0.camera.start_streaming(left_handler)
        self.cameras.cam1.camera.start_streaming(right_handler)

    def process_latest_frames(self):
        self.frame_mutex.lock()
        imgL = self.latest_raw_left.copy() if self.latest_raw_left is not None else None
        imgR = self.latest_raw_right.copy() if self.latest_raw_right is not None else None
        self.frame_mutex.unlock()

        if imgL is None or imgR is None:
            return None, None, (np.nan, np.nan, np.nan), ((np.nan, np.nan), (np.nan, np.nan))

        resultL = self.pupil_detector.DetectPupil(imgL, radiusGuess=80)
        resultR = self.pupil_detector.DetectPupil(imgR, radiusGuess=80)

        if resultL:
            drawnL, centerL, _, _ = resultL
        else:
            centerL = (np.nan, np.nan)
            drawnL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)

        if resultR:
            drawnR, centerR, _, _ = resultR
        else:
            centerR = (np.nan, np.nan)
            drawnR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

        point_3d = (np.nan, np.nan, np.nan)
        if resultL and resultR:
            pts1 = np.array([centerL], dtype=np.float32)
            pts2 = np.array([centerR], dtype=np.float32)

            pts1_rect = cv2.undistortPoints(np.expand_dims(pts1, axis=1),
                                            self.calibrator.cameraMatrix1, self.calibrator.distCoeffs1,
                                            R=self.calibrator.R1, P=self.calibrator.P1)
            pts2_rect = cv2.undistortPoints(np.expand_dims(pts2, axis=1),
                                            self.calibrator.cameraMatrix2, self.calibrator.distCoeffs2,
                                            R=self.calibrator.R2, P=self.calibrator.P2)

            point_4d = cv2.triangulatePoints(self.calibrator.P1, self.calibrator.P2, pts1_rect, pts2_rect)
            point_3d = (point_4d[:3] / point_4d[3]).reshape(-1)
            self.last_3d_point = tuple(point_3d)

        if self.laser_distance is not None:
            drawnL = self.laser_visualizer.draw_laser_marker(drawnL, self.laser_distance)

        self.last_left = drawnL
        self.last_right = drawnR

        return drawnL, drawnR, self.last_3d_point, (centerL, centerR)

    def stop_camera_stream(self):
        self.cameras.stop_stream()

    def start_camera_stream(self):
        self._setup_camera_stream()  # Reinitializes handlers if needed

    def set_laser_distance(self, distance_um: float):
        self.laser_distance = distance_um

    def get_last_3d_point(self):
        return self.last_3d_point

    def get_motor_position(self, axis: str = None):
        pos = self.motor_interface.get_position()
        return pos.get(axis.lower(), float('nan')) if axis else pos

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

    def set_exposure(self, exposure_us: float):
        self.cameras.cam0.set_exposure(exposure_us)
        self.cameras.cam1.set_exposure(exposure_us)

    def set_gain(self, gain: float):
        self.cameras.cam0.set_gain(gain)
        self.cameras.cam1.set_gain(gain)

    def set_roi(self, x: int, y: int, w: int, h: int):
        self.cameras.cam0.set_roi(x, y, w, h)
        self.cameras.cam1.set_roi(x, y, w, h)

    def close(self):
        self.cameras.close()
