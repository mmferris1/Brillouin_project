import sys
import cv2
import numpy as np
import threading
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

from src.eye_tracking.devices.connor.dual_allied_vision_cameras import DualAlliedVisionCameras
from src.eye_position.triangulation_v3 import PupilDetection, StereoCalibrator, annotate_image

class LivePupilGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Pupil Triangulation")

        # Load calibration
        calib = np.load("stereo_calibration2.npz")
        K1, D1 = calib["K1"], calib["D1"]
        K2, D2 = calib["K2"], calib["D2"]
        R, T = calib["R"], calib["T"]

        self.calibrator = StereoCalibrator(None, None)
        self.calibrator.cameraMatrix1 = K1
        self.calibrator.distCoeffs1 = D1
        self.calibrator.cameraMatrix2 = K2
        self.calibrator.distCoeffs2 = D2
        self.calibrator.R = R
        self.calibrator.T = T
        self.calibrator.compute_rectification()

        self.detector = PupilDetection()
        self.cams = DualAlliedVisionCameras()
        self.cams.start_stream()

        self.image_label = QLabel("Waiting for data...")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.snap_and_process)

        self.start_button.clicked.connect(self.start_loop)
        self.stop_button.clicked.connect(self.stop_loop)

    def start_loop(self):
        self.timer.start(200)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_loop(self):
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def snap_and_process(self):
        try:
            self.cams.clear_queues()
            self.cams.trigger_both()
            f0 = self.cams.cam0.camera.get_frame()
            f1 = self.cams.cam1.camera.get_frame()

            imgL = f0.as_numpy_ndarray()
            imgR = f1.as_numpy_ndarray()

            resultL = self.detector.DetectPupil(imgL, radiusGuess=80)
            resultR = self.detector.DetectPupil(imgR, radiusGuess=80)

            if not resultL or not resultR:
                self.image_label.setText("Pupil not detected in one or both images.")
                return

            drawnL, centerL, _, _ = resultL
            drawnR, centerR, _, _ = resultR

            pts1 = np.array([centerL], dtype=np.float32)
            pts2 = np.array([centerR], dtype=np.float32)
            point_3d = self.calibrator.triangulate(pts1, pts2)

            drawnL = annotate_image(drawnL, centerL, point_3d[0])
            drawnR = annotate_image(drawnR, centerR, point_3d[0])

            combined = np.hstack((drawnR, drawnL))
            rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))

        except Exception as e:
            self.image_label.setText(f"[ERROR] {str(e)}")

    def closeEvent(self, event):
        self.stop_loop()
        self.cams.stop_stream()
        self.cams.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = LivePupilGUI()
    gui.resize(1200, 500)
    gui.show()
    sys.exit(app.exec_())
