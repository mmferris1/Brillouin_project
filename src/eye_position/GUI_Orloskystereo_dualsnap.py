import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

from src.eye_tracking.devices.connor.dual_allied_vision_cameras import DualAlliedVisionCameras
from orlosky_pupildetection import process_frame
from triangulation_v3 import StereoCalibrator, annotate_image


class OrloskyGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Eye Tracker (Orlosky)")

        # Load stereo calibration
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

        self.cams = DualAlliedVisionCameras()
        self.cams.start_stream()

        self.label = QLabel("Press Start to begin.")
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.snap_and_process)

        self.start_btn.clicked.connect(self.start_loop)
        self.stop_btn.clicked.connect(self.stop_loop)

    def start_loop(self):
        self.timer.start(200)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_loop(self):
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def snap_and_process(self):
        try:
            self.cams.clear_queues()
            self.cams.trigger_both()

            f0 = self.cams.cam0.camera.get_frame()
            f1 = self.cams.cam1.camera.get_frame()
            imgL = f0.as_numpy_ndarray()
            imgR = f1.as_numpy_ndarray()

            ellipseL = process_frame(imgL)
            ellipseR = process_frame(imgR)

            if ellipseL is None or ellipseR is None:
                self.label.setText("Pupil not detected in one or both images.")
                return

            centerL = np.array(ellipseL[0], dtype=np.float32)
            centerR = np.array(ellipseR[0], dtype=np.float32)

            point_3d = self.calibrator.triangulate(centerL.reshape(1, 2), centerR.reshape(1, 2))
            point_3d = point_3d[0]  # extract single 3D point

            imgL = annotate_image(imgL, centerL, point_3d)
            imgR = annotate_image(imgR, centerR, point_3d)

            combined = np.hstack((imgR, imgL))
            rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qt_img))

        except Exception as e:
            self.label.setText(f"[ERROR] {str(e)}")

    def closeEvent(self, event):
        self.stop_loop()
        self.cams.stop_stream()
        self.cams.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = OrloskyGUI()
    gui.resize(1200, 600)
    gui.show()
    sys.exit(app.exec_())
