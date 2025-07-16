import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

from src.eye_tracking.devices.connor.allied_vision_camera2 import AlliedVisionCamera
from src.eye_position.triangulation_v3 import PupilDetection, StereoCalibrator, annotate_image


class LivePupilGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Pupil Triangulation")

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
        self.calibrator.compute_rectification(image_size=(2048, 2048))

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frames)

        self.detector = PupilDetection()

        # Use the stereo camera wrapper
        from src.eye_tracking.devices.connor.dual_allied_vision_cameras import DualAlliedVisionCameras
        self.cams = DualAlliedVisionCameras()

        # GUI layout
        self.image_label = QLabel("Waiting for data...")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        # Buttons
        self.start_button.clicked.connect(self.start_loop)
        self.stop_button.clicked.connect(self.stop_loop)

    def start_loop(self):
        self.cams.start_stream()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.timer.start(30)  # call every 30 ms, adjust as needed

    def stop_loop(self):
        print("[DEBUG] Stop loop clicked")
        self.timer.stop()
        self.cams.stop_stream()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def process_frames(self):
        try:
            f0, f1 = self.cams.snap_once()

            try:
                f0 = self.cams.camL.snap()
                if f0 is None:
                    print("camL returned None.")
                elif f0.get_status() != 0:
                    print("camL incomplete frame.")
                else:
                    print("camL working.")

            except Exception as e:
                print(f"[camL crash] {e}")

            # try:
            #     imgL = f0.as_numpy_ndarray()
            #     imgR = f1.as_numpy_ndarray()
            # except Exception as e:
            #     print(f"[ERROR] Buffer conversion failed: {e}")
            #     self.image_label.setText("[ERROR] Frame buffer conversion.")
            #     return
            #
            # if imgL is None or imgR is None:
            #     print("[ERROR] Null image buffer.")
            #     self.image_label.setText("Invalid image buffer.")
            #     return
            #
            # if imgL.size == 0 or imgR.size == 0:
            #     print("[ERROR] Empty frame buffer.")
            #     self.image_label.setText("Empty image frame.")
            #     return
            #
            # # Defensive image dimension check
            # if len(imgL.shape) != 2 or len(imgR.shape) != 2:
            #     print("[ERROR] Unexpected image shape.")
            #     self.image_label.setText("Unexpected image shape.")
            #     return
            #
            # # Optional safety: copy data before OpenCV calls
            # imgL = imgL.copy()
            # imgR = imgR.copy()
            #
            # print(f"[DEBUG] imgL shape: {imgL.shape}, imgR shape: {imgR.shape}")
            #
            # resultL = self.detector.DetectPupil(imgL, radiusGuess=80)
            # resultR = self.detector.DetectPupil(imgR, radiusGuess=80)
            #
            # if not resultL or not resultR:
            #     self.image_label.setText("Pupil not detected.")
            #     return
            #
            # drawnL, centerL, _, _ = resultL
            # drawnR, centerR, _, _ = resultR
            #
            # pts1 = np.array([centerL], dtype=np.float32)
            # pts2 = np.array([centerR], dtype=np.float32)
            #
            # # Rectification
            # pts1_rect = cv2.undistortPoints(np.expand_dims(pts1, axis=1),
            #                                 self.calibrator.cameraMatrix1,
            #                                 self.calibrator.distCoeffs1,
            #                                 R=self.calibrator.R1,
            #                                 P=self.calibrator.P1)
            #
            # pts2_rect = cv2.undistortPoints(np.expand_dims(pts2, axis=1),
            #                                 self.calibrator.cameraMatrix2,
            #                                 self.calibrator.distCoeffs2,
            #                                 R=self.calibrator.R2,
            #                                 P=self.calibrator.P2)
            #
            # # Triangulate
            # point_4d = cv2.triangulatePoints(self.calibrator.P1,
            #                                  self.calibrator.P2,
            #                                  pts1_rect, pts2_rect)
            # point_3d = (point_4d[:3] / point_4d[3]).reshape(-1)
            #
            # drawnL = annotate_image(drawnL, centerL, point_3d)
            # drawnR = annotate_image(drawnR, centerR)
            #
            # # Combine and convert to QImage
            # def to_rgb(img):
            #     if img.ndim == 2 or img.shape[2] == 1:
            #         return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            #     return img
            #
            # drawnL = to_rgb(drawnL)
            # drawnR = to_rgb(drawnR)
            #
            # combined = np.hstack((drawnR, drawnL))
            #
            # # Double-check shape before resize
            # if combined.size == 0:
            #     self.image_label.setText("Combined image is empty.")
            #     return
            #
            # rgb_resized = cv2.resize(combined, (0, 0), fx=0.3, fy=0.3)
            # h, w, ch = rgb_resized.shape
            # bytes_per_line = ch * w
            # qt_image = QImage(rgb_resized.data, w, h, bytes_per_line, QImage.Format_RGB888)
            #
            # self.image_label.setPixmap(QPixmap.fromImage(qt_image))

        except Exception as e:
            print(f"[EXCEPTION] {e}")
            #self.image_label.setText(f"[ERROR] {str(e)}")

    def closeEvent(self, event):
        self.timer.stop()
        self.stop_loop()
        self.cams.close()  # Close the camera on shutdown
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = LivePupilGUI()
    window.resize(1000, 200)
    window.show()
    print("[DEBUG] GUI running event loop")
    app.exec_()
    print("[DEBUG] Event loop exited")


if __name__ == "__main__":
    main()
