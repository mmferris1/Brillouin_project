import sys
import time
import cv2
import numpy as np
import threading
import queue
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage

from src.eye_tracking.devices.connor.allied_vision_camera2 import AlliedVisionCamera
from src.eye_position.triangulation_v3 import PupilDetection, StereoCalibrator, annotate_image

# Assuming DualAlliedVisionCameras is imported correctly

frame_q0 = queue.Queue()
frame_q1 = queue.Queue()

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

        self.detector = PupilDetection()

        # Use the stereo camera wrapper
        from src.eye_tracking.devices.connor.dual_allied_vision_cameras import DualAlliedVisionCameras
        self.cams = DualAlliedVisionCameras()

        # gui layout
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

        self.processing_thread = None
        self.frame_processing_flag = False

    def start_loop(self):
        # Start streaming from both cameras (continuous)
        self.cams.start_stream()
        self.frame_processing_flag = True
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_loop(self):
        print("[DEBUG] Stop loop clicked")
        self.frame_processing_flag = False
        self.cams.stop_stream()
        if self.processing_thread:
            self.processing_thread.join()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def process_frames(self):
        try:
            while self.frame_processing_flag:
                # Grab stereo image pair (streaming)
                f0, f1 = self.cams.snap_once()  # Grab frames from both cameras

                if f0.get_status() != 0 or f1.get_status() != 0:
                    print("Incomplete frame received.")
                    time.sleep(0.1)  # Sleep to avoid overloading the CPU
                    continue

                imgL = f0.as_numpy_ndarray()
                imgR = f1.as_numpy_ndarray()

                print(f"[DEBUG] imgL shape: {imgL.shape}, imgR shape: {imgR.shape}")

                # Pupil detection
                resultL = self.detector.DetectPupil(imgL, radiusGuess=80)
                resultR = self.detector.DetectPupil(imgR, radiusGuess=80)

                if not resultL or not resultR:
                    print("Pupil not detected in one or both images.")
                    time.sleep(0.1)
                    continue

                drawnL, centerL, _, _ = resultL
                drawnR, centerR, _, _ = resultR

                pts1 = np.array([centerL], dtype=np.float32)
                pts2 = np.array([centerR], dtype=np.float32)

                # Rectify and triangulate
                R1 = self.calibrator.R1
                R2 = self.calibrator.R2
                P1 = self.calibrator.P1
                P2 = self.calibrator.P2
                K1 = self.calibrator.cameraMatrix1
                D1 = self.calibrator.distCoeffs1
                K2 = self.calibrator.cameraMatrix2
                D2 = self.calibrator.distCoeffs2

                pts1_rect = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K1, D1, R=R1, P=P1)
                pts2_rect = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K2, D2, R=R2, P=P2)

                point_4d = cv2.triangulatePoints(P1, P2, pts1_rect, pts2_rect)
                point_3d = (point_4d[:3] / point_4d[3]).reshape(-1)

                drawnL = annotate_image(drawnL, centerL, point_3d)
                drawnR = annotate_image(drawnR, centerR)

                # Convert images to RGB for display
                def to_rgb(img):
                    if img.ndim == 2 or img.shape[2] == 1:
                        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    return img

                drawnL = to_rgb(drawnL)
                drawnR = to_rgb(drawnR)

                # Combine the images for display
                combined = np.hstack((drawnR, drawnL))
                rgb_resized = cv2.resize(combined, (0, 0), fx=0.3, fy=0.3)

                h, w, ch = rgb_resized.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_resized.data, w, h, bytes_per_line, QImage.Format_RGB888)

                self.image_label.setPixmap(QPixmap.fromImage(qt_image))

                time.sleep(0.1)  # Sleep to control frame processing rate

        except Exception as e:
            print(f"[ERROR] Processing frames: {str(e)}")

    def closeEvent(self, event):
        self.stop_loop()
        self.cams.close()  # Close the camera on shutdown
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = LivePupilGUI()
    window.resize(1000, 200)
    window.show()
    print("[DEBUG] gui running event loop")
    app.exec_()
    print("[DEBUG] Event loop exited")


if __name__ == "__main__":
    main()
