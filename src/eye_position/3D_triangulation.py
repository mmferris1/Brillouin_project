import cv2
import numpy as np
from src.eye_tracking.pupil_detection import PupilDetection

def triangulate_3d_point(pt1, pt2, K1, D1, K2, D2, R, T):
    pt1 = np.array([[pt1]], dtype=np.float32)
    pt2 = np.array([[pt2]], dtype=np.float32)

    pt1_undist = cv2.undistortPoints(pt1, K1, D1, P=K1)
    pt2_undist = cv2.undistortPoints(pt2, K2, D2, P=K2)

    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T))

    pt1_h = pt1_undist.reshape(2, 1)
    pt2_h = pt2_undist.reshape(2, 1)

    point_4d = cv2.triangulatePoints(P1, P2, pt1_h, pt2_h)
    point_3d = (point_4d / point_4d[3])[:3].reshape(-1)

    return point_3d  # [X, Y, Z]

def main():
    # Load stereo calibration ===
    calib = np.load("stereo_calibration.npz")
    K1, D1 = calib["K1"], calib["D1"]
    K2, D2 = calib["K2"], calib["D2"]
    R, T = calib["R"], calib["T"]

    # === Load stereo images ===
    img_left = cv2.imread("left_dummyeye.bmp")
    img_right = cv2.imread("right_dummyeye.bmp")

    # === Detect pupil centers ===
    detector = PupilDetection()

    result1 = detector.DetectPupil(img_left, radiusGuess=50)
    result2 = detector.DetectPupil(img_right, radiusGuess=50)

    if not result1 or not result2:
        print("[ERROR] Could not detect pupil in one or both images.")
        return

    _, center1, _, _ = result1
    _, center2, _, _ = result2

    print(f"[INFO] Left center: {center1}, Right center: {center2}")

    # === Triangulate ===
    point_3d = triangulate_3d_point(center1, center2, K1, D1, K2, D2, R, T)
    print(f"[RESULT] Triangulated 3D position (in meters): {point_3d}")

if __name__ == "__main__":
    main()
