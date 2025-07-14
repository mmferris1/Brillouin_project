import os
import cv2
import numpy as np
from src.eye_tracking.pupil_detection import PupilDetection

def triangulate_3d_point(pt1, pt2, K1, D1, K2, D2, R, T):
    pt1 = np.array([[pt1]], dtype=np.float32)
    pt2 = np.array([[pt2]], dtype=np.float32)

    pt1_undist = cv2.undistortPoints(pt1, K1, D1)
    pt2_undist = cv2.undistortPoints(pt2, K2, D2)

    pt1_h = pt1_undist[0].T  # shape (2, 1)
    pt2_h = pt2_undist[0].T

    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T))

    point_4d = cv2.triangulatePoints(P1, P2, pt1_h, pt2_h)
    point_3d = (point_4d / point_4d[3])[:3].reshape(-1)
    return point_3d

def annotate_image(img, center, point_3d):
    h, w = img.shape[:2]
    point_mm = point_3d * 1000  # convert from meters to millimeters
    text = f"XYZ = ({point_mm[0]:.2f}, {point_mm[1]:.2f}, {point_mm[2]:.2f}) mm"
    cv2.putText(
        img, text,
        org=(10, h - 10),  # bottom-left corner
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.5,
        color=(0, 0, 255),  # red in BGR
        thickness=2,
        lineType=cv2.LINE_AA
    )

    return img

def main():
    # === Load calibration ===
    calib = np.load("stereo_calibration.npz")
    K1, D1 = calib["K1"], calib["D1"]
    K2, D2 = calib["K2"], calib["D2"]
    R, T = calib["R"], calib["T"]

    left_dir = os.path.expanduser("/Users/margaretferris/Desktop/dummyeye/left")
    right_dir = os.path.expanduser("/Users/margaretferris/Desktop/dummyeye/right")

    detector = PupilDetection()

    print("[DEBUG] Stereo baseline (mm):", np.linalg.norm(T) * 1000)

    # Iterate over matched stereo pairs
    for i in range(1, 100):  # adjust max range based on your number of files
        left_path = os.path.join(left_dir, f"left_dummyeye_{i}.png")
        right_path = os.path.join(right_dir, f"right_dummyeye_{i}.png")

        if not os.path.exists(left_path) or not os.path.exists(right_path):
            continue

        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        result1 = detector.DetectPupil(img_left, radiusGuess=75)
        result2 = detector.DetectPupil(img_right, radiusGuess=75)

        if not result1 or not result2:
            print(f"[WARNING] Skipping frame {i}: pupil not detected.")
            continue

        drawn1, center1, _, _ = result1
        drawn2, center2, _, _ = result2

        point_3d = triangulate_3d_point(center1, center2, K1, D1, K2, D2, R, T)
        print(f"[RESULT] Frame {i} → 3D point: {point_3d}")

        # Annotate and display
        drawn1 = annotate_image(drawn1, center1, point_3d)
        drawn2 = annotate_image(drawn2, center2, point_3d)

        combined = np.hstack((drawn1, drawn2))
        cv2.imshow(f"Pupil Detection Frame {i}", combined)

        key = cv2.waitKey(0)
        if key == 27:  # ESC key to break
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
