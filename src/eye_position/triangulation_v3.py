import os
import cv2
import numpy as np
from stereocalibrator import StereoCalibrator
from src.eye_tracking.pupil_detection_laser_focus import PupilDetection

def annotate_image(img, center, point_3d):
    h, w = img.shape[:2]

    # show red dot at detected pupil center
    center_int = tuple(map(int, center))
    cv2.circle(img, center_int, 6, (0, 0, 255), -1)  #red

    # show the triangulated 3D point
    point_mm = point_3d * 1000
    text = f"XYZ = ({point_mm[0]:.2f}, {point_mm[1]:.2f}, {point_mm[2]:.2f}) mm"
    cv2.putText(
        img, text,
        org=(10, h - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(0, 255, 0),  #green
        thickness=2,
        lineType=cv2.LINE_AA
    )
    return img

def main():
    left_dir = "/Users/margaretferris/Desktop/dummyeye8/left"
    right_dir = "/Users/margaretferris/Desktop/dummyeye8/right"
    calib_file = "stereo_calibration2.npz"

    # Load calibration
    calib = np.load(calib_file)
    K1, D1 = calib["K1"], calib["D1"]
    K2, D2 = calib["K2"], calib["D2"]
    R, T = calib["R"], calib["T"]

    calibrator = StereoCalibrator(left_dir, right_dir)
    calibrator.cameraMatrix1 = K1
    calibrator.distCoeffs1 = D1
    calibrator.cameraMatrix2 = K2
    calibrator.distCoeffs2 = D2
    calibrator.R = R
    calibrator.T = T
    calibrator.compute_rectification()

    detector = PupilDetection()

    print("[DEBUG] Stereo baseline (mm):", np.linalg.norm(T) * 1000)

    for i in range(1, 100):
        left_path = os.path.join(left_dir, f"left_{i}.png")
        right_path = os.path.join(right_dir, f"right_{i}.png")

        if not os.path.exists(left_path) or not os.path.exists(right_path):
            continue

        imgL = cv2.imread(left_path)
        imgR = cv2.imread(right_path)

        resultL = detector.DetectPupil(imgL, radiusGuess=80)
        resultR = detector.DetectPupil(imgR, radiusGuess=80)

        if not resultL or not resultR:
            print(f"[WARNING] Skipping frame {i}: pupil not detected.")
            continue

        drawnL, centerL, _, _ = resultL
        drawnR, centerR, _, _ = resultR

        print(f"[DEBUG] Frame {i} - Pupil centers: Left={centerL}, Right={centerR}")

        pts1 = np.array([centerL], dtype=np.float32)
        pts2 = np.array([centerR], dtype=np.float32)

        point_3d = calibrator.triangulate(pts1, pts2)
        print(f"[RESULT] Frame {i} → 3D point: {point_3d[0]}")

        drawnL = annotate_image(drawnL, centerL, point_3d[0])
        drawnR = annotate_image(drawnR, centerR, point_3d[0])

        combined = np.hstack((drawnR, drawnL))
        cv2.imshow(f"Pupil Detection Frame {i}", combined)

        key = cv2.waitKey(0)
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
