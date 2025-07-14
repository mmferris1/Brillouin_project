import numpy as np
import cv2
from src.eye_position.stereocalibrator import StereoCalibrator

def main():
    calibrator = StereoCalibrator(
        left_dir="C:\Users\Mandelstam\Documents\Connor\data\2025-6-5\take8\left",
        right_dir="C:\Users\Mandelstam\Documents\Connor\data\2025-6-5\take8\right",
        pattern_size=(8, 6),
        square_size=0.005
    )

    # === Load individual intrinsics ===
    intrinsics = np.load("individual_intrinsics.npz")
    K1, D1 = intrinsics["K1"], intrinsics["D1"]
    K2, D2 = intrinsics["K2"], intrinsics["D2"]

    calibrator.cameraMatrix1 = K1
    calibrator.distCoeffs1 = D1
    calibrator.cameraMatrix2 = K2
    calibrator.distCoeffs2 = D2

    print("[INFO] Loaded individual intrinsics from 'individual_intrinsics.npz'.")

    # === Load image pairs ===
    left_images, right_images = calibrator.load_image_pairs()
    objp = calibrator.prepare_object_points()

    print(f"[INFO] Found {len(left_images)} left and {len(right_images)} right images.")

    for left_path, right_path in zip(left_images, right_images):
        imgL = cv2.imread(left_path)
        imgR = cv2.imread(right_path)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        foundL, cornersL = calibrator.find_corners(grayL, left_path)
        foundR, cornersR = calibrator.find_corners(grayR, right_path)

        if foundL and foundR:
            calibrator.objpoints.append(objp)
            calibrator.imgpoints1.append(cornersL)
            calibrator.imgpoints2.append(cornersR)

    if not calibrator.objpoints:
        print("[ERROR] No valid stereo corner matches.")
        return

    img_size = grayL.shape[::-1]

    print("[INFO] Starting stereo calibration...")
    retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        calibrator.objpoints,
        calibrator.imgpoints1,
        calibrator.imgpoints2,
        K1, D1, K2, D2,
        imageSize=img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print("[INFO] Stereo calibration complete.")
    print("Rotation matrix (R):\n", R)
    print("Translation vector (T):\n", T)

    np.savez("stereo_calibration.npz",
             R=R, T=T, E=E, F=F,
             K1=K1, D1=D1,
             K2=K2, D2=D2)

    print("[INFO] Calibration results saved to 'stereo_calibration.npz'.")

if __name__ == "__main__":
    main()
