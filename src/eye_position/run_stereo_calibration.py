from stereocalibrator import StereoCalibrator
import numpy as np
import cv2

def main():
    calibrator = StereoCalibrator(
        left_dir="/Users/margaretferris/Desktop/take7",
        right_dir="/Users/margaretferris/Desktop/take7",
        pattern_size=(8, 6),
        square_size=0.0005
    )

    # === Load image pairs ===
    left_images, right_images = calibrator.load_image_pairs()
    objp = calibrator.prepare_object_points()

    print(f"[INFO] Found {len(left_images)} left and {len(right_images)} right images")

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

    # === Image size from any valid image ===
    img_size = grayL.shape[::-1]

    print("[INFO] Starting stereo calibration...")
    retval, calibrator.cameraMatrix1, calibrator.distCoeffs1, \
    calibrator.cameraMatrix2, calibrator.distCoeffs2, \
    calibrator.R, calibrator.T, calibrator.E, calibrator.F = cv2.stereoCalibrate(
        calibrator.objpoints,
        calibrator.imgpoints1,
        calibrator.imgpoints2,
        None, None, None, None,
        imageSize=img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC  # or 0 if you don’t have prior intrinsics
    )

    print("[INFO] Stereo calibration complete.")
    print("Rotation matrix (R):\n", calibrator.R)
    print("Translation vector (T):\n", calibrator.T)

    np.savez("stereo_calibration.npz",
             R=calibrator.R,
             T=calibrator.T,
             E=calibrator.E,
             F=calibrator.F,
             K1=calibrator.cameraMatrix1,
             D1=calibrator.distCoeffs1,
             K2=calibrator.cameraMatrix2,
             D2=calibrator.distCoeffs2)

    print("[INFO] Calibration results saved to stereo_calibration.npz")

if __name__ == "__main__":
    main()
