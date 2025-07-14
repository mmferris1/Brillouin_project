import cv2
import numpy as np
import glob
import os
import re

class StereoCalibrator:
    def __init__(self, left_dir, right_dir, pattern_size=(8, 6), square_size=0.0215):
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.pattern_size = pattern_size
        self.square_size = square_size

        # Calibration outputs
        self.cameraMatrix1 = None
        self.distCoeffs1 = None
        self.cameraMatrix2 = None
        self.distCoeffs2 = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None

        self.objpoints = []
        self.imgpoints1 = []
        self.imgpoints2 = []

    def prepare_object_points(self):
        objp = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        objp[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    def extract_index(self, filename):
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else -1

    def load_image_pairs(self):
        left_images = glob.glob(os.path.join(self.left_dir, "*.png"))
        right_images = glob.glob(os.path.join(self.right_dir, "*.png"))

        left_map = {self.extract_index(os.path.basename(f)): f for f in left_images}
        right_map = {self.extract_index(os.path.basename(f)): f for f in right_images}

        common_indices = sorted(set(left_map) & set(right_map))
        if not common_indices:
            raise ValueError("No matching indices found between left and right images.")

        left_sorted = [left_map[i] for i in common_indices]
        right_sorted = [right_map[i] for i in common_indices]

        for l, r in zip(left_sorted, right_sorted):
            print(f"[PAIR] {os.path.basename(l)} <--> {os.path.basename(r)}")

        return left_sorted, right_sorted

    def find_corners(self, gray, image_name):
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
        else:
            print(f"[WARNING] Corners not found in {image_name}")
        return found, corners

    def calibrate(self):
        objp = self.prepare_object_points()
        left_images, right_images = self.load_image_pairs()

        for left_path, right_path in zip(left_images, right_images):
            img1 = cv2.imread(left_path)
            img2 = cv2.imread(right_path)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            found1, corners1 = self.find_corners(gray1, left_path)
            found2, corners2 = self.find_corners(gray2, right_path)

            if found1 and found2:
                self.objpoints.append(objp)
                self.imgpoints1.append(corners1)
                self.imgpoints2.append(corners2)

        ret, self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, \
        self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpoints1,
            self.imgpoints2,
            None, None,
            None, None,
            gray1.shape[::-1],
            flags=0,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        )

        print("[INFO] Stereo calibration completed.")
        print("[INFO] Stereo calibration completed.")
        print("[RESULT] Camera Matrix 1:\n", self.cameraMatrix1)
        print("[RESULT] Distortion Coeffs 1:\n", self.distCoeffs1)
        print("[RESULT] Camera Matrix 2:\n", self.cameraMatrix2)
        print("[RESULT] Distortion Coeffs 2:\n", self.distCoeffs2)
        print("[RESULT] Rotation (R):\n", self.R)
        print("[RESULT] Translation (T):\n", self.T)
        print("[RESULT] Essential Matrix (E):\n", self.E)
        print("[RESULT] Fundamental Matrix (F):\n", self.F)
        return ret

    def compute_rectification(self, image_size=None):
        if image_size is None:
            image_size = cv2.imread(sorted(glob.glob(os.path.join(self.left_dir, "*.png")))[0]).shape[1::-1]

        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.cameraMatrix1, self.distCoeffs1,
            self.cameraMatrix2, self.distCoeffs2,
            image_size, self.R, self.T
        )
        print("[INFO] Stereo rectification completed.")
        print("[INFO] Stereo rectification completed.")
        print("[RESULT] Rectification Transform (R1):\n", self.R1)
        print("[RESULT] Rectification Transform (R2):\n", self.R2)
        print("[RESULT] Projection Matrix 1 (P1):\n", self.P1)
        print("[RESULT] Projection Matrix 2 (P2):\n", self.P2)
        print("[RESULT] Disparity-to-Depth Mapping Matrix (Q):\n", self.Q)

        return self.Q

    def triangulate(self, pts1, pts2):
        if self.P1 is None or self.P2 is None:
            raise RuntimeError("Projection matrices not computed. Run compute_rectification() first.")

        pts4D = cv2.triangulatePoints(self.P1, self.P2, pts1.T, pts2.T)
        pts3D = pts4D[:3] / pts4D[3]
        return pts3D.T

    def calibrate_single_camera(self, image_dir, save_path=None):
        objp = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        objp[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        objp *= self.square_size

        objpoints = []
        imgpoints = []

        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.bmp")) +
                             glob.glob(os.path.join(image_dir, "*.png")))

        if not image_paths:
            print(f"[ERROR] No images found in {image_dir}")
            return None, None, None

        print(f"[INFO] Found {len(image_paths)} images in {image_dir}")
        for fname in image_paths:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
            if ret:
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                objpoints.append(objp)
                imgpoints.append(corners2)

                vis_img = cv2.drawChessboardCorners(img, self.pattern_size, corners2, ret)
                cv2.imshow("Corners", vis_img)
                cv2.waitKey(100)

        cv2.destroyAllWindows()

        if not objpoints:
            print("[ERROR] No valid chessboard detections.")
            return None, None, None

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        print("[INFO] Intrinsic calibration complete.")
        print("Camera matrix (K):\n", K)
        print("Distortion coefficients:\n", dist.ravel())

        if save_path:
            with open(save_path, "w") as f:
                f.write("Camera matrix (K):\n")
                for row in K:
                    f.write("  " + "  ".join(f"{val:.6f}" for val in row) + "\n")
                f.write("\nDistortion coefficients:\n")
                f.write("  " + "  ".join(f"{val:.6f}" for val in dist.ravel()) + "\n")
                f.write("\nImage size (width, height):\n")
                f.write(f"  {gray.shape[1]}, {gray.shape[0]}\n")
            print(f"[INFO] Intrinsics saved to: {save_path}")

        return K, dist, gray.shape[::-1]

    def stereo_calibrate(self, K1, D1, K2, D2, image_size):
        """Performs stereo calibration using the known intrinsics and loaded image points."""
        if not self.objpoints or not self.imgpoints_left or not self.imgpoints_right:
            raise ValueError("No object/image points loaded for stereo calibration.")

        flags = (cv2.CALIB_FIX_INTRINSIC)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpoints_left,
            self.imgpoints_right,
            K1, D1,
            K2, D2,
            imageSize=image_size,
            criteria=criteria,
            flags=flags
        )

        print("[INFO] Stereo calibration complete.")
        print("[DEBUG] Rotation matrix (R):\n", R)
        print("[DEBUG] Translation vector (T):\n", T)
        return R, T, E, F
