import cv2
import numpy as np
import glob

# === Parameters ===
CHECKERBOARD = (8, 6)  # Number of inner corners per a chessboard row and column
SQUARE_SIZE = 0.0215  # Size of a square in meters (adjust to your checkerboard)

# === Prepare object points ===
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# === Load calibration images ===
image_paths = glob.glob("calib_images/*.jpg")  # Put calibration images in 'calib_images' folder

if not image_paths:
    print("No images found in 'calib_images/' folder.")
    exit()

for fname in image_paths:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# === Calibration ===
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Camera matrix (K):")
print(K)
print("\nDistortion coefficients:")
print(dist.ravel())

# === Save to file ===
np.savez("camera_intrinsics.npz", K=K, dist=dist, image_size=gray.shape[::-1])

print("\nSaved to 'camera_intrinsics.npz'")
